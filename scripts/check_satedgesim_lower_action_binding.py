#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError

ACTION_NAMES = ["LOCAL", "NEIGHBOR", "GEO", "GROUND"]
DEFAULT_SCENARIO_PROFILE = "mixed_cost_landscape"
DEFAULT_TASK_SOURCE_MODE = "round_robin_leo"
PROBE_A = {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0}
PROBE_B = {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25}
CONTROLLED_METRIC_KEYS = (
    "estimatedTransmissionRateMbps",
    "estimatedTaskTransmissionTimeSec",
    "estimatedComputeCapacity",
    "estimatedTaskComputeTimeSec",
    "estimatedTaskCompletionTimeSec",
    "estimatedTotalDelaySec",
    "delay",
    "energyDelta",
    "stepEnergyDeltaJ",
)


class DiscreteTargetMismatch(RuntimeError):
    pass


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _continuous_map(payload: Mapping[str, Any], key: str) -> Dict[str, float]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): _to_float(v) for k, v in raw.items()}


def _maps_differ(left: Mapping[str, float], right: Mapping[str, float], *, keys: Sequence[str]) -> bool:
    return any(abs(_to_float(left.get(key)) - _to_float(right.get(key))) > 1.0e-9 for key in keys)


def _metric_differences(receipt_a: Mapping[str, Any], receipt_b: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    diffs: Dict[str, Dict[str, float]] = {}
    for key in CONTROLLED_METRIC_KEYS:
        if key not in receipt_a or key not in receipt_b:
            continue
        left = _to_float(receipt_a.get(key))
        right = _to_float(receipt_b.get(key))
        if abs(left - right) > 1.0e-9:
            diffs[key] = {"a": left, "b": right, "delta": right - left}
    return diffs


def evaluate_binding_receipts(
    receipt_a: Mapping[str, Any],
    receipt_b: Mapping[str, Any],
    *,
    require_binding_version: str,
) -> Dict[str, Any]:
    version_a = str(receipt_a.get("lowerActionBindingVersion", "unbound"))
    version_b = str(receipt_b.get("lowerActionBindingVersion", "unbound"))
    requested_a = _continuous_map(receipt_a, "requestedContinuousAction")
    requested_b = _continuous_map(receipt_b, "requestedContinuousAction")
    applied_a = _continuous_map(receipt_a, "appliedContinuousAction")
    applied_b = _continuous_map(receipt_b, "appliedContinuousAction")
    share_keys = ("cpuShare", "bandwidthShare", "txPowerRatio")
    requested_differs = _maps_differ(requested_a, requested_b, keys=share_keys)
    applied_differs = _maps_differ(applied_a, applied_b, keys=share_keys)
    metric_diffs = _metric_differences(receipt_a, receipt_b)

    violations = []
    if version_a != require_binding_version or version_b != require_binding_version:
        violations.append(f"lower_action_binding_version_mismatch:{version_a},{version_b}")
    if not requested_differs:
        violations.append("requested_continuous_action_not_distinct")
    if not applied_differs:
        violations.append("applied_continuous_action_not_distinct")
    if not metric_diffs:
        violations.append("no_controlled_physical_metric_changed")

    if require_binding_version != "unbound" and (version_a == "unbound" or version_b == "unbound"):
        status = "STAGE_BLOCKED_FOR_FULL_HYBRID_CLAIM"
    else:
        status = "LOWER_ACTION_BINDING_OK" if not violations else "LOWER_ACTION_BINDING_FAILED"
    return {
        "status": status,
        "required_binding_version": require_binding_version,
        "binding_versions": {"a": version_a, "b": version_b},
        "requested_continuous_action_a": requested_a,
        "requested_continuous_action_b": requested_b,
        "applied_continuous_action_a": applied_a,
        "applied_continuous_action_b": applied_b,
        "requested_actions_differ": requested_differs,
        "applied_actions_differ": applied_differs,
        "controlled_metric_differences": metric_diffs,
        "violations": violations,
    }


def same_discrete_target(probe_a: Mapping[str, Any], probe_b: Mapping[str, Any]) -> bool:
    action_a = probe_a.get("action") if isinstance(probe_a.get("action"), Mapping) else {}
    action_b = probe_b.get("action") if isinstance(probe_b.get("action"), Mapping) else {}
    keys = ("policyUpperAction", "abstractAction", "targetVmIndex", "targetVmId", "selectedVmId")
    return all(str(action_a.get(key)) == str(action_b.get(key)) for key in keys)


def _extract_receipt(response: Mapping[str, Any]) -> Dict[str, Any]:
    raw = response.get("receipt", response)
    return dict(raw) if isinstance(raw, Mapping) else dict(response)


def _choose_action(state: Mapping[str, Any]) -> tuple[int, int, Dict[str, Any]]:
    mask = list(abstract_action_mask_from_state(dict(state)))
    preferred = [1, 2, 3, 0]
    for action in preferred:
        if action < len(mask) and bool(mask[action]):
            target, trace = map_upper_to_target_vm_with_trace(dict(state), action, require_visible=True)
            if target >= 0:
                return action, int(target), trace
    raise RuntimeError(f"no feasible target in state mask={mask}")


def _forced_action_trace(
    state: Mapping[str, Any],
    *,
    forced_action_index: int,
    forced_target_vm_index: int,
    forced_target_vm_id: Any,
) -> tuple[int, int, Dict[str, Any]]:
    state_dict = dict(state)
    mask = list(abstract_action_mask_from_state(state_dict))
    if forced_action_index < 0 or forced_action_index >= len(mask) or not bool(mask[forced_action_index]):
        raise DiscreteTargetMismatch(f"forced action {forced_action_index} is not visible in mask={mask}")
    candidates = list(state_dict.get("candidateVms") or [])
    if forced_target_vm_index < 0 or forced_target_vm_index >= len(candidates):
        raise DiscreteTargetMismatch(f"forced targetVmIndex {forced_target_vm_index} outside candidate list")
    candidate = candidates[forced_target_vm_index]
    actual_vm_id = candidate.get("vmId", candidate.get("id"))
    if str(actual_vm_id) != str(forced_target_vm_id):
        raise DiscreteTargetMismatch(f"forced targetVmId mismatch: expected {forced_target_vm_id}, got {actual_vm_id}")
    abstract_action = int(candidate.get("abstractAction", candidate.get("_abstractAction", -1)))
    if abstract_action != int(forced_action_index):
        raise DiscreteTargetMismatch(f"forced target abstractAction mismatch: expected {forced_action_index}, got {abstract_action}")
    trace = {
        "fallback_reason": "forced_same_discrete_target",
        "mapped_upper_action": int(forced_action_index),
        "selected_abstract_action": int(abstract_action),
        "selected_abstract_action_name": ACTION_NAMES[int(abstract_action)] if 0 <= int(abstract_action) < len(ACTION_NAMES) else "UNKNOWN",
        "selected_candidate_index": int(forced_target_vm_index),
        "selected_vm_index": int(forced_target_vm_index),
        "selected_vm_id": actual_vm_id,
        "selected_logical_tier": candidate.get("logicalTier", candidate.get("_level", "")),
        "selected_rate_mbps": _to_float(candidate.get("estimatedTransmissionRateMbps", candidate.get("bw", 0.0))),
        "selected_capacity": _to_float(candidate.get("estimatedComputeCapacity", 0.0)),
    }
    return forced_action_index, forced_target_vm_index, trace


def _task_id(state: Mapping[str, Any]) -> Any:
    task = state.get("task")
    if isinstance(task, Mapping):
        return task.get("id", state.get("taskId"))
    return state.get("taskId")


def _source_device_id(state: Mapping[str, Any]) -> Any:
    task = state.get("task")
    if isinstance(task, Mapping):
        return task.get("sourceDeviceId", state.get("sourceDeviceId"))
    return state.get("sourceDeviceId")


def _wait_for_action_state(client: SatEdgeSimClient, wait_timeout_ms: int) -> Dict[str, Any]:
    deadline = time.time() + max(1.0, wait_timeout_ms / 1000.0)
    state = client.get_state()
    while state.get("status") not in {"WAITING_FOR_ACTION", "FINISHED", "CLOSED", "FAILED", "ERROR"}:
        if time.time() >= deadline:
            raise RuntimeError(f"timed out waiting for WAITING_FOR_ACTION; last_status={state.get('status')}")
        time.sleep(0.05)
        state = client.get_state()
    return state


def _advance_to_source(
    client: SatEdgeSimClient,
    state: Mapping[str, Any],
    *,
    source_device_id: Any,
    wait_timeout_ms: int,
    max_steps: int,
) -> Dict[str, Any]:
    current = dict(state)
    for _ in range(max(0, max_steps)):
        if str(_source_device_id(current)) == str(source_device_id):
            return current
        action_index, target_vm_index, mapper_trace = _choose_action(current)
        action_name = ACTION_NAMES[action_index]
        action_payload: Dict[str, Any] = {
            "decisionId": current.get("decisionId"),
            "requestId": current.get("decisionId"),
            "taskId": _task_id(current),
            "policyUpperAction": action_index,
            "policyUpperActionName": action_name,
            "abstractAction": action_index,
            "abstractActionName": action_name,
            "targetVmIndex": target_vm_index,
            "targetVmId": mapper_trace.get("selected_vm_id", -1),
            "selectedVmId": mapper_trace.get("selected_vm_id", -1),
            **PROBE_A,
        }
        client.apply_action(action_payload)
        current = _wait_for_action_state(client, wait_timeout_ms)
    if str(_source_device_id(current)) != str(source_device_id):
        raise DiscreteTargetMismatch(f"could not reproduce sourceDeviceId={source_device_id}; got {_source_device_id(current)}")
    return current


def _apply_probe(
    *,
    base_url: str,
    seed: int,
    devices_count: int,
    wait_timeout_ms: int,
    continuous_action: Mapping[str, float],
    forced_action_index: int | None = None,
    forced_target_vm_index: int | None = None,
    forced_target_vm_id: Any = None,
    forced_source_device_id: Any = None,
) -> Dict[str, Any]:
    client = SatEdgeSimClient(base_url=base_url, timeout=max(10.0, wait_timeout_ms / 1000.0 + 5.0))
    try:
        client.ensure_healthy()
        state = client.reset(
            devices_count=devices_count,
            seed=seed,
            wait_for_first_decision=True,
            wait_timeout_ms=wait_timeout_ms,
            extra={
                "scenarioProfile": DEFAULT_SCENARIO_PROFILE,
                "taskSourceMode": DEFAULT_TASK_SOURCE_MODE,
                "successProfile": "paper_strict",
                "actionMaskMode": "completion_safe",
                "maxDecisions": 3,
            },
        )
        if forced_source_device_id is not None:
            state = _advance_to_source(
                client,
                state,
                source_device_id=forced_source_device_id,
                wait_timeout_ms=wait_timeout_ms,
                max_steps=max(32, devices_count * 20),
            )
        if forced_action_index is None:
            action_index, target_vm_index, mapper_trace = _choose_action(state)
        else:
            if forced_target_vm_index is None:
                raise RuntimeError("forced_target_vm_index is required when forced_action_index is set")
            action_index, target_vm_index, mapper_trace = _forced_action_trace(
                state,
                forced_action_index=forced_action_index,
                forced_target_vm_index=forced_target_vm_index,
                forced_target_vm_id=forced_target_vm_id,
            )
        action_name = ACTION_NAMES[action_index]
        action_payload: Dict[str, Any] = {
            "decisionId": state.get("decisionId"),
            "requestId": state.get("decisionId"),
            "taskId": _task_id(state),
            "policyUpperAction": action_index,
            "policyUpperActionName": action_name,
            "abstractAction": action_index,
            "abstractActionName": action_name,
            "targetVmIndex": target_vm_index,
            "targetVmId": mapper_trace.get("selected_vm_id", -1),
            "selectedVmId": mapper_trace.get("selected_vm_id", -1),
            **{key: float(value) for key, value in continuous_action.items()},
        }
        response = client.apply_action(action_payload)
        receipt = _extract_receipt(response)
        return {
            "state": {
                "decisionId": state.get("decisionId"),
                "taskId": _task_id(state),
                "sourceDeviceId": _source_device_id(state),
                "abstractActionMask": state.get("abstractActionMask"),
            },
            "action": action_payload,
            "mapper_trace": mapper_trace,
            "receipt": receipt,
        }
    finally:
        try:
            client.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SatEdgeSim lower continuous-action binding with an A/B replay.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--require-binding-version", default="vm_network_power_binding_v1")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--max-ab-attempts", type=int, default=8)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        probe_a = _apply_probe(
            base_url=args.base_url,
            seed=args.seed,
            devices_count=args.devices_count,
            wait_timeout_ms=args.wait_timeout_ms,
            continuous_action=PROBE_A,
        )
        action_a = probe_a["action"]
        probe_b = None
        mismatch_messages = []
        for _ in range(max(1, int(args.max_ab_attempts))):
            try:
                probe_b = _apply_probe(
                    base_url=args.base_url,
                    seed=args.seed,
                    devices_count=args.devices_count,
                    wait_timeout_ms=args.wait_timeout_ms,
                    continuous_action=PROBE_B,
                    forced_action_index=int(action_a["policyUpperAction"]),
                    forced_target_vm_index=int(action_a["targetVmIndex"]),
                    forced_target_vm_id=action_a["targetVmId"],
                    forced_source_device_id=probe_a["state"].get("sourceDeviceId"),
                )
                break
            except DiscreteTargetMismatch as exc:
                mismatch_messages.append(str(exc))
        if probe_b is None:
            raise RuntimeError(
                "could not reproduce the same discrete target for A/B replay; "
                f"attempts={max(1, int(args.max_ab_attempts))}; last_mismatch={mismatch_messages[-1] if mismatch_messages else '<none>'}"
            )
        payload = evaluate_binding_receipts(
            probe_a["receipt"],
            probe_b["receipt"],
            require_binding_version=args.require_binding_version,
        )
        payload["same_discrete_target"] = same_discrete_target(probe_a, probe_b)
        if not payload["same_discrete_target"]:
            payload["violations"].append("discrete_target_not_reproduced")
            payload["status"] = "LOWER_ACTION_BINDING_FAILED"
        payload["probe_a"] = probe_a
        payload["probe_b"] = probe_b
    except (SatEdgeSimClientError, RuntimeError) as exc:
        payload = {
            "status": "LOWER_ACTION_BINDING_FAILED",
            "required_binding_version": args.require_binding_version,
            "error": str(exc),
            "violations": ["ab_replay_failed"],
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if payload.get("status") != "LOWER_ACTION_BINDING_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
