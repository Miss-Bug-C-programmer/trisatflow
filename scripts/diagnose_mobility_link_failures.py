from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError

ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
TERMINAL = {"FINISHED", "FAILED", "CLOSED", "ERROR"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mask4(raw: Any) -> List[int]:
    if isinstance(raw, list) and len(raw) >= 4:
        return [1 if bool(raw[i]) else 0 for i in range(4)]
    return [0, 0, 0, 0]


def _state_masks(state: Dict[str, Any]) -> Dict[str, List[int]]:
    mode = str(state.get("actionMaskMode") or "visible_only").strip().lower()
    visible = _mask4(state.get("abstractActionMaskVisible") or state.get("abstractActionMask"))
    mobility = _mask4(state.get("abstractActionMaskMobilitySafe"))
    completion = _mask4(state.get("abstractActionMaskCompletionSafe"))
    active = visible
    if mode == "mobility_safe" and any(mobility):
        active = mobility
    elif mode == "completion_safe" and any(completion):
        active = completion
    return {"mode": mode, "visible": visible, "mobility": mobility, "completion": completion, "active": active}


def _tier_name(idx: int) -> str:
    return ACTION_NAMES[idx] if 0 <= idx < 4 else "unknown"


def _norm_phase(state: Dict[str, Any]) -> str:
    task = state.get("task") or {}
    return str(task.get("scenarioPhase", state.get("scenarioPhase", "unknown")))


def _norm_task_type(state: Dict[str, Any]) -> str:
    task = state.get("task") or {}
    return str(task.get("taskType", state.get("taskType", "unknown")))


def _choose_round_robin_visible(mask_visible: List[int], cursor: int) -> int:
    for offset in range(4):
        idx = (cursor + offset) % 4
        if idx < len(mask_visible) and mask_visible[idx]:
            return idx
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose mobility-link failures under live SatEdgeSim states.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="mixed_cost_landscape_v2")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--success-profile", type=str, default="paper_strict")
    parser.add_argument("--action-mask-mode", type=str, default="visible_only", choices=["visible_only", "mobility_safe", "completion_safe"])
    parser.add_argument("--num-decisions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--min-link-survival-margin-sec", type=float, default=0.0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    client = SatEdgeSimClient(args.base_url, timeout=args.request_timeout)
    client.ensure_healthy()
    state = client.reset(
        devices_count=args.devices_count,
        seed=args.seed,
        wait_for_first_decision=True,
        wait_timeout_ms=args.wait_timeout_ms,
        extra={
            "scenarioProfile": args.scenario_profile,
            "taskSourceMode": args.task_source_mode,
            "successProfile": args.success_profile,
            "actionMaskMode": args.action_mask_mode,
            "minLinkSurvivalMarginSec": float(max(0.0, args.min_link_survival_margin_sec)),
            "maxDecisions": int(args.num_decisions),
        },
    )

    tier_visible = [0, 0, 0, 0]
    tier_mob_safe = [0, 0, 0, 0]
    tier_comp_safe = [0, 0, 0, 0]
    tier_expected_mobility_fail = [0, 0, 0, 0]
    tier_samples = [0, 0, 0, 0]
    tier_risk_sum = [0.0, 0.0, 0.0, 0.0]
    phase_counter = Counter()
    task_counter = Counter()
    failure_by_tier = Counter()
    failure_by_phase = Counter()
    failure_by_task = Counter()
    forced_action_counter = Counter()
    applied = 0
    observed_mobility_fail = 0
    candidate_records: List[Dict[str, Any]] = []
    cursor = 0
    step = 0

    while step < args.num_decisions:
        if state.get("status") in TERMINAL:
            break
        if state.get("status") != "WAITING_FOR_ACTION":
            time.sleep(args.poll_sleep_sec)
            state = client.get_state()
            continue
        state = client.get_state()
        if state.get("status") != "WAITING_FOR_ACTION":
            continue

        masks = _state_masks(state)
        phase = _norm_phase(state)
        task_type = _norm_task_type(state)
        phase_counter[phase] += 1
        task_counter[task_type] += 1

        candidates = list(state.get("candidateVms") or [])
        for vm in candidates:
            action = int(_to_float(vm.get("abstractAction"), -1))
            if not (0 <= action <= 3):
                continue
            if not bool(vm.get("isFeasible", False)):
                continue
            tier_samples[action] += 1
            tier_risk_sum[action] += _to_float(vm.get("mobilityRisk"), 1.0)
            candidate_records.append(
                {
                    "step": step,
                    "phase": phase,
                    "task_type": task_type,
                    "action": _tier_name(action),
                    "visible": bool(masks["visible"][action]),
                    "mobilitySafe": bool(vm.get("mobilitySafe", False)),
                    "completionSafe": bool(vm.get("completionSafe", False)),
                    "estimatedLinkLifetimeSec": _to_float(vm.get("estimatedLinkLifetimeSec"), 0.0),
                    "estimatedTaskTransmissionTimeSec": _to_float(vm.get("estimatedTaskTransmissionTimeSec"), 0.0),
                    "estimatedTaskCompletionTimeSec": _to_float(vm.get("estimatedTaskCompletionTimeSec"), 0.0),
                    "linkSurvivalMarginSec": _to_float(vm.get("linkSurvivalMarginSec"), 0.0),
                    "linkSurvivalMarginToCompletionSec": _to_float(vm.get("linkSurvivalMarginToCompletionSec"), 0.0),
                    "mobilityRisk": _to_float(vm.get("mobilityRisk"), 1.0),
                    "mobilityRiskSource": vm.get("mobilityRiskSource", "unavailable"),
                }
            )

        for a in range(4):
            tier_visible[a] += int(bool(masks["visible"][a]))
            tier_mob_safe[a] += int(bool(masks["mobility"][a]))
            tier_comp_safe[a] += int(bool(masks["completion"][a]))
            if masks["visible"][a] and not masks["completion"][a]:
                tier_expected_mobility_fail[a] += 1

        desired = cursor % 4
        cursor += 1
        chosen = desired if masks["visible"][desired] else _choose_round_robin_visible(masks["visible"], desired)
        forced_action_counter[_tier_name(chosen)] += 1
        target_idx, trace = map_upper_to_target_vm_with_trace(state, chosen, require_visible=False)
        if target_idx < 0:
            chosen = _choose_round_robin_visible(masks["visible"], chosen)
            target_idx, trace = map_upper_to_target_vm_with_trace(state, chosen, require_visible=False)
        if target_idx < 0:
            step += 1
            time.sleep(args.poll_sleep_sec)
            state = client.get_state()
            continue

        before = client.get_metrics()
        task = state.get("task") or {}
        action_payload = {
            "decisionId": int(state.get("decisionId", state.get("requestId", -1))),
            "requestId": int(state.get("decisionId", state.get("requestId", -1))),
            "taskId": int(task.get("id", state.get("taskId", -1))),
            "targetVmIndex": int(target_idx),
            "targetVmId": int(_to_float(trace.get("selected_vm_id"), -1)),
            "selectedVmId": int(_to_float(trace.get("selected_vm_id"), -1)),
            "policyUpperAction": int(chosen),
            "policyUpperActionName": ACTION_NAMES[chosen].upper(),
            "abstractAction": int(chosen),
            "abstractActionName": ACTION_NAMES[chosen].upper(),
            "cpuShare": 1.0,
            "bandwidthShare": 1.0,
            "txPowerRatio": 1.0,
            "queuePriority": 1.0,
            "extra": {"diagnoseMobility": True},
        }
        try:
            receipt = client.apply_action(action_payload)
        except SatEdgeSimClientError as exc:
            # Live bridge may advance to RUNNING between polling and POST /apply_action.
            # Treat transient "no pending decision" as a resync event instead of aborting diagnosis.
            if exc.status_code == 409 and exc.error_type == "no_pending_decision":
                time.sleep(args.poll_sleep_sec)
                state = client.get_state()
                continue
            raise
        after = client.get_metrics()
        mobility_delta = int(_to_float(after.get("tasksFailedMobility"), 0.0) - _to_float(before.get("tasksFailedMobility"), 0.0))
        if mobility_delta > 0:
            observed_mobility_fail += mobility_delta
            failure_by_tier[_tier_name(chosen)] += mobility_delta
            failure_by_phase[phase] += mobility_delta
            failure_by_task[task_type] += mobility_delta
        applied += 1
        step += 1
        state = client.get_state()
        if state.get("status") in TERMINAL:
            break

    final_metrics = client.get_metrics()
    try:
        client.close()
    except Exception:
        pass

    total_states = max(1, step)
    expected_fail_ratio = [tier_expected_mobility_fail[a] / total_states for a in range(4)]
    observed_ratio = observed_mobility_fail / max(1, applied)
    mean_risk = [0.0 if tier_samples[a] <= 0 else tier_risk_sum[a] / tier_samples[a] for a in range(4)]
    mobility_safe_ratio = [tier_mob_safe[a] / total_states for a in range(4)]
    completion_safe_ratio = [tier_comp_safe[a] / total_states for a in range(4)]

    diagnosis = []
    if any(v > 0.85 for v in [tier_visible[1] / total_states, tier_visible[2] / total_states, tier_visible[3] / total_states]) and observed_ratio > 0.20:
        diagnosis.append("mask_missing_link_lifetime")
    if any(expected_fail_ratio[a] > 0.20 for a in range(4)):
        diagnosis.append("completion_time_exceeds_link_lifetime")
    if completion_safe_ratio[3] < 0.30:
        diagnosis.append("ground_visibility_window_too_short")
    if completion_safe_ratio[1] < 0.30:
        diagnosis.append("neighbor_link_lifetime_too_short")
    if mean_risk[2] > 0.65 and completion_safe_ratio[2] > 0.5:
        diagnosis.append("geo_link_model_inconsistent")
    if max(mean_risk[1], mean_risk[2], mean_risk[3]) > 0.65:
        diagnosis.append("task_transmission_time_too_long")
    if observed_ratio > 0.20 and float(final_metrics.get("mobilityFailureRate", 0.0)) <= 0.05:
        diagnosis.append("mobility_failure_metric_mapping_bug")
    if observed_ratio > 0.15 and args.action_mask_mode == "visible_only":
        diagnosis.append("expected_mobility_stress_constraint")
    if not diagnosis:
        diagnosis = ["expected_mobility_stress_constraint"]

    output = {
        "num_states": step,
        "num_applied_actions": applied,
        "scenario_profile": args.scenario_profile,
        "task_source_mode": args.task_source_mode,
        "success_profile": args.success_profile,
        "action_mask_mode": args.action_mask_mode,
        "min_link_survival_margin_sec": float(max(0.0, args.min_link_survival_margin_sec)),
        "forced_action_distribution": dict(forced_action_counter),
        "mobility_risk_mean_by_tier": {ACTION_NAMES[a]: mean_risk[a] for a in range(4)},
        "mobility_safe_ratio_by_tier": {ACTION_NAMES[a]: mobility_safe_ratio[a] for a in range(4)},
        "completion_safe_ratio_by_tier": {ACTION_NAMES[a]: completion_safe_ratio[a] for a in range(4)},
        "expected_mobility_failure_ratio_by_tier": {ACTION_NAMES[a]: expected_fail_ratio[a] for a in range(4)},
        "observed_mobility_link_failure_ratio": observed_ratio,
        "observed_mobility_failure_count": observed_mobility_fail,
        "failure_by_tier": dict(failure_by_tier),
        "failure_by_phase": dict(failure_by_phase),
        "failure_by_task_type": dict(failure_by_task),
        "phase_distribution": dict(phase_counter),
        "task_type_distribution": dict(task_counter),
        "final_metrics": final_metrics,
        "diagnosis": diagnosis,
        "candidate_sample_size": len(candidate_records),
        "candidate_samples_head": candidate_records[:200],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
