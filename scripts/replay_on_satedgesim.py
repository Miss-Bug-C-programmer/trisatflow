from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satedgesim_semantics import (
    completion_success_ratio,
    energy_semantics,
    has_completion_evidence,
    resource_binding_semantics,
    validation_metadata,
)
from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.baselines.registry import apply_architecture_filter, normalize_architecture
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.state_adapter import (
    build_trisatflow_observation,
    is_controlled_rl_scenario_from_state,
    scenario_profile_from_state,
    source_leo_id_from_state,
    task_source_mode_from_state,
)

TERMINAL_STATUSES = {"FINISHED", "CLOSED", "FAILED", "ERROR"}
ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _flatten_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[f"{prefix}{key}"] = value
    return flat


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _action_name(action: Any) -> str:
    try:
        idx = int(action)
    except (TypeError, ValueError):
        return ""
    return ACTION_NAMES[idx].upper() if 0 <= idx < len(ACTION_NAMES) else ""


def _mask4(raw: Any) -> List[int]:
    if isinstance(raw, list) and len(raw) >= 4:
        return [1 if bool(raw[i]) else 0 for i in range(4)]
    return [0, 0, 0, 0]


def _state_masks(state: Dict[str, Any]) -> Dict[str, List[int]]:
    mode = str(state.get("actionMaskMode") or "visible_only").strip().lower()
    visible = _mask4(state.get("abstractActionMaskVisible") or state.get("abstractActionMask"))
    mobility_safe = _mask4(state.get("abstractActionMaskMobilitySafe"))
    completion_safe = _mask4(state.get("abstractActionMaskCompletionSafe"))
    active = visible
    if mode == "mobility_safe" and any(mobility_safe):
        active = mobility_safe
    elif mode == "completion_safe" and any(completion_safe):
        active = completion_safe
    return {
        "mode": mode,
        "visible": visible,
        "mobility_safe": mobility_safe,
        "completion_safe": completion_safe,
        "active": active,
    }


def _selected_candidate_from_state(state: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(state.get("candidateVms") or [])
    idx = trace_to_int(trace.get("selected_candidate_index"), -1)
    if 0 <= idx < len(candidates):
        return dict(candidates[idx] or {})
    selected_vm_id = trace_to_int(trace.get("selected_vm_id"), -1)
    if selected_vm_id >= 0:
        for vm in candidates:
            if trace_to_int((vm or {}).get("vmId"), -1) == selected_vm_id:
                return dict(vm or {})
    return {}


def _wait_for_state(client: SatEdgeSimClient, poll_sleep_sec: float, max_polls: int = 300) -> Dict[str, Any]:
    state = client.get_state()
    polls = 0
    while state.get("status") == "RUNNING" and polls < max_polls:
        time.sleep(poll_sleep_sec)
        state = client.get_state()
        polls += 1
    return state


def _parse_force_upper_action(value: str) -> int | None:
    text = str(value or "none").strip().lower()
    if text in {"", "none", "-1"}:
        return None
    if text not in ACTION_INDEX:
        raise ValueError(f"unsupported --force-upper-action={value!r}")
    return ACTION_INDEX[text]


def _random_visible_action(mask: List[int], rng: random.Random) -> int:
    visible = [idx for idx, bit in enumerate(mask[:4]) if bool(bit)]
    return rng.choice(visible) if visible else 0


def _round_robin_visible_action(mask: List[int], cursor: int) -> int:
    for offset in range(4):
        idx = (cursor + offset) % 4
        if idx < len(mask) and bool(mask[idx]):
            return idx
    return 0


def _checkpoint_action(
    policy: FrozenTriSatFlowPolicy,
    state: Dict[str, Any],
    *,
    deterministic: bool,
    eval_mode: str,
    tie_eps: float,
    stochastic_seed: int,
) -> tuple[int, List[float], Dict[str, Any]]:
    obs, edge_index, edge_attr, source_index = build_trisatflow_observation(
        state,
        node_feature_dim=policy.cfg.scenario.node_feature_dim,
        normalization_mode=policy.obs_normalization_mode,
        normalization_stats=policy.obs_normalization_stats,
    )
    raw_rows = list(state.get("denseSourceSummaries") or [])
    upper_action, lower_action, debug = policy.act(
        obs,
        edge_index,
        edge_attr,
        source_index=source_index,
        deterministic=deterministic,
        eval_mode=eval_mode,
        tie_break_eps=tie_eps,
        stochastic_seed=stochastic_seed + max(0, int(source_index)),
        raw_rows=raw_rows,
    )
    debug["source_index"] = int(source_index)
    debug["obs_feature_dim"] = int(obs.shape[1]) if obs.ndim == 2 else 0
    debug["obs_normalization_mode"] = policy.obs_normalization_mode
    debug["obs_normalization_path"] = policy.obs_normalization_path
    debug["obs_normalization_loaded"] = bool(policy.obs_normalization_loaded)
    return int(upper_action), list(lower_action), debug


def _select_action(
    state: Dict[str, Any],
    *,
    policy: FrozenTriSatFlowPolicy | None,
    deterministic: bool,
    eval_mode: str,
    tie_eps: float,
    stochastic_seed: int,
    force_policy: str,
    forced_upper_action: int | None,
    rng: random.Random,
    round_robin_cursor: int,
    architecture: str,
    continuous_resource_binding_mode: str,
) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    decision_id = int(state.get("decisionId", state.get("requestId", -1)))
    task = dict(state.get("task") or {})
    task_id = int(task.get("id", state.get("taskId", -1)))
    mask_raw = abstract_action_mask_from_state(state)
    mask = apply_architecture_filter(mask_raw, architecture)
    checkpoint_upper = None
    checkpoint_debug: Dict[str, Any] = {}
    lower_action = [1.0, 1.0, 1.0]
    source_index = 0
    inference_t0 = time.perf_counter()
    if force_policy == "checkpoint":
        if policy is not None:
            checkpoint_upper, lower_action, checkpoint_debug = _checkpoint_action(
                policy,
                state,
                deterministic=deterministic,
                eval_mode=eval_mode,
                tie_eps=tie_eps,
                stochastic_seed=stochastic_seed,
            )
            source_index = int(checkpoint_debug.get("source_index", 0))
        else:
            checkpoint_upper = int(forced_upper_action if forced_upper_action is not None else 0)
    elif force_policy == "random_visible":
        checkpoint_upper = _random_visible_action(mask, rng)
    elif force_policy == "round_robin_visible":
        checkpoint_upper = _round_robin_visible_action(mask, round_robin_cursor)
    else:
        raise ValueError(f"unsupported force policy: {force_policy}")
    inference_ms = (time.perf_counter() - inference_t0) * 1000.0

    selected_upper = checkpoint_upper
    action_origin = force_policy
    if forced_upper_action is not None:
        selected_upper = int(forced_upper_action)
        action_origin = "forced_upper_action"
    if not (0 <= int(selected_upper) < len(mask) and bool(mask[int(selected_upper)])):
        selected_upper = _random_visible_action(mask, rng)
        action_origin = f"{action_origin}_arch_filtered"

    require_visible = forced_upper_action is not None
    missing_reason = "forced_action_not_visible" if forced_upper_action is not None else "requested_action_not_visible"
    target_vm_index, mapper_trace = map_upper_to_target_vm_with_trace(
        state,
        int(selected_upper),
        require_visible=require_visible,
        missing_reason=missing_reason,
    )
    if forced_upper_action is not None and not bool(mask[int(forced_upper_action)]):
        mapper_trace["fallback_reason"] = "forced_action_not_visible"

    extra = {
        "upperAction": int(selected_upper),
        "policyUpperAction": int(selected_upper),
        "policyRawUpperAction": int(checkpoint_upper if checkpoint_upper is not None else selected_upper),
        "rawArgmaxAction": int(checkpoint_debug.get("raw_argmax_action", checkpoint_upper if checkpoint_upper is not None else selected_upper)),
        "finalPolicyAction": int(selected_upper),
        "tieBreakApplied": bool(checkpoint_debug.get("tie_break_applied", False)),
        "tieBreakCandidateActions": checkpoint_debug.get("tie_break_candidate_actions", []),
        "selectedByPolicyProb": checkpoint_debug.get("selected_by_policy_prob"),
        "selectedByCostRank": checkpoint_debug.get("selected_by_cost_rank"),
        "evalMode": eval_mode,
        "tieBreakEps": float(tie_eps),
        "actionOrigin": action_origin,
        "sourceIndex": int(source_index),
        "selectedLevel": mapper_trace.get("selected_level"),
        "desiredLevel": mapper_trace.get("desired_level"),
        "selectedAbstractAction": mapper_trace.get("selected_abstract_action"),
        "abstractActionMask": mask,
        "architecture": architecture,
        "forcePolicy": force_policy,
        "forceUpperAction": None if forced_upper_action is None else int(forced_upper_action),
        "continuous_resource_binding_mode": continuous_resource_binding_mode,
        "bindingMode": continuous_resource_binding_mode,
    }
    if policy is not None:
        extra["policyDevice"] = str(policy.device)
        extra["policyCheckpoint"] = str(policy.checkpoint_path)

    action = {
        "decisionId": decision_id,
        "requestId": decision_id,
        "taskId": task_id,
        "targetVmIndex": int(target_vm_index),
        "targetVmId": int(trace_to_int(mapper_trace.get("selected_vm_id"), -1)),
        "selectedVmId": int(trace_to_int(mapper_trace.get("selected_vm_id"), -1)),
        "policyUpperAction": int(selected_upper),
        "policyUpperActionName": ACTION_NAMES[int(selected_upper)].upper(),
        "abstractAction": int(selected_upper),
        "abstractActionName": ACTION_NAMES[int(selected_upper)].upper(),
        "cpuShare": float(lower_action[0]),
        "bandwidthShare": float(lower_action[1]),
        "txPowerRatio": float(lower_action[2]),
        "queuePriority": 1.0,
        "extra": extra,
    }
    trace = {
        "decision_id": decision_id,
        "task_id": task_id,
        "source_index": int(source_index),
        "policy_upper_action": int(selected_upper),
        "policy_raw_upper_action": int(checkpoint_upper if checkpoint_upper is not None else selected_upper),
        "raw_argmax_action": int(checkpoint_debug.get("raw_argmax_action", checkpoint_upper if checkpoint_upper is not None else selected_upper)),
        "final_policy_action": int(selected_upper),
        "tie_break_applied": int(bool(checkpoint_debug.get("tie_break_applied", False))),
        "tie_break_candidate_actions": json.dumps(checkpoint_debug.get("tie_break_candidate_actions", []), ensure_ascii=False),
        "selected_by_policy_prob": checkpoint_debug.get("selected_by_policy_prob"),
        "selected_by_cost_rank": checkpoint_debug.get("selected_by_cost_rank"),
        "target_vm_index": int(target_vm_index),
        "abstract_mask_local": int(bool(mask[0])),
        "abstract_mask_neighbor": int(bool(mask[1])),
        "abstract_mask_geo": int(bool(mask[2])),
        "abstract_mask_ground": int(bool(mask[3])),
        "inference_ms": inference_ms,
        "action_origin": action_origin,
        "eval_mode": eval_mode,
        "tie_break_eps": float(tie_eps),
    }
    trace.update(mapper_trace)
    checkpoint_debug["force_policy"] = force_policy
    checkpoint_debug["action_origin"] = action_origin
    checkpoint_debug["source_index"] = int(source_index)
    return action, checkpoint_debug, trace


def trace_to_int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SatEdgeSim with a TriSatFlow checkpoint or forced visible-action policy.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm-index", type=int, default=0)
    parser.add_argument("--architecture-index", type=int, default=0)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--eval-mode", type=str, default="raw_argmax", choices=["raw_argmax", "stochastic_eval", "margin_cost_tiebreak", "cost_greedy_baseline"])
    parser.add_argument("--tie-eps", type=float, default=0.05)
    parser.add_argument("--stochastic-seed", type=int, default=13)
    parser.add_argument("--force-upper-action", type=str, default="none", choices=["local", "neighbor", "geo", "ground", "none"])
    parser.add_argument("--force-policy", type=str, default="checkpoint", choices=["checkpoint", "random_visible", "round_robin_visible"])
    parser.add_argument("--scenario-profile", type=str, default="default")
    parser.add_argument("--task-source-mode", type=str, default="current")
    parser.add_argument("--success-profile", type=str, default="default", choices=["default", "preflight_lenient", "paper_strict"])
    parser.add_argument("--action-mask-mode", type=str, default="visible_only", choices=["visible_only", "mobility_safe", "completion_safe"])
    parser.add_argument("--min-link-survival-margin-sec", type=float, default=0.0)
    parser.add_argument("--clean-output-folder", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--retry-apply-action", type=int, default=0)
    parser.add_argument("--architecture", type=str, default="full", choices=["only_leo", "leo_geo", "leo_ground", "full"])
    parser.add_argument(
        "--continuous-resource-binding-mode",
        type=str,
        default="resource_aware_estimator_bound",
        choices=["candidate_only", "resource_aware_estimator_bound", "native_scheduler_bound"],
        help="How SatEdgeSim should bind cpuShare/bandwidthShare/txPowerRatio. Use native_scheduler_bound only with a SatEdgeSim build that exposes native binding evidence.",
    )
    args = parser.parse_args()
    args.architecture = normalize_architecture(args.architecture)

    continuous_resource_binding_mode = args.continuous_resource_binding_mode
    forced_upper_action = _parse_force_upper_action(args.force_upper_action)
    if args.force_policy == "checkpoint" and not args.checkpoint and forced_upper_action is None:
        raise SystemExit("--checkpoint is required when --force-policy=checkpoint")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = SatEdgeSimClient(args.base_url, timeout=args.request_timeout)
    started_at = time.time()
    rng = random.Random(args.seed)
    decision_rows: List[Dict[str, Any]] = []
    health: Dict[str, Any] = {}
    final_state: Dict[str, Any] = {}
    final_metrics: Dict[str, Any] = {}
    close_response: Dict[str, Any] = {"status": "SKIPPED_BY_DEFAULT"}
    failure_message: str | None = None
    policy: FrozenTriSatFlowPolicy | None = None
    state: Dict[str, Any] = {}
    round_robin_cursor = 0

    try:
        health = client.ensure_healthy()
        if args.checkpoint:
            policy = FrozenTriSatFlowPolicy(args.checkpoint, device=args.device)

        state = client.reset(
            devices_count=args.devices_count,
            algorithm_index=args.algorithm_index,
            architecture_index=args.architecture_index,
            seed=args.seed,
            clean_output_folder=args.clean_output_folder,
            wait_for_first_decision=True,
            wait_timeout_ms=args.wait_timeout_ms,
            extra={
                "scenarioProfile": args.scenario_profile,
                "taskSourceMode": args.task_source_mode,
                "successProfile": args.success_profile,
                "actionMaskMode": args.action_mask_mode,
                "minLinkSurvivalMarginSec": max(0.0, float(args.min_link_survival_margin_sec)),
                "maxDecisions": int(args.max_decisions),
            },
        )

        decision_step = 0
        while decision_step < args.max_decisions:
            if state.get("status") in TERMINAL_STATUSES:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                state = _wait_for_state(client, args.poll_sleep_sec)
                continue

            state = client.get_state()
            if state.get("status") in TERMINAL_STATUSES:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                continue

            task = dict(state.get("task") or {})
            source_leo_id = int(source_leo_id_from_state(state))
            action, debug, trace = _select_action(
                state,
                policy=policy,
                deterministic=not args.stochastic,
                eval_mode=args.eval_mode,
                tie_eps=args.tie_eps,
                stochastic_seed=args.stochastic_seed + decision_step,
                force_policy=args.force_policy,
                forced_upper_action=forced_upper_action,
                rng=rng,
                round_robin_cursor=round_robin_cursor,
                architecture=args.architecture,
                continuous_resource_binding_mode=args.continuous_resource_binding_mode,
            )
            round_robin_cursor = (round_robin_cursor + 1) % 4
            selected_candidate = _selected_candidate_from_state(state, trace)
            masks = _state_masks(state)

            row: Dict[str, Any] = {
                "step": int(decision_step),
                "state_decision_id": trace["decision_id"],
                "state_task_id": trace["task_id"],
                "task_id": trace["task_id"],
                "source_leo_id": source_leo_id,
                "scenario_profile": scenario_profile_from_state(state),
                "task_source_mode": task_source_mode_from_state(state),
                "profile_name": args.profile_name,
                "architecture": args.architecture,
                "is_controlled_rl_scenario": is_controlled_rl_scenario_from_state(state),
                "policy_upper_action": int(trace["policy_upper_action"]),
                "policy_upper_action_name": ACTION_NAMES[int(trace["policy_upper_action"])].upper(),
                "policy_raw_upper_action": int(trace["policy_raw_upper_action"]),
                "policy_raw_upper_action_name": ACTION_NAMES[int(trace["policy_raw_upper_action"])].upper(),
                "raw_argmax_action": int(trace.get("raw_argmax_action", trace["policy_raw_upper_action"])),
                "raw_argmax_action_name": ACTION_NAMES[int(trace.get("raw_argmax_action", trace["policy_raw_upper_action"]))].upper(),
                "final_policy_action": int(trace.get("final_policy_action", trace["policy_upper_action"])),
                "final_policy_action_name": ACTION_NAMES[int(trace.get("final_policy_action", trace["policy_upper_action"]))].upper(),
                "abstract_action_mask": json.dumps(
                    [int(masks["active"][0]), int(masks["active"][1]), int(masks["active"][2]), int(masks["active"][3])]
                ),
                "abstract_action_mask_visible": json.dumps([int(masks["visible"][0]), int(masks["visible"][1]), int(masks["visible"][2]), int(masks["visible"][3])]),
                "abstract_action_mask_mobility_safe": json.dumps([int(masks["mobility_safe"][0]), int(masks["mobility_safe"][1]), int(masks["mobility_safe"][2]), int(masks["mobility_safe"][3])]),
                "abstract_action_mask_completion_safe": json.dumps([int(masks["completion_safe"][0]), int(masks["completion_safe"][1]), int(masks["completion_safe"][2]), int(masks["completion_safe"][3])]),
                "action_mask_mode": masks["mode"],
                "selected_vm_id": trace.get("selected_vm_id"),
                "selected_vm_logical_tier": trace.get("selected_logical_tier"),
                "selected_vm_abstract_action": trace.get("selected_abstract_action"),
                "selectedVmId": trace.get("selected_vm_id"),
                "selectedVmLogicalTier": trace.get("selected_logical_tier"),
                "policyUpperActionName": ACTION_NAMES[int(trace["policy_upper_action"])].upper(),
                "scenario_phase": task.get("scenarioPhase", state.get("scenarioPhase")),
                "task_type": task.get("taskType", state.get("taskType")),
                "estimatedTotalDelaySec": selected_candidate.get("estimatedTotalDelaySec", trace.get("selected_delay")),
                "estimatedQueueLength": selected_candidate.get("estimatedQueueLength", trace.get("selected_queue")),
                "estimatedTransmissionRateMbps": selected_candidate.get("estimatedTransmissionRateMbps", trace.get("selected_rate_mbps")),
                "estimatedComputeCapacity": selected_candidate.get("estimatedComputeCapacity", trace.get("selected_capacity")),
                "linkAvailableNow": selected_candidate.get("linkAvailableNow", selected_candidate.get("linkAvailable")),
                "estimatedLinkLifetimeSec": selected_candidate.get("estimatedLinkLifetimeSec"),
                "estimatedTaskTransmissionTimeSec": selected_candidate.get("estimatedTaskTransmissionTimeSec", selected_candidate.get("estimatedTransmissionDelaySec")),
                "estimatedTaskComputeTimeSec": selected_candidate.get("estimatedTaskComputeTimeSec", selected_candidate.get("estimatedComputeDelaySec")),
                "estimatedTaskCompletionTimeSec": selected_candidate.get("estimatedTaskCompletionTimeSec", selected_candidate.get("estimatedTotalDelaySec")),
                "linkSurvivalMarginSec": selected_candidate.get("linkSurvivalMarginSec"),
                "linkSurvivalMarginToCompletionSec": selected_candidate.get("linkSurvivalMarginToCompletionSec"),
                "handoverRequired": selected_candidate.get("handoverRequired"),
                "handoverAvailable": selected_candidate.get("handoverAvailable"),
                "mobilityRisk": selected_candidate.get("mobilityRisk"),
                "mobilityRiskSource": selected_candidate.get("mobilityRiskSource"),
                "candidateMobilitySafe": selected_candidate.get("mobilitySafe"),
                "candidateCompletionSafe": selected_candidate.get("completionSafe"),
                "executed_logical_tier": "",
                "executed_abstract_action": None,
                "receipt_accepted": None,
                "receipt_decision_id": None,
                "receipt_task_id": None,
                "fallback_reason": str(trace.get("fallback_reason") or "none"),
                "reward": None,
                "delay": None,
                "deadline": task.get("maxLatency"),
                "queueLength": selected_candidate.get("estimatedQueueLength", trace.get("selected_queue")),
                "energy": None,
                "success": None,
                "failureReason": None,
                "deadlineMiss": None,
                "queueOverflow": None,
                "vmUnavailable": None,
                "linkUnavailable": None,
                "taskDropped": None,
                "latencyExceeded": None,
                "resourceExceeded": None,
                "server_processing_ms": None,
                "client_elapsed_ms": None,
                "http_status_code": None,
                "http_error": "",
                "decision_id": trace["decision_id"],
                "target_vm_index": trace["target_vm_index"],
                "source_index": trace.get("source_index"),
                "inference_ms": trace.get("inference_ms"),
                "action_origin": trace.get("action_origin"),
                "eval_mode": trace.get("eval_mode", args.eval_mode),
                "tie_break_eps": trace.get("tie_break_eps", args.tie_eps),
                "tie_break_applied": int(trace.get("tie_break_applied", 0)),
                "tie_break_candidate_actions": trace.get("tie_break_candidate_actions", "[]"),
                "selected_by_policy_prob": trace.get("selected_by_policy_prob"),
                "selected_by_cost_rank": trace.get("selected_by_cost_rank"),
                "desired_level": trace.get("desired_level"),
                "selected_level": trace.get("selected_level"),
                "selected_candidate_index": trace.get("selected_candidate_index"),
                "selected_distance": trace.get("selected_distance"),
                "selected_queue": trace.get("selected_queue"),
                "selected_capacity": trace.get("selected_capacity"),
                "selected_rate_mbps": trace.get("selected_rate_mbps"),
                "selected_prop_delay_sec": trace.get("selected_prop_delay_sec"),
                "cpu_share": action["cpuShare"],
                "bandwidth_share": action["bandwidthShare"],
                "tx_power_ratio": action["txPowerRatio"],
                "policy_device": str(policy.device) if policy is not None else args.device,
                "requested_device": debug.get("requested_device", args.device),
                "device_fallback_reason": debug.get("device_fallback_reason"),
                "obs_feature_dim": debug.get("obs_feature_dim"),
                "obs_normalization_mode": debug.get("obs_normalization_mode"),
                "obs_normalization_path": debug.get("obs_normalization_path"),
                "obs_normalization_loaded": int(bool(debug.get("obs_normalization_loaded", False))),
                "upper_algo": debug.get("upper_algo"),
                "lower_algo": debug.get("lower_algo"),
                "force_policy": args.force_policy,
                "force_upper_action": args.force_upper_action,
                "success_profile": args.success_profile,
                "min_link_survival_margin_sec": max(0.0, float(args.min_link_survival_margin_sec)),
                "profile_name": args.profile_name,
                "architecture": args.architecture,
            }
            row.update(_flatten_metrics(state.get("metrics") or {}, prefix="before_metric_"))

            receipt: Dict[str, Any] | None = None
            client_elapsed_ms = None
            http_error = ""
            for attempt in range(args.retry_apply_action + 1):
                t0 = time.perf_counter()
                try:
                    receipt = client.apply_action(action)
                    client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    break
                except SatEdgeSimClientError as exc:
                    client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    http_error = str(exc)
                    if attempt >= args.retry_apply_action:
                        receipt = {
                            "accepted": False,
                            "actionAccepted": False,
                            "executionScheduled": False,
                            "taskCompleted": False,
                            "taskSucceeded": False,
                            "decisionId": trace["decision_id"],
                            "taskId": trace["task_id"],
                            "executedAbstractAction": -1,
                            "executedLogicalTier": "",
                            "executedVmId": -1,
                            "intentExecutionMatch": False,
                            "fallbackReason": exc.error_type,
                            "failureReason": exc.error_type,
                            "deadlineMiss": False,
                            "queueOverflow": False,
                            "vmUnavailable": False,
                            "linkUnavailable": False,
                            "taskDropped": False,
                            "latencyExceeded": False,
                            "resourceExceeded": False,
                            "estimatedLinkLifetimeSec": 0.0,
                            "estimatedTaskTransmissionTimeSec": 0.0,
                            "estimatedTaskComputeTimeSec": 0.0,
                            "estimatedTaskCompletionTimeSec": 0.0,
                            "linkSurvivalMarginSec": -1.0,
                            "linkSurvivalMarginToCompletionSec": -1.0,
                            "linkAvailableNow": False,
                            "handoverRequired": False,
                            "handoverAvailable": False,
                            "mobilityRisk": 1.0,
                            "mobilityRiskSource": "unavailable",
                            "serverProcessingMs": None,
                            "_httpStatusCode": exc.status_code,
                            "success": False,
                            "delay": None,
                            "energyDelta": None,
                            "energyRawCounterBefore": None,
                            "energyRawCounterAfter": None,
                        }
                    else:
                        time.sleep(args.poll_sleep_sec)

            try:
                live_metrics = client.get_metrics()
            except Exception:
                live_metrics = state.get("metrics") or {}
            row.update(_flatten_metrics(live_metrics or {}, prefix="after_metric_"))
            executed_action = receipt.get("executedAbstractAction")
            try:
                executed_action = int(executed_action)
            except (TypeError, ValueError):
                executed_action = None
            fallback_reason = str(receipt.get("fallbackReason") or row["fallback_reason"] or "none")
            if row["fallback_reason"] == "forced_action_not_visible" and not bool(receipt.get("accepted", False)):
                fallback_reason = "forced_action_not_visible"

            row["receipt_accepted"] = int(bool(receipt.get("accepted", False)))
            row["receipt_decision_id"] = receipt.get("decisionId")
            row["receipt_task_id"] = receipt.get("taskId")
            row["server_processing_ms"] = receipt.get("serverProcessingMs")
            row["client_elapsed_ms"] = client_elapsed_ms
            row["http_status_code"] = receipt.get("_httpStatusCode")
            row["http_error"] = http_error
            row["reward"] = receipt.get("reward", live_metrics.get("reward"))
            row["delay"] = receipt.get("delay")
            row["deadline"] = receipt.get("deadline", row.get("deadline"))
            row["queueLength"] = receipt.get("queueLength", row.get("queueLength"))
            row["raw_energy_counter_before"] = receipt.get("energyRawCounterBefore")
            row["raw_energy_counter_after"] = receipt.get("energyRawCounterAfter")
            row["energy_raw_delta"] = receipt.get("energyDelta")
            row["energy"] = receipt.get("energyDelta")
            row["energy_source"] = receipt.get("energySource", "receipt_delta")
            row["energy_unit"] = receipt.get("energyUnit", live_metrics.get("energyCounterUnit", "Wh"))
            row["continuous_resource_binding_mode"] = receipt.get("continuous_resource_binding_mode", receipt.get("continuousResourceBindingMode"))
            row["continuous_resource_applied"] = receipt.get("continuous_resource_applied", receipt.get("continuousResourceApplied"))
            row["native_scheduler_bound"] = receipt.get("native_scheduler_bound", receipt.get("nativeSchedulerBound"))
            row["estimator_bound"] = receipt.get("estimator_bound", receipt.get("estimatorBound"))
            row["full_hybrid_closed_loop_claim_allowed"] = receipt.get(
                "full_hybrid_closed_loop_claim_allowed",
                receipt.get("fullHybridClosedLoopClaimAllowed"),
            )
            row["estimatorExpectedDelaySec"] = receipt.get("estimatorExpectedDelaySec")
            row["estimatorExpectedEnergyJ"] = receipt.get("estimatorExpectedEnergyJ")
            row["estimatorEffectiveMips"] = receipt.get("estimatorEffectiveMips")
            row["estimatorEffectiveBandwidthMbps"] = receipt.get("estimatorEffectiveBandwidthMbps")
            row["estimatorTxPowerW"] = receipt.get("estimatorTxPowerW")
            row["native_binding_applied"] = receipt.get("native_binding_applied", receipt.get("nativeBindingApplied"))
            row["native_cpu_mips_bound"] = receipt.get("native_cpu_mips_bound", receipt.get("nativeCpuMipsBound"))
            row["native_network_bandwidth_bound"] = receipt.get("native_network_bandwidth_bound", receipt.get("nativeNetworkBandwidthBound"))
            row["native_tx_power_bound"] = receipt.get("native_tx_power_bound", receipt.get("nativeTxPowerBound"))
            row["native_base_mips"] = receipt.get("native_base_mips", receipt.get("nativeBaseMips"))
            row["native_applied_mips"] = receipt.get("native_applied_mips", receipt.get("nativeAppliedMips"))
            row["native_cpu_share"] = receipt.get("native_cpu_share", receipt.get("nativeCpuShare"))
            row["native_bandwidth_share"] = receipt.get("native_bandwidth_share", receipt.get("nativeBandwidthShare"))
            row["native_tx_power_ratio"] = receipt.get("native_tx_power_ratio", receipt.get("nativeTxPowerRatio"))
            row["native_cpu_binding_scope"] = receipt.get("native_cpu_binding_scope", receipt.get("nativeCpuBindingScope"))
            row["native_network_binding_scope"] = receipt.get("native_network_binding_scope", receipt.get("nativeNetworkBindingScope"))
            row["native_tx_power_binding_scope"] = receipt.get("native_tx_power_binding_scope", receipt.get("nativeTxPowerBindingScope"))
            row["executed_logical_tier"] = str(receipt.get("executedLogicalTier") or "").upper()
            row["executed_vm_id"] = receipt.get("executedVmId")
            row["executed_abstract_action"] = executed_action
            row["executed_abstract_action_name"] = _action_name(executed_action)
            row["fallback_reason"] = fallback_reason
            row["success"] = receipt.get("taskSucceeded", receipt.get("success"))
            row["failureReason"] = receipt.get("failureReason", receipt.get("fallbackReason"))
            row["deadlineMiss"] = receipt.get("deadlineMiss")
            row["queueOverflow"] = receipt.get("queueOverflow")
            row["vmUnavailable"] = receipt.get("vmUnavailable")
            row["linkUnavailable"] = receipt.get("linkUnavailable")
            row["taskDropped"] = receipt.get("taskDropped")
            row["latencyExceeded"] = receipt.get("latencyExceeded")
            row["resourceExceeded"] = receipt.get("resourceExceeded")
            row["estimatedTotalDelaySec"] = receipt.get("estimatedTotalDelaySec", row.get("estimatedTotalDelaySec"))
            row["estimatedQueueLength"] = receipt.get("estimatedQueueLength", row.get("estimatedQueueLength"))
            row["estimatedTransmissionRateMbps"] = receipt.get("estimatedTransmissionRateMbps", row.get("estimatedTransmissionRateMbps"))
            row["estimatedComputeCapacity"] = receipt.get("estimatedComputeCapacity", row.get("estimatedComputeCapacity"))
            row["estimatedLinkLifetimeSec"] = receipt.get("estimatedLinkLifetimeSec", row.get("estimatedLinkLifetimeSec"))
            row["estimatedTaskTransmissionTimeSec"] = receipt.get("estimatedTaskTransmissionTimeSec", row.get("estimatedTaskTransmissionTimeSec"))
            row["estimatedTaskComputeTimeSec"] = receipt.get("estimatedTaskComputeTimeSec", row.get("estimatedTaskComputeTimeSec"))
            row["estimatedTaskCompletionTimeSec"] = receipt.get("estimatedTaskCompletionTimeSec", row.get("estimatedTaskCompletionTimeSec"))
            row["linkSurvivalMarginSec"] = receipt.get("linkSurvivalMarginSec", row.get("linkSurvivalMarginSec"))
            row["linkSurvivalMarginToCompletionSec"] = receipt.get("linkSurvivalMarginToCompletionSec", row.get("linkSurvivalMarginToCompletionSec"))
            row["linkAvailableNow"] = receipt.get("linkAvailableNow", row.get("linkAvailableNow"))
            row["handoverRequired"] = receipt.get("handoverRequired", row.get("handoverRequired"))
            row["handoverAvailable"] = receipt.get("handoverAvailable", row.get("handoverAvailable"))
            row["mobilityRisk"] = receipt.get("mobilityRisk", row.get("mobilityRisk"))
            row["mobilityRiskSource"] = receipt.get("mobilityRiskSource", row.get("mobilityRiskSource"))
            row["actionAccepted"] = receipt.get("actionAccepted", receipt.get("accepted"))
            row["executionScheduled"] = receipt.get("executionScheduled", receipt.get("accepted"))
            row["taskCompleted"] = receipt.get("taskCompleted")
            row["taskSucceeded"] = receipt.get("taskSucceeded", receipt.get("success"))
            row["intent_execution_match"] = int(bool(receipt.get("intentExecutionMatch", False)))
            decision_rows.append(row)
            decision_step += 1
            try:
                state = client.get_state()
            except SatEdgeSimClientError:
                state = {"status": "RUNNING"}

        if state.get("status") not in TERMINAL_STATUSES:
            state = _wait_for_state(
                client,
                args.poll_sleep_sec,
                max_polls=max(20, int(args.wait_timeout_ms / max(args.poll_sleep_sec, 1.0e-3) / 1000)),
            )
        final_state = dict(state)
        final_metrics = client.get_metrics()
    except Exception as exc:  # noqa: BLE001
        failure_message = str(exc)
        final_state = dict(state) if isinstance(state, dict) else {}
        try:
            final_metrics = client.get_metrics()
        except Exception as metrics_exc:  # noqa: BLE001
            final_metrics = {"status": "GET_METRICS_FAILED", "message": str(metrics_exc)}
    finally:
        pass

    elapsed_sec = time.time() - started_at
    summary: Dict[str, Any] = {
        **validation_metadata(),
        "status": "SATEDGESIM_REPLAY_OK" if failure_message is None else "SATEDGESIM_REPLAY_FAILED",
        "base_url": args.base_url,
        "checkpoint": args.checkpoint,
        "device": str(policy.device) if policy is not None else args.device,
        "devices_count": args.devices_count,
        "max_decisions": args.max_decisions,
        "actual_decisions": len(decision_rows),
        "seed": args.seed,
        "algorithm_index": args.algorithm_index,
        "architecture_index": args.architecture_index,
        "force_policy": args.force_policy,
        "force_upper_action": args.force_upper_action,
        "eval_mode": args.eval_mode,
        "tie_break_eps": float(args.tie_eps),
        "stochastic_seed": int(args.stochastic_seed),
        "scenario_profile": args.scenario_profile,
        "task_source_mode": args.task_source_mode,
        "profile_name": args.profile_name,
        "architecture": args.architecture,
        "success_profile": args.success_profile,
        "action_mask_mode": args.action_mask_mode,
        "min_link_survival_margin_sec": max(0.0, float(args.min_link_survival_margin_sec)),
        "elapsed_sec": elapsed_sec,
        "health": health,
        "failure_message": failure_message,
        "final_state_status": final_state.get("status") if isinstance(final_state, dict) else None,
        "final_state_message": final_state.get("message") if isinstance(final_state, dict) else None,
        "close_response": close_response,
        "final_metrics": final_metrics,
        "success_ratio_semantics": "suppressed_without_completion_evidence",
        "completion_success_available": False,
        "deprecated_success_rate_alias": False,
        "receipt_accept_ratio": 0.0,
        "scheduling_success_ratio": 0.0,
        "scheduling_acceptance_rate": 0.0,
        "receipt_accept_ratio_semantics": "RL API accepted action receipt",
        "scheduling_success_ratio_semantics": "SatEdgeSim accepted candidate scheduling",
        "scheduling_acceptance_rate_semantics": "candidate scheduling acceptance when completion evidence is unavailable",
        "intent_execution_match_ratio_semantics": "abstract policy action mapped to intended executed tier",
        "no_fallback_ratio": 0.0,
        "final_cumulative_energy": final_metrics.get("energyConsumption") if isinstance(final_metrics, dict) else None,
        "receipt_energy_delta": None,
        "energy_source": "simlog_final" if isinstance(final_metrics, dict) and final_metrics.get("energyConsumption") is not None else "unavailable",
        "energy_unit": "unknown",
        "obs_normalization_mode": policy.obs_normalization_mode if policy is not None else "legacy",
        "obs_normalization_path": policy.obs_normalization_path if policy is not None else "",
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded) if policy is not None else False,
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim) if policy is not None else None,
        "loaded_required_modules": list(policy.loaded_required_modules) if policy is not None else [],
        "loaded_optional_modules": list(policy.loaded_optional_modules) if policy is not None else [],
        "skipped_optional_modules": list(policy.skipped_optional_modules) if policy is not None else [],
        "missing_required_modules": list(policy.missing_required_modules) if policy is not None else [],
    }

    if decision_rows:
        total = len(decision_rows)
        policy_counts = [0, 0, 0, 0]
        raw_argmax_counts = [0, 0, 0, 0]
        final_policy_counts = [0, 0, 0, 0]
        executed_counts = [0, 0, 0, 0]
        visible_counts = [0, 0, 0, 0]
        mobility_safe_counts = [0, 0, 0, 0]
        completion_safe_counts = [0, 0, 0, 0]
        selected_when_visible = [0, 0, 0, 0]
        tie_break_applied_count = 0
        cost_rank_values: List[float] = []
        fallback_distribution: Dict[str, int] = {}
        intent_execution_matches = 0
        http_timeout_count = 0
        http_connection_error_count = 0
        receipt_accept_count = 0
        scheduling_success_count = 0
        server_processing_values: List[float] = []
        client_elapsed_values: List[float] = []
        mobility_risk_values: List[float] = []
        remote_available_count = 0
        for row in decision_rows:
            policy_action = int(_to_float(row.get("policy_upper_action"), -1))
            raw_argmax_action = int(_to_float(row.get("raw_argmax_action"), -1))
            final_policy_action = int(_to_float(row.get("final_policy_action"), policy_action))
            executed_action = int(_to_float(row.get("executed_abstract_action"), -1))
            if 0 <= policy_action <= 3:
                policy_counts[policy_action] += 1
            if 0 <= raw_argmax_action <= 3:
                raw_argmax_counts[raw_argmax_action] += 1
            if 0 <= final_policy_action <= 3:
                final_policy_counts[final_policy_action] += 1
            if 0 <= executed_action <= 3:
                executed_counts[executed_action] += 1
            mask = json.loads(str(row.get("abstract_action_mask") or "[1,0,0,0]"))
            visible_mask = json.loads(str(row.get("abstract_action_mask_visible") or row.get("abstract_action_mask") or "[1,0,0,0]"))
            mobility_safe_mask = json.loads(str(row.get("abstract_action_mask_mobility_safe") or "[0,0,0,0]"))
            completion_safe_mask = json.loads(str(row.get("abstract_action_mask_completion_safe") or "[0,0,0,0]"))
            for idx in range(min(4, len(visible_mask))):
                if bool(visible_mask[idx]):
                    visible_counts[idx] += 1
            for idx in range(min(4, len(mobility_safe_mask))):
                if bool(mobility_safe_mask[idx]):
                    mobility_safe_counts[idx] += 1
            for idx in range(min(4, len(completion_safe_mask))):
                if bool(completion_safe_mask[idx]):
                    completion_safe_counts[idx] += 1
            if any(bool(x) for x in mask[1:4]):
                remote_available_count += 1
            if 0 <= executed_action <= 3 and executed_action < len(mask) and bool(mask[executed_action]):
                selected_when_visible[executed_action] += 1
            fallback_reason = str(row.get("fallback_reason") or "none")
            fallback_distribution[fallback_reason] = fallback_distribution.get(fallback_reason, 0) + 1
            intent_execution_matches += int(row.get("intent_execution_match", 0) or 0)
            receipt_accept_count += int(row.get("receipt_accepted", 0) or 0)
            scheduling_success_count += int(bool(row.get("executionScheduled", row.get("actionAccepted", row.get("receipt_accepted")))))
            tie_break_applied_count += int(_to_float(row.get("tie_break_applied"), 0.0))
            if row.get("selected_by_cost_rank") not in (None, ""):
                cost_rank_values.append(_to_float(row.get("selected_by_cost_rank")))
            if fallback_reason == "http_timeout":
                http_timeout_count += 1
            if fallback_reason == "http_connection_error":
                http_connection_error_count += 1
            if row.get("server_processing_ms") not in (None, ""):
                server_processing_values.append(_to_float(row.get("server_processing_ms")))
            if row.get("client_elapsed_ms") not in (None, ""):
                client_elapsed_values.append(_to_float(row.get("client_elapsed_ms")))
            if row.get("mobilityRisk") not in (None, ""):
                mobility_risk_values.append(_to_float(row.get("mobilityRisk")))
        summary.update(
            {
                "num_decisions": total,
                "mobility_link_failure_ratio": final_metrics.get("mobilityFailureRate"),
                "latency_deadline_failure_ratio": final_metrics.get("delayFailureRate"),
                "scheduling_success_ratio": scheduling_success_count / max(1, total),
                "scheduling_acceptance_rate": scheduling_success_count / max(1, total),
                "receipt_accept_ratio": receipt_accept_count / max(1, total),
                "receipt_accept_ratio_semantics": "RL API accepted action receipt",
                "scheduling_success_ratio_semantics": "SatEdgeSim accepted candidate scheduling",
                "scheduling_acceptance_rate_semantics": "candidate scheduling acceptance when completion evidence is unavailable",
                "mean_delay": final_metrics.get("averageEteDelay"),
                "mean_energy_per_decision": sum(_to_float(row.get("energy")) for row in decision_rows) / max(1, total),
                "mean_energy_raw_delta": sum(_to_float(row.get("energy_raw_delta")) for row in decision_rows) / max(1, total),
                "energy_audit_status": "requires_manual_audit",
                "policy_local_ratio": policy_counts[0] / max(1, total),
                "policy_neighbor_ratio": policy_counts[1] / max(1, total),
                "policy_geo_ratio": policy_counts[2] / max(1, total),
                "policy_ground_ratio": policy_counts[3] / max(1, total),
                "raw_argmax_local_ratio": raw_argmax_counts[0] / max(1, total),
                "raw_argmax_neighbor_ratio": raw_argmax_counts[1] / max(1, total),
                "raw_argmax_geo_ratio": raw_argmax_counts[2] / max(1, total),
                "raw_argmax_ground_ratio": raw_argmax_counts[3] / max(1, total),
                "final_policy_local_ratio": final_policy_counts[0] / max(1, total),
                "final_policy_neighbor_ratio": final_policy_counts[1] / max(1, total),
                "final_policy_geo_ratio": final_policy_counts[2] / max(1, total),
                "final_policy_ground_ratio": final_policy_counts[3] / max(1, total),
                "executed_local_ratio": executed_counts[0] / max(1, total),
                "executed_neighbor_ratio": executed_counts[1] / max(1, total),
                "executed_geo_ratio": executed_counts[2] / max(1, total),
                "executed_ground_ratio": executed_counts[3] / max(1, total),
                "tie_break_applied_ratio": tie_break_applied_count / max(1, total),
                "cost_rank_selected_mean": sum(cost_rank_values) / max(1, len(cost_rank_values)),
                "intent_execution_match_ratio": intent_execution_matches / max(1, total),
                "intent_execution_match_ratio_semantics": "abstract policy action mapped to intended executed tier",
                "no_fallback_ratio": fallback_distribution.get("none", 0) / max(1, total),
                "receipt_success_true_ratio": sum(1 for r in decision_rows if bool(r.get("success"))) / max(1, total),
                "http_timeout_count": http_timeout_count,
                "http_connection_error_count": http_connection_error_count,
                "mean_server_processing_ms": sum(server_processing_values) / max(1, len(server_processing_values)),
                "max_server_processing_ms": max(server_processing_values or [0.0]),
                "mean_client_elapsed_ms": sum(client_elapsed_values) / max(1, len(client_elapsed_values)),
                "max_client_elapsed_ms": max(client_elapsed_values or [0.0]),
                "fallback_reason_distribution": {k: v / max(1, total) for k, v in sorted(fallback_distribution.items())},
                "local_visible_ratio": visible_counts[0] / max(1, total),
                "neighbor_visible_ratio": visible_counts[1] / max(1, total),
                "geo_visible_ratio": visible_counts[2] / max(1, total),
                "ground_visible_ratio": visible_counts[3] / max(1, total),
                "local_mobility_safe_ratio": mobility_safe_counts[0] / max(1, total),
                "neighbor_mobility_safe_ratio": mobility_safe_counts[1] / max(1, total),
                "geo_mobility_safe_ratio": mobility_safe_counts[2] / max(1, total),
                "ground_mobility_safe_ratio": mobility_safe_counts[3] / max(1, total),
                "local_completion_safe_ratio": completion_safe_counts[0] / max(1, total),
                "neighbor_completion_safe_ratio": completion_safe_counts[1] / max(1, total),
                "geo_completion_safe_ratio": completion_safe_counts[2] / max(1, total),
                "ground_completion_safe_ratio": completion_safe_counts[3] / max(1, total),
                "remote_available_ratio": remote_available_count / max(1, total),
                "mean_mobility_risk_selected": sum(mobility_risk_values) / max(1, len(mobility_risk_values)),
                "local_selected_when_visible_ratio": selected_when_visible[0] / max(1, visible_counts[0]),
                "neighbor_selected_when_visible_ratio": selected_when_visible[1] / max(1, visible_counts[1]),
                "geo_selected_when_visible_ratio": selected_when_visible[2] / max(1, visible_counts[2]),
                "ground_selected_when_visible_ratio": selected_when_visible[3] / max(1, visible_counts[3]),
            }
        )
        energy_info = energy_semantics(decision_rows, summary, final_metrics)
        binding_info = resource_binding_semantics(decision_rows, summary, final_metrics)
        summary.update(
            {
                **binding_info,
                "raw_energy_counter_final": energy_info["final_cumulative_energy"],
                "final_cumulative_energy": energy_info["final_cumulative_energy"],
                "receipt_energy_delta": energy_info["receipt_energy_delta"],
                "energy_source": energy_info["energy_source"],
                "energy_unit": energy_info["energy_unit"],
                "energy_semantics": energy_info["energy_semantics"],
            }
        )
        completion_available = has_completion_evidence(decision_rows, summary, final_metrics)
        summary["completion_success_available"] = completion_available
        summary["completion_receipt_available"] = completion_available
        summary["deprecated_success_rate_alias"] = bool(completion_available)
        if completion_available:
            completion_ratio = completion_success_ratio(decision_rows, summary, final_metrics)
            summary["completion_success_ratio"] = completion_ratio
            summary["task_completion_success_ratio"] = completion_ratio
            summary["success_ratio"] = completion_ratio
            summary["success_rate"] = completion_ratio
            summary["success_ratio_semantics"] = "deprecated alias for completion_success_ratio"
    else:
        energy_info = energy_semantics([], summary, final_metrics)
        binding_info = resource_binding_semantics([], summary, final_metrics)
        summary.update(
            {
                **binding_info,
                "raw_energy_counter_final": energy_info["final_cumulative_energy"],
                "final_cumulative_energy": energy_info["final_cumulative_energy"],
                "receipt_energy_delta": energy_info["receipt_energy_delta"],
                "energy_source": energy_info["energy_source"],
                "energy_unit": energy_info["energy_unit"],
                "energy_semantics": energy_info["energy_semantics"],
            }
        )
        completion_available = has_completion_evidence([], summary, final_metrics)
        summary["completion_success_available"] = completion_available
        summary["completion_receipt_available"] = completion_available
        summary["deprecated_success_rate_alias"] = bool(completion_available)
        if completion_available:
            completion_ratio = completion_success_ratio([], summary, final_metrics)
            summary["completion_success_ratio"] = completion_ratio
            summary["task_completion_success_ratio"] = completion_ratio
            summary["success_ratio"] = completion_ratio
            summary["success_rate"] = completion_ratio
            summary["success_ratio_semantics"] = "deprecated alias for completion_success_ratio"

    _write_csv(output_dir / "decision_log.csv", decision_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "final_metrics.json", final_metrics)

    if failure_message is not None:
        print(f"SATEDGESIM_REPLAY_FAILED decisions={len(decision_rows)} error={failure_message}")
        raise SystemExit(1)

    print(f"SATEDGESIM_REPLAY_OK decisions={len(decision_rows)} device={str(policy.device) if policy is not None else args.device}")
    print(f"decision_log={output_dir / 'decision_log.csv'}")
    print(f"summary={output_dir / 'summary.json'}")
    print(f"final_metrics={output_dir / 'final_metrics.json'}")


if __name__ == "__main__":
    main()
