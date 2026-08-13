from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from satedgesim_semantics import (
    completion_observed_ratio,
    completion_success_ratio,
    energy_semantics,
    has_completion_evidence,
    require_native_scheduler_bound_for_formal_claim,
    resource_binding_semantics,
    validation_metadata,
)


ACTION_NAMES = ["LOCAL", "NEIGHBOR", "GEO", "GROUND"]
REQUIRED_FAILURE_FIELDS = [
    "failureReason",
    "deadlineMiss",
    "queueOverflow",
    "vmUnavailable",
    "linkUnavailable",
    "taskDropped",
    "latencyExceeded",
    "resourceExceeded",
    "delay",
    "deadline",
    "queueLength",
    "selectedVmId",
    "selectedVmLogicalTier",
    "executedLogicalTier",
    "policyUpperActionName",
    "scenario_phase",
    "task_type",
    "estimatedTotalDelaySec",
    "estimatedQueueLength",
    "estimatedTransmissionRateMbps",
    "estimatedComputeCapacity",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mask(row: Dict[str, str]) -> List[int]:
    raw = row.get("abstract_action_mask", "[1,0,0,0]")
    try:
        decoded = json.loads(raw)
    except Exception:
        decoded = [1, 0, 0, 0]
    if not isinstance(decoded, list) or len(decoded) < 4:
        decoded = [1, 0, 0, 0]
    return [1 if bool(decoded[i]) else 0 for i in range(4)]


def _mask_from_field(row: Dict[str, str], field: str, fallback: str = "[0,0,0,0]") -> List[int]:
    raw = row.get(field, fallback)
    try:
        decoded = json.loads(str(raw))
    except Exception:
        decoded = [0, 0, 0, 0]
    if not isinstance(decoded, list) or len(decoded) < 4:
        decoded = [0, 0, 0, 0]
    return [1 if bool(decoded[i]) else 0 for i in range(4)]


def _norm_reason(row: Dict[str, str]) -> str:
    reason = str(row.get("failureReason") or "").strip()
    if reason:
        return reason
    fb = str(row.get("fallback_reason") or "none").strip()
    if fb and fb != "none":
        return fb
    return "unknown_failure"


def _task_failure_distribution_from_metrics(final_metrics: Dict[str, Any]) -> Dict[str, float]:
    tasks_sent = int(_to_float(final_metrics.get("tasksSent"), 0.0))
    if tasks_sent <= 0:
        return {}
    latency = int(_to_float(final_metrics.get("tasksFailedLatency"), 0.0))
    mobility = int(_to_float(final_metrics.get("tasksFailedMobility"), 0.0))
    resources = int(_to_float(final_metrics.get("tasksFailedResourcesUnavailable"), 0.0))
    dead = int(_to_float(final_metrics.get("tasksFailedBecauseDeviceDead"), 0.0))
    accounted = latency + mobility + resources + dead
    failed_total = int(_to_float(final_metrics.get("tasksFailed"), accounted))
    unknown = max(0, failed_total - accounted)
    dist = {
        "latency_deadline": float(latency / tasks_sent),
        "mobility_link": float(mobility / tasks_sent),
        "resource_unavailable": float(resources / tasks_sent),
        "device_dead_or_dropped": float(dead / tasks_sent),
    }
    if unknown > 0:
        dist["unknown_failure"] = float(unknown / tasks_sent)
    return dist


def _ratio(counter: Counter, key: str, total: int) -> float:
    return float(counter.get(key, 0) / max(1, total))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _dist(counter: Counter, denom: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in sorted(counter.items()):
        out[str(key)] = float(value / max(1, denom))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SatEdgeSim replay logs.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--require-native-scheduler-bound", action="store_true")
    parser.add_argument("--energy-source", default="auto", choices=["auto", "receipt_delta_wh", "simlog_final_wh", "estimator_expected_j"])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--allow-diagnostic-energy-missing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = _read_csv(input_dir / "decision_log.csv")
    summary_json = json.loads((input_dir / "summary.json").read_text(encoding="utf-8")) if (input_dir / "summary.json").exists() else {}
    final_metrics = json.loads((input_dir / "final_metrics.json").read_text(encoding="utf-8")) if (input_dir / "final_metrics.json").exists() else {}

    fallback_counter = Counter(str(row.get("fallback_reason", "none") or "none") for row in rows)
    total = len(rows)
    policy_counts = [0, 0, 0, 0]
    raw_argmax_counts = [0, 0, 0, 0]
    final_policy_counts = [0, 0, 0, 0]
    executed_counts = [0, 0, 0, 0]
    visible_counts = [0, 0, 0, 0]
    mobility_safe_counts = [0, 0, 0, 0]
    completion_safe_counts = [0, 0, 0, 0]
    selected_when_visible = [0, 0, 0, 0]
    intent_execution_matches = 0
    remote_visible_rows = 0
    mismatch_examples: List[Dict[str, Any]] = []
    http_timeout_count = 0
    http_connection_error_count = 0
    receipt_accept_count = 0
    scheduling_success_count = 0
    server_processing_values: List[float] = []
    client_elapsed_values: List[float] = []
    tie_break_applied_count = 0
    selected_cost_rank_values: List[float] = []
    selected_mobility_risk_values: List[float] = []

    missing_counts = {name: 0 for name in REQUIRED_FAILURE_FIELDS}
    success_flags: List[bool] = []
    pending_completion_count = 0
    delay_success: List[float] = []
    delay_failure: List[float] = []
    queue_success: List[float] = []
    queue_failure: List[float] = []
    failure_reason_counter = Counter()
    failure_by_action = Counter()
    failure_by_tier = Counter()
    failure_by_phase = Counter()
    failure_by_task_type = Counter()
    success_by_action = Counter()
    success_by_tier = Counter()
    success_by_phase = Counter()

    deadline_miss_count = 0
    queue_overflow_count = 0
    vm_unavailable_count = 0
    link_unavailable_count = 0
    task_dropped_count = 0
    latency_exceeded_count = 0
    resource_exceeded_count = 0
    unknown_failure_count = 0

    for row in rows:
        for field in REQUIRED_FAILURE_FIELDS:
            if row.get(field) in (None, ""):
                missing_counts[field] += 1

        policy_action = int(_to_float(row.get("policy_upper_action"), -1))
        raw_argmax_action = int(_to_float(row.get("raw_argmax_action"), -1))
        final_policy_action = int(_to_float(row.get("final_policy_action"), policy_action))
        executed_action = int(_to_float(row.get("executed_abstract_action"), -1))
        mask = _mask(row)
        visible_mask = _mask_from_field(row, "abstract_action_mask_visible", row.get("abstract_action_mask", "[0,0,0,0]"))
        mobility_safe_mask = _mask_from_field(row, "abstract_action_mask_mobility_safe")
        completion_safe_mask = _mask_from_field(row, "abstract_action_mask_completion_safe")
        if 0 <= policy_action <= 3:
            policy_counts[policy_action] += 1
        if 0 <= raw_argmax_action <= 3:
            raw_argmax_counts[raw_argmax_action] += 1
        if 0 <= final_policy_action <= 3:
            final_policy_counts[final_policy_action] += 1
        if 0 <= executed_action <= 3:
            executed_counts[executed_action] += 1
        for idx in range(4):
            if visible_mask[idx]:
                visible_counts[idx] += 1
            if mobility_safe_mask[idx]:
                mobility_safe_counts[idx] += 1
            if completion_safe_mask[idx]:
                completion_safe_counts[idx] += 1
        if any(mask[idx] for idx in (1, 2, 3)):
            remote_visible_rows += 1
        if 0 <= executed_action <= 3 and executed_action < len(mask) and mask[executed_action]:
            selected_when_visible[executed_action] += 1
        intent_execution_matches += int(_to_float(row.get("intent_execution_match"), 0.0))
        receipt_accept_count += int(_to_float(row.get("receipt_accepted"), 0.0))
        scheduling_success_count += int(_to_bool(row.get("executionScheduled", row.get("actionAccepted", row.get("receipt_accepted"))), False))
        tie_break_applied_count += int(_to_float(row.get("tie_break_applied"), 0.0))
        if row.get("selected_by_cost_rank") not in (None, ""):
            selected_cost_rank_values.append(_to_float(row.get("selected_by_cost_rank")))
        if str(row.get("fallback_reason") or "") == "http_timeout":
            http_timeout_count += 1
        if str(row.get("fallback_reason") or "") == "http_connection_error":
            http_connection_error_count += 1
        if row.get("server_processing_ms") not in (None, ""):
            server_processing_values.append(_to_float(row.get("server_processing_ms")))
        if row.get("client_elapsed_ms") not in (None, ""):
            client_elapsed_values.append(_to_float(row.get("client_elapsed_ms")))
        if row.get("mobilityRisk") not in (None, ""):
            selected_mobility_risk_values.append(_to_float(row.get("mobilityRisk")))
        if (
            len(mismatch_examples) < 20
            and (
                int(_to_float(row.get("intent_execution_match"), 0.0)) == 0
                or str(row.get("fallback_reason") or "none") != "none"
            )
        ):
            mismatch_examples.append(
                {
                    "step": int(_to_float(row.get("step"), -1)),
                    "state_decision_id": row.get("state_decision_id", row.get("decision_id")),
                    "receipt_decision_id": row.get("receipt_decision_id"),
                    "state_task_id": row.get("state_task_id", row.get("task_id")),
                    "receipt_task_id": row.get("receipt_task_id"),
                    "submitted_action": row.get("policy_upper_action_name"),
                    "receipt_executed_action": row.get("executed_abstract_action_name"),
                    "receipt_fallback_reason": row.get("fallback_reason"),
                    "selected_vm_id": row.get("selected_vm_id"),
                    "server_processing_ms": row.get("server_processing_ms"),
                    "client_elapsed_ms": row.get("client_elapsed_ms"),
                    "http_status_code": row.get("http_status_code"),
                    "http_error": row.get("http_error"),
                }
            )

        task_completed = _to_bool(row.get("taskCompleted"), False)
        succeeded = _to_bool(row.get("taskSucceeded", row.get("success")), False)
        reason = _norm_reason(row)
        pending_completion = (not task_completed) and reason == "pending_task_completion"
        if pending_completion:
            pending_completion_count += 1
        success_flags.append(succeeded)
        delay_val = row.get("delay")
        if delay_val in (None, ""):
            delay_val = row.get("estimatedTotalDelaySec")
        queue_val = row.get("queueLength")
        if queue_val in (None, ""):
            queue_val = row.get("estimatedQueueLength")
        delay = _to_float(delay_val, 0.0)
        queue = _to_float(queue_val, 0.0)
        phase = str(row.get("scenario_phase") or "unknown")
        task_type = str(row.get("task_type") or "unknown")
        action_name = str(row.get("policyUpperActionName") or row.get("policy_upper_action_name") or "UNKNOWN")
        tier_name = str(row.get("executedLogicalTier") or row.get("executed_logical_tier") or "UNKNOWN")
        if succeeded:
            delay_success.append(delay)
            queue_success.append(queue)
            success_by_action[action_name] += 1
            success_by_tier[tier_name] += 1
            success_by_phase[phase] += 1
        elif not pending_completion:
            delay_failure.append(delay)
            queue_failure.append(queue)
            failure_reason_counter[reason] += 1
            failure_by_action[action_name] += 1
            failure_by_tier[tier_name] += 1
            failure_by_phase[phase] += 1
            failure_by_task_type[task_type] += 1
            deadline_miss_count += int(_to_bool(row.get("deadlineMiss"), False))
            queue_overflow_count += int(_to_bool(row.get("queueOverflow"), False))
            vm_unavailable_count += int(_to_bool(row.get("vmUnavailable"), False))
            link_unavailable_count += int(_to_bool(row.get("linkUnavailable"), False))
            task_dropped_count += int(_to_bool(row.get("taskDropped"), False))
            latency_exceeded_count += int(_to_bool(row.get("latencyExceeded"), False))
            resource_exceeded_count += int(_to_bool(row.get("resourceExceeded"), False))
            unknown_failure_count += int(_to_bool(row.get("unknownFailure"), False))

    total_nonzero_policy = sum(1 for value in policy_counts if value > 0)
    total_nonzero_executed = sum(1 for value in executed_counts if value > 0)
    fallback_none_ratio = fallback_counter.get("none", 0) / max(1, total)
    warnings: List[str] = []
    if total > 0 and (intent_execution_matches / total) < 0.99:
        warnings.append("intent_execution_match_ratio_below_0.99")
    if total > 0 and fallback_none_ratio < 0.99:
        warnings.append("fallback_reason_none_ratio_below_0.99")
    if total > 0 and (receipt_accept_count / total) < 0.99:
        warnings.append("receipt_accept_ratio_below_0.99")
    for idx, action_name in enumerate(ACTION_NAMES):
        policy_ratio = final_policy_counts[idx] / max(1, total)
        executed_ratio = executed_counts[idx] / max(1, total)
        if abs(policy_ratio - executed_ratio) > 0.01:
            warnings.append(
                f"final_policy_vs_executed_ratio_mismatch_{action_name.lower()}={policy_ratio:.6f}_vs_{executed_ratio:.6f}"
            )
    if total_nonzero_policy <= 1 or total_nonzero_executed <= 1:
        warnings.append("single_action_type_detected_scene_coverage_insufficient_or_policy_collapsed")

    final_policy_ratios = [final_policy_counts[idx] / max(1, total) for idx in range(4)]
    if any(ratio >= 0.90 for ratio in final_policy_ratios):
        warnings.append("single_action_dominance")

    for idx, action_name in enumerate(ACTION_NAMES):
        visible_ratio = visible_counts[idx] / max(1, total)
        selected_ratio = final_policy_counts[idx] / max(1, total)
        if visible_ratio >= 0.95 and selected_ratio <= 0.0:
            warnings.append(f"visible_but_never_selected_{action_name.lower()}")

    completion_available = has_completion_evidence(rows, summary_json, final_metrics)
    task_completion_success_ratio = completion_success_ratio(rows, summary_json, final_metrics) if completion_available else None
    task_failure_reason_dist = _task_failure_distribution_from_metrics(final_metrics)
    if completion_available and task_completion_success_ratio is not None and task_completion_success_ratio < 0.90:
        warnings.append("low_success_ratio")
        if task_failure_reason_dist:
            dominant_reason = max(task_failure_reason_dist.items(), key=lambda kv: kv[1])[0]
            warnings.append(f"dominant_failure_reason_{dominant_reason}")
        elif failure_reason_counter:
            dominant_reason, _ = failure_reason_counter.most_common(1)[0]
            warnings.append(f"dominant_failure_reason_{dominant_reason}")

    eval_mode = str(summary_json.get("eval_mode", rows[0].get("eval_mode") if rows else "raw_argmax")).strip().lower()
    policy_ground_ratio = policy_counts[3] / max(1, total)
    if eval_mode == "raw_argmax" and policy_ground_ratio >= 0.90:
        warnings.append("raw_argmax_ground_dominance")
    neighbor_visible_ratio = visible_counts[1] / max(1, total)
    geo_visible_ratio = visible_counts[2] / max(1, total)
    neighbor_selected_ratio = final_policy_counts[1] / max(1, total)
    geo_selected_ratio = final_policy_counts[2] / max(1, total)
    if neighbor_visible_ratio >= 0.95 and neighbor_selected_ratio <= 0.0:
        warnings.append("neighbor_unused_when_visible")
    if geo_visible_ratio >= 0.95 and geo_selected_ratio <= 0.0:
        warnings.append("geo_unused_when_visible")
    if (remote_visible_rows / max(1, total)) < 0.20:
        warnings.append("remote_available_ratio_below_0.20")

    obs_norm_mode = str(summary_json.get("obs_normalization_mode", "legacy") or "legacy").strip().lower()
    obs_norm_loaded = bool(summary_json.get("obs_normalization_loaded", False))
    if obs_norm_mode == "trace_log_quantile" and not obs_norm_loaded:
        warnings.append("obs_normalization_missing")

    scheduling_success_ratio = _to_float(
        summary_json.get("scheduling_success_ratio", summary_json.get("scheduling_acceptance_rate", scheduling_success_count / max(1, total))),
        0.0,
    )
    receipt_accept_ratio = _to_float(summary_json.get("receipt_accept_ratio", receipt_accept_count / max(1, total)), 0.0)
    energy_info = energy_semantics(rows, summary_json, final_metrics, energy_source=args.energy_source)
    if args.formal and not bool(energy_info.get("energy_source_available", False)):
        raise SystemExit(f"formal SatEdgeSim summary requires energy source {args.energy_source}")
    outputs_are_diagnostic = bool((not energy_info.get("energy_source_available", False)) and args.allow_diagnostic_energy_missing)
    binding_info = resource_binding_semantics(rows, summary_json, final_metrics)
    if args.require_native_scheduler_bound:
        require_native_scheduler_bound_for_formal_claim(rows, summary_json, final_metrics)
    success_by_action_total = _dist(success_by_action, sum(success_by_action.values()))
    success_by_tier_total = _dist(success_by_tier, sum(success_by_tier.values()))
    success_by_phase_total = _dist(success_by_phase, sum(success_by_phase.values()))
    failure_by_action_total = _dist(failure_by_action, sum(failure_by_action.values()))
    failure_by_tier_total = _dist(failure_by_tier, sum(failure_by_tier.values()))
    failure_by_phase_total = _dist(failure_by_phase, sum(failure_by_phase.values()))
    failure_by_task_total = _dist(failure_by_task_type, sum(failure_by_task_type.values()))
    failure_reason_dist = _dist(failure_reason_counter, sum(failure_reason_counter.values()))
    effective_failure_total = max(1, total - pending_completion_count)
    tasks_sent = int(_to_float(final_metrics.get("tasksSent"), 0.0))
    tasks_failed = int(_to_float(final_metrics.get("tasksFailed"), 0.0))
    tasks_failed_latency = int(_to_float(final_metrics.get("tasksFailedLatency"), 0.0))
    tasks_failed_mobility = int(_to_float(final_metrics.get("tasksFailedMobility"), 0.0))
    tasks_failed_resource = int(_to_float(final_metrics.get("tasksFailedResourcesUnavailable"), 0.0))
    tasks_failed_dead = int(_to_float(final_metrics.get("tasksFailedBecauseDeviceDead"), 0.0))

    out: Dict[str, Any] = {
        **validation_metadata(),
        **binding_info,
        "input_dir": str(input_dir),
        "status": summary_json.get("status"),
        "eval_mode": eval_mode,
        "tie_break_eps": summary_json.get("tie_break_eps", _to_float(rows[0].get("tie_break_eps"), 0.05) if rows else 0.05),
        "num_decisions": total,
        "scheduling_success_ratio": scheduling_success_ratio,
        "scheduling_acceptance_rate": scheduling_success_ratio,
        "receipt_accept_ratio": receipt_accept_ratio,
        "receipt_accept_ratio_semantics": "RL API accepted action receipt",
        "scheduling_success_ratio_semantics": "SatEdgeSim accepted candidate scheduling",
        "scheduling_acceptance_rate_semantics": "candidate scheduling acceptance when completion evidence is unavailable",
        "completion_success_available": completion_available,
        "completion_receipt_available": completion_available,
        "completion_observed_ratio": completion_observed_ratio(rows),
        "require_native_scheduler_bound": bool(args.require_native_scheduler_bound),
        "success_ratio_semantics": (
            "deprecated alias for completion_success_ratio"
            if completion_available
            else "suppressed_without_completion_evidence"
        ),
        "deprecated_success_rate_alias": bool(completion_available),
        "mean_delay": summary_json.get("mean_delay", final_metrics.get("averageEteDelay")),
        "mean_energy_per_decision": summary_json.get(
            "mean_energy_per_decision",
            sum(_to_float(row.get("energy")) for row in rows) / max(1, total),
        ),
        "mean_energy_raw_delta": summary_json.get(
            "mean_energy_raw_delta",
            sum(_to_float(row.get("energy_raw_delta")) for row in rows) / max(1, total),
        ),
        "raw_energy_counter_final": energy_info["final_cumulative_energy"],
        "final_cumulative_energy": energy_info["final_cumulative_energy"],
        "receipt_energy_delta": energy_info["receipt_energy_delta"],
        "energy_source": energy_info["energy_source"],
        "energy_unit": energy_info["energy_unit"],
        "selected_energy_value": energy_info.get("selected_energy_value"),
        "energy_source_available": bool(energy_info.get("energy_source_available", False)),
        "energy_unavailable_reason": energy_info.get("energy_unavailable_reason", ""),
        "energy_semantics": energy_info["energy_semantics"],
        "energy_formal_claim_allowed": bool(energy_info.get("energy_source_available", False)),
        "outputs_are_diagnostic": outputs_are_diagnostic,
        "formal_claim_allowed": bool(not outputs_are_diagnostic and energy_info.get("energy_source_available", False)),
        "energy_audit_status": summary_json.get("energy_audit_status", "requires_manual_audit"),
        "energy_advantage_claim_allowed": bool(
            binding_info.get("native_scheduler_bound", False)
            and energy_info["energy_source"] not in {"unknown", "unavailable"}
        ),
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
        "tie_break_applied_ratio": tie_break_applied_count / max(1, total),
        "cost_rank_selected_mean": sum(selected_cost_rank_values) / max(1, len(selected_cost_rank_values)),
        "executed_local_ratio": executed_counts[0] / max(1, total),
        "executed_neighbor_ratio": executed_counts[1] / max(1, total),
        "executed_geo_ratio": executed_counts[2] / max(1, total),
        "executed_ground_ratio": executed_counts[3] / max(1, total),
        "intent_execution_match_ratio": summary_json.get("intent_execution_match_ratio", intent_execution_matches / max(1, total)),
        "intent_execution_match_ratio_semantics": "abstract policy action mapped to intended executed tier",
        "no_fallback_ratio": summary_json.get("no_fallback_ratio", fallback_none_ratio),
        "http_timeout_count": summary_json.get("http_timeout_count", http_timeout_count),
        "http_connection_error_count": summary_json.get("http_connection_error_count", http_connection_error_count),
        "mean_server_processing_ms": summary_json.get(
            "mean_server_processing_ms",
            sum(server_processing_values) / max(1, len(server_processing_values)),
        ),
        "max_server_processing_ms": summary_json.get("max_server_processing_ms", max(server_processing_values or [0.0])),
        "mean_client_elapsed_ms": summary_json.get(
            "mean_client_elapsed_ms",
            sum(client_elapsed_values) / max(1, len(client_elapsed_values)),
        ),
        "max_client_elapsed_ms": summary_json.get("max_client_elapsed_ms", max(client_elapsed_values or [0.0])),
        "fallback_reason_distribution": summary_json.get(
            "fallback_reason_distribution",
            {key: value / max(1, total) for key, value in sorted(fallback_counter.items())},
        ),
        "failure_reason_distribution": task_failure_reason_dist if task_failure_reason_dist else failure_reason_dist,
        "task_failure_reason_distribution": task_failure_reason_dist,
        "receipt_failure_reason_distribution": failure_reason_dist,
        "failure_by_action": failure_by_action_total,
        "failure_by_executed_tier": failure_by_tier_total,
        "failure_by_scenario_phase": failure_by_phase_total,
        "failure_by_task_type": failure_by_task_total,
        "success_by_action": success_by_action_total,
        "success_by_executed_tier": success_by_tier_total,
        "success_by_phase": success_by_phase_total,
        "pending_task_completion_ratio": float(pending_completion_count / max(1, total)),
        "decision_level_task_outcome_join_available": bool(sum(success_by_action.values()) > 0 or sum(failure_by_action.values()) > 0),
        "mean_delay_success": _safe_mean(delay_success),
        "mean_delay_failure": _safe_mean(delay_failure),
        "mean_queue_success": _safe_mean(queue_success),
        "mean_queue_failure": _safe_mean(queue_failure),
        "deadline_miss_ratio": float(tasks_failed_latency / max(1, tasks_sent)) if tasks_sent > 0 else float(deadline_miss_count / effective_failure_total),
        "queue_overflow_ratio": float(queue_overflow_count / effective_failure_total),
        "vm_unavailable_ratio": float(tasks_failed_resource / max(1, tasks_sent)) if tasks_sent > 0 else float(vm_unavailable_count / effective_failure_total),
        "link_unavailable_ratio": float(tasks_failed_mobility / max(1, tasks_sent)) if tasks_sent > 0 else float(link_unavailable_count / effective_failure_total),
        "task_dropped_ratio": float(tasks_failed_dead / max(1, tasks_sent)) if tasks_sent > 0 else float(task_dropped_count / effective_failure_total),
        "latency_exceeded_ratio": float(tasks_failed_latency / max(1, tasks_sent)) if tasks_sent > 0 else float(latency_exceeded_count / effective_failure_total),
        "resource_exceeded_ratio": float(tasks_failed_resource / max(1, tasks_sent)) if tasks_sent > 0 else float(resource_exceeded_count / effective_failure_total),
        "unknown_failure_ratio": float(max(0, tasks_failed - (tasks_failed_latency + tasks_failed_mobility + tasks_failed_resource + tasks_failed_dead)) / max(1, tasks_sent)) if tasks_sent > 0 else float(unknown_failure_count / effective_failure_total),
        "missing_ratio_by_field": {k: float(v / max(1, total)) for k, v in sorted(missing_counts.items())},
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
        "remote_visible_ratio": remote_visible_rows / max(1, total),
        "remote_available_ratio": remote_visible_rows / max(1, total),
        "mobility_link_failure_ratio": _to_float(final_metrics.get("mobilityFailureRate"), 0.0),
        "latency_deadline_failure_ratio": _to_float(final_metrics.get("delayFailureRate"), 0.0),
        "mean_mobility_risk_selected": _safe_mean(selected_mobility_risk_values),
        "local_selected_when_visible_ratio": selected_when_visible[0] / max(1, visible_counts[0]),
        "neighbor_selected_when_visible_ratio": selected_when_visible[1] / max(1, visible_counts[1]),
        "geo_selected_when_visible_ratio": selected_when_visible[2] / max(1, visible_counts[2]),
        "ground_selected_when_visible_ratio": selected_when_visible[3] / max(1, visible_counts[3]),
        "policy_action_distribution": {
            "local": policy_counts[0] / max(1, total),
            "neighbor": policy_counts[1] / max(1, total),
            "geo": policy_counts[2] / max(1, total),
            "ground": policy_counts[3] / max(1, total),
        },
        "raw_argmax_action_distribution": {
            "local": raw_argmax_counts[0] / max(1, total),
            "neighbor": raw_argmax_counts[1] / max(1, total),
            "geo": raw_argmax_counts[2] / max(1, total),
            "ground": raw_argmax_counts[3] / max(1, total),
        },
        "final_policy_action_distribution": {
            "local": final_policy_counts[0] / max(1, total),
            "neighbor": final_policy_counts[1] / max(1, total),
            "geo": final_policy_counts[2] / max(1, total),
            "ground": final_policy_counts[3] / max(1, total),
        },
        "executed_tier_distribution": {
            "local": executed_counts[0] / max(1, total),
            "neighbor": executed_counts[1] / max(1, total),
            "geo": executed_counts[2] / max(1, total),
            "ground": executed_counts[3] / max(1, total),
        },
        "visible_opportunity_distribution": {
            "local": visible_counts[0] / max(1, total),
            "neighbor": visible_counts[1] / max(1, total),
            "geo": visible_counts[2] / max(1, total),
            "ground": visible_counts[3] / max(1, total),
        },
        "warnings": warnings,
        "readiness": "warning" if warnings else "ready",
        "obs_normalization_mode": summary_json.get("obs_normalization_mode", "legacy"),
        "obs_normalization_path": summary_json.get("obs_normalization_path", ""),
        "obs_normalization_loaded": bool(summary_json.get("obs_normalization_loaded", False)),
        "obs_feature_dim": summary_json.get("obs_feature_dim"),
        "success_profile": summary_json.get("success_profile", "default"),
        "profile_name": summary_json.get("profile_name", ""),
        "architecture": summary_json.get("architecture", rows[0].get("architecture", "full") if rows else "full"),
        "action_mask_mode": summary_json.get("action_mask_mode", summary_json.get("actionMaskMode", "visible_only")),
        "min_link_survival_margin_sec": _to_float(summary_json.get("min_link_survival_margin_sec", summary_json.get("minLinkSurvivalMarginSec")), 0.0),
        "loaded_required_modules": summary_json.get("loaded_required_modules", []),
        "loaded_optional_modules": summary_json.get("loaded_optional_modules", []),
        "skipped_optional_modules": summary_json.get("skipped_optional_modules", []),
        "missing_required_modules": summary_json.get("missing_required_modules", []),
        "optional_module_mismatch_inference_impact": "none" if summary_json.get("skipped_optional_modules") else "",
        "mismatch_examples": mismatch_examples,
        "acceptance": {
            "intent_execution_match_ratio_ge_0_99": (intent_execution_matches / max(1, total)) >= 0.99,
            "fallback_reason_none_ratio_ge_0_99": fallback_none_ratio >= 0.99,
            "receipt_accept_ratio_ge_0_99": receipt_accept_ratio >= 0.99,
            "policy_vs_executed_ratio_diff_le_0_01": all(
                abs((final_policy_counts[idx] / max(1, total)) - (executed_counts[idx] / max(1, total))) <= 0.01
                for idx in range(4)
            ),
            "http_timeout_eq_0": http_timeout_count == 0,
            "http_connection_error_eq_0": http_connection_error_count == 0,
            "warnings_empty": len(warnings) == 0,
        },
        "mean_inference_ms": (
            sum(_to_float(row.get("inference_ms")) for row in rows) / len(rows)
            if rows
            else None
        ),
    }
    if completion_available and task_completion_success_ratio is not None:
        out["completion_success_ratio"] = task_completion_success_ratio
        out["task_completion_success_ratio"] = task_completion_success_ratio
        out["success_ratio"] = task_completion_success_ratio
        out["success_rate"] = task_completion_success_ratio
        warnings.append("deprecated_success_rate_alias")
    else:
        warnings.append("completion_receipt_unavailable_success_rate_suppressed")
    out["readiness"] = "warning" if warnings else "ready"

    output_path = Path(args.output) if args.output else input_dir / "summary_replay.json"
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatch_examples:
        (input_dir / "mismatch_examples.json").write_text(
            json.dumps(mismatch_examples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"summary_replay={output_path}")


if __name__ == "__main__":
    main()
