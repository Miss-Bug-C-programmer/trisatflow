from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import math

NA = "NA"


def _f(v: Any) -> float | None:
    try:
        if v in (None, "", "NA"):
            return None
        value = float(v)
        if not math.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _nz(v: float | None) -> Any:
    return NA if v is None else v


def _pick(summary: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in summary:
            return _f(summary.get(key))
    return None


def _ratio_failure(success: float | None) -> Any:
    if success is None:
        return NA
    return max(0.0, min(1.0, 1.0 - float(success)))


def _sum_optional(*values: float | None) -> Any:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return NA
    return float(sum(cleaned))


def unified_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    final_metrics = dict(summary.get("final_metrics") or {})
    success = _pick(summary, "task_completion_success_ratio", "success_ratio")
    local = _pick(summary, "policy_local_ratio", "final_policy_local_ratio")
    neighbor = _pick(summary, "policy_neighbor_ratio", "final_policy_neighbor_ratio")
    geo = _pick(summary, "policy_geo_ratio", "final_policy_geo_ratio")
    ground = _pick(summary, "policy_ground_ratio", "final_policy_ground_ratio")
    timeout = _f(summary.get("http_timeout_count"))
    conn_err = _f(summary.get("http_connection_error_count"))
    out: Dict[str, Any] = {
        # Performance. Missing metrics are exported as NA, never as synthetic zero.
        "normalized_system_cost": _nz(_pick(summary, "normalized_system_cost", "final_normalized_system_cost")),
        "mean_deadline_exceedance": _nz(_pick(summary, "mean_deadline_exceedance")),
        "mean_deadline_violation_ratio": _nz(_pick(summary, "mean_deadline_violation_ratio", "latency_deadline_failure_ratio", "deadline_miss_ratio")),
        "mean_delay_s": _nz(_pick(summary, "mean_delay_s")),
        "p50_delay_s": _nz(_pick(summary, "p50_delay_s")),
        "p95_delay_s": _nz(_pick(summary, "p95_delay_s")),
        "mean_energy_j": _nz(_pick(summary, "mean_energy_j", "mean_energy_per_decision")),
        "mean_queue_length_tasks": _nz(_pick(summary, "mean_queue_length_tasks")),
        "task_success_ratio": _nz(success),
        "task_failure_ratio": _ratio_failure(success),
        "mobility_link_failure_ratio": _nz(_pick(summary, "mobility_link_failure_ratio")),
        "resource_failure_ratio": _nz(_pick(summary, "resource_exceeded_ratio", "vm_unavailable_ratio")),
        # Policy
        "upper_local_ratio": _nz(local),
        "upper_neighbor_ratio": _nz(neighbor),
        "upper_geo_ratio": _nz(geo),
        "upper_ground_ratio": _nz(ground),
        "remote_ratio": _sum_optional(neighbor, geo, ground),
        "selected_when_visible_local": _nz(_pick(summary, "local_selected_when_visible_ratio")),
        "selected_when_visible_neighbor": _nz(_pick(summary, "neighbor_selected_when_visible_ratio")),
        "selected_when_visible_geo": _nz(_pick(summary, "geo_selected_when_visible_ratio")),
        "selected_when_visible_ground": _nz(_pick(summary, "ground_selected_when_visible_ratio")),
        # State-conditioned
        "raw_argmax_oracle_agreement": _nz(_pick(summary, "raw_argmax_oracle_agreement")),
        "mi_phase_argmax": _nz(_pick(summary, "mi_phase_argmax")),
        "normalized_regret": _nz(_pick(summary, "normalized_regret")),
        "near_optimal_hit_rate_05": _nz(_pick(summary, "near_optimal_hit_rate_05")),
        "near_optimal_hit_rate_10": _nz(_pick(summary, "near_optimal_hit_rate_10")),
        # Load-balance
        "load_balance_index": _nz(_pick(summary, "load_balance_index")),
        "jain_fairness_index": _nz(_pick(summary, "jain_fairness_index")),
        # Execution pipeline
        "intent_execution_match_ratio": _nz(_pick(summary, "intent_execution_match_ratio")),
        "receipt_accept_ratio": _nz(_pick(summary, "receipt_accept_ratio")),
        "fallback_none_ratio": _nz(_f((summary.get("fallback_reason_distribution") or {}).get("none"))),
        "policy_executed_ratio_diff": _nz(_pick(summary, "policy_executed_ratio_diff")),
        "http_timeout_count": "NA" if timeout is None else int(timeout),
        "http_connection_error_count": "NA" if conn_err is None else int(conn_err),
        # Complexity
        "mean_inference_ms": _nz(_pick(summary, "mean_inference_ms")),
        "p95_inference_ms": _nz(_pick(summary, "p95_inference_ms")),
        "training_time_sec": _nz(_pick(summary, "training_time_sec")),
        "replay_time_sec": _nz(_pick(summary, "elapsed_sec")),
        # Trace
        "trace_hit_ratio": _nz(_pick(summary, "trace_hit_ratio")),
        "trace_fallback_count": _nz(_pick(summary, "trace_fallback_count")),
        # Energy
        "energy_audit_status": summary.get("energy_audit_status", "requires_manual_audit"),
        "optional_metric_energy": _nz(_pick(summary, "mean_energy_per_decision")),
        # Metadata
        "profile_name": summary.get("profile_name", ""),
        "action_mask_mode": summary.get("action_mask_mode", ""),
        "success_profile": summary.get("success_profile", ""),
        "architecture": summary.get("architecture", "full"),
        "baseline_name": summary.get("baseline_name", ""),
    }
    if out["mean_delay_s"] == "NA":
        out["mean_delay_s"] = _nz(_f(final_metrics.get("averageEteDelay")))
    return out


def load_unified_metrics(summary_path: str | Path) -> Dict[str, Any]:
    p = Path(summary_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return unified_metrics_from_summary(payload)
