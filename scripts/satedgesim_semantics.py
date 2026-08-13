from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


SATEDGESIM_VALIDATION_MODE = "candidate_level_discrete_replay"
CLAIM_GUARD = (
    "Continuous resource fields were accepted by API but not applied to native "
    "VM/network/power scheduling; interpret as candidate-level replay."
)
ESTIMATOR_CLAIM_GUARD = (
    "Continuous resource fields affected SatEdgeSim resource-aware delay/energy "
    "estimator or admission/ranking, but did not bind native VM/network/power scheduling."
)


def validation_metadata() -> Dict[str, Any]:
    return {
        "satedgesim_validation_mode": SATEDGESIM_VALIDATION_MODE,
        "continuous_resource_binding_mode": "candidate_only",
        "continuous_resource_applied": False,
        "native_scheduler_bound": False,
        "estimator_bound": False,
        "full_hybrid_closed_loop_claim_allowed": False,
        "lower_continuous_allocator_validated_by_satedgesim": False,
        "continuous_resource_applied_to_native_scheduler": False,
        "cpu_share_effective": False,
        "bandwidth_share_effective": False,
        "tx_power_ratio_effective": False,
        "table5_title_suggestion": "SatEdgeSim candidate-level action-mapping replay",
        "claim_guard": CLAIM_GUARD,
    }


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def resource_binding_semantics(
    rows: Iterable[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    row_list = list(rows)
    mode = str(
        summary_json.get(
            "continuous_resource_binding_mode",
            final_metrics.get("continuous_resource_binding_mode", ""),
        )
        or ""
    ).strip()
    if not mode:
        for row in row_list:
            raw = row.get("continuous_resource_binding_mode", row.get("continuousResourceBindingMode", ""))
            if raw not in (None, ""):
                mode = str(raw).strip()
                break
    mode = mode or "candidate_only"
    mode = mode.lower()
    if mode == "native_scheduler_bound":
        native = _to_bool(
            summary_json.get("native_scheduler_bound", final_metrics.get("native_scheduler_bound")),
            False,
        )
        if not native:
            mode = "resource_aware_estimator_bound"
    if mode not in {"candidate_only", "resource_aware_estimator_bound", "native_scheduler_bound"}:
        mode = "candidate_only"

    estimator = mode == "resource_aware_estimator_bound" or _to_bool(
        summary_json.get("estimator_bound", final_metrics.get("estimator_bound")),
        False,
    )
    native_flag = mode == "native_scheduler_bound" and _to_bool(
        summary_json.get("native_scheduler_bound", final_metrics.get("native_scheduler_bound")),
        False,
    )
    native_evidence = _to_bool(
        summary_json.get("native_binding_applied", final_metrics.get("native_binding_applied")),
        False,
    )
    for row in row_list:
        native_evidence = native_evidence or _to_bool(row.get("native_binding_applied", row.get("nativeBindingApplied")), False)
        native_evidence = native_evidence or (
            _to_bool(row.get("native_cpu_mips_bound", row.get("nativeCpuMipsBound")), False)
            and _to_bool(row.get("native_network_bandwidth_bound", row.get("nativeNetworkBandwidthBound")), False)
            and _to_bool(row.get("native_tx_power_bound", row.get("nativeTxPowerBound")), False)
        )
    native = native_flag and native_evidence
    applied = native or estimator or _to_bool(
        summary_json.get("continuous_resource_applied", final_metrics.get("continuous_resource_applied")),
        False,
    )
    if native:
        mode = "native_scheduler_bound"
    elif estimator:
        mode = "resource_aware_estimator_bound"
    else:
        mode = "candidate_only"
        applied = False

    title = (
        "SatEdgeSim full hybrid native-scheduler replay"
        if native
        else "SatEdgeSim resource-aware estimator-bound replay"
        if estimator
        else "SatEdgeSim candidate-level action-mapping replay"
    )
    guard = (
        "Native VM/network/power scheduler binding is enabled and completion receipts are required for closed-loop claims."
        if native
        else ESTIMATOR_CLAIM_GUARD
        if estimator
        else CLAIM_GUARD
    )
    return {
        "satedgesim_validation_mode": (
            "full_hybrid_native_scheduler_replay"
            if native
            else "resource_aware_estimator_bound_replay"
            if estimator
            else "candidate_level_discrete_replay"
        ),
        "continuous_resource_binding_mode": mode,
        "resource_binding_mode": mode,
        "continuous_resource_applied": bool(applied),
        "native_scheduler_bound": bool(native),
        "estimator_bound": bool(estimator),
        "full_hybrid_closed_loop_claim_allowed": bool(native),
        "lower_continuous_allocator_validated_by_satedgesim": bool(native),
        "continuous_resource_applied_to_native_scheduler": bool(native),
        "cpu_share_effective": bool(native or estimator),
        "bandwidth_share_effective": bool(native or estimator),
        "tx_power_ratio_effective": bool(native or estimator),
        "native_binding_evidence": bool(native_evidence),
        "table5_title_suggestion": title,
        "claim_guard": guard,
        "energy_advantage_claim_allowed": False,
    }


def require_native_scheduler_bound_for_formal_claim(
    rows: Iterable[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> None:
    info = resource_binding_semantics(rows, summary_json, final_metrics)
    if not bool(info.get("native_scheduler_bound")):
        raise ValueError(
            "formal SatEdgeSim lower continuous validation requires native_scheduler_bound=true with "
            "completion-level evidence that CPU, bandwidth, and tx power were bound natively"
        )


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _present(value: Any) -> bool:
    return value not in (None, "")


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def is_completion_receipt(row: Mapping[str, Any]) -> bool:
    stage = str(_field(row, "receiptStage", "receipt_stage") or "").strip().lower()
    return stage == "completion" or bool(_field(row, "completion_observed")) is True


def normalize_scheduling_receipt(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    stage = str(out.get("receiptStage", out.get("receipt_stage", "")) or "").strip().lower()
    if stage == "scheduling":
        out["receiptStage"] = "scheduling"
        out["receipt_stage"] = "scheduling"
        out["taskCompleted"] = None
        out["taskSucceeded"] = None
        out["success"] = None
        out.setdefault("completion_observed", False)
    return out


def _receipt_id(row: Mapping[str, Any], *names: str) -> str:
    value = _field(row, *names)
    return "" if value in (None, "") else str(value)


def join_completion_receipt(scheduling_row: Mapping[str, Any], completion_receipts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    out = dict(scheduling_row)
    if is_completion_receipt(out):
        out["completion_observed"] = True
        return out
    decision_id = _receipt_id(out, "receipt_decision_id", "decisionId", "decision_id")
    task_id = _receipt_id(out, "receipt_task_id", "taskId", "task_id")
    match = None
    for receipt in completion_receipts:
        if not is_completion_receipt(receipt):
            continue
        rid = _receipt_id(receipt, "decisionId", "receipt_decision_id", "decision_id")
        tid = _receipt_id(receipt, "taskId", "receipt_task_id", "task_id")
        if (decision_id and rid == decision_id) or (task_id and tid == task_id):
            match = receipt
            break
    if match is None:
        out["completion_observed"] = False
        out["taskCompleted"] = None
        out["taskSucceeded"] = None
        out["success"] = None
        return out
    out["completion_observed"] = True
    out["taskCompleted"] = _field(match, "taskCompleted", "task_completed")
    out["taskSucceeded"] = _field(match, "taskSucceeded", "task_succeeded")
    out["success"] = _field(match, "taskSucceeded", "task_succeeded", "success")
    out["failureReason"] = _field(match, "failureReason", "failure_reason")
    sim_time = _field(match, "simulationTime", "simulation_time")
    if sim_time is not None:
        out["completion_simulation_time"] = sim_time
    return out


def completion_observed_ratio(rows: Iterable[Mapping[str, Any]]) -> float:
    row_list = list(rows)
    if not row_list:
        return 0.0
    observed = 0
    for row in row_list:
        observed += int(is_completion_receipt(row) or bool(row.get("completion_observed")))
    return float(observed / len(row_list))


def has_completion_evidence(
    rows: Iterable[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> bool:
    for row in rows:
        if is_completion_receipt(row) or bool(row.get("completion_observed")):
            return True
        stage = str(_field(row, "receiptStage", "receipt_stage") or "").strip().lower()
        if stage == "scheduling":
            continue
        if _present(row.get("taskCompleted")) and str(row.get("taskCompleted")).strip().lower() in {"1", "true", "yes", "y"}:
            return True
        if _present(row.get("completionReceipt")):
            return True
    if any(_present(summary_json.get(k)) for k in ("completion_success_ratio", "task_completion_success_ratio")):
        return True
    return any(
        _present(final_metrics.get(k))
        for k in ("successRate", "tasksSent", "tasksFinished", "tasksCompleted", "tasksFailed")
    )


def completion_success_ratio(
    rows: Iterable[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> Optional[float]:
    explicit = _to_float(summary_json.get("completion_success_ratio", summary_json.get("task_completion_success_ratio")))
    if explicit is not None:
        return explicit
    final_success = _to_float(final_metrics.get("successRate"))
    if final_success is not None:
        return final_success
    completed = 0
    succeeded = 0
    for row in rows:
        if is_completion_receipt(row) or bool(row.get("completion_observed")):
            completed += 1
            raw = row.get("taskSucceeded", row.get("success"))
            succeeded += int(str(raw).strip().lower() in {"1", "true", "yes", "y"})
    if completed <= 0:
        return None
    return float(succeeded / completed)


def canonical_energy_unit(raw_unit: Any) -> str:
    unit = str(raw_unit or "").strip().lower()
    if unit in {"j", "joule", "joules"}:
        return "J"
    if unit in {"wh", "watt-hour", "watt-hours", "watthour", "watthours"}:
        return "Wh"
    if unit in {"normalized", "dimensionless"}:
        return "normalized"
    if unit in {"", "unknown", "none"}:
        return "unknown"
    return "simulator_counter"


def energy_semantics(
    rows: Iterable[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    *,
    energy_source: str | None = None,
) -> Dict[str, Any]:
    row_list = list(rows)
    source_request = str(energy_source or summary_json.get("energy_source") or "").strip().lower()

    receipt_values = []
    for row in row_list:
        raw = row.get("receipt_energy_delta_wh", row.get("energy_raw_delta", row.get("energyDelta")))
        val = _to_float(raw)
        if val is not None:
            receipt_values.append(val)
    receipt_delta_wh = float(sum(receipt_values)) if receipt_values else _to_float(summary_json.get("receipt_energy_delta_wh", summary_json.get("receipt_energy_delta")))

    simlog_final_wh = _to_float(
        summary_json.get(
            "simlog_final_energy_wh",
            summary_json.get("final_cumulative_energy", summary_json.get("raw_energy_counter_final", final_metrics.get("energyConsumption"))),
        )
    )
    estimator_values = [_to_float(row.get("estimator_expected_energy_j")) for row in row_list if _to_float(row.get("estimator_expected_energy_j")) is not None]
    estimator_expected_j = float(sum(v for v in estimator_values if v is not None)) if estimator_values else _to_float(summary_json.get("estimator_expected_energy_j"))

    selected_value = None
    selected_unit = "unknown"
    unavailable_reason = ""
    selected_source = source_request
    if source_request in {"", "auto"}:
        if simlog_final_wh is not None:
            selected_source, selected_value, selected_unit = "simlog_final_wh", simlog_final_wh, "Wh"
        elif receipt_delta_wh is not None:
            selected_source, selected_value, selected_unit = "receipt_delta_wh", receipt_delta_wh, "Wh"
        elif estimator_expected_j is not None:
            selected_source, selected_value, selected_unit = "estimator_expected_j", estimator_expected_j, "J"
        else:
            selected_source, unavailable_reason = "unavailable", "energy_unavailable"
    elif source_request == "simlog_final_wh":
        selected_value, selected_unit = simlog_final_wh, "Wh"
        unavailable_reason = "simlog_final_wh_unavailable" if selected_value is None else ""
    elif source_request == "receipt_delta_wh":
        selected_value, selected_unit = receipt_delta_wh, "Wh"
        unavailable_reason = "receipt_delta_wh_unavailable" if selected_value is None else ""
    elif source_request == "estimator_expected_j":
        selected_value, selected_unit = estimator_expected_j, "J"
        unavailable_reason = "estimator_expected_j_unavailable" if selected_value is None else ""
    else:
        selected_source, unavailable_reason = "unavailable", f"unsupported_energy_source_{source_request}"

    available = selected_value is not None and selected_source != "unavailable"
    if not available:
        selected_source = "unavailable"
    raw_unit = summary_json.get("energy_unit", final_metrics.get("energyCounterUnit"))
    return {
        "simlog_final_energy_wh": simlog_final_wh,
        "final_cumulative_energy": simlog_final_wh,
        "receipt_energy_delta_wh": receipt_delta_wh,
        "receipt_energy_delta": receipt_delta_wh,
        "estimator_expected_energy_j": estimator_expected_j,
        "selected_energy_value": selected_value,
        "energy_source": selected_source,
        "energy_unit": selected_unit if available or source_request in {"simlog_final_wh", "receipt_delta_wh", "estimator_expected_j"} else canonical_energy_unit(raw_unit),
        "energy_source_available": bool(available),
        "energy_unavailable_reason": unavailable_reason,
        "energy_semantics": (
            "receipt_delta_is_not_final_task_energy"
            if selected_source == "receipt_delta_wh"
            else "final_cumulative_counter_available"
            if selected_source == "simlog_final_wh" and available
            else "estimator_expected_energy_not_simlog_final"
            if selected_source == "estimator_expected_j" and available
            else "energy_unavailable"
        ),
    }
