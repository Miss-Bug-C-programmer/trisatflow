from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError
from trisatflow.baselines.registry import apply_architecture_filter, normalize_architecture
from trisatflow.envs.physical_metrics import energy_delta_from_cumulative_wh

TERMINAL_STATUSES = {"FINISHED", "CLOSED", "FAILED", "ERROR"}
ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
EXPORTER_VERSION = "paper_v3_exporter_v1"
MASK_FIELD_KEYS = {
    "visible": ("abstractActionMaskVisible", "abstract_action_mask_visible"),
    "completion_safe": ("abstractActionMaskCompletionSafe", "abstract_action_mask_completion_safe"),
    "mobility_safe": ("abstractActionMaskMobilitySafe", "abstract_action_mask_mobility_safe"),
    "final": ("abstractActionMask", "abstract_action_mask", "abstractActionMaskFinal", "abstract_action_mask_final"),
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_is_feasible(vm: Dict[str, Any], mask: Sequence[Any], idx: int) -> bool:
    if "isFeasible" in vm:
        return bool(vm.get("isFeasible"))
    if idx < len(mask):
        return bool(mask[idx])
    if "feasible" in vm:
        return bool(vm.get("feasible"))
    return True


def _mask_field_presence(payload: Dict[str, Any]) -> Dict[str, bool]:
    return {name: any(key in payload for key in keys) for name, keys in MASK_FIELD_KEYS.items()}


def _final_mask_for_mode(
    *,
    action_mask_mode: str,
    abstract_mask: Sequence[Any],
    visible_mask: Sequence[Any],
    mobility_safe_mask: Sequence[Any],
    completion_safe_mask: Sequence[Any],
) -> List[int]:
    mode = str(action_mask_mode or "visible_only").strip().lower()
    if mode in {"none", "no_mask"}:
        raw = [1, 1, 1, 1]
    elif mode in {"full", "full_mask", "completion_safe"}:
        raw = completion_safe_mask
    elif mode in {"mobility_safe", "mobility_risk"}:
        raw = mobility_safe_mask
    elif mode in {"visible_only", "visibility", "visibility_only"}:
        raw = visible_mask
    else:
        raw = abstract_mask
    out = [int(bool(x)) for x in list(raw)[:4]]
    if len(out) < 4:
        out = (out + [0, 0, 0, 0])[:4]
    if not any(out):
        visible = [int(bool(x)) for x in list(visible_mask)[:4]]
        out = visible if any(visible) else [1, 0, 0, 0]
    return out


def _phase_id(scenario_phase: Any, traffic_phase: Any) -> str:
    scenario = str(scenario_phase or "default_phase").strip() or "default_phase"
    traffic = str(traffic_phase or "default_traffic").strip() or "default_traffic"
    return f"{scenario}:{traffic}"


def _coverage_status(*, trace_mode: str, dense_supported: bool, sparse_steps: int, missing_pairs: int, num_rows: int) -> str:
    mode = str(trace_mode or "dense_projection").strip().lower()
    if mode == "sequential_live":
        return "SEQUENTIAL_TRACE_OK" if num_rows > 0 and missing_pairs == 0 else "SEQUENTIAL_TRACE_INCOMPLETE"
    return "DENSE_TRACE_OK" if dense_supported and sparse_steps == 0 and missing_pairs == 0 else "DENSE_TRACE_INCOMPLETE"


def _annotate_contract_fields(row: Dict[str, Any], args: argparse.Namespace, *, trace_mode: str) -> Dict[str, Any]:
    out = dict(row)
    mode = str(trace_mode)
    out["trace_origin"] = "satedgesim"
    out["synthetic"] = False
    out["trace_semantic_class"] = str(getattr(args, "trace_semantic_class", "") or "actual_physical")
    out["trace_generation_mode"] = mode
    out["dense_projection_mode"] = "source_projection" if mode == "dense_projection" else "none"
    out["success_profile"] = str(args.success_profile)
    out["scenario_profile"] = str(out.get("scenario_profile", args.scenario_profile))
    out["scenario_phase"] = str(out.get("scenario_phase", "default_phase"))
    out["task_source_mode"] = str(out.get("task_source_mode", args.task_source_mode))
    queue_source = out.get("queue_estimate_source", out.get("queueEstimateSource", "unknown"))
    mobility_source = out.get("mobility_risk_source", out.get("mobilityRiskSource", "unavailable"))
    cost_version = out.get("candidate_cost_estimator_version", out.get("cost_estimator_version", "v1_unified_delay_queue"))
    semantic_class = str(out["trace_semantic_class"]).strip().lower()
    if "controlled" in semantic_class:
        queue_source = "controlled_estimate"
        mobility_source = "controlled_estimate"
    out["queue_estimate_source"] = queue_source
    out["mobility_risk_source"] = mobility_source
    out["candidate_cost_estimator_version"] = cost_version
    semantic = str(out.get("delay_semantic") or "").strip()
    if not semantic:
        semantic = "physical_seconds_controlled_estimate" if "controlled" in semantic_class else "physical_seconds_actual"
    out["delay_semantic"] = semantic
    return out


def _state_energy_counter_wh(state: Dict[str, Any]) -> float | None:
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    raw = metrics.get("energyConsumption", state.get("energyConsumption"))
    if raw in (None, ""):
        return None
    return _to_float(raw, 0.0)


def _attach_energy_delta_fields(row: Dict[str, Any], *, raw_wh: float | None, previous_wh: float | None) -> Dict[str, Any]:
    out = dict(row)
    if raw_wh is None:
        return out
    previous = raw_wh if previous_wh is None else previous_wh
    out.update(energy_delta_from_cumulative_wh(raw_wh, previous))
    return out


def _abstract_action(vm: Dict[str, Any], task: Dict[str, Any]) -> int:
    try:
        action = int(vm.get("abstractAction"))
        if 0 <= action <= 3:
            return action
    except (TypeError, ValueError):
        pass
    logical = str(vm.get("logicalTier") or "").strip().lower()
    if logical == "local":
        return 0
    if logical == "neighbor":
        return 1
    if logical in {"geo", "cloud"}:
        return 2
    if logical in {"ground", "edge"}:
        return 3
    source_dc = task.get("sourceDatacenterId")
    if source_dc is not None and str(vm.get("datacenterId")) == str(source_dc):
        return 0
    return -1


def _scaled_rate(vm: Dict[str, Any], *, rate_scale_mbps: float, max_trace_rate: float) -> float:
    raw = _to_float(vm.get("estimatedTransmissionRateMbps"), 0.0)
    if raw <= 0.0:
        raw = _to_float(vm.get("bw"), 0.0)
    return max(0.0, min(max_trace_rate, max_trace_rate * raw / max(rate_scale_mbps, 1.0e-6)))


def _source_leo_id(state: Dict[str, Any], n_leo: int) -> int:
    task = dict(state.get("task") or {})
    if task.get("sourceDeviceId") is not None:
        return int(_to_float(task.get("sourceDeviceId"), 0.0)) % max(1, n_leo)
    if task.get("sourceDatacenterId") is not None:
        return int(_to_float(task.get("sourceDatacenterId"), 0.0)) % max(1, n_leo)
    return int(_to_float(state.get("requestId"), 0.0)) % max(1, n_leo)


def _snapshot_from_state(
    state: Dict[str, Any],
    *,
    step: int,
    n_leo: int,
    rate_scale_mbps: float,
    max_trace_rate: float,
    architecture: str = "full",
) -> Dict[str, Any]:
    candidates = list(state.get("candidateVms") or [])
    action_mask = list(state.get("actionMask") or [])
    task = dict(state.get("task") or {})
    abstract_mask = abstract_action_mask_from_state(state)
    presence = _mask_field_presence(state)
    visible_mask = list(state.get("abstractActionMaskVisible") or abstract_mask)
    mobility_safe_mask = list(state.get("abstractActionMaskMobilitySafe") or [0, 0, 0, 0])
    completion_safe_mask = list(state.get("abstractActionMaskCompletionSafe") or [0, 0, 0, 0])
    architecture = normalize_architecture(architecture)
    abstract_mask = apply_architecture_filter(abstract_mask, architecture)
    visible_mask = apply_architecture_filter(visible_mask, architecture)
    mobility_safe_mask = apply_architecture_filter(mobility_safe_mask, architecture)
    completion_safe_mask = apply_architecture_filter(completion_safe_mask, architecture)
    action_mask_mode = str(state.get("actionMaskMode") or "visible_only")
    final_mask = _final_mask_for_mode(
        action_mask_mode=action_mask_mode,
        abstract_mask=abstract_mask,
        visible_mask=visible_mask,
        mobility_safe_mask=mobility_safe_mask,
        completion_safe_mask=completion_safe_mask,
    )
    min_link_margin = _to_float(state.get("minLinkSurvivalMarginSec"), 0.0)
    scenario_phase = task.get("scenarioPhase", state.get("scenarioPhase", "default_phase"))
    task_type = task.get("taskType", state.get("taskType", "unknown_task"))
    traffic_phase = task.get("trafficPhase", state.get("trafficPhase", "default_traffic"))

    counts = [0, 0, 0, 0]
    rates = [0.0, 0.0, 0.0, 0.0]
    min_dist = [None, None, None, None]
    best_queue = [None, None, None, None]
    best_prop_delay = [None, None, None, None]
    best_tx_delay = [None, None, None, None]
    best_compute_delay = [None, None, None, None]
    best_compute_capacity = [None, None, None, None]
    best_queue_delay = [None, None, None, None]
    best_delay = [None, None, None, None]
    queue_sources = set()
    mobility_sources = set()
    safe_counts = [0, 0, 0, 0]
    completion_counts = [0, 0, 0, 0]
    risk_sums = [0.0, 0.0, 0.0, 0.0]
    best_link_lifetime = [None, None, None, None]
    best_link_margin = [None, None, None, None]
    best_link_margin_to_completion = [None, None, None, None]
    handover_counts = [0, 0, 0, 0]

    for idx, vm in enumerate(candidates):
        action = _abstract_action(vm, task)
        if not (0 <= action <= 3):
            continue
        if not _candidate_is_feasible(vm, action_mask, idx):
            continue
        counts[action] += 1
        rates[action] = max(rates[action], _scaled_rate(vm, rate_scale_mbps=rate_scale_mbps, max_trace_rate=max_trace_rate))
        dist = _to_float(vm.get("sourceDistance", vm.get("distanceToSource", 0.0)), 0.0)
        queue = _to_float(vm.get("estimatedQueueLength", vm.get("assignedTasks", 0.0)), 0.0)
        prop_delay = _to_float(vm.get("propagationDelaySec"), 0.0)
        tx_delay = _to_float(vm.get("estimatedTransmissionDelaySec"), 0.0)
        compute_delay = _to_float(vm.get("estimatedComputeDelaySec"), 0.0)
        compute_capacity = _to_float(vm.get("estimatedComputeCapacity"), 0.0)
        queue_delay = _to_float(vm.get("estimatedQueueDelaySec"), queue * compute_delay)
        delay = _to_float(
            vm.get("estimatedTotalDelaySec"),
            tx_delay + prop_delay + compute_delay + queue_delay,
        )
        if bool(vm.get("mobilitySafe")):
            safe_counts[action] += 1
        if bool(vm.get("completionSafe")):
            completion_counts[action] += 1
        risk = _to_float(vm.get("mobilityRisk"), 1.0)
        risk_sums[action] += risk
        life = _to_float(vm.get("estimatedLinkLifetimeSec"), 0.0)
        margin = _to_float(vm.get("linkSurvivalMarginSec"), 0.0)
        margin_to_completion = _to_float(vm.get("linkSurvivalMarginToCompletionSec"), margin)
        best_link_lifetime[action] = life if best_link_lifetime[action] is None else min(float(best_link_lifetime[action]), life)
        best_link_margin[action] = margin if best_link_margin[action] is None else max(float(best_link_margin[action]), margin)
        best_link_margin_to_completion[action] = margin_to_completion if best_link_margin_to_completion[action] is None else max(float(best_link_margin_to_completion[action]), margin_to_completion)
        if bool(vm.get("handoverRequired", False)):
            handover_counts[action] += 1
        ms = str(vm.get("mobilityRiskSource") or "").strip()
        if ms:
            mobility_sources.add(ms)
        source = str(vm.get("queueEstimateSource") or "").strip()
        if source:
            queue_sources.add(source)
        min_dist[action] = dist if min_dist[action] is None else min(float(min_dist[action]), dist)
        if best_delay[action] is None or delay < float(best_delay[action]):
            best_queue[action] = queue
            best_prop_delay[action] = prop_delay
            best_tx_delay[action] = tx_delay
            best_compute_delay[action] = compute_delay
            best_compute_capacity[action] = compute_capacity
            best_queue_delay[action] = queue_delay
            best_delay[action] = delay

    queue_estimate_source = "unknown"
    if len(queue_sources) == 1:
        queue_estimate_source = next(iter(queue_sources))
    elif len(queue_sources) > 1:
        queue_estimate_source = "mixed"
    mobility_risk_source = "unavailable"
    if len(mobility_sources) == 1:
        mobility_risk_source = next(iter(mobility_sources))
    elif len(mobility_sources) > 1:
        mobility_risk_source = "mixed"

    return {
        "step": int(step),
        "leo_id": _source_leo_id(state, n_leo),
        "local_visible": bool(visible_mask[0]),
        "neighbor_visible": bool(visible_mask[1]),
        "geo_visible": bool(visible_mask[2]),
        "ground_visible": bool(visible_mask[3]),
        "local_rate": float(rates[0]),
        "neighbor_rate": float(rates[1]),
        "geo_rate": float(rates[2]),
        "ground_rate": float(rates[3]),
        "local_candidate_count": int(counts[0]),
        "neighbor_candidate_count": int(counts[1]),
        "geo_candidate_count": int(counts[2]),
        "ground_candidate_count": int(counts[3]),
        "neighbor_min_distance": min_dist[1],
        "geo_min_distance": min_dist[2],
        "ground_min_distance": min_dist[3],
        "local_best_queue": best_queue[0],
        "neighbor_best_queue": best_queue[1],
        "geo_best_queue": best_queue[2],
        "ground_best_queue": best_queue[3],
        "local_prop_delay": best_prop_delay[0],
        "neighbor_prop_delay": best_prop_delay[1],
        "geo_prop_delay": best_prop_delay[2],
        "ground_prop_delay": best_prop_delay[3],
        "local_tx_delay": best_tx_delay[0],
        "neighbor_tx_delay": best_tx_delay[1],
        "geo_tx_delay": best_tx_delay[2],
        "ground_tx_delay": best_tx_delay[3],
        "local_compute_delay": best_compute_delay[0],
        "neighbor_compute_delay": best_compute_delay[1],
        "geo_compute_delay": best_compute_delay[2],
        "ground_compute_delay": best_compute_delay[3],
        "local_compute_capacity": best_compute_capacity[0],
        "neighbor_compute_capacity": best_compute_capacity[1],
        "geo_compute_capacity": best_compute_capacity[2],
        "ground_compute_capacity": best_compute_capacity[3],
        "local_queue_delay": best_queue_delay[0],
        "neighbor_queue_delay": best_queue_delay[1],
        "geo_queue_delay": best_queue_delay[2],
        "ground_queue_delay": best_queue_delay[3],
        "local_total_delay": best_delay[0],
        "neighbor_total_delay": best_delay[1],
        "geo_total_delay": best_delay[2],
        "ground_total_delay": best_delay[3],
        "local_best_delay": best_delay[0],
        "neighbor_best_delay": best_delay[1],
        "geo_best_delay": best_delay[2],
        "ground_best_delay": best_delay[3],
        "abstract_action_mask": [int(x) for x in final_mask],
        "abstract_action_mask_visible": [int(bool(x)) for x in visible_mask[:4]],
        "abstract_action_mask_mobility_safe": [int(bool(x)) for x in mobility_safe_mask[:4]],
        "abstract_action_mask_completion_safe": [int(bool(x)) for x in completion_safe_mask[:4]],
        "abstract_action_mask_final": [int(x) for x in final_mask],
        "mask_field_presence": presence,
        "action_mask_mode": action_mask_mode,
        "min_link_survival_margin_sec": min_link_margin,
        "local_mobility_safe": bool(safe_counts[0] > 0),
        "neighbor_mobility_safe": bool(safe_counts[1] > 0),
        "geo_mobility_safe": bool(safe_counts[2] > 0),
        "ground_mobility_safe": bool(safe_counts[3] > 0),
        "local_completion_safe": bool(completion_counts[0] > 0),
        "neighbor_completion_safe": bool(completion_counts[1] > 0),
        "geo_completion_safe": bool(completion_counts[2] > 0),
        "ground_completion_safe": bool(completion_counts[3] > 0),
        "local_mobility_risk_mean": 0.0 if counts[0] <= 0 else risk_sums[0] / counts[0],
        "neighbor_mobility_risk_mean": 0.0 if counts[1] <= 0 else risk_sums[1] / counts[1],
        "geo_mobility_risk_mean": 0.0 if counts[2] <= 0 else risk_sums[2] / counts[2],
        "ground_mobility_risk_mean": 0.0 if counts[3] <= 0 else risk_sums[3] / counts[3],
        "local_best_link_lifetime_sec": best_link_lifetime[0],
        "neighbor_best_link_lifetime_sec": best_link_lifetime[1],
        "geo_best_link_lifetime_sec": best_link_lifetime[2],
        "ground_best_link_lifetime_sec": best_link_lifetime[3],
        "local_best_link_survival_margin_sec": best_link_margin[0],
        "neighbor_best_link_survival_margin_sec": best_link_margin[1],
        "geo_best_link_survival_margin_sec": best_link_margin[2],
        "ground_best_link_survival_margin_sec": best_link_margin[3],
        "local_best_link_survival_margin_to_completion_sec": best_link_margin_to_completion[0],
        "neighbor_best_link_survival_margin_to_completion_sec": best_link_margin_to_completion[1],
        "geo_best_link_survival_margin_to_completion_sec": best_link_margin_to_completion[2],
        "ground_best_link_survival_margin_to_completion_sec": best_link_margin_to_completion[3],
        "local_handover_required": bool(handover_counts[0] > 0),
        "neighbor_handover_required": bool(handover_counts[1] > 0),
        "geo_handover_required": bool(handover_counts[2] > 0),
        "ground_handover_required": bool(handover_counts[3] > 0),
        "mobilityRiskSource": mobility_risk_source,
        "mobility_risk_source": mobility_risk_source,
        "trace_origin": "satedgesim",
        "synthetic": False,
        "trace_semantic_class": "actual_physical",
        "delay_semantic": "physical_seconds_actual",
        "request_id": state.get("requestId"),
        "simulation_time": state.get("simulationTime"),
        "scenario_profile": state.get("scenarioProfile", "default"),
        "scenario_phase": scenario_phase,
        "phase_id": _phase_id(scenario_phase, traffic_phase),
        "task_type": task_type,
        "traffic_phase": traffic_phase,
        "task_source_mode": state.get("taskSourceMode", "current"),
        "architecture": architecture,
        "is_controlled_rl_scenario": bool(state.get("isControlledRlScenario", False)),
        "queueEstimateSource": queue_estimate_source,
        "queue_estimate_source": queue_estimate_source,
        "trace_generation_mode": "sequential_live",
        "dense_projection_mode": "none",
        "success_profile": state.get("successProfile", "default"),
        "cost_estimator_version": state.get("costEstimatorVersion", "v1_unified_delay_queue"),
        "candidate_cost_estimator_version": state.get("costEstimatorVersion", "v1_unified_delay_queue"),
    }


def _snapshot_from_dense_summary(
    summary: Dict[str, Any],
    *,
    step: int,
    leo_id: int,
) -> Dict[str, Any]:
    raw_mask = list(summary.get("abstractActionMask") or summary.get("abstract_action_mask") or [])
    presence = _mask_field_presence(summary)
    raw_visible = list(summary.get("abstractActionMaskVisible") or [])
    raw_mobility = list(summary.get("abstractActionMaskMobilitySafe") or [])
    raw_completion = list(summary.get("abstractActionMaskCompletionSafe") or [])
    if len(raw_mask) < 4:
        raw_mask = [
            int(bool(summary.get("localVisible"))),
            int(bool(summary.get("neighborVisible"))),
            int(bool(summary.get("geoVisible"))),
            int(bool(summary.get("groundVisible"))),
        ]
    if len(raw_visible) < 4:
        raw_visible = raw_mask
    if len(raw_mobility) < 4:
        raw_mobility = [
            int(bool(summary.get("localMobilitySafe"))),
            int(bool(summary.get("neighborMobilitySafe"))),
            int(bool(summary.get("geoMobilitySafe"))),
            int(bool(summary.get("groundMobilitySafe"))),
        ]
    if len(raw_completion) < 4:
        raw_completion = [
            int(bool(summary.get("localCompletionSafe"))),
            int(bool(summary.get("neighborCompletionSafe"))),
            int(bool(summary.get("geoCompletionSafe"))),
            int(bool(summary.get("groundCompletionSafe"))),
        ]
    action_mask_mode = summary.get("actionMaskMode", "visible_only")
    final_mask = _final_mask_for_mode(
        action_mask_mode=str(action_mask_mode),
        abstract_mask=raw_mask,
        visible_mask=raw_visible,
        mobility_safe_mask=raw_mobility,
        completion_safe_mask=raw_completion,
    )
    scenario_phase = summary.get("scenarioPhase", "default_phase")
    traffic_phase = summary.get("trafficPhase", "default_traffic")
    return {
        "step": int(step),
        "leo_id": int(leo_id),
        "local_visible": bool(summary.get("localVisible", raw_mask[0] if len(raw_mask) > 0 else True)),
        "neighbor_visible": bool(summary.get("neighborVisible", raw_mask[1] if len(raw_mask) > 1 else False)),
        "geo_visible": bool(summary.get("geoVisible", raw_mask[2] if len(raw_mask) > 2 else False)),
        "ground_visible": bool(summary.get("groundVisible", raw_mask[3] if len(raw_mask) > 3 else False)),
        "local_rate": _to_float(summary.get("localRate"), 0.0),
        "neighbor_rate": _to_float(summary.get("neighborRate"), 0.0),
        "geo_rate": _to_float(summary.get("geoRate"), 0.0),
        "ground_rate": _to_float(summary.get("groundRate"), 0.0),
        "local_candidate_count": int(_to_float(summary.get("localCandidateCount"), 0.0)),
        "neighbor_candidate_count": int(_to_float(summary.get("neighborCandidateCount"), 0.0)),
        "geo_candidate_count": int(_to_float(summary.get("geoCandidateCount"), 0.0)),
        "ground_candidate_count": int(_to_float(summary.get("groundCandidateCount"), 0.0)),
        "neighbor_min_distance": summary.get("neighborMinDistance"),
        "geo_min_distance": summary.get("geoMinDistance"),
        "ground_min_distance": summary.get("groundMinDistance"),
        "local_best_queue": summary.get("localBestQueue"),
        "neighbor_best_queue": summary.get("neighborBestQueue"),
        "geo_best_queue": summary.get("geoBestQueue"),
        "ground_best_queue": summary.get("groundBestQueue"),
        "local_prop_delay": summary.get("localPropDelay"),
        "neighbor_prop_delay": summary.get("neighborPropDelay"),
        "geo_prop_delay": summary.get("geoPropDelay"),
        "ground_prop_delay": summary.get("groundPropDelay"),
        "local_tx_delay": summary.get("localTxDelay"),
        "neighbor_tx_delay": summary.get("neighborTxDelay"),
        "geo_tx_delay": summary.get("geoTxDelay"),
        "ground_tx_delay": summary.get("groundTxDelay"),
        "local_compute_delay": summary.get("localComputeDelay"),
        "neighbor_compute_delay": summary.get("neighborComputeDelay"),
        "geo_compute_delay": summary.get("geoComputeDelay"),
        "ground_compute_delay": summary.get("groundComputeDelay"),
        "local_compute_capacity": summary.get("localComputeCapacity"),
        "neighbor_compute_capacity": summary.get("neighborComputeCapacity"),
        "geo_compute_capacity": summary.get("geoComputeCapacity"),
        "ground_compute_capacity": summary.get("groundComputeCapacity"),
        "local_queue_delay": summary.get("localQueueDelay"),
        "neighbor_queue_delay": summary.get("neighborQueueDelay"),
        "geo_queue_delay": summary.get("geoQueueDelay"),
        "ground_queue_delay": summary.get("groundQueueDelay"),
        "local_total_delay": summary.get("localTotalDelay", summary.get("localBestDelay")),
        "neighbor_total_delay": summary.get("neighborTotalDelay", summary.get("neighborBestDelay")),
        "geo_total_delay": summary.get("geoTotalDelay", summary.get("geoBestDelay")),
        "ground_total_delay": summary.get("groundTotalDelay", summary.get("groundBestDelay")),
        "local_best_delay": summary.get("localBestDelay"),
        "neighbor_best_delay": summary.get("neighborBestDelay"),
        "geo_best_delay": summary.get("geoBestDelay"),
        "ground_best_delay": summary.get("groundBestDelay"),
        "abstract_action_mask": [int(bool(x)) for x in final_mask[:4]],
        "abstract_action_mask_visible": [int(bool(x)) for x in raw_visible[:4]],
        "abstract_action_mask_mobility_safe": [int(bool(x)) for x in raw_mobility[:4]],
        "abstract_action_mask_completion_safe": [int(bool(x)) for x in raw_completion[:4]],
        "abstract_action_mask_final": [int(bool(x)) for x in final_mask[:4]],
        "mask_field_presence": presence,
        "action_mask_mode": action_mask_mode,
        "min_link_survival_margin_sec": _to_float(summary.get("minLinkSurvivalMarginSec"), 0.0),
        "local_mobility_safe": bool(summary.get("localMobilitySafe")),
        "neighbor_mobility_safe": bool(summary.get("neighborMobilitySafe")),
        "geo_mobility_safe": bool(summary.get("geoMobilitySafe")),
        "ground_mobility_safe": bool(summary.get("groundMobilitySafe")),
        "local_completion_safe": bool(summary.get("localCompletionSafe")),
        "neighbor_completion_safe": bool(summary.get("neighborCompletionSafe")),
        "geo_completion_safe": bool(summary.get("geoCompletionSafe")),
        "ground_completion_safe": bool(summary.get("groundCompletionSafe")),
        "local_mobility_risk_mean": _to_float(summary.get("localMobilityRiskMean"), 0.0),
        "neighbor_mobility_risk_mean": _to_float(summary.get("neighborMobilityRiskMean"), 0.0),
        "geo_mobility_risk_mean": _to_float(summary.get("geoMobilityRiskMean"), 0.0),
        "ground_mobility_risk_mean": _to_float(summary.get("groundMobilityRiskMean"), 0.0),
        "local_best_link_lifetime_sec": summary.get("localBestLinkLifetimeSec"),
        "neighbor_best_link_lifetime_sec": summary.get("neighborBestLinkLifetimeSec"),
        "geo_best_link_lifetime_sec": summary.get("geoBestLinkLifetimeSec"),
        "ground_best_link_lifetime_sec": summary.get("groundBestLinkLifetimeSec"),
        "local_best_link_survival_margin_sec": summary.get("localBestLinkSurvivalMarginSec"),
        "neighbor_best_link_survival_margin_sec": summary.get("neighborBestLinkSurvivalMarginSec"),
        "geo_best_link_survival_margin_sec": summary.get("geoBestLinkSurvivalMarginSec"),
        "ground_best_link_survival_margin_sec": summary.get("groundBestLinkSurvivalMarginSec"),
        "local_best_link_survival_margin_to_completion_sec": summary.get("localBestLinkSurvivalMarginToCompletionSec", summary.get("localBestLinkSurvivalMarginSec")),
        "neighbor_best_link_survival_margin_to_completion_sec": summary.get("neighborBestLinkSurvivalMarginToCompletionSec", summary.get("neighborBestLinkSurvivalMarginSec")),
        "geo_best_link_survival_margin_to_completion_sec": summary.get("geoBestLinkSurvivalMarginToCompletionSec", summary.get("geoBestLinkSurvivalMarginSec")),
        "ground_best_link_survival_margin_to_completion_sec": summary.get("groundBestLinkSurvivalMarginToCompletionSec", summary.get("groundBestLinkSurvivalMarginSec")),
        "local_handover_required": bool(summary.get("localHandoverRequired", False)),
        "neighbor_handover_required": bool(summary.get("neighborHandoverRequired", False)),
        "geo_handover_required": bool(summary.get("geoHandoverRequired", False)),
        "ground_handover_required": bool(summary.get("groundHandoverRequired", False)),
        "mobilityRiskSource": summary.get("mobilityRiskSource", "unavailable"),
        "mobility_risk_source": summary.get("mobilityRiskSource", "unavailable"),
        "trace_origin": "satedgesim",
        "synthetic": False,
        "trace_semantic_class": "actual_physical",
        "delay_semantic": "physical_seconds_actual",
        "dense_trace": True,
        "dense_projection_mode": "source_projection",
        "source_device_id": summary.get("sourceDeviceId"),
        "source_datacenter_id": summary.get("sourceDatacenterId"),
        "simulation_time": summary.get("simulationTime"),
        "scenario_profile": summary.get("scenarioProfile", "default"),
        "scenario_phase": scenario_phase,
        "phase_id": _phase_id(scenario_phase, traffic_phase),
        "task_type": summary.get("taskType", "unknown_task"),
        "traffic_phase": traffic_phase,
        "task_source_mode": summary.get("taskSourceMode", "current"),
        "is_controlled_rl_scenario": bool(summary.get("isControlledRlScenario", False)),
        "queueEstimateSource": summary.get("queueEstimateSource", "unknown"),
        "queue_estimate_source": summary.get("queueEstimateSource", "unknown"),
        "trace_generation_mode": summary.get("traceGenerationMode", "dense_projection"),
        "success_profile": summary.get("successProfile", "default"),
        "cost_estimator_version": summary.get("costEstimatorVersion", "v1_unified_delay_queue"),
        "candidate_cost_estimator_version": summary.get("costEstimatorVersion", "v1_unified_delay_queue"),
    }


def _current_dense_summary(state: Dict[str, Any]) -> Dict[str, Any] | None:
    summaries = list(state.get("denseSourceSummaries") or [])
    if not summaries:
        return None
    source_id = state.get("sourceDeviceId")
    if source_id is None and isinstance(state.get("task"), dict):
        source_id = state["task"].get("sourceDeviceId")
    for summary in summaries:
        if int(_to_float(summary.get("sourceDeviceId"), -1.0)) == int(_to_float(source_id, -2.0)):
            return summary
    return summaries[0]


def _advance_action(state: Dict[str, Any]) -> Dict[str, Any]:
    request_id = int(state.get("requestId", -1))
    candidates = list(state.get("candidateVms") or [])
    action_mask = list(state.get("actionMask") or [])
    task = dict(state.get("task") or {})
    preferred_order = [0, 1, 2, 3]
    target_vm_index = -1
    selected_abstract = -1
    for action in preferred_order:
        best_idx = -1
        best_delay = None
        for idx, vm in enumerate(candidates):
            if not _candidate_is_feasible(vm, action_mask, idx):
                continue
            if _abstract_action(vm, task) != action:
                continue
            delay = _to_float(vm.get("estimatedTransmissionDelaySec"), 0.0) + _to_float(vm.get("estimatedComputeDelaySec"), 0.0)
            if best_idx < 0 or delay < (best_delay if best_delay is not None else float("inf")):
                best_idx = int(vm.get("vmIndex", idx))
                best_delay = delay
        if best_idx >= 0:
            target_vm_index = best_idx
            selected_abstract = action
            break
    return {
        "requestId": request_id,
        "targetVmIndex": int(target_vm_index),
        "abstractAction": int(selected_abstract),
        "abstractActionName": ACTION_NAMES[selected_abstract] if selected_abstract >= 0 else "",
        "cpuShare": 1.0,
        "bandwidthShare": 1.0,
        "txPowerRatio": 1.0,
        "queuePriority": 1.0,
        "extra": {"traceExporter": True},
    }


def _wait_for_decision(client: SatEdgeSimClient, poll_sleep_sec: float, max_polls: int = 300) -> Dict[str, Any]:
    state = client.get_state()
    polls = 0
    while state.get("status") == "RUNNING" and polls < max_polls:
        time.sleep(poll_sleep_sec)
        state = client.get_state()
        polls += 1
    return state


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path_for_trace(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _manifest_from_export(
    *,
    trace_path: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    version: Dict[str, Any],
    metrics: Dict[str, Any],
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    semantic = str(args.trace_semantic_class)
    queue_source = "controlled_estimate" if "controlled" in semantic else "live"
    mobility_source = "controlled_estimate" if "controlled" in semantic else "live"
    if rows:
        queue_source = str(rows[0].get("queue_estimate_source", queue_source))
        mobility_source = str(rows[0].get("mobility_risk_source", mobility_source))
    return {
        "trace_sha256": _sha256_file(trace_path),
        "trace_semantic_class": semantic,
        "trace_origin": "satedgesim",
        "synthetic": False,
        "source_simulator_commit": str(version.get("git_commit", "unknown")),
        "simulator_version": str(version.get("simulator_version", "unknown")),
        "rest_api_schema_version": str(version.get("rest_api_schema_version", "unknown")),
        "state_schema_version": str(version.get("state_schema_version", "unknown")),
        "settings_root": str(version.get("settings_root", "")),
        "settings_sha256": str(version.get("settings_sha256", "")),
        "settings_files_sha256": version.get("settings_files_sha256", {}),
        "exporter_version": EXPORTER_VERSION,
        "seed": int(args.seed),
        "scenario_parameters": {
            "devices_count": args.devices_count if args.devices_count is not None else args.n_leo,
            "simulation_minutes": args.simulation_minutes,
            "tasks_generation_rate": args.tasks_generation_rate,
        },
        "scenario_profile": str(args.scenario_profile),
        "task_source_mode": str(args.task_source_mode),
        "success_profile": str(args.success_profile),
        "action_mask_mode": str(args.action_mask_mode),
        "min_link_survival_margin_sec": float(max(0.0, args.min_link_survival_margin_sec)),
        "architecture": normalize_architecture(args.architecture),
        "n_leo": int(args.n_leo),
        "num_steps": int(coverage.get("num_decision_steps", 0)),
        "num_rows": len(rows),
        "trace_generation_mode": str(args.trace_mode),
        "dense_projection_mode": "source_projection" if args.trace_mode == "dense_projection" else "none",
        "candidate_cost_estimator_version": str(
            version.get("candidate_cost_estimator_version")
            or metrics.get("candidateCostEstimatorVersion")
            or (rows[0].get("candidate_cost_estimator_version") if rows else "unknown")
        ),
        "lower_action_binding_version": str(
            version.get("lower_action_binding_version")
            or metrics.get("lowerActionBindingVersion")
            or "unknown"
        ),
        "energy_counter_unit": str(metrics.get("energyCounterUnit", "Wh")),
        "energy_counter_semantics": str(
            metrics.get("energyCounterSemantics", "cumulative_total_across_all_datacenters")
        ),
        "queue_estimate_source": queue_source,
        "mobility_risk_source": mobility_source,
        "coverage": coverage,
    }


def _synthetic_row(step: int, leo_id: int, rng: random.Random) -> Dict[str, Any]:
    phase = 0.31 * step + 0.77 * leo_id
    neighbor_visible = math.sin(phase) > -0.20
    geo_visible = math.cos(0.41 * step - 0.53 * leo_id) > 0.18
    ground_visible = math.sin(0.27 * step + 0.63 * leo_id) > -0.05
    mask = [1, int(neighbor_visible), int(geo_visible), int(ground_visible)]
    completion_mask = [1, int(neighbor_visible and (step + leo_id) % 4 != 0), int(geo_visible and step % 3 != 0), int(ground_visible and leo_id % 3 != 1)]
    mobility_mask = [1, int(completion_mask[1] and step % 5 != 0), int(completion_mask[2] and (step + leo_id) % 5 != 2), int(completion_mask[3] and step % 4 != 1)]

    def remote_rate(base: float, visible: bool, scale: float) -> float:
        if not visible:
            return 0.0
        return round(base * (0.55 + 0.45 * max(0.0, math.sin(scale * phase))) + rng.random() * 0.5, 6)

    neighbor_count = rng.randint(1, 3) if neighbor_visible else 0
    geo_count = rng.randint(1, 2) if geo_visible else 0
    ground_count = rng.randint(1, 2) if ground_visible else 0
    return {
        "step": step,
        "leo_id": leo_id,
        "local_visible": True,
        "neighbor_visible": neighbor_visible,
        "geo_visible": geo_visible,
        "ground_visible": ground_visible,
        "local_rate": 24.0,
        "neighbor_rate": remote_rate(12.0, neighbor_visible, 1.0),
        "geo_rate": remote_rate(10.0, geo_visible, 0.7),
        "ground_rate": remote_rate(14.0, ground_visible, 1.3),
        "local_candidate_count": 1,
        "neighbor_candidate_count": neighbor_count,
        "geo_candidate_count": geo_count,
        "ground_candidate_count": ground_count,
        "neighbor_min_distance": round(350.0 + 800.0 * rng.random(), 6) if neighbor_visible else None,
        "geo_min_distance": round(35000.0 + 4000.0 * rng.random(), 6) if geo_visible else None,
        "ground_min_distance": round(700.0 + 1500.0 * rng.random(), 6) if ground_visible else None,
        "local_best_queue": rng.randint(0, 4),
        "neighbor_best_queue": rng.randint(0, 6) if neighbor_visible else 0,
        "geo_best_queue": rng.randint(0, 4) if geo_visible else 0,
        "ground_best_queue": rng.randint(0, 6) if ground_visible else 0,
        "local_best_delay": round(0.01 + 0.01 * rng.random(), 6),
        "neighbor_best_delay": round(0.01 + 0.04 * rng.random(), 6) if neighbor_visible else None,
        "geo_best_delay": round(0.16 + 0.14 * rng.random(), 6) if geo_visible else None,
        "ground_best_delay": round(0.04 + 0.08 * rng.random(), 6) if ground_visible else None,
        "abstract_action_mask": mask,
        "abstract_action_mask_visible": mask,
        "abstract_action_mask_completion_safe": completion_mask,
        "abstract_action_mask_mobility_safe": mobility_mask,
        "abstract_action_mask_final": mobility_mask,
        "mask_field_presence": {"visible": True, "completion_safe": True, "mobility_safe": True, "final": True},
        "phase_id": f"synthetic_{step % 3}",
        "action_mask_mode": "mobility_safe",
        "trace_origin": "synthetic",
        "synthetic": True,
        "trace_semantic_class": "synthetic_debug",
        "delay_semantic": "legacy_unknown",
        "trace_generation_mode": "synthetic",
        "dense_projection_mode": "none",
        "scenario_profile": "synthetic",
        "scenario_phase": "synthetic",
        "task_source_mode": "synthetic",
        "success_profile": "synthetic",
        "queue_estimate_source": "synthetic",
        "mobility_risk_source": "synthetic",
        "candidate_cost_estimator_version": "synthetic",
        "is_controlled_rl_scenario": False,
    }


def _export_synthetic(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    rows = [
        _synthetic_row(step, leo_id, rng)
        for step in range(args.max_decisions)
        for leo_id in range(max(1, args.n_leo))
    ]
    _write_jsonl(Path(args.output), rows)
    print(f"SATEDGESIM_TRACE_EXPORTED_SYNTHETIC output={args.output} rows={len(rows)}")
    return len(rows)


def _export_from_server(args: argparse.Namespace) -> int:
    client = SatEdgeSimClient(args.base_url, timeout=args.request_timeout)
    try:
        client.ensure_healthy()
        version_payload = client.version()
    except SatEdgeSimClientError as exc:
        raise SystemExit(
            "SatEdgeSim REST server is unavailable. "
            "Use --synthetic for a local smoke-trace, or start the Java REST server first. "
            f"Details: {exc}"
        ) from exc

    reset_extra: Dict[str, Any] = {}
    if args.simulation_minutes is not None:
        reset_extra["simulationTimeMinutes"] = float(args.simulation_minutes)
    if args.tasks_generation_rate is not None:
        reset_extra["tasksGenerationRate"] = int(args.tasks_generation_rate)
    elif args.max_decisions > 0:
        reset_extra["tasksGenerationRate"] = max(2, math.ceil(args.max_decisions / 120.0))
    reset_extra["scenarioProfile"] = str(args.scenario_profile)
    reset_extra["taskSourceMode"] = str(args.task_source_mode)
    reset_extra["successProfile"] = str(args.success_profile)
    reset_extra["actionMaskMode"] = str(args.action_mask_mode)
    reset_extra["minLinkSurvivalMarginSec"] = float(max(0.0, args.min_link_survival_margin_sec))
    reset_extra["maxDecisions"] = int(args.max_decisions)
    state = client.reset(
        devices_count=args.devices_count if args.devices_count is not None else args.n_leo,
        algorithm_index=args.algorithm_index,
        architecture_index=args.architecture_index,
        seed=args.seed,
        clean_output_folder=args.clean_output_folder,
        wait_for_first_decision=True,
        wait_timeout_ms=args.wait_timeout_ms,
        extra=reset_extra or None,
    )
    rows: List[Dict[str, Any]] = []
    dense_steps = 0
    sparse_steps = 0
    missing_pairs = 0
    dense_supported = False
    source_id_to_leo: Dict[int, int] = {}
    output_path = Path(args.output)
    coverage_path = output_path.with_suffix(output_path.suffix + ".coverage.json")
    manifest_path = _manifest_path_for_trace(output_path)
    metrics_payload: Dict[str, Any] = {}
    try:
        decision_step = 0
        previous_energy_counter_wh: float | None = None
        while decision_step < args.max_decisions:
            if state.get("status") in TERMINAL_STATUSES:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                state = _wait_for_decision(client, args.poll_sleep_sec)
                continue
            state = client.get_state()
            if state.get("status") != "WAITING_FOR_ACTION":
                continue
            dense_summaries = list(state.get("denseSourceSummaries") or [])
            if args.trace_mode == "sequential_live":
                raw_energy_counter_wh = _state_energy_counter_wh(state)
                summary = _current_dense_summary(state)
                if summary is not None:
                    leo_id = int(_to_float(summary.get("sourceDeviceId"), state.get("sourceDeviceId", 0.0)))
                    row = _snapshot_from_dense_summary(summary, step=decision_step, leo_id=leo_id)
                    row = _attach_energy_delta_fields(row, raw_wh=raw_energy_counter_wh, previous_wh=previous_energy_counter_wh)
                    rows.append(_annotate_contract_fields(row, args, trace_mode="sequential_live"))
                else:
                    row = _snapshot_from_state(
                            state,
                            step=decision_step,
                            n_leo=args.n_leo,
                            rate_scale_mbps=args.rate_scale_mbps,
                            max_trace_rate=args.max_trace_rate,
                            architecture=args.architecture,
                        )
                    row = _attach_energy_delta_fields(row, raw_wh=raw_energy_counter_wh, previous_wh=previous_energy_counter_wh)
                    rows.append(_annotate_contract_fields(row, args, trace_mode="sequential_live"))
                if raw_energy_counter_wh is not None:
                    previous_energy_counter_wh = raw_energy_counter_wh
                dense_steps += 1
            elif dense_summaries:
                raw_energy_counter_wh = _state_energy_counter_wh(state)
                dense_supported = True
                ordered = sorted(dense_summaries, key=lambda item: int(_to_float(item.get("sourceDeviceId"), -1.0)))
                if not source_id_to_leo:
                    unique_source_ids = [int(_to_float(item.get("sourceDeviceId"), -1.0)) for item in ordered]
                    unique_source_ids = [source_id for source_id in unique_source_ids if source_id >= 0]
                    unique_source_ids = sorted(dict.fromkeys(unique_source_ids))
                    if len(unique_source_ids) < args.n_leo:
                        missing_pairs += (args.n_leo - len(unique_source_ids))
                    source_id_to_leo = {source_id: leo_id for leo_id, source_id in enumerate(unique_source_ids[: args.n_leo])}
                step_seen = set()
                for summary in ordered:
                    source_device_id = int(_to_float(summary.get("sourceDeviceId"), -1.0))
                    if source_device_id not in source_id_to_leo:
                        continue
                    leo_id = source_id_to_leo[source_device_id]
                    step_seen.add(leo_id)
                    row = _snapshot_from_dense_summary(summary, step=decision_step, leo_id=leo_id)
                    row = _attach_energy_delta_fields(row, raw_wh=raw_energy_counter_wh, previous_wh=previous_energy_counter_wh)
                    rows.append(_annotate_contract_fields(row, args, trace_mode="dense_projection"))
                if raw_energy_counter_wh is not None:
                    previous_energy_counter_wh = raw_energy_counter_wh
                dense_steps += 1
                missing_pairs += max(0, args.n_leo - len(step_seen))
            else:
                raw_energy_counter_wh = _state_energy_counter_wh(state)
                sparse_steps += 1
                missing_pairs += 0 if args.trace_mode == "sequential_live" else max(0, args.n_leo - 1)
                row = _snapshot_from_state(
                        state,
                        step=decision_step,
                        n_leo=args.n_leo,
                        rate_scale_mbps=args.rate_scale_mbps,
                        max_trace_rate=args.max_trace_rate,
                        architecture=args.architecture,
                    )
                row = _attach_energy_delta_fields(row, raw_wh=raw_energy_counter_wh, previous_wh=previous_energy_counter_wh)
                rows.append(_annotate_contract_fields(row, args, trace_mode=args.trace_mode))
                if raw_energy_counter_wh is not None:
                    previous_energy_counter_wh = raw_energy_counter_wh
            decision_step += 1
            state = client.step(_advance_action(state), wait_timeout_ms=args.wait_timeout_ms)
        try:
            metrics_payload = client.get_metrics()
        except SatEdgeSimClientError:
            metrics_payload = {}
    finally:
        try:
            client.close()
        except Exception:
            pass
    coverage = {
        "trace": str(output_path),
        "status": _coverage_status(
            trace_mode=args.trace_mode,
            dense_supported=dense_supported,
            sparse_steps=sparse_steps,
            missing_pairs=missing_pairs,
            num_rows=len(rows),
        ),
        "dense_supported": dense_supported,
        "dense_steps": dense_steps,
        "sparse_steps": sparse_steps,
        "num_rows": len(rows),
        "num_decision_steps": dense_steps + sparse_steps,
        "expected_dense_rows": args.n_leo * max(0, dense_steps + sparse_steps),
        "dense_coverage_ratio": len(rows) / max(1, args.n_leo * max(0, dense_steps + sparse_steps)),
        "missing_pairs": missing_pairs,
        "n_leo_requested": args.n_leo,
        "devices_count_used": args.devices_count if args.devices_count is not None else args.n_leo,
        "source_id_to_leo": source_id_to_leo,
        "scenario_profile": args.scenario_profile,
        "task_source_mode": args.task_source_mode,
        "success_profile": args.success_profile,
        "architecture": normalize_architecture(args.architecture),
        "action_mask_mode": args.action_mask_mode,
        "min_link_survival_margin_sec": float(max(0.0, args.min_link_survival_margin_sec)),
        "trace_generation_mode": args.trace_mode,
        "dense_projection_mode": "source_projection" if args.trace_mode == "dense_projection" else "none",
    }
    _write_jsonl(output_path, rows)
    _write_json(coverage_path, coverage)
    manifest = _manifest_from_export(
        trace_path=output_path,
        rows=rows,
        args=args,
        version=version_payload,
        metrics=metrics_payload,
        coverage=coverage,
    )
    _write_json(manifest_path, manifest)
    print(
        "SATEDGESIM_TRACE_EXPORTED "
        f"output={args.output} rows={len(rows)} dense_supported={dense_supported} "
        f"dense_coverage_ratio={coverage['dense_coverage_ratio']:.6f}"
    )
    print(f"coverage_report={coverage_path}")
    print(f"manifest={manifest_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SatEdgeSim topology traces for TriSatFlow.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--devices-count", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm-index", type=int, default=0)
    parser.add_argument("--architecture-index", type=int, default=0)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--clean-output-folder", action="store_true")
    parser.add_argument("--simulation-minutes", type=float, default=None)
    parser.add_argument("--tasks-generation-rate", type=int, default=None)
    parser.add_argument("--rate-scale-mbps", type=float, default=1000.0)
    parser.add_argument("--max-trace-rate", type=float, default=24.0)
    parser.add_argument("--scenario-profile", type=str, default="default")
    parser.add_argument("--task-source-mode", type=str, default="current")
    parser.add_argument("--success-profile", type=str, default="default", choices=["default", "preflight_lenient", "paper_strict"])
    parser.add_argument("--action-mask-mode", type=str, default="visible_only", choices=["visible_only", "mobility_safe", "completion_safe"])
    parser.add_argument("--architecture", type=str, default="full", choices=["only_leo", "leo_geo", "leo_ground", "full"])
    parser.add_argument("--min-link-survival-margin-sec", type=float, default=0.0)
    parser.add_argument("--trace-mode", type=str, default="dense_projection", choices=["dense_projection", "sequential_live"])
    parser.add_argument("--trace-semantic-class", type=str, default="actual_physical")
    parser.add_argument("--synthetic", action="store_true", help="Generate a clearly marked synthetic trace without contacting SatEdgeSim.")
    args = parser.parse_args()

    if args.synthetic:
        _export_synthetic(args)
        return
    _export_from_server(args)


if __name__ == "__main__":
    main()
