from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from trisatflow.envs.obs_builder import build_shared_observation, dense_rows_from_state, ring_edge_index

NODE_FEATURE_DIM = 12
EDGE_FEATURE_DIM = 4

ACTION_LOCAL = 0
ACTION_NEIGHBOR = 1
ACTION_GEO = 2
ACTION_GROUND = 3


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_max(values: Iterable[float], default: float = 1.0) -> float:
    values = [float(v) for v in values if v is not None]
    if not values:
        return default
    return max(max(values), default)


def _candidate_is_feasible(vm: Dict[str, Any], action_mask: Sequence[Any], index: int) -> bool:
    if "isFeasible" in vm:
        return bool(vm.get("isFeasible"))
    if index < len(action_mask):
        return bool(action_mask[index])
    if "feasible" in vm:
        return bool(vm.get("feasible"))
    return True


def _is_local_candidate(vm: Dict[str, Any], task: Dict[str, Any]) -> bool:
    if "isLocalToSource" in vm:
        return bool(vm.get("isLocalToSource"))
    source_dc = task.get("sourceDatacenterId")
    return source_dc is not None and str(vm.get("datacenterId")) == str(source_dc)


def _abstract_action(vm: Dict[str, Any], task: Dict[str, Any]) -> int:
    try:
        action = int(vm.get("abstractAction"))
        if 0 <= action <= 3:
            return action
    except (TypeError, ValueError):
        pass
    logical = str(vm.get("logicalTier") or vm.get("level") or "").strip().lower()
    if logical == "local":
        return ACTION_LOCAL
    if logical == "neighbor":
        return ACTION_NEIGHBOR
    if logical in {"geo", "cloud"}:
        return ACTION_GEO
    if logical in {"ground", "edge"}:
        return ACTION_GROUND
    dc_type = str(vm.get("datacenterType", "")).lower()
    if "cloud" in dc_type or "geo" in dc_type:
        return ACTION_GEO
    if "edge_datacenter" in dc_type or "ground" in dc_type or ("edge" in dc_type and "device" not in dc_type):
        return ACTION_GROUND
    if "edge_device" in dc_type or "mist" in dc_type or "device" in dc_type:
        return ACTION_LOCAL if _is_local_candidate(vm, task) else ACTION_NEIGHBOR
    return -1


def _abstract_action_mask(state: Dict[str, Any], candidate_vms: List[Dict[str, Any]], action_mask: Sequence[Any], task: Dict[str, Any]) -> List[int]:
    explicit = state.get("abstractActionMask")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) and len(explicit) >= 4:
        return [1 if bool(explicit[i]) else 0 for i in range(4)]
    mask = [0, 0, 0, 0]
    for i, vm in enumerate(candidate_vms):
        if not _candidate_is_feasible(vm, action_mask, i):
            continue
        action = _abstract_action(vm, task)
        if 0 <= action <= 3:
            mask[action] = 1
    return mask


def source_leo_id_from_state(state: Dict[str, Any], *, fallback_n: int | None = None) -> int:
    if state.get("sourceLeoId") is not None:
        source = _to_int(state.get("sourceLeoId"), 0)
        if fallback_n is not None and fallback_n > 0:
            return source % fallback_n
        return source
    task = dict(state.get("task") or {})
    if task.get("sourceDeviceId") is not None:
        source = _to_int(task.get("sourceDeviceId"), 0)
        if fallback_n is not None and fallback_n > 0:
            return source % fallback_n
        return source
    if task.get("sourceDatacenterId") is not None:
        source = _to_int(task.get("sourceDatacenterId"), 0)
        if fallback_n is not None and fallback_n > 0:
            return source % fallback_n
        return source
    source = _to_int(state.get("decisionId", state.get("requestId")), 0)
    if fallback_n is not None and fallback_n > 0:
        return source % fallback_n
    return source


def scenario_profile_from_state(state: Dict[str, Any]) -> str:
    task = dict(state.get("task") or {})
    return str(task.get("scenarioProfile") or state.get("scenarioProfile") or "default")


def task_source_mode_from_state(state: Dict[str, Any]) -> str:
    task = dict(state.get("task") or {})
    return str(task.get("taskSourceMode") or state.get("taskSourceMode") or "current")


def is_controlled_rl_scenario_from_state(state: Dict[str, Any]) -> bool:
    task = dict(state.get("task") or {})
    return bool(task.get("isControlledRlScenario", state.get("isControlledRlScenario", False)))


def _rate_norm(vm: Dict[str, Any], max_bw: float) -> float:
    rate = _to_float(vm.get("estimatedTransmissionRateMbps"), 0.0)
    if rate <= 0.0:
        rate = _to_float(vm.get("bw"), 0.0)
    return max(0.0, min(rate / max(max_bw, 1e-6), 1.0))


def build_trisatflow_observation(
    state: Dict[str, Any],
    *,
    node_feature_dim: int = NODE_FEATURE_DIM,
    normalization_mode: str = "legacy",
    normalization_stats: Mapping[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Convert SatEdgeSim REST state into shared per-source observations."""

    rows = dense_rows_from_state(state)
    if not rows:
        rows = [_fallback_dense_row_from_candidate_state(state)]
    source_index = _infer_shared_source_index(state, rows)
    batch = build_shared_observation(
        rows,
        source_index=source_index,
        node_feature_dim=node_feature_dim,
        normalization_mode=normalization_mode,
        normalization_stats=normalization_stats,
    )
    edge_index, edge_attr = ring_edge_index(batch.obs.shape[0])
    return batch.obs, edge_index, edge_attr, batch.source_index


def _infer_source_index(state: Dict[str, Any], n: int) -> int:
    if n <= 1:
        return 0
    task = dict(state.get("task") or {})
    source_dc = task.get("sourceDatacenterId")
    candidate_vms: List[Dict[str, Any]] = list(state.get("candidateVms") or [])
    if source_dc is not None:
        for i, vm in enumerate(candidate_vms):
            if str(vm.get("datacenterId")) == str(source_dc):
                return i
    return source_leo_id_from_state(state, fallback_n=n)


def _infer_shared_source_index(state: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    source_leo = source_leo_id_from_state(state, fallback_n=max(1, len(rows)))
    for idx, row in enumerate(rows):
        leo_id = int(_to_float(row.get("leo_id", row.get("sourceDeviceId", idx)), idx))
        if leo_id == source_leo:
            return idx
    return max(0, min(source_leo, len(rows) - 1))


def _fallback_dense_row_from_candidate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    candidate_vms: List[Dict[str, Any]] = list(state.get("candidateVms") or [])
    action_mask: List[Any] = list(state.get("actionMask") or [])
    task: Dict[str, Any] = dict(state.get("task") or {})

    raw = {
        "leo_id": source_leo_id_from_state(state, fallback_n=1),
        "local_visible": 1,
        "neighbor_visible": 0,
        "geo_visible": 0,
        "ground_visible": 0,
        "local_rate": 1000.0,
        "neighbor_rate": 0.0,
        "geo_rate": 0.0,
        "ground_rate": 0.0,
        "local_delay": 0.02,
        "neighbor_delay": 0.0,
        "geo_delay": 0.0,
        "ground_delay": 0.0,
        "local_queue": 0.0,
        "neighbor_queue": 0.0,
        "geo_queue": 0.0,
        "ground_queue": 0.0,
    }
    for idx, vm in enumerate(candidate_vms):
        if not _candidate_is_feasible(vm, action_mask, idx):
            continue
        action = _abstract_action(vm, task)
        queue = _to_float(vm.get("estimatedQueueLength", vm.get("assignedTasks", 0.0)), 0.0)
        delay = _to_float(
            vm.get("estimatedTotalDelaySec"),
            _to_float(vm.get("estimatedTransmissionDelaySec"), 0.0) + _to_float(vm.get("estimatedComputeDelaySec"), 0.0),
        )
        rate = _to_float(vm.get("estimatedTransmissionRateMbps"), 0.0) or _to_float(vm.get("bw"), 0.0)
        if action == ACTION_LOCAL:
            raw["local_queue"] = queue
            raw["local_delay"] = delay
            raw["local_visible"] = 1
        elif action == ACTION_NEIGHBOR:
            raw["neighbor_visible"] = 1
            raw["neighbor_rate"] = max(raw["neighbor_rate"], rate)
            raw["neighbor_queue"] = queue if raw["neighbor_queue"] == 0.0 else min(raw["neighbor_queue"], queue)
            raw["neighbor_delay"] = delay if raw["neighbor_delay"] == 0.0 else min(raw["neighbor_delay"], delay)
        elif action == ACTION_GEO:
            raw["geo_visible"] = 1
            raw["geo_rate"] = max(raw["geo_rate"], rate)
            raw["geo_queue"] = queue if raw["geo_queue"] == 0.0 else min(raw["geo_queue"], queue)
            raw["geo_delay"] = delay if raw["geo_delay"] == 0.0 else min(raw["geo_delay"], delay)
        elif action == ACTION_GROUND:
            raw["ground_visible"] = 1
            raw["ground_rate"] = max(raw["ground_rate"], rate)
            raw["ground_queue"] = queue if raw["ground_queue"] == 0.0 else min(raw["ground_queue"], queue)
            raw["ground_delay"] = delay if raw["ground_delay"] == 0.0 else min(raw["ground_delay"], delay)
    return raw


def _build_candidate_graph(
    candidate_vms: List[Dict[str, Any]],
    action_mask: Sequence[Any],
    *,
    max_bw: float,
    max_dist: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(candidate_vms)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    src: List[int] = []
    dst: List[int] = []
    attrs: List[List[float]] = []

    def add_edge(i: int, j: int, edge_type: float) -> None:
        vm_j = candidate_vms[j]
        feasible = 1.0 if _candidate_is_feasible(vm_j, action_mask, j) else 0.0
        rate = (_to_float(vm_j.get("estimatedTransmissionRateMbps"), 0.0) or _to_float(vm_j.get("bw"), 0.0)) / max(max_bw, 1e-6)
        delay_scale = _to_float(vm_j.get("sourceDistance", vm_j.get("distanceToSource", 0.0)), 0.0) / max(max_dist, 1e-6)
        src.append(i)
        dst.append(j)
        attrs.append(
            [
                max(0.0, min(rate, 1.0)),
                max(0.0, min(delay_scale, 1.0)),
                feasible,
                edge_type,
            ]
        )

    for i in range(n):
        add_edge(i, (i - 1) % n, 0.0)
        add_edge(i, (i + 1) % n, 0.0)
        if n > 3:
            add_edge(i, (i + 2) % n, 1.0)

    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(attrs, dtype=torch.float32)
