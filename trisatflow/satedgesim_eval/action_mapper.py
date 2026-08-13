from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

ACTION_LOCAL = 0
ACTION_NEIGHBOR = 1
ACTION_GEO = 2
ACTION_GROUND = 3

ACTION_TO_LEVEL = {
    ACTION_LOCAL: "LOCAL",
    ACTION_NEIGHBOR: "NEIGHBOR",
    ACTION_GEO: "GEO",
    ACTION_GROUND: "GROUND",
}

_LEVEL_ALIASES = {
    "local": "LOCAL",
    "neighbor": "NEIGHBOR",
    "mistremote": "NEIGHBOR",
    "leo_remote": "NEIGHBOR",
    "leo-neighbor": "NEIGHBOR",
    "geo": "GEO",
    "cloud": "GEO",
    "ground": "GROUND",
    "edge": "GROUND",
    "edge_datacenter": "GROUND",
    "edge device": "LEO",
    "edge_device": "LEO",
    "mist": "LEO",
    "leo": "LEO",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _candidate_is_feasible(vm: Dict[str, Any], mask: Sequence[Any], idx: int, mode: str = "visible_only") -> bool:
    if "isFeasible" in vm:
        base = bool(vm.get("isFeasible"))
    elif idx < len(mask):
        base = bool(mask[idx])
    elif "feasible" in vm:
        base = bool(vm.get("feasible"))
    else:
        base = True
    if not base:
        return False
    mode = str(mode or "visible_only").strip().lower()
    if mode in {"mobility_safe", "mobility_risk"}:
        return bool(vm.get("mobilitySafe", False))
    if mode in {"completion_safe", "full", "full_mask"}:
        return bool(vm.get("completionSafe", False))
    return True


def _candidate_index(vm: Dict[str, Any]) -> int:
    """SatEdgeSim RlAction.targetVmIndex expects the VM-list/candidate index."""

    return int(vm.get("vmIndex", vm.get("_candidateIndex", -1)))


def _dc_type(vm: Dict[str, Any]) -> str:
    return str(vm.get("datacenterType") or vm.get("type") or "").lower()


def _canonical_level(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    key = text.lower().replace(" ", "_")
    return _LEVEL_ALIASES.get(key, text.upper())


def _is_local(vm: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if "isLocalToSource" in vm:
        return _to_bool(vm.get("isLocalToSource"))
    task = dict(state.get("task") or {})
    source_dc = task.get("sourceDatacenterId")
    return source_dc is not None and str(vm.get("datacenterId")) == str(source_dc)


def _abstract_action(vm: Dict[str, Any], state: Dict[str, Any]) -> int:
    explicit = vm.get("abstractAction")
    try:
        action = int(explicit)
        if 0 <= action <= 3:
            return action
    except (TypeError, ValueError):
        pass

    level = _level(vm, state)
    if level == "LOCAL":
        return ACTION_LOCAL
    if level == "NEIGHBOR":
        return ACTION_NEIGHBOR
    if level == "GEO":
        return ACTION_GEO
    if level == "GROUND":
        return ACTION_GROUND

    dc_type = _dc_type(vm)
    if "cloud" in dc_type or "geo" in dc_type:
        return ACTION_GEO
    if "edge_datacenter" in dc_type or "ground" in dc_type or ("edge" in dc_type and "device" not in dc_type):
        return ACTION_GROUND
    if "edge_device" in dc_type or "mist" in dc_type or "device" in dc_type:
        return ACTION_LOCAL if _is_local(vm, state) else ACTION_NEIGHBOR
    return -1


def _level(vm: Dict[str, Any], state: Dict[str, Any] | None = None) -> str:
    explicit = _canonical_level(vm.get("logicalTier") or vm.get("level"))
    if explicit in {"LOCAL", "NEIGHBOR", "GEO", "GROUND"}:
        return explicit
    dc_type = _dc_type(vm)
    if "cloud" in dc_type or "geo" in dc_type:
        return "GEO"
    if "edge_datacenter" in dc_type or "ground" in dc_type or ("edge" in dc_type and "device" not in dc_type):
        return "GROUND"
    if "edge_device" in dc_type or "mist" in dc_type or "device" in dc_type:
        if state is not None and _is_local(vm, state):
            return "LOCAL"
        return "NEIGHBOR"
    return "UNKNOWN"


def _feasible_candidates(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    vms = list(state.get("candidateVms") or [])
    mask = list(state.get("actionMask") or [])
    mode = str(state.get("actionMaskMode") or "visible_only")
    out = []
    for idx, vm in enumerate(vms):
        item = dict(vm)
        item["_candidateIndex"] = idx
        if _candidate_is_feasible(item, mask, idx, mode=mode):
            item["_abstractAction"] = _abstract_action(item, state)
            item["_level"] = _level(item, state)
            item["_isLocal"] = _is_local(item, state)
            item["_requiresNetworkTransfer"] = not item["_isLocal"]
            out.append(item)
    return out


def abstract_action_mask_from_state(state: Dict[str, Any]) -> List[int]:
    """Return [local, neighbor, geo, ground] feasibility from SatEdgeSim state."""

    mode = str(state.get("actionMaskMode") or "visible_only").strip().lower()
    explicit = None
    if mode in {"mobility_safe", "mobility_risk"}:
        explicit = state.get("abstractActionMaskMobilitySafe")
    elif mode in {"completion_safe", "full", "full_mask"}:
        explicit = state.get("abstractActionMaskCompletionSafe")
    elif mode in {"none", "no_mask"}:
        explicit = [1, 1, 1, 1]
    if explicit is None:
        explicit = state.get("abstractActionMask")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) and len(explicit) >= 4:
        return [1 if bool(explicit[i]) else 0 for i in range(4)]

    mask = [0, 0, 0, 0]
    for vm in _feasible_candidates(state):
        action = int(vm.get("_abstractAction", -1))
        if 0 <= action <= 3:
            mask[action] = 1
    return mask


def _queue(vm: Dict[str, Any]) -> float:
    return _to_float(vm.get("estimatedQueueLength", vm.get("assignedTasks", 0.0)), 0.0)


def _distance(vm: Dict[str, Any]) -> float:
    return _to_float(vm.get("sourceDistance", vm.get("distanceToSource", 0.0)), 0.0)


def _capacity(vm: Dict[str, Any]) -> float:
    if vm.get("estimatedComputeCapacity") is not None:
        return _to_float(vm.get("estimatedComputeCapacity"), 0.0)
    return _to_float(vm.get("mips"), 0.0) * max(_to_float(vm.get("pesNumber"), 1.0), 1.0)


def _best(pool: List[Dict[str, Any]], *, prefer_capacity: bool = True) -> Dict[str, Any]:
    def total_delay(vm: Dict[str, Any]) -> float:
        tx = _to_float(vm.get("estimatedTransmissionDelaySec"), 0.0)
        compute = _to_float(vm.get("estimatedComputeDelaySec"), 0.0)
        if tx > 0.0 or compute > 0.0:
            return tx + compute
        return _queue(vm) + _distance(vm) / 1000.0

    if prefer_capacity:
        return min(pool, key=lambda vm: (total_delay(vm), _queue(vm), _distance(vm), -_capacity(vm)))
    return min(pool, key=lambda vm: (total_delay(vm), _distance(vm), _queue(vm), -_capacity(vm)))


def _trace(upper_action: int, desired_level: str, selected: Dict[str, Any] | None, fallback_reason: str) -> Dict[str, Any]:
    if selected is None:
        return {
            "desired_level": desired_level,
            "desired_abstract_action_name": ACTION_TO_LEVEL.get(int(upper_action), "UNKNOWN"),
            "fallback_reason": fallback_reason or "no_feasible_candidates",
            "selected_level": None,
            "selected_logical_tier": None,
            "selected_abstract_action": None,
            "selected_abstract_action_name": None,
            "selected_is_local": None,
            "selected_is_remote": None,
            "selected_datacenter_id": None,
            "selected_vm_id": None,
            "selected_candidate_index": -1,
            "selected_vm_index": -1,
            "selected_queue": None,
            "selected_distance": None,
            "selected_capacity": None,
            "selected_rate_mbps": None,
            "selected_prop_delay_sec": None,
            "mapped_upper_action": int(upper_action),
        }
    return {
        "desired_level": desired_level,
        "desired_abstract_action_name": ACTION_TO_LEVEL.get(int(upper_action), "UNKNOWN"),
        "fallback_reason": fallback_reason or "none",
        "selected_level": selected.get("_level") or _level(selected),
        "selected_logical_tier": selected.get("_level") or _level(selected),
        "selected_abstract_action": int(selected.get("_abstractAction", _abstract_action(selected, {}))),
        "selected_abstract_action_name": ACTION_TO_LEVEL.get(int(selected.get("_abstractAction", _abstract_action(selected, {}))), "UNKNOWN"),
        "selected_is_local": bool(selected.get("_isLocal")),
        "selected_is_remote": not bool(selected.get("_isLocal")),
        "selected_datacenter_id": selected.get("datacenterId"),
        "selected_vm_id": selected.get("vmId"),
        "selected_candidate_index": int(selected.get("_candidateIndex", -1)),
        "selected_vm_index": _candidate_index(selected),
        "selected_queue": _queue(selected),
        "selected_distance": _distance(selected),
        "selected_capacity": _capacity(selected),
        "selected_rate_mbps": _to_float(selected.get("estimatedTransmissionRateMbps", selected.get("bw", 0.0)), 0.0),
        "selected_prop_delay_sec": _to_float(selected.get("propagationDelaySec"), 0.0),
        "mapped_upper_action": int(upper_action),
    }


def _pool_for_action(candidates: List[Dict[str, Any]], action: int) -> List[Dict[str, Any]]:
    return [vm for vm in candidates if int(vm.get("_abstractAction", -1)) == int(action)]


def map_upper_to_target_vm_with_trace(
    state: Dict[str, Any],
    upper_action: int,
    *,
    require_visible: bool = False,
    missing_reason: str = "requested_action_not_visible",
) -> Tuple[int, Dict[str, Any]]:
    """Map TriSatFlow's abstract action to a SatEdgeSim VM-list index.

    The preferred path uses SatEdgeSim's explicit abstractAction field.  Legacy
    SatEdgeSim builds are still supported by inferring the four tiers from
    datacenterType and sourceDatacenterId.
    """

    candidates = _feasible_candidates(state)
    desired_level = ACTION_TO_LEVEL.get(int(upper_action), "UNKNOWN")
    if not candidates:
        return -1, _trace(upper_action, desired_level, None, "no_feasible_candidates")
    if require_visible:
        mask = abstract_action_mask_from_state(state)
        if not (0 <= int(upper_action) < len(mask)) or not bool(mask[int(upper_action)]):
            return -1, _trace(upper_action, desired_level, None, missing_reason)

    selected: Dict[str, Any] | None = None
    fallback_reason = ""
    pool = _pool_for_action(candidates, int(upper_action))

    if not pool:
        if upper_action == ACTION_LOCAL:
            pool = [vm for vm in candidates if vm.get("_isLocal")]
        elif upper_action == ACTION_NEIGHBOR:
            pool = [vm for vm in candidates if not vm.get("_isLocal") and vm.get("_level") == "NEIGHBOR"]
        elif upper_action == ACTION_GEO:
            pool = [vm for vm in candidates if vm.get("_level") == "GEO"]
        elif upper_action == ACTION_GROUND:
            pool = [vm for vm in candidates if vm.get("_level") == "GROUND"]
        if pool:
            fallback_reason = "legacy_tier_inference_used"
        else:
            return -1, _trace(upper_action, desired_level, None, "no_feasible_candidate_for_requested_tier")

    selected = _best(pool, prefer_capacity=(upper_action != ACTION_NEIGHBOR))
    selected_index = _candidate_index(selected)
    if selected_index < 0:
        return -1, _trace(upper_action, desired_level, None, "selected_candidate_missing_vm_index")
    return selected_index, _trace(upper_action, desired_level, selected, fallback_reason)


def map_upper_to_target_vm(state: Dict[str, Any], upper_action: int) -> int:
    target, _ = map_upper_to_target_vm_with_trace(state, upper_action)
    return target
