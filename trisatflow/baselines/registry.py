from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import warnings
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence

ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
FORMAL_BASELINE_NAMES = [
    "local_only",
    "neighbor_only",
    "geo_only",
    "ground_only",
    "random_visible",
    "min_delay_greedy",
    "min_energy_greedy",
    "queue_aware_greedy",
    "mobility_risk_greedy",
    "lyapunov_dpp_greedy",
    "optimized_lyapunov_dpp",
]
PAPER_READY_LEARNING_BASELINE_NAMES = [
    "flat_ppo",
    "flat_mappo",
    "hierarchical_no_gnn",
]
LEGACY_BASELINE_ALIASES = {
    "random": "random_visible",
    "greedy_delay": "min_delay_greedy",
    "greedy_energy": "min_energy_greedy",
    "greedy_queue": "queue_aware_greedy",
}

ARCHITECTURE_ALLOWED_ACTIONS: Dict[str, List[int]] = {
    "only_leo": [0, 1],
    "leo_geo": [0, 1, 2],
    "leo_ground": [0, 1, 3],
    "full": [0, 1, 2, 3],
}

BASELINE_TYPES = {"static", "heuristic", "optimization", "rl", "placeholder"}


class BaselinePolicy(Protocol):
    name: str

    def select_action(
        self,
        obs: Any,
        state: Mapping[str, Any],
        mask: Sequence[int],
        candidate_info: Mapping[int, Mapping[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class BaselineContext:
    baseline_name: str
    architecture: str = "full"
    action_mask_mode: str = "visible_only"
    fallback_policy: str = "cost_greedy"


@dataclass(frozen=True)
class BaselineMetadata:
    name: str
    type: str
    uses_oracle: bool
    uses_privileged_info: bool
    trainable: bool
    paper_ready: bool
    implemented: bool = True
    requires_checkpoint: bool = False
    checkpoint_loaded: bool = False
    is_placeholder: bool = False
    allows_formal_eval: bool = True
    fallback_policy: str | None = None
    update_implemented: bool = True

    @property
    def baseline_name(self) -> str:
        return self.name

    def with_checkpoint_loaded(self, loaded: bool) -> "BaselineMetadata":
        return BaselineMetadata(
            name=self.name,
            type=self.type,
            uses_oracle=self.uses_oracle,
            uses_privileged_info=self.uses_privileged_info,
            trainable=self.trainable,
            paper_ready=self.paper_ready,
            implemented=self.implemented,
            requires_checkpoint=self.requires_checkpoint,
            checkpoint_loaded=bool(loaded),
            is_placeholder=self.is_placeholder,
            allows_formal_eval=(
                self.implemented
                and self.paper_ready
                and not self.is_placeholder
                and (not self.requires_checkpoint or bool(loaded))
            ),
            fallback_policy=self.fallback_policy,
            update_implemented=self.update_implemented,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["baseline_name"] = self.name
        return payload


def normalize_architecture(value: str | None) -> str:
    text = str(value or "full").strip().lower()
    if text not in ARCHITECTURE_ALLOWED_ACTIONS:
        raise ValueError(f"unsupported architecture={value!r}; choose from {sorted(ARCHITECTURE_ALLOWED_ACTIONS)}")
    return text


def allowed_actions_for_architecture(architecture: str) -> List[int]:
    return list(ARCHITECTURE_ALLOWED_ACTIONS[normalize_architecture(architecture)])


def action_mask_for_architecture(architecture: str) -> List[int]:
    allowed = set(allowed_actions_for_architecture(architecture))
    return [1 if idx in allowed else 0 for idx in range(4)]


def apply_architecture_filter(mask: Sequence[int], architecture: str) -> List[int]:
    arch_mask = action_mask_for_architecture(architecture)
    out: List[int] = []
    for i in range(4):
        bit = 1 if (i < len(mask) and bool(mask[i]) and bool(arch_mask[i])) else 0
        out.append(bit)
    return out


def state_action_mask(state: Mapping[str, Any], action_mask_mode: str = "visible_only") -> List[int]:
    mode = str(action_mask_mode or state.get("actionMaskMode") or "visible_only").strip().lower()
    if mode in {"no_mask", "none"}:
        return [1, 1, 1, 1]
    if mode in {"visibility", "visibility_only"}:
        mode = "visible_only"
    elif mode in {"mobility_risk"}:
        mode = "mobility_safe"
    elif mode in {"full", "full_mask"}:
        mode = "completion_safe"

    def _mask4(raw: Any) -> List[int]:
        if isinstance(raw, list) and len(raw) >= 4:
            return [1 if bool(raw[i]) else 0 for i in range(4)]
        return [0, 0, 0, 0]

    visible = _mask4(state.get("abstractActionMaskVisible") or state.get("abstractActionMask"))
    mobility = _mask4(state.get("abstractActionMaskMobilitySafe"))
    completion = _mask4(state.get("abstractActionMaskCompletionSafe"))
    if mode == "mobility_safe":
        return mobility if any(mobility) else visible
    if mode == "completion_safe":
        return completion if any(completion) else visible
    return visible


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_cost(vm: Mapping[str, Any]) -> float:
    delay = max(0.0, _to_float(vm.get("estimatedTotalDelaySec", vm.get("estimatedComputeDelaySec", 0.0))))
    queue = max(0.0, _to_float(vm.get("estimatedQueueLength", 0.0)))
    risk = max(0.0, min(1.0, _to_float(vm.get("mobilityRisk", 0.0))))
    return float(delay + 0.5 * queue + 0.2 * risk)


def extract_candidate_info(state: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for vm in list(state.get("candidateVms") or []):
        if not bool(vm.get("isFeasible", False)):
            continue
        action = int(_to_float(vm.get("abstractAction"), -1))
        if action < 0 or action > 3:
            continue
        cost = _best_cost(vm)
        risk = max(0.0, min(1.0, _to_float(vm.get("mobilityRisk", 1.0))))
        item = out.get(action)
        if item is None or cost < float(item["estimated_cost"]):
            out[action] = {
                "action": action,
                "tier": ACTION_NAMES[action],
                "estimated_cost": cost,
                "mobility_risk": risk,
                "estimated_delay": _to_float(vm.get("estimatedTotalDelaySec", 0.0)),
                "estimated_queue": _to_float(vm.get("estimatedQueueLength", 0.0)),
                "estimated_energy_j": _to_float(
                    vm.get("estimatedEnergyJ", vm.get("estimatedTxEnergyJ", vm.get("estimatedComputeEnergyJ", 0.0)))
                ),
                "selected_vm_id": int(_to_float(vm.get("vmId"), -1)),
                "selected_candidate_index": int(_to_float(vm.get("vmIndex", vm.get("_candidateIndex", -1)), -1)),
                "raw": dict(vm),
            }
    return out


def _cost_rank(candidate_info: Mapping[int, Mapping[str, Any]], action: int) -> float:
    rows = sorted(
        [(a, _to_float(info.get("estimated_cost"), float("inf"))) for a, info in candidate_info.items()],
        key=lambda x: x[1],
    )
    for idx, (a, _) in enumerate(rows, start=1):
        if int(a) == int(action):
            return float(idx)
    return float("inf")


def _fallback_action(
    fallback_policy: str,
    mask: Sequence[int],
    candidate_info: Mapping[int, Mapping[str, Any]],
    rng: random.Random,
) -> int:
    visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
    if not visible:
        return 0
    policy = str(fallback_policy or "cost_greedy").strip().lower()
    if policy == "random_visible":
        return int(rng.choice(visible))
    if policy == "local":
        return 0 if 0 in visible else int(visible[0])
    # default: min delay/cost greedy
    ranked = sorted(
        visible,
        key=lambda a: (
            _to_float(candidate_info.get(a, {}).get("estimated_cost"), float("inf")),
            _to_float(candidate_info.get(a, {}).get("mobility_risk"), 1.0),
            a,
        ),
    )
    return int(ranked[0])


def finalize_baseline_decision(
    *,
    baseline_name: str,
    requested_action: int,
    mask: Sequence[int],
    candidate_info: Mapping[int, Mapping[str, Any]],
    rng: random.Random,
    fallback_policy: str = "cost_greedy",
    selection_reason: str = "baseline_rule",
    extra_info: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    fallback_used = False
    fallback_reason = "none"
    selected_action = int(requested_action)
    if not (0 <= selected_action <= 3 and selected_action < len(mask) and bool(mask[selected_action])):
        fallback_used = True
        fallback_reason = "target_action_not_available"
        selected_action = _fallback_action(fallback_policy, mask, candidate_info, rng)
    info = dict(candidate_info.get(selected_action) or {})
    selected_tier = ACTION_NAMES[selected_action]
    decision_info: Dict[str, Any] = {
        "baseline_name": baseline_name,
        "requested_action": int(requested_action),
        "selected_action": selected_action,
        "selected_tier": selected_tier,
        "selection_reason": selection_reason,
        "target_tier": ACTION_NAMES[int(max(0, min(3, requested_action)))],
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "cost_rank": _cost_rank(candidate_info, selected_action),
        "mobility_risk": _to_float(info.get("mobility_risk"), 1.0),
        "estimated_cost": _to_float(info.get("estimated_cost"), float("inf")),
        "estimated_delay_s": _to_float(info.get("estimated_delay"), 0.0),
        "estimated_queue": _to_float(info.get("estimated_queue"), 0.0),
        "estimated_energy_j": _to_float(info.get("estimated_energy_j"), 0.0),
        "selected_vm_id": int(_to_float(info.get("selected_vm_id"), -1)),
        "selected_candidate_index": int(_to_float(info.get("selected_candidate_index"), -1)),
    }
    if extra_info:
        decision_info.update(dict(extra_info))
    return {
        "upper_action": int(selected_action),
        "lower_action": [1.0, 1.0, 1.0],
        "action_name": ACTION_NAMES[selected_action],
        "decision_info": decision_info,
    }


def _meta(
    name: str,
    type_: str,
    uses_oracle: bool = False,
    uses_privileged_info: bool = False,
    trainable: bool = False,
    paper_ready: bool = True,
    *,
    implemented: bool = True,
    requires_checkpoint: bool = False,
    checkpoint_loaded: bool = False,
    is_placeholder: bool = False,
    allows_formal_eval: bool | None = None,
    fallback_policy: str | None = None,
    update_implemented: bool = True,
) -> BaselineMetadata:
    if allows_formal_eval is None:
        allows_formal_eval = (
            implemented
            and paper_ready
            and not is_placeholder
            and (not requires_checkpoint or checkpoint_loaded)
        )
    return BaselineMetadata(
        name=name,
        type=type_,
        uses_oracle=uses_oracle,
        uses_privileged_info=uses_privileged_info,
        trainable=trainable,
        paper_ready=paper_ready,
        implemented=implemented,
        requires_checkpoint=requires_checkpoint,
        checkpoint_loaded=checkpoint_loaded,
        is_placeholder=is_placeholder,
        allows_formal_eval=bool(allows_formal_eval),
        fallback_policy=fallback_policy,
        update_implemented=update_implemented,
    )


def _baseline_metadata_table(*, checkpoint_loaded: bool = False) -> Dict[str, BaselineMetadata]:
    table = {
        "local_only": _meta("local_only", "static"),
        "neighbor_only": _meta("neighbor_only", "static"),
        "geo_only": _meta("geo_only", "static"),
        "ground_only": _meta("ground_only", "static"),
        "remote_only": _meta("remote_only", "static", paper_ready=False, allows_formal_eval=False),
        "random_visible": _meta("random_visible", "heuristic", fallback_policy="random_visible"),
        "min_delay_greedy": _meta("min_delay_greedy", "heuristic", fallback_policy="cost_greedy"),
        "min_energy_greedy": _meta("min_energy_greedy", "heuristic", fallback_policy="cost_greedy"),
        "queue_aware_greedy": _meta("queue_aware_greedy", "heuristic", fallback_policy="cost_greedy"),
        "mobility_risk_greedy": _meta("mobility_risk_greedy", "heuristic", fallback_policy="cost_greedy"),
        "lyapunov_dpp_greedy": _meta("lyapunov_dpp_greedy", "optimization", fallback_policy="cost_greedy"),
        "optimized_lyapunov_dpp": _meta("optimized_lyapunov_dpp", "optimization", fallback_policy="cost_greedy"),
        "lyapunov_dpp_optimized": _meta("lyapunov_dpp_optimized", "optimization", fallback_policy="cost_greedy"),
        "tri_mappo_maddpg": _meta(
            "tri_mappo_maddpg",
            "rl",
            trainable=True,
            paper_ready=True,
            requires_checkpoint=True,
            checkpoint_loaded=checkpoint_loaded,
            fallback_policy=None,
        ),
        "flat_ppo": _meta("flat_ppo", "rl", trainable=True, requires_checkpoint=True, checkpoint_loaded=checkpoint_loaded),
        "flat_mappo": _meta("flat_mappo", "rl", trainable=True, requires_checkpoint=True, checkpoint_loaded=checkpoint_loaded),
        "hierarchical_no_gnn": _meta("hierarchical_no_gnn", "rl", trainable=True, requires_checkpoint=True, checkpoint_loaded=checkpoint_loaded),
        "hmadrl_maddqn_ddpg": _meta(
            "hmadrl_maddqn_ddpg",
            "placeholder",
            paper_ready=False,
            implemented=False,
            is_placeholder=True,
            allows_formal_eval=False,
            fallback_policy="random_visible",
            update_implemented=False,
        ),
        "random_mobility_safe": _meta("random_mobility_safe", "heuristic", paper_ready=False, allows_formal_eval=False, fallback_policy="random_visible"),
        "round_robin_visible": _meta("round_robin_visible", "heuristic", paper_ready=False, allows_formal_eval=False),
        "weight_greedy": _meta("weight_greedy", "heuristic", paper_ready=False, allows_formal_eval=False),
        "cost_greedy": _meta("cost_greedy", "heuristic", paper_ready=False, allows_formal_eval=False),
        "random": _meta("random", "heuristic", paper_ready=False, allows_formal_eval=False, fallback_policy="random_visible"),
        "greedy_delay": _meta("greedy_delay", "heuristic", paper_ready=False, allows_formal_eval=False),
        "greedy_energy": _meta("greedy_energy", "heuristic", paper_ready=False, allows_formal_eval=False),
        "greedy_queue": _meta("greedy_queue", "heuristic", paper_ready=False, allows_formal_eval=False),
        "only_leo": _meta("only_leo", "placeholder", paper_ready=False, implemented=False, is_placeholder=True, allows_formal_eval=False),
        "leo_geo": _meta("leo_geo", "placeholder", paper_ready=False, implemented=False, is_placeholder=True, allows_formal_eval=False),
        "leo_ground": _meta("leo_ground", "placeholder", paper_ready=False, implemented=False, is_placeholder=True, allows_formal_eval=False),
        "full": _meta("full", "placeholder", paper_ready=False, implemented=False, is_placeholder=True, allows_formal_eval=False),
    }
    bad = [k for k, v in table.items() if v.type not in BASELINE_TYPES]
    if bad:
        raise ValueError(f"invalid baseline metadata type for {bad}; supported={sorted(BASELINE_TYPES)}")
    return table


def baseline_metadata_registry(*, checkpoint_loaded: bool = False) -> Dict[str, BaselineMetadata]:
    return dict(_baseline_metadata_table(checkpoint_loaded=checkpoint_loaded))


def baseline_metadata(name: str, *, checkpoint_loaded: bool = False) -> BaselineMetadata:
    key = str(name or "").strip().lower()
    table = _baseline_metadata_table(checkpoint_loaded=checkpoint_loaded)
    if key not in table:
        raise ValueError(f"unsupported baseline={name!r}; choose from {sorted(table)}")
    return table[key]


def baseline_metadata_json(*, checkpoint_loaded: bool = False) -> Dict[str, Dict[str, Any]]:
    return {name: meta.to_dict() for name, meta in _baseline_metadata_table(checkpoint_loaded=checkpoint_loaded).items()}


def validate_baseline_for_formal(name: str, *, checkpoint_loaded: bool = False) -> BaselineMetadata:
    meta = baseline_metadata(name, checkpoint_loaded=checkpoint_loaded)
    if meta.is_placeholder:
        raise ValueError(f"baseline {name!r} is_placeholder=true and cannot enter formal evaluation")
    if not meta.implemented:
        raise ValueError(f"baseline {name!r} implemented=false and cannot enter formal evaluation")
    if not meta.paper_ready:
        raise ValueError(f"baseline {name!r} paper_ready=false and cannot enter formal evaluation")
    if meta.requires_checkpoint and not meta.checkpoint_loaded:
        raise ValueError(f"baseline {name!r} requires_checkpoint=true but checkpoint_loaded=false")
    if not meta.allows_formal_eval:
        raise ValueError(f"baseline {name!r} allows_formal_eval=false")
    return meta


def formal_baseline_metadata_registry(*, checkpoint_loaded: bool = False) -> Dict[str, BaselineMetadata]:
    out: Dict[str, BaselineMetadata] = {}
    for name, meta in _baseline_metadata_table(checkpoint_loaded=checkpoint_loaded).items():
        if meta.allows_formal_eval:
            out[name] = meta
    return out


def formal_baseline_names(*, checkpoint_loaded: bool = False) -> List[str]:
    return sorted(formal_baseline_metadata_registry(checkpoint_loaded=checkpoint_loaded))


def assert_no_placeholder_baselines(rows: List[Mapping[str, Any]], *, context: str) -> None:
    blocked: List[str] = []
    table = _baseline_metadata_table()
    for row in rows:
        name = str(row.get("baseline", "") or "").strip().lower()
        if not name:
            continue
        meta = table.get(name)
        if meta is not None and (meta.type == "placeholder" or not bool(meta.paper_ready)):
            blocked.append(name)
    if blocked:
        raise ValueError(
            f"{context} contains non-paper-ready/placeholder baselines: {sorted(set(blocked))}. "
            "Complete training, checkpoint evaluation, and literature mapping before exporting formal tables."
        )


def paper_ready_baseline_names() -> List[str]:
    return [name for name, meta in _baseline_metadata_table().items() if meta.paper_ready and not meta.is_placeholder and not meta.requires_checkpoint]


def baseline_names(*, paper_ready_only: bool = False, include_placeholder: bool = True) -> List[str]:
    names = sorted(_policy_factories().keys())
    if not paper_ready_only and include_placeholder:
        return names
    out: List[str] = []
    table = _baseline_metadata_table()
    for name in names:
        meta = table[name]
        if paper_ready_only and not bool(meta.paper_ready):
            continue
        if not include_placeholder and meta.type == "placeholder":
            continue
        out.append(name)
    return out


def _policy_factories() -> Dict[str, Callable[[], BaselinePolicy]]:
    from trisatflow.baselines.hmadrl_baseline import HMADRLMaddqnDdpgBaseline, TriMappoMaddpgBaseline
    from trisatflow.baselines.heuristic_policies import (
        LyapunovDppGreedyPolicy,
        MinDelayGreedyPolicy,
        MinEnergyGreedyPolicy,
        MobilityRiskGreedyPolicy,
        QueueAwareGreedyPolicy,
        RandomMobilitySafePolicy,
        RandomVisiblePolicy,
        RoundRobinVisiblePolicy,
        WeightGreedyPolicy,
    )
    from trisatflow.baselines.optimized_dpp import LyapunovDppOptimizedPolicy, OptimizedLyapunovDppPolicy
    from trisatflow.baselines.static_policies import (
        GeoOnlyPolicy,
        GroundOnlyPolicy,
        LocalOnlyPolicy,
        NeighborOnlyPolicy,
        RemoteOnlyPolicy,
    )

    return {
        "local_only": LocalOnlyPolicy,
        "neighbor_only": NeighborOnlyPolicy,
        "geo_only": GeoOnlyPolicy,
        "ground_only": GroundOnlyPolicy,
        "remote_only": RemoteOnlyPolicy,
        "random_visible": RandomVisiblePolicy,
        "min_delay_greedy": MinDelayGreedyPolicy,
        "min_energy_greedy": MinEnergyGreedyPolicy,
        "queue_aware_greedy": QueueAwareGreedyPolicy,
        "mobility_risk_greedy": MobilityRiskGreedyPolicy,
        "lyapunov_dpp_greedy": LyapunovDppGreedyPolicy,
        "optimized_lyapunov_dpp": OptimizedLyapunovDppPolicy,
        "lyapunov_dpp_optimized": LyapunovDppOptimizedPolicy,
        "tri_mappo_maddpg": TriMappoMaddpgBaseline,
        "hmadrl_maddqn_ddpg": HMADRLMaddqnDdpgBaseline,
        # compatibility aliases
        "random_mobility_safe": RandomMobilitySafePolicy,
        "round_robin_visible": RoundRobinVisiblePolicy,
        "weight_greedy": WeightGreedyPolicy,
        "cost_greedy": MinDelayGreedyPolicy,
        "random": RandomVisiblePolicy,
        "greedy_delay": MinDelayGreedyPolicy,
        "greedy_energy": MinEnergyGreedyPolicy,
        "greedy_queue": QueueAwareGreedyPolicy,
        "only_leo": MinDelayGreedyPolicy,
        "leo_geo": MinDelayGreedyPolicy,
        "leo_ground": MinDelayGreedyPolicy,
        "full": MinDelayGreedyPolicy,
    }


def build_baseline_policy(name: str) -> BaselinePolicy:
    key = str(name or "").strip().lower()
    if key in LEGACY_BASELINE_ALIASES:
        canonical = LEGACY_BASELINE_ALIASES[key]
        warnings.warn(
            f"baseline name {key!r} is deprecated; use {canonical!r}",
            UserWarning,
            stacklevel=2,
        )
        key = canonical
    factories = _policy_factories()
    if key not in factories:
        raise ValueError(f"unsupported baseline={name!r}; choose from {sorted(factories)}")
    return factories[key]()
