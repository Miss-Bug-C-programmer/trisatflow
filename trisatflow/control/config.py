"""Dataclass configuration for the outer control loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import PlannerFidelity, PlanningBudget


@dataclass
class MonitorConfig:
    interval: float = 1.0
    true_cheap_required: bool = False
    fallback_mode: str = "compatibility_preflight"


@dataclass
class ViabilityConfig:
    uncertainty_margin: float = 0.0
    feasibility_margin: float = 0.0
    performance_risk_threshold: float = 0.5
    contact_predictability: bool = True
    evaluation_horizon_sec: float = 10.0
    service_safety_fraction: float = 0.1


@dataclass
class ScopeConfig:
    max_candidate_scopes: int = 4
    max_scope_entities: int | None = None
    include_global_candidate: bool = True


@dataclass
class PlannerArbitrationConfig:
    enabled_backends: list[str] = field(default_factory=lambda: ["greedy_weighted_cost"])
    fidelity_levels: list[str] = field(default_factory=lambda: ["light", "medium", "high"])
    budget_levels: dict[str, dict[str, Any]] = field(default_factory=dict)
    no_decision_cost: bool = False
    hold_cost_weight: float = 1.0


@dataclass
class DecisionCostConfig:
    enabled: bool = True
    obs_price: float = 1.0
    sync_price: float = 1.0
    solve_price: float = 1.0
    signal_price: float = 1.0
    reconfiguration_price: float = 1.0
    average_decision_energy_budget: float | None = None
    average_control_bytes_budget: float | None = None
    average_decision_compute_budget: float | None = None
    observation_byte_price: float | None = None
    observation_latency_price: float | None = None
    observation_energy_price: float | None = None
    sync_byte_price: float | None = None
    sync_latency_price: float | None = None
    sync_energy_price: float | None = None
    solve_wallclock_price: float | None = None
    solve_latency_price: float | None = None
    solve_compute_price: float | None = None
    solve_energy_price: float | None = None
    signal_byte_price: float | None = None
    signal_latency_price: float | None = None
    signal_energy_price: float | None = None
    reconfiguration_byte_price: float | None = None
    reconfiguration_volume_price: float | None = None
    reconfiguration_assignment_price: float | None = None
    reconfiguration_resource_price: float | None = None
    reconfiguration_route_price: float | None = None


@dataclass
class BenefitConfig:
    evaluation_horizon_sec: float = 10.0
    score_mode: str = "mean"
    lcb_beta: float = 1.0
    objective_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionDelayConfig:
    mode: str = "none"
    require_physical_enforcement: bool = False
    modeled_components: tuple[str, ...] = ("solver",)
    use_wallclock_as_simulated: bool = False


@dataclass
class SMDPConfig:
    discount_mode: str = "power"
    gamma: float = 0.99


@dataclass
class AblationConfig:
    fixed_period_replanning: bool = False
    fixed_period: int = 1
    state_change_trigger: bool = False
    global_only_intervention: bool = False
    fixed_intervention_scope: ReconfigurationScope | None = None
    no_decision_cost: bool = False
    solver_latency_only: bool = False
    reward_penalty_delay: bool = False
    no_contact_predictability: bool = False
    no_uncertainty_margin: bool = False
    always_high_fidelity: bool = False
    cost_blind_planner_selection: bool = False
    heuristic_fidelity_multiplier: bool = False
    mean_voc: bool = False
    lcb_voc: bool = False
    full_state_acquisition_compatibility: bool = False


@dataclass
class ControllerConfig:
    mode: str = "endogenous_replanning"
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    viability: ViabilityConfig = field(default_factory=ViabilityConfig)
    benefit: BenefitConfig = field(default_factory=BenefitConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    planner: PlannerArbitrationConfig = field(default_factory=PlannerArbitrationConfig)
    decision_cost: DecisionCostConfig = field(default_factory=DecisionCostConfig)
    decision_delay: DecisionDelayConfig = field(default_factory=DecisionDelayConfig)
    smdp: SMDPConfig = field(default_factory=SMDPConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    seed: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ControllerConfig":
        data = dict(payload or {})
        nested = {}
        for name, type_ in (
            ("monitor", MonitorConfig),
            ("viability", ViabilityConfig),
            ("benefit", BenefitConfig),
            ("scope", ScopeConfig),
            ("planner", PlannerArbitrationConfig),
            ("decision_cost", DecisionCostConfig),
            ("decision_delay", DecisionDelayConfig),
            ("smdp", SMDPConfig),
            ("ablations", AblationConfig),
        ):
            value = data.get(name, {})
            if isinstance(value, type_):
                nested[name] = value
            elif isinstance(value, Mapping):
                value = dict(value)
                if name == "ablations" and isinstance(value.get("fixed_intervention_scope"), Mapping):
                    value["fixed_intervention_scope"] = ReconfigurationScope(**value["fixed_intervention_scope"])
                if name == "decision_delay" and isinstance(value.get("modeled_components"), list):
                    value["modeled_components"] = tuple(value["modeled_components"])
                known = set(type_.__dataclass_fields__)
                nested[name] = type_(**{key: item for key, item in value.items() if key in known})
        data.update(nested)
        data["mode"] = str(data.get("mode", "endogenous_replanning"))
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})

    def resolved_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        fixed_scope = self.ablations.fixed_intervention_scope
        if fixed_scope is not None:
            payload["ablations"]["fixed_intervention_scope"] = fixed_scope.to_dict()
        payload["decision_delay"]["modeled_components"] = list(self.decision_delay.modeled_components)
        return payload


def budget_from_mapping(payload: Mapping[str, Any] | None) -> PlanningBudget:
    data = dict(payload or {})
    valid = set(PlanningBudget.__dataclass_fields__)
    return PlanningBudget(**{key: value for key, value in data.items() if key in valid})


def load_controller_config(path: str | Path | None = None) -> ControllerConfig:
    """Load a control-only YAML mapping without changing legacy TrainConfig."""

    if path is None:
        return ControllerConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Controller YAML must contain a mapping")
    control = payload.get("controller", payload.get("control", payload))
    if not isinstance(control, Mapping):
        raise ValueError("controller/control YAML section must contain a mapping")
    return ControllerConfig.from_mapping(control)


def save_controller_config(config: ControllerConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.resolved_dict(), handle, sort_keys=False, allow_unicode=True)
