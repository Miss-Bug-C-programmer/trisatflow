"""Shared, serialisable data types for the outer decision loop.

These types deliberately contain observations available at decision time only.
In particular, :class:`MonitorState` has no future stochastic workload, queue,
or channel realisation fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class FeasibilityStatus(str, Enum):
    VIABLE = "VIABLE"
    INVIABLE = "INVIABLE"
    UNCERTAIN = "UNCERTAIN"


class PlannerFidelity(str, Enum):
    """Planner resource tiers with operational, not merely cosmetic, meaning."""

    LIGHT = "light"
    LOW = "light"  # readable alias used by some experiment configurations
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PlanningBudget:
    """Concrete limits passed to a planner and enforced by adapters."""

    max_candidate_count: Optional[int] = None
    max_scope_entities: Optional[int] = None
    max_planner_evaluations: Optional[int] = None
    max_iterations: Optional[int] = None
    max_coordination_bytes: Optional[int] = None
    max_compute_budget: Optional[float] = None
    time_budget_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def restrict_count(self, values: Sequence[Any]) -> list[Any]:
        if self.max_candidate_count is None:
            return list(values)
        return list(values[: max(0, int(self.max_candidate_count))])

    def restrict_scope(self, scope: Any) -> Any:
        if self.max_scope_entities is None or not hasattr(scope, "truncate"):
            return scope
        return scope.truncate(max(0, int(self.max_scope_entities)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClockState:
    """The three clocks used by the controller.

    ``physical_time_sec`` and ``physical_slot`` describe the simulator.  The
    monitor and intervention epochs count control-plane events and therefore do
    not imply a fixed physical period.
    """

    physical_time_sec: float = 0.0
    physical_slot: int = 0
    monitor_epoch: int = 0
    intervention_epoch: int = 0
    configuration_age_sec: float = 0.0
    time_since_last_intervention_sec: float = 0.0

    def advance(self, delta_sec: float, *, physical_slot_delta: int = 0) -> "ClockState":
        delta = max(0.0, float(delta_sec))
        self.physical_time_sec += delta
        self.physical_slot += max(0, int(physical_slot_delta))
        self.configuration_age_sec += delta
        self.time_since_last_intervention_sec += delta
        return self

    def mark_monitor(self) -> "ClockState":
        self.monitor_epoch += 1
        return self

    def mark_intervention(self) -> "ClockState":
        self.intervention_epoch += 1
        self.configuration_age_sec = 0.0
        self.time_since_last_intervention_sec = 0.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerContext:
    """Runtime context passed to monitor, viability and planner components."""

    clock: ClockState = field(default_factory=ClockState)
    current_config_id: Optional[str] = None
    current_config_version: Optional[int] = None
    seed: Optional[int] = None
    run_id: str = ""
    backend_source: str = "unknown"
    topology_source: str = "unknown"
    physical_delay_enforced: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["clock"] = self.clock.to_dict()
        return payload


@dataclass
class MonitorAcquisitionMetadata:
    obs_bytes: int = 0
    num_queries: int = 0
    latency_sec: float = 0.0
    source: str = "unknown"
    is_true_cheap_monitor: bool = False
    monitor_http_calls: int = 0
    monitor_bytes: int = 0
    monitor_latency_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorState:
    """Low-cost state safe for KEEP decisions.

    The fields intentionally contain summaries and cached/predictable contact
    information only.  A backend must use :class:`PlannerState` for detailed
    candidate and graph acquisition after escalation.
    """

    simulation_time: float = 0.0
    current_config_id: Optional[str] = None
    current_config_version: Optional[int] = None
    local_queue_summary: Dict[str, float] = field(default_factory=dict)
    source_queue_summary: Dict[str, float] = field(default_factory=dict)
    remaining_workload_summary: Dict[str, float] = field(default_factory=dict)
    deadline_slack: Dict[str, float] = field(default_factory=dict)
    local_load_summary: Dict[str, float] = field(default_factory=dict)
    remaining_contact_lifetime: Dict[str, float] = field(default_factory=dict)
    next_contact_summary: Dict[str, Any] = field(default_factory=dict)
    contact_slack: Dict[str, float] = field(default_factory=dict)
    small_neighborhood_state: Dict[str, Any] = field(default_factory=dict)
    cached_state: Dict[str, Any] = field(default_factory=dict)
    prediction_uncertainty: Dict[str, float] = field(default_factory=dict)
    degradation_indicators: Dict[str, float] = field(default_factory=dict)
    acquisition: MonitorAcquisitionMetadata = field(default_factory=MonitorAcquisitionMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["acquisition"] = self.acquisition.to_dict()
        return payload


@dataclass
class PlannerState:
    """Heavy state acquired only after intervention escalation."""

    simulation_time: float = 0.0
    candidate_vms: list[Dict[str, Any]] = field(default_factory=list)
    queue_load: Dict[str, Any] = field(default_factory=dict)
    detailed_resources: Dict[str, Any] = field(default_factory=dict)
    topology: Dict[str, Any] = field(default_factory=dict)
    candidate_links: Dict[str, Any] = field(default_factory=dict)
    candidate_execution_costs: Dict[str, Any] = field(default_factory=dict)
    observation: Any = None
    acquisition: MonitorAcquisitionMetadata = field(default_factory=MonitorAcquisitionMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["acquisition"] = self.acquisition.to_dict()
        return payload


@dataclass
class PlannerCapabilities:
    supports_scope_restriction: bool = False
    supports_candidate_restriction: bool = False
    supports_budget_limits: bool = False
    supports_checkpoint: bool = False
    supports_upper_lower_hierarchy: bool = False
    supports_cost_estimation: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerResult:
    configuration: Any
    planner_name: str = "unknown"
    planner_family: str = "unknown"
    fidelity: PlannerFidelity = PlannerFidelity.LIGHT
    budget: PlanningBudget = field(default_factory=PlanningBudget)
    planned_at_sim_time: float = 0.0
    planning_delay_sec: float = 0.0
    decision_cost: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SMDPTransition:
    """Variable-duration outer transition."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    holding_time: float
    start_time: float = 0.0
    end_time: float = 0.0
    physical_slot: int = 0
    monitor_epoch: int = 0
    intervention_epoch: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_discount(self, gamma: float, *, mode: str = "power") -> float:
        if str(mode).lower() in {"one", "fixed", "legacy"}:
            return float(gamma)
        return float(gamma) ** max(0.0, float(self.holding_time))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce_fidelity(value: PlannerFidelity | str) -> PlannerFidelity:
    if isinstance(value, PlannerFidelity):
        return value
    text = str(value).strip().lower()
    if text in {"low", "light", "cheap"}:
        return PlannerFidelity.LIGHT
    if text in {"medium", "mid"}:
        return PlannerFidelity.MEDIUM
    if text in {"high", "full", "hierarchical"}:
        return PlannerFidelity.HIGH
    raise ValueError(f"Unknown planner fidelity: {value!r}")


def mapping_float(mapping: Mapping[str, Any] | None, default: float = 0.0) -> float:
    """Sum numeric values in a summary mapping without reading hidden future state."""

    if not mapping:
        return float(default)
    values = []
    for value in mapping.values():
        if isinstance(value, Mapping):
            values.append(mapping_float(value))
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return float(sum(values)) if values else float(default)
