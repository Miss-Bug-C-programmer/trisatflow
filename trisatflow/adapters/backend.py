"""Capability-aware physical backend protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class BackendCapabilities:
    server_version: str = "unknown"
    control_physical_contract_version: str = "unknown"
    supports_cheap_monitor: bool = False
    supports_monitor_state: bool = False
    supports_planner_state: bool = False
    supports_scoped_planner_state: bool = False
    supports_budget_aware_planner_state: bool = False
    supports_contact_plan: bool = False
    supports_topology_snapshot: bool = False
    supports_configuration_apply: bool = False
    supports_persistent_configuration: bool = False
    supports_persistent_configuration_execution: bool = False
    supports_configuration_dispatch: bool = False
    supports_physical_decision_delay: bool = False
    supports_advance_world: bool = False
    supports_scope_aware_planner_state: bool = False
    supports_budget_aware_planner_state: bool = False
    supports_configuration_validation: bool = False
    supports_verified_delay_receipt: bool = False
    supports_mid_transfer_contact_enforcement: bool = False
    future_stochastic_truth_exposed: bool = False
    physical_decision_delay_semantics_version: str = "unknown"
    configuration_semantics_version: str = "unknown"
    scope_dimensions: set[str] = field(default_factory=set)
    supported_budget_dimensions: set[str] = field(default_factory=set)
    persistent_rule_dimensions: set[str] = field(default_factory=set)
    backend_source: str = "unknown"
    topology_source: str = "unknown"
    monitor_state_source: str = "unknown"
    authoritative_physical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supported_budget_dimensions"] = sorted(self.supported_budget_dimensions)
        payload["scope_dimensions"] = sorted(self.scope_dimensions)
        payload["persistent_rule_dimensions"] = sorted(self.persistent_rule_dimensions)
        return payload


class PhysicalBackend(Protocol):
    capabilities: BackendCapabilities

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_monitor_state(self, context: Any | None = None) -> Any:
        ...

    def get_planner_state(self, context: Any | None = None, scope: Any | None = None, budget: Any | None = None) -> Any:
        ...

    def get_contact_forecast(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_current_topology(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def materialize_current_configuration(self, configuration: Any, task: Any | None = None) -> Any:
        ...

    def apply_configuration(self, configuration: Any) -> Any:
        ...

    def dispatch_under_configuration(self, configuration: Any, task: Any | None = None) -> Any:
        ...

    def advance_world(self, delta_sec: float) -> Any:
        ...

    def current_time(self) -> float:
        ...

    def validate_configuration(self, configuration: Any) -> Any:
        ...
