"""Planner adapter protocol.

Existing MAPPO/MADDPG code stays an inner planner implementation.  The outer
controller only depends on this narrow protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import PlannerCapabilities, PlannerFidelity, PlannerResult, PlanningBudget, PlanningDescriptor


class PlannerBackend(Protocol):
    name: str
    family: str
    fidelity: PlannerFidelity

    def capabilities(self) -> PlannerCapabilities:
        ...

    def estimate_decision_cost(
        self,
        planner_state: Any,
        current_config: Any,
        scope: ReconfigurationScope,
        budget: PlanningBudget,
    ) -> DecisionCostBreakdown:
        ...

    def describe_planning(
        self,
        monitor_state: Any,
        current_config: Any,
        scope: ReconfigurationScope,
        budget: PlanningBudget,
    ) -> PlanningDescriptor:
        ...

    def plan(
        self,
        planner_state: Any,
        current_config: Any,
        scope: ReconfigurationScope,
        budget: PlanningBudget,
    ) -> PlannerResult:
        ...


@dataclass
class PlannerSpec:
    name: str
    family: str
    fidelity: PlannerFidelity
    backend: PlannerBackend
    budget: PlanningBudget = field(default_factory=PlanningBudget)
    estimated_cost: DecisionCostBreakdown = field(default_factory=DecisionCostBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "fidelity": self.fidelity.value,
            "budget": self.budget.to_dict(),
            "estimated_cost": self.estimated_cost.to_dict(),
            "metadata": dict(self.metadata),
        }
