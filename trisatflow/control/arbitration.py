"""Delay-aware Value of Computation and planner arbitration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.types import PlannerFidelity, PlanningBudget, coerce_fidelity


@dataclass
class PlannerCandidate:
    scope: Any
    fidelity: PlannerFidelity
    budget: PlanningBudget
    planner_name: str
    estimated_improvement: float = 0.0
    estimated_hold_cost: float = 0.0
    estimated_candidate_cost: DecisionCostBreakdown = field(default_factory=DecisionCostBreakdown)
    delay: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value_of_computation(self) -> float:
        return float(self.estimated_improvement + self.estimated_hold_cost - self.estimated_candidate_cost.intervention_cost)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fidelity"] = self.fidelity.value
        payload["budget"] = self.budget.to_dict()
        payload["estimated_candidate_cost"] = self.estimated_candidate_cost.to_dict()
        payload["value_of_computation"] = self.value_of_computation
        return payload


@dataclass
class VoC:
    selected: PlannerCandidate | None
    estimated_hold_cost: float
    estimated_candidate_cost: float
    value: float
    reason: str = ""

    @property
    def keep(self) -> bool:
        return self.selected is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "estimated_hold_cost": self.estimated_hold_cost,
            "estimated_candidate_cost": self.estimated_candidate_cost,
            "VoC": self.value,
            "value": self.value,
            "reason": self.reason,
        }


class PlannerArbitrator:
    def __init__(self, *, include_decision_cost: bool = True, no_decision_cost: bool = False) -> None:
        self.include_decision_cost = bool(include_decision_cost) and not bool(no_decision_cost)

    def choose(self, candidates: Iterable[PlannerCandidate], *, hold_cost: float = 0.0) -> VoC:
        candidates = list(candidates)
        if not candidates:
            return VoC(None, float(hold_cost), 0.0, 0.0, "no_candidate")
        best = max(
            candidates,
            key=lambda item: item.estimated_improvement + item.estimated_hold_cost
            - (item.estimated_candidate_cost.intervention_cost if self.include_decision_cost else 0.0),
        )
        candidate_cost = best.estimated_candidate_cost.intervention_cost if self.include_decision_cost else 0.0
        value = float(best.estimated_improvement + best.estimated_hold_cost - candidate_cost)
        if value <= 0.0:
            return VoC(None, float(hold_cost), candidate_cost, value, "voc_non_positive")
        return VoC(best, float(hold_cost), candidate_cost, value, "best_positive_voc")

    @staticmethod
    def build_candidates(
        scopes: Iterable[Any],
        planner_specs: Iterable[Any],
        *,
        estimated_improvement: float = 0.0,
        hold_cost: float = 0.0,
    ) -> list[PlannerCandidate]:
        result: list[PlannerCandidate] = []
        for scope in scopes:
            for spec in planner_specs:
                fidelity = coerce_fidelity(getattr(spec, "fidelity", PlannerFidelity.LIGHT))
                budget = getattr(spec, "budget", PlanningBudget())
                cost = getattr(spec, "estimated_cost", DecisionCostBreakdown())
                result.append(
                    PlannerCandidate(
                        scope=scope,
                        fidelity=fidelity,
                        budget=budget,
                        planner_name=str(getattr(spec, "name", "unknown")),
                        estimated_improvement=float(estimated_improvement),
                        estimated_hold_cost=float(hold_cost),
                        estimated_candidate_cost=cost,
                    )
                )
        return result
