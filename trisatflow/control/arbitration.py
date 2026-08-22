"""Delay-aware Value of Computation and planner arbitration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from trisatflow.control.benefit import BenefitEstimate
from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.types import PlannerFidelity, PlanningBudget, PlanningDescriptor, coerce_fidelity


@dataclass
class PlannerCandidate:
    scope: Any
    fidelity: PlannerFidelity
    budget: PlanningBudget
    planner_name: str
    observation_scope: Any | None = None
    modification_scope: Any | None = None
    estimated_improvement: float = 0.0
    estimated_hold_cost: float = 0.0
    estimated_candidate_cost: DecisionCostBreakdown = field(default_factory=DecisionCostBreakdown)
    delay: Any = None
    planner_descriptor: PlanningDescriptor | None = None
    benefit_estimate: BenefitEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value_of_computation(self) -> float:
        if self.benefit_estimate is None:
            # Compatibility/test fixture path only. Proposed controller
            # candidates always carry a BenefitEstimate.
            benefit = float(self.estimated_improvement + self.estimated_hold_cost)
        else:
            benefit = float(self.benefit_estimate.risk_adjusted_benefit)
        return float(benefit - self.estimated_candidate_cost.intervention_cost)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fidelity"] = self.fidelity.value
        payload["budget"] = self.budget.to_dict()
        payload["estimated_candidate_cost"] = self.estimated_candidate_cost.to_dict()
        payload["planner_descriptor"] = self.planner_descriptor.to_dict() if self.planner_descriptor else None
        payload["benefit_estimate"] = self.benefit_estimate.to_dict() if self.benefit_estimate else None
        payload["value_of_computation"] = self.value_of_computation
        return payload


@dataclass
class VoC:
    selected: PlannerCandidate | None
    estimated_hold_cost: float
    estimated_candidate_cost: float
    value: float
    reason: str = ""
    benefit_estimate: BenefitEstimate | None = None

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
            "benefit_estimate": self.benefit_estimate.to_dict() if self.benefit_estimate else None,
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
            key=lambda item: self._benefit(item)
            - (item.estimated_candidate_cost.intervention_cost if self.include_decision_cost else 0.0),
        )
        candidate_cost = best.estimated_candidate_cost.intervention_cost if self.include_decision_cost else 0.0
        value = float(self._benefit(best) - candidate_cost)
        if value <= 0.0:
            return VoC(None, float(hold_cost), candidate_cost, value, "voc_non_positive", best.benefit_estimate)
        return VoC(best, float(hold_cost), candidate_cost, value, "best_positive_voc", best.benefit_estimate)

    @staticmethod
    def _benefit(candidate: PlannerCandidate) -> float:
        if candidate.benefit_estimate is not None:
            return float(candidate.benefit_estimate.risk_adjusted_benefit)
        return float(candidate.estimated_improvement + candidate.estimated_hold_cost)

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
