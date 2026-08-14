"""Adapter for the existing hierarchical MAPPO/MADDPG implementation.

The adapter intentionally does not rewrite or retrain the legacy agents.  A
caller supplies an action/configuration provider backed by a configured
``HierarchicalTrainer`` and can use checkpoints through that provider.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import PlannerCapabilities, PlannerFidelity, PlannerResult, PlanningBudget


class HierarchicalMARLPlannerAdapter:
    name = "hierarchical_mappo_maddpg"
    family = "hierarchical_marl"
    fidelity = PlannerFidelity.HIGH

    def __init__(self, *, action_provider: Callable[..., Any] | None = None, trainer: Any | None = None, checkpoint: str | None = None) -> None:
        self.action_provider = action_provider
        self.trainer = trainer
        self.checkpoint = checkpoint

    def capabilities(self) -> PlannerCapabilities:
        return PlannerCapabilities(
            supports_scope_restriction=False,
            supports_candidate_restriction=False,
            supports_budget_limits=True,
            supports_checkpoint=self.checkpoint is not None,
            supports_upper_lower_hierarchy=True,
            metadata={
                "upper_algorithm": "MAPPO",
                "lower_algorithm": "MADDPG",
                "legacy_role": "inner_planner",
                "checkpoint": self.checkpoint,
            },
        )

    def estimate_decision_cost(self, planner_state: Any, current_config: Any, scope: ReconfigurationScope, budget: PlanningBudget) -> DecisionCostBreakdown:
        candidate_count = len(list(getattr(planner_state, "candidate_vms", []) or []))
        return DecisionCostBreakdown(
            observation_bytes=int(getattr(getattr(planner_state, "acquisition", None), "obs_bytes", 0)),
            sync_bytes=max(0, candidate_count * 16),
            solver_compute_proxy=float(max(1, budget.max_iterations or 1) * max(1, candidate_count)),
            signal_bytes=max(1, scope.cardinality),
            metadata={"scope_execution_restricted": True, "scope_computation_restricted": False},
        )

    def plan(self, planner_state: Any, current_config: Any, scope: ReconfigurationScope, budget: PlanningBudget) -> PlannerResult:
        if self.action_provider is None and self.trainer is None:
            raise RuntimeError(
                "HierarchicalMARLPlannerAdapter requires a configured trainer/action_provider; "
                "no checkpoint-backed DRL planner was silently invented."
            )
        started = time.perf_counter()
        if self.action_provider is not None:
            raw = self.action_provider(planner_state, current_config, scope, budget)
        elif hasattr(self.trainer, "plan_configuration"):
            raw = self.trainer.plan_configuration(planner_state, current_config, scope, budget)
        elif hasattr(self.trainer, "select_action"):
            raw = self.trainer.select_action(planner_state)
        else:
            raise RuntimeError("Configured hierarchical trainer exposes no planning method")

        if isinstance(raw, PersistentConfiguration):
            next_config = raw
        elif isinstance(raw, dict) and isinstance(raw.get("configuration"), PersistentConfiguration):
            next_config = raw["configuration"]
        elif isinstance(raw, dict):
            next_config = current_config.apply_patch(raw)
        else:
            raise TypeError(f"Hierarchical planner provider returned unsupported result: {type(raw)!r}")
        next_config.planner_name = self.name
        next_config.planner_fidelity = self.fidelity.value
        next_config.planning_budget = budget.to_dict()
        next_config.scope_from_previous = scope
        cost = self.estimate_decision_cost(planner_state, current_config, scope, budget)
        cost.solver_wallclock_sec = max(0.0, time.perf_counter() - started)
        return PlannerResult(
            configuration=next_config,
            planner_name=self.name,
            planner_family=self.family,
            fidelity=self.fidelity,
            budget=budget,
            planned_at_sim_time=float(getattr(planner_state, "simulation_time", 0.0)),
            planning_delay_sec=cost.solver_wallclock_sec,
            decision_cost=cost,
            metadata={"scope_execution_restricted": True, "scope_computation_restricted": False, "checkpoint": self.checkpoint},
        )
