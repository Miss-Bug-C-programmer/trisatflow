"""Dependency-light planner backend for tests, preflight and low fidelity."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Mapping

from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import PlannerCapabilities, PlannerFidelity, PlannerResult, PlanningBudget, PlanningDescriptor


class GreedyPlanner:
    name = "greedy_weighted_cost"
    family = "deterministic_greedy"
    fidelity = PlannerFidelity.LIGHT

    def __init__(self, *, score_key: str = "estimatedTotalDelaySec", source_name: str = "test_backend") -> None:
        self.score_key = score_key
        self.source_name = source_name

    def capabilities(self) -> PlannerCapabilities:
        return PlannerCapabilities(
            supports_scope_restriction=True,
            supports_candidate_restriction=True,
            supports_budget_limits=True,
            supports_scope_aware_acquisition=True,
            supports_budget_aware_acquisition=True,
            supported_budget_dimensions={"max_candidate_count", "max_scope_entities", "max_planner_evaluations", "max_coordination_bytes"},
            supports_cost_estimation=True,
            metadata={"backend_type": self.source_name, "truth_authoritative": False},
        )

    def estimate_decision_cost(self, planner_state: Any, current_config: Any, scope: ReconfigurationScope, budget: PlanningBudget) -> DecisionCostBreakdown:
        descriptor_count = getattr(planner_state, "estimated_candidate_count", None)
        candidates = list(getattr(planner_state, "candidate_vms", []) or [])
        if descriptor_count is not None and not candidates:
            candidates = [None] * max(0, int(descriptor_count))
        candidates = budget.restrict_count(candidates)
        simulated_latency = float((budget.metadata or {}).get("simulated_latency_sec", 0.0) or 0.0)
        return DecisionCostBreakdown(
            observation_bytes=int(getattr(planner_state, "estimated_observation_bytes", getattr(getattr(planner_state, "acquisition", None), "obs_bytes", 0))),
            solver_compute_proxy=float(max(1, len(candidates))),
            solver_simulated_latency_sec=max(0.0, simulated_latency),
            solver_wallclock_sec=0.0,
            signal_bytes=max(1, scope.cardinality),
            metadata={"candidate_count": len(candidates), "scope_execution_restricted": True},
        )

    def describe_planning(self, monitor_state: Any, current_config: Any, scope: ReconfigurationScope, budget: PlanningBudget) -> PlanningDescriptor:
        estimated_count = int((monitor_state.metadata or {}).get("candidate_count_hint", 0))
        return PlanningDescriptor(
            planner_name=self.name,
            planner_family=self.family,
            fidelity=self.fidelity,
            scope_cardinality=scope.cardinality,
            scope_normalized_volume=scope.normalized_volume(),
            estimated_candidate_count=min(estimated_count, int(budget.max_candidate_count)) if budget.max_candidate_count is not None else estimated_count,
            estimated_observation_bytes=int((getattr(monitor_state, "metadata", {}) or {}).get("planner_observation_bytes_hint", 0)),
            estimated_solver_latency_sec=0.0,
            expected_benefit_mean=float((monitor_state.metadata or {}).get("greedy_expected_benefit", 0.0)),
            expected_benefit_uncertainty=float(max(0.0, _max_numeric(getattr(monitor_state, "prediction_uncertainty", {})))),
            supports_scope_aware_acquisition=True,
            supports_budget_aware_acquisition=True,
            source="greedy_cached_descriptor",
        )

    def plan(self, planner_state: Any, current_config: Any, scope: ReconfigurationScope, budget: PlanningBudget) -> PlannerResult:
        started = time.perf_counter()
        scope = budget.restrict_scope(scope)
        candidates = budget.restrict_count(list(getattr(planner_state, "candidate_vms", []) or []))
        if budget.max_planner_evaluations is not None:
            candidates = candidates[: max(0, int(budget.max_planner_evaluations))]
        if budget.max_compute_budget is not None and float(budget.max_compute_budget) <= 0.0:
            candidates = []
        if budget.max_iterations is not None and int(budget.max_iterations) <= 0:
            candidates = []
        chosen_by_source: dict[str, Mapping[str, Any]] = {}
        for candidate in candidates:
            if budget.time_budget_ms is not None and (time.perf_counter() - started) * 1000.0 > float(budget.time_budget_ms):
                break
            if not isinstance(candidate, Mapping):
                continue
            source = str(candidate.get("sourceId", candidate.get("source_id", candidate.get("leoId", "default"))))
            try:
                score = float(candidate.get(self.score_key, candidate.get("score", 0.0)))
            except (TypeError, ValueError):
                score = 0.0
            previous = chosen_by_source.get(source)
            if previous is None or score < float(previous.get(self.score_key, previous.get("score", 0.0)) or 0.0):
                chosen_by_source[source] = candidate

        assignments = deepcopy(getattr(current_config, "assignments", {}) or {})
        resources = deepcopy(getattr(current_config, "resource_allocations", {}) or {})
        for source, candidate in chosen_by_source.items():
            if scope.is_empty or scope.contains(source, "source") or scope.contains(source, "node"):
                assignments[source] = deepcopy(dict(candidate))
                if "resourceAllocation" in candidate:
                    resources[source] = deepcopy(candidate["resourceAllocation"])

        next_config = current_config.clone(
            config_id=f"{getattr(current_config, 'config_id', 'config')}.v{int(getattr(current_config, 'version', 0)) + 1}",
            version=int(getattr(current_config, "version", 0)) + 1,
        )
        next_config.assignments = assignments
        next_config.resource_allocations = resources
        next_config.planner_name = self.name
        next_config.planner_fidelity = self.fidelity.value
        next_config.planning_budget = budget.to_dict()
        next_config.scope_from_previous = scope
        next_config.metadata = {**getattr(next_config, "metadata", {}), "backend_source": self.source_name}
        delay = max(0.0, time.perf_counter() - started)
        cost = self.estimate_decision_cost(planner_state, current_config, scope, budget)
        cost.solver_wallclock_sec = delay
        if budget.max_coordination_bytes is not None:
            cost.signal_bytes = min(cost.signal_bytes, max(0, int(budget.max_coordination_bytes)))
        return PlannerResult(
            configuration=next_config,
            planner_name=self.name,
            planner_family=self.family,
            fidelity=self.fidelity,
            budget=budget,
            planned_at_sim_time=float(getattr(planner_state, "simulation_time", 0.0)),
            planning_delay_sec=delay,
            decision_cost=cost,
            metadata={"candidate_count": len(candidates), "scope_execution_restricted": True, "scope_computation_restricted": True},
        )


def _max_numeric(mapping: Any) -> float:
    values = []
    for value in (mapping or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0
