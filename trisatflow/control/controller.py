"""Endogenous, variable-duration outer controller."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.arbitration import PlannerArbitrator, PlannerCandidate, VoC
from trisatflow.control.config import ControllerConfig, budget_from_mapping
from trisatflow.control.decision_cost import DecisionCostBreakdown, ResourceBudgetState, cost_from_monitor
from trisatflow.control.decision_delay import DecisionDelayBreakdown, DecisionDelayModel, PostDelayRevalidator
from trisatflow.control.metrics import ControlMetrics
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope, ScopeGenerator
from trisatflow.control.types import (
    ClockState,
    ControllerContext,
    FeasibilityStatus,
    MonitorState,
    PlannerResult,
    PlannerState,
    SMDPTransition,
    coerce_fidelity,
)
from trisatflow.control.viability import ConservativeViabilityEstimator, ViabilityReport
from trisatflow.planners.base import PlannerBackend, PlannerSpec
from trisatflow.planners.greedy_planner import GreedyPlanner
from trisatflow.planners.registry import PlannerRegistry


@dataclass
class ControlDecision:
    action: str
    monitor_state: MonitorState | None = None
    viability_report: ViabilityReport | None = None
    scope: ReconfigurationScope = field(default_factory=ReconfigurationScope)
    planner_name: str = ""
    planner_fidelity: str = ""
    planning_budget: dict[str, Any] = field(default_factory=dict)
    voc: VoC | None = None
    planner_result: PlannerResult | None = None
    delay: DecisionDelayBreakdown | None = None
    stale_plan_rejection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def replanned(self) -> bool:
        return self.action == "INTERVENE" and self.planner_result is not None and not self.stale_plan_rejection


class EndogenousReplanningController:
    """Controller implementing KEEP / selective intervention outside the inner MARL loop.

    The controller never calls ``get_planner_state`` on the KEEP path.  A
    backend can therefore enforce a real cheap-monitor boundary and expose
    compatibility fallbacks explicitly in its metadata.
    """

    def __init__(
        self,
        backend: Any,
        *,
        config: ControllerConfig | Mapping[str, Any] | None = None,
        viability: Any | None = None,
        scope_generator: ScopeGenerator | None = None,
        planner_backends: Iterable[PlannerBackend] | None = None,
        arbitrator: PlannerArbitrator | None = None,
        delay_model: DecisionDelayModel | None = None,
        revalidator: PostDelayRevalidator | None = None,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        config_was_omitted = config is None
        self.config = config if isinstance(config, ControllerConfig) else ControllerConfig.from_mapping(config)
        self.seed = self.config.seed if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.clock = ClockState()
        self.context = ControllerContext(seed=self.seed)
        self.current_configuration: PersistentConfiguration | None = None
        self.viability = viability or ConservativeViabilityEstimator(
            uncertainty_margin=0.0 if self.config.ablations.no_uncertainty_margin else self.config.viability.uncertainty_margin,
            feasibility_margin=self.config.viability.feasibility_margin,
            performance_risk_threshold=self.config.viability.performance_risk_threshold,
            contact_predictability=not self.config.ablations.no_contact_predictability and self.config.viability.contact_predictability,
        )
        self.scope_generator = scope_generator or ScopeGenerator(
            max_candidate_scopes=self.config.scope.max_candidate_scopes,
            max_scope_entities=self.config.scope.max_scope_entities,
            include_global_candidate=self.config.scope.include_global_candidate,
        )
        self.planners = PlannerRegistry(list(planner_backends or []))
        if not self.planners.values():
            self.planners.register(GreedyPlanner(source_name="test_backend"))
        elif config_was_omitted:
            # An explicitly supplied registry is itself the experiment's
            # resolved planner set; do not accidentally filter it through the
            # default greedy-only config.
            self.config.planner.enabled_backends = [str(getattr(item, "name")) for item in self.planners.values()]
        self.arbitrator = arbitrator or PlannerArbitrator(
            include_decision_cost=self.config.decision_cost.enabled,
            no_decision_cost=self.config.planner.no_decision_cost or self.config.ablations.no_decision_cost,
        )
        self.delay_model = delay_model or DecisionDelayModel(
            mode=self._resolved_delay_mode(),
            require_physical_enforcement=self.config.decision_delay.require_physical_enforcement,
            modeled_components=self.config.decision_delay.modeled_components,
        )
        self.revalidator = revalidator or PostDelayRevalidator()
        self.metrics = ControlMetrics()
        self.resource_budget = ResourceBudgetState(
            average_decision_energy_budget=self.config.decision_cost.average_decision_energy_budget,
            average_control_bytes_budget=self.config.decision_cost.average_control_bytes_budget,
            average_decision_compute_budget=self.config.decision_cost.average_decision_compute_budget,
        )
        self._last_monitor: MonitorState | None = None
        self._last_viability: ViabilityReport | None = None
        self._last_intervention_time = 0.0
        self._initialised = False

    def initialize(
        self,
        initial_configuration: PersistentConfiguration | None = None,
        *,
        initial_plan: bool = True,
        planner_name: str | None = None,
    ) -> PersistentConfiguration:
        self._sync_clock()
        if initial_configuration is not None:
            self.current_configuration = initial_configuration.clone()
            self.current_configuration.applied_at_sim_time = self.clock.physical_time_sec
            self.current_configuration.last_validated_at_sim_time = self.clock.physical_time_sec
            self._apply_backend_configuration(self.current_configuration)
        elif initial_plan:
            self.current_configuration = self._make_empty_configuration()
            backend = self._select_backend(planner_name)
            try:
                state = self.backend.get_planner_state(self.context, ReconfigurationScope(), None)
            except TypeError:
                state = self.backend.get_planner_state()
            result = backend.plan(state, self.current_configuration, self._global_scope(state), self._budget_for(backend))
            self.current_configuration = result.configuration
            self.current_configuration.created_at_sim_time = self.clock.physical_time_sec
            self.current_configuration.applied_at_sim_time = self.clock.physical_time_sec
            self.current_configuration.last_validated_at_sim_time = self.clock.physical_time_sec
            self._apply_backend_configuration(self.current_configuration)
            self._record_initial_plan(result)
        else:
            self.current_configuration = self._make_empty_configuration()
        self._initialised = True
        self._update_context()
        return self.current_configuration

    def evaluate_current_configuration(self, monitor_state: MonitorState | None = None) -> ViabilityReport:
        if self.current_configuration is None:
            self.initialize(initial_plan=False)
        monitor = monitor_state or self._acquire_monitor_state()
        report = self.viability.evaluate(monitor, self.current_configuration)
        self._last_monitor = monitor
        self._last_viability = report
        return report

    def on_monitor_epoch(self, *, dispatch_task: Any | None = None) -> ControlDecision:
        if not self._initialised:
            self.initialize(initial_plan=True)
        self._sync_clock()
        self.clock.mark_monitor()
        self._update_context()
        monitor = self._acquire_monitor_state()
        direct_state_change = self.config.ablations.state_change_trigger and bool((monitor.metadata or {}).get("state_changed", False))
        if direct_state_change:
            # A2 is intentionally a trigger-only ablation: it bypasses the
            # proposed viability gate rather than calling it and relabeling the
            # result afterwards.
            report = ViabilityReport(
                feasibility_status=FeasibilityStatus.UNCERTAIN,
                affected_entities=self.current_configuration.affected_entities(),
                reason_codes=["ablation_state_change_trigger"],
                needs_intervention=True,
                confidence=0.0,
                evaluated_at=monitor.simulation_time,
                metadata={"ablation": "state_change_trigger"},
            )
        else:
            report = self.viability.evaluate(monitor, self.current_configuration)
        self._last_monitor = monitor
        self._last_viability = report

        forced = self._ablation_forces_intervention(monitor)
        if forced and not report.needs_intervention:
            report.needs_intervention = True
            report.reason_codes = [*report.reason_codes, "ablation_forced_intervention"]
            if report.affected_entities.is_empty:
                report.affected_entities = self.current_configuration.affected_entities()
        if self.config.mode == "legacy_slotwise" or (not report.needs_intervention and not forced):
            decision = ControlDecision(action="KEEP", monitor_state=monitor, viability_report=report)
            self._record_decision(decision)
            if dispatch_task is not None:
                self.dispatch_under_current_configuration(dispatch_task, monitor_state=monitor)
            return decision

        decision = self.decide_intervention(monitor, report)
        if dispatch_task is not None and decision.action == "KEEP":
            self.dispatch_under_current_configuration(dispatch_task, monitor_state=monitor)
        return decision

    def decide_intervention(self, monitor_state: MonitorState, report: ViabilityReport) -> ControlDecision:
        if report.affected_entities.is_empty and not self.config.ablations.global_only_intervention:
            decision = ControlDecision(
                action="KEEP",
                monitor_state=monitor_state,
                viability_report=report,
                metadata={"reason": "empty_scope_is_keep", "planner_state_acquired": False},
            )
            self._record_decision(decision)
            return decision
        planner_state, candidates = self.select_scope_planner_budget(monitor_state, report)
        hold_cost = self._hold_cost(report)
        voc = self.arbitrator.choose(candidates, hold_cost=hold_cost)
        if voc.keep:
            decision = ControlDecision(
                action="KEEP",
                monitor_state=monitor_state,
                viability_report=report,
                voc=voc,
                metadata={"reason": voc.reason, "planner_state_acquired": True},
            )
            self._record_decision(decision)
            return decision

        selected = voc.selected
        assert selected is not None
        backend = self.planners.get(selected.planner_name)
        result, delay, projected, accepted = self.execute_intervention(
            planner_state,
            backend,
            selected.scope,
            selected.budget,
            report=report,
        )
        decision = ControlDecision(
            action="INTERVENE" if accepted else "KEEP",
            monitor_state=monitor_state,
            viability_report=report,
            scope=selected.scope,
            planner_name=selected.planner_name,
            planner_fidelity=getattr(getattr(selected, "fidelity", None), "value", selected.fidelity),
            planning_budget=selected.budget.to_dict(),
            voc=voc,
            planner_result=result,
            delay=delay,
            stale_plan_rejection=not accepted,
            metadata={"projected_configuration": projected.to_dict() if projected else None},
        )
        self._record_decision(decision)
        return decision

    def select_scope_planner_budget(self, monitor_state: MonitorState, report: ViabilityReport) -> tuple[PlannerState, list[PlannerCandidate]]:
        try:
            planner_state = self.backend.get_planner_state(self.context, None, None)
        except TypeError:
            planner_state = self.backend.get_planner_state()
        scopes = self.scope_generator.generate(self.current_configuration, monitor_state, planner_state, report)
        if self.config.ablations.global_only_intervention:
            global_scope = self.scope_generator._global_scope(self.current_configuration, planner_state)
            scopes = [global_scope] if not global_scope.is_empty else scopes
        if self.config.ablations.fixed_intervention_scope is not None:
            scopes = [self.config.ablations.fixed_intervention_scope]
        if not scopes:
            scopes = [self._global_scope(planner_state)]

        candidates: list[PlannerCandidate] = []
        hold_cost = self._hold_cost(report)
        improvement = max(0.0, -min(report.service_margin, report.contact_margin, report.deadline_margin))
        improvement += report.performance_risk
        for backend in self.planners.values():
            fidelity = coerce_fidelity(getattr(backend, "fidelity", "light"))
            if self.config.planner.fidelity_levels and fidelity.value not in {str(v).lower() for v in self.config.planner.fidelity_levels}:
                continue
            budget = self._budget_for(backend)
            for scope in scopes:
                cost = backend.estimate_decision_cost(planner_state, self.current_configuration, scope, budget)
                self._price_cost(cost)
                # Higher fidelity can produce a larger expected improvement, but
                # the arbitrator may reject it when its real cost dominates.
                fidelity_multiplier = {"light": 0.75, "medium": 1.0, "high": 1.25}.get(fidelity.value, 1.0)
                candidates.append(
                    PlannerCandidate(
                        scope=scope,
                        fidelity=fidelity,
                        budget=budget,
                        planner_name=str(getattr(backend, "name", "unknown")),
                        estimated_improvement=improvement * fidelity_multiplier,
                        estimated_hold_cost=hold_cost,
                        estimated_candidate_cost=cost,
                        metadata={
                            "planner_family": getattr(backend, "family", "unknown"),
                            "scope_execution_restricted": bool(getattr(backend.capabilities(), "supports_scope_restriction", False)),
                            "scope_computation_restricted": bool(getattr(backend.capabilities(), "supports_scope_restriction", False)),
                        },
                    )
                )
        return planner_state, candidates

    def execute_intervention(
        self,
        planner_state: PlannerState,
        backend: PlannerBackend,
        scope: ReconfigurationScope,
        budget: Any,
        *,
        report: ViabilityReport | None = None,
    ) -> tuple[PlannerResult | None, DecisionDelayBreakdown, PersistentConfiguration | None, bool]:
        previous_configuration = self.current_configuration
        result = backend.plan(planner_state, self.current_configuration, scope, budget)
        result.metadata["planner_state_bytes"] = int(getattr(getattr(planner_state, "acquisition", None), "obs_bytes", 0))
        cost = result.decision_cost if isinstance(result.decision_cost, DecisionCostBreakdown) else DecisionCostBreakdown()
        self._price_cost(cost)
        delay = self.delay_model.estimate(cost)
        planned_at = self.clock.physical_time_sec
        if self.config.ablations.solver_latency_only or self.config.ablations.reward_penalty_delay:
            # These are explicit baselines: they model a solver/penalty signal
            # without advancing the physical world during the decision.
            delay.physical_delay_enforced = False
            delay.metadata["physical_evolution_suppressed_by_ablation"] = True
        else:
            self.delay_model.enforce(self.backend, delay)
        self._sync_clock()
        applied_at = self.clock.physical_time_sec
        projected = self._project_configuration(self.current_configuration, result.configuration, scope)
        validation = self.revalidator.revalidate(self.backend, projected, planned_at=planned_at, applied_at=applied_at)
        if not validation.accepted:
            self.metrics.stale_plan_rejection += 1
            return result, delay, projected, False
        projected.applied_at_sim_time = applied_at
        projected.last_validated_at_sim_time = applied_at
        projected.scope_from_previous = scope
        self._apply_backend_configuration(projected)
        self.current_configuration = projected
        self._last_intervention_time = applied_at
        if previous_configuration is not None:
            self.metrics.add_lifetime(max(0.0, applied_at - float(previous_configuration.applied_at_sim_time)))
        self.clock.mark_intervention()
        self._update_context()
        self.resource_budget.update(cost)
        return result, delay, projected, True

    def dispatch_under_current_configuration(self, task: Any | None = None, *, monitor_state: MonitorState | None = None) -> Any:
        if self.current_configuration is None:
            self.initialize(initial_plan=True)
        result = self.backend.dispatch_under_configuration(self.current_configuration, task)
        monitor = monitor_state or self._last_monitor
        self.metrics.record(
            self._record_payload(
                "DISPATCH",
                monitor,
                self._last_viability,
                metadata={"execution_dispatch": True, "replanning": False},
            )
        )
        return result

    def step(self, **kwargs: Any) -> ControlDecision:
        return self.on_monitor_epoch(**kwargs)

    def run(self, num_monitor_epochs: int, *, dispatch_tasks: Iterable[Any] | None = None) -> list[ControlDecision]:
        tasks = iter(dispatch_tasks or ())
        decisions: list[ControlDecision] = []
        for _ in range(max(0, int(num_monitor_epochs))):
            task = next(tasks, None)
            decisions.append(self.on_monitor_epoch(dispatch_task=task))
        return decisions

    def smdp_transition(self, state: Any, action: Any, reward: float, next_state: Any, *, start_time: float, end_time: float) -> SMDPTransition:
        return SMDPTransition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            holding_time=max(0.0, float(end_time) - float(start_time)),
            start_time=float(start_time),
            end_time=float(end_time),
            physical_slot=self.clock.physical_slot,
            monitor_epoch=self.clock.monitor_epoch,
            intervention_epoch=self.clock.intervention_epoch,
        )

    def resolved_metadata(self) -> dict[str, Any]:
        capabilities = getattr(self.backend, "capabilities", None)
        if isinstance(capabilities, BackendCapabilities):
            capability_payload = capabilities.to_dict()
        elif hasattr(capabilities, "to_dict"):
            capability_payload = capabilities.to_dict()
        else:
            capability_payload = {}
        return {
            "resolved_control_config": self.config.resolved_dict(),
            "backend_source": capability_payload.get("backend_source", "unknown"),
            "topology_source": capability_payload.get("topology_source", "unknown"),
            "physical_delay_enforced": bool(self.context.physical_delay_enforced),
            "monitor_state_source": capability_payload.get("monitor_state_source", "unknown"),
            "oracle_evaluation_only": True,
            "planner_registry": self.planners.metadata(),
        }

    def _acquire_monitor_state(self) -> MonitorState:
        try:
            monitor = self.backend.get_monitor_state(self.context)
        except TypeError:
            monitor = self.backend.get_monitor_state()
        if not isinstance(monitor, MonitorState):
            raise TypeError(f"backend.get_monitor_state must return MonitorState, got {type(monitor)!r}")
        if self.config.monitor.true_cheap_required and not monitor.acquisition.is_true_cheap_monitor:
            if self.config.monitor.fallback_mode == "error":
                raise RuntimeError("Configured true_cheap_required but backend returned a compatibility monitor")
        self._sync_clock(monitor.simulation_time)
        return monitor

    def _ablation_forces_intervention(self, monitor: MonitorState) -> bool:
        if self.config.ablations.fixed_period_replanning:
            period = max(1, int(self.config.ablations.fixed_period))
            return self.clock.monitor_epoch % period == 0
        if self.config.ablations.state_change_trigger:
            return bool((monitor.metadata or {}).get("state_changed", False))
        return False

    def _record_initial_plan(self, result: PlannerResult) -> None:
        self.metrics.record(
            self._record_payload(
                "INITIAL_PLAN",
                None,
                None,
                planner_result=result,
                metadata={"initial_plan": True, "configuration_changed": True},
            )
        )

    def _record_decision(self, decision: ControlDecision) -> None:
        self.metrics.record(
            self._record_payload(
                decision.action,
                decision.monitor_state,
                decision.viability_report,
                planner_result=decision.planner_result,
                decision=decision,
                metadata=decision.metadata,
            )
        )

    def _record_payload(
        self,
        action: str,
        monitor: MonitorState | None,
        report: ViabilityReport | None,
        *,
        planner_result: PlannerResult | None = None,
        decision: ControlDecision | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.current_configuration
        cost = planner_result.decision_cost if planner_result is not None and isinstance(planner_result.decision_cost, DecisionCostBreakdown) else DecisionCostBreakdown()
        scope = decision.scope if decision else ReconfigurationScope()
        payload = {
            "control_action": action,
            "physical_time_sec": self.clock.physical_time_sec,
            "physical_slot": self.clock.physical_slot,
            "monitor_epoch": self.clock.monitor_epoch,
            "intervention_epoch": self.clock.intervention_epoch,
            "config_id": getattr(config, "config_id", None),
            "config_version": getattr(config, "version", None),
            "configuration_age_sec": self.clock.configuration_age_sec,
            "holding_time_since_last_intervention": self.clock.time_since_last_intervention_sec,
            "viability_status": report.viability_status if report else "UNKNOWN",
            "performance_risk": report.performance_risk if report else 0.0,
            "service_margin": report.service_margin if report else 0.0,
            "contact_margin": report.contact_margin if report else 0.0,
            "deadline_margin": report.deadline_margin if report else 0.0,
            "uncertainty_margin": report.uncertainty_margin if report else 0.0,
            "viability_reason": ";".join(report.reason_codes) if report else "",
            "intervention_required": bool(report.needs_intervention) if report else False,
            "scope_cardinality": scope.cardinality,
            "scope_normalized_volume": scope.normalized_volume(),
            "affected_task_count": len(scope.task_ids),
            "affected_node_count": len(scope.node_ids),
            "affected_link_count": len(scope.link_ids),
            "affected_resource_count": len(scope.resource_keys),
            "planner_name": getattr(planner_result, "planner_name", "none"),
            "planner_family": getattr(planner_result, "planner_family", "none"),
            "planner_fidelity": getattr(getattr(planner_result, "fidelity", None), "value", "none"),
            "planning_budget": getattr(getattr(planner_result, "budget", None), "to_dict", lambda: {})(),
            "VoC": decision.voc.value if decision and decision.voc else 0.0,
            "estimated_hold_cost": decision.voc.estimated_hold_cost if decision and decision.voc else 0.0,
            "estimated_candidate_cost": decision.voc.estimated_candidate_cost if decision and decision.voc else 0.0,
            "obs_cost": cost.obs_cost,
            "sync_cost": cost.sync_cost,
            "solve_cost": cost.solve_cost,
            "signal_cost": cost.signal_cost,
            "decision_cost": cost.decision_cost,
            "reconfiguration_cost": cost.reconfiguration_cost,
            "total_intervention_cost": cost.intervention_cost,
            "solver_wallclock_sec": cost.solver_wallclock_sec,
            "physical_delay_enforced": bool(self.context.physical_delay_enforced),
            "monitor_is_true_cheap": bool(monitor.acquisition.is_true_cheap_monitor) if monitor else False,
            "monitor_bytes": int(monitor.acquisition.obs_bytes) if monitor else 0,
            "planner_state_bytes": int((getattr(planner_result, "metadata", {}) or {}).get("planner_state_bytes", 0)) if planner_result else 0,
            "control_plane_bytes": int(cost.observation_bytes + cost.sync_bytes + cost.signal_bytes),
            "reconfiguration_volume": scope.normalized_volume(),
            "num_dispatches": self.metrics.num_dispatches,
            "num_replans": self.metrics.num_replans,
            "num_configuration_changes": self.metrics.num_configuration_changes,
            "stale_plan_rejection": bool(decision.stale_plan_rejection) if decision else False,
            "backend_source": self.context.backend_source,
            "topology_source": self.context.topology_source,
            "oracle_evaluation_only": True,
            "metadata": dict(metadata or {}),
        }
        if decision and decision.delay is not None:
            payload["modeled_decision_delay_sec"] = decision.delay.total_delay_sec
            payload["physical_delay_enforced"] = decision.delay.physical_delay_enforced
        return payload

    def _make_empty_configuration(self) -> PersistentConfiguration:
        return PersistentConfiguration(config_id="config-0", version=0, created_at_sim_time=self.clock.physical_time_sec)

    def _select_backend(self, name: str | None = None) -> PlannerBackend:
        if name:
            return self.planners.get(name)
        enabled = self.config.planner.enabled_backends
        for backend in self.planners.values():
            if not enabled or getattr(backend, "name", "") in enabled:
                return backend
        return self.planners.values()[0]

    def _budget_for(self, backend: PlannerBackend) -> Any:
        fidelity = getattr(getattr(backend, "fidelity", None), "value", getattr(backend, "fidelity", "light"))
        return budget_from_mapping(self.config.planner.budget_levels.get(str(fidelity), {}))

    @staticmethod
    def _global_scope(planner_state: PlannerState) -> ReconfigurationScope:
        nodes = set(str(key) for key in (planner_state.detailed_resources or {}))
        sources = set(str(item.get("sourceId", item.get("source_id", ""))) for item in planner_state.candidate_vms if isinstance(item, Mapping))
        sources.discard("")
        return ReconfigurationScope(node_ids=nodes, source_ids=sources)

    @staticmethod
    def _project_configuration(current: PersistentConfiguration, proposed: PersistentConfiguration, scope: ReconfigurationScope) -> PersistentConfiguration:
        result = current.clone(
            config_id=proposed.config_id,
            version=proposed.version,
        )
        for name in ("assignments", "resource_allocations", "routes"):
            old = dict(getattr(current, name) or {})
            new = dict(getattr(proposed, name) or {})
            merged = dict(old)
            for key, value in new.items():
                if scope.contains(key):
                    merged[key] = value
            setattr(result, name, merged)
        result.planner_name = proposed.planner_name
        result.planner_fidelity = proposed.planner_fidelity
        result.planning_budget = dict(proposed.planning_budget)
        result.metadata = {**current.metadata, **proposed.metadata, "scope_execution_restricted": True}
        return result

    def _apply_backend_configuration(self, configuration: PersistentConfiguration) -> Any:
        if not hasattr(self.backend, "apply_configuration"):
            raise RuntimeError("Physical backend does not expose apply_configuration")
        result = self.backend.apply_configuration(configuration)
        self.context.current_config_id = configuration.config_id
        self.context.current_config_version = configuration.version
        return result

    def _price_cost(self, cost: DecisionCostBreakdown) -> None:
        cost.obs_price = self.config.decision_cost.obs_price
        cost.sync_price = self.config.decision_cost.sync_price
        cost.solve_price = self.config.decision_cost.solve_price
        cost.signal_price = self.config.decision_cost.signal_price
        cost.reconfiguration_price = self.config.decision_cost.reconfiguration_price

    def _hold_cost(self, report: ViabilityReport) -> float:
        negative_margin = max(0.0, -min(report.service_margin, report.contact_margin, report.deadline_margin))
        return self.config.planner.hold_cost_weight * (negative_margin + report.performance_risk)

    def _resolved_delay_mode(self) -> str:
        if self.config.ablations.solver_latency_only:
            return "modeled"
        if self.config.ablations.reward_penalty_delay:
            return "reward_penalty"
        return self.config.decision_delay.mode

    def _sync_clock(self, observed_time: float | None = None) -> None:
        if observed_time is None and hasattr(self.backend, "current_time"):
            try:
                observed_time = float(self.backend.current_time())
            except Exception:  # noqa: BLE001
                observed_time = self.clock.physical_time_sec
        if observed_time is None:
            return
        observed_time = float(observed_time)
        delta = observed_time - self.clock.physical_time_sec
        if delta > 0.0:
            self.clock.advance(delta, physical_slot_delta=max(0, int(round(delta))))

    def _update_context(self) -> None:
        capabilities = getattr(self.backend, "capabilities", None)
        self.context.clock = self.clock
        self.context.current_config_id = getattr(self.current_configuration, "config_id", None)
        self.context.current_config_version = getattr(self.current_configuration, "version", None)
        self.context.backend_source = getattr(capabilities, "backend_source", "unknown")
        self.context.topology_source = getattr(capabilities, "topology_source", "unknown")
        self.context.physical_delay_enforced = bool(getattr(capabilities, "supports_physical_decision_delay", False))
