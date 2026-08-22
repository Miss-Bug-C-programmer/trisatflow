from __future__ import annotations

from dataclasses import dataclass

import pytest

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.adapters.legacy_env_backend import LegacyEnvBackendAdapter
from trisatflow.control.arbitration import PlannerArbitrator, PlannerCandidate
from trisatflow.control.config import ControllerConfig
from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.decision_delay import DecisionDelayBreakdown, DecisionDelayModel, PostDelayRevalidator, RevalidationResult
from trisatflow.control.controller import EndogenousReplanningController
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import MonitorAcquisitionMetadata, MonitorState, PlannerCapabilities, PlannerFidelity, PlannerResult, PlannerState, PlanningBudget
from trisatflow.control.viability import ConservativeViabilityEstimator, FeasibilityStatus
from trisatflow.planners.greedy_planner import GreedyPlanner


class FakeBackend:
    def __init__(self, *, physical_delay: bool = True) -> None:
        self.time = 0.0
        self.monitor_calls = 0
        self.planner_state_calls = 0
        self.applied: list[tuple[float, PersistentConfiguration]] = []
        self.dispatches = 0
        self.stale = False
        self.capabilities = BackendCapabilities(
            supports_monitor_state=True,
            supports_planner_state=True,
            supports_configuration_apply=True,
            supports_persistent_configuration=True,
            supports_physical_decision_delay=physical_delay,
            supports_advance_world=physical_delay,
            backend_source="fake",
            topology_source="fake_topology",
            monitor_state_source="fake_monitor",
            authoritative_physical=False,
        )

    def current_time(self) -> float:
        return self.time

    def get_monitor_state(self, context=None) -> MonitorState:
        self.monitor_calls += 1
        degraded = self.monitor_calls >= 3
        return MonitorState(
            simulation_time=self.time,
            current_config_id="cfg",
            current_config_version=0,
            source_queue_summary={"s1": 1.0 if not degraded else 9.0},
            remaining_workload_summary={"s1": 1.0 if not degraded else 9.0},
            deadline_slack={"s1": 10.0 if not degraded else -1.0},
            local_load_summary={"service": 10.0},
            service_rate_lower_bound=10.0,
            service_bound_certified=True,
            service_horizon_sec=10.0,
            contact_slack={"l1": 10.0 if not degraded else -1.0},
            prediction_uncertainty={"service": 0.0},
            uncertainty_evidence_available=True,
            degradation_indicators={"risk": 0.0 if not degraded else 0.9},
            acquisition=MonitorAcquisitionMetadata(obs_bytes=10, num_queries=1, source="fake", is_true_cheap_monitor=True),
            metadata={"affected_entities": {"source_ids": {"s1"}}, "future_queue_truth": "must_not_be_consumed"},
        )

    def get_planner_state(self, context=None, scope=None, budget=None) -> PlannerState:
        self.planner_state_calls += 1
        return PlannerState(
            simulation_time=self.time,
            candidate_vms=[
                {"sourceId": "s1", "vmIndex": 1, "estimatedTotalDelaySec": 1.0, "resourceAllocation": {"cpu": 1.0}},
                {"sourceId": "s2", "vmIndex": 2, "estimatedTotalDelaySec": 2.0, "resourceAllocation": {"cpu": 2.0}},
            ],
            detailed_resources={"n1": {"cpu": 1.0}, "n2": {"cpu": 2.0}},
        )

    def apply_configuration(self, configuration):
        self.applied.append((self.time, configuration.clone()))
        return {"accepted": True}

    def dispatch_under_configuration(self, configuration, task=None):
        self.dispatches += 1
        return configuration.materialize_execution_rule(task or {"task_id": "s1"})

    def advance_world(self, delta_sec: float):
        self.time += float(delta_sec)

    def validate_configuration(self, configuration):
        return RevalidationResult(not self.stale, ["target_unavailable"] if self.stale else [])


class DelayedPlanner(GreedyPlanner):
    name = "delayed_greedy"
    fidelity = PlannerFidelity.HIGH

    def estimate_decision_cost(self, planner_state, current_config, scope, budget):
        return DecisionCostBreakdown(solver_simulated_latency_sec=2.0, solver_compute_proxy=1.0)


def config() -> PersistentConfiguration:
    return PersistentConfiguration(
        config_id="cfg",
        version=0,
        assignments={"s1": {"target": "old"}, "s2": {"target": "old2"}},
        covered_source_ids={"s1", "s2"},
    )


def test_destination_only_candidates_install_a_reusable_default_rule() -> None:
    planner = GreedyPlanner(source_name="satedgesim")
    current = PersistentConfiguration(config_id="cfg", version=0)
    state = PlannerState(
        simulation_time=0.0,
        candidate_vms=[
            {"datacenterDeviceId": 7, "vmIndex": 7, "estimatedTotalDelaySec": 0.01, "isFeasible": False},
            {"datacenterDeviceId": 8, "vmIndex": 8, "estimatedTotalDelaySec": 0.20, "isFeasible": True},
            {"datacenterDeviceId": 9, "vmIndex": 9, "estimatedTotalDelaySec": 0.05, "isFeasible": True},
        ],
    )
    result = planner.plan(state, current, ReconfigurationScope(node_ids={"7", "8", "9"}), PlanningBudget())

    assert set(result.configuration.assignments) == {"default"}
    assert set(result.configuration.reusable_rules) == {"default"}
    assert result.configuration.materialize_execution_rule({"task_id": "future", "source_id": "new-source"})["vmIndex"] == 9


def test_persistent_configuration_survives_physical_slots_and_keep_does_not_plan():
    backend = FakeBackend()
    controller = EndogenousReplanningController(backend, config={"planner": {"enabled_backends": []}})
    controller.initialize(config(), initial_plan=False)
    controller.on_monitor_epoch()
    backend.time += 1.0
    controller.on_monitor_epoch()
    assert controller.current_configuration.config_id == "cfg"
    assert controller.current_configuration.version == 0
    assert backend.planner_state_calls == 0


def test_true_cheap_keep_path_never_acquires_planner_state():
    backend = FakeBackend()
    controller = EndogenousReplanningController(backend)
    controller.initialize(config(), initial_plan=False)
    decision = controller.on_monitor_epoch()
    assert decision.action == "KEEP"
    assert backend.planner_state_calls == 0
    assert decision.monitor_state.acquisition.is_true_cheap_monitor


def test_scope_empty_is_keep_and_scope_projection_freezes_outside_entries():
    assert ReconfigurationScope().is_empty
    current = config()
    proposed = current.apply_patch({"assignments": {"s1": {"target": "new"}, "s2": {"target": "bad"}}})
    projected = EndogenousReplanningController._project_configuration(current, proposed, ReconfigurationScope(source_ids={"s1"}))
    assert projected.assignments["s1"]["target"] == "new"
    assert projected.assignments["s2"]["target"] == "old2"


def test_feasibility_and_performance_viability_are_separate():
    estimator = ConservativeViabilityEstimator(performance_risk_threshold=0.5)
    monitor = MonitorState(
        remaining_workload_summary={"s": 1.0},
        local_load_summary={"service": 10.0},
        service_rate_lower_bound=10.0,
        service_bound_certified=True,
        service_horizon_sec=10.0,
        deadline_slack={"s": 10.0},
        contact_slack={"l": 10.0},
        prediction_uncertainty={"service": 0.0},
        uncertainty_evidence_available=True,
        degradation_indicators={"risk": 0.0},
        acquisition=MonitorAcquisitionMetadata(is_true_cheap_monitor=True),
    )
    safe = estimator.evaluate(monitor, config())
    assert safe.feasibility_status == FeasibilityStatus.VIABLE
    assert not safe.needs_intervention
    monitor.degradation_indicators = {"risk": 0.9}
    risky = estimator.evaluate(monitor, config())
    assert risky.feasibility_status == FeasibilityStatus.VIABLE
    assert risky.needs_intervention


def test_contact_or_deadline_margin_triggers_intervention():
    estimator = ConservativeViabilityEstimator()
    monitor = MonitorState(
        remaining_workload_summary={"s": 1.0},
        local_load_summary={"service": 10.0},
        service_rate_lower_bound=10.0,
        service_bound_certified=True,
        service_horizon_sec=10.0,
        deadline_slack={"s": -0.1},
        contact_slack={"l": 10.0},
        prediction_uncertainty={"service": 0.0},
        uncertainty_evidence_available=True,
        acquisition=MonitorAcquisitionMetadata(is_true_cheap_monitor=True),
    )
    report = estimator.evaluate(monitor, config())
    assert report.needs_intervention
    assert "deadline_margin_negative" in report.reason_codes


def test_decision_cost_has_no_reconfiguration_double_count():
    cost = DecisionCostBreakdown(
        observation_bytes=1,
        sync_bytes=1,
        solver_compute_proxy=1,
        signal_bytes=1,
        migration_volume=1,
    )
    assert cost.decision_cost == pytest.approx(4.0)
    assert cost.intervention_cost == pytest.approx(5.0)


def test_voc_can_reject_expensive_candidate():
    candidate = PlannerCandidate(
        scope=ReconfigurationScope(source_ids={"s"}),
        fidelity=PlannerFidelity.HIGH,
        budget=PlanningBudget(),
        planner_name="expensive",
        estimated_improvement=1.0,
        estimated_candidate_cost=DecisionCostBreakdown(solver_compute_proxy=100.0),
    )
    voc = PlannerArbitrator().choose([candidate], hold_cost=0.0)
    assert voc.keep
    assert voc.value < 0.0


def test_physical_delay_is_explicit_and_world_advances_when_supported():
    backend = FakeBackend(physical_delay=True)
    model = DecisionDelayModel(mode="modeled", modeled_components=("solver",))
    delay = model.estimate(DecisionCostBreakdown(solver_simulated_latency_sec=2.0))
    model.enforce(backend, delay)
    assert delay.total_delay_sec == pytest.approx(2.0)
    assert delay.physical_delay_enforced is True
    assert backend.time == pytest.approx(2.0)

    unsupported = FakeBackend(physical_delay=False)
    delay2 = model.enforce(unsupported, DecisionDelayBreakdown(total_delay_sec=2.0))
    assert delay2.physical_delay_enforced is False
    assert unsupported.time == 0.0


def test_controller_executes_delay_and_keeps_old_configuration_during_delay():
    backend = FakeBackend(physical_delay=True)
    planner = DelayedPlanner()
    cfg = ControllerConfig.from_mapping(
        {
            "decision_delay": {"mode": "modeled", "modeled_components": ["solver"]},
            "decision_cost": {"obs_price": 0.0, "sync_price": 0.0, "solve_price": 0.0, "signal_price": 0.0, "reconfiguration_price": 0.0},
        }
    )
    controller = EndogenousReplanningController(backend, config=cfg, planner_backends=[planner])
    controller.initialize(config(), initial_plan=False)
    decision = controller.on_monitor_epoch()
    # First monitor is safe; force a second monitor degradation to trigger.
    decision = controller.on_monitor_epoch()
    assert decision.action == "KEEP"
    decision = controller.on_monitor_epoch()
    assert decision.action == "INTERVENE"
    assert decision.delay is not None and decision.delay.physical_delay_enforced
    assert backend.time == pytest.approx(2.0)
    assert backend.applied[-1][0] == pytest.approx(2.0)


def test_post_delay_stale_plan_is_rejected():
    backend = FakeBackend()
    backend.stale = True
    result = PostDelayRevalidator().revalidate(backend, config(), planned_at=0.0, applied_at=1.0)
    assert not result.accepted
    assert "target_unavailable" in result.reason_codes


def test_smdp_transition_records_holding_time():
    backend = FakeBackend()
    controller = EndogenousReplanningController(backend)
    transition = controller.smdp_transition("s0", "u", 1.0, "s1", start_time=1.0, end_time=3.5)
    assert transition.holding_time == pytest.approx(2.5)
    assert transition.effective_discount(0.9) == pytest.approx(0.9**2.5)


def test_monitor_state_does_not_contain_future_stochastic_truth():
    backend = FakeBackend()
    state = backend.get_monitor_state()
    assert not hasattr(state, "future_queue")
    assert not hasattr(state, "future_channel")
    assert state.metadata["future_queue_truth"] == "must_not_be_consumed"


def test_legacy_backend_is_explicitly_non_authoritative():
    class Env:
        t = 0
        queue = [0.0]
        class cfg:
            n_leo = 1
            max_queue = 10.0
            deadline_threshold = 8.0
            leo_cpu_capacity = 5.0
        def reset(self):
            return None

    backend = LegacyEnvBackendAdapter(Env())
    assert backend.capabilities.authoritative_physical is False
    assert backend.capabilities.backend_source == "legacy_env"
    assert backend.get_monitor_state().acquisition.is_true_cheap_monitor
