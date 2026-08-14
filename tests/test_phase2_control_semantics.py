from __future__ import annotations

import pytest

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.benefit import ConservativeAnalyticalBenefitEstimator
from trisatflow.control.decision_cost import DecisionCostBreakdown, ResourceBudgetState
from trisatflow.control.decision_delay import DecisionDelayModel
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import (
    MonitorAcquisitionMetadata,
    MonitorState,
    PlannerFidelity,
    PlanningBudget,
    PlanningDescriptor,
)


def _monitor() -> MonitorState:
    return MonitorState(
        simulation_time=5.0,
        remaining_workload_summary={"s1": 10.0},
        source_queue_summary={"s1": 4.0},
        local_load_summary={"service_rate": 2.0},
        deadline_slack={"s1": 4.0},
        prediction_uncertainty={"service": 0.2},
        acquisition=MonitorAcquisitionMetadata(is_true_cheap_monitor=True),
        metadata={"future_queue": [9999], "future_stochastic_truth_used": False},
    )


def test_common_horizon_benefit_records_delay_interval_without_future_truth() -> None:
    estimator = ConservativeAnalyticalBenefitEstimator(score_mode="lower_confidence_bound", lcb_beta=0.5)
    descriptor = PlanningDescriptor(
        planner_name="p",
        planner_family="test",
        fidelity=PlannerFidelity.MEDIUM,
        expected_benefit_mean=1.0,
        expected_benefit_uncertainty=0.1,
    )
    benefit = estimator.estimate_candidate(
        _monitor(),
        descriptor,
        PersistentConfiguration(config_id="c"),
        ReconfigurationScope(source_ids={"s1"}),
        PlannerFidelity.MEDIUM,
        PlanningBudget(),
        DecisionDelayModel(mode="modeled").estimate(DecisionCostBreakdown(solver_simulated_latency_sec=2.0)),
        10.0,
    )
    assert benefit.evaluation_start_time == pytest.approx(5.0)
    assert benefit.evaluation_end_time == pytest.approx(15.0)
    assert benefit.candidate_outcome is not None
    assert benefit.candidate_outcome.forecast_start_time == pytest.approx(7.0)
    assert benefit.future_stochastic_truth_used is False
    assert benefit.score_mode == "lower_confidence_bound"


def test_wallclock_is_not_simulated_delay_without_explicit_ablation() -> None:
    cost = DecisionCostBreakdown(solver_wallclock_sec=3.0)
    delay = DecisionDelayModel(mode="modeled").estimate(cost)
    assert delay.wallclock_solver_sec == pytest.approx(3.0)
    assert delay.solver_delay_sec == 0.0


def test_raw_resource_units_are_priced_independently() -> None:
    cost = DecisionCostBreakdown(
        observation_bytes=10,
        observation_latency_sec=2.0,
        observation_energy=3.0,
        observation_byte_price=0.1,
        observation_latency_price=2.0,
        observation_energy_price=4.0,
    )
    assert cost.obs_cost == pytest.approx(17.0)
    assert cost.units["observation_bytes"] == "byte"


def test_configuration_change_counts_are_not_scope_volume() -> None:
    before = PersistentConfiguration(
        config_id="before",
        assignments={"s1": {"target": "a"}, "s2": {"target": "b"}},
        covered_source_ids={"s1", "s2"},
    )
    after = before.clone(config_id="after")
    after.assignments["s1"] = {"target": "c"}
    counts = before.change_counts(after)
    assert counts["num_changed_assignments"] == 1
    assert counts["num_changed_resources"] == 0
    assert counts["migration_volume"] >= 0.0
    assert counts["reconfiguration_bytes"] > 0


def test_resource_dual_uses_current_cost_and_physical_duration() -> None:
    state = ResourceBudgetState(average_control_bytes_budget=1.0, step_size=1.0)
    state.update(DecisionCostBreakdown(signal_bytes=10), holding_time_sec=2.0)
    assert state.last_consumption["bytes_rate"] == pytest.approx(5.0)
    assert state.dual_bytes == pytest.approx(4.0)


def test_unverified_physical_advance_is_rejected_when_strict() -> None:
    class NoAdvance:
        capabilities = BackendCapabilities(
            supports_physical_decision_delay=True,
            supports_advance_world=True,
            supports_verified_delay_receipt=True,
        )

        def current_time(self) -> float:
            return 0.0

        def advance_world(self, delta_sec: float):
            return {"accepted": True}

    delay = DecisionDelayModel(mode="modeled", require_physical_enforcement=True).estimate(
        DecisionCostBreakdown(solver_simulated_latency_sec=1.0)
    )
    with pytest.raises(RuntimeError, match="verifiable physical time"):
        DecisionDelayModel(mode="modeled", require_physical_enforcement=True).enforce(NoAdvance(), delay)
