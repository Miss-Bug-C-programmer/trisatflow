"""Causal, delay-aware benefit estimation for planner arbitration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from trisatflow.control.decision_delay import DecisionDelayBreakdown
from trisatflow.control.types import MonitorState, PlannerFidelity, PlannerState, PlanningBudget, PlanningDescriptor


@dataclass
class OutcomeEstimate:
    expected_task_cost: float = 0.0
    expected_delay: float = 0.0
    expected_energy: float = 0.0
    deadline_violation_risk: float = 0.0
    queue_cost: float = 0.0
    configuration_risk: float = 0.0
    prediction_uncertainty: float = 0.0
    confidence: float = 0.0
    forecast_start_time: float = 0.0
    forecast_end_time: float = 0.0
    estimator_source: str = "unknown"
    future_stochastic_truth_used: bool = False
    objective_units: str = "normalized_data_plane_cost"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(
            self.expected_task_cost
            + self.expected_delay
            + self.expected_energy
            + self.deadline_violation_risk
            + self.queue_cost
            + self.configuration_risk
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_cost"] = self.total_cost
        return payload


CostToGoEstimate = OutcomeEstimate


@dataclass
class BenefitEstimate:
    hold_cost: float = 0.0
    candidate_total_cost: float = 0.0
    candidate_cost_after_delay: float = 0.0
    cost_accumulated_during_decision_delay: float = 0.0
    gross_benefit: float = 0.0
    benefit_uncertainty: float = 0.0
    lower_confidence_benefit: float = 0.0
    evaluation_start_time: float = 0.0
    evaluation_end_time: float = 0.0
    decision_delay_sec: float = 0.0
    score_mode: str = "mean"
    lcb_beta: float = 0.0
    estimator_source: str = "unknown"
    future_stochastic_truth_used: bool = False
    hold_outcome: OutcomeEstimate | None = None
    candidate_outcome: OutcomeEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_adjusted_benefit(self) -> float:
        return self.lower_confidence_benefit if self.score_mode == "lower_confidence_bound" else self.gross_benefit

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_adjusted_benefit"] = self.risk_adjusted_benefit
        if self.hold_outcome is not None:
            payload["hold_outcome"] = self.hold_outcome.to_dict()
        if self.candidate_outcome is not None:
            payload["candidate_outcome"] = self.candidate_outcome.to_dict()
        return payload


class BenefitEstimator(Protocol):
    def estimate_hold(
        self,
        monitor_state: MonitorState,
        current_configuration: Any,
        evaluation_horizon: float,
        *,
        forecast_start_time: float | None = None,
    ) -> OutcomeEstimate:
        ...

    def estimate_candidate(
        self,
        monitor_state: MonitorState,
        planner_descriptor: PlanningDescriptor,
        current_configuration: Any,
        scope: Any,
        fidelity: PlannerFidelity,
        budget: PlanningBudget,
        decision_delay: DecisionDelayBreakdown,
        evaluation_horizon: float,
        *,
        forecast_start_time: float | None = None,
    ) -> BenefitEstimate:
        ...


class ConservativeAnalyticalBenefitEstimator:
    """Executable causal estimator using current summaries and cached forecasts.

    This is an implementation realization, not the paper abstraction itself.
    It never consumes future realized queues, arrivals, channels or remote load.
    """

    def __init__(
        self,
        *,
        score_mode: str = "mean",
        lcb_beta: float = 1.0,
        objective_weights: Mapping[str, float] | None = None,
    ) -> None:
        mode = str(score_mode).strip().lower()
        if mode not in {"mean", "lower_confidence_bound", "lcb"}:
            raise ValueError("score_mode must be 'mean' or 'lower_confidence_bound'")
        self.score_mode = "lower_confidence_bound" if mode == "lcb" else mode
        self.lcb_beta = float(lcb_beta)
        self.objective_weights = {
            "task": 1.0,
            "delay": 1.0,
            "energy": 1.0,
            "deadline": 1.0,
            "queue": 1.0,
            "configuration": 1.0,
            **dict(objective_weights or {}),
        }

    def estimate_hold(
        self,
        monitor_state: MonitorState,
        current_configuration: Any,
        evaluation_horizon: float,
        *,
        forecast_start_time: float | None = None,
    ) -> OutcomeEstimate:
        start = float(monitor_state.simulation_time if forecast_start_time is None else forecast_start_time)
        horizon = max(0.0, float(evaluation_horizon))
        workload = _sum_numeric(monitor_state.remaining_workload_summary)
        queue = _sum_numeric(monitor_state.source_queue_summary or monitor_state.local_queue_summary)
        rate = _service_rate(monitor_state)
        expected_delay = workload / max(rate, 1.0e-9)
        deadline_risk = _deadline_risk(monitor_state)
        config_risk = max(_max_numeric(monitor_state.degradation_indicators), _max_numeric(monitor_state.prediction_uncertainty))
        uncertainty = max(
            _max_numeric(monitor_state.prediction_uncertainty),
            0.5 if not monitor_state.acquisition.is_true_cheap_monitor else 0.0,
        )
        energy = float((monitor_state.metadata or {}).get("expected_energy_rate", 0.0)) * horizon
        total = self._objective(task=workload, delay=expected_delay, energy=energy, deadline=deadline_risk, queue=queue, configuration=config_risk)
        return OutcomeEstimate(
            expected_task_cost=workload * self.objective_weights["task"],
            expected_delay=expected_delay * self.objective_weights["delay"],
            expected_energy=energy * self.objective_weights["energy"],
            deadline_violation_risk=deadline_risk * self.objective_weights["deadline"],
            queue_cost=queue * self.objective_weights["queue"],
            configuration_risk=config_risk * self.objective_weights["configuration"],
            prediction_uncertainty=uncertainty,
            confidence=max(0.0, min(1.0, 0.9 if monitor_state.acquisition.is_true_cheap_monitor else 0.5)),
            forecast_start_time=start,
            forecast_end_time=start + horizon,
            estimator_source="conservative_analytical",
            future_stochastic_truth_used=False,
            metadata={"horizon_sec": horizon, "service_rate_lower_bound": rate, "objective_total": total},
        )

    def estimate_candidate(
        self,
        monitor_state: MonitorState,
        planner_descriptor: PlanningDescriptor,
        current_configuration: Any,
        scope: Any,
        fidelity: PlannerFidelity,
        budget: PlanningBudget,
        decision_delay: DecisionDelayBreakdown,
        evaluation_horizon: float,
        *,
        forecast_start_time: float | None = None,
    ) -> BenefitEstimate:
        start = float(monitor_state.simulation_time if forecast_start_time is None else forecast_start_time)
        horizon = max(0.0, float(evaluation_horizon))
        end = start + horizon
        hold = self.estimate_hold(monitor_state, current_configuration, horizon, forecast_start_time=start)
        delay = max(0.0, min(float(decision_delay.total_delay_sec), horizon))

        descriptor_benefit = float(planner_descriptor.expected_benefit_mean)
        if descriptor_benefit == 0.0:
            # A descriptor may have no learned/historical benefit. In that case
            # the conservative default is no improvement, never a free fidelity
            # multiplier or a full planner look-ahead.
            descriptor_benefit = 0.0
        candidate_uncertainty = max(0.0, float(planner_descriptor.expected_benefit_uncertainty))
        candidate_base = max(0.0, hold.total_cost - descriptor_benefit)
        delay_cost = hold.total_cost * (delay / horizon) if horizon > 0.0 else 0.0
        after_delay = candidate_base * ((horizon - delay) / horizon) if horizon > 0.0 else 0.0
        candidate_total = delay_cost + after_delay
        gross = hold.total_cost - candidate_total
        uncertainty = hold.prediction_uncertainty + candidate_uncertainty
        lcb = gross - self.lcb_beta * uncertainty
        candidate_outcome = OutcomeEstimate(
            expected_task_cost=after_delay,
            expected_delay=delay,
            expected_energy=0.0,
            deadline_violation_risk=0.0,
            queue_cost=0.0,
            configuration_risk=candidate_uncertainty,
            prediction_uncertainty=candidate_uncertainty,
            confidence=max(0.0, 1.0 - min(1.0, candidate_uncertainty)),
            forecast_start_time=start + delay,
            forecast_end_time=end,
            estimator_source="conservative_analytical",
            future_stochastic_truth_used=False,
            metadata={"descriptor": planner_descriptor.to_dict(), "fidelity_is_implementation_label": True},
        )
        return BenefitEstimate(
            hold_cost=hold.total_cost,
            candidate_total_cost=candidate_total,
            candidate_cost_after_delay=after_delay,
            cost_accumulated_during_decision_delay=delay_cost,
            gross_benefit=gross,
            benefit_uncertainty=uncertainty,
            lower_confidence_benefit=lcb,
            evaluation_start_time=start,
            evaluation_end_time=end,
            decision_delay_sec=delay,
            score_mode=self.score_mode,
            lcb_beta=self.lcb_beta,
            estimator_source="conservative_analytical",
            future_stochastic_truth_used=False,
            hold_outcome=hold,
            candidate_outcome=candidate_outcome,
            metadata={
                "common_absolute_evaluation_end": end,
                "planning_delay_old_configuration_active": True,
                "decision_plane_cost_excluded_from_data_plane_cost": True,
            },
        )

    def _objective(self, *, task: float, delay: float, energy: float, deadline: float, queue: float, configuration: float) -> float:
        return sum(
            value * self.objective_weights[key]
            for key, value in {
                "task": task,
                "delay": delay,
                "energy": energy,
                "deadline": deadline,
                "queue": queue,
                "configuration": configuration,
            }.items()
        )


def _sum_numeric(mapping: Mapping[str, Any] | None) -> float:
    total = 0.0
    for value in (mapping or {}).values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _max_numeric(mapping: Mapping[str, Any] | None) -> float:
    values = []
    for value in (mapping or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _deadline_risk(monitor_state: MonitorState) -> float:
    values = []
    for value in (monitor_state.deadline_slack or {}).values():
        try:
            values.append(1.0 if float(value) < 0.0 else 0.0)
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


def _service_rate(monitor_state: MonitorState) -> float:
    explicit = getattr(monitor_state, "service_rate_lower_bound", None)
    if explicit is not None:
        return max(0.0, float(explicit))
    for key, value in (monitor_state.local_load_summary or {}).items():
        if any(token in str(key).lower() for token in ("service", "capacity", "rate")):
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    return max(0.0, _sum_numeric(monitor_state.local_load_summary))
