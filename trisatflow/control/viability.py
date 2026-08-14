"""Configuration feasibility and performance viability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import FeasibilityStatus, MonitorState, mapping_float


@dataclass
class ViabilityReport:
    feasibility_status: FeasibilityStatus = FeasibilityStatus.UNCERTAIN
    performance_risk: float = 0.0
    service_margin: float = 0.0
    contact_margin: float = 0.0
    deadline_margin: float = 0.0
    uncertainty_margin: float = 0.0
    uncertainty_components: dict[str, float] = field(default_factory=dict)
    horizon_sec: float = 0.0
    cumulative_service_lower_bound: float = 0.0
    opportunity_score: float = 0.0
    affected_entities: ReconfigurationScope = field(default_factory=ReconfigurationScope)
    reason_codes: list[str] = field(default_factory=list)
    needs_intervention: bool = False
    confidence: float = 0.0
    evaluated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def worth_keeping(self) -> bool:
        return self.certifies_keep

    @property
    def certifies_keep(self) -> bool:
        """Cheap certification only; final intervention remains VoC's job."""

        return (
            self.feasibility_status == FeasibilityStatus.VIABLE
            and not self.needs_intervention
            and self.confidence > 0.0
        )

    @property
    def needs_escalation(self) -> bool:
        return not self.certifies_keep

    @property
    def viability_status(self) -> str:
        return self.feasibility_status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasibility_status": self.feasibility_status.value,
            "performance_risk": self.performance_risk,
            "service_margin": self.service_margin,
            "contact_margin": self.contact_margin,
            "deadline_margin": self.deadline_margin,
            "uncertainty_margin": self.uncertainty_margin,
            "uncertainty_components": dict(self.uncertainty_components),
            "horizon_sec": self.horizon_sec,
            "cumulative_service_lower_bound": self.cumulative_service_lower_bound,
            "opportunity_score": self.opportunity_score,
            "affected_entities": self.affected_entities.to_dict(),
            "reason_codes": list(self.reason_codes),
            "needs_intervention": self.needs_intervention,
            "confidence": self.confidence,
            "evaluated_at": self.evaluated_at,
            "metadata": dict(self.metadata),
        }


class ViabilityEstimator(Protocol):
    def evaluate(self, monitor_state: MonitorState, current_config: Any, *, planner_state: Any | None = None) -> ViabilityReport:
        ...


def _summary_min(mapping: Mapping[str, Any] | None, default: float) -> float:
    values: list[float] = []
    for value in (mapping or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return min(values) if values else float(default)


def _summary_max(mapping: Mapping[str, Any] | None, default: float) -> float:
    values: list[float] = []
    for value in (mapping or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else float(default)


class ConservativeViabilityEstimator:
    """Deterministic estimator using observable/cached summaries only."""

    def __init__(
        self,
        *,
        uncertainty_margin: float = 0.0,
        feasibility_margin: float = 0.0,
        performance_risk_threshold: float = 0.5,
        contact_predictability: bool = True,
        use_performance_viability: bool = True,
        evaluation_horizon_sec: float = 10.0,
        service_safety_fraction: float = 0.1,
    ) -> None:
        self.uncertainty_margin = float(uncertainty_margin)
        self.feasibility_margin = float(feasibility_margin)
        self.performance_risk_threshold = float(performance_risk_threshold)
        self.contact_predictability = bool(contact_predictability)
        self.use_performance_viability = bool(use_performance_viability)
        self.evaluation_horizon_sec = max(0.0, float(evaluation_horizon_sec))
        self.service_safety_fraction = max(0.0, min(1.0, float(service_safety_fraction)))

    def evaluate(
        self,
        monitor_state: MonitorState,
        current_config: Any,
        *,
        planner_state: Any | None = None,
    ) -> ViabilityReport:
        workload = mapping_float(monitor_state.remaining_workload_summary)
        capacity = mapping_float(
            monitor_state.local_load_summary,
            default=float(monitor_state.metadata.get("lower_bound_service_capacity", 0.0)),
        )
        horizon = max(0.0, float(monitor_state.service_horizon_sec or self.evaluation_horizon_sec))
        explicit_rate = monitor_state.service_rate_lower_bound
        if explicit_rate is None:
            explicit_rate = monitor_state.metadata.get("lower_bound_service_rate", capacity)
        contact_horizon = _summary_min(monitor_state.remaining_contact_lifetime, default=horizon)
        effective_horizon = min(horizon, max(0.0, contact_horizon)) if contact_horizon != float("inf") else horizon
        cumulative_service = max(0.0, float(explicit_rate)) * effective_horizon * (1.0 - self.service_safety_fraction)
        service_margin = cumulative_service - workload - self.feasibility_margin

        contact_values = list(monitor_state.contact_slack.values())
        if contact_values:
            contact_margin = min(float(value) for value in contact_values) - self.uncertainty_margin
        else:
            lifetime = _summary_min(monitor_state.remaining_contact_lifetime, default=float("inf"))
            required = float(monitor_state.metadata.get("required_completion_time", 0.0))
            contact_margin = lifetime - required - self.uncertainty_margin

        deadline_margin = _summary_min(monitor_state.deadline_slack, default=float("inf"))
        uncertainty = _summary_max(monitor_state.prediction_uncertainty, default=0.0)
        degradation = _summary_max(monitor_state.degradation_indicators, default=0.0)
        # ``local_load_summary`` may also carry capacity hints (for example a
        # key named ``service_capacity``).  Treat only explicit load/utilisation
        # summaries as risk; capacity belongs to the feasibility margin.
        load_values = {
            key: value
            for key, value in (monitor_state.local_load_summary or {}).items()
            if any(token in str(key).lower() for token in ("load", "util", "pressure"))
        }
        load = _summary_max(load_values, default=0.0)

        reasons: list[str] = []
        if service_margin < 0.0:
            reasons.append("service_margin_negative")
        if contact_margin < 0.0:
            reasons.append("contact_margin_negative")
        if deadline_margin < 0.0:
            reasons.append("deadline_margin_negative")
        if uncertainty > self.uncertainty_margin > 0.0:
            reasons.append("uncertainty_margin_exceeded")

        if reasons:
            feasibility = FeasibilityStatus.INVIABLE
        elif (
            not self.contact_predictability
            or uncertainty > 0.0
            or monitor_state.acquisition.source in {"compatibility_preflight", "full_get_state_fallback"}
        ):
            feasibility = FeasibilityStatus.UNCERTAIN
        else:
            feasibility = FeasibilityStatus.VIABLE

        # Performance viability is deliberately conservative and based on
        # current/cached degradation, never future stochastic truth.
        performance_risk = max(0.0, min(1.0, max(degradation, load)))
        performance_worth_replanning = self.use_performance_viability and performance_risk > self.performance_risk_threshold
        if performance_worth_replanning:
            reasons.append("performance_risk_threshold_exceeded")

        needs_intervention = feasibility != FeasibilityStatus.VIABLE or performance_worth_replanning
        confidence = 0.9 if monitor_state.acquisition.is_true_cheap_monitor else 0.5
        if feasibility == FeasibilityStatus.UNCERTAIN:
            confidence *= 0.7

        affected = self._affected_entities(monitor_state, current_config, reasons)
        return ViabilityReport(
            feasibility_status=feasibility,
            performance_risk=performance_risk,
            service_margin=service_margin,
            contact_margin=contact_margin,
            deadline_margin=deadline_margin,
            uncertainty_margin=self.uncertainty_margin,
            affected_entities=affected,
            reason_codes=reasons,
            needs_intervention=needs_intervention,
            confidence=confidence,
            evaluated_at=float(monitor_state.simulation_time),
            uncertainty_components=dict(monitor_state.prediction_uncertainty),
            horizon_sec=horizon,
            cumulative_service_lower_bound=cumulative_service,
            opportunity_score=performance_risk,
            metadata={
                "oracle_evaluation_only": False,
                "future_stochastic_truth_used": False,
                "feasibility_viability": feasibility.value,
                "performance_viability": performance_worth_replanning,
                "service_rate_lower_bound": float(explicit_rate),
                "effective_horizon_sec": effective_horizon,
                "horizon_semantics": "lower_bound_cumulative_service_minus_remaining_workload",
            },
        )

    @staticmethod
    def _affected_entities(monitor_state: MonitorState, current_config: Any, reasons: list[str]) -> ReconfigurationScope:
        metadata = monitor_state.metadata or {}
        explicit = metadata.get("affected_entities")
        if isinstance(explicit, ReconfigurationScope):
            return explicit
        if isinstance(explicit, Mapping):
            return ReconfigurationScope(**{key: value for key, value in explicit.items() if key in ReconfigurationScope._fields()})
        source_ids = set()
        for field_name in ("source_queue_summary", "deadline_slack", "remaining_workload_summary"):
            source_ids.update(str(key) for key in (getattr(monitor_state, field_name, {}) or {}))
        if not source_ids:
            source_ids.update(getattr(current_config, "covered_source_ids", set()) or set())
        return ReconfigurationScope(
            source_ids=source_ids,
            task_ids=set(getattr(current_config, "covered_task_ids", set()) or set()),
            link_ids={str(key) for key in (monitor_state.contact_slack or {}) if "contact" in " ".join(reasons)},
        )
