"""Configuration feasibility and performance viability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from trisatflow.control.scope import ReconfigurationScope, extract_typed_affected_entities
from trisatflow.control.types import FeasibilityStatus, MonitorState


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
            and bool(self.metadata.get("required_evidence_complete", False))
            and bool(self.metadata.get("config_truth_consistent", False))
            and not self.metadata.get("evidence_missing", [])
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
        metadata = monitor_state.metadata or {}
        workload = self._remaining_workload(monitor_state.remaining_workload_summary)
        idle = workload <= 0.0
        evidence_missing: list[str] = []

        service_bound_certified = bool(monitor_state.service_bound_certified)
        service_rate = monitor_state.service_rate_lower_bound
        horizon_available = monitor_state.service_horizon_sec is not None
        contact_required = self._contact_required(monitor_state, idle)
        service_evidence_available = service_bound_certified and service_rate is not None
        if not idle and not service_evidence_available:
            evidence_missing.append("service_bound_unavailable")
        if not idle and not horizon_available:
            evidence_missing.append("service_horizon_unavailable")

        horizon = max(0.0, float(monitor_state.service_horizon_sec)) if horizon_available else 0.0
        if service_evidence_available and horizon_available:
            contact_horizon = _summary_min(monitor_state.remaining_contact_lifetime, default=horizon) if contact_required else horizon
            effective_horizon = min(horizon, max(0.0, contact_horizon)) if contact_horizon != float("inf") else horizon
            cumulative_service = max(0.0, float(service_rate)) * effective_horizon * (1.0 - self.service_safety_fraction)
            service_margin = cumulative_service - workload - self.feasibility_margin
        else:
            effective_horizon = 0.0
            cumulative_service = 0.0
            service_margin = 0.0

        contact_values = [float(value) for value in (monitor_state.contact_slack or {}).values() if _is_number(value)]
        contact_evidence_available = bool(contact_values)
        if contact_required:
            if contact_values:
                contact_margin = min(contact_values) - self.uncertainty_margin
            else:
                required = _optional_number(metadata.get("required_completion_time"))
                lifetime = _summary_min(monitor_state.remaining_contact_lifetime, default=float("nan"))
                if required is not None and lifetime == lifetime:
                    contact_margin = lifetime - required - self.uncertainty_margin
                    contact_evidence_available = True
                else:
                    contact_margin = 0.0
                    evidence_missing.append("contact_evidence_unavailable")
        else:
            contact_margin = 0.0

        deadline_values = [float(value) for value in (monitor_state.deadline_slack or {}).values() if _is_number(value)]
        if idle:
            deadline_margin = 0.0
        elif deadline_values:
            deadline_margin = min(deadline_values)
        else:
            deadline_margin = 0.0
            evidence_missing.append("deadline_evidence_unavailable")

        uncertainty_values = [float(value) for value in (monitor_state.prediction_uncertainty or {}).values() if _is_number(value)]
        uncertainty_applicable = bool(metadata.get("uncertainty_evidence_applicable", not idle))
        uncertainty_evidence_available = (
            not uncertainty_applicable
            or (bool(monitor_state.uncertainty_evidence_available) and bool(uncertainty_values))
        )
        if uncertainty_applicable and not uncertainty_evidence_available:
            evidence_missing.append("uncertainty_unavailable")
        # Numeric uncertainty without an explicit evidence marker is not a
        # calibrated risk estimate.  Keep it out of the negative-margin path;
        # the missing-evidence code above drives the result to UNCERTAIN.
        uncertainty = max(uncertainty_values) if uncertainty_evidence_available and uncertainty_values else 0.0
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

        config_truth_consistent, config_truth_required = self._configuration_truth(monitor_state, current_config)
        if not config_truth_consistent:
            evidence_missing.append("configuration_state_mismatch")
        if not monitor_state.acquisition.is_true_cheap_monitor:
            evidence_missing.append("cheap_monitor_unverified")

        reasons: list[str] = []
        if service_margin < 0.0:
            reasons.append("service_margin_negative")
        if contact_margin < 0.0:
            reasons.append("contact_margin_negative")
        if deadline_margin < 0.0:
            reasons.append("deadline_margin_negative")
        if uncertainty > self.uncertainty_margin:
            reasons.append("uncertainty_margin_exceeded")
        reasons.extend(code for code in evidence_missing if code not in reasons)

        negative_margin = any(
            code in reasons
            for code in (
                "service_margin_negative",
                "contact_margin_negative",
                "deadline_margin_negative",
                "uncertainty_margin_exceeded",
            )
        )
        if negative_margin:
            feasibility = FeasibilityStatus.INVIABLE
        elif (
            evidence_missing
            or (contact_required and not self.contact_predictability)
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
        confidence = 0.9 if monitor_state.acquisition.is_true_cheap_monitor else 0.0
        if feasibility == FeasibilityStatus.UNCERTAIN:
            confidence = min(confidence, 0.5)

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
                "service_rate_lower_bound": float(service_rate) if service_rate is not None else None,
                "service_rate_observed": monitor_state.service_rate_observed,
                "service_rate_source": monitor_state.service_rate_source,
                "service_bound_semantics": monitor_state.service_bound_semantics,
                "service_evidence_available": service_evidence_available or idle,
                "service_bound_certified": service_bound_certified,
                "service_horizon_available": horizon_available or idle,
                "contact_evidence_required": contact_required,
                "contact_evidence_available": contact_evidence_available or not contact_required,
                "uncertainty_evidence_available": uncertainty_evidence_available,
                "configuration_truth_required": config_truth_required,
                "config_truth_consistent": config_truth_consistent,
                "evidence_missing": list(evidence_missing),
                "required_evidence_complete": not evidence_missing,
                "effective_horizon_sec": effective_horizon,
                "horizon_semantics": "certified_lower_bound_only" if service_bound_certified else "not_certified",
            },
        )

    @staticmethod
    def _affected_entities(monitor_state: MonitorState, current_config: Any, reasons: list[str]) -> ReconfigurationScope:
        return extract_typed_affected_entities(monitor_state, current_config)

    @staticmethod
    def _remaining_workload(summary: Mapping[str, Any] | None) -> float:
        values = summary or {}
        if "total" in values and _is_number(values.get("total")):
            return max(0.0, float(values["total"]))
        return max(0.0, sum(float(value) for value in values.values() if _is_number(value)))

    @staticmethod
    def _contact_required(monitor_state: MonitorState, idle: bool) -> bool:
        if idle:
            return False
        metadata = monitor_state.metadata or {}
        for key in ("contact_evidence_required", "contactEvidenceRequired"):
            if key in metadata:
                return bool(metadata[key])
        if bool(metadata.get("execution_is_local", False)) or bool(metadata.get("contact_not_applicable", False)):
            return False
        return bool(
            monitor_state.contact_slack
            or monitor_state.remaining_contact_lifetime
            or not metadata.get("contact_applicability_known", False)
        )

    @staticmethod
    def _configuration_truth(monitor_state: MonitorState, current_config: Any) -> tuple[bool, bool]:
        metadata = monitor_state.metadata or {}
        expected_id = getattr(current_config, "config_id", None)
        expected_version = getattr(current_config, "version", None)
        observed_id = monitor_state.current_config_id
        observed_version = monitor_state.current_config_version
        authoritative = bool(metadata.get("authoritative_physical", False))
        declared = bool(metadata.get("configuration_truth_available", False)) or observed_id is not None or observed_version is not None
        required = authoritative or declared
        if not required:
            return True, False
        if expected_id is None and expected_version is None:
            return observed_id is not None and observed_version is not None, True
        consistent = (
            observed_id is not None
            and observed_version is not None
            and str(observed_id) == str(expected_id)
            and int(observed_version) == int(expected_version)
        )
        return consistent, True


def _is_number(value: Any) -> bool:
    try:
        return float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _optional_number(value: Any) -> float | None:
    return float(value) if _is_number(value) else None
