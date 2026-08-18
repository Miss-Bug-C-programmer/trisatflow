from __future__ import annotations

from trisatflow.control.benefit import ConservativeAnalyticalBenefitEstimator
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.types import FeasibilityStatus, MonitorAcquisitionMetadata, MonitorState, PlanningBudget, PlannerFidelity, PlanningDescriptor
from trisatflow.control.viability import ConservativeViabilityEstimator


def _config() -> PersistentConfiguration:
    return PersistentConfiguration(config_id="cfg", version=1, covered_source_ids={"1"})


def _monitor(**changes) -> MonitorState:
    payload = {
        "remaining_workload_summary": {"total": 10.0, "source:1": 10.0},
        "deadline_slack": {"42": 5.0},
        "service_rate_lower_bound": 2.0,
        "service_bound_certified": True,
        "service_horizon_sec": 10.0,
        "contact_slack": {},
        "prediction_uncertainty": {"service": 0.0},
        "uncertainty_evidence_available": True,
        "acquisition": MonitorAcquisitionMetadata(is_true_cheap_monitor=True),
        "metadata": {"contact_evidence_required": False},
    }
    payload.update(changes)
    return MonitorState(**payload)


def test_complete_non_idle_evidence_certifies_keep() -> None:
    report = ConservativeViabilityEstimator().evaluate(_monitor(), _config())
    assert report.feasibility_status == FeasibilityStatus.VIABLE
    assert report.certifies_keep
    assert report.metadata["service_evidence_status"] == "AVAILABLE"


def test_positive_uncertainty_is_a_robust_margin_penalty_not_automatic_invalidity() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(prediction_uncertainty={"service": 0.2}), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.VIABLE
    assert report.certifies_keep
    assert report.metadata["robust_service_margin"] > 0.0


def test_uncertainty_crossing_robust_margin_is_inviable() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(prediction_uncertainty={"service": 20.0}), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.INVIABLE
    assert "service_margin_negative" in report.reason_codes
    assert report.metadata["robust_service_margin"] < 0.0


def test_explicit_local_contact_is_not_applicable() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(contact_slack={}, metadata={"contact_evidence_required": False}), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.VIABLE
    assert report.metadata["contact_evidence_status"] == "NOT_APPLICABLE"


def test_unknown_remote_contact_applicability_remains_uncertain() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(
            contact_slack={},
            metadata={"contact_applicability_known": False},
        ),
        _config(),
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "contact_evidence_unavailable" in report.reason_codes


def test_benefit_aggregation_prefers_total_and_unfinished_queue_count() -> None:
    monitor = _monitor(
        remaining_workload_summary={"total": 75.0, "source:1": 50.0, "source:2": 25.0},
        source_queue_summary={"arrivedTaskCount": 10.0, "unfinishedTaskCount": 4.0, "pendingDecision": 1.0},
        service_rate_lower_bound=None,
        service_bound_certified=False,
        service_rate_observed=5.0,
    )
    outcome = ConservativeAnalyticalBenefitEstimator().estimate_hold(monitor, _config(), 10.0)
    assert outcome.expected_task_cost == 75.0
    assert outcome.queue_cost == 4.0
    assert outcome.metadata["service_rate_source"] == "service_rate_observed"
    assert outcome.metadata["service_rate_semantics"] == "observed_not_certified"


def test_unavailable_soft_service_rate_is_not_invented() -> None:
    monitor = _monitor(
        service_rate_lower_bound=None,
        service_bound_certified=False,
        service_rate_observed=None,
        local_load_summary={"service": 99.0},
    )
    outcome = ConservativeAnalyticalBenefitEstimator().estimate_hold(monitor, _config(), 10.0)
    assert outcome.metadata["service_rate_source"] == "unavailable"
    assert outcome.metadata["service_rate_semantics"] == "unavailable"

