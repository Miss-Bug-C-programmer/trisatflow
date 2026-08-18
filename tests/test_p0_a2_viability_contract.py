from __future__ import annotations

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope, extract_typed_affected_entities
from trisatflow.control.types import FeasibilityStatus, MonitorAcquisitionMetadata, MonitorState
from trisatflow.control.viability import ConservativeViabilityEstimator, ViabilityReport


def _config() -> PersistentConfiguration:
    return PersistentConfiguration(config_id="cfg", version=1, covered_source_ids={"1"})


def _monitor(**changes) -> MonitorState:
    payload = {
        "remaining_workload_summary": {"total": 10.0, "source:1": 10.0},
        "deadline_slack": {"42": 5.0},
        "service_rate_lower_bound": 2.0,
        "service_bound_certified": True,
        "service_horizon_sec": 10.0,
        "contact_slack": {"transfer:42:TASK": 5.0},
        "prediction_uncertainty": {"service": 0.0},
        "uncertainty_evidence_available": True,
        "acquisition": MonitorAcquisitionMetadata(is_true_cheap_monitor=True),
    }
    payload.update(changes)
    return MonitorState(**payload)


def test_missing_service_bound_is_uncertain_and_cannot_keep() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(service_rate_lower_bound=None, service_bound_certified=False), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "service_bound_unavailable" in report.reason_codes
    assert not report.certifies_keep


def test_observed_rate_is_not_accepted_as_a_certified_bound() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(service_rate_observed=2.0, service_rate_lower_bound=2.0, service_bound_certified=False), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert report.metadata["horizon_semantics"] == "not_certified"


def test_missing_uncertainty_is_not_zero_uncertainty() -> None:
    report = ConservativeViabilityEstimator().evaluate(_monitor(prediction_uncertainty={}), _config())
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "uncertainty_unavailable" in report.reason_codes


def test_unmarked_zero_uncertainty_is_not_calibrated_evidence() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(prediction_uncertainty={"service": 0.0}, uncertainty_evidence_available=False), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "uncertainty_unavailable" in report.reason_codes
    assert not report.certifies_keep


def test_missing_service_horizon_is_not_replaced_by_default_horizon() -> None:
    report = ConservativeViabilityEstimator(evaluation_horizon_sec=999.0).evaluate(
        _monitor(service_horizon_sec=None), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "service_horizon_unavailable" in report.reason_codes
    assert report.horizon_sec == 0.0


def test_negative_deadline_margin_is_inviable() -> None:
    report = ConservativeViabilityEstimator().evaluate(_monitor(deadline_slack={"42": -0.1}), _config())
    assert report.feasibility_status == FeasibilityStatus.INVIABLE
    assert "deadline_margin_negative" in report.reason_codes


def test_negative_contact_margin_is_inviable_when_contact_is_required() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(contact_slack={"transfer:42:TASK": -0.1}), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.INVIABLE
    assert "contact_margin_negative" in report.reason_codes


def test_missing_contact_evidence_is_uncertain_when_required() -> None:
    report = ConservativeViabilityEstimator().evaluate(_monitor(contact_slack={}), _config())
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "contact_evidence_unavailable" in report.reason_codes


def test_explicitly_local_execution_does_not_require_contact_evidence() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(contact_slack={}, metadata={"contact_evidence_required": False}), _config()
    )
    assert report.feasibility_status == FeasibilityStatus.VIABLE
    assert report.certifies_keep


def test_idle_workload_has_explicit_non_applicable_service_semantics() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        MonitorState(acquisition=MonitorAcquisitionMetadata(is_true_cheap_monitor=True), remaining_workload_summary={"total": 0.0}),
        _config(),
    )
    assert report.feasibility_status == FeasibilityStatus.VIABLE
    assert report.certifies_keep
    assert report.metadata["service_evidence_available"] is True


def test_typed_source_and_task_parsing() -> None:
    monitor = MonitorState(
        remaining_workload_summary={"total": 75.0, "source:1": 75.0},
        deadline_slack={"42": 8.0},
        source_queue_summary={"arrivedTaskCount": 2.0, "unfinishedTaskCount": 1.0, "pendingDecision": 0.0},
    )
    scope = extract_typed_affected_entities(monitor)
    assert scope.source_ids == {"1"}
    assert scope.task_ids == {"42"}


def test_transfer_contact_key_is_task_evidence_not_link_id() -> None:
    monitor = MonitorState(contact_slack={"transfer:42:TASK": -1.0})
    scope = extract_typed_affected_entities(monitor)
    assert scope.task_ids == {"42"}
    assert scope.link_ids == set()


def test_aggregate_keys_do_not_enter_any_entity_set() -> None:
    monitor = MonitorState(
        remaining_workload_summary={"total": 75.0, "source:1": 75.0},
        deadline_slack={"42": 8.0},
        source_queue_summary={"arrivedTaskCount": 2.0, "unfinishedTaskCount": 1.0, "pendingDecision": 0.0},
        contact_slack={"transfer:42:TASK": 1.0},
    )
    scope = extract_typed_affected_entities(monitor)
    all_ids = scope.task_ids | scope.source_ids | scope.link_ids | scope.resource_keys
    assert {"total", "arrivedTaskCount", "unfinishedTaskCount", "pendingDecision"}.isdisjoint(all_ids)
    assert "42" not in scope.source_ids
    assert not scope.link_ids


def test_explicit_typed_hints_are_canonicalized_and_aggregates_removed() -> None:
    monitor = MonitorState(
        metadata={
            "affected_entity_hints": {
                "affectedSourceIds": ["source:1", "total"],
                "affectedTaskIds": ["task:42", "pendingDecision"],
                "affectedLinkIds": ["link:7"],
            }
        }
    )
    scope = extract_typed_affected_entities(monitor)
    assert scope.source_ids == {"1"}
    assert scope.task_ids == {"42"}
    assert scope.link_ids == {"7"}


def test_server_observed_configuration_overrides_python_cache_in_monitor_state() -> None:
    backend = object.__new__(SatEdgeSimBackend)
    backend._configuration = _config()
    backend._capabilities = BackendCapabilities(authoritative_physical=True)
    monitor = backend._monitor_from_payload(
        {
            "payloadKind": "cheap_monitor",
            "configId": "server-cfg",
            "configVersion": 2,
            "remainingWorkload": {"total": 0.0},
            "instrumentation": {"candidateEvaluations": 0, "fullStateBuilderInvoked": False},
            "containsFutureStochasticState": False,
        },
        source="/get_monitor_state",
        true_cheap=True,
    )
    assert monitor.current_config_id == "server-cfg"
    assert monitor.current_config_version == 2
    assert monitor.metadata["expected_config_id"] == "cfg"
    assert monitor.metadata["configuration_state_mismatch"] is True


def test_backend_reset_drops_previous_expected_configuration_cache() -> None:
    class ResetClient:
        def reset(self, **kwargs):
            return {"accepted": True}

    backend = object.__new__(SatEdgeSimBackend)
    backend.client = ResetClient()
    backend._configuration = _config()
    backend._last_state = {"configId": "cfg"}
    assert backend.reset() == {"accepted": True}
    assert backend._configuration is None
    assert backend._last_state == {}


def test_configuration_mismatch_cannot_certify_keep() -> None:
    report = ConservativeViabilityEstimator().evaluate(
        _monitor(
            current_config_id="server-cfg",
            current_config_version=2,
            metadata={"authoritative_physical": True},
        ),
        _config(),
    )
    assert report.feasibility_status == FeasibilityStatus.UNCERTAIN
    assert "configuration_state_mismatch" in report.reason_codes
    assert not report.certifies_keep


def test_report_without_evidence_metadata_cannot_certify_keep() -> None:
    report = ViabilityReport(
        feasibility_status=FeasibilityStatus.VIABLE,
        confidence=1.0,
        needs_intervention=False,
    )
    assert not report.certifies_keep


def test_typed_scope_falls_back_only_to_typed_configuration_coverage() -> None:
    monitor = MonitorState(source_queue_summary={"total": 2.0, "pendingDecision": 1.0})
    scope = extract_typed_affected_entities(monitor, _config())
    assert scope.source_ids == {"1"}
    assert "total" not in scope.source_ids
