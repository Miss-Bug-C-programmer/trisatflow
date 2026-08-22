from __future__ import annotations

import pytest

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend
from trisatflow.control.config import ControllerConfig
from trisatflow.control.controller import EndogenousReplanningController
from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope, ScopeGenerator
from trisatflow.control.types import MonitorAcquisitionMetadata, MonitorState, PlannerState
from trisatflow.control.viability import ConservativeViabilityEstimator


def _configuration() -> PersistentConfiguration:
    return PersistentConfiguration(
        config_id="cfg",
        version=4,
        assignments={"42": {"target": "n1"}},
        covered_task_ids={"42"},
        covered_source_ids={"s1"},
    )


def _monitor(**changes: object) -> MonitorState:
    payload: dict[str, object] = {
        "simulation_time": 12.0,
        "remaining_workload_summary": {"total": 10.0, "source:s1": 10.0},
        "deadline_slack": {"42": 8.0},
        "contact_slack": {"transfer:42:link-1": 8.0},
        "service_rate_lower_bound": 4.0,
        "service_bound_certified": True,
        "service_horizon_sec": 10.0,
        "prediction_uncertainty": {"service": 0.1},
        "uncertainty_evidence_available": True,
        "acquisition": MonitorAcquisitionMetadata(is_true_cheap_monitor=True, source="contract-test"),
        "metadata": {"contact_evidence_required": True, "monitor_epoch": 6},
    }
    payload.update(changes)
    return MonitorState(**payload)


def _report(monitor: MonitorState | None = None):
    return ConservativeViabilityEstimator().evaluate(monitor or _monitor(), _configuration())


def test_t1_strong_robust_margins_certify_safe() -> None:
    report = _report()
    assert report.certificate is not None
    assert report.certificate.certified_safe is True
    assert report.certificate.service_margin is not None and report.certificate.service_margin > 0.0
    assert report.certificate.monitor_epoch == 6
    assert report.certificate.units["service_lower_bound"] == "MI"


def test_t2_service_lower_bound_failure_has_provenance() -> None:
    report = _report(_monitor(service_rate_lower_bound=0.5))
    assert report.certificate_safe is False
    assert "SERVICE_DEFICIT" in report.certificate_failure_reasons
    assert any(item.violation_type == "SERVICE_DEFICIT" for item in report.constraint_provenance)
    assert "42" in report.constraint_provenance[0].task_ids


def test_t3_deadline_and_contact_failures_have_correct_provenance() -> None:
    report = _report(_monitor(deadline_slack={"42": -1.0}, contact_slack={"transfer:42:link-1": -1.0}))
    kinds = {item.violation_type for item in report.constraint_provenance}
    assert {"DEADLINE_DEFICIT", "CONTACT_DEFICIT"}.issubset(kinds)
    assert report.trigger_reason == "CERTIFICATE_FAILURE"


def test_t4_certificate_safe_configuration_naturally_keeps() -> None:
    report = _report()
    assert report.certifies_keep is True
    assert report.needs_intervention is False
    assert report.trigger_reason == "KEEP_SAFE"


def test_t5_empty_scope_is_keep_without_synthetic_action_class() -> None:
    empty = ReconfigurationScope()
    assert empty.is_empty
    assert empty.to_dict() == {
        "task_ids": [], "source_ids": [], "node_ids": [], "link_ids": [], "route_ids": [], "resource_keys": []
    }
    assert not hasattr(empty, "local")
    assert empty.derived_bucket() == "keep"


def test_t6_single_task_violation_generates_minimal_candidate_first() -> None:
    report = _report(_monitor(deadline_slack={"42": -1.0}, contact_slack={"transfer:42:link-1": 8.0}))
    candidates = ScopeGenerator(include_global_candidate=True).generate(_configuration(), _monitor(), None, report)
    assert candidates
    assert candidates[0].task_ids == {"42"}
    assert candidates[0].metadata["expansion_reason"] == "minimal_direct_implication"
    assert candidates[0].metadata["size_primitives"]["task_ids"] == 1


def test_t7_dependency_expansion_is_deterministic_and_typed() -> None:
    monitor = _monitor(metadata={"contact_evidence_required": True})
    report = _report(_monitor(deadline_slack={"42": -1.0}))
    report.metadata["dependency_graph"] = {"42": ["link:l1", "node:n1"]}
    first = ScopeGenerator(include_global_candidate=False).generate(_configuration(), monitor, None, report)
    second = ScopeGenerator(include_global_candidate=False).generate(_configuration(), monitor, None, report)
    assert first[1].to_dict() == second[1].to_dict()
    assert first[1].link_ids == {"l1"}
    assert first[1].node_ids == {"n1"}
    assert first[1].metadata["expansion_reason"] == "dependency_expansion"


def test_t8_scope_serialization_preserves_resume_recompute_semantics() -> None:
    report = _report(_monitor(deadline_slack={"42": -1.0}))
    scope = ScopeGenerator(include_global_candidate=False).generate(_configuration(), _monitor(), None, report)[0]
    serialized = scope.to_dict()
    assert serialized["metadata"]["preserve_resume_recompute"] == {
        "preserve": True, "resume": True, "recompute": False
    }
    restored = ReconfigurationScope(**serialized)
    assert restored.metadata["preserve_resume_recompute"]["resume"] is True


def test_t9_modification_scope_cannot_silently_widen() -> None:
    current = _configuration()
    proposed = current.clone(version=5)
    proposed.resource_allocations = {"other-task": {"cpuShare": 0.5}}
    with pytest.raises(RuntimeError, match="outside the requested modification scope"):
        EndogenousReplanningController._assert_modification_scope(
            current, proposed, ReconfigurationScope(task_ids={"42"})
        )


def test_t10_estimated_and_realized_costs_are_distinct() -> None:
    cost = DecisionCostBreakdown(reconfiguration_bytes=100, migration_volume=2.0, num_changed_resources=1)
    assert cost.realized_reconfiguration_cost is None
    assert cost.estimated_reconfiguration_cost > 0.0
    cost.actual_reconfiguration_bytes = 7
    cost.actual_migration_volume = 0.25
    cost.metadata["reconfiguration_receipt_status"] = "verified"
    assert cost.realized_reconfiguration_cost is not None
    assert cost.realized_reconfiguration_cost != cost.estimated_reconfiguration_cost

    measured = DecisionCostBreakdown(reconfiguration_bytes=100, migration_volume=2.0)
    EndogenousReplanningController._merge_apply_receipt_cost(
        measured,
        {
            "accepted": True,
            "changed": True,
            "scopeInvariantSatisfied": True,
            "realizedScope": {"task_ids": ["42"]},
            "actualChangedEntities": ["task:42:resource"],
            "realizedReconfigurationVolume": {
                "changedEntityCount": 1,
                "resourceChanges": 1,
                "bytesStateTransferred": 0,
            },
        },
        {"reconfiguration_bytes": 100, "migration_volume": 2.0},
    )
    assert measured.metadata["reconfiguration_receipt_status"] == "verified"
    assert measured.actual_reconfiguration_bytes == 0
    assert measured.actual_migration_volume == 0.0
    assert measured.num_changed_resources == 1


def test_configuration_patch_contract_sends_exact_after_values_and_versions() -> None:
    class Client:
        def __init__(self) -> None:
            self.payload = None

        def _request(self, method, path, **kwargs):
            assert method == "POST"
            assert path == "/configuration/patch"
            self.payload = kwargs["json"]
            return {
                "accepted": True,
                "changed": True,
                "requestedScope": self.payload["requestedScope"],
                "realizedScope": {"task_ids": ["42"]},
                "actualChangedEntities": ["task:42:assignment"],
                "realizedReconfigurationVolume": {"changedEntityCount": 1, "bytesStateTransferred": 0},
            }

    client = Client()
    backend = object.__new__(SatEdgeSimBackend)
    backend.client = client
    backend._last_state = {"worldVersion": 17}
    backend._configuration = None
    backend._capabilities = BackendCapabilities(
        supports_configuration_patch=True,
        authoritative_physical=True,
    )
    current = _configuration()
    proposed = current.clone(version=5)
    proposed.assignments = {"42": {"targetVmId": 3}}
    receipt = backend.apply_configuration_patch(
        current,
        proposed,
        ReconfigurationScope(task_ids={"42"}),
        preserve_resume_recompute={"preserve": True, "resume": True, "recompute": False},
        acquisition_epoch=9,
        intervention_id="int-1",
    )
    assert receipt["accepted"] is True
    assert client.payload["baseConfigurationVersion"] == 4
    assert client.payload["baseWorldVersion"] == 17
    assert client.payload["acquisitionEpoch"] == 9
    assert client.payload["taskAssignmentChanges"]["42"] == {"targetVmId": 3}
    assert "before" not in client.payload["taskAssignmentChanges"]["42"]
    assert backend._configuration is proposed


def test_t11_missing_backend_evidence_is_not_certified_safe() -> None:
    report = _report(_monitor(
        service_rate_lower_bound=None,
        service_bound_certified=False,
        prediction_uncertainty={},
        uncertainty_evidence_available=False,
    ))
    assert report.certificate_safe is False
    assert report.certificate.uncertainty_bound is None
    assert report.certificate.evidence_complete is False


class _IntegrationBackend:
    def __init__(self, *, missing_certificate: bool = False) -> None:
        self.time = 12.0
        self.missing_certificate = missing_certificate
        self.planner_state_calls = 0
        self.applied: list[PersistentConfiguration] = []
        self.capabilities = BackendCapabilities(
            supports_monitor_state=True,
            supports_planner_state=True,
            supports_configuration_apply=True,
            supports_persistent_configuration=True,
            supports_advance_world=True,
            supports_physical_decision_delay=True,
            authoritative_physical=False,
            backend_source="contract-integration",
        )

    def get_monitor_state(self, context=None) -> MonitorState:
        changes = {
            "simulation_time": self.time,
            "deadline_slack": {"42": -1.0},
            "metadata": {"affected_entities": {"source_ids": {"s1"}}, "contact_evidence_required": False},
        }
        if self.missing_certificate:
            changes.update({
                "service_rate_lower_bound": None,
                "service_bound_certified": False,
                "prediction_uncertainty": {},
                "uncertainty_evidence_available": False,
            })
        return _monitor(
            **changes,
        )

    def get_planner_state(self, context=None, scope=None, budget=None) -> PlannerState:
        self.planner_state_calls += 1
        return PlannerState(
            simulation_time=self.time,
            candidate_vms=[{"sourceId": "s1", "vmIndex": 1, "score": 0.1}],
            detailed_resources={"n1": {"cpu": 1.0}},
        )

    def apply_configuration(self, configuration: PersistentConfiguration):
        self.applied.append(configuration.clone())
        return {"accepted": True, "compatibility_fallback": True}

    def advance_world(self, delta_sec: float):
        self.time += float(delta_sec)
        return {"accepted": True}

    def validate_configuration(self, configuration: PersistentConfiguration):
        return {"accepted": True}

    def dispatch_under_configuration(self, configuration, task=None):
        return configuration.materialize_execution_rule(task or {"task_id": "42"})


def test_t12_strict_mode_fails_closed_for_missing_certificate_evidence() -> None:
    backend = _IntegrationBackend(missing_certificate=True)
    controller = EndogenousReplanningController(backend, config={"strict_publication_mode": True})
    controller.initialize(_configuration(), initial_plan=False)
    # This is a strict contract check, not an experiment run.
    with pytest.raises(RuntimeError, match="complete robust viability certificate"):
        controller.on_monitor_epoch()


def test_integration_monitor_certificate_scope_acquisition_planner_path() -> None:
    backend = _IntegrationBackend()
    controller = EndogenousReplanningController(
        backend,
        config={
            "planner": {"enabled_backends": []},
            "decision_cost": {
                "obs_price": 0.0, "sync_price": 0.0, "solve_price": 0.0,
                "signal_price": 0.0, "reconfiguration_price": 0.0,
            },
        },
    )
    controller.initialize(_configuration(), initial_plan=False)
    decision = controller.on_monitor_epoch()
    assert decision.monitor_state.metadata["monitor_epoch"] == 1
    assert decision.viability_report.certificate is not None
    assert decision.observation_scope == decision.modification_scope
    assert decision.scope.metadata["provenance"]
    assert backend.planner_state_calls == 1
    assert backend.applied
