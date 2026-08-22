from __future__ import annotations

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend
from trisatflow.control.decision_delay import DecisionDelayBreakdown, DecisionDelayModel, PostDelayRevalidator
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope


class _PhysicalBackend:
    capabilities = BackendCapabilities(
        supports_physical_decision_delay=True,
        supports_advance_world=True,
    )

    def __init__(self) -> None:
        self.time = 4.0

    def current_time(self) -> float:
        return self.time

    def advance_world(self, delta_sec: float) -> dict[str, object]:
        before = self.time
        self.time += delta_sec
        return {
            "accepted": True,
            "physicalClockAdvanced": True,
            "physicalStateChanged": True,
            "physicalProgressBefore": {"remainingTaskWorkload": 10.0},
            "physicalProgressAfter": {"remainingTaskWorkload": 9.0},
            "simulationTimeBeforeSec": before,
            "simulationTimeSec": self.time,
        }


def test_delay_keeps_native_progression_receipt() -> None:
    backend = _PhysicalBackend()
    delay = DecisionDelayModel(mode="modeled").enforce(
        backend,
        DecisionDelayBreakdown(total_delay_sec=1.0),
    )
    assert delay.physical_receipt_verified is True
    assert delay.metadata["physical_advance_receipt"]["physicalStateChanged"] is True
    assert delay.metadata["physical_advance_receipt"]["physicalProgressAfter"]["remainingTaskWorkload"] == 9.0


def test_revalidation_preserves_world_token_metadata() -> None:
    class Backend:
        def validate_configuration(self, configuration, **kwargs):
            return {
                "accepted": True,
                "worldVersion": 8,
                "decisionStatus": "APPLY",
                "reasonCodes": [],
            }

    result = PostDelayRevalidator().revalidate(
        Backend(),
        object(),
        planned_at=1.0,
        applied_at=2.0,
        observed_world_version=7,
        intervention_id="int-1",
    )
    assert result.accepted is True
    assert result.metadata["worldVersion"] == 8
    assert result.metadata["decisionStatus"] == "APPLY"


def test_live_patch_payload_contains_observation_and_revalidation_identity() -> None:
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
                "afterConfiguration": {
                    "configId": "cfg.v2",
                    "version": 2,
                    "assignments": {"1": {"targetVmIndex": 0}},
                },
                "evidenceId": "e-1",
                "decisionStatus": "APPLY",
                "scopeInvariantSatisfied": True,
                "realizedScope": {"task_ids": ["1"]},
                "actualChangedEntities": ["task:1:persistentRuleChanges"],
                "realizedReconfigurationVolume": {"changedEntityCount": 1},
            }

    client = Client()
    backend = object.__new__(SatEdgeSimBackend)
    backend.client = client
    backend._last_state = {}
    backend._observed_world_version = 4
    backend._observed_control_epoch = 3
    backend._last_revalidated_world_version = 5
    backend._configuration = None
    backend._capabilities = BackendCapabilities(
        supports_configuration_patch=True,
        authoritative_physical=True,
    )
    current = PersistentConfiguration(config_id="cfg", version=1, assignments={"1": {"targetVmIndex": 0}})
    proposed = current.clone(version=2)
    proposed.reusable_rules = {"1": {"selector": {"taskId": "1"}, "assignment": {"targetVmIndex": 0}}}
    receipt = backend.apply_configuration_patch(
        current,
        proposed,
        ReconfigurationScope(task_ids={"1"}),
        acquisition_epoch=3,
        intervention_id="int-1",
        observed_world_version=4,
        observed_control_epoch=3,
        revalidated_world_version=5,
        planning_delay={"total_delay_sec": 1.0},
        acquisition_metadata={"source": "cheap_monitor"},
    )
    assert receipt["evidenceId"] == "e-1"
    assert client.payload["baseWorldVersion"] == 4
    assert client.payload["observedWorldVersion"] == 4
    assert client.payload["observedControlEpoch"] == 3
    assert client.payload["revalidatedWorldVersion"] == 5
    assert client.payload["planningDelayMetadata"]["total_delay_sec"] == 1.0
    assert client.payload["acquisitionMetadata"]["source"] == "cheap_monitor"
