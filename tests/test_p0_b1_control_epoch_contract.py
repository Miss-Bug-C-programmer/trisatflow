from __future__ import annotations

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.decision_delay import DecisionDelayBreakdown, DecisionDelayModel
from trisatflow.control.persistent_configuration import PersistentConfiguration


class ControlEpochFixtureBackend:
    """Unit fixture for protocol selection; not a physical simulator proof."""

    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(
            supports_physical_decision_delay=True,
            supports_advance_world=True,
            supports_control_monitor_epoch=True,
            supports_control_epoch_resume=True,
        )
        self.time = 10.0
        self.advance_calls: list[float] = []
        self.resume_calls = 0

    def current_time(self) -> float:
        return self.time

    def advance_control_epoch(self, delta_sec: float) -> dict[str, object]:
        self.advance_calls.append(float(delta_sec))
        before = self.time
        self.time += float(delta_sec)
        return {
            "accepted": True,
            "physicalClockAdvanced": True,
            "simulationTimeBeforeSec": before,
            "simulationTimeSec": self.time,
            "controlEpoch": True,
            "pausedForConfigurationActivation": True,
        }

    def advance_world(self, delta_sec: float) -> dict[str, object]:
        raise AssertionError("control epoch path must be preferred when declared")

    def resume_control_epoch(self) -> dict[str, object]:
        self.resume_calls += 1
        return {"accepted": True, "resumed": True}


def test_decision_delay_prefers_control_monitor_epoch_path() -> None:
    backend = ControlEpochFixtureBackend()
    delay = DecisionDelayModel(
        mode="modeled",
        modeled_components=("solver",),
        require_physical_enforcement=True,
    ).estimate(DecisionCostBreakdown(solver_simulated_latency_sec=2.5))

    result = DecisionDelayModel(
        mode="modeled",
        modeled_components=("solver",),
        require_physical_enforcement=True,
    ).enforce(backend, delay)

    assert result.physical_delay_enforced is True
    assert result.physical_receipt_verified is True
    assert result.actual_delta_sec == 2.5
    assert result.metadata["physical_progression_path"] == "advance_control_epoch"
    assert result.metadata["control_monitor_epoch"] is True
    assert result.metadata["control_epoch_paused_for_activation"] is True
    assert backend.advance_calls == [2.5]


def test_control_epoch_resume_is_explicit_after_stale_plan() -> None:
    backend = ControlEpochFixtureBackend()
    assert backend.resume_control_epoch()["accepted"] is True
    assert backend.resume_calls == 1


def test_unsupported_backend_does_not_claim_control_epoch() -> None:
    class LegacyBackend:
        capabilities = BackendCapabilities(
            supports_physical_decision_delay=True,
            supports_advance_world=False,
        )

    delay = DecisionDelayBreakdown(total_delay_sec=1.0)
    result = DecisionDelayModel(mode="modeled").enforce(LegacyBackend(), delay)
    assert result.physical_delay_enforced is False
    assert result.metadata.get("physical_progression_path") is None


def test_source_keyed_persistent_assignment_materializes_before_default_rule() -> None:
    configuration = PersistentConfiguration(
        config_id="source-keyed",
        assignments={"source-7": {"targetVmId": 7}},
        reusable_rules={"default": {"selector": {}, "assignment": {"targetVmId": 9}}},
    )
    assert configuration.materialize_execution_rule({"task_id": "future-1", "source_id": "source-7"}) == {
        "targetVmId": 7
    }


def test_rejected_control_epoch_cannot_be_certified_as_physical_delay() -> None:
    class RejectedBackend(ControlEpochFixtureBackend):
        def advance_control_epoch(self, delta_sec: float) -> dict[str, object]:
            self.time += float(delta_sec)
            return {
                "accepted": False,
                "reason": "persistent_configuration_cannot_resolve_task_during_control_epoch",
                "physicalClockAdvanced": True,
            }

    backend = RejectedBackend()
    model = DecisionDelayModel(
        mode="modeled",
        modeled_components=("solver",),
        require_physical_enforcement=True,
    )
    delay = model.estimate(DecisionCostBreakdown(solver_simulated_latency_sec=1.0))

    try:
        model.enforce(backend, delay)
    except RuntimeError as error:
        assert "verifiable physical time advancement" in str(error)
    else:
        raise AssertionError("rejected control epoch must fail closed")
