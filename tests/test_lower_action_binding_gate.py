from __future__ import annotations

from scripts.check_satedgesim_lower_action_binding import evaluate_binding_receipts, same_discrete_target


def test_binding_receipts_accept_distinct_applied_continuous_action_and_metric_change() -> None:
    receipt_a = {
        "lowerActionBindingVersion": "vm_network_power_binding_v1",
        "requestedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "appliedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "estimatedTransmissionRateMbps": 100.0,
    }
    receipt_b = {
        "lowerActionBindingVersion": "vm_network_power_binding_v1",
        "requestedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "appliedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "estimatedTransmissionRateMbps": 25.0,
    }

    payload = evaluate_binding_receipts(
        receipt_a,
        receipt_b,
        require_binding_version="vm_network_power_binding_v1",
    )

    assert payload["status"] == "LOWER_ACTION_BINDING_OK"
    assert payload["requested_actions_differ"] is True
    assert payload["applied_actions_differ"] is True
    assert "estimatedTransmissionRateMbps" in payload["controlled_metric_differences"]


def test_binding_receipts_block_full_hybrid_claim_when_server_reports_unbound() -> None:
    receipt_a = {
        "lowerActionBindingVersion": "unbound",
        "requestedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "appliedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "estimatedTransmissionRateMbps": 100.0,
    }
    receipt_b = {
        "lowerActionBindingVersion": "unbound",
        "requestedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "appliedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "estimatedTransmissionRateMbps": 25.0,
    }

    payload = evaluate_binding_receipts(
        receipt_a,
        receipt_b,
        require_binding_version="vm_network_power_binding_v1",
    )

    assert payload["status"] == "STAGE_BLOCKED_FOR_FULL_HYBRID_CLAIM"
    assert "lower_action_binding_version_mismatch:unbound,unbound" in payload["violations"]


def test_binding_receipts_reject_api_echo_without_physical_effect() -> None:
    receipt_a = {
        "lowerActionBindingVersion": "vm_network_power_binding_v1",
        "requestedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "appliedContinuousAction": {"cpuShare": 1.0, "bandwidthShare": 1.0, "txPowerRatio": 1.0},
        "estimatedTransmissionRateMbps": 100.0,
    }
    receipt_b = {
        "lowerActionBindingVersion": "vm_network_power_binding_v1",
        "requestedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "appliedContinuousAction": {"cpuShare": 0.25, "bandwidthShare": 0.25, "txPowerRatio": 0.25},
        "estimatedTransmissionRateMbps": 100.0,
    }

    payload = evaluate_binding_receipts(
        receipt_a,
        receipt_b,
        require_binding_version="vm_network_power_binding_v1",
    )

    assert payload["status"] == "LOWER_ACTION_BINDING_FAILED"
    assert "no_controlled_physical_metric_changed" in payload["violations"]


def test_same_discrete_target_requires_action_and_vm_match() -> None:
    probe_a = {"action": {"policyUpperAction": 1, "abstractAction": 1, "targetVmIndex": 8, "targetVmId": 8, "selectedVmId": 8}}
    probe_b = {"action": {"policyUpperAction": 1, "abstractAction": 1, "targetVmIndex": 8, "targetVmId": 8, "selectedVmId": 8}}
    probe_c = {"action": {"policyUpperAction": 2, "abstractAction": 2, "targetVmIndex": 164, "targetVmId": 164, "selectedVmId": 164}}

    assert same_discrete_target(probe_a, probe_b) is True
    assert same_discrete_target(probe_a, probe_c) is False
