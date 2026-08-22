"""Real cross-repository protocol smoke for the TriSatFlow/SatEdgeSim loop.

This script intentionally requires a live SatEdgeSim REST server.  It never
constructs a fake HTTP response or substitutes a legacy backend.  It checks
the physical advance receipt and the native intervention evidence round-trip
for a small, deterministic session.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.satedgesim_eval.client import SatEdgeSimClient


def request(base: str, method: str, path: str, **kwargs: object) -> dict[str, object]:
    response = requests.request(method, f"{base}{path}", timeout=30.0, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    capabilities = request(base, "GET", "/capabilities")
    required = (
        "supportsConfigurationPatch", "supportsActualAppliedInterventionEvidence",
        "supportsPhysicalDecisionDelay", "supportsAdvanceWorld",
        "supportsConfigurationValidation", "supportsPersistentNativeResourceActuation",
    )
    missing = [key for key in required if not capabilities.get(key)]
    if missing:
        raise RuntimeError(f"authoritative integration capabilities missing: {missing}")

    reset = request(
        base,
        "POST",
        "/reset",
        json={
            "devicesCount": 2,
            "waitForFirstDecision": True,
            "waitTimeoutMs": 10000,
            "simulationTimeMinutes": 0.2,
            "tasksGenerationRate": 1,
            "strictPhysicalClaims": True,
        },
    )
    if reset.get("status") not in {"WAITING_FOR_ACTION", "RUNNING"}:
        raise RuntimeError(f"reset did not create a live decision session: {reset}")

    # Use the real TriSatFlow adapter for monitor/configuration/patch calls;
    # raw requests below are only used to query the server evidence endpoints.
    backend = SatEdgeSimBackend(SatEdgeSimClient(base, timeout=30.0), compatibility_preflight=False)
    monitor_object = backend.get_monitor_state()
    monitor = monitor_object.to_dict()
    monitor_wire = request(base, "GET", "/get_monitor_state")
    if not monitor_object.acquisition.cheap_monitor_verified or monitor_wire.get("payloadKind") != "cheap_monitor" or monitor_wire.get("containsFutureStochasticState"):
        raise RuntimeError(f"cheap monitor contract failed: {monitor}")
    task_id = str(monitor_wire.get("currentTaskId"))
    if task_id in {"None", "-1"}:
        raise RuntimeError(f"no current native task for integration smoke: {monitor}")

    state = request(base, "GET", "/get_state")
    candidates = [item for item in state.get("candidateVms", []) if isinstance(item, dict)]
    local = next((item for item in candidates if item.get("isLocalToSource") and item.get("isFeasible", item.get("feasible"))), None)
    if local is None:
        local = next((item for item in candidates if item.get("isFeasible", item.get("feasible"))), None)
    if local is None:
        raise RuntimeError("no feasible native candidate in live planner state")
    assignment = {"targetVmIndex": int(float(local["vmIndex"]))}

    # Install one persistent execution configuration for the actual pending
    # task, then resolve that task through the canonical dispatch endpoint.
    initial_configuration = PersistentConfiguration(
        config_id="integration-config",
        version=1,
        assignments={task_id: assignment},
        metadata={"integration": "live_cross_repo", "taskId": task_id},
    )
    configuration = backend.apply_configuration(initial_configuration)
    if not configuration.get("accepted"):
        raise RuntimeError(f"initial persistent configuration rejected: {configuration}")
    dispatched = backend.dispatch_under_configuration(initial_configuration, {"taskId": task_id})
    if not dispatched.get("accepted"):
        raise RuntimeError(f"native persistent dispatch rejected: {dispatched}")

    # Give the native CloudSim task a physical interval to progress.  If the
    # simulation has already reached the next external decision, install the
    # current task's real local binding and use the control-epoch route; that
    # route is the canonical pause-for-revalidation path.
    advance = request(base, "POST", "/advance_world", json={"deltaSec": 0.5})
    if not advance.get("accepted") and advance.get("reason") == "simulation_waiting_for_decision":
        pending = request(base, "GET", "/get_monitor_state")
        pending_task_id = str(pending.get("currentTaskId"))
        pending_state = request(base, "GET", "/get_state")
        pending_candidates = [item for item in pending_state.get("candidateVms", []) if isinstance(item, dict)]
        pending_local = next((item for item in pending_candidates if item.get("isLocalToSource") and item.get("isFeasible", item.get("feasible"))), None)
        if pending_local is None:
            pending_local = next((item for item in pending_candidates if item.get("isFeasible", item.get("feasible"))), None)
        if pending_local is None or pending_task_id in {"None", "-1"}:
            raise RuntimeError(f"pending task cannot be resolved for control-epoch advance: {pending}")
        current_before_epoch = request(base, "GET", "/configuration/current")
        epoch_object = PersistentConfiguration(
            config_id="integration-config-epoch",
            version=int(current_before_epoch.get("version", 0)) + 1,
            assignments={pending_task_id: {"targetVmIndex": int(float(pending_local["vmIndex"]))}},
        )
        epoch_configuration = backend.apply_configuration(epoch_object)
        if not epoch_configuration.get("accepted"):
            raise RuntimeError(f"control-epoch configuration rejected: {epoch_configuration}")
        advance = backend.advance_control_epoch(0.5)
    if not advance.get("accepted") or not advance.get("physicalClockAdvanced"):
        raise RuntimeError(f"physical advance failed: {advance}")
    before = advance.get("physicalProgressBefore")
    after = advance.get("physicalProgressAfter")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError(f"physical progression evidence missing: {advance}")
    if before == after:
        raise RuntimeError(f"physical delay advanced only a clock in this qualifying run: {advance}")

    current = request(base, "GET", "/configuration/current")
    current_version = int(current.get("version", 0))
    current_world = int(current.get("worldVersion", 0))
    backend.get_monitor_state()
    current_object = backend._configuration
    if current_object is None or current_object.version != current_version:
        current_object = PersistentConfiguration(
            config_id=str(current.get("configId")),
            version=current_version,
            assignments=dict((current.get("configuration") or {}).get("assignments", {}) or {}),
            reusable_rules=dict((current.get("configuration") or {}).get("reusableRules", {}) or {}),
        )
    intervention_id = "live-integration-intervention-1"
    scope = {"task_ids": [task_id], "source_ids": [], "node_ids": [], "link_ids": [], "route_ids": [], "resource_keys": []}

    # This is a real selective persistent-rule patch.  It changes only the
    # task-scoped rule and returns native applied/rejected evidence.
    proposed_object = current_object.clone(version=current_version + 1)
    proposed_object.reusable_rules = {
        **dict(current_object.reusable_rules or {}),
        task_id: {"selector": {"taskId": task_id}, "assignment": assignment},
    }
    validation = backend.validate_configuration(
        proposed_object,
        current_configuration=current_object,
        scope=ReconfigurationScope(task_ids={task_id}),
        observed_world_version=current_world,
        observed_control_epoch=1,
        planning_delay={"mode": "live_smoke", "physical": True},
        intervention_id=intervention_id,
    )
    if not validation.get("accepted"):
        raise RuntimeError(f"live adapter revalidation rejected valid patch: {validation}")
    patch = backend.apply_configuration_patch(
        current_object,
        proposed_object,
        ReconfigurationScope(task_ids={task_id}),
        preserve_resume_recompute={"preserve": True, "resume": True, "recompute": False},
        acquisition_epoch=1,
        intervention_id=intervention_id,
        observed_world_version=current_world,
        observed_control_epoch=1,
        revalidated_world_version=int(validation.get("worldVersion", current_world)),
        planning_delay={"mode": "live_smoke", "physical": True},
        acquisition_metadata={"source": "live_cheap_monitor"},
    )
    if not patch.get("accepted") or patch.get("decisionStatus") != "APPLY":
        raise RuntimeError(f"valid selective patch was not applied: {patch}")
    if not patch.get("evidenceId") or not patch.get("scopeInvariantSatisfied"):
        raise RuntimeError(f"applied evidence is incomplete: {patch}")

    # The old version must not be applied again after the real world/config
    # changed.  The response is evidence, not an exception-only side effect.
    stale_object = current_object.clone(version=current_version + 1)
    stale_object.reusable_rules = {
        **dict(current_object.reusable_rules or {}),
        task_id: {"selector": {"taskId": task_id}, "assignment": {"targetVmIndex": 0}},
    }
    stale = backend.apply_configuration_patch(
        current_object,
        stale_object,
        ReconfigurationScope(task_ids={task_id}),
        intervention_id=intervention_id + "-stale",
        observed_world_version=current_world,
        revalidated_world_version=None,
    )
    if stale.get("accepted") or stale.get("decisionStatus") != "REJECT_STALE":
        raise RuntimeError(f"stale patch was not rejected: {stale}")

    applied_object = backend._configuration or proposed_object
    unsupported_object = applied_object.clone(version=applied_object.version + 1)
    unsupported_object.routes = {"route-not-supported": {"nextHop": "n2"}}
    unsupported = backend.apply_configuration_patch(
        applied_object,
        unsupported_object,
        ReconfigurationScope(task_ids={task_id}, route_ids={"route-not-supported"}),
        intervention_id=intervention_id + "-unsupported",
        observed_world_version=int(patch.get("worldVersion", current_world)),
        revalidated_world_version=int(patch.get("worldVersion", current_world)),
    )
    if unsupported.get("accepted") or not unsupported.get("rejectedChanges"):
        raise RuntimeError(f"unsupported partial patch did not fail closed: {unsupported}")

    evidence = request(base, "GET", "/intervention_evidence")
    events = request(base, "GET", "/protocol_events")
    validation = request(base, "GET", "/dynamic_validation/report")
    if not any(item.get("evidenceId") == patch.get("evidenceId") for item in evidence.get("evidence", [])):
        raise RuntimeError("applied intervention evidence was not queryable by evidenceId")
    if not any(item.get("payload", {}).get("interventionId") == intervention_id for item in events.get("events", [])):
        raise RuntimeError("intervention identity was not preserved in protocol events")

    print(json.dumps({
        "status": "INTEGRATION PASS",
        "taskId": task_id,
        "physicalAdvance": {"before": before, "after": after},
        "appliedPatch": patch,
        "stalePatch": stale,
        "unsupportedPatch": unsupported,
        "evidenceCount": evidence.get("evidenceCount"),
        "eventCount": events.get("eventCount"),
        "dynamicValidation": validation,
    }, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
