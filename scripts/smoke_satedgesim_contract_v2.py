"""Capability-gated cross-repository SatEdgeSim contract v2 smoke test.

The Java REST server must already be running. This script exercises the real
TriSatFlow adapter, not a fake backend or a compatibility endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope
from trisatflow.control.types import PlanningBudget
from trisatflow.satedgesim_eval.client import SatEdgeSimClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--devices", type=int, default=2)
    args = parser.parse_args()

    client = SatEdgeSimClient(args.base_url, timeout=30.0)
    client.reset(
        devices_count=args.devices,
        wait_for_first_decision=True,
        wait_timeout_ms=10000,
        extra={"simulationTimeMinutes": 1.0, "tasksGenerationRate": 1},
    )
    backend = SatEdgeSimBackend(client, compatibility_preflight=True)
    required = (
        "supports_cheap_monitor",
        "supports_scoped_planner_state",
        "supports_budget_aware_planner_state",
        "supports_configuration_apply",
        "supports_persistent_configuration_execution",
        "supports_configuration_dispatch",
        "supports_configuration_validation",
        "supports_physical_decision_delay",
        "supports_advance_world",
    )
    missing = [name for name in required if not getattr(backend.capabilities, name)]
    if missing:
        raise RuntimeError(f"contract capabilities missing: {missing}")

    monitor = backend.get_monitor_state()
    if not monitor.acquisition.cheap_monitor_verified:
        raise RuntimeError(f"cheap monitor proof failed: {monitor.to_dict()}")

    planner = backend.get_planner_state(
        scope=ReconfigurationScope(),
        budget=PlanningBudget(max_candidate_count=2),
    )
    if not planner.metadata.get("budget_applied_during_acquisition") or planner.metadata.get("post_filter_only"):
        raise RuntimeError(f"planner acquisition proof failed: {planner.metadata}")
    feasible = next(
        (item for item in planner.candidate_vms if item.get("isFeasible", item.get("feasible", False))),
        planner.candidate_vms[0],
    )
    configuration = PersistentConfiguration(
        config_id="contract-v2-smoke",
        version=1,
        reusable_rules={
            "default": {
                "selector": {},
                "assignment": {"abstractAction": feasible.get("abstractAction", 0)},
            }
        },
    )
    validation = backend.validate_configuration(configuration)
    applied = backend.apply_configuration(configuration)
    dispatched = backend.dispatch_under_configuration(configuration)
    advance = backend.advance_world(0.01)
    if not validation.get("accepted") or not applied.get("accepted") or not dispatched.get("accepted"):
        raise RuntimeError({"validation": validation, "apply": applied, "dispatch": dispatched})
    if not advance.get("accepted") or not advance.get("physicalClockAdvanced"):
        raise RuntimeError(f"physical advance proof failed: {advance}")

    print(
        json.dumps(
            {
                "contract": backend.capabilities.to_dict(),
                "cheap_monitor": monitor.acquisition.to_dict(),
                "planner": planner.metadata,
                "configuration_validation": validation,
                "configuration_apply": applied,
                "configuration_dispatch": dispatched,
                "advance_world": advance,
                "mid_transfer_capability": backend.capabilities.supports_mid_transfer_contact_enforcement,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
