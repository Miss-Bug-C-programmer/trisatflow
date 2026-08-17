"""Run the endogenous controller against a live SatEdgeSim REST server.

This script is intentionally a capability gate.  It never substitutes the
legacy environment and never labels a compatibility fallback as a physical
SatEdgeSim result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend, SatEdgeSimCapabilityError
from trisatflow.control.config import ControllerConfig
from trisatflow.control.controller import EndogenousReplanningController
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError
from trisatflow.planners.greedy_planner import GreedyPlanner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = SatEdgeSimClient(base_url=args.base_url, timeout=args.timeout)
    try:
        health = client.ensure_healthy()
        client.reset(seed=args.seed, wait_for_first_decision=True, wait_timeout_ms=10000)
        backend = SatEdgeSimBackend(client, compatibility_preflight=True)
    except (SatEdgeSimClientError, SatEdgeSimCapabilityError, OSError) as exc:
        print(f"BLOCKED: SatEdgeSim live smoke unavailable: {exc}")
        return 2

    required = {
        "supports_cheap_monitor": backend.capabilities.supports_cheap_monitor,
        "supports_monitor_state": backend.capabilities.supports_monitor_state,
        "supports_planner_state": backend.capabilities.supports_planner_state,
        "supports_configuration_apply": backend.capabilities.supports_configuration_apply,
        "supports_persistent_configuration": backend.capabilities.supports_persistent_configuration,
        "supports_persistent_configuration_execution": backend.capabilities.supports_persistent_configuration_execution,
        "supports_configuration_dispatch": backend.capabilities.supports_configuration_dispatch,
        "supports_scope_aware_planner_state": backend.capabilities.supports_scope_aware_planner_state,
        "supports_budget_aware_planner_state": backend.capabilities.supports_budget_aware_planner_state,
        "supports_configuration_validation": backend.capabilities.supports_configuration_validation,
        "supports_physical_decision_delay": backend.capabilities.supports_physical_decision_delay,
        "supports_advance_world": backend.capabilities.supports_advance_world,
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing:
        print(f"BLOCKED: SatEdgeSim capability contract is incomplete: {missing}")
        print(f"capabilities={backend.capabilities.to_dict()}")
        return 2

    config = ControllerConfig.from_mapping(
        {
            "monitor": {"true_cheap_required": True, "fallback_mode": "error"},
            "decision_delay": {"mode": "modeled", "modeled_components": ["solver"], "require_physical_enforcement": True},
            "planner": {"budget_levels": {"light": {"metadata": {"simulated_latency_sec": 0.01}}}},
        }
    )
    controller = EndogenousReplanningController(
        backend,
        config=config,
        planner_backends=[GreedyPlanner(source_name="satedgesim")],
        seed=args.seed,
    )
    try:
        controller.initialize(initial_plan=True)
        decision = controller.on_monitor_epoch()
    except (SatEdgeSimClientError, SatEdgeSimCapabilityError, RuntimeError, TypeError, ValueError) as exc:
        print(f"BLOCKED: live controller flow could not be verified: {exc}")
        return 2

    print(f"health={health}")
    print(f"backend_source={backend.capabilities.backend_source}")
    print(f"capabilities={backend.capabilities.to_dict()}")
    print(f"action={decision.action}")
    print(f"physical_time_sec={controller.clock.physical_time_sec}")
    print(f"physical_delay_receipt_verified={decision.delay.physical_receipt_verified if decision.delay else False}")
    print(f"future_stochastic_truth_used={bool((decision.metadata or {}).get('future_stochastic_truth_used', False))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
