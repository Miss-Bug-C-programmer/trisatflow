"""Minimal executable smoke for the endogenous replanning skeleton.

It uses a deliberately non-authoritative fake physical backend.  The output
is therefore an engineering smoke result, never a SatEdgeSim paper result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.config import ControllerConfig
from trisatflow.control.controller import EndogenousReplanningController
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.types import MonitorAcquisitionMetadata, MonitorState, PlannerState
from trisatflow.planners.greedy_planner import GreedyPlanner


class SmokeBackend:
    def __init__(self) -> None:
        self.time = 0.0
        self.monitor_count = 0
        self.applied: list[tuple[float, PersistentConfiguration]] = []
        self.capabilities = BackendCapabilities(
            supports_monitor_state=True,
            supports_planner_state=True,
            supports_configuration_apply=True,
            supports_persistent_configuration=True,
            supports_physical_decision_delay=True,
            supports_advance_world=True,
            supports_mid_transfer_contact_enforcement=False,
            backend_source="fake_smoke_backend",
            topology_source="fake_fixture",
            monitor_state_source="fake_cheap_monitor",
            authoritative_physical=False,
        )

    def current_time(self) -> float:
        return self.time

    def reset(self) -> None:
        self.time = 0.0
        self.monitor_count = 0

    def get_monitor_state(self, context=None) -> MonitorState:
        self.monitor_count += 1
        degraded = self.monitor_count >= 3
        return MonitorState(
            simulation_time=self.time,
            current_config_id=self.applied[-1][1].config_id if self.applied else None,
            current_config_version=self.applied[-1][1].version if self.applied else None,
            source_queue_summary={"source-1": 1.0 if not degraded else 8.0},
            remaining_workload_summary={"source-1": 1.0 if not degraded else 8.0},
            deadline_slack={"source-1": 8.0 if not degraded else -1.0},
            local_load_summary={"service_capacity": 10.0},
            contact_slack={"link-1": 8.0 if not degraded else -1.0},
            degradation_indicators={"queue_pressure": 0.0 if not degraded else 0.8},
            acquisition=MonitorAcquisitionMetadata(
                obs_bytes=128,
                num_queries=1,
                source="fake_cheap_monitor",
                is_true_cheap_monitor=True,
            ),
            metadata={"affected_entities": {"source_ids": {"source-1"}}, "future_stochastic_truth_used": False},
        )

    def get_planner_state(self, context=None, scope=None, budget=None) -> PlannerState:
        candidates = [
            {"sourceId": "source-1", "vmIndex": 1, "estimatedTotalDelaySec": 1.0, "resourceAllocation": {"cpu": 1.0}},
            {"sourceId": "source-2", "vmIndex": 2, "estimatedTotalDelaySec": 2.0, "resourceAllocation": {"cpu": 0.5}},
        ]
        if budget is not None and hasattr(budget, "restrict_count"):
            candidates = budget.restrict_count(candidates)
        return PlannerState(
            simulation_time=self.time,
            candidate_vms=candidates,
            detailed_resources={"node-1": {"cpu": 1.0}, "node-2": {"cpu": 0.5}},
            metadata={"backend_source": "fake_smoke_backend"},
        )

    def apply_configuration(self, configuration: PersistentConfiguration):
        self.applied.append((self.time, configuration.clone()))
        return {"accepted": True}

    def dispatch_under_configuration(self, configuration, task=None):
        return configuration.materialize_execution_rule(task or {"task_id": "source-1"})

    def advance_world(self, delta_sec: float):
        self.time += max(0.0, float(delta_sec))

    def validate_configuration(self, configuration):
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-epochs", type=int, default=3)
    args = parser.parse_args()

    backend = SmokeBackend()
    backend.reset()
    config = ControllerConfig.from_mapping(
        {
            "decision_cost": {"obs_price": 0.0, "sync_price": 0.0, "solve_price": 0.0, "signal_price": 0.0, "reconfiguration_price": 0.0},
            "decision_delay": {"mode": "modeled", "modeled_components": ["solver"]},
            "planner": {"budget_levels": {"light": {"metadata": {"simulated_latency_sec": 0.01}}}},
            "monitor": {"true_cheap_required": True},
        }
    )
    controller = EndogenousReplanningController(backend, config=config, planner_backends=[GreedyPlanner(source_name="test_backend")])
    initial = controller.initialize(initial_plan=True)
    decisions = []
    for _ in range(max(1, args.monitor_epochs)):
        decisions.append(controller.on_monitor_epoch())

    intervention = next((item for item in decisions if item.action == "INTERVENE"), None)
    print(f"backend_source={backend.capabilities.backend_source}")
    print(f"config 0 lifetime={controller.metrics.configuration_lifetimes[0] if controller.metrics.configuration_lifetimes else 0.0:.6f}")
    print(f"KEEP count={controller.metrics.keep_count}")
    print(f"intervention count={controller.metrics.num_replans}")
    print(f"selected scope={intervention.scope.to_dict() if intervention else {}}")
    print(f"selected fidelity={intervention.planner_fidelity if intervention else 'none'}")
    print(f"selected budget={intervention.planning_budget if intervention else {}}")
    print(f"decision cost={intervention.voc.estimated_candidate_cost if intervention and intervention.voc else 0.0}")
    print(f"modeled decision delay={intervention.delay.total_delay_sec if intervention and intervention.delay else 0.0}")
    print(f"physical_delay_enforced={intervention.delay.physical_delay_enforced if intervention and intervention.delay else False}")
    print(f"config 1 applied time={backend.applied[-1][0] if len(backend.applied) > 1 else 'not-applied'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
