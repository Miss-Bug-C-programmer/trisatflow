"""Compatibility wrapper around the existing GeoLeoGroundEnv.

This adapter is intentionally marked non-authoritative.  It is suitable for
unit tests, smoke tests and legacy slotwise experiments, not for claiming a
SatEdgeSim physical run.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.types import MonitorAcquisitionMetadata, MonitorState, PlannerState


class LegacyEnvBackendAdapter:
    def __init__(self, env: Any, *, source_name: str = "legacy_env") -> None:
        self.env = env
        self.source_name = source_name
        self._configuration: PersistentConfiguration | None = None
        self._dispatches = 0
        self._current_time = 0.0
        self.capabilities = BackendCapabilities(
            supports_monitor_state=True,
            supports_planner_state=True,
            supports_contact_plan=False,
            supports_configuration_apply=True,
            supports_persistent_configuration=True,
            supports_physical_decision_delay=hasattr(env, "advance_world"),
            supports_advance_world=hasattr(env, "advance_world"),
            supports_scope_aware_planner_state=False,
            supports_budget_aware_planner_state=False,
            supports_configuration_validation=True,
            supports_verified_delay_receipt=hasattr(env, "advance_world"),
            supports_mid_transfer_contact_enforcement=False,
            backend_source=source_name,
            topology_source="analytic_or_trace_legacy",
            monitor_state_source=source_name,
            authoritative_physical=False,
            metadata={"mode": "legacy_slotwise", "truth_authoritative": False},
        )

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        self._current_time = 0.0
        self._dispatches = 0
        return result

    def current_time(self) -> float:
        return float(getattr(self.env, "t", self._current_time))

    def get_monitor_state(self, context: Any | None = None) -> MonitorState:
        queue = getattr(self.env, "queue", None)
        queue_values = self._vector_summary(queue)
        cfg = self._configuration
        cfg_id = getattr(cfg, "config_id", None)
        cfg_version = getattr(cfg, "version", None)
        metadata = {
            "backend_source": self.source_name,
            "topology_source": self.capabilities.topology_source,
            "future_stochastic_truth_used": False,
            "lower_bound_service_capacity": self._capacity_hint(),
        }
        if cfg is not None:
            metadata["current_configuration"] = cfg.to_dict()
        raw = {
            "queue": queue_values,
            "time": self.current_time(),
            "config": cfg_id,
        }
        return MonitorState(
            simulation_time=self.current_time(),
            current_config_id=cfg_id,
            current_config_version=cfg_version,
            local_queue_summary={str(i): value for i, value in enumerate(queue_values)},
            source_queue_summary={str(i): value for i, value in enumerate(queue_values)},
            remaining_workload_summary={str(i): value for i, value in enumerate(queue_values)},
            deadline_slack={str(i): float(getattr(getattr(self.env, "cfg", None), "deadline_threshold", 0.0)) - value for i, value in enumerate(queue_values)},
            local_load_summary={"service_capacity": self._capacity_hint()},
            remaining_contact_lifetime={},
            contact_slack={},
            degradation_indicators={"queue_pressure": min(1.0, sum(queue_values) / max(1.0, self._max_queue()))},
            acquisition=MonitorAcquisitionMetadata(
                obs_bytes=len(json.dumps(raw)),
                num_queries=1,
                source=self.source_name,
                is_true_cheap_monitor=True,
            ),
            metadata=metadata,
        )

    def get_planner_state(self, context: Any | None = None, scope: Any | None = None, budget: Any | None = None) -> PlannerState:
        n = int(getattr(getattr(self.env, "cfg", None), "n_leo", 1))
        candidates = [{"sourceId": str(i), "vmIndex": i, "estimatedTotalDelaySec": float(i), "feasible": True} for i in range(n)]
        if budget is not None and hasattr(budget, "restrict_count"):
            candidates = budget.restrict_count(candidates)
        return PlannerState(
            simulation_time=self.current_time(),
            candidate_vms=candidates,
            queue_load={"source_queue": self._vector_summary(getattr(self.env, "queue", None))},
            detailed_resources={str(i): {"cpu": self._capacity_hint()} for i in range(n)},
            topology={"source": self.capabilities.topology_source},
            acquisition=MonitorAcquisitionMetadata(
                obs_bytes=len(json.dumps(candidates)), num_queries=1, source=self.source_name, is_true_cheap_monitor=False
            ),
            metadata={"backend_source": self.source_name, "future_stochastic_truth_used": False},
        )

    def get_contact_forecast(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"available": None, "source": "legacy_backend_not_authoritative"}

    def get_current_topology(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"source": self.capabilities.topology_source}

    def materialize_current_configuration(self, configuration: PersistentConfiguration, task: Any | None = None) -> Any:
        return configuration.materialize_execution_rule(task or {"task_id": "default"})

    def apply_configuration(self, configuration: PersistentConfiguration) -> PersistentConfiguration:
        self._configuration = configuration
        return configuration

    def dispatch_under_configuration(self, configuration: PersistentConfiguration, task: Any | None = None) -> Any:
        self._dispatches += 1
        return self.materialize_current_configuration(configuration, task)

    def advance_world(self, delta_sec: float) -> None:
        if hasattr(self.env, "advance_world"):
            self.env.advance_world(float(delta_sec))
        else:
            self._current_time += max(0.0, float(delta_sec))

    def validate_configuration(self, configuration: PersistentConfiguration) -> bool:
        return isinstance(configuration, PersistentConfiguration)

    @staticmethod
    def _vector_summary(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            result: list[float] = []
            for item in value:
                try:
                    result.append(float(item))
                except (TypeError, ValueError):
                    continue
            return result
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []

    def _max_queue(self) -> float:
        return float(getattr(getattr(self.env, "cfg", None), "max_queue", 1.0))

    def _capacity_hint(self) -> float:
        return float(getattr(getattr(self.env, "cfg", None), "leo_cpu_capacity", 1.0))
