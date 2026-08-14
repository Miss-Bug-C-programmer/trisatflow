"""Capability-aware adapter built on the existing SatEdgeSim REST client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.types import MonitorAcquisitionMetadata, MonitorState, PlannerState
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError


class SatEdgeSimCapabilityError(RuntimeError):
    pass


@dataclass
class SatEdgeSimBackend:
    client: SatEdgeSimClient
    compatibility_preflight: bool = True

    def __post_init__(self) -> None:
        self._last_state: dict[str, Any] = {}
        self._configuration: PersistentConfiguration | None = None
        self._capabilities = self._detect_capabilities()

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def reset(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.client.reset(**kwargs)

    def current_time(self) -> float:
        state = self._last_state or self.client.get_state()
        return self._extract_time(state)

    def get_monitor_state(self, context: Any | None = None) -> MonitorState:
        state, source, true_cheap = self._optional_or_compat("/get_monitor_state", method="GET")
        if state is None:
            state = self.client.get_state()
            source = "compatibility_preflight"
            true_cheap = False
        self._last_state = dict(state)
        return self._monitor_from_payload(state, source=source, true_cheap=true_cheap)

    def get_planner_state(self, context: Any | None = None, scope: Any | None = None, budget: Any | None = None) -> PlannerState:
        if scope is not None and self.capabilities.supports_scope_aware_planner_state:
            state = self.client._request("POST", "/get_planner_state_scoped", json={"scope": _to_dict(scope)})
            source = "/get_planner_state_scoped"
        elif budget is not None and self.capabilities.supports_budget_aware_planner_state:
            state = self.client._request(
                "POST", "/get_planner_state_budgeted", json={"budget": _to_dict(budget)}
            )
            source = "/get_planner_state_budgeted"
        else:
            state, source, _ = self._optional_or_compat("/get_planner_state", method="GET")
        if state is None:
            state = self._last_state or self.client.get_state()
            source = "compatibility_preflight_get_state"
        candidates = list(state.get("candidateVms", state.get("candidate_vms", [])) or [])
        if budget is not None and hasattr(budget, "restrict_count"):
            candidates = budget.restrict_count(candidates)
        return PlannerState(
            simulation_time=self._extract_time(state),
            candidate_vms=[dict(item) for item in candidates if isinstance(item, Mapping)],
            queue_load=dict(state.get("queue", state.get("queueLoad", {})) or {}),
            detailed_resources=dict(state.get("resources", state.get("resourceState", {})) or {}),
            topology=dict(state.get("topology", {}) or {}),
            candidate_links=dict(state.get("candidateLinks", {}) or {}),
            candidate_execution_costs={str(i): item for i, item in enumerate(candidates)},
            acquisition=MonitorAcquisitionMetadata(
                obs_bytes=len(json.dumps(state, default=str).encode("utf-8")),
                num_queries=1,
                source=source,
                is_true_cheap_monitor=False,
            ),
            metadata={
                "backend_source": "satedgesim",
                "topology_source": self.capabilities.topology_source,
                "future_stochastic_truth_used": False,
            },
        )

    def get_contact_forecast(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.capabilities.supports_contact_plan:
            raise SatEdgeSimCapabilityError("SatEdgeSim v22 contact-plan endpoint is not available")
        return self.client._request("POST", "/topology/contact_plan", json=dict(request))

    def get_current_topology(self) -> Mapping[str, Any]:
        if not self._endpoint_available("/topology/current", method="GET"):
            raise SatEdgeSimCapabilityError("SatEdgeSim topology endpoint is not available")
        return self.client._request("GET", "/topology/current")

    def materialize_current_configuration(self, configuration: PersistentConfiguration, task: Any | None = None) -> Any:
        return configuration.materialize_execution_rule(task or {"task_id": "default"})

    def apply_configuration(self, configuration: PersistentConfiguration) -> Mapping[str, Any]:
        if not self.capabilities.supports_configuration_apply:
            raise SatEdgeSimCapabilityError(
                "SatEdgeSim v22 exposes action dispatch but no persistent configuration apply endpoint"
            )
        payload = configuration.to_dict()
        result = self.client._request("POST", "/configuration/apply", json={"configuration": payload})
        if isinstance(result, Mapping) and "accepted" in result and not bool(result.get("accepted")):
            raise SatEdgeSimCapabilityError(f"SatEdgeSim rejected configuration apply: {result}")
        self._configuration = configuration
        return result

    def dispatch_under_configuration(self, configuration: PersistentConfiguration, task: Any | None = None) -> Mapping[str, Any]:
        if not bool(self.capabilities.metadata.get("endpoint_probe", {}).get("/configuration/dispatch", False)):
            raise SatEdgeSimCapabilityError(
                "SatEdgeSim exposes no native persistent-configuration dispatch endpoint; "
                "apply_action cannot be relabeled as execution under Γ_k"
            )
        return self.client._request(
            "POST",
            "/configuration/dispatch",
            json={"configuration": configuration.to_dict(), "task": task},
        )

    def advance_world(self, delta_sec: float) -> Mapping[str, Any]:
        if not self.capabilities.supports_advance_world:
            raise SatEdgeSimCapabilityError(
                "SatEdgeSim v22 has no explicit advance-world/physical-decision-delay endpoint; "
                "wall-clock sleep is not a physical delay substitute"
            )
        result = self.client._request("POST", "/advance_world", json={"deltaSec": float(delta_sec)})
        # A pre-advance cache must not be used as proof of physical evolution.
        self._last_state = dict(result) if isinstance(result, Mapping) and _has_time(result) else {}
        return result

    def validate_configuration(self, configuration: PersistentConfiguration) -> Mapping[str, Any]:
        if not self.capabilities.supports_configuration_validation:
            raise SatEdgeSimCapabilityError(
                "SatEdgeSim v22 exposes no configuration validation endpoint; "
                "post-delay acceptance cannot be claimed"
            )
        result = self.client._request(
            "POST", "/configuration/validate", json={"configuration": configuration.to_dict()}
        )
        if not isinstance(result, Mapping):
            raise SatEdgeSimCapabilityError("SatEdgeSim configuration validation returned a non-structured receipt")
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_source": self.capabilities.backend_source,
            "topology_source": self.capabilities.topology_source,
            "monitor_state_source": self.capabilities.monitor_state_source,
            "physical_delay_enforced": self.capabilities.supports_physical_decision_delay,
            "capabilities": self.capabilities.to_dict(),
        }

    def _detect_capabilities(self) -> BackendCapabilities:
        endpoints = {
            "/get_monitor_state": self._endpoint_available("/get_monitor_state", method="GET"),
            "/get_planner_state": self._endpoint_available("/get_planner_state", method="GET"),
            "/topology/contact_plan": self._endpoint_available("/topology/contact_plan", method="POST"),
            "/topology/current": self._endpoint_available("/topology/current", method="GET"),
            "/configuration/apply": self._endpoint_available("/configuration/apply", method="POST"),
            "/configuration/dispatch": self._endpoint_available("/configuration/dispatch", method="POST"),
            "/configuration/validate": self._endpoint_available("/configuration/validate", method="POST"),
            "/advance_world": self._endpoint_available("/advance_world", method="POST"),
            "/get_planner_state_scoped": self._endpoint_available("/get_planner_state_scoped", method="POST"),
            "/get_planner_state_budgeted": self._endpoint_available("/get_planner_state_budgeted", method="POST"),
        }
        return BackendCapabilities(
            supports_monitor_state=endpoints["/get_monitor_state"],
            supports_planner_state=endpoints["/get_planner_state"],
            supports_contact_plan=endpoints["/topology/contact_plan"],
            supports_configuration_apply=endpoints["/configuration/apply"],
            supports_persistent_configuration=(endpoints["/configuration/apply"] and endpoints["/configuration/dispatch"]),
            supports_physical_decision_delay=endpoints["/advance_world"],
            supports_advance_world=endpoints["/advance_world"],
            supports_scope_aware_planner_state=endpoints["/get_planner_state_scoped"],
            supports_budget_aware_planner_state=endpoints["/get_planner_state_budgeted"],
            supports_configuration_validation=endpoints["/configuration/validate"],
            # Endpoint discovery is not proof that the server returns a
            # verifiable physical-time receipt.  The delay model verifies
            # before/after world time at runtime instead.
            supports_verified_delay_receipt=False,
            supports_mid_transfer_contact_enforcement=False,
            supported_budget_dimensions={
                value
                for value, enabled in (
                    ("scope", endpoints["/get_planner_state_scoped"]),
                    ("budget", endpoints["/get_planner_state_budgeted"]),
                )
                if enabled
            },
            backend_source="satedgesim",
            topology_source="deterministic_contact_plan" if endpoints["/topology/contact_plan"] else "get_state_candidate_fallback",
            monitor_state_source="/get_monitor_state" if endpoints["/get_monitor_state"] else "compatibility_preflight",
            authoritative_physical=True,
            metadata={
                "endpoint_probe": endpoints,
                "compatibility_preflight": bool(self.compatibility_preflight),
            },
        )

    def _endpoint_available(self, path: str, *, method: str) -> bool:
        if not self.compatibility_preflight:
            return False
        try:
            self.client._request(method, path, **({"json": {}} if method == "POST" else {}))
            return True
        except SatEdgeSimClientError as exc:
            # Connection failures and 5xx responses are not evidence that an
            # endpoint exists.  A 400 can still be useful evidence for a POST
            # endpoint because the route may reject an intentionally empty
            # probe payload.
            return exc.status_code is not None and 200 <= int(exc.status_code) < 500 and exc.status_code not in {404, 405}
        except Exception:
            return False

    def _optional_or_compat(self, path: str, *, method: str) -> tuple[dict[str, Any] | None, str, bool]:
        endpoint_supported = (
            (path == "/get_monitor_state" and self.capabilities.supports_monitor_state)
            or (path == "/get_planner_state" and self.capabilities.supports_planner_state)
        )
        if endpoint_supported:
            try:
                payload = self.client._request(method, path)
                return payload, path, path == "/get_monitor_state"
            except SatEdgeSimClientError:
                pass
        return None, "compatibility_preflight", False

    @staticmethod
    def _extract_time(state: Mapping[str, Any]) -> float:
        for key in ("simulationTimeSec", "simulation_time", "simulationTime", "time"):
            try:
                return float(state.get(key, 0.0))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _monitor_from_payload(self, state: Mapping[str, Any], *, source: str, true_cheap: bool) -> MonitorState:
        payload_bytes = len(json.dumps(state, default=str).encode("utf-8"))
        queue = state.get("queueSummary", state.get("queue", {})) or {}
        if not isinstance(queue, Mapping):
            queue = {"total": float(queue)}
        task = state.get("task", {}) or {}
        deadline = state.get("deadlineSlack", {}) or {}
        if not isinstance(deadline, Mapping):
            deadline = {str(task.get("taskId", "current")): float(deadline)}
        contact = state.get("contactSlack", {}) or {}
        if not isinstance(contact, Mapping):
            contact = {"current": float(contact)}
        acquisition = MonitorAcquisitionMetadata(
            obs_bytes=payload_bytes,
            num_queries=1,
            latency_sec=0.0,
            source=source,
            is_true_cheap_monitor=true_cheap,
            monitor_http_calls=1,
            monitor_bytes=payload_bytes,
            response_bytes=payload_bytes,
            cheap_monitor_verified=true_cheap,
        )
        return MonitorState(
            simulation_time=self._extract_time(state),
            current_config_id=(self._configuration.config_id if self._configuration else state.get("configId")),
            current_config_version=(self._configuration.version if self._configuration else state.get("configVersion")),
            local_queue_summary={str(k): float(v) for k, v in queue.items() if _is_number(v)},
            source_queue_summary={str(k): float(v) for k, v in queue.items() if _is_number(v)},
            remaining_workload_summary={str(k): float(v) for k, v in (state.get("remainingWorkload", queue) or {}).items() if _is_number(v)},
            deadline_slack={str(k): float(v) for k, v in deadline.items() if _is_number(v)},
            local_load_summary={str(k): float(v) for k, v in (state.get("loadSummary", {}) or {}).items() if _is_number(v)},
            remaining_contact_lifetime={str(k): float(v) for k, v in (state.get("remainingContactLifetime", {}) or {}).items() if _is_number(v)},
            next_contact_summary=dict(state.get("nextContact", {}) or {}),
            contact_slack={str(k): float(v) for k, v in contact.items() if _is_number(v)},
            small_neighborhood_state=dict(state.get("smallNeighborhood", {}) or {}),
            cached_state=dict(state.get("cachedState", {}) or {}),
            prediction_uncertainty={str(k): float(v) for k, v in (state.get("predictionUncertainty", {}) or {}).items() if _is_number(v)},
            degradation_indicators={str(k): float(v) for k, v in (state.get("degradationIndicators", {}) or {}).items() if _is_number(v)},
            acquisition=acquisition,
            metadata={
                "backend_source": "satedgesim",
                "topology_source": self.capabilities.topology_source,
                "monitor_source": source,
                "future_stochastic_truth_used": False,
                "compatibility_preflight": source == "compatibility_preflight",
            },
        )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _has_time(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("simulationTimeSec", "simulation_time", "simulationTime", "time"))


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        raw = value.to_dict()
        return dict(raw) if isinstance(raw, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}
