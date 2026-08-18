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
        result = self.client.reset(**kwargs)
        # A reset creates a new authoritative physical session.  Do not let
        # an expected configuration from the previous session masquerade as
        # the current controller state during the next monitor epoch.
        self._last_state = {}
        self._configuration = None
        return result

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
        context_metadata = getattr(context, "metadata", {}) or {}
        payload = {
            "scope": _to_dict(scope) if scope is not None else {},
            "budget": _to_dict(budget) if budget is not None else {},
            "fidelityHint": context_metadata.get("planner_fidelity") if isinstance(context_metadata, Mapping) else None,
        }
        if self.capabilities.supports_planner_state:
            state = self.client._request("POST", "/get_planner_state", json=payload)
            source = "/get_planner_state"
        else:
            state, source, _ = self._optional_or_compat("/get_planner_state", method="GET")
            if state is None:
                state = self._last_state or self.client.get_state()
                source = "compatibility_preflight_get_state"
        candidates = list(state.get("candidateVms", state.get("candidate_vms", [])) or [])
        acquisition_payload = state.get("acquisition", {}) or {}
        requested_scope = state.get("requestedScope", payload["scope"])
        requested_budget = state.get("requestedBudget", payload["budget"])
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
                request_bytes=len(json.dumps(payload, default=str).encode("utf-8")),
                response_bytes=len(json.dumps(state, default=str).encode("utf-8")),
                entity_count=len(candidates),
            ),
            metadata={
                "backend_source": "satedgesim",
                "topology_source": self.capabilities.topology_source,
                "future_stochastic_truth_used": False,
                "requested_scope": requested_scope,
                "requested_budget": requested_budget,
                "applied_scope": state.get("appliedScope", requested_scope),
                "applied_budget": state.get("appliedBudget", requested_budget),
                "scope_restriction_applied": bool(state.get("scopeRestrictionApplied", False)),
                "budget_restriction_applied": bool(state.get("budgetRestrictionApplied", False)),
                "budget_applied_during_acquisition": bool(state.get("budgetAppliedDuringAcquisition", False)),
                "post_filter_only": bool(state.get("postFilterOnly", False)),
                "full_state_equivalent": bool(state.get("fullStateEquivalent", False)),
                "candidate_count_before_restriction": state.get("candidateCountBeforeRestriction"),
                "candidate_count_after_restriction": state.get("candidateCountAfterRestriction", len(candidates)),
                "acquisition": acquisition_payload,
            },
        )

    def get_contact_forecast(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.capabilities.supports_contact_plan:
            raise SatEdgeSimCapabilityError("SatEdgeSim v22 contact-plan endpoint is not available")
        return self.client._request("POST", "/topology/contact_plan", json=dict(request))

    def get_current_topology(self) -> Mapping[str, Any]:
        if not self.capabilities.supports_topology_snapshot:
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
        if not self.capabilities.supports_configuration_dispatch:
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
        # The backend resumes after reaching the target.  Do not retain the
        # target receipt as a stale current-time cache; the next read must
        # observe the live CloudSim state.
        self._last_state = {}
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
        try:
            declared = self.client._request("GET", "/capabilities")
            if isinstance(declared, Mapping) and declared.get("controlPhysicalContractVersion"):
                budget_dimensions = declared.get("budgetDimensions", []) or []
                scope_dimensions = declared.get("scopeDimensions", []) or []
                persistent_dimensions = declared.get("persistentRuleDimensions", []) or []
                return BackendCapabilities(
                    server_version=str(declared.get("serverVersion", "unknown")),
                    control_physical_contract_version=str(declared.get("controlPhysicalContractVersion", "unknown")),
                    supports_cheap_monitor=bool(declared.get("supportsCheapMonitor", False)),
                    supports_monitor_state=bool(declared.get("supportsCheapMonitor", declared.get("supportsMonitorState", False))),
                    supports_planner_state=bool(declared.get("supportsScopedPlannerState", declared.get("supportsPlannerState", False))),
                    supports_scoped_planner_state=bool(declared.get("supportsScopedPlannerState", False)),
                    supports_budget_aware_planner_state=bool(declared.get("supportsBudgetAwarePlannerState", False)),
                    supports_scope_aware_planner_state=bool(declared.get("supportsScopedPlannerState", False)),
                    supports_contact_plan=bool(declared.get("supportsContactPlan", False)),
                    supports_topology_snapshot=bool(declared.get("supportsTopologySnapshot", False)),
                    supports_configuration_apply=bool(declared.get("supportsConfigurationApply", False)),
                    supports_persistent_configuration=bool(declared.get("supportsPersistentConfigurationExecution", False)),
                    supports_persistent_configuration_execution=bool(declared.get("supportsPersistentConfigurationExecution", False)),
                    supports_persistent_native_resource_actuation=bool(declared.get("supportsPersistentNativeResourceActuation", False)),
                    supports_persistent_route_actuation=bool(declared.get("supportsPersistentRouteActuation", False)),
                    supports_configuration_dispatch=bool(declared.get("supportsConfigurationDispatch", False)),
                    supports_physical_decision_delay=bool(declared.get("supportsPhysicalDecisionDelay", False)),
                    supports_advance_world=bool(declared.get("supportsAdvanceWorld", False)),
                    supports_configuration_validation=bool(declared.get("supportsConfigurationValidation", False)),
                    supports_mid_transfer_contact_enforcement=bool(declared.get("supportsMidTransferContactEnforcement", False)),
                    supports_verified_delay_receipt=False,
                    future_stochastic_truth_exposed=bool(declared.get("futureStochasticTruthExposed", False)),
                    physical_decision_delay_semantics_version=str(declared.get("physicalDecisionDelaySemanticsVersion", "unknown")),
                    configuration_semantics_version=str(declared.get("configurationSemanticsVersion", "unknown")),
                    scope_dimensions={str(value) for value in scope_dimensions},
                    supported_budget_dimensions={str(value) for value in budget_dimensions},
                    persistent_rule_dimensions={str(value) for value in persistent_dimensions},
                    backend_source="satedgesim",
                    topology_source=str(declared.get("topologySource", "unknown")),
                    monitor_state_source=str(declared.get("monitorSource", "unknown")),
                    authoritative_physical=bool(declared.get("authoritativePhysicalBackend", False)),
                    metadata={"capability_declaration": dict(declared), "compatibility_preflight": bool(self.compatibility_preflight)},
                )
        except Exception:
            # Older SatEdgeSim builds have no declaration endpoint.  Keep the
            # probe path as an explicitly labelled compatibility fallback.
            pass
        endpoints = {
            "/get_monitor_state": self._endpoint_available("/get_monitor_state", method="GET"),
            "/get_planner_state": self._endpoint_available("/get_planner_state", method="GET"),
            "/topology/contact_plan": self._endpoint_available("/topology/contact_plan", method="POST"),
            "/topology/current": self._endpoint_available("/topology/current", method="GET"),
            "/configuration/apply": self._endpoint_available("/configuration/apply", method="POST"),
            "/configuration/dispatch": self._endpoint_available("/configuration/dispatch", method="POST"),
            "/configuration/validate": self._endpoint_available("/configuration/validate", method="POST"),
            "/advance_world": self._endpoint_available("/advance_world", method="POST"),
        }
        return BackendCapabilities(
            server_version="legacy-probe",
            control_physical_contract_version="legacy",
            supports_cheap_monitor=endpoints["/get_monitor_state"],
            supports_monitor_state=endpoints["/get_monitor_state"],
            supports_planner_state=endpoints["/get_planner_state"],
            supports_scoped_planner_state=False,
            supports_budget_aware_planner_state=False,
            supports_contact_plan=endpoints["/topology/contact_plan"],
            supports_topology_snapshot=endpoints["/topology/current"],
            supports_configuration_apply=endpoints["/configuration/apply"],
            supports_persistent_configuration=(endpoints["/configuration/apply"] and endpoints["/configuration/dispatch"]),
            supports_persistent_configuration_execution=(endpoints["/configuration/apply"] and endpoints["/configuration/dispatch"]),
            supports_configuration_dispatch=endpoints["/configuration/dispatch"],
            supports_physical_decision_delay=endpoints["/advance_world"],
            supports_advance_world=endpoints["/advance_world"],
            supports_scope_aware_planner_state=False,
            supports_configuration_validation=endpoints["/configuration/validate"],
            # Endpoint discovery is not proof that the server returns a
            # verifiable physical-time receipt.  The delay model verifies
            # before/after world time at runtime instead.
            supports_verified_delay_receipt=False,
            supports_mid_transfer_contact_enforcement=False,
            supported_budget_dimensions={
                value
                for value, enabled in (
                    ("scope", False),
                    ("budget", False),
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
                "formal_capabilities_unavailable": True,
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
        instrumentation = state.get("instrumentation", {}) or {}
        payload_is_cheap = state.get("payloadKind") == "cheap_monitor"
        true_cheap = bool(
            true_cheap
            and payload_is_cheap
            and instrumentation.get("candidateEvaluations") == 0
            and instrumentation.get("fullStateBuilderInvoked") is False
            and state.get("containsFutureStochasticState") is False
        )
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
        prediction_uncertainty = {
            str(k): float(v)
            for k, v in (state.get("predictionUncertainty", {}) or {}).items()
            if _is_number(v)
        }
        observed_config_id = state.get("configId", state.get("config_id"))
        observed_config_version = state.get("configVersion", state.get("config_version"))
        expected_config_id = self._configuration.config_id if self._configuration else None
        expected_config_version = self._configuration.version if self._configuration else None
        expected_config_version_int = _optional_int(expected_config_version)
        service_rate_observed = _optional_float(
            state.get("serviceRateObserved", state.get("service_rate_observed"))
        )
        service_bound_certified = bool(
            state.get("serviceBoundCertified", state.get("service_bound_certified", False))
        )
        service_rate_lower_bound = (
            _optional_float(state.get("serviceRateLowerBound", state.get("service_rate_lower_bound")))
            if service_bound_certified
            else None
        )
        uncertainty_marker = _optional_bool(
            state.get(
                "uncertaintyEvidenceAvailable",
                state.get("uncertainty_evidence_available", instrumentation.get("predictionUncertaintyAvailable")),
            )
        )
        uncertainty_evidence_available = bool(uncertainty_marker) and bool(prediction_uncertainty)
        uncertainty_source = state.get(
            "uncertaintySource",
            state.get("uncertainty_source", instrumentation.get("predictionUncertaintySource")),
        )
        observed_config_version_int = _optional_int(observed_config_version)
        contact_evidence_required = state.get("contactEvidenceRequired", state.get("contact_evidence_required"))
        contact_applicability_known = _optional_bool(
            state.get("contactApplicabilityKnown", state.get("contact_applicability_known"))
        )
        service_evidence_applicable = _optional_bool(
            state.get("serviceEvidenceApplicable", state.get("service_evidence_applicable"))
        )
        deadline_evidence_applicable = _optional_bool(
            state.get("deadlineEvidenceApplicable", state.get("deadline_evidence_applicable"))
        )
        deadline_evidence_available = _optional_bool(
            state.get("deadlineEvidenceAvailable", state.get("deadline_evidence_available"))
        )
        uncertainty_evidence_applicable = _optional_bool(
            state.get("uncertaintyEvidenceApplicable", state.get("uncertainty_evidence_applicable"))
        )
        service_evidence_status = state.get("serviceEvidenceStatus", state.get("service_evidence_status"))
        contact_evidence_status = state.get("contactEvidenceStatus", state.get("contact_evidence_status"))
        deadline_evidence_status = state.get("deadlineEvidenceStatus", state.get("deadline_evidence_status"))
        uncertainty_evidence_status = state.get("uncertaintyEvidenceStatus", state.get("uncertainty_evidence_status"))
        configuration_truth_available = observed_config_id is not None and observed_config_version_int is not None
        configuration_state_mismatch = bool(
            expected_config_id is not None
            and (
                not configuration_truth_available
                or str(observed_config_id) != str(expected_config_id)
                or expected_config_version_int is None
                or observed_config_version_int != expected_config_version_int
            )
        )
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
            current_config_id=observed_config_id,
            current_config_version=observed_config_version_int,
            configuration_age_sec=_optional_float(state.get("configurationAgeSec", state.get("configuration_age_sec"))),
            local_queue_summary={str(k): float(v) for k, v in queue.items() if _is_number(v)},
            source_queue_summary={str(k): float(v) for k, v in queue.items() if _is_number(v)},
            remaining_workload_summary={str(k): float(v) for k, v in (state.get("remainingWorkload", queue) or {}).items() if _is_number(v)},
            deadline_slack={str(k): float(v) for k, v in deadline.items() if _is_number(v)},
            local_load_summary={str(k): float(v) for k, v in (state.get("loadSummary", {}) or {}).items() if _is_number(v)},
            service_rate_observed=service_rate_observed,
            service_rate_lower_bound=service_rate_lower_bound,
            service_bound_certified=service_bound_certified,
            service_horizon_sec=_optional_float(state.get("serviceHorizonSec", state.get("service_horizon_sec"))),
            service_rate_source=state.get("serviceRateSource", state.get("service_rate_source")),
            service_bound_semantics=state.get("serviceBoundSemantics", state.get("service_bound_semantics")),
            service_evidence_status=service_evidence_status,
            service_horizon_source=state.get("serviceHorizonSource", state.get("service_horizon_source")),
            remaining_contact_lifetime={str(k): float(v) for k, v in (state.get("remainingContactLifetime", {}) or {}).items() if _is_number(v)},
            next_contact_summary=dict(state.get("nextContact", {}) or {}),
            contact_slack={str(k): float(v) for k, v in contact.items() if _is_number(v)},
            small_neighborhood_state=dict(state.get("smallNeighborhood", {}) or {}),
            cached_state=dict(state.get("cachedState", {}) or {}),
            prediction_uncertainty=prediction_uncertainty,
            uncertainty_evidence_available=uncertainty_evidence_available,
            uncertainty_source=uncertainty_source,
            contact_evidence_status=contact_evidence_status,
            deadline_evidence_status=deadline_evidence_status,
            uncertainty_evidence_status=uncertainty_evidence_status,
            compute_ready_workload_mi=_optional_float(state.get("computeReadyWorkloadMi", state.get("compute_ready_workload_mi"))),
            executing_workload_mi=_optional_float(state.get("executingWorkloadMi", state.get("executing_workload_mi"))),
            waiting_dispatch_workload_mi=_optional_float(state.get("waitingDispatchWorkloadMi", state.get("waiting_dispatch_workload_mi"))),
            network_remaining_bits=_optional_float(state.get("networkRemainingBits", state.get("network_remaining_bits"))),
            phase_state_uncertain=bool(state.get("phaseStateUncertain", state.get("phase_state_uncertain", False))),
            degradation_indicators={str(k): float(v) for k, v in (state.get("degradationIndicators", {}) or {}).items() if _is_number(v)},
            acquisition=acquisition,
            metadata={
                "backend_source": "satedgesim",
                "topology_source": self.capabilities.topology_source,
                "monitor_source": source,
                "future_stochastic_truth_used": False,
                "compatibility_preflight": source == "compatibility_preflight",
                "payload_kind": state.get("payloadKind", "unknown"),
                "cheap_monitor_instrumentation": dict(instrumentation),
                "authoritative_physical": bool(getattr(self.capabilities, "authoritative_physical", False)),
                "configuration_truth_available": configuration_truth_available,
                "configuration_state_mismatch": configuration_state_mismatch,
                "expected_config_id": expected_config_id,
                "expected_config_version": expected_config_version,
                "observed_config_id": observed_config_id,
                "observed_config_version": observed_config_version_int,
                "service_rate_observed_available": service_rate_observed is not None,
                "service_rate_lower_bound_available": service_rate_lower_bound is not None,
                "service_bound_certified": service_bound_certified,
                "service_horizon_available": _optional_float(state.get("serviceHorizonSec", state.get("service_horizon_sec"))) is not None,
                "service_evidence_applicable": service_evidence_applicable,
                "service_evidence_status": service_evidence_status,
                "service_horizon_source": state.get("serviceHorizonSource", state.get("service_horizon_source")),
                "uncertainty_evidence_available": uncertainty_evidence_available,
                "uncertainty_source": uncertainty_source,
                "uncertainty_evidence_applicable": uncertainty_evidence_applicable,
                "uncertainty_evidence_status": uncertainty_evidence_status,
                "contact_applicability_known": contact_applicability_known,
                "contact_evidence_status": contact_evidence_status,
                "deadline_evidence_applicable": deadline_evidence_applicable,
                "deadline_evidence_available": deadline_evidence_available,
                "deadline_evidence_status": deadline_evidence_status,
                **({"contact_evidence_required": contact_evidence_required} if contact_evidence_required is not None else {}),
                "affected_entity_hints": {
                    key: state[key]
                    for key in (
                        "affectedTaskIds",
                        "affectedSourceIds",
                        "affectedNodeIds",
                        "affectedLinkIds",
                        "affectedRouteIds",
                        "affectedResourceKeys",
                    )
                    if key in state
                },
            },
        )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _optional_int(value: Any) -> int | None:
    if not _is_number(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _has_time(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("simulationTimeSec", "simulation_time", "simulationTime", "time"))


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        raw = value.to_dict()
        return dict(raw) if isinstance(raw, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}
