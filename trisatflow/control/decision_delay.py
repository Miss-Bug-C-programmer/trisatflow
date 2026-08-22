"""Explicit contract for modeled planning delay and post-delay validity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class DecisionDelayBreakdown:
    observation_delay_sec: float = 0.0
    synchronization_delay_sec: float = 0.0
    solver_delay_sec: float = 0.0
    signal_delay_sec: float = 0.0
    total_delay_sec: float = 0.0
    mode: str = "none"
    physical_delay_enforced: bool = False
    modeled_components: tuple[str, ...] = ()
    wallclock_solver_sec: float = 0.0
    requested_delta_sec: float = 0.0
    actual_delta_sec: float = 0.0
    world_time_before: float | None = None
    world_time_after: float | None = None
    physical_receipt_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def recompute(self) -> "DecisionDelayBreakdown":
        self.total_delay_sec = sum(
            float(value)
            for value in (
                self.observation_delay_sec,
                self.synchronization_delay_sec,
                self.solver_delay_sec,
                self.signal_delay_sec,
            )
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modeled_components"] = list(self.modeled_components)
        return payload


class DecisionDelayModel:
    def __init__(
        self,
        *,
        mode: str = "none",
        require_physical_enforcement: bool = False,
        modeled_components: tuple[str, ...] = ("solver",),
        use_wallclock_as_simulated: bool = False,
        receipt_tolerance_sec: float = 1.0e-9,
    ) -> None:
        self.mode = str(mode).lower()
        self.require_physical_enforcement = bool(require_physical_enforcement)
        self.modeled_components = tuple(modeled_components)
        self.use_wallclock_as_simulated = bool(use_wallclock_as_simulated)
        self.receipt_tolerance_sec = max(0.0, float(receipt_tolerance_sec))

    def estimate(self, cost: Any) -> DecisionDelayBreakdown:
        if self.mode in {"none", "instantaneous"}:
            return DecisionDelayBreakdown(mode=self.mode, modeled_components=self.modeled_components).recompute()
        result = DecisionDelayBreakdown(mode=self.mode, modeled_components=self.modeled_components)
        result.wallclock_solver_sec = float(getattr(cost, "solver_wallclock_sec", 0.0))
        if "observation" in self.modeled_components:
            result.observation_delay_sec = float(getattr(cost, "observation_latency_sec", 0.0))
        if "sync" in self.modeled_components or "synchronization" in self.modeled_components:
            result.synchronization_delay_sec = float(getattr(cost, "sync_latency_sec", 0.0))
        if "solver" in self.modeled_components:
            result.solver_delay_sec = float(getattr(cost, "solver_simulated_latency_sec", 0.0))
            if result.solver_delay_sec <= 0.0 and self.use_wallclock_as_simulated:
                result.solver_delay_sec = result.wallclock_solver_sec
                result.metadata["simulated_delay_source"] = "explicit_wallclock_ablation"
            else:
                result.metadata["simulated_delay_source"] = "modeled_solver_latency"
        if "signal" in self.modeled_components:
            result.signal_delay_sec = float(getattr(cost, "signal_latency_sec", 0.0))
        return result.recompute()

    def enforce(self, backend: Any, delay: DecisionDelayBreakdown) -> DecisionDelayBreakdown:
        capabilities = getattr(backend, "capabilities", None)
        supported = bool(getattr(capabilities, "supports_physical_decision_delay", False)) and bool(
            getattr(capabilities, "supports_advance_world", True)
        )
        delay.requested_delta_sec = max(0.0, float(delay.total_delay_sec))
        if delay.total_delay_sec <= 0.0:
            delay.physical_delay_enforced = False
            delay.metadata["physical_receipt_status"] = "not_requested"
            return delay
        control_epoch = bool(
            getattr(capabilities, "supports_control_monitor_epoch", False)
            and hasattr(backend, "advance_control_epoch")
        )
        advance_method = "advance_control_epoch" if control_epoch else "advance_world"
        if supported and hasattr(backend, advance_method):
            before = None
            if hasattr(backend, "current_time"):
                try:
                    before = float(backend.current_time())
                except Exception:  # noqa: BLE001
                    before = None
            receipt = getattr(backend, advance_method)(delay.total_delay_sec)
            delay.metadata["physical_progression_path"] = advance_method
            delay.metadata["control_monitor_epoch"] = control_epoch
            if isinstance(receipt, Mapping):
                # Keep the backend's native progression receipt attached to
                # the decision.  A boolean ``physical_delay_enforced`` is not
                # sufficient evidence for publication claims: the outcome
                # path must be able to inspect the before/after simulation
                # state reported by /advance_world.
                delay.metadata["physical_advance_receipt"] = dict(receipt)
                delay.metadata["control_epoch_paused_for_activation"] = bool(
                    receipt.get("pausedForConfigurationActivation", False)
                )
                delay.metadata["physical_state_changed"] = bool(
                    receipt.get("physicalStateChanged", receipt.get("physical_state_changed", False))
                )
            after = None
            if hasattr(backend, "current_time"):
                try:
                    after = float(backend.current_time())
                except Exception:  # noqa: BLE001
                    after = None
            delay.world_time_before = before
            delay.world_time_after = after
            if before is not None and after is not None:
                delay.actual_delta_sec = max(0.0, after - before)
                verified = delay.actual_delta_sec + self.receipt_tolerance_sec >= delay.requested_delta_sec
            else:
                receipt_delta = None
                if isinstance(receipt, Mapping):
                    for key in ("actualDeltaSec", "actual_delta_sec", "physicalDeltaSec"):
                        if key in receipt:
                            try:
                                receipt_delta = float(receipt[key])
                            except (TypeError, ValueError):
                                receipt_delta = None
                            break
                if receipt_delta is not None:
                    delay.actual_delta_sec = max(0.0, receipt_delta)
                    verified = receipt_delta + self.receipt_tolerance_sec >= delay.requested_delta_sec
                else:
                    verified = bool(isinstance(receipt, Mapping) and receipt.get("accepted", receipt.get("physicalDelayEnforced", False)))
            if isinstance(receipt, Mapping) and "accepted" in receipt and not bool(receipt.get("accepted")):
                verified = False
                delay.metadata["physical_receipt_rejected"] = receipt.get("reason", "backend_rejected")
            delay.physical_receipt_verified = verified
            delay.physical_delay_enforced = verified
            delay.metadata["physical_receipt_status"] = "verified" if verified else "unverified"
            if not verified and self.require_physical_enforcement:
                raise RuntimeError("Backend physical delay path returned no verifiable physical time advancement")
            return delay
        delay.physical_delay_enforced = False
        if self.require_physical_enforcement:
            raise RuntimeError(
                "Physical decision delay was required but the backend does not expose "
                "supports_physical_decision_delay + advance_world"
            )
        return delay


@dataclass
class RevalidationResult:
    accepted: bool
    reason_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PostDelayRevalidator:
    """Backend-mediated stale-plan check; no future oracle is inferred here."""

    def revalidate(
        self,
        backend: Any,
        configuration: Any,
        *,
        planned_at: float,
        applied_at: float,
        current_configuration: Any | None = None,
        scope: Any | None = None,
        observed_world_version: int | None = None,
        observed_control_epoch: int | None = None,
        planning_delay: Mapping[str, Any] | None = None,
        intervention_id: str | None = None,
    ) -> RevalidationResult:
        if hasattr(backend, "validate_configuration"):
            try:
                raw = backend.validate_configuration(
                    configuration,
                    current_configuration=current_configuration,
                    scope=scope,
                    observed_world_version=observed_world_version,
                    observed_control_epoch=observed_control_epoch,
                    planning_delay=planning_delay,
                    intervention_id=intervention_id,
                )
            except TypeError:
                # Compatibility adapters keep the old one-argument contract;
                # they do not participate in the authoritative patch token.
                raw = backend.validate_configuration(configuration)
            if isinstance(raw, RevalidationResult):
                return raw
            if isinstance(raw, Mapping):
                raw_reasons = raw.get("reason_codes", raw.get("reasonCodes", []))
                if isinstance(raw_reasons, str):
                    raw_reasons = [raw_reasons]
                metadata = dict(raw.get("metadata", raw.get("details", {})) or {})
                # The v2 server keeps these fields at response top level.  Do
                # not drop them when translating into the canonical Python
                # revalidation result; the apply request needs the exact
                # world version that was validated after physical delay.
                for key in (
                    "worldVersion", "world_version", "configurationVersion",
                    "configurationVersionBefore", "resultingConfigurationVersion",
                    "simulationTimeSec", "decisionStatus", "staleBaseRejected",
                ):
                    if key in raw:
                        metadata[key] = raw[key]
                return RevalidationResult(
                    accepted=bool(raw.get("accepted", raw.get("valid", False))),
                    reason_codes=[str(v) for v in raw_reasons],
                    metadata=metadata,
                )
            return RevalidationResult(accepted=bool(raw))

        if hasattr(backend, "get_monitor_state"):
            monitor = backend.get_monitor_state()
            if hasattr(configuration, "is_applicable_to"):
                target = getattr(monitor, "metadata", {}).get("current_task")
                if target is not None and not configuration.is_applicable_to(target):
                    return RevalidationResult(False, ["target_unavailable"])
        return RevalidationResult(
            accepted=True,
            metadata={"revalidation_fallback": True, "planned_at": planned_at, "applied_at": applied_at},
        )
