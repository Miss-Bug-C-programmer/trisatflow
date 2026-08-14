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
    ) -> None:
        self.mode = str(mode).lower()
        self.require_physical_enforcement = bool(require_physical_enforcement)
        self.modeled_components = tuple(modeled_components)

    def estimate(self, cost: Any) -> DecisionDelayBreakdown:
        if self.mode in {"none", "instantaneous"}:
            return DecisionDelayBreakdown(mode=self.mode, modeled_components=self.modeled_components).recompute()
        result = DecisionDelayBreakdown(mode=self.mode, modeled_components=self.modeled_components)
        if "observation" in self.modeled_components:
            result.observation_delay_sec = float(getattr(cost, "observation_latency_sec", 0.0))
        if "sync" in self.modeled_components or "synchronization" in self.modeled_components:
            result.synchronization_delay_sec = float(getattr(cost, "sync_latency_sec", 0.0))
        if "solver" in self.modeled_components:
            solver = float(getattr(cost, "solver_simulated_latency_sec", 0.0))
            result.solver_delay_sec = solver if solver > 0 else float(getattr(cost, "solver_wallclock_sec", 0.0))
        if "signal" in self.modeled_components:
            result.signal_delay_sec = float(getattr(cost, "signal_latency_sec", 0.0))
        return result.recompute()

    def enforce(self, backend: Any, delay: DecisionDelayBreakdown) -> DecisionDelayBreakdown:
        supported = bool(getattr(getattr(backend, "capabilities", None), "supports_physical_decision_delay", False))
        if delay.total_delay_sec <= 0.0:
            delay.physical_delay_enforced = False
            return delay
        if supported and hasattr(backend, "advance_world"):
            backend.advance_world(delay.total_delay_sec)
            delay.physical_delay_enforced = True
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

    def revalidate(self, backend: Any, configuration: Any, *, planned_at: float, applied_at: float) -> RevalidationResult:
        if hasattr(backend, "validate_configuration"):
            raw = backend.validate_configuration(configuration)
            if isinstance(raw, RevalidationResult):
                return raw
            if isinstance(raw, Mapping):
                return RevalidationResult(
                    accepted=bool(raw.get("accepted", raw.get("valid", False))),
                    reason_codes=[str(v) for v in raw.get("reason_codes", [])],
                    metadata=dict(raw.get("metadata", {})),
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
