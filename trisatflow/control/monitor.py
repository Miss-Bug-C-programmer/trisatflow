"""Monitor facade enforcing the cheap/heavy acquisition boundary."""

from __future__ import annotations

from typing import Any

from trisatflow.control.types import MonitorState, PlannerState


class CheapConfigurationMonitor:
    def __init__(self, backend: Any, *, true_cheap_required: bool = False, fallback_mode: str = "compatibility_preflight") -> None:
        self.backend = backend
        self.true_cheap_required = bool(true_cheap_required)
        self.fallback_mode = fallback_mode
        self.monitor_calls = 0
        self.planner_state_calls = 0

    def acquire(self, context: Any | None = None) -> MonitorState:
        self.monitor_calls += 1
        try:
            state = self.backend.get_monitor_state(context)
        except TypeError:
            state = self.backend.get_monitor_state()
        if not isinstance(state, MonitorState):
            raise TypeError(f"Expected MonitorState, got {type(state)!r}")
        if self.true_cheap_required and not state.acquisition.is_true_cheap_monitor and self.fallback_mode == "error":
            raise RuntimeError("Backend did not provide a true cheap monitor")
        return state

    def acquire_planner_state(self, context: Any | None = None, scope: Any | None = None, budget: Any | None = None) -> PlannerState:
        self.planner_state_calls += 1
        try:
            state = self.backend.get_planner_state(context, scope, budget)
        except TypeError:
            state = self.backend.get_planner_state()
        if not isinstance(state, PlannerState):
            raise TypeError(f"Expected PlannerState, got {type(state)!r}")
        return state


__all__ = ["CheapConfigurationMonitor", "MonitorState", "PlannerState"]
