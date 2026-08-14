"""Small registry for explicitly selected planner backends."""

from __future__ import annotations

from typing import Any

from trisatflow.planners.base import PlannerBackend


class PlannerRegistry:
    def __init__(self, backends: list[PlannerBackend] | None = None) -> None:
        self._backends: dict[str, PlannerBackend] = {}
        for backend in backends or []:
            self.register(backend)

    def register(self, backend: PlannerBackend) -> None:
        name = str(getattr(backend, "name", "")).strip()
        if not name:
            raise ValueError("Planner backend must define a non-empty name")
        self._backends[name] = backend

    def get(self, name: str) -> PlannerBackend:
        try:
            return self._backends[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown planner backend {name!r}; available={sorted(self._backends)}") from exc

    def values(self) -> list[PlannerBackend]:
        return list(self._backends.values())

    def metadata(self) -> dict[str, Any]:
        return {
            name: {
                "family": getattr(backend, "family", "unknown"),
                "fidelity": getattr(getattr(backend, "fidelity", None), "value", getattr(backend, "fidelity", "unknown")),
                "capabilities": _capabilities_payload(backend),
            }
            for name, backend in self._backends.items()
        }


def _capabilities_payload(backend: PlannerBackend) -> dict[str, Any]:
    if not hasattr(backend, "capabilities"):
        return {}
    capabilities = backend.capabilities()
    if hasattr(capabilities, "to_dict"):
        return dict(capabilities.to_dict())
    return dict(getattr(capabilities, "metadata", {}) or {})
