"""Generic selective reconfiguration subsets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _normalise(values: Iterable[Any] | None) -> set[str]:
    return {str(value) for value in (values or ())}


@dataclass
class ReconfigurationScope:
    """A subset of entities eligible for reconfiguration.

    ``KEEP`` is represented by an empty scope.  ``LOCAL``, ``REGIONAL`` and
    ``GLOBAL`` are intentionally not actions; they may only be derived later as
    reporting buckets.
    """

    task_ids: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    node_ids: set[str] = field(default_factory=set)
    link_ids: set[str] = field(default_factory=set)
    route_ids: set[str] = field(default_factory=set)
    resource_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for name in (
            "task_ids",
            "source_ids",
            "node_ids",
            "link_ids",
            "route_ids",
            "resource_keys",
        ):
            setattr(self, name, _normalise(getattr(self, name)))

    @property
    def is_empty(self) -> bool:
        return self.cardinality == 0

    @property
    def cardinality(self) -> int:
        return sum(len(getattr(self, name)) for name in self._fields())

    def normalized_volume(self, universe: Mapping[str, int] | None = None) -> float:
        if self.is_empty:
            return 0.0
        if universe:
            numerator = 0.0
            denominator = 0.0
            for name in self._fields():
                numerator += len(getattr(self, name))
                denominator += max(0, int(universe.get(name, 0)))
            if denominator > 0:
                return min(1.0, numerator / denominator)
        return 1.0

    def contains(self, entity: Any, entity_type: str | None = None) -> bool:
        if entity_type:
            name = self._canonical_field(entity_type)
            return str(entity) in getattr(self, name)
        if isinstance(entity, Mapping):
            for key, field_name in {
                "task_id": "task_ids",
                "source_id": "source_ids",
                "node_id": "node_ids",
                "link_id": "link_ids",
                "route_id": "route_ids",
                "resource_key": "resource_keys",
            }.items():
                if key in entity and str(entity[key]) in getattr(self, field_name):
                    return True
            return False
        value = str(entity)
        return any(value in getattr(self, name) for name in self._fields())

    def union(self, other: "ReconfigurationScope") -> "ReconfigurationScope":
        return ReconfigurationScope(
            **{name: getattr(self, name) | getattr(other, name) for name in self._fields()}
        )

    def intersection(self, other: "ReconfigurationScope") -> "ReconfigurationScope":
        return ReconfigurationScope(
            **{name: getattr(self, name) & getattr(other, name) for name in self._fields()}
        )

    def truncate(self, max_entities: int) -> "ReconfigurationScope":
        remaining = max(0, int(max_entities))
        payload: dict[str, set[str]] = {}
        for name in self._fields():
            values = sorted(getattr(self, name))[:remaining]
            payload[name] = set(values)
            remaining = max(0, remaining - len(values))
        return ReconfigurationScope(**payload)

    def affected_entities(self) -> dict[str, set[str]]:
        return {name: set(getattr(self, name)) for name in self._fields()}

    def derived_bucket(self, universe: Mapping[str, int] | None = None) -> str:
        """Reporting-only size bucket; never used as a controller action."""

        volume = self.normalized_volume(universe)
        if self.is_empty:
            return "keep"
        if volume < 0.25:
            return "small"
        if volume < 0.75:
            return "medium"
        return "global"

    def to_dict(self) -> dict[str, list[str]]:
        return {name: sorted(getattr(self, name)) for name in self._fields()}

    @staticmethod
    def _fields() -> tuple[str, ...]:
        return (
            "task_ids",
            "source_ids",
            "node_ids",
            "link_ids",
            "route_ids",
            "resource_keys",
        )

    @staticmethod
    def _canonical_field(value: str) -> str:
        text = str(value).strip().lower()
        if text.endswith("s"):
            text = text[:-1]
        candidate = f"{text}_ids" if text != "resource_key" else "resource_keys"
        if candidate not in ReconfigurationScope._fields():
            raise ValueError(f"Unknown scope entity type: {value!r}")
        return candidate


class ScopeGenerator:
    """Generate candidate subsets from observed impact and dependencies."""

    def __init__(
        self,
        *,
        max_candidate_scopes: int = 4,
        max_scope_entities: int | None = None,
        include_global_candidate: bool = True,
    ) -> None:
        self.max_candidate_scopes = max(1, int(max_candidate_scopes))
        self.max_scope_entities = max_scope_entities
        self.include_global_candidate = bool(include_global_candidate)

    def generate(
        self,
        current_config: Any,
        monitor_state: Any,
        planner_state: Any | None,
        viability_report: Any,
    ) -> list[ReconfigurationScope]:
        base = getattr(viability_report, "affected_entities", None)
        if isinstance(base, ReconfigurationScope):
            impacted = base
        elif isinstance(base, Mapping):
            impacted = ReconfigurationScope(**{k: v for k, v in base.items() if k in ReconfigurationScope._fields()})
        else:
            impacted = self._from_observation(monitor_state, current_config)

        candidates: list[ReconfigurationScope] = []
        if not impacted.is_empty:
            candidates.append(self._limit(impacted))

            graph = {}
            metadata = getattr(viability_report, "metadata", {}) or {}
            if isinstance(metadata, Mapping):
                graph = metadata.get("dependency_graph", {}) or {}
            expanded = self._expand_dependencies(impacted, graph)
            if expanded != impacted:
                candidates.append(self._limit(expanded))

        if self.include_global_candidate:
            global_scope = self._global_scope(current_config, planner_state)
            if not global_scope.is_empty:
                candidates.append(self._limit(global_scope))

        unique: list[ReconfigurationScope] = []
        seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
        for scope in candidates:
            key = tuple((name, tuple(sorted(getattr(scope, name)))) for name in scope._fields())
            if key not in seen and not scope.is_empty:
                seen.add(key)
                unique.append(scope)
        return unique[: self.max_candidate_scopes]

    def _limit(self, scope: ReconfigurationScope) -> ReconfigurationScope:
        if self.max_scope_entities is None:
            return scope
        return scope.truncate(self.max_scope_entities)

    @staticmethod
    def _from_observation(monitor_state: Any, current_config: Any) -> ReconfigurationScope:
        tasks = set()
        sources = set()
        for key in ("remaining_workload_summary", "deadline_slack", "source_queue_summary"):
            value = getattr(monitor_state, key, {}) or {}
            if isinstance(value, Mapping):
                sources.update(str(k) for k in value)
        tasks.update(str(v) for v in (getattr(current_config, "covered_task_ids", set()) or set()))
        return ReconfigurationScope(task_ids=tasks, source_ids=sources)

    @staticmethod
    def _expand_dependencies(scope: ReconfigurationScope, graph: Mapping[str, Any]) -> ReconfigurationScope:
        expanded = scope
        for entity, neighbours in graph.items():
            if not isinstance(neighbours, Iterable) or isinstance(neighbours, (str, bytes)):
                continue
            if scope.contains(entity):
                expanded = expanded.union(ReconfigurationScope(node_ids={str(v) for v in neighbours}))
        return expanded

    @staticmethod
    def _global_scope(current_config: Any, planner_state: Any | None) -> ReconfigurationScope:
        payload: dict[str, set[str]] = {}
        for name in ReconfigurationScope._fields():
            values = set(getattr(current_config, f"covered_{name}", set()) or set())
            if planner_state is not None and name == "node_ids":
                values.update(str(k) for k in (getattr(planner_state, "detailed_resources", {}) or {}))
            payload[name] = values
        return ReconfigurationScope(**payload)
