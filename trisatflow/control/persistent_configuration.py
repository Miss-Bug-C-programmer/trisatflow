"""First-class execution configurations that persist across physical slots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from trisatflow.control.scope import ReconfigurationScope


@dataclass
class PersistentConfiguration:
    config_id: str
    version: int = 0
    created_at_sim_time: float = 0.0
    applied_at_sim_time: float = 0.0
    last_validated_at_sim_time: float = 0.0
    source_state_id: str | None = None
    source_decision_id: str | None = None
    assignments: dict[str, Any] = field(default_factory=dict)
    resource_allocations: dict[str, Any] = field(default_factory=dict)
    routes: dict[str, Any] = field(default_factory=dict)
    covered_task_ids: set[str] = field(default_factory=set)
    covered_source_ids: set[str] = field(default_factory=set)
    covered_node_ids: set[str] = field(default_factory=set)
    covered_link_ids: set[str] = field(default_factory=set)
    covered_resource_keys: set[str] = field(default_factory=set)
    planner_name: str = "unknown"
    planner_fidelity: str = "light"
    planning_budget: dict[str, Any] = field(default_factory=dict)
    scope_from_previous: ReconfigurationScope = field(default_factory=ReconfigurationScope)
    expected_horizon: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "covered_task_ids",
            "covered_source_ids",
            "covered_node_ids",
            "covered_link_ids",
            "covered_resource_keys",
        ):
            setattr(self, name, {str(v) for v in (getattr(self, name) or set())})
        if not isinstance(self.scope_from_previous, ReconfigurationScope):
            self.scope_from_previous = ReconfigurationScope(**(self.scope_from_previous or {}))

    def clone(self, *, config_id: str | None = None, version: int | None = None) -> "PersistentConfiguration":
        cloned = deepcopy(self)
        if config_id is not None:
            cloned.config_id = str(config_id)
        if version is not None:
            cloned.version = int(version)
        return cloned

    def diff(self, other: "PersistentConfiguration") -> dict[str, Any]:
        if not isinstance(other, PersistentConfiguration):
            raise TypeError("diff expects another PersistentConfiguration")
        changed: dict[str, Any] = {}
        for name in ("assignments", "resource_allocations", "routes"):
            before = getattr(self, name)
            after = getattr(other, name)
            if before != after:
                changed[name] = {"before": deepcopy(before), "after": deepcopy(after)}
        for name in (
            "covered_task_ids",
            "covered_source_ids",
            "covered_node_ids",
            "covered_link_ids",
            "covered_resource_keys",
        ):
            before = getattr(self, name)
            after = getattr(other, name)
            if before != after:
                changed[name] = {
                    "added": sorted(after - before),
                    "removed": sorted(before - after),
                }
        return changed

    def apply_patch(self, patch: Mapping[str, Any] | None = None, **kwargs: Any) -> "PersistentConfiguration":
        """Return a patched copy, preserving immutable historical versions."""

        payload: dict[str, Any] = dict(patch or {})
        payload.update(kwargs)
        result = self.clone()
        for name, value in payload.items():
            if not hasattr(result, name):
                raise AttributeError(f"Unknown PersistentConfiguration field: {name}")
            if name == "scope_from_previous" and not isinstance(value, ReconfigurationScope):
                value = ReconfigurationScope(**dict(value or {}))
            if name in {
                "covered_task_ids",
                "covered_source_ids",
                "covered_node_ids",
                "covered_link_ids",
                "covered_resource_keys",
            }:
                value = set(str(v) for v in value)
            setattr(result, name, deepcopy(value))
        return result

    def affected_entities(self) -> ReconfigurationScope:
        return ReconfigurationScope(
            task_ids=self.covered_task_ids,
            source_ids=self.covered_source_ids,
            node_ids=self.covered_node_ids,
            link_ids=self.covered_link_ids,
            resource_keys=self.covered_resource_keys,
        )

    def reconfiguration_volume(self, other: "PersistentConfiguration", universe: Mapping[str, int] | None = None) -> float:
        changed = self.diff(other)
        if not changed:
            return 0.0
        left = self.affected_entities()
        right = other.affected_entities()
        entity_scope = left.union(right)
        if universe:
            return entity_scope.normalized_volume(universe)
        # A conservative structural proxy when the backend does not expose its
        # universe.  It counts changed assignment/resource bindings exactly once.
        entity_count = max(1, entity_scope.cardinality)
        changed_count = sum(
            1
            for name in ("assignments", "resource_allocations", "routes")
            if name in changed
        )
        return min(1.0, max(changed_count, entity_count) / max(entity_count, 1))

    def is_applicable_to(self, target: Any, *, target_type: str | None = None) -> bool:
        if target is None:
            return False
        if isinstance(target, Mapping):
            for key, field_name in {
                "task_id": "covered_task_ids",
                "source_id": "covered_source_ids",
                "node_id": "covered_node_ids",
                "link_id": "covered_link_ids",
                "resource_key": "covered_resource_keys",
            }.items():
                if key in target:
                    values = getattr(self, field_name)
                    return not values or str(target[key]) in values
            return True
        if target_type:
            field_name = {
                "task": "covered_task_ids",
                "source": "covered_source_ids",
                "node": "covered_node_ids",
                "link": "covered_link_ids",
                "resource": "covered_resource_keys",
            }.get(str(target_type).lower())
            if field_name is None:
                raise ValueError(f"Unknown target type: {target_type!r}")
            values = getattr(self, field_name)
            return not values or str(target) in values
        return any(
            not values or str(target) in values
            for values in (
                self.covered_task_ids,
                self.covered_source_ids,
                self.covered_node_ids,
                self.covered_link_ids,
                self.covered_resource_keys,
            )
        )

    def materialize_execution_rule(self, task: Mapping[str, Any] | Any) -> Any:
        task_id = str(task.get("task_id", task.get("taskId"))) if isinstance(task, Mapping) else str(task)
        if task_id in self.assignments:
            return deepcopy(self.assignments[task_id])
        return deepcopy(self.assignments.get("default"))

    def to_dict(self) -> dict[str, Any]:
        payload = deepcopy(self.__dict__)
        payload["scope_from_previous"] = self.scope_from_previous.to_dict()
        for name in (
            "covered_task_ids",
            "covered_source_ids",
            "covered_node_ids",
            "covered_link_ids",
            "covered_resource_keys",
        ):
            payload[name] = sorted(payload[name])
        return payload
