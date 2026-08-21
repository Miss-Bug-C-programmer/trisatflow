"""First-class execution configurations that persist across physical slots."""

from __future__ import annotations

from copy import deepcopy
import json
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
    # Reusable rules are selector-based execution bindings.  A rule may match
    # source/application/traffic/flow/node/route/resource dimensions and is
    # reusable for tasks that have not been enumerated at configuration time.
    reusable_rules: dict[str, Any] = field(default_factory=dict)
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
        for name in ("assignments", "reusable_rules", "resource_allocations", "routes"):
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

    def change_counts(self, other: "PersistentConfiguration") -> dict[str, Any]:
        """Return structural changes between two applied configurations.

        This is an execution-side accounting primitive.  It deliberately does
        not use the requested control scope as a proxy: unchanged entries,
        additions and removals are counted from the two concrete
        configurations.
        """

        if not isinstance(other, PersistentConfiguration):
            raise TypeError("change_counts expects another PersistentConfiguration")

        def mapping_changes(name: str) -> tuple[int, int]:
            before = dict(getattr(self, name) or {})
            after = dict(getattr(other, name) or {})
            keys = set(before) | set(after)
            changed = sum(1 for key in keys if before.get(key) != after.get(key))
            return changed, len(keys)

        assignments, assignment_universe = mapping_changes("assignments")
        rules, rule_universe = mapping_changes("reusable_rules")
        resources, resource_universe = mapping_changes("resource_allocations")
        routes, route_universe = mapping_changes("routes")
        diff = self.diff(other)
        encoded = json.dumps(diff, default=str, sort_keys=True, separators=(",", ":"))
        return {
            "num_changed_assignments": assignments,
            "num_changed_reusable_rules": rules,
            "num_changed_resources": resources,
            "num_changed_routes": routes,
            "reconfiguration_bytes": len(encoded.encode("utf-8")) if diff else 0,
            "migration_volume": self.reconfiguration_volume(other),
            "changed_field_count": len(diff),
            "assignment_universe": assignment_universe,
            "rule_universe": rule_universe,
            "resource_universe": resource_universe,
            "route_universe": route_universe,
            "diff": diff,
        }

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
            for name in ("assignments", "reusable_rules", "resource_allocations", "routes")
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
        context = dict(task) if isinstance(task, Mapping) else {"task_id": str(task)}
        task_id = str(context.get("task_id", context.get("taskId", "")))
        if task_id in self.assignments:
            return deepcopy(self.assignments[task_id])
        source_id = context.get("source_id", context.get("sourceId"))
        if source_id is not None and str(source_id) in self.assignments:
            return deepcopy(self.assignments[str(source_id)])
        if source_id is not None and f"source:{source_id}" in self.assignments:
            return deepcopy(self.assignments[f"source:{source_id}"])
        # Explicit task overrides remain the strongest binding.
        best_rule: Any = None
        best_score = -1
        for rule_id, raw_rule in (self.reusable_rules or {}).items():
            if not isinstance(raw_rule, Mapping):
                continue
            selector = raw_rule.get("selector", raw_rule.get("match", {})) or {}
            if not _selector_matches(selector, context):
                continue
            score = _selector_specificity(selector)
            if str(rule_id).lower() == "default":
                score = max(0, score)
            if score > best_score:
                best_score = score
                best_rule = raw_rule
        if best_rule is not None:
            return deepcopy(best_rule.get("assignment", best_rule.get("action", best_rule)))
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


def _selector_matches(selector: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Match only decision-time task metadata, never future workload truth."""

    if not selector:
        return True
    aliases = {
        "task": ("task_id", "taskId"),
        "source": ("source_id", "sourceId"),
        "application": ("application_id", "applicationId"),
        "traffic": ("traffic_phase", "trafficPhase"),
        "flow": ("flow_id", "flowId"),
        "node": ("node_id", "nodeId"),
        "route": ("route_id", "routeId"),
        "resource": ("resource_key", "resourceKey"),
    }
    for key, expected in selector.items():
        field_names = aliases.get(str(key), (str(key),))
        actual = next((context[name] for name in field_names if name in context), None)
        values = expected if isinstance(expected, (list, tuple, set, frozenset)) else (expected,)
        if actual is None or str(actual) not in {str(value) for value in values}:
            return False
    return True


def _selector_specificity(selector: Mapping[str, Any]) -> int:
    return sum(1 for value in selector.values() if value not in (None, "", [], (), set()))
