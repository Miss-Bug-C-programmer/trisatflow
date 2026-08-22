"""Generic selective reconfiguration subsets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping


_AGGREGATE_KEYS = {
    "total",
    "current",
    "arrivedtaskcount",
    "unfinishedtaskcount",
    "pendingdecision",
    "futuretaskcount",
    "queue",
    "load",
    "servicerate",
}

_ENTITY_TYPES = {
    "task_ids": "task",
    "source_ids": "source",
    "node_ids": "node",
    "link_ids": "link",
    "route_ids": "route",
    "resource_keys": "resource",
}


def _normalise(values: Iterable[Any] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        return {str(values)}
    return {str(value) for value in (values or ())}


def _clean_typed_id(value: Any, entity_type: str) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if _is_aggregate_id(lowered):
        return None
    prefixes = (f"{entity_type}:", f"{entity_type}_id:", f"{entity_type}id:")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            return text or None
    if entity_type == "source":
        return None
    return text


def _is_aggregate_id(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in _AGGREGATE_KEYS:
        return True
    prefixes = (
        "task:", "task_id:", "taskid:",
        "source:", "source_id:", "sourceid:",
        "node:", "node_id:", "nodeid:",
        "link:", "link_id:", "linkid:",
        "route:", "route_id:", "routeid:",
        "resource:", "resource_key:", "resourcekey:",
    )
    return any(text.startswith(prefix) and text[len(prefix):] in _AGGREGATE_KEYS for prefix in prefixes)


def _clean_explicit_id(value: Any, entity_type: str) -> str | None:
    """Normalize an explicit typed hint while accepting canonical bare ids."""

    text = str(value).strip()
    if not text or _is_aggregate_id(text):
        return None
    lowered = text.lower()
    prefixes = (f"{entity_type}:", f"{entity_type}_id:", f"{entity_type}id:")
    if entity_type == "resource":
        prefixes = (*prefixes, "resource_key:", "resourcekey:")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            return text if text and not _is_aggregate_id(text) else None
    return text


def _values_from_mapping(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        return value.keys()
    if isinstance(value, (str, bytes)):
        return (value,)
    return value or ()


def _scope_from_typed_mapping(mapping: Mapping[str, Any]) -> "ReconfigurationScope":
    payload: dict[str, set[str]] = {}
    for field_name, entity_type in _ENTITY_TYPES.items():
        raw = mapping.get(field_name)
        if raw is None:
            continue
        payload[field_name] = {
            parsed
            for value in _values_from_mapping(raw)
            if (parsed := _clean_explicit_id(value, entity_type)) is not None
        }
    return ReconfigurationScope(**payload)


@dataclass
class ViolationProvenance:
    """Evidence-backed explanation for why an entity entered a scope.

    This is deliberately a small provenance object, not a second scope model.
    Its entity sets are converted to :class:`ReconfigurationScope` at the
    canonical candidate-generation boundary.
    """

    violation_type: str
    task_ids: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    node_ids: set[str] = field(default_factory=set)
    link_ids: set[str] = field(default_factory=set)
    route_ids: set[str] = field(default_factory=set)
    resource_keys: set[str] = field(default_factory=set)
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    severity: float | None = None
    source: str = "monitor"

    def __post_init__(self) -> None:
        for name in ReconfigurationScope._fields():
            setattr(self, name, {
                parsed
                for value in _normalise(getattr(self, name))
                if (parsed := _clean_explicit_id(value, _ENTITY_TYPES[name])) is not None
            })
        self.violation_type = str(self.violation_type).upper()

    def to_scope(self) -> "ReconfigurationScope":
        return ReconfigurationScope(**{name: set(getattr(self, name)) for name in ReconfigurationScope._fields()})

    def identity(self) -> str:
        payload = {
            "violation_type": self.violation_type,
            **{name: sorted(getattr(self, name)) for name in ReconfigurationScope._fields()},
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "severity": self.severity,
            "source": self.source,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            **{name: sorted(getattr(self, name)) for name in ReconfigurationScope._fields()},
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "severity": self.severity,
            "source": self.source,
            "provenance_id": self.identity() if self.violation_type else "",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ViolationProvenance":
        aliases = {
            "affectedTaskIds": "task_ids", "affectedSourceIds": "source_ids",
            "affectedNodeIds": "node_ids", "affectedLinkIds": "link_ids",
            "affectedRouteIds": "route_ids", "affectedResourceKeys": "resource_keys",
            "type": "violation_type", "violationType": "violation_type",
        }
        payload: dict[str, Any] = {}
        for key, item in value.items():
            payload[aliases.get(key, key)] = item
        return cls(**{key: item for key, item in payload.items() if key in cls.__dataclass_fields__})


# Compatibility name used by external contract/tests.
ConstraintViolation = ViolationProvenance


def extract_constraint_provenance(viability_report: Any, monitor_state: Any | None = None) -> list[ViolationProvenance]:
    """Read canonical violation provenance without inventing favorable IDs."""

    metadata = getattr(viability_report, "metadata", {}) or {}
    raw_values: list[Any] = []
    direct = getattr(viability_report, "constraint_provenance", None)
    if direct:
        raw_values.extend(direct)
    for key in ("constraint_provenance", "violations", "violation_provenance"):
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            raw_values.extend(value)
    result: list[ViolationProvenance] = []
    seen: set[str] = set()
    seen_types: set[str] = set()
    for value in raw_values:
        item = value if isinstance(value, ViolationProvenance) else (
            ViolationProvenance.from_mapping(value) if isinstance(value, Mapping) else None
        )
        if item is not None and item.identity() not in seen:
            result.append(item)
            seen.add(item.identity())
            seen_types.add(item.violation_type)
    affected = getattr(viability_report, "affected_entities", ReconfigurationScope())
    if not isinstance(affected, ReconfigurationScope):
        affected = ReconfigurationScope(**{k: v for k, v in (affected or {}).items() if k in ReconfigurationScope._fields()})
    reasons = list(getattr(viability_report, "certificate_failure_reasons", []) or [])
    reasons.extend(str(value) for value in (getattr(viability_report, "reason_codes", []) or []))
    reason_map = (
        ("service", "SERVICE_DEFICIT"), ("deadline", "DEADLINE_DEFICIT"),
        ("contact", "CONTACT_DEFICIT"), ("queue", "QUEUE_OVERLOAD"),
        ("resource", "RESOURCE_CONTENTION"), ("uncertainty", "UNCERTAINTY"),
    )
    for token, violation_type in reason_map:
        matching = [reason for reason in reasons if token in reason.lower()]
        if not matching or violation_type in seen_types:
            continue
        item = ViolationProvenance(
            violation_type=violation_type,
            task_ids=set(affected.task_ids), source_ids=set(affected.source_ids),
            node_ids=set(affected.node_ids), link_ids=set(affected.link_ids),
            route_ids=set(affected.route_ids), resource_keys=set(affected.resource_keys),
            reason=";".join(sorted(set(matching))),
            evidence={"monitor_epoch": (getattr(monitor_state, "metadata", {}) or {}).get("monitor_epoch")},
            source="viability_certificate",
        )
        if item.identity() not in seen:
            result.append(item)
            seen.add(item.identity())
            seen_types.add(item.violation_type)
    if not result and not affected.is_empty and reasons:
        result.append(ViolationProvenance(
            violation_type="UNKNOWN", reason=";".join(sorted(set(reasons))),
            **affected.affected_entities(), source="viability_certificate",
        ))
    return result


def _typed_metadata_scope(metadata: Mapping[str, Any]) -> "ReconfigurationScope":
    aliases = {
        "affectedTaskIds": "task_ids",
        "affectedSourceIds": "source_ids",
        "affectedNodeIds": "node_ids",
        "affectedLinkIds": "link_ids",
        "affectedRouteIds": "route_ids",
        "affectedResourceKeys": "resource_keys",
    }
    nested = metadata.get("affected_entity_hints")
    source_metadata = nested if isinstance(nested, Mapping) else metadata
    canonical = {
        field_name: source_metadata[key]
        for key, field_name in aliases.items()
        if key in source_metadata
    }
    return _scope_from_typed_mapping(canonical)


def _remove_aggregate_ids(scope: "ReconfigurationScope") -> "ReconfigurationScope":
    payload = {}
    for field_name in ReconfigurationScope._fields():
        payload[field_name] = {
            value for value in getattr(scope, field_name)
            if not _is_aggregate_id(value)
        }
    return ReconfigurationScope(**payload)


def extract_typed_affected_entities(monitor_state: Any, current_config: Any | None = None) -> "ReconfigurationScope":
    """Extract typed entity ids without treating aggregate keys as entities."""

    metadata = getattr(monitor_state, "metadata", {}) or {}
    explicit = metadata.get("affected_entities")
    if isinstance(explicit, ReconfigurationScope):
        result = _remove_aggregate_ids(explicit)
    elif isinstance(explicit, Mapping):
        result = _remove_aggregate_ids(_scope_from_typed_mapping(explicit))
    else:
        result = ReconfigurationScope()

    result = _remove_aggregate_ids(result.union(_typed_metadata_scope(metadata)))
    sources: set[str] = set()
    for field_name in ("remaining_workload_summary", "source_queue_summary"):
        mapping = getattr(monitor_state, field_name, {}) or {}
        if isinstance(mapping, Mapping):
            for key in mapping:
                parsed = _clean_typed_id(key, "source")
                if parsed is not None:
                    sources.add(parsed)

    tasks: set[str] = set()
    deadlines = getattr(monitor_state, "deadline_slack", {}) or {}
    if isinstance(deadlines, Mapping):
        for key in deadlines:
            parsed = _clean_typed_id(key, "task")
            if parsed is not None:
                tasks.add(parsed)

    for field_name in ("contact_slack", "remaining_contact_lifetime"):
        mapping = getattr(monitor_state, field_name, {}) or {}
        if isinstance(mapping, Mapping):
            for key in mapping:
                parsed = _task_id_from_transfer_key(key)
                if parsed is not None:
                    tasks.add(parsed)

    result = _remove_aggregate_ids(result.union(ReconfigurationScope(task_ids=tasks, source_ids=sources)))
    if result.is_empty and current_config is not None:
        result = result.union(_remove_aggregate_ids(current_config.affected_entities()))
    return _remove_aggregate_ids(result)


def _task_id_from_transfer_key(value: Any) -> str | None:
    parts = str(value).strip().split(":", 2)
    if len(parts) != 3 or parts[0].lower() != "transfer":
        return None
    return _clean_explicit_id(parts[1], "task")


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "task_ids",
            "source_ids",
            "node_ids",
            "link_ids",
            "route_ids",
            "resource_keys",
        ):
            setattr(
                self,
                name,
                {
                    value
                    for value in _normalise(getattr(self, name))
                    if not _is_aggregate_id(value)
                },
            )

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
        metadata = dict(self.metadata)
        metadata.update(other.metadata)
        return ReconfigurationScope(
            **{name: getattr(self, name) | getattr(other, name) for name in self._fields()},
            metadata=metadata,
        )

    def intersection(self, other: "ReconfigurationScope") -> "ReconfigurationScope":
        return ReconfigurationScope(
            **{name: getattr(self, name) & getattr(other, name) for name in self._fields()},
            metadata=dict(self.metadata),
        )

    def truncate(self, max_entities: int) -> "ReconfigurationScope":
        remaining = max(0, int(max_entities))
        payload: dict[str, set[str]] = {}
        for name in self._fields():
            values = sorted(getattr(self, name))[:remaining]
            payload[name] = set(values)
            remaining = max(0, remaining - len(values))
        return ReconfigurationScope(**payload, metadata=dict(self.metadata))

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

    @property
    def scope_id(self) -> str:
        explicit = self.metadata.get("scope_id")
        if explicit:
            return str(explicit)
        payload = {name: sorted(getattr(self, name)) for name in self._fields()}
        return "scope-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {name: sorted(getattr(self, name)) for name in self._fields()}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

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
        provenance = extract_constraint_provenance(viability_report, monitor_state)
        base = getattr(viability_report, "affected_entities", None)
        if isinstance(base, ReconfigurationScope):
            impacted = base
        elif isinstance(base, Mapping):
            impacted = ReconfigurationScope(**{k: v for k, v in base.items() if k in ReconfigurationScope._fields()})
        else:
            impacted = self._from_observation(monitor_state, current_config)

        candidates: list[ReconfigurationScope] = []
        if not impacted.is_empty:
            direct = ReconfigurationScope()
            for item in provenance:
                direct = direct.union(item.to_scope())
            direct = direct if not direct.is_empty else impacted
            candidates.append(self._annotate(self._limit(direct), provenance, "minimal_direct_implication"))

            graph = {}
            metadata = getattr(viability_report, "metadata", {}) or {}
            if isinstance(metadata, Mapping):
                graph = metadata.get("dependency_graph", {}) or {}
            expanded = self._expand_dependencies(direct, graph)
            if expanded != direct:
                candidates.append(self._annotate(
                    self._limit(expanded), provenance, "dependency_expansion",
                    extra_provenance=[ViolationProvenance(
                        violation_type="DEPENDENCY_EXPANSION", reason="typed_dependency_graph",
                        **expanded.intersection(self._limit(expanded)).affected_entities(), source="scope_generator"
                    )],
                ))

        # A wider candidate is only meaningful after a directly implicated
        # entity exists.  Missing provenance must not be converted into an
        # implicit global refresh.
        if self.include_global_candidate and not impacted.is_empty:
            global_scope = self._global_scope(current_config, planner_state)
            if not global_scope.is_empty:
                candidates.append(self._annotate(self._limit(global_scope), provenance, "wider_repair_allowed"))

        unique: list[ReconfigurationScope] = []
        seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
        for scope in candidates:
            key = tuple((name, tuple(sorted(getattr(scope, name)))) for name in scope._fields())
            if key not in seen and not scope.is_empty:
                seen.add(key)
                unique.append(scope)
        return unique[: self.max_candidate_scopes]

    @staticmethod
    def _annotate(
        scope: ReconfigurationScope,
        provenance: list[ViolationProvenance],
        expansion_reason: str,
        *,
        extra_provenance: list[ViolationProvenance] | None = None,
    ) -> ReconfigurationScope:
        all_provenance = [*(provenance or []), *(extra_provenance or [])]
        required = [name for name in ReconfigurationScope._fields() if getattr(scope, name)]
        metadata = {
            "scope_id": scope.scope_id,
            "provenance": [item.to_dict() for item in all_provenance],
            "expansion_reason": expansion_reason,
            "size_primitives": {name: len(getattr(scope, name)) for name in ReconfigurationScope._fields()},
            "estimated_recoverability": None,
            "required_acquisition_components": required,
            "preserve_resume_recompute": {"preserve": True, "resume": True, "recompute": False},
        }
        metadata["scope_id"] = "scope-" + hashlib.sha256(
            json.dumps({name: sorted(getattr(scope, name)) for name in scope._fields()} | {
                "provenance": [item.identity() for item in all_provenance],
                "reason": expansion_reason,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return ReconfigurationScope(**scope.affected_entities(), metadata=metadata)

    def _limit(self, scope: ReconfigurationScope) -> ReconfigurationScope:
        if self.max_scope_entities is None:
            return scope
        return scope.truncate(self.max_scope_entities)

    @staticmethod
    def _from_observation(monitor_state: Any, current_config: Any) -> ReconfigurationScope:
        return extract_typed_affected_entities(monitor_state, current_config)

    @staticmethod
    def _expand_dependencies(scope: ReconfigurationScope, graph: Mapping[str, Any]) -> ReconfigurationScope:
        expanded = scope
        for entity, neighbours in sorted(graph.items(), key=lambda item: str(item[0])):
            if not isinstance(neighbours, Iterable) or isinstance(neighbours, (str, bytes)):
                continue
            if ScopeGenerator._scope_contains_graph_entity(scope, str(entity)):
                for value in sorted((str(v) for v in neighbours), key=str):
                    expanded = expanded.union(ScopeGenerator._typed_entity_scope(value))
        return expanded

    @staticmethod
    def _scope_contains_graph_entity(scope: ReconfigurationScope, value: str) -> bool:
        lowered = value.lower()
        for entity_type in ("task", "source", "node", "link", "route", "resource"):
            if lowered.startswith((f"{entity_type}:", f"{entity_type}_id:", f"{entity_type}id:")):
                parsed = _clean_explicit_id(value, entity_type)
                return bool(parsed and scope.contains(parsed, entity_type))
        return scope.contains(value)

    @staticmethod
    def _typed_entity_scope(value: str) -> ReconfigurationScope:
        text = str(value)
        lowered = text.lower()
        for entity_type, field_name in (("task", "task_ids"), ("source", "source_ids"), ("node", "node_ids"),
                                        ("link", "link_ids"), ("route", "route_ids"), ("resource", "resource_keys")):
            if lowered.startswith((f"{entity_type}:", f"{entity_type}_id:", f"{entity_type}id:")):
                parsed = _clean_explicit_id(text, entity_type)
                return ReconfigurationScope(**{field_name: {parsed}}) if parsed else ReconfigurationScope()
        return ReconfigurationScope(node_ids={text}) if text else ReconfigurationScope()

    @staticmethod
    def _global_scope(current_config: Any, planner_state: Any | None) -> ReconfigurationScope:
        payload: dict[str, set[str]] = {}
        for name in ReconfigurationScope._fields():
            values = set(getattr(current_config, f"covered_{name}", set()) or set())
            if planner_state is not None and name == "node_ids":
                values.update(str(k) for k in (getattr(planner_state, "detailed_resources", {}) or {}))
            payload[name] = values
        return ReconfigurationScope(**payload)
