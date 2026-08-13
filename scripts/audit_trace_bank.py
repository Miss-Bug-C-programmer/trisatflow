#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REQUIRED_MANIFEST_FIELDS = {
    "trace_sha256",
    "trace_semantic_class",
    "trace_origin",
    "synthetic",
    "source_simulator_commit",
    "simulator_version",
    "rest_api_schema_version",
    "state_schema_version",
    "settings_sha256",
    "exporter_version",
    "seed",
    "scenario_parameters",
    "scenario_profile",
    "task_source_mode",
    "success_profile",
    "action_mask_mode",
    "min_link_survival_margin_sec",
    "architecture",
    "n_leo",
    "num_steps",
    "trace_generation_mode",
    "dense_projection_mode",
    "candidate_cost_estimator_version",
    "lower_action_binding_version",
    "energy_counter_unit",
    "energy_counter_semantics",
}
MASK_FIELDS = (
    "abstract_action_mask_visible",
    "abstract_action_mask_completion_safe",
    "abstract_action_mask_mobility_safe",
    "abstract_action_mask_final",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _mask(row: Dict[str, Any], key: str) -> List[int]:
    raw = row.get(key)
    if isinstance(raw, str) and raw.startswith("["):
        raw = json.loads(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return [1 if _bool(raw[i]) else 0 for i in range(4)]
    return [0, 0, 0, 0]


def _transition_count(rows: List[Dict[str, Any]]) -> int:
    grouped: Dict[int, List[tuple[int, List[int]]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("leo_id", 0))].append((int(row.get("step", 0)), _mask(row, "abstract_action_mask_final")))
    count = 0
    for items in grouped.values():
        items.sort(key=lambda item: item[0])
        for (_, prev), (_, curr) in zip(items, items[1:]):
            count += int(prev[1:] != curr[1:])
    return count


def _prune_ratio(rows: List[Dict[str, Any]], before_key: str, after_key: str) -> float:
    opportunities = 0
    pruned = 0
    for row in rows:
        before = _mask(row, before_key)
        after = _mask(row, after_key)
        for idx in (1, 2, 3):
            opportunities += int(before[idx])
            pruned += int(before[idx] and not after[idx])
    return pruned / max(1, opportunities)


def _split_name(path: Path) -> str:
    for part in path.parts:
        if part in {"train", "validation", "test"}:
            return part
    return "unknown"


def _class_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "actual_projection" in parts:
        return "actual_projection"
    if "actual_sequential_live" in parts:
        return "actual_sequential_live"
    if "controlled_stress_projection" in parts:
        return "controlled_stress_projection"
    return "unknown"


def _audit_trace(path: Path, *, require_provenance: bool, paper_strict: bool) -> Dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    coverage_path = path.with_suffix(path.suffix + ".coverage.json")
    failures: List[str] = []
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    coverage = _read_json(coverage_path) if coverage_path.is_file() else {}
    rows = _read_jsonl(path)

    if not manifest_path.is_file():
        failures.append("missing_manifest")
    if not coverage_path.is_file():
        failures.append("missing_coverage")
    missing_fields = sorted(field for field in REQUIRED_MANIFEST_FIELDS if manifest.get(field) in (None, ""))
    if missing_fields:
        failures.append(f"manifest_missing_fields={missing_fields}")
    if manifest.get("trace_sha256") and manifest.get("trace_sha256") != _sha256_file(path):
        failures.append("trace_sha256_mismatch")
    if paper_strict and str(manifest.get("success_profile")) != "paper_strict":
        failures.append(f"success_profile_not_paper_strict={manifest.get('success_profile')}")
    if str(manifest.get("success_profile")) == "preflight_lenient":
        failures.append("preflight_lenient_in_paper_bank")
    if manifest.get("synthetic") or any(_bool(row.get("synthetic")) for row in rows):
        failures.append("synthetic_trace_in_paper_bank")
    if str(manifest.get("trace_origin")) != "satedgesim":
        failures.append(f"trace_origin_not_satedgesim={manifest.get('trace_origin')}")

    if require_provenance:
        for field in (
            "source_simulator_commit",
            "simulator_version",
            "rest_api_schema_version",
            "state_schema_version",
            "settings_sha256",
            "candidate_cost_estimator_version",
            "lower_action_binding_version",
        ):
            value = str(manifest.get(field, "")).strip().lower()
            if not value or value == "unknown":
                failures.append(f"provenance_missing_or_unknown={field}")

    row_classes = Counter(str(row.get("trace_semantic_class", "")) for row in rows)
    row_modes = Counter(str(row.get("trace_generation_mode", "")) for row in rows)
    bank_class = _class_from_path(path)
    semantic = str(manifest.get("trace_semantic_class", ""))
    mode = str(manifest.get("trace_generation_mode", ""))
    dense_mode = str(manifest.get("dense_projection_mode", ""))

    if bank_class == "actual_projection":
        if semantic != "actual_physical_projection" or mode != "dense_projection" or dense_mode != "source_projection":
            failures.append("actual_projection_semantic_mismatch")
        if str(manifest.get("queue_estimate_source", "live")) == "controlled_estimate":
            failures.append("actual_bank_uses_controlled_queue_estimate")
        if str(manifest.get("mobility_risk_source", "live")) == "controlled_estimate":
            failures.append("actual_bank_uses_controlled_mobility_estimate")
    elif bank_class == "actual_sequential_live":
        if semantic != "actual_physical_sequential_live" or mode != "sequential_live" or dense_mode != "none":
            failures.append("actual_sequential_live_semantic_mismatch")
    elif bank_class == "controlled_stress_projection":
        if semantic != "controlled_stress_projection" or mode != "dense_projection" or dense_mode != "source_projection":
            failures.append("controlled_stress_semantic_mismatch")
        if str(manifest.get("queue_estimate_source")) != "controlled_estimate":
            failures.append("controlled_stress_queue_source_not_controlled")
        if str(manifest.get("mobility_risk_source")) != "controlled_estimate":
            failures.append("controlled_stress_mobility_source_not_controlled")
    else:
        failures.append("unknown_bank_class")

    if len(row_classes) != 1 or next(iter(row_classes), "") != semantic:
        failures.append(f"row_semantic_class_mismatch={dict(row_classes)}")
    if len(row_modes) != 1 or next(iter(row_modes), "") != mode:
        failures.append(f"row_trace_generation_mode_mismatch={dict(row_modes)}")

    missing_layer_rows = sum(1 for row in rows if any(field not in row for field in MASK_FIELDS))
    if missing_layer_rows:
        failures.append(f"missing_layered_mask_rows={missing_layer_rows}")

    if mode == "dense_projection":
        if coverage.get("status") != "DENSE_TRACE_OK":
            failures.append(f"dense_projection_coverage_not_ok={coverage.get('status')}")
        if float(coverage.get("dense_coverage_ratio", 0.0)) < 0.99:
            failures.append(f"dense_projection_coverage_ratio={coverage.get('dense_coverage_ratio')}")
    if mode == "sequential_live" and any(str(row.get("dense_projection_mode")) != "none" for row in rows):
        failures.append("sequential_live_rows_use_dense_projection_mode")

    transition_count = _transition_count(rows)
    phase_count = len({str(row.get("phase_id", row.get("scenario_phase", ""))) for row in rows})
    completion_prune = _prune_ratio(rows, "abstract_action_mask_visible", "abstract_action_mask_completion_safe")
    mobility_prune = _prune_ratio(rows, "abstract_action_mask_visible", "abstract_action_mask_mobility_safe")
    if bank_class == "controlled_stress_projection":
        if phase_count < 2:
            failures.append(f"stress_bank_phase_count={phase_count}")
        if mobility_prune <= 0.0:
            failures.append("stress_bank_missing_mobility_prune")
        if completion_prune <= 0.0:
            failures.append("stress_bank_missing_completion_prune")

    return {
        "path": str(path),
        "split": _split_name(path),
        "bank_class": bank_class,
        "trace_sha256": manifest.get("trace_sha256", ""),
        "num_rows": len(rows),
        "num_steps": manifest.get("num_steps"),
        "semantic": semantic,
        "mode": mode,
        "phase_count": phase_count,
        "transition_count": transition_count,
        "completion_prune_ratio": completion_prune,
        "mobility_prune_ratio": mobility_prune,
        "failures": failures,
    }


def _write_index(trace_root: Path, records: Iterable[Dict[str, Any]]) -> None:
    payload = {
        "trace_root": str(trace_root),
        "schema_version": "paper_v3_trace_bank_index_v1",
        "traces": list(records),
    }
    (trace_root / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper-ready v3 SatEdgeSim trace bank.")
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--require-disjoint-splits", action="store_true")
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--paper-strict", action="store_true")
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    traces = sorted(path for path in trace_root.rglob("*.jsonl") if path.is_file())
    failures: List[str] = []
    if not traces:
        failures.append("no_jsonl_traces")

    records = [
        _audit_trace(path, require_provenance=args.require_provenance, paper_strict=args.paper_strict)
        for path in traces
    ]
    for record in records:
        failures.extend(f"{record['path']}: {failure}" for failure in record["failures"])

    if args.require_disjoint_splits:
        by_split: Dict[str, set[str]] = defaultdict(set)
        for record in records:
            by_split[str(record["split"])].add(str(record.get("trace_sha256", "")))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = by_split[left] & by_split[right]
            overlap.discard("")
            if overlap:
                failures.append(f"split_sha256_overlap:{left}:{right}:{sorted(overlap)}")

    classes = Counter(str(record["bank_class"]) for record in records)
    for required in ("actual_projection", "actual_sequential_live", "controlled_stress_projection"):
        if classes.get(required, 0) <= 0:
            failures.append(f"missing_required_bank_class={required}")
    actual_transition_count = sum(
        int(record.get("transition_count", 0))
        for record in records
        if str(record.get("bank_class", "")).startswith("actual")
    )
    if actual_transition_count <= 0:
        failures.append("actual_bank_missing_topology_transition")

    _write_index(trace_root, records)
    summary = {
        "status": "TRACE_BANK_AUDIT_OK" if not failures else "TRACE_BANK_AUDIT_FAILED",
        "trace_root": str(trace_root),
        "num_traces": len(records),
        "bank_classes": dict(classes),
        "records": records,
        "failures": failures,
        "index": str(trace_root / "index.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
