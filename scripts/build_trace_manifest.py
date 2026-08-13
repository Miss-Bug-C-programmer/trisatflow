from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trisatflow.data.trace_hashing import content_fingerprint, hash_mapping, sha256_file
from trisatflow.data.trace_manifest import detect_oracle_field_names, infer_split, missing_metadata, normalize_manifest_record, write_manifest_file


TRACE_SUFFIXES = {".jsonl", ".json", ".csv", ".npz"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _walk_values(obj: Any) -> Iterable[str]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in {"topology_trace_path", "trace_path", "trace"} and isinstance(value, str):
                yield value
            yield from _walk_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_values(item)


def _resolve_path(raw: str, *, source_file: Path, project_root: Path) -> Path:
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([project_root / p, source_file.parent / p])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _first_jsonl_row(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() != ".jsonl" or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        return json.loads(line) if line else {}
    except Exception:
        return {}


def _scan_jsonl_metadata(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() != ".jsonl" or not path.exists():
        return {}
    rows = 0
    first: Dict[str, Any] = {}
    last: Dict[str, Any] = {}
    min_step = None
    max_step = None
    min_time = None
    max_time = None
    leo_ids: set[int] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if not first:
                    first = row
                last = row
                rows += 1
                step = row.get("step")
                if step is not None:
                    try:
                        value = int(step)
                        min_step = value if min_step is None else min(min_step, value)
                        max_step = value if max_step is None else max(max_step, value)
                    except Exception:
                        pass
                sim_time = row.get("simulation_time") or row.get("time_s") or row.get("timestamp_s")
                if sim_time is not None:
                    try:
                        value = float(sim_time)
                        min_time = value if min_time is None else min(min_time, value)
                        max_time = value if max_time is None else max(max_time, value)
                    except Exception:
                        pass
                leo_id = row.get("leo_id")
                if leo_id is not None:
                    try:
                        leo_ids.add(int(leo_id))
                    except Exception:
                        pass
    except Exception:
        return {}
    num_steps = (max_step + 1) if max_step is not None and min_step in {0, None} else None
    return {
        "num_rows_scanned": rows,
        "num_steps_from_rows": num_steps,
        "n_leo_from_rows": len(leo_ids) if leo_ids else None,
        "time_start_s_from_rows": min_time,
        "time_end_s_from_rows": (max_time + 1.0) if max_time is not None else None,
        "first_row": first,
        "last_row": last,
    }


def _load_sidecar_manifest(trace_path: Path) -> Dict[str, Any]:
    candidates = [
        trace_path.with_name(trace_path.name + ".manifest.json"),
        trace_path.with_suffix(trace_path.suffix + ".manifest.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {"sidecar_manifest_error": str(candidate)}
    return {}


def _load_coverage(trace_path: Path) -> Dict[str, Any]:
    return _load_json(trace_path.with_name(trace_path.name + ".coverage.json"))


def _truthy_first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _infer_source(path: Path, metadata: Mapping[str, Any]) -> str:
    text = path.as_posix().lower()
    if bool(metadata.get("synthetic")) or "synthetic" in text:
        return "synthetic"
    if "satedgesim" in text or str(metadata.get("trace_origin", "")).lower() == "satedgesim":
        return "satedgesim_export"
    if "sgp4" in text:
        return "sgp4_generated"
    return "legacy"


def _infer_seed(path: Path, metadata: Mapping[str, Any]) -> int | None:
    for key in ("generation_seed", "seed"):
        if metadata.get(key) is not None:
            try:
                return int(metadata[key])
            except Exception:
                return None
    stem = path.stem.lower()
    if "seed_" in stem:
        tail = stem.split("seed_", 1)[1].split(".", 1)[0]
    elif "seed" in stem:
        tail = stem.split("seed", 1)[1].split(".", 1)[0]
    else:
        return None
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def _trace_index_entries(project_root: Path) -> Dict[Path, Tuple[Path, Dict[str, Any]]]:
    entries: Dict[Path, Tuple[Path, Dict[str, Any]]] = {}
    traces_root = project_root / "traces"
    if not traces_root.exists():
        return entries
    for index_path in traces_root.rglob("index.json"):
        payload = _load_json(index_path)
        for item in payload.get("traces", []) if isinstance(payload.get("traces"), list) else []:
            if not isinstance(item, Mapping) or not item.get("path"):
                continue
            trace_path = _resolve_path(str(item["path"]), source_file=index_path, project_root=project_root)
            enriched = dict(item)
            enriched["trace_bank_index"] = index_path.as_posix()
            entries[trace_path] = (index_path, enriched)
    return entries


def _is_trace_data_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".manifest.json") or name.endswith(".coverage.json") or name.startswith("obs_norm"):
        return False
    if name == "index.json":
        return False
    if name.startswith("_tmp"):
        return False
    return path.suffix.lower() in TRACE_SUFFIXES


def build_record(trace_path: Path, *, project_root: Path, source_file: Path | None = None, config_payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    exists = trace_path.exists()
    sidecar = _load_sidecar_manifest(trace_path) if exists else {}
    coverage = _load_coverage(trace_path) if exists else {}
    first_row = _first_jsonl_row(trace_path) if exists else {}
    scanned = _scan_jsonl_metadata(trace_path) if exists else {}
    scenario = dict(config_payload.get("scenario", {})) if isinstance(config_payload, Mapping) and isinstance(config_payload.get("scenario"), Mapping) else {}
    index_entry = dict(config_payload.get("trace_index_entry", {})) if isinstance(config_payload, Mapping) and isinstance(config_payload.get("trace_index_entry"), Mapping) else {}
    oracle_names = sorted(set(detect_oracle_field_names(sidecar)) | set(detect_oracle_field_names(first_row)))
    rel_path = trace_path
    try:
        rel_path = trace_path.relative_to(project_root)
    except Exception:
        pass
    merged_source_metadata = {**index_entry, **coverage, **first_row, **sidecar}
    slot_duration = _truthy_first(sidecar.get("slot_duration_s"), scenario.get("slot_duration_s"), 1.0)
    num_steps = _truthy_first(
        sidecar.get("num_steps"),
        sidecar.get("num_decision_steps"),
        coverage.get("num_decision_steps"),
        coverage.get("dense_steps"),
        index_entry.get("num_steps"),
        scanned.get("num_steps_from_rows"),
        first_row.get("numSteps"),
    )
    duration = sidecar.get("duration_s")
    if duration is None and num_steps is not None:
        try:
            duration = float(num_steps) * float(slot_duration)
        except Exception:
            duration = None
    time_start_s = _truthy_first(sidecar.get("time_start_s"), scanned.get("time_start_s_from_rows"))
    time_end_s = _truthy_first(sidecar.get("time_end_s"), scanned.get("time_end_s_from_rows"))
    n_leo = _truthy_first(
        sidecar.get("n_leo"),
        sidecar.get("scenario_parameters", {}).get("devices_count"),
        coverage.get("n_leo_requested"),
        coverage.get("devices_count_used"),
        scenario.get("n_leo"),
        scanned.get("n_leo_from_rows"),
    )
    scenario_profile = _truthy_first(
        sidecar.get("scenario_profile"),
        coverage.get("scenario_profile"),
        first_row.get("scenario_profile"),
        scenario.get("profile_name"),
        sidecar.get("bank_class"),
        index_entry.get("bank_class"),
        "unknown",
    )
    trace_semantic_class = _truthy_first(
        sidecar.get("trace_semantic_class"),
        first_row.get("trace_semantic_class"),
        index_entry.get("semantic"),
        sidecar.get("trace_generation_mode"),
        coverage.get("trace_generation_mode"),
    )
    trace_generation_mode = _truthy_first(sidecar.get("trace_generation_mode"), coverage.get("trace_generation_mode"), first_row.get("trace_generation_mode"), index_entry.get("mode"))
    settings_sha256 = _truthy_first(sidecar.get("settings_sha256"), index_entry.get("settings_sha256"))
    generation_seed = _infer_seed(trace_path, merged_source_metadata)
    split = _truthy_first(sidecar.get("split"), index_entry.get("split"), infer_split(rel_path.as_posix()))
    record: Dict[str, Any] = {
        "trace_id": rel_path.as_posix(),
        "trace_path": rel_path.as_posix(),
        "source": _infer_source(trace_path, merged_source_metadata),
        "generation_seed": generation_seed,
        "tle_epoch": sidecar.get("tle_epoch"),
        "n_leo": n_leo,
        "n_geo": sidecar.get("n_geo") or scenario.get("n_geo") or 1,
        "n_ground": sidecar.get("n_ground") or scenario.get("n_ground") or 1,
        "duration_s": duration,
        "slot_duration_s": slot_duration,
        "time_start_s": time_start_s,
        "time_end_s": time_end_s,
        "scenario_profile": scenario_profile,
        "split": split,
        "sha256": sha256_file(trace_path) if exists else "missing",
        "content_fingerprint": content_fingerprint(trace_path),
        "contains_oracle_fields": bool(oracle_names),
        "oracle_field_names": oracle_names,
        "safe_observable_excludes_oracle_fields": True,
        "generator_config_hash": sidecar.get("generator_config_hash") or hash_mapping(
            {
                "source": _infer_source(trace_path, merged_source_metadata),
                "seed": generation_seed,
                "scenario_profile": scenario_profile,
                "settings_sha256": settings_sha256,
                "trace_generation_mode": trace_generation_mode,
                "trace_semantic_class": trace_semantic_class,
            }
        ),
        "created_at": sidecar.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "notes": sidecar.get("notes") or "generated by build_trace_manifest.py; verify source metadata before paper claims",
        "status": "ok" if exists else "missing",
        "same_distribution_non_overlapping": sidecar.get("same_distribution_non_overlapping"),
        "split_independence_evidence": sidecar.get("split_independence_evidence") or (
            "trace_bank_index_seed_split" if index_entry.get("split") and generation_seed is not None else None
        ),
        "non_overlap_evidence": sidecar.get("non_overlap_evidence") or ("jsonl_simulation_time_scan" if time_start_s is not None and time_end_s is not None else None),
        "domain_shift": sidecar.get("domain_shift") or ("stress" in rel_path.as_posix().lower()),
        "source_file": str(source_file) if source_file else "",
        "trace_semantic_class": trace_semantic_class,
        "trace_generation_mode": trace_generation_mode,
        "settings_sha256": settings_sha256,
        "num_steps": num_steps,
        "num_rows": _truthy_first(sidecar.get("num_rows"), coverage.get("num_rows"), index_entry.get("num_rows"), scanned.get("num_rows_scanned")),
        "metadata_derivation": {
            "sidecar_manifest": bool(sidecar),
            "coverage_json": bool(coverage),
            "trace_bank_index": bool(index_entry),
            "jsonl_scanned": bool(scanned),
        },
    }
    return normalize_manifest_record(record)


def discover_trace_paths(project_root: Path) -> List[tuple[Path, Path | None, Dict[str, Any]]]:
    discovered: Dict[Path, tuple[Path | None, Dict[str, Any]]] = {}
    index_entries = _trace_index_entries(project_root)
    for root_name in ("trisatflow/configs", "outputs"):
        root = project_root / root_name
        if not root.exists():
            continue
        for cfg_path in root.rglob("*.yaml"):
            payload = _load_yaml(cfg_path)
            for raw in _walk_values(payload):
                if not raw:
                    continue
                path = _resolve_path(raw, source_file=cfg_path, project_root=project_root)
                discovered[path] = (cfg_path, payload)
    traces_root = project_root / "traces"
    if traces_root.exists():
        for item in traces_root.rglob("*"):
            if not item.is_file():
                continue
            if not _is_trace_data_file(item):
                continue
            resolved = item.resolve()
            index_source, index_entry = index_entries.get(resolved, (None, {}))
            existing_source, existing_payload = discovered.get(resolved, (None, {}))
            payload = dict(existing_payload or {})
            if index_entry:
                payload["trace_index_entry"] = index_entry
            discovered[resolved] = (existing_source or index_source, payload)
    return [(path, source, payload) for path, (source, payload) in sorted(discovered.items(), key=lambda x: str(x[0]))]


def build_manifests(project_root: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    active_manifest_files: List[str] = []
    for trace_path, source_file, payload in discover_trace_paths(project_root):
        record = build_record(trace_path, project_root=project_root, source_file=source_file, config_payload=payload)
        out_name = record["trace_id"].replace("/", "__").replace("\\", "__").replace(":", "_") + ".json"
        write_manifest_file(output_dir / out_name, record)
        active_manifest_files.append(out_name)
        rows.append(record)
    real_rows = [r for r in rows if r.get("status") == "ok" and r.get("source") != "synthetic"]
    incomplete = [{"trace_id": r.get("trace_id"), "missing": missing_metadata(r)} for r in rows if missing_metadata(r)]
    status = "ok"
    if not real_rows:
        status = "audit_failed_missing_manifest"
    elif incomplete:
        status = "audit_failed_incomplete_manifest_metadata"
    summary = {
        "manifest_build_status": status,
        "num_manifest_records": len(rows),
        "num_real_trace_records": len(real_rows),
        "output_dir": str(output_dir),
        "incomplete_records": incomplete,
        "active_manifest_files": sorted(active_manifest_files),
        "excluded_non_trace_patterns": ["*.manifest.json", "*.coverage.json", "obs_norm*", "index.json", "_tmp*"],
        "toy_manifest_is_not_evidence": True,
    }
    with (output_dir / "manifest_build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-empty-for-unit-test", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    summary = build_manifests(project_root, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["manifest_build_status"] == "audit_failed_missing_manifest" and not args.allow_empty_for_unit_test:
        return 2
    if summary["manifest_build_status"] == "audit_failed_incomplete_manifest_metadata":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
