from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.config import load_config
from trisatflow.envs.physical_metrics import (
    DEPRECATED_METRIC_ALIASES,
    ENERGY_CONVERSION_RULE_WH_TO_J,
    METRIC_SCHEMA_VERSION,
    METRIC_UNITS,
    infer_trace_delay_semantic,
    is_paper_safe_delay_semantic,
)
from trisatflow.envs.units import (
    TraceDelayInterpretation,
    classify_legacy_trace_delay_rows,
    has_physical_unit_suffix,
    is_normalized_or_training_field,
    validate_metric_field_names,
)

TIERS = ["local", "neighbor", "geo", "ground"]
MIN_PHASE_RATIO = 0.05
MIN_TASK_TYPE_RATIO = 0.05
MAX_PHASE_RATIO = 0.60
LOCAL_PROP_DELAY_MAX_MEAN = 1.0e-3
REQUIRED_STAGE3_METRIC_FIELDS = (
    "mean_deadline_exceedance",
    "mean_deadline_violation_ratio",
    "mean_delay_s",
    "mean_energy_j",
    "mean_queue_length_tasks",
    "normalized_system_cost",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            arr = payload.get("rows", payload.get("snapshots", []))
            if isinstance(arr, list):
                return [dict(item) for item in arr]
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(item) for item in csv.DictReader(f)]


def _series(rows: Iterable[Mapping[str, Any]], key: str) -> List[float]:
    return [max(0.0, _to_float(row.get(key), 0.0)) for row in rows if row.get(key) not in (None, "")]


def _range(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
    }


def _tier_ranges(rows: List[Dict[str, Any]], suffix: str) -> Dict[str, Dict[str, float]]:
    return {tier: _range(_series(rows, f"{tier}_{suffix}")) for tier in TIERS}


def _total_delay(row: Mapping[str, Any], tier: str) -> float:
    explicit = row.get(f"{tier}_total_delay")
    if explicit not in (None, ""):
        return max(0.0, _to_float(explicit, 0.0))
    return sum(
        max(0.0, _to_float(row.get(f"{tier}_{component}"), 0.0))
        for component in ("prop_delay", "tx_delay", "compute_delay", "queue_delay")
    )


def _visible(row: Mapping[str, Any], tier: str) -> bool:
    if row.get(f"{tier}_visible") in (None, ""):
        return tier == "local"
    return bool(row.get(f"{tier}_visible"))


def _distribution(counter: Counter[str], total: int) -> Dict[str, float]:
    return {key: float(value / max(1, total)) for key, value in sorted(counter.items())}


def _nearly_constant(stats: Dict[str, float]) -> bool:
    return stats["max"] - stats["min"] <= 1.0e-9 or stats["std"] <= 1.0e-6


def _resolve_trace_path(raw_trace: str, *, config_path: Path | None) -> Path:
    explicit = Path(raw_trace)
    candidates = [explicit]
    if not explicit.is_absolute():
        candidates.append(Path.cwd() / explicit)
        if config_path is not None:
            candidates.append(config_path.parent / explicit)
        candidates.append(Path(__file__).resolve().parents[1] / explicit)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"trace not found: {raw_trace}")


def _collect_row_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(str(key))
    return keys


def _row_bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _audit_output_root(input_root: Path, *, strict: bool) -> Dict[str, Any]:
    violations: List[str] = []
    metrics_path = input_root / "metrics.csv"
    manifest_path = input_root / "manifest.json"
    metadata_path = input_root / "run_metadata.json"

    if not metrics_path.is_file():
        violations.append("missing_metrics_csv")
        metric_rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []
    else:
        metric_rows = _read_rows(metrics_path)
        fieldnames = _collect_row_keys(metric_rows)
        for field in REQUIRED_STAGE3_METRIC_FIELDS:
            if field not in fieldnames:
                violations.append(f"missing_stage3_metric_field:{field}")
        if "mean_deadline_violation" in fieldnames and (
            "mean_deadline_exceedance" not in fieldnames or "mean_deadline_violation_ratio" not in fieldnames
        ):
            violations.append("legacy_deadline_violation_without_split_fields")
        if "mean_system_cost" in fieldnames and "normalized_system_cost" not in fieldnames:
            violations.append("legacy_system_cost_without_normalized_system_cost")

    manifest = _read_json(manifest_path)
    metadata = _read_json(metadata_path)
    schema_payload: Dict[str, Any] = {**metadata, **manifest}
    schema_version = str(schema_payload.get("metric_schema_version", ""))
    if schema_version != METRIC_SCHEMA_VERSION:
        violations.append(f"metric_schema_version_mismatch:{schema_version or '<missing>'}")

    metric_units = schema_payload.get("metric_units")
    if not isinstance(metric_units, Mapping):
        violations.append("missing_metric_units")
        metric_units = {}
    for field, unit in METRIC_UNITS.items():
        if str(metric_units.get(field, "")) != str(unit):
            violations.append(f"metric_unit_mismatch:{field}")

    for required_key in ("delay_unit", "energy_unit", "queue_unit"):
        if required_key not in schema_payload:
            violations.append(f"missing_manifest_unit:{required_key}")

    cost_normalization = schema_payload.get("cost_normalization")
    if not isinstance(cost_normalization, Mapping):
        violations.append("missing_cost_normalization")
        cost_components = {}
    else:
        cost_components = cost_normalization.get("components")
        if not isinstance(cost_components, Mapping) or not cost_components:
            violations.append("missing_normalized_system_cost_components")
            cost_components = {}
        for component in ("delay", "queue", "transmission", "compute", "energy", "feasibility"):
            if component not in cost_components:
                violations.append(f"missing_normalized_system_cost_component:{component}")

    if "reward_weights" not in schema_payload:
        violations.append("missing_reward_weights")

    trace_semantic = str(schema_payload.get("trace_delay_source_semantic", "")).strip().lower()
    if not trace_semantic:
        violations.append("missing_trace_delay_source_semantic")
    if strict and trace_semantic in {"seconds", "physical_seconds", "physical seconds"}:
        violations.append("score_like_trace_delay_marked_as_seconds")
    if strict and "legacy_trace_delay_score" in fieldnames and "excluded from *_delay_s" not in trace_semantic:
        violations.append("legacy_trace_delay_score_without_physical_delay_exclusion_semantic")
    delay_semantic = str(schema_payload.get("delay_semantic", "")).strip()
    if strict and delay_semantic == "legacy_unknown":
        violations.append("delay_semantic_legacy_unknown")
    if strict and delay_semantic == "normalized_score":
        for field in fieldnames:
            if field.endswith("_delay_s") or field in {"mean_delay_s", "p95_delay_s", "physical_delay_s"}:
                violations.append(f"normalized_score_written_to_delay_seconds:{field}")

    aliases = schema_payload.get("deprecated_metric_aliases")
    if not isinstance(aliases, Mapping):
        aliases = {}
    for deprecated in DEPRECATED_METRIC_ALIASES:
        if deprecated in fieldnames and deprecated not in aliases:
            violations.append(f"deprecated_alias_not_declared:{deprecated}")

    normalized_named_as_physical = validate_metric_field_names(fieldnames)
    for key in normalized_named_as_physical:
        violations.append(f"normalized_field_misnamed_as_physical:{key}")

    return {
        "input_root": str(input_root),
        "strict": bool(strict),
        "metrics_csv": str(metrics_path),
        "manifest": str(manifest_path),
        "run_metadata": str(metadata_path),
        "metric_schema_version": schema_version,
        "metric_fields": fieldnames,
        "metric_units": dict(metric_units),
        "normalized_system_cost_components": dict(cost_components),
        "violations": sorted(set(violations)),
        "status": "PHYSICAL_PLAUSIBILITY_OK" if not violations else "PHYSICAL_PLAUSIBILITY_FAILED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check physical plausibility and metric naming in SatEdgeSim-like traces.")
    parser.add_argument("--trace", type=str, default="", help="Trace path (.jsonl/.json/.csv).")
    parser.add_argument("--config", type=str, default="", help="TriSatFlow YAML config; trace path is read from scenario.topology_trace_path.")
    parser.add_argument("--input-root", type=str, default="", help="Experiment output directory containing metrics.csv and manifest.json.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing Stage-3 output contract fields.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--delay-anomaly-threshold-s", type=float, default=None)
    args = parser.parse_args()

    if args.input_root:
        payload = _audit_output_root(Path(args.input_root), strict=bool(args.strict))
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        if payload["violations"]:
            raise SystemExit(1)
        return

    if not args.trace and not args.config:
        raise SystemExit("either --trace, --config, or --input-root must be provided")

    cfg = None
    cfg_path: Path | None = None
    raw_trace = str(args.trace or "").strip()
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg = load_config(cfg_path)
        if not raw_trace:
            raw_trace = str(getattr(cfg.scenario, "topology_trace_path", "") or "").strip()
        if not raw_trace:
            raise SystemExit("scenario.topology_trace_path is empty in config; pass --trace explicitly")

    trace_path = _resolve_trace_path(raw_trace, config_path=cfg_path)
    rows = _read_rows(trace_path)

    threshold = (
        float(args.delay_anomaly_threshold_s)
        if args.delay_anomaly_threshold_s is not None
        else float(getattr(getattr(cfg, "scenario", object()), "trace_delay_anomaly_threshold_s", 1.0e3))
    )
    interpretation = TraceDelayInterpretation(
        anomaly_threshold_s=threshold,
        treat_anomaly_as_legacy_score=bool(
            getattr(getattr(cfg, "scenario", object()), "trace_treat_large_delay_as_legacy_score", True)
        ),
    )

    phase_counter: Counter[str] = Counter(str(row.get("scenario_phase", "unknown_phase")) for row in rows)
    task_type_counter: Counter[str] = Counter(str(row.get("task_type", "unknown_task")) for row in rows)
    delay_ranges = _tier_ranges(rows, "total_delay")
    rate_ranges = _tier_ranges(rows, "rate")
    queue_ranges = _tier_ranges(rows, "best_queue")
    compute_capacity_ranges = _tier_ranges(rows, "compute_capacity")
    propagation_ranges = _tier_ranges(rows, "prop_delay")

    violations: List[str] = []
    delay_anomaly_count = 0
    delay_semantic_counter: Counter[str] = Counter()
    energy_rows = 0
    energy_conversion_violations = 0
    cumulative_energy_as_step_count = 0
    for tier in TIERS:
        for row in rows:
            delay_val = _total_delay(row, tier)
            delay_semantic = infer_trace_delay_semantic(row)
            delay_semantic_counter[delay_semantic] += 1
            if delay_val > threshold and delay_semantic == "legacy_unknown":
                delay_anomaly_count += 1
        if any(_to_float(row.get(f"{tier}_total_delay"), 0.0) < 0.0 for row in rows if row.get(f"{tier}_total_delay") not in (None, "")):
            violations.append(f"{tier}_total_delay_negative")
        if any(_to_float(row.get(f"{tier}_rate"), 0.0) <= 0.0 for row in rows if _visible(row, tier)):
            violations.append(f"{tier}_rate_non_positive")
        if any(_to_float(row.get(f"{tier}_best_queue"), 0.0) < 0.0 for row in rows if row.get(f"{tier}_best_queue") not in (None, "")):
            violations.append(f"{tier}_queue_negative")

    if delay_anomaly_count > 0:
        violations.append(f"delay_anomaly_count={delay_anomaly_count}")

    for row in rows:
        delay_semantic = infer_trace_delay_semantic(row)
        if args.strict and not is_paper_safe_delay_semantic(delay_semantic):
            violations.append(f"paper_unsafe_delay_semantic:{delay_semantic}")
        if args.strict and delay_semantic == "normalized_score":
            for key in row.keys():
                if str(key).endswith("_delay_s"):
                    violations.append(f"normalized_score_written_to_delay_seconds:{key}")
        if row.get("raw_energy_counter_wh") not in (None, "") or row.get("step_energy_delta_wh") not in (None, ""):
            energy_rows += 1
            raw = _to_float(row.get("raw_energy_counter_wh"), 0.0)
            previous = _to_float(row.get("previous_raw_energy_counter_wh"), raw)
            delta_wh = _to_float(row.get("step_energy_delta_wh"), -1.0)
            delta_j = _to_float(row.get("step_energy_delta_j"), -1.0)
            rule = str(row.get("energy_conversion_rule") or "")
            expected_wh = max(0.0, raw - previous)
            expected_j = expected_wh * 3600.0
            if abs(delta_wh - expected_wh) > 1.0e-9 or abs(delta_j - expected_j) > 1.0e-6 or rule != ENERGY_CONVERSION_RULE_WH_TO_J:
                energy_conversion_violations += 1
            if row.get("physical_energy_j") not in (None, "") and abs(_to_float(row.get("physical_energy_j"), 0.0) - raw) <= 1.0e-9:
                cumulative_energy_as_step_count += 1
            if row.get("energy_unit") == "J" and row.get("energy_raw_delta") not in (None, "") and row.get("step_energy_delta_j") in (None, ""):
                energy_conversion_violations += 1
    if args.strict and energy_conversion_violations > 0:
        violations.append(f"energy_conversion_violations={energy_conversion_violations}")
    if args.strict and cumulative_energy_as_step_count > 0:
        violations.append(f"cumulative_energy_counter_used_as_step_energy={cumulative_energy_as_step_count}")

    if propagation_ranges["geo"]["mean"] <= max(propagation_ranges["neighbor"]["mean"], propagation_ranges["local"]["mean"]):
        violations.append("geo_propagation_not_greater_than_neighbor_local")
    if propagation_ranges["local"]["mean"] > LOCAL_PROP_DELAY_MAX_MEAN:
        violations.append("local_propagation_not_near_zero")

    # CPU capacity positivity checks from config and/or trace fields.
    if cfg is not None:
        scenario = cfg.scenario
        if float(scenario.leo_cpu_capacity) <= 0.0:
            violations.append("leo_cpu_capacity_non_positive")
        if float(scenario.geo_cpu_capacity) <= 0.0:
            violations.append("geo_cpu_capacity_non_positive")
        if float(scenario.ground_cpu_capacity) <= 0.0:
            violations.append("ground_cpu_capacity_non_positive")
    for tier in TIERS:
        if any(_to_float(row.get(f"{tier}_compute_capacity"), 0.0) <= 0.0 for row in rows if row.get(f"{tier}_compute_capacity") not in (None, "")):
            violations.append(f"{tier}_compute_capacity_non_positive")

    # Energy non-negativity if trace contains energy metrics.
    energy_keys = [k for k in _collect_row_keys(rows) if "energy" in k.lower()]
    for key in energy_keys:
        if any(_to_float(row.get(key), 0.0) < 0.0 for row in rows if row.get(key) not in (None, "")):
            violations.append(f"{key}_negative")

    keys = _collect_row_keys(rows)
    normalized_named_as_physical = validate_metric_field_names(keys)
    for key in normalized_named_as_physical:
        violations.append(f"normalized_field_misnamed_as_physical:{key}")

    # Also detect suspicious physical-suffix fields that are clearly marked as scores.
    suspicious_score_fields = [
        key for key in keys
        if has_physical_unit_suffix(key)
        and any(token in key.lower() for token in ("legacy", "score", "normalized"))
    ]
    for key in suspicious_score_fields:
        violations.append(f"suspicious_physical_field_name:{key}")

    phase_distribution = _distribution(phase_counter, len(rows))
    task_type_distribution = _distribution(task_type_counter, len(rows))
    for phase, ratio in phase_distribution.items():
        if ratio < MIN_PHASE_RATIO:
            violations.append(f"phase_ratio_too_low:{phase}:{ratio:.6f}")
        if ratio > MAX_PHASE_RATIO:
            violations.append(f"phase_ratio_too_high:{phase}:{ratio:.6f}")
    for task_type, ratio in task_type_distribution.items():
        if ratio < MIN_TASK_TYPE_RATIO:
            violations.append(f"task_type_ratio_too_low:{task_type}:{ratio:.6f}")

    oracle_winners: Counter[str] = Counter()
    for row in rows:
        visible_tiers = [tier for tier in TIERS if _visible(row, tier)]
        if not visible_tiers:
            continue
        winner = min(visible_tiers, key=lambda tier: _total_delay(row, tier))
        oracle_winners[winner] += 1
    if oracle_winners:
        dominant_tier, dominant_count = max(oracle_winners.items(), key=lambda item: item[1])
        if dominant_count == len(rows):
            violations.append(f"single_tier_always_min_delay:{dominant_tier}")

    for tier in TIERS:
        if _nearly_constant(queue_ranges[tier]):
            violations.append(f"{tier}_queue_nearly_constant")
        if _nearly_constant(delay_ranges[tier]):
            violations.append(f"{tier}_delay_nearly_constant")

    legacy_normalized_field_candidates = [
        key for key in keys
        if is_normalized_or_training_field(key) and not has_physical_unit_suffix(key)
    ]
    legacy_delay_rows = classify_legacy_trace_delay_rows(rows, interpretation=interpretation)

    payload: Dict[str, Any] = {
        "trace": str(trace_path),
        "config": str(cfg_path) if cfg_path else "",
        "num_rows": len(rows),
        "delay_range_by_tier": delay_ranges,
        "rate_range_by_tier": rate_ranges,
        "queue_range_by_tier": queue_ranges,
        "compute_capacity_range_by_tier": compute_capacity_ranges,
        "propagation_delay_range_by_tier": propagation_ranges,
        "phase_distribution": phase_distribution,
        "task_type_distribution": task_type_distribution,
        "delay_anomaly_threshold_s": threshold,
        "delay_anomaly_count": delay_anomaly_count,
        "delay_semantic_distribution": dict(delay_semantic_counter),
        "energy_rows_with_counters": energy_rows,
        "energy_conversion_violations": energy_conversion_violations,
        "cumulative_energy_as_step_count": cumulative_energy_as_step_count,
        "legacy_trace_delay_row_count": legacy_delay_rows,
        "normalized_field_name_violations": normalized_named_as_physical,
        "legacy_normalized_field_candidates": sorted(set(legacy_normalized_field_candidates)),
        "violations": sorted(set(violations)),
        "status": "PHYSICAL_PLAUSIBILITY_OK" if not violations else "PHYSICAL_PLAUSIBILITY_FAILED",
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
