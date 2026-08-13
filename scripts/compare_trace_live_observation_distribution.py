from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import FIELD_NAMES
from trisatflow.satedgesim_eval.inspection import (
    collect_live_states,
    current_dense_row_from_state,
    dense_rows_from_state,
    load_trace_groups,
    load_trace_rows,
    raw_field_series,
    summarize_field_stats,
)

DELAY_COMPONENT_FIELDS = [
    "local_prop_delay",
    "local_tx_delay",
    "local_compute_delay",
    "local_queue_delay",
    "local_total_delay",
    "neighbor_prop_delay",
    "neighbor_tx_delay",
    "neighbor_compute_delay",
    "neighbor_queue_delay",
    "neighbor_total_delay",
    "geo_prop_delay",
    "geo_tx_delay",
    "geo_compute_delay",
    "geo_queue_delay",
    "geo_total_delay",
    "ground_prop_delay",
    "ground_tx_delay",
    "ground_compute_delay",
    "ground_queue_delay",
    "ground_total_delay",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _shift_score(trace_stats: Dict[str, float], live_stats: Dict[str, float]) -> float:
    mean_gap = abs(trace_stats["mean"] - live_stats["mean"]) / (trace_stats["std"] + live_stats["std"] + 1.0e-6)
    std_gap = abs(math.log((trace_stats["std"] + 1.0e-6) / (live_stats["std"] + 1.0e-6)))
    return float(mean_gap + std_gap)


def _component_series(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[float]]:
    out = {field: [] for field in DELAY_COMPONENT_FIELDS}
    for row in rows:
        for field in DELAY_COMPONENT_FIELDS:
            camel = "".join(part.capitalize() if idx > 0 else part for idx, part in enumerate(field.split("_")))
            out[field].append(_to_float(row.get(field, row.get(camel)), 0.0))
    return out


def _flatten_dense_trace(trace_path: str, *, n_leo: int, num_rows: int) -> List[Dict[str, Any]]:
    trace_groups = load_trace_groups(trace_path, n_leo=n_leo, num_states=max(1, num_rows // max(1, n_leo)))
    trace_rows: List[Dict[str, Any]] = []
    for group in trace_groups:
        trace_rows.extend(group[:n_leo])
        if len(trace_rows) >= num_rows:
            break
    return trace_rows[:num_rows]


def _collect_live_rows(*, states: List[Dict[str, Any]], trace_mode: str, n_leo: int, num_rows: int) -> List[Dict[str, Any]]:
    live_rows: List[Dict[str, Any]] = []
    for state in states:
        if trace_mode == "sequential_live":
            live_rows.append(current_dense_row_from_state(state))
        else:
            live_rows.extend(dense_rows_from_state(state, n_leo=n_leo))
        if len(live_rows) >= num_rows:
            break
    return live_rows[:num_rows]


def _infer_causes(field_stats: Dict[str, Any], *, trace_mode: str) -> List[str]:
    causes: List[str] = []
    local_total = field_stats.get("local_total_delay", {})
    local_queue = field_stats.get("local_queue_delay", {})
    local_compute = field_stats.get("local_compute_delay", {})
    ground_total = field_stats.get("ground_total_delay", {})
    ground_tx = field_stats.get("ground_tx_delay", {})
    geo_total = field_stats.get("geo_total_delay", {})
    geo_prop = field_stats.get("geo_prop_delay", {})

    if local_total and local_total["distribution_shift_score"] > 2.0:
        if local_queue and local_queue["distribution_shift_score"] > 2.0:
            causes.append("local_delay_shift_driven_by_queue_delay")
        elif local_compute and local_compute["distribution_shift_score"] > 2.0:
            causes.append("local_delay_shift_driven_by_compute_delay")
        else:
            causes.append("local_delay_shift_requires_formula_audit")
    if ground_total and ground_total["distribution_shift_score"] > 2.0:
        if ground_tx and ground_tx["distribution_shift_score"] > 2.0:
            causes.append("ground_delay_shift_driven_by_tx_delay")
        else:
            causes.append("ground_delay_shift_requires_formula_audit")
    if geo_total and geo_total["distribution_shift_score"] > 2.0:
        if geo_prop and geo_prop["distribution_shift_score"] > 2.0:
            causes.append("geo_delay_shift_driven_by_propagation_delay")
        else:
            causes.append("geo_delay_shift_requires_estimator_audit")
    if trace_mode == "sequential_live":
        if local_queue and local_queue["distribution_shift_score"] > 2.0:
            causes.append("sequential_backlog_queue_evolution_may_contribute")
        elif local_total and local_total["distribution_shift_score"] > 2.0:
            causes.append("reset_strategy_or_backlog_warmup_may_contribute")
        elif causes:
            causes.append("residual_difference_requires_estimator_audit")
    if trace_mode == "dense_projection" and causes:
        causes.append("dense_projection_vs_sequential_live_may_contribute")
    return causes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trace and live observation distributions, including delay decomposition.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--trace-mode", type=str, default="dense_projection", choices=["dense_projection", "sequential_live"])
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="balanced_four_tier")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--num-states", type=int, default=2000)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.trace_mode == "sequential_live":
        trace_rows = load_trace_rows(args.trace, num_rows=args.num_states)
        live_state_count = args.num_states
    else:
        trace_rows = _flatten_dense_trace(args.trace, n_leo=args.n_leo, num_rows=args.num_states)
        live_state_count = max(1, args.num_states // max(1, args.n_leo)) + 1

    live_states = collect_live_states(
        base_url=args.base_url,
        scenario_profile=args.scenario_profile,
        task_source_mode=args.task_source_mode,
        num_states=live_state_count,
        request_timeout=args.request_timeout,
    )
    live_rows = _collect_live_rows(states=live_states, trace_mode=args.trace_mode, n_leo=args.n_leo, num_rows=args.num_states)

    trace_series = raw_field_series(trace_rows)
    live_series = raw_field_series(live_rows)
    trace_component_series = _component_series(trace_rows)
    live_component_series = _component_series(live_rows)

    field_stats: Dict[str, Any] = {}
    warnings: List[str] = []
    max_shift = 0.0
    for field in FIELD_NAMES:
        trace_stats = summarize_field_stats(trace_series[field])
        live_stats = summarize_field_stats(live_series[field])
        shift_score = _shift_score(trace_stats, live_stats)
        max_shift = max(max_shift, shift_score)
        field_stats[field] = {"trace": trace_stats, "live": live_stats, "distribution_shift_score": shift_score}
        if shift_score > 2.0:
            warnings.append(f"distribution_shift_{field}")
        if trace_stats["p95"] > 0.0 and live_stats["p95"] > 10.0 * trace_stats["p95"]:
            warnings.append(f"unit_mismatch_suspected_{field}")
        if trace_stats["max"] == 0.0 and live_stats["max"] == 0.0:
            warnings.append(f"all_zero_{field}")
        if trace_stats["std"] <= 1.0e-9 and live_stats["std"] <= 1.0e-9:
            warnings.append(f"nearly_constant_{field}")

    for field in DELAY_COMPONENT_FIELDS:
        trace_stats = summarize_field_stats(trace_component_series[field])
        live_stats = summarize_field_stats(live_component_series[field])
        shift_score = _shift_score(trace_stats, live_stats)
        max_shift = max(max_shift, shift_score)
        field_stats[field] = {"trace": trace_stats, "live": live_stats, "distribution_shift_score": shift_score}
        if field.endswith("_total_delay") and shift_score > 2.0:
            warnings.append(f"distribution_shift_{field}")
        if trace_stats["p95"] > 0.0 and live_stats["p95"] > 10.0 * trace_stats["p95"]:
            warnings.append(f"unit_mismatch_suspected_{field}")
        if trace_stats["max"] == 0.0 and live_stats["max"] == 0.0:
            warnings.append(f"all_zero_{field}")
        if trace_stats["std"] <= 1.0e-9 and live_stats["std"] <= 1.0e-9:
            warnings.append(f"nearly_constant_{field}")

    if max_shift > 2.0:
        warnings.append("train_replay_distribution_shift")

    causes = _infer_causes(field_stats, trace_mode=args.trace_mode)
    payload = {
        "status": "TRACE_LIVE_OBS_COMPARE_OK",
        "trace_mode": args.trace_mode,
        "num_trace_rows": len(trace_rows),
        "num_live_rows": len(live_rows),
        "field_stats": field_stats,
        "warnings": sorted(set(warnings)),
        "suspected_causes": causes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
