from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tail_mean(rows: List[Dict[str, str]], key: str, tail_window: int) -> float:
    tail = rows[-tail_window:] if tail_window > 0 else rows
    if not tail:
        return 0.0
    return sum(_to_float(row.get(key)) for row in tail) / len(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check four-tier policy health on balanced strict-trace runs.")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--tail-window", type=int, default=5)
    parser.add_argument("--min-remote-ratio", type=float, default=0.05)
    parser.add_argument("--min-feasibility", type=float, default=0.85)
    parser.add_argument("--min-geo-visible-ratio", type=float, default=0.05)
    parser.add_argument("--min-ground-visible-ratio", type=float, default=0.05)
    parser.add_argument("--min-neighbor-visible-ratio", type=float, default=0.05)
    parser.add_argument("--trace-semantic-class", default="", help="Trace semantic label for reporting.")
    parser.add_argument("--fail-on-deterministic-dominance", action="store_true")
    parser.add_argument("--max-single-action-ratio", type=float, default=0.98)
    parser.add_argument("--min-phase-action-divergence", type=float, default=0.0)
    parser.add_argument("--min-selected-tier-count", type=int, default=1)
    parser.add_argument("--min-feasible-response-ratio", type=float, default=0.0)
    args = parser.parse_args()

    rows = _read_rows(Path(args.metrics))
    if not rows:
        raise SystemExit("metrics.csv is empty")

    visible = {
        "neighbor": _tail_mean(rows, "neighbor_visible_ratio", args.tail_window),
        "geo": _tail_mean(rows, "geo_visible_ratio", args.tail_window),
        "ground": _tail_mean(rows, "ground_visible_ratio", args.tail_window),
    }
    selected_when_visible = {
        "neighbor": _tail_mean(rows, "neighbor_selected_when_visible_ratio", args.tail_window),
        "geo": _tail_mean(rows, "geo_selected_when_visible_ratio", args.tail_window),
        "ground": _tail_mean(rows, "ground_selected_when_visible_ratio", args.tail_window),
        "remote": _tail_mean(rows, "remote_selected_when_visible_ratio", args.tail_window),
    }
    actions = {
        "local": _tail_mean(rows, "upper_local_ratio", args.tail_window),
        "neighbor": _tail_mean(rows, "upper_neighbor_ratio", args.tail_window),
        "geo": _tail_mean(rows, "upper_geo_ratio", args.tail_window),
        "ground": _tail_mean(rows, "upper_ground_ratio", args.tail_window),
        "remote": _tail_mean(rows, "upper_remote_ratio", args.tail_window),
    }
    mean_feasibility = _tail_mean(rows, "mean_feasibility", args.tail_window)
    trace_hit_ratio = _tail_mean(rows, "trace_hit_ratio", args.tail_window)
    trace_fallback_count = _tail_mean(rows, "trace_fallback_count", args.tail_window)
    remote_available_ratio = _tail_mean(rows, "remote_available_ratio", args.tail_window)
    eval_policy_entropy = _tail_mean(rows, "eval_policy_entropy", args.tail_window)

    warnings: List[str] = []
    failures: List[str] = []

    if visible["neighbor"] < args.min_neighbor_visible_ratio:
        warnings.append("scene_coverage_insufficient_neighbor")
    if visible["geo"] < args.min_geo_visible_ratio:
        warnings.append("scene_coverage_insufficient_geo")
    if visible["ground"] < args.min_ground_visible_ratio:
        warnings.append("scene_coverage_insufficient_ground")

    for tier in ("neighbor", "geo", "ground"):
        if visible[tier] >= 0.10 and selected_when_visible[tier] < 0.02:
            warnings.append(f"policy_avoids_available_tier_{tier}")

    dominant = max(actions.items(), key=lambda item: item[1])
    if dominant[1] > args.max_single_action_ratio and dominant[0] in {"local", "neighbor", "geo", "ground"}:
        marker = f"single_action_dominance_{dominant[0]}"
        if args.fail_on_deterministic_dominance:
            failures.append(marker)
        else:
            warnings.append(marker)
    eval_actions = {
        "local": _tail_mean(rows, "eval_argmax_local_ratio", args.tail_window),
        "neighbor": _tail_mean(rows, "eval_argmax_neighbor_ratio", args.tail_window),
        "geo": _tail_mean(rows, "eval_argmax_geo_ratio", args.tail_window),
        "ground": _tail_mean(rows, "eval_argmax_ground_ratio", args.tail_window),
    }
    eval_dominant = max(eval_actions.items(), key=lambda item: item[1])
    if eval_dominant[1] > args.max_single_action_ratio:
        marker = f"deterministic_policy_single_action_dominance_{eval_dominant[0]}"
        if args.fail_on_deterministic_dominance and eval_policy_entropy <= 0.25:
            failures.append(marker)
        elif args.fail_on_deterministic_dominance:
            warnings.append(f"argmax_single_action_dominance_high_entropy_{eval_dominant[0]}")
        else:
            warnings.append(marker)

    selected_tier_count = sum(1 for tier in ("local", "neighbor", "geo", "ground") if actions[tier] >= 0.01)
    feasible_response = max(
        selected_when_visible["neighbor"],
        selected_when_visible["geo"],
        selected_when_visible["ground"],
        actions["neighbor"],
        actions["geo"],
        actions["ground"],
    )

    if mean_feasibility < args.min_feasibility:
        failures.append("feasibility_violation")
    if trace_hit_ratio < 0.999 or trace_fallback_count > 0.0:
        failures.append("trace_not_strict")
    if remote_available_ratio >= args.min_remote_ratio and actions["remote"] < args.min_remote_ratio:
        failures.append("local_collapse")
    if selected_tier_count < args.min_selected_tier_count:
        failures.append(f"selected_tier_count={selected_tier_count}<min_selected_tier_count={args.min_selected_tier_count}")
    if remote_available_ratio >= args.min_feasible_response_ratio and feasible_response < args.min_feasible_response_ratio:
        failures.append(
            f"feasible_response_ratio={feasible_response:.6f}<min_feasible_response_ratio={args.min_feasible_response_ratio}"
        )

    status = "FOUR_TIER_POLICY_HEALTH_OK" if not failures else "FOUR_TIER_POLICY_HEALTH_FAILED"
    payload = {
        "status": status,
        "trace_semantic_class": args.trace_semantic_class,
        "tail_window": args.tail_window,
        "mean_feasibility": mean_feasibility,
        "trace_hit_ratio": trace_hit_ratio,
        "trace_fallback_count": trace_fallback_count,
        "remote_available_ratio": remote_available_ratio,
        "visible_ratio": visible,
        "selected_when_visible_ratio": selected_when_visible,
        "action_ratio": actions,
        "eval_argmax_ratio": eval_actions,
        "eval_policy_entropy": eval_policy_entropy,
        "selected_tier_count": selected_tier_count,
        "feasible_response_ratio": feasible_response,
        "min_phase_action_divergence": args.min_phase_action_divergence,
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
