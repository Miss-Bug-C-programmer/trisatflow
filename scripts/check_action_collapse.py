from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ACTION_COLUMNS = [
    "upper_local_ratio",
    "upper_neighbor_ratio",
    "upper_geo_ratio",
    "upper_ground_ratio",
]
ACTION_NAMES = ("local", "neighbor", "geo", "ground")
EVAL_COLUMNS = [
    "eval_argmax_local_ratio",
    "eval_argmax_neighbor_ratio",
    "eval_argmax_geo_ratio",
    "eval_argmax_ground_ratio",
]
REMOTE_COLUMNS = ACTION_COLUMNS[1:]


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a TriSatFlow training run collapsed to one upper action.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--tail-window", type=int, default=20)
    parser.add_argument("--max-local-ratio", type=float, default=0.95)
    parser.add_argument("--min-remote-ratio", type=float, default=0.05)
    parser.add_argument("--min-feasibility", type=float, default=0.90)
    parser.add_argument("--trace-semantic-class", default="", help="Trace semantic label for reporting.")
    parser.add_argument("--fail-on-deterministic-dominance", action="store_true")
    parser.add_argument("--max-single-action-ratio", type=float, default=0.98)
    parser.add_argument("--min-phase-action-divergence", type=float, default=0.0)
    parser.add_argument("--min-selected-tier-count", type=int, default=1)
    parser.add_argument("--min-feasible-response-ratio", type=float, default=0.0)
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("ACTION_COLLAPSE_FAILED reason=empty_metrics")

    tail = rows[-max(1, args.tail_window) :]
    local = mean(_to_float(row.get("upper_local_ratio")) for row in tail)
    remote = mean(sum(_to_float(row.get(col)) for col in REMOTE_COLUMNS) for row in tail)
    feasibility = mean(_to_float(row.get("mean_feasibility"), 1.0) for row in tail)
    system_cost = mean(_to_float(row.get("mean_system_cost")) for row in tail)
    ratios = {col: mean(_to_float(row.get(col)) for row in tail) for col in ACTION_COLUMNS}
    action_ratios = dict(zip(ACTION_NAMES, (ratios[col] for col in ACTION_COLUMNS)))
    eval_ratios = dict(
        zip(
            ACTION_NAMES,
            (mean(_to_float(row.get(col)) for row in tail) for col in EVAL_COLUMNS),
        )
    )
    selected_tier_count = sum(1 for value in action_ratios.values() if value >= 0.01)
    dominant_action, dominant_ratio = max(action_ratios.items(), key=lambda item: item[1])
    det_action, det_ratio = max(eval_ratios.items(), key=lambda item: item[1])
    eval_policy_entropy = mean(_to_float(row.get("eval_policy_entropy")) for row in tail)
    visible_remote = max(
        mean(_to_float(row.get("neighbor_visible_ratio")) for row in tail),
        mean(_to_float(row.get("geo_visible_ratio")) for row in tail),
        mean(_to_float(row.get("ground_visible_ratio")) for row in tail),
    )
    feasible_response = max(
        mean(_to_float(row.get("neighbor_selected_when_visible_ratio")) for row in tail),
        mean(_to_float(row.get("geo_selected_when_visible_ratio")) for row in tail),
        mean(_to_float(row.get("ground_selected_when_visible_ratio")) for row in tail),
        action_ratios["neighbor"],
        action_ratios["geo"],
        action_ratios["ground"],
    )

    print("tail_window", len(tail))
    print("mean_system_cost", f"{system_cost:.6f}")
    print("mean_feasibility", f"{feasibility:.6f}")
    for col in ACTION_COLUMNS:
        print(col, f"{ratios[col]:.6f}")
    print("upper_remote_ratio", f"{remote:.6f}")
    print("dominant_action", dominant_action, f"{dominant_ratio:.6f}")
    print("deterministic_dominant_action", det_action, f"{det_ratio:.6f}")
    print("eval_policy_entropy", f"{eval_policy_entropy:.6f}")
    print("selected_tier_count", selected_tier_count)
    print("feasible_response_ratio", f"{feasible_response:.6f}")

    failed = []
    if local > args.max_local_ratio:
        failed.append(f"local_ratio={local:.6f}>max_local_ratio={args.max_local_ratio}")
    if remote < args.min_remote_ratio:
        failed.append(f"remote_ratio={remote:.6f}<min_remote_ratio={args.min_remote_ratio}")
    if feasibility < args.min_feasibility:
        failed.append(f"feasibility={feasibility:.6f}<min_feasibility={args.min_feasibility}")
    if args.fail_on_deterministic_dominance and dominant_ratio > args.max_single_action_ratio:
        failed.append(f"single_action_dominance:{dominant_action}:{dominant_ratio:.6f}")
    if args.fail_on_deterministic_dominance and det_ratio > args.max_single_action_ratio:
        if eval_policy_entropy <= 0.25:
            failed.append(f"deterministic_single_action_dominance:{det_action}:{det_ratio:.6f}")
        else:
            print(
                "warning",
                f"argmax_single_action_dominance_high_entropy:{det_action}:{det_ratio:.6f}:entropy={eval_policy_entropy:.6f}",
            )
    if selected_tier_count < args.min_selected_tier_count:
        failed.append(f"selected_tier_count={selected_tier_count}<min_selected_tier_count={args.min_selected_tier_count}")
    if visible_remote >= args.min_feasible_response_ratio and feasible_response < args.min_feasible_response_ratio:
        failed.append(
            f"feasible_response_ratio={feasible_response:.6f}<min_feasible_response_ratio={args.min_feasible_response_ratio}"
        )

    if failed:
        raise SystemExit("ACTION_COLLAPSE_FAILED reason=" + ";".join(failed))
    print(
        json.dumps(
            {
                "status": "ACTION_DIVERSITY_OK",
                "trace_semantic_class": args.trace_semantic_class,
                "action_ratio": action_ratios,
                "deterministic_action_ratio": eval_ratios,
                "dominant_action": dominant_action,
                "dominant_action_ratio": dominant_ratio,
                "deterministic_dominant_action": det_action,
                "deterministic_dominant_action_ratio": det_ratio,
                "eval_policy_entropy": eval_policy_entropy,
                "selected_tier_count": selected_tier_count,
                "feasible_response_ratio": feasible_response,
                "min_phase_action_divergence": args.min_phase_action_divergence,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
