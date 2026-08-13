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
    parser = argparse.ArgumentParser(description="Inspect deterministic eval action ratios from training metrics.csv.")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = _read_rows(Path(args.metrics))
    if not rows:
        raise SystemExit("metrics.csv is empty")

    sampled = {
        "local": _tail_mean(rows, "upper_local_ratio", args.tail_window),
        "neighbor": _tail_mean(rows, "upper_neighbor_ratio", args.tail_window),
        "geo": _tail_mean(rows, "upper_geo_ratio", args.tail_window),
        "ground": _tail_mean(rows, "upper_ground_ratio", args.tail_window),
        "remote": _tail_mean(rows, "upper_remote_ratio", args.tail_window),
    }
    argmax = {
        "local": _tail_mean(rows, "eval_argmax_local_ratio", args.tail_window),
        "neighbor": _tail_mean(rows, "eval_argmax_neighbor_ratio", args.tail_window),
        "geo": _tail_mean(rows, "eval_argmax_geo_ratio", args.tail_window),
        "ground": _tail_mean(rows, "eval_argmax_ground_ratio", args.tail_window),
        "remote": _tail_mean(rows, "eval_argmax_remote_ratio", args.tail_window),
    }
    entropy = _tail_mean(rows, "eval_policy_entropy", args.tail_window)
    eval_warning = str(rows[-1].get("eval_warning", "") or "")

    classification = "deterministic_policy_diverse"
    if argmax["ground"] > 0.98:
        classification = "deterministic_argmax_ground_collapse"
    elif argmax["geo"] > 0.98:
        classification = "deterministic_argmax_geo_collapse"
    elif argmax["local"] > 0.98:
        classification = "deterministic_argmax_local_collapse"
    elif argmax["neighbor"] > 0.98:
        classification = "deterministic_argmax_neighbor_collapse"

    sampled_max = max(sampled.values())
    argmax_max = max(argmax.values())
    sampled_diverse = sampled_max < 0.98
    argmax_single = argmax_max > 0.98
    if sampled_diverse and argmax_single:
        classification = "stochastic_diversity_only"

    sampled_dominant = max(sampled.items(), key=lambda item: item[1])
    argmax_dominant = max(argmax.items(), key=lambda item: item[1])
    stochastic_vs_argmax_gap = {
        key: sampled[key] - argmax.get(key, 0.0)
        for key in ("local", "neighbor", "geo", "ground", "remote")
    }

    payload = {
        "metrics": args.metrics,
        "tail_window": args.tail_window,
        "sampled_action_ratio": sampled,
        "eval_argmax_ratio": argmax,
        "eval_policy_entropy": entropy,
        "eval_warning": eval_warning,
        "sampled_dominant_action": sampled_dominant[0],
        "sampled_dominant_ratio": sampled_dominant[1],
        "argmax_dominant_action": argmax_dominant[0],
        "argmax_dominant_ratio": argmax_dominant[1],
        "stochastic_vs_argmax_gap": stochastic_vs_argmax_gap,
        "classification": classification,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
