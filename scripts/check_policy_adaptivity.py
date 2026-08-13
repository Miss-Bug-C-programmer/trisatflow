#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

ACTION_NAMES = ("local", "neighbor", "geo", "ground")
ACTION_COLS = {
    "local": "upper_local_ratio",
    "neighbor": "upper_neighbor_ratio",
    "geo": "upper_geo_ratio",
    "ground": "upper_ground_ratio",
}
EVAL_COLS = {
    "local": "eval_argmax_local_ratio",
    "neighbor": "eval_argmax_neighbor_ratio",
    "geo": "eval_argmax_geo_ratio",
    "ground": "eval_argmax_ground_ratio",
}
MASK_KEY = "abstract_action_mask_final"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NA"):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tail_mean(rows: List[Dict[str, str]], key: str, tail_window: int, default: float = 0.0) -> float:
    tail = rows[-max(1, tail_window) :]
    if not tail:
        return default
    return mean(_to_float(row.get(key), default) for row in tail)


def _distribution_from_metrics(rows: List[Dict[str, str]], columns: Dict[str, str], tail_window: int) -> Dict[str, float]:
    raw = {name: _tail_mean(rows, col, tail_window) for name, col in columns.items()}
    total = sum(max(0.0, value) for value in raw.values())
    if total <= 0.0:
        return {name: 0.0 for name in columns}
    return {name: max(0.0, value) / total for name, value in raw.items()}


def _entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log(p, 2) for p in dist.values() if p > 0.0)


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    midpoint = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}

    def kl(lhs: Dict[str, float], rhs: Dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            a = lhs.get(key, 0.0)
            b = rhs.get(key, 0.0)
            if a > 0.0 and b > 0.0:
                total += a * math.log(a / b, 2)
        return total

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _mask(raw: Any) -> List[int]:
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return [1 if bool(raw[i]) else 0 for i in range(4)]
    return [0, 0, 0, 0]


def _read_jsonl(path: Path, max_rows: int) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows > 0 and idx >= max_rows:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _trace_files(root: Path) -> List[Path]:
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def _trace_summary(trace_root: Path, semantic_class: str, max_rows_per_trace: int) -> Dict[str, Any]:
    phase_counts: Counter[str] = Counter()
    phase_masks: Dict[str, List[List[int]]] = defaultdict(list)
    feasible_counts = Counter()
    total_rows = 0
    semantic_counts = Counter()
    legality_denominator = 0
    legality_pruned = 0
    transitions = 0
    previous_by_leo: Dict[tuple[str, int], tuple[int, List[int]]] = {}

    for path in _trace_files(trace_root):
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_semantic = ""
        if manifest_path.is_file():
            try:
                manifest_semantic = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("trace_semantic_class", ""))
            except json.JSONDecodeError:
                manifest_semantic = ""
        if manifest_semantic and manifest_semantic != semantic_class:
            continue
        for row in _read_jsonl(path, max_rows_per_trace):
            row_semantic = str(row.get("trace_semantic_class", ""))
            semantic_counts[row_semantic] += 1
            if row_semantic != semantic_class:
                continue
            total_rows += 1
            phase = str(row.get("phase_id") or row.get("scenario_phase") or row.get("scenarioPhase") or "default_phase")
            mask = _mask(row.get(MASK_KEY, row.get("abstract_action_mask", [])))
            phase_counts[phase] += 1
            phase_masks[phase].append(mask)
            for idx, name in enumerate(ACTION_NAMES):
                feasible_counts[name] += int(mask[idx])
            legal = sum(mask)
            legality_denominator += 1
            legality_pruned += int(legal < 4)
            leo = int(_to_float(row.get("leo_id"), 0.0))
            step = int(_to_float(row.get("step"), 0.0))
            key = (str(path), leo)
            prev = previous_by_leo.get(key)
            if prev is not None and step >= prev[0] and mask != prev[1]:
                transitions += 1
            previous_by_leo[key] = (step, mask)

    if total_rows <= 0:
        raise SystemExit(f"POLICY_ADAPTIVITY_FAILED reason=no_rows_for_semantic_class:{semantic_class}")
    feasible_ratio = {name: feasible_counts[name] / total_rows for name in ACTION_NAMES}
    phase_feasible_distribution: Dict[str, Dict[str, float]] = {}
    for phase, masks in phase_masks.items():
        counts = [sum(mask[idx] for mask in masks) for idx in range(4)]
        total = sum(counts)
        if total <= 0:
            phase_feasible_distribution[phase] = {name: 0.0 for name in ACTION_NAMES}
        else:
            phase_feasible_distribution[phase] = {name: counts[idx] / total for idx, name in enumerate(ACTION_NAMES)}
    phase_js_values = []
    phases = sorted(phase_feasible_distribution)
    for i, left in enumerate(phases):
        for right in phases[i + 1 :]:
            phase_js_values.append(_js_divergence(phase_feasible_distribution[left], phase_feasible_distribution[right]))
    return {
        "num_rows": total_rows,
        "semantic_counts": dict(semantic_counts),
        "phase_counts": dict(phase_counts),
        "phase_count": len(phase_counts),
        "feasible_ratio": feasible_ratio,
        "phase_feasible_distribution": phase_feasible_distribution,
        "phase_feasible_js_mean": mean(phase_js_values) if phase_js_values else 0.0,
        "phase_feasible_js_max": max(phase_js_values or [0.0]),
        "topology_transition_count": transitions,
        "mask_pruned_ratio": legality_pruned / max(1, legality_denominator),
    }


def _phase_action_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "source": str(path),
            "available": False,
            "phase_counts": {},
            "phase_action_distribution": {},
            "phase_action_js_mean": 0.0,
            "phase_action_js_max": 0.0,
            "time_split_action_js": 0.0,
        }
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    phase_action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    first_half: Counter[str] = Counter()
    second_half: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        phase = str(row.get("scenario_phase") or "unknown_phase")
        action_raw = row.get("selected_action")
        try:
            action = ACTION_NAMES[int(action_raw)]
        except (TypeError, ValueError, IndexError):
            continue
        phase_action_counts[phase][action] += 1
        if idx < len(rows) / 2:
            first_half[action] += 1
        else:
            second_half[action] += 1

    def normalize(counter: Counter[str]) -> Dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {name: 0.0 for name in ACTION_NAMES}
        return {name: counter.get(name, 0) / total for name in ACTION_NAMES}

    phase_dist = {phase: normalize(counter) for phase, counter in phase_action_counts.items()}
    phase_js_values = []
    phases = sorted(phase_dist)
    for i, left in enumerate(phases):
        for right in phases[i + 1 :]:
            phase_js_values.append(_js_divergence(phase_dist[left], phase_dist[right]))
    return {
        "source": str(path),
        "available": True,
        "num_rows": len(rows),
        "phase_counts": {phase: sum(counter.values()) for phase, counter in phase_action_counts.items()},
        "phase_action_distribution": phase_dist,
        "phase_action_js_mean": mean(phase_js_values) if phase_js_values else 0.0,
        "phase_action_js_max": max(phase_js_values or [0.0]),
        "time_split_action_js": _js_divergence(normalize(first_half), normalize(second_half)),
    }


def _selected_when_visible(rows: List[Dict[str, str]], tail_window: int) -> Dict[str, float]:
    return {
        "neighbor": _tail_mean(rows, "neighbor_selected_when_visible_ratio", tail_window),
        "geo": _tail_mean(rows, "geo_selected_when_visible_ratio", tail_window),
        "ground": _tail_mean(rows, "ground_selected_when_visible_ratio", tail_window),
        "remote": _tail_mean(rows, "remote_selected_when_visible_ratio", tail_window),
    }


def _selected_tier_count(dist: Dict[str, float], min_ratio: float = 0.01) -> int:
    return sum(1 for value in dist.values() if value >= min_ratio)


def _remote_congested_response(trace: Dict[str, Any], selected: Dict[str, float]) -> float | None:
    phase_dist = trace["phase_feasible_distribution"]
    congested = [
        name
        for name in phase_dist
        if "remote_congested" in name or "remote_pressure" in name or "ground_congested" in name
    ]
    if not congested:
        return None
    remote_selected = selected["neighbor"] + selected["geo"] + selected["ground"]
    remote_feasible = mean(
        sum(phase_dist[name].get(tier, 0.0) for tier in ("neighbor", "geo", "ground"))
        for name in congested
    )
    return max(0.0, remote_feasible - remote_selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit policy action diversity and topology/phase adaptivity.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--trace-semantic-class", required=True)
    parser.add_argument("--tail-window", type=int, default=20)
    parser.add_argument("--fail-on-deterministic-dominance", action="store_true")
    parser.add_argument("--max-single-action-ratio", type=float, default=0.98)
    parser.add_argument("--min-phase-action-divergence", type=float, default=0.0)
    parser.add_argument("--min-selected-tier-count", type=int, default=2)
    parser.add_argument("--min-feasible-response-ratio", type=float, default=0.02)
    parser.add_argument("--max-rows-per-trace", type=int, default=5000)
    parser.add_argument("--rollout-debug", default="", help="Optional rollout_debug.csv with scenario_phase and selected_action.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    trace_root = Path(args.trace_root)
    rows = _read_metrics(metrics_path)
    if not rows:
        raise SystemExit("POLICY_ADAPTIVITY_FAILED reason=empty_metrics")
    trace = _trace_summary(trace_root, args.trace_semantic_class, args.max_rows_per_trace)
    rollout_debug = Path(args.rollout_debug) if args.rollout_debug else metrics_path.parent / "rollout_debug.csv"
    phase_actions = _phase_action_summary(rollout_debug)
    selected = _distribution_from_metrics(rows, ACTION_COLS, args.tail_window)
    deterministic = _distribution_from_metrics(rows, EVAL_COLS, args.tail_window)
    selected_when_visible = _selected_when_visible(rows, args.tail_window)
    mean_feasibility = _tail_mean(rows, "mean_feasibility", args.tail_window, default=1.0)
    invalid_action_ratio = _tail_mean(rows, "invalid_action_ratio", args.tail_window, default=0.0)
    eval_policy_entropy = _tail_mean(rows, "eval_policy_entropy", args.tail_window, default=0.0)
    trace_hit_ratio = _tail_mean(rows, "trace_hit_ratio", args.tail_window, default=1.0)
    trace_fallback_count = _tail_mean(rows, "trace_fallback_count", args.tail_window, default=0.0)

    dominant_action, dominant_ratio = max(selected.items(), key=lambda item: item[1])
    det_action, det_ratio = max(deterministic.items(), key=lambda item: item[1])
    selected_count = _selected_tier_count(selected)
    feasible_response = 0.0
    visible_remote = max(
        trace["feasible_ratio"]["neighbor"],
        trace["feasible_ratio"]["geo"],
        trace["feasible_ratio"]["ground"],
    )
    if visible_remote > 0.0:
        feasible_response = max(
            selected_when_visible["neighbor"],
            selected_when_visible["geo"],
            selected_when_visible["ground"],
            selected["neighbor"],
            selected["geo"],
            selected["ground"],
        )

    failures: List[str] = []
    warnings: List[str] = []
    semantic = args.trace_semantic_class
    is_controlled = semantic.startswith("controlled_stress")
    is_actual = semantic.startswith("actual_physical")

    if trace_hit_ratio < 0.999 or trace_fallback_count > 0.0:
        failures.append("trace_not_strict")
    if mean_feasibility < 0.85:
        failures.append(f"mean_feasibility={mean_feasibility:.6f}<0.85")
    if invalid_action_ratio > 0.0:
        failures.append(f"selected_action_legality_violation:invalid_action_ratio={invalid_action_ratio:.6f}")
    if args.fail_on_deterministic_dominance and dominant_ratio > args.max_single_action_ratio:
        failures.append(f"single_action_dominance:{dominant_action}:{dominant_ratio:.6f}")
    if args.fail_on_deterministic_dominance and det_ratio > args.max_single_action_ratio:
        if eval_policy_entropy <= 0.25:
            failures.append(f"deterministic_single_action_dominance:{det_action}:{det_ratio:.6f}")
        else:
            warnings.append(
                f"argmax_single_action_dominance_high_entropy:{det_action}:{det_ratio:.6f}:entropy={eval_policy_entropy:.6f}"
            )
    if selected_count < args.min_selected_tier_count:
        failures.append(f"selected_tier_count={selected_count}<min_selected_tier_count={args.min_selected_tier_count}")
    if visible_remote >= args.min_feasible_response_ratio and feasible_response < args.min_feasible_response_ratio:
        failures.append(
            f"feasible_response_ratio={feasible_response:.6f}<min_feasible_response_ratio={args.min_feasible_response_ratio}"
        )

    if is_actual:
        if trace["topology_transition_count"] <= 0:
            failures.append("actual_trace_missing_topology_transition")
        if phase_actions["available"] and trace["topology_transition_count"] > 0 and phase_actions["time_split_action_js"] <= 0.0:
            warnings.append("actual_topology_transition_without_time_split_action_change")
        if trace["mask_pruned_ratio"] <= 0.0:
            warnings.append("actual_trace_has_no_mask_prune")
    elif is_controlled:
        if trace["phase_count"] < 2:
            failures.append("controlled_stress_missing_phase_diversity")
        if not phase_actions["available"]:
            failures.append("missing_rollout_debug_phase_action_distribution")
        phase_divergence = (
            phase_actions["phase_action_js_max"]
            if phase_actions["available"]
            else trace["phase_feasible_js_max"]
        )
        if phase_divergence < args.min_phase_action_divergence:
            failures.append(
                f"phase_action_divergence={phase_divergence:.6f}<min_phase_action_divergence={args.min_phase_action_divergence}"
            )
        congested_response = _remote_congested_response(trace, selected)
        if congested_response is not None and congested_response <= 0.0:
            warnings.append("remote_congested_phase_no_global_remote_reduction")
    else:
        failures.append(f"unsupported_trace_semantic_class={semantic}")

    payload = {
        "status": "POLICY_ADAPTIVITY_OK" if not failures else "POLICY_ADAPTIVITY_FAILED",
        "metrics": str(metrics_path),
        "trace_root": str(trace_root),
        "trace_semantic_class": semantic,
        "tail_window": args.tail_window,
        "action_distribution": selected,
        "deterministic_action_distribution": deterministic,
        "selected_when_visible": selected_when_visible,
        "conditional_action_entropy": _entropy(selected),
        "deterministic_action_entropy": _entropy(deterministic),
        "eval_policy_entropy": eval_policy_entropy,
        "dominant_action": dominant_action,
        "dominant_action_ratio": dominant_ratio,
        "deterministic_dominant_action": det_action,
        "deterministic_dominant_action_ratio": det_ratio,
        "selected_tier_count": selected_count,
        "feasible_response_ratio": feasible_response,
        "mean_feasibility": mean_feasibility,
        "invalid_action_ratio": invalid_action_ratio,
        "trace_hit_ratio": trace_hit_ratio,
        "trace_fallback_count": trace_fallback_count,
        "trace": trace,
        "phase_action": phase_actions,
        "warnings": warnings,
        "failures": failures,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
