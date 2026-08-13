from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.inspection import (
    collect_live_states,
    current_dense_row_from_state,
    dense_rows_from_state,
    load_trace_groups,
    load_trace_rows,
)

TIERS = list(ACTION_NAMES)
OVERALL_GATE = {
    "local": 0.03,
    "neighbor": 0.05,
    "geo": 0.05,
    "ground": 0.05,
    "max_ratio": 0.80,
}
PHASE_TARGETS = {
    "local_favorable_phase": "local",
    "neighbor_favorable_phase": "neighbor",
    "geo_favorable_phase": "geo",
    "ground_favorable_phase": "ground",
}
MIN_PHASE_DELTA = 0.02
MIN_PHASE_MULTIPLIER = 1.20
MAX_PHASE_DOMINANCE = 0.95


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.source == "trace":
        if args.n_leo > 1:
            groups = load_trace_groups(args.trace, n_leo=args.n_leo, num_states=max(1, args.num_states // max(1, args.n_leo)))
            rows: List[Dict[str, Any]] = []
            for group in groups:
                rows.extend(group[: args.n_leo])
                if len(rows) >= args.num_states:
                    break
            if rows:
                return rows[: args.num_states]
        return load_trace_rows(args.trace, num_rows=args.num_states)

    live_states = collect_live_states(
        base_url=args.base_url,
        scenario_profile=args.scenario_profile,
        task_source_mode=args.task_source_mode,
        num_states=max(1, args.num_states // max(1, args.n_leo)) + 1 if args.n_leo > 1 else args.num_states,
        request_timeout=args.request_timeout,
    )
    rows: List[Dict[str, Any]] = []
    for state in live_states:
        state_rows = dense_rows_from_state(state)
        if state_rows:
            rows.extend(state_rows[: args.n_leo] if args.n_leo > 1 else [current_dense_row_from_state(state)])
        else:
            rows.append(current_dense_row_from_state(state))
        if len(rows) >= args.num_states:
            break
    return rows[: args.num_states]


def _visible(row: Mapping[str, Any], tier: str) -> bool:
    return _to_bool(row.get(f"{tier}_visible", row.get(f"{tier}Visible")), tier == "local")


def _component(row: Mapping[str, Any], tier: str, component: str) -> float:
    snake = f"{tier}_{component}"
    camel = tier + "".join(part.capitalize() for part in component.split("_"))
    return max(0.0, _to_float(row.get(snake, row.get(camel)), 0.0))


def _total_delay(row: Mapping[str, Any], tier: str) -> float:
    explicit = row.get(f"{tier}_total_delay", row.get(f"{tier}TotalDelay"))
    if explicit not in (None, ""):
        return max(0.0, _to_float(explicit, 0.0))
    return (
        _component(row, tier, "prop_delay")
        + _component(row, tier, "tx_delay")
        + _component(row, tier, "compute_delay")
        + _component(row, tier, "queue_delay")
    )


def _ratio(counter: Counter[int], total: int, idx: int) -> float:
    return float(counter.get(idx, 0) / max(1, total))


def _distribution(counter: Counter[int], total: int) -> Dict[str, float]:
    return {f"oracle_{tier}_ratio": _ratio(counter, total, idx) for idx, tier in enumerate(TIERS)}


def _group_distribution(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Counter[int]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for record in records:
        key = str(record.get(field) or "unknown")
        grouped[key][int(record["oracle_action"])] += 1
        counts[key] += 1
    payload: Dict[str, Dict[str, float]] = {}
    for key, counter in sorted(grouped.items()):
        payload[key] = _distribution(counter, counts[key])
        payload[key]["num_rows"] = int(counts[key])
    return payload


def _cost_component_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tier in TIERS:
        props = [_component(record["row"], tier, "prop_delay") for record in records if _visible(record["row"], tier)]
        txs = [_component(record["row"], tier, "tx_delay") for record in records if _visible(record["row"], tier)]
        computes = [_component(record["row"], tier, "compute_delay") for record in records if _visible(record["row"], tier)]
        queues = [_component(record["row"], tier, "queue_delay") for record in records if _visible(record["row"], tier)]
        totals = [_total_delay(record["row"], tier) for record in records if _visible(record["row"], tier)]
        out[tier] = {
            "prop_delay_mean": float(sum(props) / max(1, len(props))),
            "tx_delay_mean": float(sum(txs) / max(1, len(txs))),
            "compute_delay_mean": float(sum(computes) / max(1, len(computes))),
            "queue_delay_mean": float(sum(queues) / max(1, len(queues))),
            "total_delay_mean": float(sum(totals) / max(1, len(totals))),
        }
    return out


def _diagnosis(overall: Dict[str, float], component_summary: Dict[str, Dict[str, float]]) -> List[str]:
    diagnoses: List[str] = []
    dominant_tier = max(TIERS, key=lambda tier: overall[f"oracle_{tier}_ratio"])
    dominant_ratio = overall[f"oracle_{dominant_tier}_ratio"]
    if dominant_ratio > OVERALL_GATE["max_ratio"]:
        diagnoses.append(f"oracle_{dominant_tier}_dominant")
    for tier in TIERS:
        if overall[f"oracle_{tier}_ratio"] < OVERALL_GATE[tier]:
            diagnoses.append(f"insufficient_{tier}_opportunity")

    totals = sorted((values["total_delay_mean"], tier) for tier, values in component_summary.items())
    if len(totals) >= 2 and totals[0][0] < 0.60 * max(1.0e-9, totals[1][0]):
        diagnoses.append("cost_component_imbalance")
    for tier, values in component_summary.items():
        total = max(values["total_delay_mean"], 1.0e-9)
        if values["queue_delay_mean"] / total > 0.92 or values["compute_delay_mean"] / total > 0.92:
            diagnoses.append("cost_component_imbalance")
            break
    return sorted(set(diagnoses))


def _phase_gate(overall: Dict[str, float], phase_distribution: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for phase, target in PHASE_TARGETS.items():
        phase_stats = phase_distribution.get(phase)
        if not phase_stats:
            checks[phase] = {"status": "missing_phase"}
            failures.append(f"missing_phase={phase}")
            continue
        phase_ratio = float(phase_stats.get(f"oracle_{target}_ratio", 0.0))
        overall_ratio = float(overall.get(f"oracle_{target}_ratio", 0.0))
        boosted = phase_ratio >= overall_ratio + MIN_PHASE_DELTA and phase_ratio >= overall_ratio * MIN_PHASE_MULTIPLIER
        phase_max = max(float(phase_stats.get(f"oracle_{tier}_ratio", 0.0)) for tier in TIERS)
        not_degenerate = phase_max <= MAX_PHASE_DOMINANCE
        checks[phase] = {
            "target_tier": target,
            "phase_ratio": phase_ratio,
            "overall_ratio": overall_ratio,
            "boosted_vs_overall": boosted,
            "max_phase_action_ratio": phase_max,
            "not_degenerate": not_degenerate,
        }
        if not boosted:
            failures.append(f"phase_target_not_boosted={phase}")
        if not not_degenerate:
            failures.append(f"phase_degenerate={phase}")
    return {
        "status": "PHASE_ORACLE_SANITY_OK" if not failures else "PHASE_ORACLE_SANITY_FAILED",
        "checks": checks,
        "failures": failures,
    }


def _overall_gate(overall: Dict[str, float]) -> Dict[str, Any]:
    failures: List[str] = []
    for tier in TIERS:
        key = f"oracle_{tier}_ratio"
        if overall[key] < OVERALL_GATE[tier]:
            failures.append(f"{key}={overall[key]:.6f} < {OVERALL_GATE[tier]:.2f}")
    if overall["max_oracle_action_ratio"] > OVERALL_GATE["max_ratio"]:
        failures.append(
            f"max_oracle_action_ratio={overall['max_oracle_action_ratio']:.6f} > {OVERALL_GATE['max_ratio']:.2f}"
        )
    return {
        "status": "OVERALL_ORACLE_GATE_OK" if not failures else "OVERALL_ORACLE_GATE_FAILED",
        "failures": failures,
        "thresholds": {
            "oracle_local_ratio_min": OVERALL_GATE["local"],
            "oracle_neighbor_ratio_min": OVERALL_GATE["neighbor"],
            "oracle_geo_ratio_min": OVERALL_GATE["geo"],
            "oracle_ground_ratio_min": OVERALL_GATE["ground"],
            "max_oracle_action_ratio_max": OVERALL_GATE["max_ratio"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze phase-aware oracle tier selection from SatEdgeSim traces or live states.")
    parser.add_argument("--source", type=str, required=True, choices=["trace", "live"])
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="balanced_four_tier")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=1024)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    raw_rows = _collect_rows(args)
    records: List[Dict[str, Any]] = []
    oracle_counter: Counter[int] = Counter()
    for row in raw_rows:
        tier_costs = [math.inf, math.inf, math.inf, math.inf]
        for idx, tier in enumerate(TIERS):
            if _visible(row, tier):
                tier_costs[idx] = _total_delay(row, tier)
        oracle_action = min(range(4), key=lambda idx: tier_costs[idx])
        if not math.isfinite(tier_costs[oracle_action]):
            continue
        oracle_counter[oracle_action] += 1
        records.append(
            {
                "row": dict(row),
                "scenario_phase": str(row.get("scenario_phase", row.get("scenarioPhase", "unknown_phase"))),
                "task_type": str(row.get("task_type", row.get("taskType", "unknown_task"))),
                "oracle_action": int(oracle_action),
                "oracle_action_name": TIERS[oracle_action],
            }
        )

    overall = _distribution(oracle_counter, len(records))
    overall["max_oracle_action_ratio"] = max((overall[f"oracle_{tier}_ratio"] for tier in TIERS), default=0.0)
    phase_distribution = _group_distribution(records, "scenario_phase")
    task_type_distribution = _group_distribution(records, "task_type")
    component_summary = _cost_component_summary(records)
    diagnosis = _diagnosis(overall, component_summary)
    overall_gate = _overall_gate(overall)
    phase_gate = _phase_gate(overall, phase_distribution)

    payload: Dict[str, Any] = {
        "source": args.source,
        "num_rows": len(records),
        **overall,
        "phase_oracle_distribution": phase_distribution,
        "task_type_oracle_distribution": task_type_distribution,
        "oracle_cost_component_summary": component_summary,
        "failure_diagnosis": diagnosis,
        "overall_oracle_gate": overall_gate,
        "phase_oracle_sanity_gate": phase_gate,
        "status": "ORACLE_TRAINING_GATE_OK"
        if overall_gate["status"] == "OVERALL_ORACLE_GATE_OK" and phase_gate["status"] == "PHASE_ORACLE_SANITY_OK"
        else "ORACLE_TRAINING_GATE_FAILED",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
