from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

ACTIONS = ("local", "neighbor", "geo", "ground")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_metrics_last(metrics_path: Path) -> Dict[str, str]:
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return {}
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _get_nested(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return _to_float(cur, default)


def _fallback_none_ratio(summary: Dict[str, Any]) -> float:
    dist = summary.get("fallback_reason_distribution")
    if isinstance(dist, dict):
        return _to_float(dist.get("none"), 0.0)
    return 0.0


def _policy_executed_diff(summary: Dict[str, Any]) -> float:
    max_diff = 0.0
    for a in ACTIONS:
        p = _to_float(summary.get(f"final_policy_{a}_ratio"), 0.0)
        e = _to_float(summary.get(f"executed_{a}_ratio"), 0.0)
        max_diff = max(max_diff, abs(p - e))
    return max_diff


def _replay_mode_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    block = {
        "intent_execution_match_ratio": _to_float(summary.get("intent_execution_match_ratio"), 0.0),
        "fallback_none_ratio": _fallback_none_ratio(summary),
        "receipt_accept_ratio": _to_float(summary.get("receipt_accept_ratio"), 0.0),
        "http_timeout_count": int(_to_float(summary.get("http_timeout_count"), 0.0)),
        "http_connection_error_count": int(_to_float(summary.get("http_connection_error_count"), 0.0)),
        "policy_executed_max_abs_diff": _policy_executed_diff(summary),
    }
    block["pass"] = bool(
        block["intent_execution_match_ratio"] >= 0.99
        and block["fallback_none_ratio"] >= 0.99
        and block["receipt_accept_ratio"] >= 0.99
        and block["http_timeout_count"] == 0
        and block["http_connection_error_count"] == 0
        and block["policy_executed_max_abs_diff"] <= 0.01
    )
    return block


def _regret_overall(regret_payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    mode_results = regret_payload.get("mode_results") if isinstance(regret_payload, dict) else {}
    if not isinstance(mode_results, dict):
        mode_results = {}
    mode_block = mode_results.get(mode)
    if not isinstance(mode_block, dict):
        mode_block = {}
    overall = mode_block.get("overall")
    if not isinstance(overall, dict):
        overall = {}
    return overall


def _mode_dist(eval_modes_payload: Dict[str, Any], key: str) -> Dict[str, float]:
    d = eval_modes_payload.get(key)
    if not isinstance(d, dict):
        return {a: 0.0 for a in ACTIONS}
    return {a: _to_float(d.get(a), 0.0) for a in ACTIONS}


def _flatten_numeric(prefix: str, payload: Dict[str, Any], out: Dict[str, float]) -> None:
    for k, v in payload.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, (int, float)):
            vv = float(v)
            if not (math.isnan(vv) or math.isinf(vv)):
                out[key] = vv
        elif isinstance(v, dict):
            _flatten_numeric(f"{key}.", v, out)


def _agg(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed GPU preflight outputs.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="13,17,23")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    per_seed: Dict[str, Any] = {}
    numeric_rows: List[Dict[str, float]] = []
    readiness_counter = {
        "strong_pass": 0,
        "conditional_pass": 0,
        "calibration_required": 0,
        "fail": 0,
        "unknown": 0,
    }

    for seed in seeds:
        run_dir = run_root / f"seed_{seed}" / "upper_mappo__lower_maddpg"
        eval_dir = run_dir / "eval"

        metrics_last = _read_metrics_last(run_dir / "metrics.csv")
        state_diag = _load_json(eval_dir / "state_conditioned.json")
        eval_modes = _load_json(eval_dir / "eval_modes.json")
        regret = _load_json(eval_dir / "regret.json")
        readiness = _load_json(eval_dir / "readiness.json")

        raw_summary = _load_json(run_dir / "replay_raw_argmax" / "summary_compact.json")
        stochastic_summary = _load_json(run_dir / "replay_stochastic_eval" / "summary_compact.json")
        margin_summary = _load_json(run_dir / "replay_margin_cost_tiebreak" / "summary_compact.json")

        training_block = {
            "final_mean_system_cost": _to_float(metrics_last.get("mean_system_cost"), 0.0),
            "final_mean_delay": _to_float(metrics_last.get("mean_delay"), 0.0),
            "final_mean_queue": _to_float(metrics_last.get("mean_queue"), 0.0),
            "final_mean_feasibility": _to_float(metrics_last.get("mean_feasibility"), 0.0),
            "trace_hit_ratio": _to_float(metrics_last.get("trace_hit_ratio"), 0.0),
            "trace_fallback_count": _to_float(metrics_last.get("trace_fallback_count"), 0.0),
            "upper_local_ratio": _to_float(metrics_last.get("upper_local_ratio"), 0.0),
            "upper_neighbor_ratio": _to_float(metrics_last.get("upper_neighbor_ratio"), 0.0),
            "upper_geo_ratio": _to_float(metrics_last.get("upper_geo_ratio"), 0.0),
            "upper_ground_ratio": _to_float(metrics_last.get("upper_ground_ratio"), 0.0),
        }

        state_block = {
            "raw_argmax_oracle_agreement": _to_float(state_diag.get("raw_argmax_oracle_agreement"), 0.0),
            "mi_phase_argmax": _get_nested(state_diag, ["state_sensitivity", "mutual_information_phase_argmax"], 0.0),
            "MI(phase,argmax)": _get_nested(state_diag, ["state_sensitivity", "mutual_information_phase_argmax"], 0.0),
            "prob_oracle_action_mean": _to_float(state_diag.get("prob_oracle_action_mean"), 0.0),
            "raw_argmax_distribution": _mode_dist(state_diag, "raw_argmax_distribution"),
        }

        regret_raw = _regret_overall(regret, "raw_argmax")
        regret_stochastic = _regret_overall(regret, "stochastic_eval")
        regret_margin = _regret_overall(regret, "margin_cost_tiebreak")
        regret_cost = _regret_overall(regret, "cost_greedy_baseline")

        regret_block = {
            "raw": {
                "mean_normalized_regret": _to_float(regret_raw.get("mean_normalized_regret"), 0.0),
                "near_optimal_hit_rate_05": _to_float(regret_raw.get("near_optimal_hit_rate_05"), 0.0),
                "near_optimal_hit_rate_10": _to_float(regret_raw.get("near_optimal_hit_rate_10"), 0.0),
            },
            "stochastic": {
                "mean_normalized_regret": _to_float(regret_stochastic.get("mean_normalized_regret"), 0.0),
            },
            "margin": {
                "mean_normalized_regret": _to_float(regret_margin.get("mean_normalized_regret"), 0.0),
            },
            "cost_greedy": {
                "mean_normalized_regret": _to_float(regret_cost.get("mean_normalized_regret"), 0.0),
            },
        }

        raw_block = _replay_mode_block(raw_summary)
        stochastic_block = _replay_mode_block(stochastic_summary)
        margin_block = _replay_mode_block(margin_summary)

        replay_block = {
            "raw": raw_block,
            "stochastic": stochastic_block,
            "margin": margin_block,
            "http_timeout_count": max(
                int(raw_block["http_timeout_count"]),
                int(stochastic_block["http_timeout_count"]),
                int(margin_block["http_timeout_count"]),
            ),
            "http_connection_error_count": max(
                int(raw_block["http_connection_error_count"]),
                int(stochastic_block["http_connection_error_count"]),
                int(margin_block["http_connection_error_count"]),
            ),
        }

        readiness_value = str(readiness.get("readiness") or "unknown")
        if readiness_value not in readiness_counter:
            readiness_value = "unknown"
        readiness_counter[readiness_value] += 1

        acceptance = {
            "trace_strict_ok": bool(training_block["trace_hit_ratio"] >= 1.0 - 1.0e-9 and training_block["trace_fallback_count"] <= 0.0),
            "feasibility_ok": bool(training_block["final_mean_feasibility"] >= 0.95),
            "replay_raw_ok": bool(raw_block["pass"]),
            "replay_stochastic_ok": bool(stochastic_block["pass"]),
            "replay_margin_ok": bool(margin_block["pass"]),
            "replay_all_ok": bool(raw_block["pass"] and stochastic_block["pass"] and margin_block["pass"]),
            "readiness_not_fail": readiness_value != "fail",
        }

        seed_payload = {
            "seed": seed,
            "run_dir": str(run_dir),
            "training": training_block,
            "state_conditioned": state_block,
            "regret": regret_block,
            "replay": replay_block,
            "readiness": readiness_value,
            "readiness_reasons": readiness.get("reasons", []),
            "acceptance": acceptance,
        }
        per_seed[str(seed)] = seed_payload

        row_numeric: Dict[str, float] = {}
        _flatten_numeric("training.", training_block, row_numeric)
        _flatten_numeric("state_conditioned.", state_block, row_numeric)
        _flatten_numeric("regret.", regret_block, row_numeric)
        _flatten_numeric("replay.", replay_block, row_numeric)
        row_numeric["acceptance.trace_strict_ok"] = 1.0 if acceptance["trace_strict_ok"] else 0.0
        row_numeric["acceptance.feasibility_ok"] = 1.0 if acceptance["feasibility_ok"] else 0.0
        row_numeric["acceptance.replay_raw_ok"] = 1.0 if acceptance["replay_raw_ok"] else 0.0
        row_numeric["acceptance.replay_stochastic_ok"] = 1.0 if acceptance["replay_stochastic_ok"] else 0.0
        row_numeric["acceptance.replay_margin_ok"] = 1.0 if acceptance["replay_margin_ok"] else 0.0
        row_numeric["acceptance.replay_all_ok"] = 1.0 if acceptance["replay_all_ok"] else 0.0
        row_numeric["acceptance.readiness_not_fail"] = 1.0 if acceptance["readiness_not_fail"] else 0.0
        row_numeric["readiness.strong_pass"] = 1.0 if readiness_value == "strong_pass" else 0.0
        row_numeric["readiness.conditional_pass"] = 1.0 if readiness_value == "conditional_pass" else 0.0
        row_numeric["readiness.fail"] = 1.0 if readiness_value == "fail" else 0.0
        numeric_rows.append(row_numeric)

    all_keys = sorted({k for row in numeric_rows for k in row.keys()})
    aggregate_numeric: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        vals = [row[key] for row in numeric_rows if key in row]
        aggregate_numeric[key] = _agg(vals)

    num_pass = int(readiness_counter.get("strong_pass", 0))
    num_cond = int(readiness_counter.get("conditional_pass", 0))
    num_fail = int(readiness_counter.get("fail", 0))

    global_acceptance = {
        "num_pass": num_pass,
        "num_conditional_pass": num_cond,
        "num_fail": num_fail,
        "at_least_2_of_3_pass_or_conditional": bool(num_pass + num_cond >= 2),
        "no_replay_fail": all(
            bool(seed_data.get("acceptance", {}).get("replay_all_ok", False))
            for seed_data in per_seed.values()
        ),
        "no_trace_fallback": all(
            _to_float(seed_data.get("training", {}).get("trace_fallback_count"), 1.0) <= 0.0
            for seed_data in per_seed.values()
        ),
        "no_readiness_fail": all(
            str(seed_data.get("readiness")) != "fail"
            for seed_data in per_seed.values()
        ),
    }

    output_payload = {
        "run_root": str(run_root),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "numeric": aggregate_numeric,
            "readiness_counts": readiness_counter,
            "num_pass": num_pass,
            "num_conditional_pass": num_cond,
            "num_fail": num_fail,
            "global_acceptance": global_acceptance,
        },
        "note": "energy remains requires_manual_audit and is excluded from pass/fail gating.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
