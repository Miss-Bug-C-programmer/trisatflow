from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trisatflow.config import load_config
from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.inspection import load_trace_rows

import diagnose_reward_oracle_alignment as diag


TIERS = list(ACTION_NAMES)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit-test oracle cost and env reward alignment on sampled trace states.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_trace_rows(args.trace, num_rows=args.num_states)
    if not rows:
        raise RuntimeError("trace has no rows")

    reward_by_action: Dict[str, List[float]] = defaultdict(list)
    cost_by_action: Dict[str, List[float]] = defaultdict(list)
    spearman_vals: List[float] = []
    kendall_vals: List[float] = []
    agreement_hits = 0
    agreement_total = 0

    for row in rows:
        evaluated = diag._evaluate_row(row, cfg)
        feasible = [idx for idx, tier in enumerate(TIERS) if diag._visible(row, tier)]
        if not feasible:
            feasible = [0]

        oracle = [evaluated[idx]["oracle_cost"] for idx in feasible]
        env_neg_reward = [-evaluated[idx]["env_reward"] for idx in feasible]
        if len(feasible) >= 2:
            spearman_vals.append(diag._spearman(oracle, env_neg_reward))
            kendall_vals.append(diag._kendall(oracle, env_neg_reward))

        best_oracle = min(feasible, key=lambda idx: evaluated[idx]["oracle_cost"])
        best_env = min(feasible, key=lambda idx: env_neg_reward[feasible.index(idx)])
        agreement_hits += int(best_oracle == best_env)
        agreement_total += 1

        for idx in feasible:
            tier = TIERS[idx]
            reward_by_action[tier].append(float(evaluated[idx]["env_reward"]))
            cost_by_action[tier].append(float(evaluated[idx]["env_cost"]))

    agreement = float(agreement_hits / max(1, agreement_total))
    spearman = _mean(spearman_vals)
    kendall = _mean(kendall_vals)

    per_action_reward_mean = {k: _mean(v) for k, v in reward_by_action.items()}
    per_action_cost_mean = {k: _mean(v) for k, v in cost_by_action.items()}
    reward_values = list(per_action_reward_mean.values())
    reward_scale_span = (max(reward_values) - min(reward_values)) if reward_values else 0.0

    checks = {
        "oracle_env_best_action_agreement_ge_0_60": agreement >= 0.60,
        "spearman_rank_corr_mean_ge_0_50": spearman >= 0.50,
        "no_order_of_magnitude_action_bias": reward_scale_span <= 100.0,
    }

    status = "PASS" if all(checks.values()) else "FAIL"
    payload: Dict[str, Any] = {
        "status": status,
        "config": args.config,
        "trace": args.trace,
        "num_rows": len(rows),
        "oracle_env_best_action_agreement": agreement,
        "spearman_rank_corr_mean": spearman,
        "kendall_rank_corr_mean": kendall,
        "env_reward_mean_by_action": per_action_reward_mean,
        "env_cost_mean_by_action": per_action_cost_mean,
        "reward_scale_span": reward_scale_span,
        "checks": checks,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
