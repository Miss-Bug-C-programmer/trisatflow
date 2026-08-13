from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_builder import canonical_row
from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import collect_live_states
from trisatflow.satedgesim_eval.state_adapter import build_trisatflow_observation


ACTION_TIER = {0: "local", 1: "neighbor", 2: "geo", 3: "ground"}
MODES = ["raw_argmax", "stochastic_eval", "margin_cost_tiebreak", "cost_greedy_baseline"]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _visible(canonical: Dict[str, float], action: int) -> bool:
    tier = ACTION_TIER[action]
    return bool(_to_float(canonical.get(f"{tier}_visible"), 0.0) > 0.5)


def _action_costs(canonical: Dict[str, float], policy: FrozenTriSatFlowPolicy) -> List[float]:
    costs: List[float] = []
    reward = policy.cfg.reward
    for action in range(4):
        tier = ACTION_TIER[action]
        if not _visible(canonical, action):
            costs.append(float("inf"))
            continue
        delay = max(0.0, _to_float(canonical.get(f"{tier}_delay"), 0.0))
        queue = max(0.0, _to_float(canonical.get(f"{tier}_queue"), 0.0))
        rate = max(1.0e-9, _to_float(canonical.get(f"{tier}_rate"), 0.0))
        tx = 0.0 if action == 0 else (1.0 / rate)
        compute = max(0.0, delay - tx)
        costs.append(
            reward.delay_weight * delay
            + reward.queue_weight * queue
            + reward.transmission_weight * tx
            + reward.compute_weight * compute
        )
    return costs


def _dist(actions: List[int]) -> Dict[str, float]:
    n = max(1, len(actions))
    return {name: float(sum(1 for a in actions if a == i) / n) for i, name in enumerate(ACTION_NAMES)}


def _safe_regret(selected: float, oracle: float) -> float:
    if not (selected < float("inf") and oracle < float("inf")):
        return 1.0e6
    return float(max(0.0, (selected - oracle) / max(1.0e-6, oracle)))


def _near(selected: float, oracle: float, eps: float = 0.05) -> bool:
    if not (selected < float("inf") and oracle < float("inf")):
        return False
    return bool(selected <= oracle * (1.0 + eps) + 1.0e-12)


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose live policy regret on checkpoint-normalized live states.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario-profile", type=str, default="mixed_cost_landscape_v2")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--num-states", type=int, default=500)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tie-eps", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    states = collect_live_states(
        base_url=args.base_url,
        scenario_profile=args.scenario_profile,
        task_source_mode=args.task_source_mode,
        num_states=args.num_states,
        devices_count=args.devices_count,
        seed=args.seed,
        request_timeout=args.request_timeout,
    )

    rng = {m: random.Random(13 + idx * 17) for idx, m in enumerate(MODES)}
    selected_actions: Dict[str, List[int]] = {m: [] for m in MODES}
    selected_costs: Dict[str, List[float]] = {m: [] for m in MODES}
    selected_probs: Dict[str, List[float]] = {m: [] for m in MODES}
    selected_ranks: Dict[str, List[float]] = {m: [] for m in MODES}
    phase_actions: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {m: [] for m in MODES + ["oracle"]})
    oracle_actions: List[int] = []
    oracle_costs: List[float] = []
    prob_sum = torch.zeros(4, dtype=torch.float64)
    cost_sum = torch.zeros(4, dtype=torch.float64)
    cost_count = torch.zeros(4, dtype=torch.float64)

    for state in states:
        obs, edge_index, edge_attr, source_index = build_trisatflow_observation(
            state,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=policy.obs_normalization_mode,
            normalization_stats=policy.obs_normalization_stats,
        )
        diagnostics = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        probs = diagnostics["probs"][source_index].detach().cpu().float()
        prob_sum += probs.to(dtype=torch.float64)

        raw_rows = list(state.get("denseSourceSummaries") or [])
        if raw_rows and 0 <= source_index < len(raw_rows):
            canonical = canonical_row(raw_rows[source_index])
        elif raw_rows:
            canonical = canonical_row(raw_rows[0])
        else:
            canonical = {
                "local_visible": 1.0,
                "neighbor_visible": 0.0,
                "geo_visible": 0.0,
                "ground_visible": 0.0,
                "local_rate": 1000.0,
                "neighbor_rate": 0.0,
                "geo_rate": 0.0,
                "ground_rate": 0.0,
                "local_delay": 0.02,
                "neighbor_delay": 0.0,
                "geo_delay": 0.0,
                "ground_delay": 0.0,
                "local_queue": 0.0,
                "neighbor_queue": 0.0,
                "geo_queue": 0.0,
                "ground_queue": 0.0,
            }
        costs = _action_costs(canonical, policy)
        finite = [a for a in range(4) if costs[a] < float("inf")]
        if not finite:
            finite = [0]
        oracle = min(finite, key=lambda a: costs[a])
        oracle_actions.append(int(oracle))
        oracle_costs.append(float(costs[oracle]))
        phase = str((state.get("task") or {}).get("scenarioPhase") or state.get("scenarioPhase") or "unknown")
        phase_actions[phase]["oracle"].append(int(oracle))

        for a in range(4):
            if costs[a] < float("inf"):
                cost_sum[a] += float(costs[a])
                cost_count[a] += 1.0

        for mode in MODES:
            sel = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=source_index,
                raw_rows=raw_rows,
                eval_mode=mode,
                tie_break_eps=args.tie_eps,
                rng=rng[mode],
            )
            a = int(sel["final_action"])
            selected_actions[mode].append(a)
            phase_actions[phase][mode].append(a)
            selected_cost = float(costs[a]) if 0 <= a < 4 else float("inf")
            selected_costs[mode].append(selected_cost)
            selected_probs[mode].append(float(probs[a].item()) if 0 <= a < 4 else 0.0)
            ordered = sorted(finite, key=lambda x: costs[x])
            rank = float(ordered.index(a) + 1) if a in ordered else float(len(ordered) + 1)
            selected_ranks[mode].append(rank)

    mode_payload: Dict[str, Any] = {}
    for mode in MODES:
        regrets = [_safe_regret(selected_costs[mode][i], oracle_costs[i]) for i in range(len(oracle_costs))]
        near05 = float(sum(1 for i in range(len(oracle_costs)) if _near(selected_costs[mode][i], oracle_costs[i], 0.05)) / max(1, len(oracle_costs)))
        agree = float(sum(1 for i in range(len(oracle_costs)) if selected_actions[mode][i] == oracle_actions[i]) / max(1, len(oracle_costs)))
        mode_payload[mode] = {
            "selected_action_distribution": _dist(selected_actions[mode]),
            "mean_selected_cost": _mean(selected_costs[mode]),
            "mean_oracle_cost": _mean(oracle_costs),
            "mean_normalized_regret": _mean(regrets),
            "near_optimal_hit_rate_05": near05,
            "selected_oracle_agreement": agree,
            "mean_selected_policy_probability": _mean(selected_probs[mode]),
            "mean_selected_cost_rank": _mean(selected_ranks[mode]),
        }

    per_phase = {}
    for phase, bucket in sorted(phase_actions.items()):
        per_phase[phase] = {
            "raw_argmax_distribution": _dist(bucket["raw_argmax"]),
            "stochastic_eval_distribution": _dist(bucket["stochastic_eval"]),
            "margin_cost_tiebreak_distribution": _dist(bucket["margin_cost_tiebreak"]),
            "cost_greedy_distribution": _dist(bucket["cost_greedy_baseline"]),
            "oracle_distribution": _dist(bucket["oracle"]),
        }

    cost_by_action_mean = {
        ACTION_NAMES[a]: float((cost_sum[a] / cost_count[a]).item()) if cost_count[a] > 0 else float("inf")
        for a in range(4)
    }
    prob_mean = {
        ACTION_NAMES[a]: float((prob_sum[a] / max(1, len(states))).item()) for a in range(4)
    }

    raw_ground = float(mode_payload["raw_argmax"]["selected_action_distribution"].get("ground", 0.0))
    greedy_ground = float(mode_payload["cost_greedy_baseline"]["selected_action_distribution"].get("ground", 0.0))
    oracle_ground = float(_dist(oracle_actions).get("ground", 0.0))
    near_geo_neighbor_raw = (
        float(mode_payload["raw_argmax"]["selected_action_distribution"].get("neighbor", 0.0))
        + float(mode_payload["raw_argmax"]["selected_action_distribution"].get("geo", 0.0))
    )

    flags = {
        "live_distribution_shift": bool(raw_ground >= 0.85 and greedy_ground < 0.75),
        "raw_policy_ground_bias": bool(raw_ground >= 0.85 and greedy_ground < 0.75),
        "live_cost_landscape_ground_dominant": bool(greedy_ground >= 0.85 and oracle_ground >= 0.85),
        "normalization_mismatch_resolved_but_policy_bad": bool(
            raw_ground >= 0.85 and greedy_ground < 0.75 and near_geo_neighbor_raw <= 0.05
        ),
    }

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "num_states": len(states),
        "obs_normalization_mode": policy.obs_normalization_mode,
        "obs_normalization_path": policy.obs_normalization_path,
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "live_raw_argmax_distribution": mode_payload["raw_argmax"]["selected_action_distribution"],
        "live_cost_greedy_distribution": mode_payload["cost_greedy_baseline"]["selected_action_distribution"],
        "live_oracle_distribution": _dist(oracle_actions),
        "live_raw_regret": mode_payload["raw_argmax"]["mean_normalized_regret"],
        "live_near_optimal_hit_rate_05": mode_payload["raw_argmax"]["near_optimal_hit_rate_05"],
        "live_selected_oracle_agreement": mode_payload["raw_argmax"]["selected_oracle_agreement"],
        "per_phase_live_distribution": per_phase,
        "policy_probability_mean": prob_mean,
        "cost_by_action_mean": cost_by_action_mean,
        "raw_action_vs_live_cost_rank_mean": mode_payload["raw_argmax"]["mean_selected_cost_rank"],
        "mode_results": mode_payload,
        "diagnosis": flags,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
