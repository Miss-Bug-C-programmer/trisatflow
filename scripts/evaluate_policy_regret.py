from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.baselines.registry import apply_architecture_filter, normalize_architecture
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


EVAL_MODES_SUPPORTED = {"raw_argmax", "stochastic_eval", "margin_cost_tiebreak", "cost_greedy_baseline"}
ACTION_TO_TIER = {0: "local", 1: "neighbor", 2: "geo", 3: "ground"}
MODE_SEED_OFFSET = {
    "raw_argmax": 101,
    "stochastic_eval": 0,
    "margin_cost_tiebreak": 202,
    "cost_greedy_baseline": 303,
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value in (None, ""):
        return default
    return bool(value)

def _canonical_from_trace_row(row: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tier in ("local", "neighbor", "geo", "ground"):
        out[f"{tier}_visible"] = 1.0 if _to_bool(row.get(f"{tier}_visible", row.get(f"{tier}Visible")), tier == "local") else 0.0
        out[f"{tier}_rate"] = max(0.0, _to_float(row.get(f"{tier}_rate", row.get(f"{tier}Rate")), 0.0))
        delay = row.get(f"{tier}_delay")
        if delay in (None, ""):
            delay = row.get(f"{tier}_best_delay", row.get(f"{tier}BestDelay"))
        if delay in (None, ""):
            delay = row.get(f"{tier}Delay")
        out[f"{tier}_delay"] = max(0.0, _to_float(delay, 0.0))
        queue = row.get(f"{tier}_queue")
        if queue in (None, ""):
            queue = row.get(f"{tier}_best_queue", row.get(f"{tier}BestQueue"))
        if queue in (None, ""):
            queue = row.get(f"{tier}Queue")
        out[f"{tier}_queue"] = max(0.0, _to_float(queue, 0.0))
    return out


def _tier_cost(canonical: Mapping[str, float], tier: str) -> float:
    delay = max(0.0, _to_float(canonical.get(f"{tier}_delay"), 0.0))
    queue = max(0.0, _to_float(canonical.get(f"{tier}_queue"), 0.0))
    rate = max(1.0e-6, _to_float(canonical.get(f"{tier}_rate"), 0.0))
    tx = 0.0 if tier == "local" else (1.0 / rate)
    compute = max(0.0, delay - tx)
    return float(delay + 0.5 * queue + 0.2 * tx + 0.2 * compute)


def _action_costs(
    canonical: Mapping[str, float],
    *,
    architecture: str = "full",
    include_failure_risk: bool = False,
    failure_penalty_weight: float = 0.0,
    failure_risk_by_tier: Mapping[str, float] | None = None,
) -> List[float]:
    arch_mask = apply_architecture_filter([1, 1, 1, 1], normalize_architecture(architecture))
    out: List[float] = []
    for action in range(4):
        if not bool(arch_mask[action]):
            out.append(float("inf"))
            continue
        tier = ACTION_TO_TIER[action]
        visible = bool(_to_float(canonical.get(f"{tier}_visible"), 0.0) > 0.5)
        if not visible:
            out.append(float("inf"))
            continue
        base = _tier_cost(canonical, tier)
        if include_failure_risk:
            risk = _to_float((failure_risk_by_tier or {}).get(tier), 0.0)
            base += max(0.0, float(failure_penalty_weight)) * max(0.0, min(1.0, risk))
        out.append(base)
    return out


def _load_failure_risk(path: str) -> Dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    source = payload.get("failure_risk_by_tier", payload)
    if not isinstance(source, dict):
        return {}
    out: Dict[str, float] = {}
    for key in ("local", "neighbor", "geo", "ground"):
        out[key] = max(0.0, min(1.0, _to_float(source.get(key), 0.0)))
    return out


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q), dtype=torch.float32)))


def _distribution(actions: List[int]) -> Dict[str, float]:
    if not actions:
        return {name: 0.0 for name in ACTION_NAMES}
    t = torch.tensor(actions, dtype=torch.long)
    return {name: float((t == idx).float().mean().item()) for idx, name in enumerate(ACTION_NAMES)}


def _safe_regret(selected_cost: float, oracle_cost: float) -> float:
    if not (selected_cost < float("inf") and oracle_cost < float("inf")):
        return 1.0e6
    denom = max(1.0e-6, float(oracle_cost))
    return float(max(0.0, (float(selected_cost) - float(oracle_cost)) / denom))


def _is_near_optimal(selected_cost: float, oracle_cost: float, eps: float) -> bool:
    if not (selected_cost < float("inf") and oracle_cost < float("inf")):
        return False
    return bool(selected_cost <= oracle_cost * (1.0 + eps) + 1.0e-12)


def _mode_summary(
    *,
    selected_actions: List[int],
    oracle_actions: List[int],
    selected_costs: List[float],
    oracle_costs: List[float],
    normalized_regrets: List[float],
    phases: List[str],
) -> Dict[str, Any]:
    n = len(selected_actions)
    if n == 0:
        return {
            "overall": {},
            "per_phase": [],
            "per_action": {},
        }

    mean_selected_cost = float(sum(selected_costs) / n)
    mean_oracle_cost = float(sum(oracle_costs) / n)
    mean_regret = float(sum(normalized_regrets) / n)
    median_regret = _quantile(normalized_regrets, 0.50)
    p90_regret = _quantile(normalized_regrets, 0.90)
    mean_cost_ratio = float(sum((selected_costs[i] / max(1.0e-6, oracle_costs[i])) for i in range(n)) / n)

    near01 = float(sum(1.0 for i in range(n) if _is_near_optimal(selected_costs[i], oracle_costs[i], 0.01)) / n)
    near05 = float(sum(1.0 for i in range(n) if _is_near_optimal(selected_costs[i], oracle_costs[i], 0.05)) / n)
    near10 = float(sum(1.0 for i in range(n) if _is_near_optimal(selected_costs[i], oracle_costs[i], 0.10)) / n)

    selected_dist = _distribution(selected_actions)
    oracle_dist = _distribution(oracle_actions)
    selected_oracle_agreement = float(sum(1.0 for i in range(n) if selected_actions[i] == oracle_actions[i]) / n)

    per_phase_rows: List[Dict[str, Any]] = []
    phase_index: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(phases):
        phase_index[str(p)].append(i)
    for phase in sorted(phase_index):
        idxs = phase_index[phase]
        pa = [selected_actions[i] for i in idxs]
        po = [oracle_actions[i] for i in idxs]
        pr = [normalized_regrets[i] for i in idxs]
        psel = [selected_costs[i] for i in idxs]
        pora = [oracle_costs[i] for i in idxs]
        per_phase_rows.append(
            {
                "phase": phase,
                "mean_selected_cost": float(sum(psel) / len(idxs)),
                "mean_oracle_cost": float(sum(pora) / len(idxs)),
                "mean_normalized_regret": float(sum(pr) / len(idxs)),
                "near_optimal_hit_rate_05": float(sum(1.0 for i in idxs if _is_near_optimal(selected_costs[i], oracle_costs[i], 0.05)) / len(idxs)),
                "selected_action_distribution": _distribution(pa),
                "oracle_action_distribution": _distribution(po),
            }
        )

    per_action: Dict[str, Dict[str, float]] = {}
    for action in range(4):
        idxs = [i for i, a in enumerate(selected_actions) if a == action]
        name = ACTION_NAMES[action]
        if not idxs:
            per_action[name] = {
                "selected_count": 0,
                "mean_regret_when_selected": 0.0,
                "mean_cost_when_selected": 0.0,
            }
            continue
        per_action[name] = {
            "selected_count": int(len(idxs)),
            "mean_regret_when_selected": float(sum(normalized_regrets[i] for i in idxs) / len(idxs)),
            "mean_cost_when_selected": float(sum(selected_costs[i] for i in idxs) / len(idxs)),
        }

    return {
        "overall": {
            "mean_selected_cost": mean_selected_cost,
            "mean_oracle_cost": mean_oracle_cost,
            "mean_normalized_regret": mean_regret,
            "median_normalized_regret": median_regret,
            "p90_normalized_regret": p90_regret,
            "mean_cost_ratio": mean_cost_ratio,
            "near_optimal_hit_rate_01": near01,
            "near_optimal_hit_rate_05": near05,
            "near_optimal_hit_rate_10": near10,
            "selected_action_distribution": selected_dist,
            "oracle_action_distribution": oracle_dist,
            "selected_oracle_agreement": selected_oracle_agreement,
        },
        "per_phase": per_phase_rows,
        "per_action": per_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy regret under multiple eval modes against oracle-aligned per-action costs.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--oracle", type=str, default="")
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=8192)
    parser.add_argument("--eval-modes", type=str, default="raw_argmax,stochastic_eval,margin_cost_tiebreak,cost_greedy_baseline")
    parser.add_argument("--tie-eps", type=float, default=0.05)
    parser.add_argument("--stochastic-seed", type=int, default=13)
    parser.add_argument("--include-failure-risk", action="store_true")
    parser.add_argument("--failure-risk-json", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="full", choices=["only_leo", "leo_geo", "leo_ground", "full"])
    args = parser.parse_args()
    args.architecture = normalize_architecture(args.architecture)

    eval_modes = [m.strip() for m in str(args.eval_modes).split(",") if m.strip()]
    unknown = [m for m in eval_modes if m not in EVAL_MODES_SUPPORTED]
    if unknown:
        raise ValueError(f"Unsupported eval modes: {unknown}; supported={sorted(EVAL_MODES_SUPPORTED)}")
    if not eval_modes:
        raise ValueError("No eval modes specified.")

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    failure_risk_by_tier = _load_failure_risk(args.failure_risk_json)
    include_failure_risk = bool(args.include_failure_risk)
    failure_penalty_weight = _to_float(getattr(policy.cfg.reward, "failure_penalty_weight", 0.0), 0.0)
    groups = load_trace_groups(
        args.trace,
        n_leo=args.n_leo,
        num_states=max(1, args.num_states // max(1, args.n_leo)),
    )

    rng_map: Dict[str, random.Random] = {}
    for mode in eval_modes:
        seed = int(args.stochastic_seed) + int(MODE_SEED_OFFSET.get(mode, 0))
        rng_map[mode] = random.Random(seed)

    phases: List[str] = []
    oracle_actions: List[int] = []
    oracle_costs: List[float] = []

    selected_actions_by_mode: Dict[str, List[int]] = {m: [] for m in eval_modes}
    selected_costs_by_mode: Dict[str, List[float]] = {m: [] for m in eval_modes}
    regrets_by_mode: Dict[str, List[float]] = {m: [] for m in eval_modes}

    total = 0
    for group in groups:
        if total >= args.num_states:
            break
        obs, edge_index, edge_attr = shared_batch_from_trace_group(
            group,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=policy.obs_normalization_mode,
            normalization_stats=policy.obs_normalization_stats,
        )
        diagnostics = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        for i, row in enumerate(group):
            if total >= args.num_states:
                break
            total += 1
            phase = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            canonical = _canonical_from_trace_row(row)
            costs = _action_costs(
                canonical,
                architecture=args.architecture,
                include_failure_risk=include_failure_risk,
                failure_penalty_weight=failure_penalty_weight,
                failure_risk_by_tier=failure_risk_by_tier,
            )
            oracle_action = int(min(range(4), key=lambda a: costs[a]))
            oracle_cost = float(costs[oracle_action])
            phases.append(phase)
            oracle_actions.append(oracle_action)
            oracle_costs.append(oracle_cost)

            raw_rows = [dict(r) for r in group]
            for mode in eval_modes:
                res = policy.select_action_from_diagnostics(
                    diagnostics,
                    source_index=i,
                    raw_rows=raw_rows,
                    eval_mode=mode,
                    tie_break_eps=args.tie_eps,
                    rng=rng_map[mode],
                )
                action = int(res["final_action"])
                selected_cost = float(costs[action]) if 0 <= action < 4 else float("inf")
                regret = _safe_regret(selected_cost, oracle_cost)

                selected_actions_by_mode[mode].append(action)
                selected_costs_by_mode[mode].append(selected_cost)
                regrets_by_mode[mode].append(regret)

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "oracle": args.oracle,
        "num_states": int(total),
        "n_leo": int(args.n_leo),
        "eval_modes": eval_modes,
        "tie_eps": float(args.tie_eps),
        "stochastic_seed": int(args.stochastic_seed),
        "policy_head": str(getattr(policy.cfg.algo, "policy_head", "unknown")),
        "obs_normalization_mode": policy.obs_normalization_mode,
        "obs_normalization_path": policy.obs_normalization_path,
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim),
        "include_failure_risk": include_failure_risk,
        "failure_penalty_weight": failure_penalty_weight,
        "failure_risk_by_tier": failure_risk_by_tier,
        "architecture": args.architecture,
        "failure_risk_by_phase": {},
        "failure_risk_estimate_source": args.failure_risk_json if args.failure_risk_json else "none",
        "mode_results": {},
    }

    for mode in eval_modes:
        payload["mode_results"][mode] = _mode_summary(
            selected_actions=selected_actions_by_mode[mode],
            oracle_actions=oracle_actions,
            selected_costs=selected_costs_by_mode[mode],
            oracle_costs=oracle_costs,
            normalized_regrets=regrets_by_mode[mode],
            phases=phases,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
