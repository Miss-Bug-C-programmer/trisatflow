from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.config import TrainConfig, load_config
from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.inspection import load_trace_rows


TIERS = list(ACTION_NAMES)
EPS = 1.0e-9


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


def _visible(row: Mapping[str, Any], tier: str) -> bool:
    return _to_bool(row.get(f"{tier}_visible"), tier == "local")


def _component(row: Mapping[str, Any], tier: str, component: str) -> float:
    return max(0.0, _to_float(row.get(f"{tier}_{component}"), 0.0))


def _total_delay(row: Mapping[str, Any], tier: str) -> float:
    explicit = row.get(f"{tier}_total_delay", row.get(f"{tier}_best_delay"))
    if explicit not in (None, ""):
        return max(0.0, _to_float(explicit, 0.0))
    return (
        _component(row, tier, "prop_delay")
        + _component(row, tier, "tx_delay")
        + _component(row, tier, "compute_delay")
        + _component(row, tier, "queue_delay")
    )


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    rx = _ranks(x)
    ry = _ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(max(EPS, vx) * max(EPS, vy))
    return float(cov / denom)


def _kendall(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    concordant = 0
    discordant = 0
    ties = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) <= EPS or abs(dy) <= EPS:
                ties += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant + ties
    if denom <= 0:
        return 0.0
    return float((concordant - discordant) / denom)


def _normalize_by_feasible(values: Sequence[float], feasible: Sequence[bool], default_scale: float) -> List[float]:
    vis = [v for v, ok in zip(values, feasible) if ok]
    if not vis:
        return [v / max(default_scale, EPS) for v in values]
    ref = sum(vis) / max(1, len(vis))
    return [v / max(ref, EPS) for v in values]


def _evaluate_row(row: Mapping[str, Any], cfg: TrainConfig) -> Dict[int, Dict[str, float]]:
    rw = cfg.reward
    sc = cfg.scenario
    mode = str(getattr(rw, "mode", "legacy_remote_biased") or "legacy_remote_biased").strip().lower()
    remote_bonus_coef = rw.remote_bonus if abs(rw.remote_bonus) > 0.0 else rw.remote_feasible_bonus

    feasible = [_visible(row, tier) for tier in TIERS]
    prop = [_component(row, tier, "prop_delay") for tier in TIERS]
    tx = [_component(row, tier, "tx_delay") for tier in TIERS]
    compute = [_component(row, tier, "compute_delay") for tier in TIERS]
    queue_delay = [_component(row, tier, "queue_delay") for tier in TIERS]
    queue_len = [max(0.0, _to_float(row.get(f"{tier}_best_queue", row.get(f"{tier}_queue")), 0.0)) for tier in TIERS]
    total_delay = [_total_delay(row, tier) for tier in TIERS]
    rates = [max(EPS, _to_float(row.get(f"{tier}_rate"), 0.0)) for tier in TIERS]

    delay_scale = max(1.0e-6, float(sc.deadline_threshold))
    queue_scale = max(1.0, float(sc.max_queue))
    if rw.cost_normalization_enabled:
        if rw.per_tier_cost_normalization:
            norm_prop = _normalize_by_feasible(prop, feasible, delay_scale)
            norm_tx = _normalize_by_feasible(tx, feasible, delay_scale)
            norm_compute = _normalize_by_feasible(compute, feasible, delay_scale)
            norm_queue_delay = _normalize_by_feasible(queue_delay, feasible, delay_scale)
            norm_queue_len = _normalize_by_feasible(queue_len, feasible, queue_scale)
            norm_total_delay = _normalize_by_feasible(total_delay, feasible, delay_scale)
        else:
            norm_prop = [v / delay_scale for v in prop]
            norm_tx = [v / delay_scale for v in tx]
            norm_compute = [v / delay_scale for v in compute]
            norm_queue_delay = [v / delay_scale for v in queue_delay]
            norm_queue_len = [v / queue_scale for v in queue_len]
            norm_total_delay = [v / delay_scale for v in total_delay]
    else:
        norm_prop = prop
        norm_tx = tx
        norm_compute = compute
        norm_queue_delay = queue_delay
        norm_queue_len = queue_len
        norm_total_delay = total_delay

    out: Dict[int, Dict[str, float]] = {}
    for idx, tier in enumerate(TIERS):
        feasible_flag = bool(feasible[idx])
        infeasible = 0.0 if feasible_flag else 1.0
        remote = 1.0 if idx != 0 and feasible_flag else 0.0
        local_pen = rw.local_penalty + rw.local_queue_penalty * norm_queue_delay[idx] if idx == 0 else 0.0
        neighbor_pen = rw.neighbor_penalty + rw.neighbor_link_penalty / max(rates[idx], EPS) if idx == 1 else 0.0
        geo_pen = rw.geo_penalty + rw.geo_delay_penalty * norm_prop[idx] if idx == 2 else 0.0
        ground_pen = rw.ground_penalty + rw.ground_congestion_penalty * norm_queue_delay[idx] if idx == 3 else 0.0
        penalty = local_pen + neighbor_pen + geo_pen + ground_pen

        if mode == "oracle_aligned_cost":
            queue_share = queue_delay[idx] / max(total_delay[idx], EPS)
            tx_share = tx[idx] / max(total_delay[idx], EPS)
            compute_share = compute[idx] / max(total_delay[idx], EPS)
            delay_cost = rw.delay_weight * norm_total_delay[idx]
            queue_cost = rw.queue_weight * (0.75 * queue_share + 0.25 * norm_queue_len[idx])
            tx_cost = rw.transmission_weight * tx_share
            compute_cost = rw.compute_weight * compute_share
            feasibility_penalty = rw.feasibility_weight * infeasible
            energy_cost = 0.0
            if rw.include_energy:
                energy_cost = rw.energy * 0.0
            bonus = remote_bonus_coef * remote + rw.selected_when_visible_bonus * (1.0 if feasible_flag else 0.0)
            normalized_cost = delay_cost + queue_cost + tx_cost + compute_cost + feasibility_penalty + penalty - bonus
            raw_cost = (
                rw.delay_weight * total_delay[idx]
                + rw.queue_weight * (0.75 * queue_share + 0.25 * (queue_len[idx] / max(queue_scale, EPS)))
                + rw.transmission_weight * tx_share
                + rw.compute_weight * compute_share
                + rw.feasibility_weight * infeasible
                + penalty
                - bonus
            )
            env_cost = normalized_cost
            env_reward = -normalized_cost
            lower_effect = 0.0
        else:
            delay_term = norm_total_delay[idx]
            queue_term = norm_queue_len[idx]
            violation = max(0.0, norm_total_delay[idx] - 1.0)
            delay_cost = rw.delay * delay_term
            queue_cost = rw.queue * queue_term
            tx_cost = rw.delay * norm_tx[idx]
            compute_cost = rw.delay * norm_compute[idx]
            energy_cost = 0.0
            feasibility_penalty = rw.infeasible * infeasible
            bonus = (
                remote_bonus_coef * remote
                + rw.selected_when_visible_bonus * (1.0 if feasible_flag else 0.0)
                + rw.action_balance_bonus * 0.0
                + rw.offload_gain * 0.0
            )
            env_cost = (
                delay_cost
                + queue_cost
                + rw.violation * violation
                + feasibility_penalty
                + rw.load_balance * 0.0
                + rw.local_queue_pressure * 0.0
                + penalty
                - bonus
            )
            env_reward = -env_cost
            raw_cost = env_cost
            normalized_cost = env_cost
            lower_effect = 0.0

        out[idx] = {
            "oracle_cost": total_delay[idx],
            "env_reward": float(env_reward),
            "env_cost": float(env_cost),
            "delay_cost": float(delay_cost),
            "queue_cost": float(queue_cost),
            "transmission_cost": float(tx_cost),
            "compute_cost": float(compute_cost),
            "feasibility_cost": float(feasibility_penalty),
            "remote_bonus": float(remote_bonus_coef * remote),
            "tier_penalty": float(penalty),
            "action_balance_bonus": float(rw.action_balance_bonus * 0.0),
            "selected_when_visible_bonus": float(rw.selected_when_visible_bonus * (1.0 if feasible_flag else 0.0)),
            "lower_effect": float(lower_effect),
            "normalized_cost": float(normalized_cost),
            "raw_cost": float(raw_cost),
            "feasible": 1.0 if feasible_flag else 0.0,
        }
    return out


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _audit_classification(payload: Dict[str, Any], cfg: TrainConfig) -> List[str]:
    labels: List[str] = []
    agreement = float(payload.get("oracle_best_action_env_best_action_agreement", 0.0))
    spearman = float(payload.get("spearman_corr_oracle_cost_vs_negative_env_reward", 0.0))
    if agreement < 0.60 or spearman < 0.50:
        labels.append("reward_oracle_mismatch")

    reward_by_action = payload.get("env_reward_mean_by_action", {})
    local = float(reward_by_action.get("local", 0.0))
    neighbor = float(reward_by_action.get("neighbor", 0.0))
    geo = float(reward_by_action.get("geo", 0.0))
    ground = float(reward_by_action.get("ground", 0.0))
    if min(local, neighbor) < (max(geo, ground) - 2.0):
        labels.append("local_neighbor_over_penalized")

    remote_bonus = payload.get("bonus_mean_by_action", {})
    if max(float(remote_bonus.get("neighbor", 0.0)), float(remote_bonus.get("geo", 0.0)), float(remote_bonus.get("ground", 0.0))) > 0.5:
        labels.append("remote_bonus_too_large")

    penalty = payload.get("penalty_mean_by_action", {})
    if any(float(v) < -EPS for v in penalty.values()):
        labels.append("tier_penalty_wrong_sign")

    lower_effect = payload.get("lower_effect_mean_by_action", {})
    if max(abs(float(v)) for v in lower_effect.values()) > 0.5:
        labels.append("lower_allocation_bias")

    if not bool(cfg.reward.cost_normalization_enabled):
        labels.append("normalization_not_applied")

    env_reward_abs = [abs(float(v)) for v in reward_by_action.values()]
    if env_reward_abs and max(env_reward_abs) > 100.0:
        labels.append("reward_scale_explosion")

    if not labels:
        labels.append("oracle_env_alignment_ok")
    return sorted(set(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose oracle-cost vs environment-reward alignment from trace states.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found: {resolved}")
    cfg = load_config(resolved)

    rows = load_trace_rows(args.trace, num_rows=args.num_states)
    if not rows:
        raise RuntimeError("trace has no rows")

    oracle_cost_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    oracle_rank_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    env_reward_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    env_cost_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    delay_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    queue_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    transmission_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    compute_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    bonus_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    penalty_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    lower_effect_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    normalized_cost_mean_by_action: Dict[str, List[float]] = defaultdict(list)
    raw_cost_mean_by_action: Dict[str, List[float]] = defaultdict(list)

    spearman_vals: List[float] = []
    kendall_vals: List[float] = []
    agreement_hits = 0
    agreement_total = 0

    for row in rows:
        evaluated = _evaluate_row(row, cfg)
        feasible_actions = [idx for idx, tier in enumerate(TIERS) if _visible(row, tier)]
        if not feasible_actions:
            feasible_actions = [0]

        state_oracle = [evaluated[idx]["oracle_cost"] for idx in feasible_actions]
        state_env_neg_reward = [-evaluated[idx]["env_reward"] for idx in feasible_actions]
        if len(feasible_actions) >= 2:
            spearman_vals.append(_spearman(state_oracle, state_env_neg_reward))
            kendall_vals.append(_kendall(state_oracle, state_env_neg_reward))

        best_oracle = min(feasible_actions, key=lambda idx: evaluated[idx]["oracle_cost"])
        best_env = min(feasible_actions, key=lambda idx: state_env_neg_reward[feasible_actions.index(idx)])
        agreement_hits += int(best_oracle == best_env)
        agreement_total += 1

        sorted_cost = sorted((evaluated[idx]["oracle_cost"], idx) for idx in feasible_actions)
        rank_lookup = {idx: rank + 1 for rank, (_, idx) in enumerate(sorted_cost)}

        for idx in feasible_actions:
            name = TIERS[idx]
            rec = evaluated[idx]
            oracle_cost_mean_by_action[name].append(rec["oracle_cost"])
            oracle_rank_mean_by_action[name].append(float(rank_lookup[idx]))
            env_reward_mean_by_action[name].append(rec["env_reward"])
            env_cost_mean_by_action[name].append(rec["env_cost"])
            delay_mean_by_action[name].append(rec["delay_cost"])
            queue_mean_by_action[name].append(rec["queue_cost"])
            transmission_mean_by_action[name].append(rec["transmission_cost"])
            compute_mean_by_action[name].append(rec["compute_cost"])
            bonus_mean_by_action[name].append(rec["remote_bonus"] + rec["selected_when_visible_bonus"] + rec["action_balance_bonus"])
            penalty_mean_by_action[name].append(rec["tier_penalty"] + rec["feasibility_cost"])
            lower_effect_mean_by_action[name].append(rec["lower_effect"])
            normalized_cost_mean_by_action[name].append(rec["normalized_cost"])
            raw_cost_mean_by_action[name].append(rec["raw_cost"])

    payload: Dict[str, Any] = {
        "trace": args.trace,
        "run_dir": str(run_dir),
        "num_rows": len(rows),
        "oracle_cost_mean_by_action": {k: _mean(v) for k, v in oracle_cost_mean_by_action.items()},
        "oracle_rank_mean_by_action": {k: _mean(v) for k, v in oracle_rank_mean_by_action.items()},
        "env_reward_mean_by_action": {k: _mean(v) for k, v in env_reward_mean_by_action.items()},
        "env_cost_mean_by_action": {k: _mean(v) for k, v in env_cost_mean_by_action.items()},
        "delay_mean_by_action": {k: _mean(v) for k, v in delay_mean_by_action.items()},
        "queue_mean_by_action": {k: _mean(v) for k, v in queue_mean_by_action.items()},
        "transmission_mean_by_action": {k: _mean(v) for k, v in transmission_mean_by_action.items()},
        "compute_mean_by_action": {k: _mean(v) for k, v in compute_mean_by_action.items()},
        "bonus_mean_by_action": {k: _mean(v) for k, v in bonus_mean_by_action.items()},
        "penalty_mean_by_action": {k: _mean(v) for k, v in penalty_mean_by_action.items()},
        "lower_effect_mean_by_action": {k: _mean(v) for k, v in lower_effect_mean_by_action.items()},
        "normalized_cost_mean_by_action": {k: _mean(v) for k, v in normalized_cost_mean_by_action.items()},
        "raw_cost_mean_by_action": {k: _mean(v) for k, v in raw_cost_mean_by_action.items()},
        "spearman_corr_oracle_cost_vs_negative_env_reward": _mean(spearman_vals),
        "kendall_corr_oracle_cost_vs_negative_env_reward": _mean(kendall_vals),
        "oracle_best_action_env_best_action_agreement": float(agreement_hits / max(1, agreement_total)),
    }
    payload["diagnosis"] = _audit_classification(payload, cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
