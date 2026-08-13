from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _ratio(actions: torch.Tensor, action: int) -> float:
    if actions.numel() == 0:
        return 0.0
    return float((actions == action).float().mean())


def _distribution(actions: torch.Tensor) -> Dict[str, float]:
    return {name: _ratio(actions, idx) for idx, name in enumerate(ACTION_NAMES)}


def _is_visible(row: Mapping[str, Any], tier: str) -> bool:
    raw = row.get(f"{tier}_visible", row.get(f"{tier}Visible"))
    if raw in (None, ""):
        return tier == "local"
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return bool(raw)


def _tier_cost_components(row: Mapping[str, Any], tier: str) -> Dict[str, float]:
    delay = max(0.0, _to_float(row.get(f"{tier}_delay", row.get(f"{tier}_best_delay")), 0.0))
    queue = max(0.0, _to_float(row.get(f"{tier}_queue", row.get(f"{tier}_best_queue")), 0.0))
    rate = max(1.0e-6, _to_float(row.get(f"{tier}_rate"), 0.0))
    tx = 0.0 if tier == "local" else (1.0 / rate)
    compute = max(0.0, delay - tx)
    # Keep aligned with current oracle-aligned weighting defaults.
    cost = delay + 0.5 * queue + 0.2 * tx + 0.2 * compute
    return {
        "delay": delay,
        "queue": queue,
        "tx": tx,
        "compute": compute,
        "cost": cost,
    }


def _oracle_action_and_costs(row: Mapping[str, Any]) -> tuple[int, List[float], Dict[str, Dict[str, float]]]:
    tiers = ["local", "neighbor", "geo", "ground"]
    costs: List[float] = [float("inf")] * 4
    components: Dict[str, Dict[str, float]] = {}
    for i, tier in enumerate(tiers):
        comp = _tier_cost_components(row, tier)
        components[tier] = comp
        if _is_visible(row, tier):
            costs[i] = comp["cost"]
    best = min(range(4), key=lambda i: costs[i])
    return int(best), costs, components


def _agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.shape != b.shape:
        return 0.0
    return float((a == b).float().mean())


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q), dtype=torch.float32)))


def _corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).sum() * y.pow(2).sum()).clamp_min(1.0e-12))
    return float((x * y).sum() / denom)


def _mi(x: List[int], y: List[int]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    n = float(len(x))
    joint = Counter(zip(x, y))
    px = Counter(x)
    py = Counter(y)
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / n
        pa = px[a] / n
        pb = py[b] / n
        mi += pxy * math.log(max(1.0e-12, pxy / max(1.0e-12, pa * pb)))
    return float(mi)


def _pairwise_dist_stats(logits: torch.Tensor, probs: torch.Tensor, max_pairs: int = 8192) -> tuple[float, float]:
    n = logits.shape[0]
    if n <= 1:
        return 0.0, 0.0
    rng = random.Random(13)
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    logit_ds: List[float] = []
    prob_ds: List[float] = []
    for _ in range(num_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        logit_ds.append(float(torch.norm(logits[i] - logits[j], p=2)))
        prob_ds.append(float(torch.norm(probs[i] - probs[j], p=2)))
    if not logit_ds:
        return 0.0, 0.0
    return float(sum(logit_ds) / len(logit_ds)), float(sum(prob_ds) / len(prob_ds))


def _classify(
    *,
    raw_argmax_distribution: Dict[str, float],
    stochastic_distribution: Dict[str, float],
    oracle_distribution: Dict[str, float],
    raw_agreement: float,
    stochastic_agreement: float,
    phase_raw_vs_oracle_gap_mean: float,
    logit_std_mean: float,
    prob_std_mean: float,
    corr_neg_cost_prob: float,
) -> str:
    raw_geo = float(raw_argmax_distribution.get("geo", 0.0))
    if raw_geo > 0.98 and logit_std_mean < 0.01 and prob_std_mean < 0.01:
        return "constant_policy_with_global_geo_bias"
    dist_gap = sum(abs(float(stochastic_distribution.get(k, 0.0)) - float(oracle_distribution.get(k, 0.0))) for k in ACTION_NAMES)
    if dist_gap < 0.15 and stochastic_agreement < 0.45:
        return "weak_state_conditioning"
    if phase_raw_vs_oracle_gap_mean > 0.25 and stochastic_agreement < 0.50:
        return "weak_state_conditioning"
    if raw_geo > 0.98 and corr_neg_cost_prob > 0.30 and raw_agreement < 0.45:
        return "state_conditioned_but_argmax_biased"
    if raw_agreement >= 0.60 and stochastic_agreement >= 0.60:
        return "policy_matches_oracle_conditionally"
    if stochastic_agreement >= raw_agreement + 0.10 and raw_geo > 0.90:
        return "stochastic_policy_reasonable_but_deterministic_bad"
    return "weak_state_conditioning"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether a checkpoint learned state-conditioned policy decisions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--oracle", type=str, default="")
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=8192)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    groups = load_trace_groups(args.trace, n_leo=args.n_leo, num_states=max(1, args.num_states // max(1, args.n_leo)))

    raw_actions: List[int] = []
    stochastic_actions: List[int] = []
    oracle_actions: List[int] = []
    cost_greedy_actions: List[int] = []
    oracle_probs: List[float] = []
    oracle_ranks: List[float] = []
    entropy_values: List[float] = []
    all_logits: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    phase_ids: List[str] = []
    phase_raw: Dict[str, List[int]] = defaultdict(list)
    phase_stochastic: Dict[str, List[int]] = defaultdict(list)
    phase_oracle: Dict[str, List[int]] = defaultdict(list)
    phase_prob_oracle: Dict[str, List[float]] = defaultdict(list)
    phase_index_map: Dict[str, int] = {}
    phase_seq: List[int] = []

    corr_cost_adv_x: List[float] = []
    corr_cost_adv_y: List[float] = []
    corr_neg_cost_x: List[float] = []
    corr_neg_cost_y: List[float] = []
    tier_component_records: List[Dict[str, Any]] = []
    obs_feature_values: List[torch.Tensor] = []

    rng_stoch = random.Random(13)
    total = 0
    for group in groups:
        obs, edge_index, edge_attr = shared_batch_from_trace_group(
            group,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=policy.obs_normalization_mode,
            normalization_stats=policy.obs_normalization_stats,
        )
        d = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        probs = d["probs"].detach().cpu()
        logits = d["masked_logits"].detach().cpu()
        mask = d["mask"].detach().cpu().bool()
        entropy = d["entropy"].detach().cpu()

        for i, row in enumerate(group):
            if total >= args.num_states:
                break
            total += 1
            phase = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            task_type = str(row.get("task_type", row.get("taskType", "unknown")))
            oracle, oracle_costs, tier_components = _oracle_action_and_costs(row)
            row_dicts = [dict(r) for r in group]
            raw = policy.select_action_from_diagnostics(
                d, source_index=i, raw_rows=row_dicts, eval_mode="raw_argmax", tie_break_eps=0.05, rng=random.Random(i)
            )
            stochastic = policy.select_action_from_diagnostics(
                d, source_index=i, raw_rows=row_dicts, eval_mode="stochastic_eval", tie_break_eps=0.05, rng=rng_stoch
            )
            cost = policy.select_action_from_diagnostics(
                d, source_index=i, raw_rows=row_dicts, eval_mode="cost_greedy_baseline", tie_break_eps=0.05, rng=random.Random(i + 17)
            )

            raw_action = int(raw["final_action"])
            stochastic_action = int(stochastic["final_action"])
            cost_action = int(cost["final_action"])
            raw_actions.append(raw_action)
            stochastic_actions.append(stochastic_action)
            oracle_actions.append(oracle)
            cost_greedy_actions.append(cost_action)
            all_logits.append(logits[i].clone())
            all_probs.append(probs[i].clone())
            obs_feature_values.append(obs[i].detach().cpu())
            entropy_values.append(float(entropy[i].item()))

            p_oracle = float(probs[i, oracle].item())
            oracle_probs.append(p_oracle)
            rank = int((torch.argsort(probs[i], descending=True) == oracle).nonzero(as_tuple=False)[0].item() + 1)
            oracle_ranks.append(float(rank))

            phase_raw[phase].append(raw_action)
            phase_stochastic[phase].append(stochastic_action)
            phase_oracle[phase].append(oracle)
            phase_prob_oracle[phase].append(p_oracle)
            phase_ids.append(phase)
            if phase not in phase_index_map:
                phase_index_map[phase] = len(phase_index_map)
            phase_seq.append(phase_index_map[phase])

            finite = [c for c in oracle_costs if math.isfinite(c)]
            if finite:
                sorted_costs = sorted(finite)
                cost_adv = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else 0.0
                best_other = max(float(probs[i, j].item()) for j in range(4) if j != oracle)
                corr_cost_adv_x.append(float(cost_adv))
                corr_cost_adv_y.append(float(probs[i, oracle].item()) - best_other)

            for j in range(4):
                c = oracle_costs[j]
                if math.isfinite(c):
                    corr_neg_cost_x.append(-float(c))
                    corr_neg_cost_y.append(float(probs[i, j].item()))

            tier_component_records.append(
                {
                    "phase": phase,
                    "task_type": task_type,
                    "oracle_action": ACTION_NAMES[oracle],
                    "tier_components": tier_components,
                }
            )
        if total >= args.num_states:
            break

    raw_t = torch.tensor(raw_actions, dtype=torch.long)
    stochastic_t = torch.tensor(stochastic_actions, dtype=torch.long)
    oracle_t = torch.tensor(oracle_actions, dtype=torch.long)
    cost_t = torch.tensor(cost_greedy_actions, dtype=torch.long)
    logits_t = torch.stack(all_logits, dim=0) if all_logits else torch.zeros((0, 4), dtype=torch.float32)
    probs_t = torch.stack(all_probs, dim=0) if all_probs else torch.zeros((0, 4), dtype=torch.float32)
    obs_t = torch.stack(obs_feature_values, dim=0) if obs_feature_values else torch.zeros((0, 0), dtype=torch.float32)

    phase_policy_distribution_raw_argmax = {}
    phase_policy_distribution_stochastic = {}
    phase_oracle_distribution = {}
    phase_raw_oracle_agreement = {}
    phase_stochastic_oracle_agreement = {}
    phase_prob_oracle_action_mean = {}
    phase_gap_values: List[float] = []

    for phase in sorted(phase_oracle):
        raw_phase = torch.tensor(phase_raw[phase], dtype=torch.long)
        st_phase = torch.tensor(phase_stochastic[phase], dtype=torch.long)
        or_phase = torch.tensor(phase_oracle[phase], dtype=torch.long)
        phase_policy_distribution_raw_argmax[phase] = _distribution(raw_phase)
        phase_policy_distribution_stochastic[phase] = _distribution(st_phase)
        phase_oracle_distribution[phase] = _distribution(or_phase)
        phase_raw_oracle_agreement[phase] = _agreement(raw_phase, or_phase)
        phase_stochastic_oracle_agreement[phase] = _agreement(st_phase, or_phase)
        phase_prob_oracle_action_mean[phase] = float(sum(phase_prob_oracle[phase]) / max(1, len(phase_prob_oracle[phase])))
        phase_gap_values.append(
            sum(abs(phase_policy_distribution_raw_argmax[phase][k] - phase_oracle_distribution[phase][k]) for k in ACTION_NAMES)
        )

    mean_pairwise_logit_distance, mean_pairwise_prob_distance = _pairwise_dist_stats(logits_t, probs_t)
    corr_cost_adv_logit = _corr(corr_cost_adv_x, corr_cost_adv_y)
    corr_neg_cost_prob = _corr(corr_neg_cost_x, corr_neg_cost_y)
    mi_phase_argmax = _mi(phase_seq, raw_actions)
    mi_oracle_argmax = _mi(oracle_actions, raw_actions)

    logit_std = {ACTION_NAMES[i]: float(logits_t[:, i].std(unbiased=False)) if logits_t.numel() else 0.0 for i in range(4)}
    prob_std = {ACTION_NAMES[i]: float(probs_t[:, i].std(unbiased=False)) if probs_t.numel() else 0.0 for i in range(4)}

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "oracle_input": args.oracle,
        "policy_head": str(getattr(policy.cfg.algo, "policy_head", "gnn_only")),
        "obs_normalization_mode": policy.obs_normalization_mode,
        "obs_normalization_path": policy.obs_normalization_path,
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim),
        "num_states": int(total),
        "raw_argmax_distribution": _distribution(raw_t),
        "stochastic_distribution": _distribution(stochastic_t),
        "oracle_distribution": _distribution(oracle_t),
        "cost_greedy_distribution": _distribution(cost_t),
        "raw_argmax_oracle_agreement": _agreement(raw_t, oracle_t),
        "stochastic_oracle_agreement": _agreement(stochastic_t, oracle_t),
        "prob_oracle_action_mean": float(sum(oracle_probs) / max(1, len(oracle_probs))),
        "prob_oracle_action_p50": _quantile(oracle_probs, 0.50),
        "rank_of_oracle_action_mean": float(sum(oracle_ranks) / max(1, len(oracle_ranks))),
        "entropy_mean": float(sum(entropy_values) / max(1, len(entropy_values))),
        "logit_std_across_states_by_action": logit_std,
        "logit_std_across_states_by_action_mean": float(sum(logit_std.values()) / max(1, len(logit_std))),
        "prob_std_across_states_by_action": prob_std,
        "prob_std_across_states_by_action_mean": float(sum(prob_std.values()) / max(1, len(prob_std))),
        "phase_policy_distribution_raw_argmax": phase_policy_distribution_raw_argmax,
        "phase_policy_distribution_stochastic": phase_policy_distribution_stochastic,
        "phase_oracle_distribution": phase_oracle_distribution,
        "phase_raw_oracle_agreement": phase_raw_oracle_agreement,
        "phase_stochastic_oracle_agreement": phase_stochastic_oracle_agreement,
        "phase_prob_oracle_action_mean": phase_prob_oracle_action_mean,
        "state_sensitivity": {
            "mean_pairwise_logit_distance": mean_pairwise_logit_distance,
            "mean_pairwise_prob_distance": mean_pairwise_prob_distance,
            "correlation_between_cost_advantage_and_policy_logit": corr_cost_adv_logit,
            "correlation_between_negative_cost_and_policy_prob": corr_neg_cost_prob,
            "mutual_information_phase_argmax": mi_phase_argmax,
            "mutual_information_oracle_argmax": mi_oracle_argmax,
        },
        "tier_cost_component_records_sample": tier_component_records[:32],
    }
    saturation_fields = [
        "local_visible",
        "neighbor_visible",
        "geo_visible",
        "ground_visible",
        "local_rate",
        "neighbor_rate",
        "geo_rate",
        "ground_rate",
        "local_delay",
        "neighbor_delay",
        "geo_delay",
        "ground_delay",
        "local_queue",
        "neighbor_queue",
        "geo_queue",
        "ground_queue",
    ]
    feature_saturation_ratio_by_field: Dict[str, float] = {}
    saturation_warnings: List[str] = []
    if obs_t.numel() > 0:
        dim = min(obs_t.shape[1], len(saturation_fields))
        for i in range(dim):
            ratio = float((obs_t[:, i] >= 0.999).float().mean())
            feature_saturation_ratio_by_field[saturation_fields[i]] = ratio
            if ratio > 0.30:
                saturation_warnings.append(f"{saturation_fields[i]}_saturation_ratio={ratio:.4f}")
    payload["feature_saturation_ratio_by_field"] = feature_saturation_ratio_by_field
    payload["feature_saturation_warning"] = saturation_warnings

    payload["classification"] = _classify(
        raw_argmax_distribution=payload["raw_argmax_distribution"],
        stochastic_distribution=payload["stochastic_distribution"],
        oracle_distribution=payload["oracle_distribution"],
        raw_agreement=float(payload["raw_argmax_oracle_agreement"]),
        stochastic_agreement=float(payload["stochastic_oracle_agreement"]),
        phase_raw_vs_oracle_gap_mean=float(sum(phase_gap_values) / max(1, len(phase_gap_values))),
        logit_std_mean=float(sum(logit_std.values()) / max(1, len(logit_std))),
        prob_std_mean=float(sum(prob_std.values()) / max(1, len(prob_std))),
        corr_neg_cost_prob=float(corr_neg_cost_prob),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
