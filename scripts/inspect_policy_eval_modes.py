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


def _visible(row: Mapping[str, Any], tier: str) -> bool:
    raw = row.get(f"{tier}_visible")
    if raw is None:
        return tier == "local"
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return bool(raw)


def _total_delay(row: Mapping[str, Any], tier: str) -> float:
    explicit = row.get(f"{tier}_total_delay", row.get(f"{tier}_best_delay"))
    if explicit not in (None, ""):
        return max(0.0, _to_float(explicit, 0.0))
    return (
        max(0.0, _to_float(row.get(f"{tier}_prop_delay"), 0.0))
        + max(0.0, _to_float(row.get(f"{tier}_tx_delay"), 0.0))
        + max(0.0, _to_float(row.get(f"{tier}_compute_delay"), 0.0))
        + max(0.0, _to_float(row.get(f"{tier}_queue_delay"), 0.0))
    )


def _oracle_action(row: Mapping[str, Any]) -> int:
    names = ["local", "neighbor", "geo", "ground"]
    best_idx = 0
    best_cost = float("inf")
    for idx, name in enumerate(names):
        if not _visible(row, name):
            continue
        cost = _total_delay(row, name)
        if cost < best_cost:
            best_cost = cost
            best_idx = idx
    return int(best_idx)


def _distribution(actions: torch.Tensor) -> Dict[str, float]:
    return {name: _ratio(actions, idx) for idx, name in enumerate(ACTION_NAMES)}


def _agreement(pred: torch.Tensor, oracle: torch.Tensor) -> float:
    if pred.numel() == 0 or oracle.numel() == 0 or pred.shape != oracle.shape:
        return 0.0
    return float((pred == oracle).float().mean())


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q), dtype=torch.float32)))


def _phase_distribution(actions_by_phase: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for phase in sorted(actions_by_phase):
        t = torch.tensor(actions_by_phase[phase], dtype=torch.long)
        out[phase] = _distribution(t)
    return out


def _phase_agreement(
    pred_by_phase: Dict[str, List[int]],
    oracle_by_phase: Dict[str, List[int]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    phases = sorted(set(pred_by_phase) | set(oracle_by_phase))
    for phase in phases:
        pred = torch.tensor(pred_by_phase.get(phase, []), dtype=torch.long)
        oracle = torch.tensor(oracle_by_phase.get(phase, []), dtype=torch.long)
        out[phase] = _agreement(pred, oracle)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect policy distributions under raw/stochastic/tiebreak/cost-greedy eval modes.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--tie-eps", type=float, default=0.05)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    groups = load_trace_groups(
        args.trace,
        n_leo=args.n_leo,
        num_states=max(1, args.num_states // max(1, args.n_leo)),
    )

    raw_argmax_actions: List[int] = []
    stochastic_actions: List[int] = []
    tiebreak_actions: List[int] = []
    cost_greedy_actions: List[int] = []
    oracle_actions: List[int] = []
    prob_oracle_action: List[float] = []
    rank_oracle_action: List[float] = []

    top1_prob_margins: List[float] = []
    top1_logit_margins: List[float] = []
    tie_break_applied = 0
    near_tie_count = 0
    total_obs = 0
    phase_raw: Dict[str, List[int]] = defaultdict(list)
    phase_stochastic: Dict[str, List[int]] = defaultdict(list)
    phase_tiebreak: Dict[str, List[int]] = defaultdict(list)
    phase_cost: Dict[str, List[int]] = defaultdict(list)
    phase_oracle: Dict[str, List[int]] = defaultdict(list)
    phase_prob_oracle: Dict[str, List[float]] = defaultdict(list)

    stochastic_rng = random.Random(13)
    tiebreak_rng = random.Random(13)
    cost_rng = random.Random(13)

    for group in groups:
        obs, edge_index, edge_attr = shared_batch_from_trace_group(
            group,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=policy.obs_normalization_mode,
            normalization_stats=policy.obs_normalization_stats,
        )
        diagnostics = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        probs = diagnostics["probs"].detach().cpu()
        logits = diagnostics["masked_logits"].detach().cpu()
        mask = diagnostics["mask"].detach().cpu().bool()

        for idx, row in enumerate(group):
            total_obs += 1
            phase = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            oracle = _oracle_action(row)
            oracle_actions.append(oracle)

            raw = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=idx,
                raw_rows=[dict(r) for r in group],
                eval_mode="raw_argmax",
                tie_break_eps=args.tie_eps,
                rng=random.Random(idx),
            )
            stochastic = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=idx,
                raw_rows=[dict(r) for r in group],
                eval_mode="stochastic_eval",
                tie_break_eps=args.tie_eps,
                rng=stochastic_rng,
            )
            tiebreak = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=idx,
                raw_rows=[dict(r) for r in group],
                eval_mode="margin_cost_tiebreak",
                tie_break_eps=args.tie_eps,
                rng=tiebreak_rng,
            )
            cost = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=idx,
                raw_rows=[dict(r) for r in group],
                eval_mode="cost_greedy_baseline",
                tie_break_eps=args.tie_eps,
                rng=cost_rng,
            )

            raw_argmax_actions.append(int(raw["final_action"]))
            stochastic_actions.append(int(stochastic["final_action"]))
            tiebreak_actions.append(int(tiebreak["final_action"]))
            cost_greedy_actions.append(int(cost["final_action"]))
            phase_raw[phase].append(int(raw["final_action"]))
            phase_stochastic[phase].append(int(stochastic["final_action"]))
            phase_tiebreak[phase].append(int(tiebreak["final_action"]))
            phase_cost[phase].append(int(cost["final_action"]))
            phase_oracle[phase].append(int(oracle))

            tie_break_applied += int(bool(tiebreak["tie_break_applied"]))
            top1_prob_margins.append(float(tiebreak["top1_prob_margin"]))
            feasible = [a for a in range(4) if bool(mask[idx, a])]
            if not feasible:
                feasible = [0]
            ordered = sorted(feasible, key=lambda a: float(probs[idx, a]), reverse=True)
            a1 = ordered[0]
            a2 = ordered[1] if len(ordered) > 1 else ordered[0]
            top1_logit_margins.append(float(logits[idx, a1] - logits[idx, a2]))
            near_tie_count += int(float(tiebreak["top1_prob_margin"]) <= args.tie_eps + 1.0e-12)
            p_oracle = float(probs[idx, oracle].item())
            prob_oracle_action.append(p_oracle)
            rank = int((torch.argsort(probs[idx], descending=True) == oracle).nonzero(as_tuple=False)[0].item() + 1)
            rank_oracle_action.append(float(rank))
            phase_prob_oracle[phase].append(p_oracle)

        if total_obs >= args.num_states:
            break

    raw_t = torch.tensor(raw_argmax_actions, dtype=torch.long)
    stochastic_t = torch.tensor(stochastic_actions, dtype=torch.long)
    tiebreak_t = torch.tensor(tiebreak_actions, dtype=torch.long)
    cost_t = torch.tensor(cost_greedy_actions, dtype=torch.long)
    oracle_t = torch.tensor(oracle_actions, dtype=torch.long)

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "policy_head": str(getattr(policy.cfg.algo, "policy_head", "gnn_only")),
        "obs_normalization_mode": policy.obs_normalization_mode,
        "obs_normalization_path": policy.obs_normalization_path,
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim),
        "num_observations": int(total_obs),
        "tie_eps": float(args.tie_eps),
        "raw_argmax_distribution": _distribution(raw_t),
        "stochastic_eval_distribution": _distribution(stochastic_t),
        "margin_cost_tiebreak_distribution": _distribution(tiebreak_t),
        "cost_greedy_baseline_distribution": _distribution(cost_t),
        "oracle_distribution": _distribution(oracle_t),
        "raw_argmax_vs_oracle_agreement": _agreement(raw_t, oracle_t),
        "stochastic_eval_vs_oracle_agreement": _agreement(stochastic_t, oracle_t),
        "margin_cost_tiebreak_vs_oracle_agreement": _agreement(tiebreak_t, oracle_t),
        "cost_greedy_vs_oracle_agreement": _agreement(cost_t, oracle_t),
        "prob_oracle_action_mean": float(sum(prob_oracle_action) / max(1, len(prob_oracle_action))),
        "prob_oracle_action_p50": _quantile(prob_oracle_action, 0.50),
        "oracle_action_rank_mean": float(sum(rank_oracle_action) / max(1, len(rank_oracle_action))),
        "oracle_action_rank_p50": _quantile(rank_oracle_action, 0.50),
        "tie_break_applied_ratio": float(tie_break_applied / max(1, total_obs)),
        "near_tie_ratio": float(near_tie_count / max(1, total_obs)),
        "top1_top2_margin_stats": {
            "prob_margin_mean": float(sum(top1_prob_margins) / max(1, len(top1_prob_margins))),
            "prob_margin_p50": _quantile(top1_prob_margins, 0.50),
            "prob_margin_p90": _quantile(top1_prob_margins, 0.90),
            "logit_margin_mean": float(sum(top1_logit_margins) / max(1, len(top1_logit_margins))),
            "logit_margin_p50": _quantile(top1_logit_margins, 0.50),
            "logit_margin_p90": _quantile(top1_logit_margins, 0.90),
        },
        "phase_policy_distribution_raw_argmax": _phase_distribution(phase_raw),
        "phase_policy_distribution_stochastic_eval": _phase_distribution(phase_stochastic),
        "phase_policy_distribution_margin_cost_tiebreak": _phase_distribution(phase_tiebreak),
        "phase_policy_distribution_cost_greedy": _phase_distribution(phase_cost),
        "phase_oracle_distribution": _phase_distribution(phase_oracle),
        "phase_raw_argmax_oracle_agreement": _phase_agreement(phase_raw, phase_oracle),
        "phase_stochastic_oracle_agreement": _phase_agreement(phase_stochastic, phase_oracle),
        "phase_margin_tiebreak_oracle_agreement": _phase_agreement(phase_tiebreak, phase_oracle),
        "phase_cost_greedy_oracle_agreement": _phase_agreement(phase_cost, phase_oracle),
        "phase_prob_oracle_action_mean": {k: float(sum(v) / max(1, len(v))) for k, v in sorted(phase_prob_oracle.items())},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
