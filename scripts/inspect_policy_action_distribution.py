from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import (
    collect_live_states,
    describe_mask_distribution,
    load_trace_groups,
    shared_batch_from_live_state,
    shared_batch_from_trace_group,
)


def _action_ratio(actions: torch.Tensor, action: int) -> float:
    if actions.numel() == 0:
        return 0.0
    return float((actions == action).float().mean())


def _summarize(
    policy: FrozenTriSatFlowPolicy,
    batches: List[Dict[str, Any]],
    *,
    source: str,
    eval_mode: str,
    tie_eps: float,
    stochastic_seed: int,
) -> Dict[str, Any]:
    argmax_actions = []
    sampled_actions = []
    final_actions = []
    probs = []
    logits = []
    entropies = []
    masks = []
    tie_break_applied = 0
    cost_rank_values: List[float] = []
    rng = random.Random(int(stochastic_seed))
    for item in batches:
        obs = item["obs"]
        edge_index = item["edge_index"]
        edge_attr = item["edge_attr"]
        raw_rows = item.get("raw_rows") or []
        diagnostics = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        argmax_actions.append(diagnostics["argmax_action"].detach().cpu())
        sampled_actions.append(diagnostics["sampled_action"].detach().cpu())
        probs.append(diagnostics["probs"].detach().cpu())
        logits.append(diagnostics["masked_logits"].detach().cpu())
        entropies.append(diagnostics["entropy"].detach().cpu())
        masks.append(diagnostics["mask"].detach().cpu())
        final_batch = []
        for agent_idx in range(diagnostics["probs"].shape[0]):
            selected = policy.select_action_from_diagnostics(
                diagnostics,
                source_index=agent_idx,
                raw_rows=raw_rows,
                eval_mode=eval_mode,
                tie_break_eps=tie_eps,
                rng=rng,
            )
            final_batch.append(int(selected["final_action"]))
            tie_break_applied += int(bool(selected["tie_break_applied"]))
            cost_rank_values.append(float(selected["selected_by_cost_rank"]))
        final_actions.append(torch.tensor(final_batch, dtype=torch.long))
    argmax_tensor = torch.cat(argmax_actions, dim=0) if argmax_actions else torch.empty(0, dtype=torch.long)
    sampled_tensor = torch.cat(sampled_actions, dim=0) if sampled_actions else torch.empty(0, dtype=torch.long)
    final_tensor = torch.cat(final_actions, dim=0) if final_actions else torch.empty(0, dtype=torch.long)
    prob_tensor = torch.cat(probs, dim=0) if probs else torch.empty((0, 4), dtype=torch.float32)
    logit_tensor = torch.cat(logits, dim=0) if logits else torch.empty((0, 4), dtype=torch.float32)
    entropy_tensor = torch.cat(entropies, dim=0) if entropies else torch.empty(0, dtype=torch.float32)
    mask_tensor = torch.cat(masks, dim=0) if masks else torch.empty((0, 4), dtype=torch.bool)
    total_obs = int(mask_tensor.shape[0]) if mask_tensor.ndim == 2 else 0

    result: Dict[str, Any] = {
        "num_states": len(batches),
        "num_agent_observations": total_obs,
        "source": source,
        "eval_mode": eval_mode,
        "tie_break_eps": float(tie_eps),
        "stochastic_seed": int(stochastic_seed),
        "checkpoint_path": str(policy.checkpoint_path),
        "obs_normalization_mode": policy.obs_normalization_mode,
        "obs_normalization_path": policy.obs_normalization_path,
        "obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim),
        "raw_argmax_local_ratio": _action_ratio(argmax_tensor, 0),
        "raw_argmax_neighbor_ratio": _action_ratio(argmax_tensor, 1),
        "raw_argmax_geo_ratio": _action_ratio(argmax_tensor, 2),
        "raw_argmax_ground_ratio": _action_ratio(argmax_tensor, 3),
        "argmax_local_ratio": _action_ratio(argmax_tensor, 0),
        "argmax_neighbor_ratio": _action_ratio(argmax_tensor, 1),
        "argmax_geo_ratio": _action_ratio(argmax_tensor, 2),
        "argmax_ground_ratio": _action_ratio(argmax_tensor, 3),
        "sampled_local_ratio": _action_ratio(sampled_tensor, 0),
        "sampled_neighbor_ratio": _action_ratio(sampled_tensor, 1),
        "sampled_geo_ratio": _action_ratio(sampled_tensor, 2),
        "sampled_ground_ratio": _action_ratio(sampled_tensor, 3),
        "final_policy_local_ratio": _action_ratio(final_tensor, 0),
        "final_policy_neighbor_ratio": _action_ratio(final_tensor, 1),
        "final_policy_geo_ratio": _action_ratio(final_tensor, 2),
        "final_policy_ground_ratio": _action_ratio(final_tensor, 3),
        "prob_local_mean": float(prob_tensor[:, 0].mean()) if prob_tensor.numel() else 0.0,
        "prob_neighbor_mean": float(prob_tensor[:, 1].mean()) if prob_tensor.numel() else 0.0,
        "prob_geo_mean": float(prob_tensor[:, 2].mean()) if prob_tensor.numel() else 0.0,
        "prob_ground_mean": float(prob_tensor[:, 3].mean()) if prob_tensor.numel() else 0.0,
        "prob_local_std": float(prob_tensor[:, 0].std(unbiased=False)) if prob_tensor.numel() else 0.0,
        "prob_neighbor_std": float(prob_tensor[:, 1].std(unbiased=False)) if prob_tensor.numel() else 0.0,
        "prob_geo_std": float(prob_tensor[:, 2].std(unbiased=False)) if prob_tensor.numel() else 0.0,
        "prob_ground_std": float(prob_tensor[:, 3].std(unbiased=False)) if prob_tensor.numel() else 0.0,
        "logit_local_mean": float(logit_tensor[:, 0].mean()) if logit_tensor.numel() else 0.0,
        "logit_neighbor_mean": float(logit_tensor[:, 1].mean()) if logit_tensor.numel() else 0.0,
        "logit_geo_mean": float(logit_tensor[:, 2].mean()) if logit_tensor.numel() else 0.0,
        "logit_ground_mean": float(logit_tensor[:, 3].mean()) if logit_tensor.numel() else 0.0,
        "entropy_mean": float(entropy_tensor.mean()) if entropy_tensor.numel() else 0.0,
        "mask_distribution": describe_mask_distribution(mask_tensor),
        "all_actions_visible_ratio": float(mask_tensor.all(dim=-1).float().mean()) if mask_tensor.numel() else 0.0,
        "tie_break_applied_ratio": float(tie_break_applied / max(1, total_obs)),
        "cost_rank_selected_mean": float(sum(cost_rank_values) / max(1, len(cost_rank_values))),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect deterministic and stochastic upper-policy action distributions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, choices=["trace", "live"])
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=1024)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="balanced_four_tier")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--eval-mode", type=str, default="raw_argmax", choices=["raw_argmax", "stochastic_eval", "margin_cost_tiebreak", "cost_greedy_baseline"])
    parser.add_argument("--tie-eps", type=float, default=0.05)
    parser.add_argument("--stochastic-seed", type=int, default=13)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    batches: List[Dict[str, Any]] = []
    if args.source == "trace":
        groups = load_trace_groups(args.trace, n_leo=args.n_leo, num_states=args.num_states)
        for group in groups:
            obs, edge_index, edge_attr = shared_batch_from_trace_group(
                group,
                node_feature_dim=policy.cfg.scenario.node_feature_dim,
                normalization_mode=policy.obs_normalization_mode,
                normalization_stats=policy.obs_normalization_stats,
            )
            batches.append({"obs": obs, "edge_index": edge_index, "edge_attr": edge_attr, "raw_rows": [dict(row) for row in group]})
    else:
        states = collect_live_states(
            base_url=args.base_url,
            scenario_profile=args.scenario_profile,
            task_source_mode=args.task_source_mode,
            num_states=args.num_states,
            request_timeout=args.request_timeout,
        )
        for state in states:
            obs, edge_index, edge_attr, _, raw_rows = shared_batch_from_live_state(
                state,
                node_feature_dim=policy.cfg.scenario.node_feature_dim,
                normalization_mode=policy.obs_normalization_mode,
                normalization_stats=policy.obs_normalization_stats,
            )
            batches.append({"obs": obs, "edge_index": edge_index, "edge_attr": edge_attr, "raw_rows": raw_rows})

    payload = _summarize(
        policy,
        batches,
        source=args.source,
        eval_mode=args.eval_mode,
        tie_eps=args.tie_eps,
        stochastic_seed=args.stochastic_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
