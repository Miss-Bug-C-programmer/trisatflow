from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


def _ratio(actions: torch.Tensor, action: int) -> float:
    if actions.numel() == 0:
        return 0.0
    return float((actions == action).float().mean())


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q), dtype=torch.float32)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose deterministic argmax tie bias on trace states.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    groups = load_trace_groups(
        args.trace,
        n_leo=args.n_leo,
        num_states=max(1, args.num_states // max(1, args.n_leo)),
    )

    argmax_actions = []
    sampled_actions = []
    probs_all = []
    logits_all = []
    entropy_all = []

    top1_prob_margins: List[float] = []
    top1_logit_margins: List[float] = []
    top1_counter = [0, 0, 0, 0]
    top2_counter = [0, 0, 0, 0]
    near_tie_001 = 0
    near_tie_002 = 0
    near_tie_005 = 0
    near_tie_010 = 0

    rng = random.Random(13)
    processed = 0
    for group in groups:
        obs, edge_index, edge_attr = shared_batch_from_trace_group(group, node_feature_dim=policy.cfg.scenario.node_feature_dim)
        d = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        probs = d["probs"].detach().cpu()
        logits = d["masked_logits"].detach().cpu()
        mask = d["mask"].detach().cpu().bool()
        argmax = d["argmax_action"].detach().cpu()
        sampled = d["sampled_action"].detach().cpu()

        argmax_actions.append(argmax)
        sampled_actions.append(sampled)
        probs_all.append(probs)
        logits_all.append(logits)
        entropy_all.append(d["entropy"].detach().cpu())

        for i in range(probs.shape[0]):
            feasible = [idx for idx in range(4) if bool(mask[i, idx])]
            if not feasible:
                feasible = [0]
            sorted_actions = sorted(feasible, key=lambda idx: float(probs[i, idx]), reverse=True)
            top1 = sorted_actions[0]
            top2 = sorted_actions[1] if len(sorted_actions) > 1 else sorted_actions[0]
            top1_counter[top1] += 1
            top2_counter[top2] += 1
            top1_prob = float(probs[i, top1])
            top2_prob = float(probs[i, top2])
            top1_logit = float(logits[i, top1])
            top2_logit = float(logits[i, top2])
            prob_margin = top1_prob - top2_prob
            logit_margin = top1_logit - top2_logit
            top1_prob_margins.append(prob_margin)
            top1_logit_margins.append(logit_margin)
            near_tie_001 += int(prob_margin <= 0.01 + 1.0e-12)
            near_tie_002 += int(prob_margin <= 0.02 + 1.0e-12)
            near_tie_005 += int(prob_margin <= 0.05 + 1.0e-12)
            near_tie_010 += int(prob_margin <= 0.10 + 1.0e-12)
            processed += 1

        if processed >= args.num_states:
            break

    argmax_t = torch.cat(argmax_actions, dim=0) if argmax_actions else torch.empty(0, dtype=torch.long)
    sampled_t = torch.cat(sampled_actions, dim=0) if sampled_actions else torch.empty(0, dtype=torch.long)
    probs_t = torch.cat(probs_all, dim=0) if probs_all else torch.empty((0, 4), dtype=torch.float32)
    logits_t = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0, 4), dtype=torch.float32)
    entropy_t = torch.cat(entropy_all, dim=0) if entropy_all else torch.empty(0, dtype=torch.float32)

    total = int(max(1, processed))
    argmax_dist = {name: _ratio(argmax_t, idx) for idx, name in enumerate(ACTION_NAMES)}
    sampled_dist = {name: _ratio(sampled_t, idx) for idx, name in enumerate(ACTION_NAMES)}
    prob_mean = {name: float(probs_t[:, idx].mean()) if probs_t.numel() else 0.0 for idx, name in enumerate(ACTION_NAMES)}
    logit_mean = {name: float(logits_t[:, idx].mean()) if logits_t.numel() else 0.0 for idx, name in enumerate(ACTION_NAMES)}

    top1_dist = {name: float(top1_counter[idx] / total) for idx, name in enumerate(ACTION_NAMES)}
    top2_dist = {name: float(top2_counter[idx] / total) for idx, name in enumerate(ACTION_NAMES)}

    near_tie_ratio_001 = float(near_tie_001 / total)
    near_tie_ratio_002 = float(near_tie_002 / total)
    near_tie_ratio_005 = float(near_tie_005 / total)
    near_tie_ratio_010 = float(near_tie_010 / total)

    entropy_mean = float(entropy_t.mean()) if entropy_t.numel() else 0.0
    max_argmax_ratio = max(argmax_dist.values()) if argmax_dist else 0.0
    max_sampled_ratio = max(sampled_dist.values()) if sampled_dist else 0.0
    geo_prob_adv = prob_mean.get("geo", 0.0) - max(prob_mean.get("local", 0.0), prob_mean.get("neighbor", 0.0), prob_mean.get("ground", 0.0))
    logit_std = {name: float(logits_t[:, idx].std(unbiased=False)) if logits_t.numel() else 0.0 for idx, name in enumerate(ACTION_NAMES)}

    classification = "unresolved"
    if max_argmax_ratio > 0.98 and near_tie_ratio_005 > 0.60 and entropy_mean > 1.1 and max_sampled_ratio < 0.80 and geo_prob_adv < 0.08:
        classification = "near_tie_argmax_artifact"
    elif max_argmax_ratio > 0.98 and _quantile(top1_prob_margins, 0.50) >= 0.05 and near_tie_ratio_005 < 0.50:
        classification = "true_deterministic_collapse"
    elif max_argmax_ratio > 0.98 and max(logit_std.values()) < 1.0e-2:
        classification = "checkpoint_or_logit_bias"

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "num_states": int(processed),
        "argmax_action_distribution": argmax_dist,
        "sampled_action_distribution": sampled_dist,
        "prob_mean_by_action": prob_mean,
        "logit_mean_by_action": logit_mean,
        "entropy_mean": entropy_mean,
        "top1_prob_margin_mean": float(sum(top1_prob_margins) / max(1, len(top1_prob_margins))),
        "top1_prob_margin_p50": _quantile(top1_prob_margins, 0.50),
        "top1_prob_margin_p90": _quantile(top1_prob_margins, 0.90),
        "top1_logit_margin_mean": float(sum(top1_logit_margins) / max(1, len(top1_logit_margins))),
        "top1_logit_margin_p50": _quantile(top1_logit_margins, 0.50),
        "top1_logit_margin_p90": _quantile(top1_logit_margins, 0.90),
        "near_tie_ratio_eps_0_01": near_tie_ratio_001,
        "near_tie_ratio_eps_0_02": near_tie_ratio_002,
        "near_tie_ratio_eps_0_05": near_tie_ratio_005,
        "near_tie_ratio_eps_0_10": near_tie_ratio_010,
        "top1_action_ratio": top1_dist,
        "top2_action_ratio": top2_dist,
        "classification": classification,
        "logit_std_by_action": logit_std,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
