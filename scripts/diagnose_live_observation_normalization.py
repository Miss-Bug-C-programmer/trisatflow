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

from trisatflow.envs.obs_schema import ACTION_NAMES, FIELD_NAMES
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import collect_live_states
from trisatflow.satedgesim_eval.state_adapter import build_trisatflow_observation


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def _ratio(xs: List[int], target: int) -> float:
    if not xs:
        return 0.0
    return float(sum(1 for x in xs if x == target) / len(xs))


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1.0e-12)
    q = q.clamp_min(1.0e-12)
    return float(torch.sum(p * (p.log() - q.log())).item())


def _dist(actions: List[int]) -> Dict[str, float]:
    return {name: _ratio(actions, i) for i, name in enumerate(ACTION_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose live observation normalization mismatch between legacy and checkpoint modes.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario-profile", type=str, default="mixed_cost_landscape_v2")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--num-states", type=int, default=500)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
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

    n_fields = int(policy.cfg.scenario.node_feature_dim)
    if n_fields > len(FIELD_NAMES):
        n_fields = len(FIELD_NAMES)

    legacy_values: Dict[str, List[float]] = {FIELD_NAMES[i]: [] for i in range(n_fields)}
    checkpoint_values: Dict[str, List[float]] = {FIELD_NAMES[i]: [] for i in range(n_fields)}
    legacy_sat: Dict[str, int] = {FIELD_NAMES[i]: 0 for i in range(n_fields)}
    checkpoint_sat: Dict[str, int] = {FIELD_NAMES[i]: 0 for i in range(n_fields)}

    legacy_actions: List[int] = []
    checkpoint_actions: List[int] = []
    legacy_prob_sum = torch.zeros(4, dtype=torch.float64)
    checkpoint_prob_sum = torch.zeros(4, dtype=torch.float64)
    kl_legacy_to_ckpt: List[float] = []
    kl_ckpt_to_legacy: List[float] = []
    prob_l1_diff: List[float] = []

    rng_legacy = random.Random(13)
    rng_ckpt = random.Random(13)
    action_diff_count = 0

    for state in states:
        obs_legacy, ei_legacy, ea_legacy, src_legacy = build_trisatflow_observation(
            state,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode="legacy",
            normalization_stats=None,
        )
        obs_ckpt, ei_ckpt, ea_ckpt, src_ckpt = build_trisatflow_observation(
            state,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=policy.obs_normalization_mode,
            normalization_stats=policy.obs_normalization_stats,
        )

        d_legacy = policy.inspect_upper_policy(obs_legacy, ei_legacy, ea_legacy)
        d_ckpt = policy.inspect_upper_policy(obs_ckpt, ei_ckpt, ea_ckpt)
        p_legacy = d_legacy["probs"][src_legacy].detach().cpu().float()
        p_ckpt = d_ckpt["probs"][src_ckpt].detach().cpu().float()
        legacy_prob_sum += p_legacy.to(dtype=torch.float64)
        checkpoint_prob_sum += p_ckpt.to(dtype=torch.float64)

        sel_legacy = policy.select_action_from_diagnostics(
            d_legacy,
            source_index=src_legacy,
            raw_rows=list(state.get("denseSourceSummaries") or []),
            eval_mode="raw_argmax",
            tie_break_eps=0.05,
            rng=rng_legacy,
        )
        sel_ckpt = policy.select_action_from_diagnostics(
            d_ckpt,
            source_index=src_ckpt,
            raw_rows=list(state.get("denseSourceSummaries") or []),
            eval_mode="raw_argmax",
            tie_break_eps=0.05,
            rng=rng_ckpt,
        )
        a_legacy = int(sel_legacy["final_action"])
        a_ckpt = int(sel_ckpt["final_action"])
        legacy_actions.append(a_legacy)
        checkpoint_actions.append(a_ckpt)
        action_diff_count += int(a_legacy != a_ckpt)

        kl_legacy_to_ckpt.append(_kl(p_legacy, p_ckpt))
        kl_ckpt_to_legacy.append(_kl(p_ckpt, p_legacy))
        prob_l1_diff.append(float(torch.abs(p_legacy - p_ckpt).mean().item()))

        src_obs_legacy = obs_legacy[src_legacy].detach().cpu().float()
        src_obs_ckpt = obs_ckpt[src_ckpt].detach().cpu().float()
        for i in range(n_fields):
            field = FIELD_NAMES[i]
            v_a = float(src_obs_legacy[i].item())
            v_b = float(src_obs_ckpt[i].item())
            legacy_values[field].append(v_a)
            checkpoint_values[field].append(v_b)
            if v_a <= 1.0e-6 or v_a >= 1.0 - 1.0e-6:
                legacy_sat[field] += 1
            if v_b <= 1.0e-6 or v_b >= 1.0 - 1.0e-6:
                checkpoint_sat[field] += 1

    n = max(1, len(states))
    feature_summary = {}
    for i in range(n_fields):
        field = FIELD_NAMES[i]
        feature_summary[field] = {
            "legacy": _stats(legacy_values[field]),
            "checkpoint": _stats(checkpoint_values[field]),
            "saturation_ratio_legacy": float(legacy_sat[field] / n),
            "saturation_ratio_checkpoint": float(checkpoint_sat[field] / n),
        }

    legacy_dist = _dist(legacy_actions)
    checkpoint_dist = _dist(checkpoint_actions)
    legacy_prob_mean = (legacy_prob_sum / float(n)).tolist()
    checkpoint_prob_mean = (checkpoint_prob_sum / float(n)).tolist()

    legacy_ground = float(legacy_dist.get("ground", 0.0))
    ckpt_ground = float(checkpoint_dist.get("ground", 0.0))
    mismatch_confirmed = (
        (legacy_ground >= 0.85 and ckpt_ground < 0.85)
        or (legacy_ground >= 0.75 and ckpt_ground <= 0.50 and float(action_diff_count / n) >= 0.20)
    )
    normalization_fixed = mismatch_confirmed
    unresolved = not mismatch_confirmed

    payload: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "num_states": len(states),
        "obs_feature_dim": int(policy.cfg.scenario.node_feature_dim),
        "checkpoint_obs_normalization_mode": policy.obs_normalization_mode,
        "checkpoint_obs_normalization_path": policy.obs_normalization_path,
        "checkpoint_obs_normalization_loaded": bool(policy.obs_normalization_loaded),
        "feature_stats_by_field": feature_summary,
        "policy_raw_argmax_distribution_legacy_obs": legacy_dist,
        "policy_raw_argmax_distribution_checkpoint_obs": checkpoint_dist,
        "policy_probability_mean_legacy_obs": {
            ACTION_NAMES[i]: float(legacy_prob_mean[i]) for i in range(4)
        },
        "policy_probability_mean_checkpoint_obs": {
            ACTION_NAMES[i]: float(checkpoint_prob_mean[i]) for i in range(4)
        },
        "probability_difference": {
            "mean_kl_legacy_to_checkpoint": float(sum(kl_legacy_to_ckpt) / max(1, len(kl_legacy_to_ckpt))),
            "mean_kl_checkpoint_to_legacy": float(sum(kl_ckpt_to_legacy) / max(1, len(kl_ckpt_to_legacy))),
            "mean_l1_prob_diff": float(sum(prob_l1_diff) / max(1, len(prob_l1_diff))),
        },
        "selected_action_difference_ratio": float(action_diff_count / n),
        "conclusion": {
            "train_live_normalization_mismatch_confirmed": bool(mismatch_confirmed),
            "normalization_fixed": bool(normalization_fixed),
            "unresolved": bool(unresolved),
            "note": (
                "legacy obs is ground-heavy while checkpoint-normalized obs is not"
                if mismatch_confirmed
                else "both modes still similar or ground-heavy; continue with live distribution-shift diagnosis"
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
