from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.models import CentralValue
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


TRAIN_FIELDS = [
    "upper_policy_loss",
    "upper_value_loss",
    "upper_entropy",
    "upper_approx_kl",
    "upper_clip_fraction",
    "upper_grad_norm",
    "upper_advantage_mean",
    "upper_advantage_std",
    "upper_return_mean",
    "upper_return_std",
    "upper_value_mean",
    "upper_value_std",
    "upper_ratio_mean",
    "upper_ratio_std",
    "upper_old_logprob_mean",
    "upper_new_logprob_mean",
]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _series_stats(rows: List[Dict[str, str]], key: str) -> Dict[str, float] | None:
    vals = [_to_float(r.get(key), float("nan")) for r in rows if r.get(key) not in (None, "")]
    vals = [v for v in vals if v == v]
    if not vals:
        return None
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "last": float(t[-1]),
    }


def _action_ratio(actions: torch.Tensor, action_idx: int) -> float:
    if actions.numel() == 0:
        return 0.0
    return float((actions == action_idx).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MAPPO upper-layer training dynamics and collapse signals.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint.pt not found: {checkpoint_path}")

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("metrics.csv is empty")

    policy = FrozenTriSatFlowPolicy(checkpoint_path, device="cpu")
    payload = torch.load(checkpoint_path, map_location="cpu")

    groups = load_trace_groups(
        args.trace,
        n_leo=args.n_leo,
        num_states=max(1, args.num_states // max(1, args.n_leo)),
    )
    batches: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for group in groups:
        batches.append(shared_batch_from_trace_group(group, node_feature_dim=policy.cfg.scenario.node_feature_dim))
        if len(batches) * args.n_leo >= args.num_states:
            break
    if not batches:
        raise RuntimeError("no trace groups available for diagnosis")

    # Optional direct-load mismatch check.
    encoder2 = policy._build_encoder().to(policy.device)  # type: ignore[attr-defined]
    actor2 = policy._build_upper_actor().to(policy.device)  # type: ignore[attr-defined]
    encoder2.load_state_dict(payload["encoder"])
    actor2.load_state_dict(payload["upper_actor"])
    encoder2.eval()
    actor2.eval()

    critic_value_stats: List[float] = []
    critic = None
    if "upper_critic" in payload:
        critic = CentralValue(
            policy.cfg.algo.gnn_hidden_dim,
            policy.cfg.scenario.n_leo,
            policy.cfg.algo.policy_hidden_dim,
        ).to(policy.device)
        critic.load_state_dict(payload["upper_critic"])
        critic.eval()

    argmax_actions = []
    sampled_actions = []
    logits = []
    probs = []
    entropy = []
    masks = []
    max_reload_logit_diff = 0.0
    for obs, edge_index, edge_attr in batches:
        d = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        argmax_actions.append(d["argmax_action"].detach().cpu())
        sampled_actions.append(d["sampled_action"].detach().cpu())
        logits.append(d["masked_logits"].detach().cpu())
        probs.append(d["probs"].detach().cpu())
        entropy.append(d["entropy"].detach().cpu())
        masks.append(d["mask"].detach().cpu())

        with torch.no_grad():
            embed = policy.encoder(obs.to(policy.device), edge_index.to(policy.device), edge_attr.to(policy.device))
            if critic is not None:
                v = critic(embed)
                critic_value_stats.append(float(v.detach().cpu()))

            logits_reload = actor2.net(encoder2(obs.to(policy.device), edge_index.to(policy.device), edge_attr.to(policy.device)))
            mask_reload = d["mask"].to(policy.device)
            logits_reload = logits_reload.masked_fill(~mask_reload, torch.finfo(logits_reload.dtype).min / 4)
            diff = torch.max(torch.abs(logits_reload.detach().cpu() - d["masked_logits"].detach().cpu()))
            max_reload_logit_diff = max(max_reload_logit_diff, float(diff))

    argmax_t = torch.cat(argmax_actions, dim=0)
    sampled_t = torch.cat(sampled_actions, dim=0)
    logit_t = torch.cat(logits, dim=0)
    prob_t = torch.cat(probs, dim=0)
    entropy_t = torch.cat(entropy, dim=0)
    mask_t = torch.cat(masks, dim=0)

    training_series: Dict[str, Any] = {}
    missing_training_fields: List[str] = []
    for key in TRAIN_FIELDS:
        stats = _series_stats(rows, key)
        if stats is None:
            missing_training_fields.append(key)
        training_series[key] = stats

    # Per-action log/reward/value summaries from metrics tail if available.
    def _tail_mean(field: str, window: int = 10) -> float:
        tail = rows[-window:]
        vals = [_to_float(r.get(field), float("nan")) for r in tail]
        vals = [v for v in vals if v == v]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    per_action_summary = {}
    for idx, name in enumerate(ACTION_NAMES):
        per_action_summary[name] = {
            "logit_mean": float(logit_t[:, idx].mean()),
            "logit_std": float(logit_t[:, idx].std(unbiased=False)),
            "prob_mean": float(prob_t[:, idx].mean()),
            "prob_std": float(prob_t[:, idx].std(unbiased=False)),
            "argmax_ratio": _action_ratio(argmax_t, idx),
            "sampled_ratio": _action_ratio(sampled_t, idx),
            "mean_advantage_when_selected": _tail_mean(f"mean_advantage_{name}_selected"),
            "mean_reward_when_selected": _tail_mean(f"mean_reward_{name}_selected"),
            "mean_return_when_selected": _tail_mean(f"mean_return_{name}_selected"),
            "mean_value_when_selected": _tail_mean(f"mean_value_{name}_selected"),
        }

    # Failure checks
    geo_logit_mean = per_action_summary["geo"]["logit_mean"]
    other_logits = [per_action_summary[name]["logit_mean"] for name in ACTION_NAMES if name != "geo"]
    geo_adv = per_action_summary["geo"]["mean_advantage_when_selected"]
    other_adv = [per_action_summary[name]["mean_advantage_when_selected"] for name in ACTION_NAMES if name != "geo"]
    kl_last = (training_series.get("upper_approx_kl") or {}).get("last", 0.0) if isinstance(training_series.get("upper_approx_kl"), dict) else 0.0
    clip_last = (training_series.get("upper_clip_fraction") or {}).get("last", 0.0) if isinstance(training_series.get("upper_clip_fraction"), dict) else 0.0
    ratio_mean_last = (training_series.get("upper_ratio_mean") or {}).get("last", 0.0) if isinstance(training_series.get("upper_ratio_mean"), dict) else 0.0
    ratio_std_last = (training_series.get("upper_ratio_std") or {}).get("last", 0.0) if isinstance(training_series.get("upper_ratio_std"), dict) else 0.0
    ent_last = (training_series.get("upper_entropy") or {}).get("last", 0.0) if isinstance(training_series.get("upper_entropy"), dict) else 0.0
    old_lp_last = (training_series.get("upper_old_logprob_mean") or {}).get("last", 0.0) if isinstance(training_series.get("upper_old_logprob_mean"), dict) else 0.0
    new_lp_last = (training_series.get("upper_new_logprob_mean") or {}).get("last", 0.0) if isinstance(training_series.get("upper_new_logprob_mean"), dict) else 0.0
    value_std = float(torch.tensor(critic_value_stats).std(unbiased=False)) if critic_value_stats else 0.0

    checks = {
        "geo_logit_bias": geo_logit_mean > max(other_logits) + 0.5,
        "geo_advantage_bias": geo_adv > max(other_adv) + 0.05,
        "value_collapse": value_std < 1.0e-4,
        "entropy_not_applied": ent_last < 0.05 and float(policy.cfg.algo.entropy_coef) > 0.0,
        "ratio_explosion": ratio_mean_last > 2.0 or ratio_std_last > 1.0,
        "kl_explosion": kl_last > 0.2,
        "clip_fraction_abnormal": clip_last > 0.8,
        "old_new_logprob_mismatch": abs(new_lp_last - old_lp_last) > 3.0,
        "mask_update_mismatch": float(mask_t.all(dim=-1).float().mean()) < 0.95
        and float(sum(per_action_summary[name]["argmax_ratio"] for name in ACTION_NAMES)) > 0.0,
        "checkpoint_load_mismatch": max_reload_logit_diff > 1.0e-6,
    }

    output = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_path),
        "num_metric_rows": len(rows),
        "num_states": len(batches),
        "num_agent_observations": int(mask_t.shape[0]),
        "policy_summary": {
            "entropy_mean": float(entropy_t.mean()),
            "all_actions_visible_ratio": float(mask_t.all(dim=-1).float().mean()),
        },
        "critic_summary": {
            "value_mean": float(torch.tensor(critic_value_stats).mean()) if critic_value_stats else 0.0,
            "value_std": value_std,
            "value_min": float(torch.tensor(critic_value_stats).min()) if critic_value_stats else 0.0,
            "value_max": float(torch.tensor(critic_value_stats).max()) if critic_value_stats else 0.0,
        },
        "training_series": training_series,
        "missing_training_fields": missing_training_fields,
        "per_action_summary": per_action_summary,
        "checks": checks,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
