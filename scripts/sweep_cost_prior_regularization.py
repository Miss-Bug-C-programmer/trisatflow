from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import load_config, save_config
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


def _distribution(actions: torch.Tensor) -> Dict[str, float]:
    if actions.numel() == 0:
        return {name: 0.0 for name in ACTION_NAMES}
    return {name: float((actions == i).float().mean()) for i, name in enumerate(ACTION_NAMES)}


def _mi(x: List[int], y: List[int]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    n = float(len(x))
    px: Dict[int, int] = {}
    py: Dict[int, int] = {}
    pxy: Dict[Tuple[int, int], int] = {}
    for a, b in zip(x, y):
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
        pxy[(a, b)] = pxy.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), c in pxy.items():
        p_ab = c / n
        p_a = px[a] / n
        p_b = py[b] / n
        mi += p_ab * math.log(max(1.0e-12, p_ab / max(1.0e-12, p_a * p_b)))
    return float(mi)


def _tier_cost(row: Mapping[str, Any], tier: str) -> float:
    delay = max(0.0, _to_float(row.get(f"{tier}_delay", row.get(f"{tier}_best_delay")), 0.0))
    queue = max(0.0, _to_float(row.get(f"{tier}_queue", row.get(f"{tier}_best_queue")), 0.0))
    rate = max(1.0e-6, _to_float(row.get(f"{tier}_rate"), 0.0))
    tx = 0.0 if tier == "local" else (1.0 / rate)
    compute = max(0.0, delay - tx)
    return delay + 0.5 * queue + 0.2 * tx + 0.2 * compute


def _oracle_action(row: Mapping[str, Any]) -> int:
    tiers = ["local", "neighbor", "geo", "ground"]
    costs: List[float] = []
    for t in tiers:
        vis = row.get(f"{t}_visible", row.get(f"{t}Visible"))
        visible = True if t == "local" else bool(vis)
        if isinstance(vis, str):
            visible = vis.strip().lower() in {"1", "true", "yes", "y"}
        if not visible:
            costs.append(float("inf"))
        else:
            costs.append(_tier_cost(row, t))
    return int(min(range(4), key=lambda i: costs[i]))


def _classify(raw_dist: Dict[str, float], logit_std_mean: float, prob_std_mean: float, mi_phase: float, raw_oracle: float) -> str:
    if max(raw_dist.values()) > 0.999 and mi_phase <= 1.0e-8:
        return "constant_policy_with_global_geo_bias"
    if max(raw_dist.values()) > 0.999 and (logit_std_mean > 0.01 or prob_std_mean > 0.002):
        return "state_conditioned_but_argmax_biased"
    if raw_oracle > 0.55 and mi_phase > 0.05:
        return "policy_matches_oracle_conditionally"
    return "weak_state_conditioning"


def _load_norm_from_checkpoint_cfg(policy: FrozenTriSatFlowPolicy) -> tuple[str, Dict[str, Any] | None]:
    mode = str(getattr(policy.cfg.scenario, "obs_normalization_mode", "legacy") or "legacy")
    path = str(getattr(policy.cfg.scenario, "obs_normalization_path", "") or "").strip()
    if not path:
        return mode, None
    p = Path(path)
    if not p.exists():
        return mode, None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return mode, dict(payload.get("fields") or payload)
    return mode, None


def _diagnose_checkpoint(checkpoint: Path, trace: str, n_leo: int, num_states: int) -> Dict[str, Any]:
    policy = FrozenTriSatFlowPolicy(checkpoint, device="cpu")
    norm_mode, norm_stats = _load_norm_from_checkpoint_cfg(policy)
    groups = load_trace_groups(trace, n_leo=n_leo, num_states=max(1, num_states // max(1, n_leo)))
    raw_actions: List[int] = []
    stochastic_actions: List[int] = []
    oracle_actions: List[int] = []
    probs_oracle: List[float] = []
    entropy_vals: List[float] = []
    phase_seq: List[int] = []
    phase_map: Dict[str, int] = {}
    logits_all: List[torch.Tensor] = []
    probs_all: List[torch.Tensor] = []
    total = 0
    rng = random.Random(13)
    for group in groups:
        if total >= num_states:
            break
        obs, edge_index, edge_attr = shared_batch_from_trace_group(
            group,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=norm_mode,
            normalization_stats=norm_stats,
        )
        d = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        probs = d["probs"].detach().cpu()
        logits = d["logits"].detach().cpu()
        entropy = d["entropy"].detach().cpu()
        for i, row in enumerate(group):
            if total >= num_states:
                break
            total += 1
            phase = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            if phase not in phase_map:
                phase_map[phase] = len(phase_map)
            phase_seq.append(phase_map[phase])
            oracle = _oracle_action(row)
            raw = int(torch.argmax(probs[i]).item())
            raw_actions.append(raw)
            oracle_actions.append(oracle)
            st = policy.select_action_from_diagnostics(
                d,
                source_index=i,
                raw_rows=[dict(r) for r in group],
                eval_mode="stochastic_eval",
                tie_break_eps=0.05,
                rng=rng,
            )
            stochastic_actions.append(int(st["final_action"]))
            probs_oracle.append(float(probs[i, oracle].item()))
            entropy_vals.append(float(entropy[i].item()))
            logits_all.append(logits[i])
            probs_all.append(probs[i])
        if total >= num_states:
            break
    raw_t = torch.tensor(raw_actions, dtype=torch.long)
    stochastic_t = torch.tensor(stochastic_actions, dtype=torch.long)
    oracle_t = torch.tensor(oracle_actions, dtype=torch.long)
    logits_t = torch.stack(logits_all, dim=0) if logits_all else torch.zeros((0, 4))
    probs_t = torch.stack(probs_all, dim=0) if probs_all else torch.zeros((0, 4))
    raw_oracle_agreement = float((raw_t == oracle_t).float().mean()) if raw_t.numel() else 0.0
    logit_std_mean = float(logits_t.std(dim=0, unbiased=False).mean()) if logits_t.numel() else 0.0
    prob_std_mean = float(probs_t.std(dim=0, unbiased=False).mean()) if probs_t.numel() else 0.0
    mi_phase = _mi(phase_seq, raw_actions)
    raw_dist = _distribution(raw_t)
    return {
        "num_states": int(total),
        "raw_argmax_distribution": raw_dist,
        "raw_argmax_oracle_agreement": raw_oracle_agreement,
        "stochastic_oracle_agreement": float((stochastic_t == oracle_t).float().mean()) if stochastic_t.numel() else 0.0,
        "prob_oracle_action_mean": float(sum(probs_oracle) / max(1, len(probs_oracle))),
        "entropy_mean": float(sum(entropy_vals) / max(1, len(entropy_vals))),
        "mutual_information_phase_argmax": mi_phase,
        "logit_std_mean": logit_std_mean,
        "prob_std_mean": prob_std_mean,
        "classification": _classify(raw_dist, logit_std_mean, prob_std_mean, mi_phase, raw_oracle_agreement),
    }


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_text_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep cost_prior_ce weight/temperature under upper-only per-agent training.")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--temperatures", type=str, required=True)
    parser.add_argument("--entropy-coefs", type=str, default="0.02")
    parser.add_argument("--entropy-schedules", type=str, default="constant")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-states", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None, help="Override config device: cpu|cuda|cuda:0|auto")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    args = parser.parse_args()

    weights = _parse_float_list(args.weights)
    temps = _parse_float_list(args.temperatures)
    entropy_coefs = _parse_float_list(args.entropy_coefs)
    entropy_schedules = _parse_text_list(args.entropy_schedules)
    base = load_config(args.base_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for w in weights:
        for t in temps:
            for ent in entropy_coefs:
                for ent_sched in entropy_schedules:
                    sched = "" if ent_sched == "constant" else ent_sched
                    run_name = f"w{w:g}_t{t:g}_e{ent:g}_es{ent_sched}".replace(".", "p")
                    run_root = output_root / run_name
                    cfg = deepcopy(base)
                    cfg.scenario.n_leo = int(args.n_leo)
                    cfg.scenario.seed = int(args.seed)
                    cfg.total_episodes = int(args.episodes)
                    cfg.scenario.episode_len = int(args.steps)
                    cfg.algo.upper_algo = "mappo"
                    cfg.algo.lower_algo = "maddpg"
                    cfg.algo.credit_assignment = "per_agent"
                    cfg.algo.policy_head = "hybrid_gnn_cost"
                    cfg.algo.logit_centering = True
                    cfg.algo.action_bias_regularization = 0.01
                    cfg.algo.entropy_coef = float(ent)
                    cfg.algo.entropy_coef_schedule = str(sched)
                    cfg.policy_regularization.enabled = True
                    cfg.policy_regularization.mode = "cost_prior_ce"
                    cfg.policy_regularization.weight = float(w)
                    cfg.policy_regularization.temperature = float(t)
                    cfg.lower_training_enabled = False
                    cfg.lower_action_mode = "neutral_allocator"
                    cfg.scenario.include_cost_features_in_obs = True
                    if args.device is not None:
                        cfg.device = str(args.device)
                    cfg.output_dir = str(run_root / "seed_13" / "upper_mappo__lower_maddpg")
                    save_config(cfg, run_root / "resolved_config.yaml")

                    status = "ok"
                    error = ""
                    try:
                        trainer = HierarchicalTrainer(cfg)
                        history = trainer.train()
                        ckpt = Path(cfg.output_dir) / "checkpoint.pt"
                        trainer.save_checkpoint(ckpt)
                        diag = _diagnose_checkpoint(ckpt, cfg.scenario.topology_trace_path, args.n_leo, args.num_states)
                        policy_cost_prior_agreement = 0.0
                        if history:
                            tail = history[-min(5, len(history)) :]
                            policy_cost_prior_agreement = float(
                                sum(_to_float(item.get("policy_cost_prior_agreement"), 0.0) for item in tail) / max(1, len(tail))
                            )
                        row = {
                            "status": status,
                            "error": error,
                            "weight": float(w),
                            "temperature": float(t),
                            "entropy_coef": float(ent),
                            "entropy_schedule": str(ent_sched),
                            "output_dir": cfg.output_dir,
                            "requested_device": getattr(cfg, "requested_device", cfg.device),
                            "actual_device": getattr(cfg, "actual_device", cfg.device),
                            "device_fallback_reason": getattr(cfg, "device_fallback_reason", ""),
                            "raw_argmax_distribution": json.dumps(diag["raw_argmax_distribution"], ensure_ascii=False),
                            "raw_argmax_oracle_agreement": diag["raw_argmax_oracle_agreement"],
                            "stochastic_oracle_agreement": diag["stochastic_oracle_agreement"],
                            "prob_oracle_action_mean": diag["prob_oracle_action_mean"],
                            "entropy_mean": diag["entropy_mean"],
                            "mutual_information_phase_argmax": diag["mutual_information_phase_argmax"],
                            "logit_std_mean": diag["logit_std_mean"],
                            "prob_std_mean": diag["prob_std_mean"],
                            "policy_cost_prior_agreement": policy_cost_prior_agreement,
                            "classification": diag["classification"],
                        }
                    except Exception as exc:
                        status = "failed"
                        error = repr(exc)
                        row = {
                            "status": status,
                            "error": error,
                            "weight": float(w),
                            "temperature": float(t),
                            "entropy_coef": float(ent),
                            "entropy_schedule": str(ent_sched),
                            "output_dir": str(run_root),
                            "requested_device": str(args.device or base.device),
                            "actual_device": str(args.device or base.device),
                            "device_fallback_reason": "",
                            "raw_argmax_distribution": "{}",
                            "raw_argmax_oracle_agreement": 0.0,
                            "stochastic_oracle_agreement": 0.0,
                            "prob_oracle_action_mean": 0.0,
                            "entropy_mean": 0.0,
                            "mutual_information_phase_argmax": 0.0,
                            "logit_std_mean": 0.0,
                            "prob_std_mean": 0.0,
                            "policy_cost_prior_agreement": 0.0,
                            "classification": "failed",
                        }
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    fieldnames = [
        "status",
        "error",
        "weight",
        "temperature",
        "entropy_coef",
        "entropy_schedule",
        "output_dir",
        "requested_device",
        "actual_device",
        "device_fallback_reason",
        "raw_argmax_distribution",
        "raw_argmax_oracle_agreement",
        "stochastic_oracle_agreement",
        "prob_oracle_action_mean",
        "entropy_mean",
        "mutual_information_phase_argmax",
        "logit_std_mean",
        "prob_std_mean",
        "policy_cost_prior_agreement",
        "classification",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"SWEEP_DONE rows={len(rows)} summary={summary_path}")


if __name__ == "__main__":
    main()
