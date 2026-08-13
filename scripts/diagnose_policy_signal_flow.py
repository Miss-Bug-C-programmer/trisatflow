from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_builder import build_shared_observation, canonical_row
from trisatflow.envs.obs_schema import ACTION_NAMES, FIELD_NAMES, SHARED_NODE_FEATURE_DIM, SHARED_NODE_FEATURE_DIM_WITH_COST
from trisatflow.models import upper_action_mask_from_obs
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _corr(xs: Iterable[float], ys: Iterable[float]) -> float:
    xv = torch.tensor(list(xs), dtype=torch.float32)
    yv = torch.tensor(list(ys), dtype=torch.float32)
    if xv.numel() < 2 or yv.numel() != xv.numel():
        return 0.0
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = torch.sqrt((xv.pow(2).sum() * yv.pow(2).sum()).clamp_min(1.0e-12))
    return float((xv * yv).sum() / denom)


def _pairwise_distance_mean(x: torch.Tensor, max_pairs: int = 4096) -> float:
    if x.ndim != 2 or x.shape[0] <= 1:
        return 0.0
    n = x.shape[0]
    rng = torch.Generator().manual_seed(13)
    num = min(max_pairs, n * (n - 1) // 2)
    if num <= 0:
        return 0.0
    idx_i = torch.randint(0, n, (num,), generator=rng)
    idx_j = torch.randint(0, n, (num,), generator=rng)
    valid = idx_i != idx_j
    if not bool(valid.any()):
        return 0.0
    diff = x[idx_i[valid]] - x[idx_j[valid]]
    return float(torch.norm(diff, p=2, dim=-1).mean())


def _group_separability(features: torch.Tensor, labels: List[str]) -> float:
    if features.ndim != 2 or features.shape[0] == 0 or len(labels) != features.shape[0]:
        return 0.0
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[str(label)].append(i)
    if len(groups) <= 1:
        return 0.0
    means: List[torch.Tensor] = []
    within_vars: List[torch.Tensor] = []
    for idxs in groups.values():
        sub = features[idxs]
        means.append(sub.mean(dim=0))
        within_vars.append(sub.std(dim=0, unbiased=False).mean())
    between = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            between.append(torch.norm(means[i] - means[j], p=2))
    between_mean = torch.stack(between).mean() if between else torch.tensor(0.0)
    within_mean = torch.stack(within_vars).mean() if within_vars else torch.tensor(0.0)
    return float(between_mean / (within_mean + 1.0e-6))


def _normalized_cost_from_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM_WITH_COST:
        return obs[:, 16:20]
    if obs.shape[-1] < SHARED_NODE_FEATURE_DIM:
        return torch.ones((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
    rates = obs[:, 4:8]
    delays = obs[:, 8:12]
    queues = obs[:, 12:16]
    visible = obs[:, :4] > 0.5
    tx = torch.zeros_like(rates)
    tx[:, 1:] = 1.0 / rates[:, 1:].clamp_min(1.0e-6)
    compute = torch.relu(delays - tx)
    raw = delays + 0.5 * queues + 0.2 * tx + 0.2 * compute
    raw = raw.masked_fill(~visible, torch.finfo(raw.dtype).max / 4)
    row_min = raw.min(dim=-1, keepdim=True).values
    row_max = raw.max(dim=-1, keepdim=True).values
    norm = (raw - row_min) / (row_max - row_min).clamp_min(1.0e-6)
    return torch.where(visible, norm, torch.ones_like(norm))


def _classify(
    *,
    phase_sep_obs: float,
    phase_sep_embed: float,
    phase_sep_logits: float,
    raw_argmax_geo_ratio: float,
    logit_mean: Dict[str, float],
    saturation: Dict[str, float],
) -> str:
    if phase_sep_obs < 0.08:
        return "obs_signal_too_weak"
    if any(v > 0.30 for k, v in saturation.items() if "delay" in k or "rate" in k):
        return "normalization_saturates_features"
    if phase_sep_obs >= 0.12 and phase_sep_embed < 0.05:
        return "encoder_washes_out_state_signal"
    if phase_sep_embed >= 0.10 and phase_sep_logits < 0.05:
        return "policy_head_washes_out_state_signal"
    dominant_logit = max(logit_mean.values()) if logit_mean else 0.0
    min_logit = min(logit_mean.values()) if logit_mean else 0.0
    if raw_argmax_geo_ratio > 0.98 and (dominant_logit - min_logit) > 0.08:
        return "logits_have_global_bias"
    return "state_signal_flow_ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose obs->embedding->logits signal flow for upper policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--oracle", type=str, default="")
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=8192)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    policy = FrozenTriSatFlowPolicy(args.checkpoint, device="cpu")
    norm_mode, norm_stats = _load_norm_from_checkpoint_cfg(policy)
    groups = load_trace_groups(args.trace, n_leo=args.n_leo, num_states=max(1, args.num_states // max(1, args.n_leo)))

    raw_obs_rows: List[torch.Tensor] = []
    normalized_obs_rows: List[torch.Tensor] = []
    embed_rows: List[torch.Tensor] = []
    hidden_rows: List[torch.Tensor] = []
    logits_rows: List[torch.Tensor] = []
    probs_rows: List[torch.Tensor] = []
    phase_labels: List[str] = []
    oracle_labels: List[str] = []
    raw_argmax: List[int] = []

    phase_group_logits: Dict[str, List[torch.Tensor]] = defaultdict(list)
    oracle_group_logits: Dict[str, List[torch.Tensor]] = defaultdict(list)
    phase_group_embed: Dict[str, List[torch.Tensor]] = defaultdict(list)
    phase_group_hidden: Dict[str, List[torch.Tensor]] = defaultdict(list)

    corr_norm_cost_logit: Dict[str, List[float]] = {name: [] for name in ACTION_NAMES}
    corr_norm_cost_logit_y: Dict[str, List[float]] = {name: [] for name in ACTION_NAMES}
    corr_neg_cost_prob_x: Dict[str, List[float]] = {name: [] for name in ACTION_NAMES}
    corr_neg_cost_prob_y: Dict[str, List[float]] = {name: [] for name in ACTION_NAMES}

    total = 0
    for group in groups:
        if total >= args.num_states:
            break
        # policy input (normalized obs)
        obs, edge_index, edge_attr = shared_batch_from_trace_group(
            group,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=norm_mode,
            normalization_stats=norm_stats,
        )
        obs = obs.detach().cpu()
        d = policy.inspect_upper_policy(obs, edge_index, edge_attr)
        logits = d["logits"].detach().cpu()
        probs = d["probs"].detach().cpu()
        embeds = d.get("embed")
        hidden = d.get("policy_hidden")
        if embeds is None or hidden is None:
            with torch.no_grad():
                obs_dev = obs.to(policy.device)
                edge_index_dev = edge_index.to(policy.device)
                edge_attr_dev = edge_attr.to(policy.device)
                embed_dev = policy.encoder(obs_dev, edge_index_dev, edge_attr_dev)
                logits_dev, details = policy.upper_actor.compute_logits(embed_dev, obs=obs_dev, return_details=True)
                embeds = embed_dev.detach().cpu()
                hidden = details["policy_hidden"].detach().cpu()
                logits = logits_dev.detach().cpu()
                mask = upper_action_mask_from_obs(obs_dev).detach().cpu()
                logits_masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
                probs = torch.softmax(logits_masked, dim=-1)
        else:
            embeds = embeds.detach().cpu()
            hidden = hidden.detach().cpu()

        # raw canonical observations
        canon = [canonical_row(row) for row in group]
        raw_fields = []
        for row in canon:
            values = [float(row.get(f, 0.0)) for f in FIELD_NAMES]
            raw_fields.append(values)
        raw_tensor = torch.tensor(raw_fields, dtype=torch.float32)

        # normalized observation reconstructed with checkpoint normalization setup.
        normalized_batch = build_shared_observation(
            group,
            source_index=0,
            node_feature_dim=policy.cfg.scenario.node_feature_dim,
            normalization_mode=norm_mode,
            normalization_stats=norm_stats,
        ).obs.detach().cpu()

        norm_cost = _normalized_cost_from_obs(normalized_batch)
        for i, row in enumerate(group):
            if total >= args.num_states:
                break
            total += 1
            phase = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            oracle = str(row.get("oracle_action", row.get("oracleAction", ""))).strip().lower()
            if oracle not in ACTION_NAMES:
                oracle_idx = int(torch.argmin(norm_cost[i]).item())
                oracle = ACTION_NAMES[oracle_idx]
            raw_obs_rows.append(raw_tensor[i])
            normalized_obs_rows.append(normalized_batch[i])
            embed_rows.append(embeds[i])
            hidden_rows.append(hidden[i])
            logits_rows.append(logits[i])
            probs_rows.append(probs[i])
            phase_labels.append(phase)
            oracle_labels.append(oracle)
            argmax_idx = int(torch.argmax(probs[i]).item())
            raw_argmax.append(argmax_idx)
            phase_group_logits[phase].append(logits[i])
            oracle_group_logits[oracle].append(logits[i])
            phase_group_embed[phase].append(embeds[i])
            phase_group_hidden[phase].append(hidden[i])

            for a, name in enumerate(ACTION_NAMES):
                corr_norm_cost_logit[name].append(float(norm_cost[i, a].item()))
                corr_norm_cost_logit_y[name].append(float(logits[i, a].item()))
                corr_neg_cost_prob_x[name].append(float(-norm_cost[i, a].item()))
                corr_neg_cost_prob_y[name].append(float(probs[i, a].item()))

    if total == 0:
        raise RuntimeError("no states collected from trace")

    raw_obs_t = torch.stack(raw_obs_rows, dim=0)
    norm_obs_t = torch.stack(normalized_obs_rows, dim=0)
    embed_t = torch.stack(embed_rows, dim=0)
    hidden_t = torch.stack(hidden_rows, dim=0)
    logits_t = torch.stack(logits_rows, dim=0)
    probs_t = torch.stack(probs_rows, dim=0)
    raw_argmax_t = torch.tensor(raw_argmax, dtype=torch.long)

    obs_std_by_field = {
        FIELD_NAMES[i]: float(raw_obs_t[:, i].std(unbiased=False)) for i in range(min(raw_obs_t.shape[1], len(FIELD_NAMES)))
    }
    normalized_obs_std_by_field = {
        FIELD_NAMES[i]: float(norm_obs_t[:, i].std(unbiased=False)) for i in range(min(norm_obs_t.shape[1], len(FIELD_NAMES)))
    }
    saturation = {
        FIELD_NAMES[i]: float((norm_obs_t[:, i] >= 0.999).float().mean())
        for i in range(min(norm_obs_t.shape[1], len(FIELD_NAMES)))
    }

    phase_sep_obs = _group_separability(norm_obs_t, phase_labels)
    phase_sep_embed = _group_separability(embed_t, phase_labels)
    phase_sep_logits = _group_separability(logits_t, phase_labels)
    oracle_sep_logits = _group_separability(logits_t, oracle_labels)

    phase_stats: Dict[str, Dict[str, float]] = {}
    for phase, items in phase_group_logits.items():
        t = torch.stack(items, dim=0)
        phase_stats[phase] = {
            "logit_std_mean": float(t.std(dim=0, unbiased=False).mean()),
            "logit_pairwise_distance_mean": _pairwise_distance_mean(t),
            "count": float(t.shape[0]),
        }

    oracle_stats: Dict[str, Dict[str, float]] = {}
    for action, items in oracle_group_logits.items():
        t = torch.stack(items, dim=0)
        oracle_stats[action] = {
            "logit_std_mean": float(t.std(dim=0, unbiased=False).mean()),
            "logit_pairwise_distance_mean": _pairwise_distance_mean(t),
            "count": float(t.shape[0]),
        }

    corr_norm_cost_logit_out = {name: _corr(corr_norm_cost_logit[name], corr_norm_cost_logit_y[name]) for name in ACTION_NAMES}
    corr_neg_cost_prob_out = {name: _corr(corr_neg_cost_prob_x[name], corr_neg_cost_prob_y[name]) for name in ACTION_NAMES}
    logit_mean = {name: float(logits_t[:, i].mean()) for i, name in enumerate(ACTION_NAMES)}
    logit_std = {name: float(logits_t[:, i].std(unbiased=False)) for i, name in enumerate(ACTION_NAMES)}
    prob_std = {name: float(probs_t[:, i].std(unbiased=False)) for i, name in enumerate(ACTION_NAMES)}
    raw_dist = {name: float((raw_argmax_t == i).float().mean()) for i, name in enumerate(ACTION_NAMES)}

    classification = _classify(
        phase_sep_obs=phase_sep_obs,
        phase_sep_embed=phase_sep_embed,
        phase_sep_logits=phase_sep_logits,
        raw_argmax_geo_ratio=raw_dist.get("geo", 0.0),
        logit_mean=logit_mean,
        saturation=saturation,
    )

    payload = {
        "checkpoint": args.checkpoint,
        "trace": args.trace,
        "oracle": args.oracle,
        "num_states": total,
        "policy_head": str(getattr(policy.cfg.algo, "policy_head", "gnn_only")),
        "logit_centering": bool(getattr(policy.cfg.algo, "logit_centering", False)),
        "raw_argmax_distribution": raw_dist,
        "obs_std_by_field": obs_std_by_field,
        "normalized_obs_std_by_field": normalized_obs_std_by_field,
        "feature_saturation_ratio_by_field": saturation,
        "embedding_std_mean": float(embed_t.std(dim=0, unbiased=False).mean()),
        "embedding_pairwise_distance_mean": _pairwise_distance_mean(embed_t),
        "hidden_std_mean": float(hidden_t.std(dim=0, unbiased=False).mean()),
        "hidden_pairwise_distance_mean": _pairwise_distance_mean(hidden_t),
        "logit_std_by_action": logit_std,
        "prob_std_by_action": prob_std,
        "correlation_normalized_cost_feature_vs_logit_action": corr_norm_cost_logit_out,
        "correlation_negative_cost_vs_prob_action": corr_neg_cost_prob_out,
        "phase_separability_obs": phase_sep_obs,
        "phase_separability_embedding": phase_sep_embed,
        "phase_separability_logits": phase_sep_logits,
        "oracle_action_separability_logits": oracle_sep_logits,
        "phase_group_logit_stats": phase_stats,
        "oracle_group_logit_stats": oracle_stats,
        "classification": classification,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
