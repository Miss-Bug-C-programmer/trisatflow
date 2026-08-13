from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.models import TopologyEncoder, UpperMAPPOPolicy, upper_action_mask_from_obs
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


TIERS = list(ACTION_NAMES)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v: Any, default: bool = False) -> bool:
    if v in (None, ""):
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def _visible(row: Dict[str, Any], tier: str) -> bool:
    return _to_bool(row.get(f"{tier}_visible"), tier == "local")


def _component(row: Dict[str, Any], tier: str, name: str) -> float:
    return max(0.0, _to_float(row.get(f"{tier}_{name}"), 0.0))


def _total_delay(row: Dict[str, Any], tier: str) -> float:
    explicit = row.get(f"{tier}_total_delay")
    if explicit not in (None, ""):
        return max(0.0, _to_float(explicit, 0.0))
    return (
        _component(row, tier, "prop_delay")
        + _component(row, tier, "tx_delay")
        + _component(row, tier, "compute_delay")
        + _component(row, tier, "queue_delay")
    )


def _oracle_label(row: Dict[str, Any]) -> int:
    costs = [math.inf, math.inf, math.inf, math.inf]
    for idx, tier in enumerate(TIERS):
        if _visible(row, tier):
            costs[idx] = _total_delay(row, tier)
    label = min(range(4), key=lambda i: costs[i])
    if not math.isfinite(costs[label]):
        return 0
    return int(label)


def _action_ratio(t: torch.Tensor, idx: int) -> float:
    if t.numel() == 0:
        return 0.0
    return float((t == idx).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train upper-policy oracle imitation debug model from mixed_v2 trace.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-states", type=int, default=1024)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    groups = load_trace_groups(args.trace, n_leo=args.n_leo, num_states=args.num_states)
    if not groups:
        raise RuntimeError("no trace groups loaded")

    batches: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for group in groups:
        obs, edge_index, edge_attr = shared_batch_from_trace_group(group, node_feature_dim=16)
        labels = torch.tensor([_oracle_label(dict(row)) for row in group[: args.n_leo]], dtype=torch.long)
        batches.append((obs, edge_index, edge_attr, labels))

    device = torch.device("cpu")
    encoder = TopologyEncoder(node_dim=16, edge_dim=4, hidden_dim=64).to(device)
    actor = UpperMAPPOPolicy(embed_dim=64, hidden_dim=128, n_actions=4).to(device)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(actor.parameters()), lr=3.0e-4)

    for _ in range(args.epochs):
        for obs, edge_index, edge_attr, labels in batches:
            obs = obs.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            labels = labels.to(device)
            embed = encoder(obs, edge_index, edge_attr)
            logits = actor.net(embed)
            mask = upper_action_mask_from_obs(obs)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            loss = F.cross_entropy(masked_logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        all_argmax = []
        all_labels = []
        all_entropy = []
        for obs, edge_index, edge_attr, labels in batches:
            obs = obs.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            labels = labels.to(device)
            embed = encoder(obs, edge_index, edge_attr)
            logits = actor.net(embed)
            mask = upper_action_mask_from_obs(obs)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            probs = torch.softmax(masked_logits, dim=-1)
            argmax = probs.argmax(dim=-1)
            entropy = torch.distributions.Categorical(probs=probs).entropy()
            all_argmax.append(argmax.cpu())
            all_labels.append(labels.cpu())
            all_entropy.append(entropy.cpu())

    argmax_t = torch.cat(all_argmax, dim=0)
    label_t = torch.cat(all_labels, dim=0)
    entropy_t = torch.cat(all_entropy, dim=0)
    acc = float((argmax_t == label_t).float().mean())

    output = {
        "num_states": len(batches),
        "num_agent_observations": int(label_t.numel()),
        "oracle_imitation_accuracy": acc,
        "argmax_action_distribution": {
            "argmax_local_ratio": _action_ratio(argmax_t, 0),
            "argmax_neighbor_ratio": _action_ratio(argmax_t, 1),
            "argmax_geo_ratio": _action_ratio(argmax_t, 2),
            "argmax_ground_ratio": _action_ratio(argmax_t, 3),
        },
        "label_distribution": {
            "label_local_ratio": _action_ratio(label_t, 0),
            "label_neighbor_ratio": _action_ratio(label_t, 1),
            "label_geo_ratio": _action_ratio(label_t, 2),
            "label_ground_ratio": _action_ratio(label_t, 3),
        },
        "entropy_mean": float(entropy_t.mean()) if entropy_t.numel() else 0.0,
        "status": "ORACLE_IMITATION_OK" if acc >= 0.70 else "ORACLE_IMITATION_LOW_ACCURACY",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
