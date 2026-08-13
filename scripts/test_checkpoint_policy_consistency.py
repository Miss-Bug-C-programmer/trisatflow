from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy
from trisatflow.satedgesim_eval.inspection import load_trace_groups, shared_batch_from_trace_group


def _ratio(actions: torch.Tensor, idx: int) -> float:
    if actions.numel() == 0:
        return 0.0
    return float((actions == idx).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check checkpoint save/load consistency for policy logits/probs/actions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=16)
    parser.add_argument("--num-states", type=int, default=1024)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu")

    with tempfile.TemporaryDirectory(prefix="ckpt_consistency_") as tmp_dir:
        tmp_ckpt = Path(tmp_dir) / "roundtrip.pt"
        torch.save(payload, tmp_ckpt)

        policy_a = FrozenTriSatFlowPolicy(ckpt_path, device="cpu")
        policy_b = FrozenTriSatFlowPolicy(tmp_ckpt, device="cpu")

        groups = load_trace_groups(
            args.trace,
            n_leo=args.n_leo,
            num_states=max(1, args.num_states // max(1, args.n_leo)),
        )
        batches: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for group in groups:
            batches.append(shared_batch_from_trace_group(group, node_feature_dim=policy_a.cfg.scenario.node_feature_dim))
            if len(batches) * args.n_leo >= args.num_states:
                break
        if not batches:
            raise RuntimeError("no trace batches available")

        max_logit_diff = 0.0
        max_prob_diff = 0.0
        argmax_a = []
        for obs, edge_index, edge_attr in batches:
            da = policy_a.inspect_upper_policy(obs, edge_index, edge_attr)
            db = policy_b.inspect_upper_policy(obs, edge_index, edge_attr)
            diff_l = torch.max(torch.abs(da["masked_logits"].detach().cpu() - db["masked_logits"].detach().cpu()))
            diff_p = torch.max(torch.abs(da["probs"].detach().cpu() - db["probs"].detach().cpu()))
            max_logit_diff = max(max_logit_diff, float(diff_l))
            max_prob_diff = max(max_prob_diff, float(diff_p))
            argmax_a.append(da["argmax_action"].detach().cpu())

    argmax_t = torch.cat(argmax_a, dim=0) if argmax_a else torch.empty(0, dtype=torch.long)
    argmax_distribution = {
        "argmax_local_ratio": _ratio(argmax_t, 0),
        "argmax_neighbor_ratio": _ratio(argmax_t, 1),
        "argmax_geo_ratio": _ratio(argmax_t, 2),
        "argmax_ground_ratio": _ratio(argmax_t, 3),
    }

    inspect_path = Path("outputs/policy_inspection_mixed_v2_trace_cpu.json")
    inspect_match = False
    inspect_ref: Dict[str, Any] = {}
    if inspect_path.exists():
        inspect_ref = json.loads(inspect_path.read_text(encoding="utf-8"))
        diffs = []
        for k, v in argmax_distribution.items():
            diffs.append(abs(float(inspect_ref.get(k, 0.0)) - float(v)))
        inspect_match = max(diffs) < 1.0e-6

    output = {
        "checkpoint": str(ckpt_path),
        "num_states": len(argmax_t) // max(1, args.n_leo),
        "num_agent_observations": int(argmax_t.numel()),
        "max_logit_diff_after_reload": max_logit_diff,
        "max_prob_diff_after_reload": max_prob_diff,
        "argmax_distribution": argmax_distribution,
        "inspection_reference_path": str(inspect_path),
        "inspection_reference": inspect_ref,
        "argmax_distribution_matches_inspection": inspect_match,
        "status": "CHECKPOINT_POLICY_CONSISTENT"
        if max_logit_diff < 1.0e-6 and max_prob_diff < 1.0e-6 and inspect_match
        else "CHECKPOINT_POLICY_INCONSISTENT",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "CHECKPOINT_POLICY_CONSISTENT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
