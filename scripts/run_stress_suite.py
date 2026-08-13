from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trisatflow.config import load_config
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv


FIXED_SIZE_DEPENDENCIES = [
    {
        "module": "lower centralized critic",
        "fixed_size_dependency": "critic inputs commonly flatten all LEO embeddings/actions into checkpoint-shaped vectors",
        "required_refactor": "replace flatten critic with permutation-invariant pooling or graph-level readout",
        "current_status": "transfer_blocked_by_fixed_size_module_for_checkpoint_16_to_32_64",
    },
    {
        "module": "QMIX/VDN mixing networks",
        "fixed_size_dependency": "mixers are initialized with a fixed n_agents dimension",
        "required_refactor": "agent-count invariant mixer or per-agent pooling adapter",
        "current_status": "checkpoint_transfer_not_claimed",
    },
    {
        "module": "replay buffer tensors",
        "fixed_size_dependency": "stored transitions are shaped by training n_leo",
        "required_refactor": "do not mix train and transfer replay batches without padding/masking",
        "current_status": "evaluation_reset_only_safe",
    },
    {
        "module": "obs_builder candidate features",
        "fixed_size_dependency": "per-node observations are variable-size, but downstream flatten consumers may be fixed",
        "required_refactor": "keep per-node tensors until pooled policy/critic readout",
        "current_status": "reset_shape_check_required",
    },
    {
        "module": "graph encoder pooling",
        "fixed_size_dependency": "message passing can be variable-size if pooled before fixed heads",
        "required_refactor": "verify checkpoint heads consume pooled or shared per-node features",
        "current_status": "not_sufficient_for_transfer_claim_alone",
    },
    {
        "module": "upper action head",
        "fixed_size_dependency": "shared per-agent action head is reset-compatible; checkpoint compatibility depends on encoder/critic",
        "required_refactor": "shape-test checkpoint forward on 32/64 before transfer claims",
        "current_status": "inductive_transfer_unproven",
    },
]


def _load_raw_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    parent = data.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        base = _load_raw_yaml(parent_path)
        return _deep_merge(base, data)
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _float_mean(info: Dict[str, Any], key: str) -> float:
    value = info.get(key)
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.float().mean().detach().cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def _random_visible_action(env: GeoLeoGroundEnv, rng: random.Random) -> torch.Tensor:
    mask = env._upper_action_mask_at_step(env.t).detach().cpu()  # private by design for smoke-only diagnostics
    actions = []
    for row in mask:
        feasible = [idx for idx, bit in enumerate(row.tolist()) if bool(bit)]
        actions.append(rng.choice(feasible) if feasible else 0)
    return torch.tensor(actions, dtype=torch.long, device=env.device)


def _rule_action(env: GeoLeoGroundEnv) -> torch.Tensor:
    mask = env._upper_action_mask_at_step(env.t)
    preferred = [0, 1, 2, 3]
    actions = []
    for row in mask.detach().cpu().tolist():
        selected = 0
        for action in preferred:
            if action < len(row) and bool(row[action]):
                selected = action
                break
        actions.append(selected)
    return torch.tensor(actions, dtype=torch.long, device=env.device)


def _select_action(env: GeoLeoGroundEnv, policy: str, rng: random.Random) -> torch.Tensor:
    if policy == "random_visible":
        return _random_visible_action(env, rng)
    if policy == "rule":
        return _rule_action(env)
    return _random_visible_action(env, rng)


def transfer_blockers_for_config(*, n_leo: int, train_n_leo: int, policy: str, checkpoint: str | None) -> List[Dict[str, str]]:
    blockers = list(FIXED_SIZE_DEPENDENCIES)
    if policy == "checkpoint":
        if not checkpoint:
            blockers.append(
                {
                    "module": "checkpoint loader",
                    "fixed_size_dependency": "checkpoint policy requested without checkpoint path",
                    "required_refactor": "pass --checkpoint and run an explicit forward shape test",
                    "current_status": "transfer_blocked_missing_checkpoint",
                }
            )
        elif int(n_leo) != int(train_n_leo):
            blockers.append(
                {
                    "module": "checkpoint shape compatibility",
                    "fixed_size_dependency": f"train_n_leo={train_n_leo}, test_n_leo={n_leo}",
                    "required_refactor": "run checkpoint forward on target n_leo and inspect incompatible state_dict keys",
                    "current_status": "transfer_blocked_by_fixed_n_leo_checkpoint_until_shape_test_passes",
                }
            )
    return blockers


def run_one(config_path: Path, *, policy: str, checkpoint: str | None, episodes: int, steps: int, device: str) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    raw = _load_raw_yaml(config_path)
    stress = dict(raw.get("stress") or {})
    cfg = load_config(config_path)
    cfg.total_episodes = int(episodes)
    cfg.steps_per_episode = int(steps)
    cfg.device = device
    cfg.scenario.episode_len = int(steps)
    cfg.scenario.seed = int(getattr(cfg.scenario, "seed", 7))
    train_n_leo = int(stress.get("train_n_leo", 16))
    rng = random.Random(cfg.scenario.seed)
    row: Dict[str, Any] = {
        "stress_name": stress.get("stress_name") or config_path.stem,
        "n_leo": int(cfg.scenario.n_leo),
        "n_geo": int(cfg.scenario.n_geo),
        "n_ground": int(cfg.scenario.n_ground),
        "train_n_leo": train_n_leo,
        "test_n_leo": int(cfg.scenario.n_leo),
        "gateway_visibility_mode": stress.get("gateway_visibility_mode", "unknown"),
        "isl_density_mode": stress.get("isl_density_mode", "unknown"),
        "burst_prob": float(stress.get("burst_prob", getattr(cfg.scenario, "burst_prob", 0.0))),
        "deadline_mode": stress.get("deadline_mode", "unknown"),
        "mask_noise_level": float(stress.get("mask_noise_level", 0.0)),
        "domain_shift": bool(stress.get("domain_shift", False)),
        "policy_source": policy,
        "cost": 0.0,
        "delay": 0.0,
        "energy": 0.0,
        "violation": 0.0,
        "reset_ok": False,
        "step_ok": False,
        "checkpoint_compatible": bool(checkpoint) and policy == "checkpoint" and int(cfg.scenario.n_leo) == train_n_leo,
        "transfer_blocked": False,
        "blocker_reason": "",
    }
    blockers = transfer_blockers_for_config(n_leo=cfg.scenario.n_leo, train_n_leo=train_n_leo, policy=policy, checkpoint=checkpoint)
    try:
        env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, device=device)
        env.reset()
        row["reset_ok"] = True
        metric_sums = {"cost": 0.0, "delay": 0.0, "energy": 0.0, "violation": 0.0}
        n = 0
        for _ in range(int(steps)):
            upper = _select_action(env, policy, rng)
            lower = torch.ones((cfg.scenario.n_leo, 3), dtype=torch.float32, device=env.device)
            out = env.step(upper, lower, minimal_info=True)
            info = out.info
            metric_sums["cost"] += _float_mean(info, "normalized_system_cost")
            metric_sums["delay"] += _float_mean(info, "delay")
            metric_sums["energy"] += _float_mean(info, "energy")
            metric_sums["violation"] += _float_mean(info, "deadline_violation_flag")
            n += 1
            if out.done:
                break
        if n:
            for key in metric_sums:
                row[key] = metric_sums[key] / n
        row["step_ok"] = n > 0
    except Exception as exc:
        row["transfer_blocked"] = True
        row["blocker_reason"] = f"reset_or_step_failed: {type(exc).__name__}: {exc}"
        blockers.append(
            {
                "module": "environment reset/step",
                "fixed_size_dependency": f"n_leo={cfg.scenario.n_leo}",
                "required_refactor": "inspect env tensors and topology trace shapes",
                "current_status": row["blocker_reason"],
            }
        )
    if policy == "checkpoint" and not row["checkpoint_compatible"]:
        row["transfer_blocked"] = True
        row["blocker_reason"] = row["blocker_reason"] or "checkpoint transfer not shape-verified"
    return row, blockers


def run_suite(configs: Sequence[Path], *, policy: str, checkpoint: str | None, episodes: int, steps: int, device: str, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    blockers: List[Dict[str, str]] = []
    for config in configs:
        row, found = run_one(config, policy=policy, checkpoint=checkpoint, episodes=episodes, steps=steps, device=device)
        rows.append(row)
        blockers.extend(found)
    csv_path = output_dir / "stress_results.csv"
    fields = [
        "stress_name",
        "n_leo",
        "n_geo",
        "n_ground",
        "train_n_leo",
        "test_n_leo",
        "gateway_visibility_mode",
        "isl_density_mode",
        "burst_prob",
        "deadline_mode",
        "mask_noise_level",
        "domain_shift",
        "policy_source",
        "cost",
        "delay",
        "energy",
        "violation",
        "reset_ok",
        "step_ok",
        "checkpoint_compatible",
        "transfer_blocked",
        "blocker_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stress_rows": len(rows),
        "stress_results_csv": str(csv_path),
        "reset_ok_count": sum(1 for r in rows if r.get("reset_ok")),
        "step_ok_count": sum(1 for r in rows if r.get("step_ok")),
        "transfer_claim_supported": False,
        "transfer_claim_guard": "16_to_32_64 inductive transfer is not supported until checkpoint forward/evaluation passes on target n_leo.",
        "blocked_count": sum(1 for r in rows if r.get("transfer_blocked")),
    }
    with (output_dir / "stress_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    dedup = {(b["module"], b["current_status"]): b for b in blockers}
    with (output_dir / "transfer_blockers.json").open("w", encoding="utf-8") as f:
        json.dump(list(dedup.values()), f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def _parse_configs(value: str) -> List[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-configs", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--policy", choices=["random_visible", "rule", "checkpoint"], default="random_visible")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-transfer-blocker", choices=["true", "false"], default="false")
    args = parser.parse_args()
    configs = _parse_configs(args.stress_configs)
    summary = run_suite(
        configs,
        policy=args.policy,
        checkpoint=args.checkpoint or None,
        episodes=min(int(args.episodes), 2),
        steps=min(int(args.steps), 8),
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_transfer_blocker == "true" and summary.get("blocked_count", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
