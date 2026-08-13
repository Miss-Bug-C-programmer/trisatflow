from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import TrainConfig, load_config


ABLATION_ROOT = REPO_ROOT / "trisatflow" / "configs" / "ablations"
DEFAULT_VARIANTS = "safe_observable_full,safe_no_mask,safe_no_gnn,safe_no_lyapunov"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _ablation_metadata(path: Path) -> dict[str, Any]:
    payload = _read_yaml(path)
    meta = payload.get("ablation")
    if not isinstance(meta, dict):
        meta = {}
    return {
        "name": str(meta.get("name", path.stem)),
        "main_ablation_deployable": bool(meta.get("main_ablation_deployable", path.stem.startswith("safe_"))),
        "diagnostic_only": bool(meta.get("diagnostic_only", False)),
    }


def _variant_path(name: str) -> Path:
    path = ABLATION_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown safe ablation variant={name!r}: {path}")
    return path


def _prepare_smoke_cfg(cfg: TrainConfig, *, episodes: int, steps: int, device: str, output_dir: Path) -> TrainConfig:
    cfg.total_episodes = int(episodes)
    cfg.scenario.episode_len = int(steps)
    cfg.steps_per_episode = int(steps)
    cfg.scenario.n_leo = min(int(cfg.scenario.n_leo), 4)
    cfg.scenario.topology_mode = "analytic"
    cfg.scenario.topology_trace_path = ""
    cfg.scenario.topology_trace_strict = False
    cfg.device = str(device)
    cfg.requested_device = str(device)
    cfg.output_dir = str(output_dir)
    cfg.lower_training_enabled = False
    cfg.lower_action_mode = "neutral_allocator"
    cfg.upper_pretrain_episodes = 0
    cfg.joint_train_episodes = 0
    cfg.log_interval = 1
    cfg.algo.gnn_hidden_dim = min(int(cfg.algo.gnn_hidden_dim), 16)
    cfg.algo.policy_hidden_dim = min(int(cfg.algo.policy_hidden_dim), 32)
    cfg.algo.upper_batch_size = min(int(cfg.algo.upper_batch_size), 4)
    cfg.algo.lower_batch_size = min(int(cfg.algo.lower_batch_size), 4)
    cfg.algo.upper_warmup = min(int(cfg.algo.upper_warmup), 4)
    cfg.algo.lower_warmup = min(int(cfg.algo.lower_warmup), 4)
    return cfg


def run_variant(name: str, *, episodes: int, steps: int, device: str, output_dir: Path) -> dict[str, Any]:
    config_path = _variant_path(name)
    meta = _ablation_metadata(config_path)
    cfg = _prepare_smoke_cfg(
        load_config(config_path),
        episodes=episodes,
        steps=steps,
        device=device,
        output_dir=output_dir / name,
    )
    if str(cfg.observation.mode) == "safe_observable":
        if bool(cfg.observation.include_cost_prior_features) or bool(cfg.observation.include_oracle_cost):
            raise ValueError(f"{name} is safe_observable but exposes privileged observation fields")
    history = HierarchicalTrainer(cfg).train()
    last = dict(history[-1] if history else {})
    row = {
        "variant": name,
        "config": str(config_path),
        "normalized_cost": float(last.get("normalized_system_cost", 0.0)),
        "delay": float(last.get("mean_delay_s", last.get("mean_delay", 0.0))),
        "energy": float(last.get("mean_energy_j", last.get("mean_energy", 0.0))),
        "deadline_violation_ratio": float(last.get("mean_deadline_violation_ratio", 0.0)),
        "mask_violation_ratio": float(last.get("invalid_action_ratio", 0.0)),
        "action_mix": {
            "local": float(last.get("upper_local_ratio", 0.0)),
            "neighbor": float(last.get("upper_neighbor_ratio", 0.0)),
            "geo": float(last.get("upper_geo_ratio", 0.0)),
            "ground": float(last.get("upper_ground_ratio", 0.0)),
        },
        "observation_policy": str(cfg.observation.mode),
        "uses_cost_prior": bool(cfg.observation.include_cost_prior_features),
        "uses_oracle_cost": bool(cfg.observation.include_oracle_cost),
        "main_ablation_deployable": bool(meta["main_ablation_deployable"]),
        "diagnostic_only": bool(meta["diagnostic_only"]),
        "enable_gnn": str(cfg.model.topology_encoder) != "no_gnn",
        "topology_encoder": str(cfg.model.topology_encoder),
        "action_mask_mode": str(cfg.scenario.action_mask_layer_mode),
        "enable_lyapunov_reward": bool(cfg.scenario.enable_lyapunov_reward),
        "enable_cross_layer_feedback": bool(cfg.scenario.enable_cross_layer_feedback),
        "episodes": int(episodes),
        "steps": int(steps),
        "device": str(device),
    }
    return row


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": rows,
        "row_count": len(rows),
        "main_ablation_deployable": all(bool(row["main_ablation_deployable"]) for row in rows),
        "uses_cost_prior": any(bool(row["uses_cost_prior"]) for row in rows),
        "uses_oracle_cost": any(bool(row["uses_oracle_cost"]) for row in rows),
        "note": "Main safe ablation fixes observation_policy=safe_observable and toggles only GNN/mask/Lyapunov/cross-layer feedback.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "variant",
            "normalized_cost",
            "delay",
            "energy",
            "deadline_violation_ratio",
            "mask_violation_ratio",
            "action_mix",
            "observation_policy",
            "uses_cost_prior",
            "uses_oracle_cost",
            "main_ablation_deployable",
            "diagnostic_only",
            "topology_encoder",
            "action_mask_mode",
            "enable_lyapunov_reward",
            "enable_cross_layer_feedback",
            "episodes",
            "steps",
            "device",
            "config",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["action_mix"] = json.dumps(row["action_mix"], sort_keys=True, separators=(",", ":"))
            writer.writerow({key: item.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny deployable safe-observable ablation smoke suite.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "reviewer_repair" / "safe_ablation"))
    args = parser.parse_args()
    if int(args.episodes) > 2 or int(args.steps) > 8:
        raise ValueError("CPU smoke guard: use episodes<=2 and steps<=8.")
    output_dir = Path(args.output_dir)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    rows = [
        run_variant(name, episodes=int(args.episodes), steps=int(args.steps), device=str(args.device), output_dir=output_dir)
        for name in variants
    ]
    _write_outputs(rows, output_dir)
    print(json.dumps({"rows": rows, "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
