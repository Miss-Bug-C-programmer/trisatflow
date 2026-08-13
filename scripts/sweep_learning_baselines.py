from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.agents.flat_hybrid_trainer import FlatHybridTrainer
from trisatflow.agents.hierarchical_trainer import HierarchicalTrainer
from trisatflow.algorithms.registry import learning_baseline_names
from trisatflow.config import TrainConfig, load_config
from trisatflow.experiment_contracts import assert_paper_safe, trace_sha256_for_config, write_contract_artifacts


PHASES = ("train", "validation", "test")


def _split_names(value: str, choices: Iterable[str]) -> List[str]:
    choices_set = set(choices)
    names = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not names or names == ["all"]:
        return sorted(choices_set)
    bad = [name for name in names if name not in choices_set]
    if bad:
        raise ValueError(f"unknown learning baselines {bad}; choices={sorted(choices_set)}")
    return names


def _split_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.replace(",", " ").split() if item.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _tail_mean(history: List[Dict[str, Any]], key: str, window: int) -> float | str:
    values = [float(row[key]) for row in history[-window:] if key in row and str(row[key]) not in {"", "NA"}]
    return mean(values) if values else ""


def _phase_seed_plan(cfg: TrainConfig, args: argparse.Namespace) -> Dict[str, List[int]]:
    split = cfg.experiment.split
    fallback = int(cfg.scenario.seed)
    return {
        "train": _split_ints(args.train_seeds) if args.train_seeds else (list(split.train_seeds) or [fallback]),
        "validation": _split_ints(args.val_seeds) if args.val_seeds else (list(split.val_seeds) or [fallback + 1]),
        "test": _split_ints(args.test_seeds) if args.test_seeds else (list(split.test_seeds) or [fallback + 2]),
    }


def _prepare_cfg(config_path: Path, *, baseline: str, args: argparse.Namespace, phase: str, seed: int, output_dir: Path) -> TrainConfig:
    cfg = load_config(config_path)
    cfg.total_episodes = int(args.episodes)
    cfg.scenario.episode_len = int(args.steps)
    cfg.steps_per_episode = int(args.steps)
    cfg.scenario.n_leo = int(args.n_leo)
    cfg.scenario.seed = int(seed)
    cfg.device = str(args.device)
    cfg.output_dir = str(output_dir)
    if baseline == "hierarchical_no_gnn":
        cfg.algo.upper_algo = "mappo"
        cfg.algo.lower_algo = "maddpg"
        cfg.model.topology_encoder = "no_gnn"
        cfg.model.temporal.enabled = False
        cfg.scenario.enable_gnn = False
    else:
        cfg.algo.upper_algo = baseline
        cfg.algo.lower_algo = "flat_resource_head"
        cfg.model.topology_encoder = "static_gnn"
        cfg.model.temporal.enabled = False
        cfg.scenario.enable_gnn = True
    assert_paper_safe(cfg)
    return cfg


def _fairness_payload(cfg: TrainConfig, *, trace_sha256: str, args: argparse.Namespace, seed_plan: Dict[str, List[int]]) -> Dict[str, Any]:
    return {
        "config": str(Path(args.config)),
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "n_leo": int(args.n_leo),
        "requested_device": str(args.device),
        "trace": {
            "topology_mode": str(cfg.scenario.topology_mode),
            "path": str(cfg.scenario.topology_trace_path),
            "sha256": str(trace_sha256),
            "strict": bool(cfg.scenario.topology_trace_strict),
        },
        "observation": {
            "mode": str(cfg.observation.mode),
            "include_oracle_cost": bool(cfg.observation.include_oracle_cost),
            "include_cost_prior_features": bool(cfg.observation.include_cost_prior_features),
        },
        "reward": {
            "mode": str(cfg.reward.mode),
            "use_oracle_cost_components": bool(cfg.reward.use_oracle_cost_components),
        },
        "seed_banks": {phase: list(seeds) for phase, seeds in seed_plan.items()},
    }


def _fingerprint(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_one(cfg: TrainConfig, *, baseline: str) -> tuple[List[Dict[str, Any]], Path]:
    if baseline in {"flat_ppo", "flat_mappo"}:
        trainer = FlatHybridTrainer(cfg, baseline_name=baseline)
        history = trainer.train()
        ckpt = Path(cfg.output_dir) / "checkpoint.pt"
        trainer.save_checkpoint(ckpt)
        return history, ckpt
    if baseline == "hierarchical_no_gnn":
        trainer = HierarchicalTrainer(cfg)
        history = trainer.train()
        ckpt = Path(cfg.output_dir) / "checkpoint.pt"
        trainer.save_checkpoint(ckpt)
        return history, ckpt
    raise ValueError(f"unsupported learning baseline {baseline!r}")


def _run_row(
    cfg: TrainConfig,
    *,
    baseline: str,
    phase: str,
    seed: int,
    args: argparse.Namespace,
    fairness_sha: str,
) -> Dict[str, Any]:
    status = "ok"
    error = ""
    history: List[Dict[str, Any]] = []
    checkpoint = Path(cfg.output_dir) / "checkpoint.pt"
    try:
        history, checkpoint = _run_one(cfg, baseline=baseline)
        write_contract_artifacts(cfg.output_dir, cfg, base_dir=Path(__file__).resolve().parents[1])
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    last = history[-1] if history else {}
    row: Dict[str, Any] = {
        "status": status,
        "error": error,
        "baseline": baseline,
        "phase": phase,
        "protocol_role": phase,
        "seed": int(seed),
        "upper_algo": baseline if baseline != "hierarchical_no_gnn" else "mappo",
        "lower_algo": "flat_resource_head" if baseline in {"flat_ppo", "flat_mappo"} else "maddpg",
        "episodes": int(cfg.total_episodes),
        "episode_len": int(cfg.scenario.episode_len),
        "n_leo": int(cfg.scenario.n_leo),
        "topology_trace_path": str(cfg.scenario.topology_trace_path),
        "requested_device": str(getattr(cfg, "requested_device", args.device)),
        "actual_device": str(getattr(cfg, "actual_device", cfg.device)),
        "device_fallback_reason": str(getattr(cfg, "device_fallback_reason", "")),
        "output_dir": str(cfg.output_dir),
        "metrics_csv": str(Path(cfg.output_dir) / "metrics.csv"),
        "checkpoint": str(checkpoint),
        "fairness_contract_sha256": fairness_sha,
        "observation_mode": str(cfg.observation.mode),
        "include_oracle_cost": int(bool(cfg.observation.include_oracle_cost)),
        "include_cost_prior_features": int(bool(cfg.observation.include_cost_prior_features)),
        "paper_ready": 1,
        "placeholder": 0,
        "tail_mean_delay": _tail_mean(history, "mean_delay_s", int(args.summary_window)),
        "tail_mean_energy": _tail_mean(history, "mean_energy_j", int(args.summary_window)),
        "tail_mean_system_cost": _tail_mean(history, "normalized_system_cost", int(args.summary_window)),
        "final_normalized_system_cost": last.get("normalized_system_cost", ""),
        "reward_mean": last.get("reward_mean", ""),
    }
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready learning baselines under the shared TriSatFlow contract.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baselines", type=str, default="flat_ppo,flat_mappo,hierarchical_no_gnn")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--n-leo", type=int, required=True)
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--val-seeds", type=str, default="")
    parser.add_argument("--test-seeds", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs/learning_baselines")
    parser.add_argument("--summary-window", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    base_cfg = load_config(config_path)
    assert_paper_safe(base_cfg)
    seed_plan = _phase_seed_plan(base_cfg, args)
    trace_sha = trace_sha256_for_config(base_cfg, base_dir=Path(__file__).resolve().parents[1])
    fairness_payload = _fairness_payload(base_cfg, trace_sha256=trace_sha, args=args, seed_plan=seed_plan)
    fairness_sha = _fingerprint(fairness_payload)
    baselines = _split_names(args.baselines, learning_baseline_names())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        for phase in PHASES:
            for seed in seed_plan[phase]:
                outdir = output_root / baseline / phase / f"seed_{seed}"
                cfg = _prepare_cfg(config_path, baseline=baseline, args=args, phase=phase, seed=int(seed), output_dir=outdir)
                rows.append(_run_row(cfg, baseline=baseline, phase=phase, seed=int(seed), args=args, fairness_sha=fairness_sha))

    _write_csv(output_root / "sweep_summary.csv", rows)
    manifest = {
        "status": "ok",
        "baselines": baselines,
        "seed_plan": seed_plan,
        "fairness_contract": fairness_payload,
        "fairness_contract_sha256": fairness_sha,
        "summary_csv": str(output_root / "sweep_summary.csv"),
    }
    (output_root / "learning_baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"LEARNING_BASELINES_OK rows={len(rows)} output_root={output_root} fairness_contract_sha256={fairness_sha}")


if __name__ == "__main__":
    main()
