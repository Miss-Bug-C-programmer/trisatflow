from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

import yaml

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import TrainConfig, load_config
from trisatflow.experiment_contracts import (
    assert_paper_safe,
    assert_same_contract,
    contract_diff_paths,
    contract_sha256,
    resolve_contract,
)


DEFAULT_ABLATIONS = (
    "no_mask",
    "visibility_only",
    "completion_safe",
    "full_mask",
    "no_gnn",
    "static_gnn",
    "temporal_gnn",
    "no_cost_prior",
)


def _split_names(value: str, choices: Iterable[str]) -> List[str]:
    choices_list = list(choices)
    if value == "all":
        return choices_list
    raw = value.replace(",", " ").split()
    names = [item.strip() for item in raw if item.strip()]
    bad = [name for name in names if name not in choices_list]
    if bad:
        raise ValueError(f"Unknown ablations {bad}; choices={choices_list}")
    return names


def _split_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.replace(",", " ").split() if item.strip()]


def _tail_mean(history: List[Dict[str, float]], key: str, window: int) -> float:
    vals = [float(row[key]) for row in history[-window:] if key in row]
    return mean(vals) if vals else float("nan")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _ci95(vals: List[float]) -> float:
    return 1.96 * pstdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must contain a mapping: {path}")
    return data


def _ablation_spec(path: Path) -> Dict[str, Any]:
    data = _read_yaml(path)
    spec = data.get("ablation")
    if not isinstance(spec, dict):
        raise ValueError(f"ablation metadata missing in {path}")
    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError(f"ablation.name missing in {path}")
    allowed = spec.get("allowed_contract_diff_paths", [])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError(f"ablation.allowed_contract_diff_paths must be a string list in {path}")
    return {"name": name, "allowed_contract_diff_paths": [str(item) for item in allowed]}


def _config_files(root: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in sorted(root.glob("*.yaml")):
        spec = _ablation_spec(path)
        name = str(spec["name"])
        if name in files:
            raise ValueError(f"duplicate ablation name {name}: {files[name]} and {path}")
        files[name] = path
    return files


def _phase_seed_plan(cfg: TrainConfig, *, smoke: bool, args: argparse.Namespace) -> Dict[str, List[int]]:
    split = cfg.experiment.split
    train = _split_ints(args.train_seeds) if args.train_seeds else list(split.train_seeds)
    val = _split_ints(args.val_seeds) if args.val_seeds else list(split.val_seeds)
    test = _split_ints(args.test_seeds) if args.test_seeds else list(split.test_seeds)
    fallback = int(getattr(cfg.scenario, "seed", 13))
    plan = {
        "train": train or [fallback],
        "validation": val or [fallback + 1],
        "test": test or [fallback + 2],
    }
    if smoke:
        return {phase: seeds[:1] for phase, seeds in plan.items()}
    return plan


def _prepare_cfg(base: TrainConfig, *, args: argparse.Namespace, phase: str, seed: int, output_dir: Path) -> TrainConfig:
    cfg = load_config(Path(base.config_source_chain[-1]))
    cfg.algo.upper_algo = str(args.upper)
    cfg.algo.lower_algo = str(args.lower)
    cfg.scenario.seed = int(seed)
    cfg.output_dir = str(output_dir)
    if args.device is not None:
        cfg.device = str(args.device)
    if args.smoke:
        cfg.total_episodes = int(args.episodes if args.episodes is not None else 1)
        cfg.scenario.episode_len = int(args.steps if args.steps is not None else 4)
        cfg.steps_per_episode = cfg.scenario.episode_len
        cfg.scenario.n_leo = int(args.n_leo if args.n_leo is not None else 4)
    else:
        if args.episodes is not None:
            cfg.total_episodes = int(args.episodes)
        if args.steps is not None:
            cfg.scenario.episode_len = int(args.steps)
            cfg.steps_per_episode = int(args.steps)
        if args.n_leo is not None:
            cfg.scenario.n_leo = int(args.n_leo)
    return cfg


def _validate_contract(base_cfg: TrainConfig, cfg: TrainConfig, config_path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    assert_paper_safe(cfg)
    base_source = str(Path(base_cfg.config_source_chain[-1]).resolve())
    if base_source not in {str(Path(item).resolve()) for item in cfg.config_source_chain}:
        raise ValueError(f"{config_path} does not inherit the paper-safe base config {base_source}")
    if str(config_path.resolve()) != str(Path(cfg.config_source_chain[-1]).resolve()):
        raise ValueError(f"{config_path} is not the final config in its resolved source chain")
    base_contract = resolve_contract(base_cfg, "stage10_contract_trace_sha")
    ablation_contract = resolve_contract(cfg, "stage10_contract_trace_sha")
    allowed = list(spec["allowed_contract_diff_paths"])
    assert_same_contract(base_contract, ablation_contract, allowed)
    diffs = contract_diff_paths(base_contract, ablation_contract)
    return {
        "config": str(config_path),
        "ablation": spec["name"],
        "allowed_contract_diff_paths": allowed,
        "contract_diff_paths": diffs,
        "contract_sha256": contract_sha256(ablation_contract),
    }


def _run_one(cfg: TrainConfig, *, ablation: str, phase: str, seed: int, args: argparse.Namespace, contract: Dict[str, Any]) -> Dict[str, object]:
    start = time.time()
    status = "ok"
    error = ""
    history: List[Dict[str, float]]
    try:
        trainer = HierarchicalTrainer(cfg)
        history = trainer.train()
        trainer.save_checkpoint(Path(cfg.output_dir) / "checkpoint.pt")
    except Exception as exc:
        history = []
        status = "failed"
        error = repr(exc)
    elapsed = time.time() - start
    row: Dict[str, object] = {
        "status": status,
        "error": error,
        "ablation": ablation,
        "phase": phase,
        "protocol_role": phase,
        "seed": seed,
        "upper_algo": cfg.algo.upper_algo,
        "lower_algo": cfg.algo.lower_algo,
        "episodes": cfg.total_episodes,
        "episode_len": cfg.scenario.episode_len,
        "n_leo": cfg.scenario.n_leo,
        "requested_device": getattr(cfg, "requested_device", cfg.device),
        "actual_device": getattr(cfg, "actual_device", cfg.device),
        "device_fallback_reason": getattr(cfg, "device_fallback_reason", ""),
        "output_dir": cfg.output_dir,
        "metrics_csv": str(Path(cfg.output_dir) / "metrics.csv"),
        "checkpoint": str(Path(cfg.output_dir) / "checkpoint.pt"),
        "elapsed_sec": round(elapsed, 4),
        "experiment_contract_sha256": contract["contract_sha256"],
        "contract_diff_paths": "|".join(contract["contract_diff_paths"]),
        "allowed_contract_diff_paths": "|".join(contract["allowed_contract_diff_paths"]),
        "tail_mean_delay": _tail_mean(history, "mean_delay", args.summary_window) if history else "",
        "tail_mean_energy": _tail_mean(history, "mean_energy", args.summary_window) if history else "",
        "tail_mean_queue": _tail_mean(history, "mean_queue", args.summary_window) if history else "",
        "tail_mean_system_cost": _tail_mean(history, "mean_system_cost", args.summary_window) if history else "",
        "tail_mean_deadline_violation": _tail_mean(history, "mean_deadline_violation", args.summary_window) if history else "",
        "tail_mean_feasibility": _tail_mean(history, "mean_feasibility", args.summary_window) if history else "",
        "tail_mean_virtual_delay_queue": _tail_mean(history, "mean_virtual_delay_queue", args.summary_window) if history else "",
    }
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriSatFlow reviewer-facing ablation suite.")
    parser.add_argument("--config-root", type=str, default="", help="Directory containing Stage 10 ablation YAML files.")
    parser.add_argument("--base-config", type=str, default="trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml")
    parser.add_argument("--config", type=str, default="", help="Legacy single base config path.")
    parser.add_argument("--ablations", type=str, default="all")
    parser.add_argument("--upper", type=str, default="mappo")
    parser.add_argument("--lower", type=str, default="maddpg")
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--val-seeds", type=str, default="")
    parser.add_argument("--test-seeds", type=str, default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-leo", type=int, default=None)
    parser.add_argument("--summary-window", type=int, default=5)
    parser.add_argument("--output-root", type=str, default="outputs/ablation_suite")
    parser.add_argument("--device", type=str, default=None, help="Override config device: cpu|cuda|cuda:0|auto")
    parser.add_argument("--smoke", action="store_true", help="Run one short train/validation/test seed per ablation.")
    args = parser.parse_args()

    if not args.config_root:
        if not args.config:
            raise SystemExit("--config-root is required for Stage 10 ablation contracts")
        raise SystemExit("legacy --config mode is no longer paper-ready; use --config-root")

    config_root = Path(args.config_root)
    available = _config_files(config_root)
    missing = [name for name in DEFAULT_ABLATIONS if name not in available]
    if missing:
        raise SystemExit(f"missing required ablation configs: {missing}")
    names = _split_names(args.ablations, available.keys())
    base_cfg = load_config(args.base_config)
    assert_paper_safe(base_cfg)
    output_root = Path(args.output_root)
    run_rows: List[Dict[str, object]] = []

    contracts: Dict[str, Dict[str, Any]] = {}
    for name in names:
        path = available[name]
        cfg = load_config(path)
        spec = _ablation_spec(path)
        if spec["name"] != name:
            raise SystemExit(f"ablation filename/name mismatch: {path} name={spec['name']}")
        contracts[name] = _validate_contract(base_cfg, cfg, path, spec)

    for name in names:
        config_path = available[name]
        cfg_for_plan = load_config(config_path)
        seed_plan = _phase_seed_plan(cfg_for_plan, smoke=bool(args.smoke), args=args)
        for phase, seeds in seed_plan.items():
            for seed in seeds:
                run_dir = output_root / name / phase / f"seed_{seed}"
                cfg = _prepare_cfg(cfg_for_plan, args=args, phase=phase, seed=seed, output_dir=run_dir)
                run_rows.append(_run_one(cfg, ablation=name, phase=phase, seed=seed, args=args, contract=contracts[name]))

    failures = [row for row in run_rows if row["status"] != "ok"]
    _write_csv(output_root / "ablation_runs.csv", run_rows)
    summary_rows = []
    metric_cols = [
        "tail_mean_delay",
        "tail_mean_energy",
        "tail_mean_queue",
        "tail_mean_system_cost",
        "tail_mean_deadline_violation",
        "tail_mean_feasibility",
        "tail_mean_virtual_delay_queue",
    ]
    for name in names:
        subset = [row for row in run_rows if row["ablation"] == name and row["status"] == "ok"]
        phases = sorted({str(row["phase"]) for row in subset})
        summary: Dict[str, object] = {
            "ablation": name,
            "num_successful_runs": len(subset),
            "phases": ",".join(phases),
            "all_required_phases_complete": set(phases) == {"test", "train", "validation"},
            "contract_diff_paths": "|".join(contracts[name]["contract_diff_paths"]),
            "allowed_contract_diff_paths": "|".join(contracts[name]["allowed_contract_diff_paths"]),
        }
        for col in metric_cols:
            vals = [float(row[col]) for row in subset if row[col] != ""]
            if vals:
                summary[f"{col}_mean"] = mean(vals)
                summary[f"{col}_ci95"] = _ci95(vals)
        summary_rows.append(summary)
    _write_csv(output_root / "ablation_summary.csv", summary_rows)
    if failures:
        raise SystemExit(f"ABLATION_SUITE_FAILED failed_runs={len(failures)} runs_csv={output_root / 'ablation_runs.csv'}")
    incomplete = [row["ablation"] for row in summary_rows if not row["all_required_phases_complete"]]
    if incomplete:
        raise SystemExit(f"ABLATION_SUITE_FAILED missing_required_phases={incomplete}")
    print(f"ABLATION_SUITE_OK runs_csv={output_root / 'ablation_runs.csv'} summary_csv={output_root / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
