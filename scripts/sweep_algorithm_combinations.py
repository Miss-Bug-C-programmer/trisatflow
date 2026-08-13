from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import platform
import shlex
import socket
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch

from trisatflow.agents import HierarchicalTrainer
from trisatflow.algorithms import lower_algorithm_names, supported_algorithm_matrix, upper_algorithm_names
from trisatflow.config import load_config
from trisatflow.experiment_contracts import write_contract_artifacts
from trisatflow.models import upper_action_mask_from_obs


def _split_algos(value: str, choices: List[str]) -> List[str]:
    if value == "all":
        return choices
    algos = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in algos if item not in choices]
    if bad:
        raise ValueError(f"Unsupported algorithms {bad}; choices={choices}")
    return algos


def _parse_seed_list(text: str | None) -> List[int]:
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _tail_mean(history: List[Dict[str, float]], key: str, window: int) -> float:
    values = [float(item[key]) for item in history[-window:] if key in item]
    return mean(values) if values else float("nan")


def _append_summary(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _checkpoint_id(checkpoint: str | Path, *, output_root: Path) -> str:
    path = Path(checkpoint)
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except Exception:
        return str(path)


def _is_paper_config(config_path: str) -> bool:
    parts = Path(config_path).parts
    return "paper" in parts


def _apply_observation_ablation(cfg, mode: str) -> None:
    mode = str(mode or "no-cost-prior").strip().lower()
    cfg.observation.mode = "safe_observable"
    cfg.observation.include_oracle_cost = False
    cfg.observation.include_cost_prior_features = False
    cfg.reward.mode = "physical_weighted"
    cfg.reward.use_oracle_cost_components = False
    cfg.policy_regularization.enabled = False
    cfg.policy_regularization.mode = "none"
    cfg.policy_regularization.weight = 0.0
    cfg.algo.policy_head = "gnn_only"

    if mode == "no-cost-prior":
        return
    if mode == "cost-prior-features-only":
        cfg.observation.mode = "cost_prior_ablation"
        cfg.observation.include_cost_prior_features = True
        cfg.algo.policy_head = "hybrid_gnn_cost"
        return
    if mode == "cost-prior-regularization":
        cfg.observation.mode = "cost_prior_ablation"
        cfg.observation.include_cost_prior_features = True
        cfg.algo.policy_head = "hybrid_gnn_cost"
        cfg.policy_regularization.enabled = True
        cfg.policy_regularization.mode = "cost_prior_ce"
        cfg.policy_regularization.weight = max(0.0, float(cfg.policy_regularization.weight or 0.2)) or 0.2
        return
    if mode == "oracle-debug":
        cfg.observation.mode = "oracle_debug"
        cfg.observation.include_oracle_cost = True
        cfg.observation.include_cost_prior_features = True
        cfg.reward.mode = "oracle_aligned_cost"
        cfg.reward.use_oracle_cost_components = True
        cfg.algo.policy_head = "hybrid_gnn_cost"
        return
    raise ValueError(
        f"Unsupported --observation-ablation={mode!r}; expected "
        "'no-cost-prior'|'cost-prior-features-only'|'cost-prior-regularization'|'oracle-debug'."
    )


def _git_commit_hash(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _runtime_metadata(seed: int) -> Dict[str, Any]:
    commit_hash = _git_commit_hash(Path(__file__).resolve().parents[1]) or "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git_commit_hash": commit_hash,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint_into_trainer(trainer: HierarchicalTrainer, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location=trainer.device)
    if "encoder" in payload:
        trainer.encoder.load_state_dict(payload["encoder"], strict=False)
    upper = trainer.upper_agent
    lower = trainer.lower_agent
    for name in ["actor", "critic", "value", "q_net", "mixer", "target_q_net", "target_mixer"]:
        module = getattr(upper, name, None)
        key = f"upper_{name}"
        if module is not None and key in payload and hasattr(module, "load_state_dict"):
            module.load_state_dict(payload[key], strict=False)
    for name in ["encoder", "target_encoder", "actor", "critic", "target_actor", "target_critic"]:
        module = getattr(lower, name, None)
        key = f"lower_{name}"
        if module is not None and key in payload and hasattr(module, "load_state_dict"):
            module.load_state_dict(payload[key], strict=False)


def _mean_info(infos: List[Dict[str, torch.Tensor]], key: str) -> float:
    if not infos:
        return 0.0
    vals = [info[key].float().mean().detach().cpu() for info in infos if key in info]
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean())


def _reset_temporal_state(module: object) -> None:
    reset_fn = getattr(module, "reset_temporal_state", None)
    if callable(reset_fn):
        reset_fn()


@torch.no_grad()
def _evaluate_checkpoint(
    cfg,
    checkpoint: Path,
    *,
    seed: int,
    episodes: int,
    steps: int | None,
    n_leo: int | None,
) -> Dict[str, float]:
    eval_cfg = deepcopy(cfg)
    eval_cfg.total_episodes = int(max(1, episodes))
    eval_cfg.scenario.seed = int(seed)
    eval_cfg.upper_pretrain_episodes = 0
    eval_cfg.joint_train_episodes = 0
    eval_cfg.lower_training_enabled = False
    eval_cfg.lower_action_mode = "learned"
    if steps is not None:
        eval_cfg.steps_per_episode = int(steps)
        eval_cfg.scenario.episode_len = int(steps)
    if n_leo is not None:
        eval_cfg.scenario.n_leo = int(n_leo)

    trainer = HierarchicalTrainer(eval_cfg)
    _load_checkpoint_into_trainer(trainer, checkpoint)

    infos_all: List[Dict[str, torch.Tensor]] = []
    for _ in range(eval_cfg.total_episodes):
        _reset_temporal_state(trainer.encoder)
        _reset_temporal_state(getattr(trainer.lower_agent, "encoder", None))
        _reset_temporal_state(getattr(trainer.lower_agent, "target_encoder", None))
        obs, edge_index, edge_attr = trainer.env.reset()
        done = False
        while not done:
            embed = trainer.encoder(obs, edge_index, edge_attr)
            if hasattr(trainer.upper_agent, "actor"):
                logits = trainer.upper_agent.actor.compute_logits(embed, obs=obs)
            else:
                logits = trainer.upper_agent.q_net(embed)
            mask = upper_action_mask_from_obs(obs)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            upper_action = masked_logits.argmax(dim=-1)
            lower_action = trainer.lower_agent.act(
                embed.detach(),
                upper_action,
                explore=False,
                obs=obs,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            step = trainer.env.step(upper_action, lower_action)
            infos_all.append(step.info)
            obs, edge_index, edge_attr, done = step.obs, step.edge_index, step.edge_attr, step.done

    return {
        "normalized_system_cost": _mean_info(infos_all, "normalized_system_cost"),
        "mean_system_cost": _mean_info(infos_all, "system_cost"),
        "mean_delay": _mean_info(infos_all, "delay"),
        "mean_delay_s": _mean_info(infos_all, "physical_delay_s"),
        "mean_energy": _mean_info(infos_all, "energy"),
        "mean_energy_j": _mean_info(infos_all, "physical_energy_j"),
        "mean_queue": _mean_info(infos_all, "queue"),
        "mean_queue_length_tasks": _mean_info(infos_all, "physical_queue_length_tasks"),
        "mean_feasibility": _mean_info(infos_all, "feasible"),
        "mean_deadline_exceedance": _mean_info(infos_all, "deadline_exceedance"),
        "mean_deadline_violation_ratio": _mean_info(infos_all, "deadline_violation_flag"),
        "mean_deadline_violation": _mean_info(infos_all, "deadline_violation"),
    }


def _build_train_row(
    *,
    status: str,
    error: str,
    seed: int,
    phase: str,
    upper: str,
    lower: str,
    cfg,
    output_dir: Path,
    elapsed: float,
    history: List[Dict[str, float]],
    summary_window: int,
    split_id: str,
    selected_checkpoint: str,
    checkpoint_selection_mode: str,
    checkpoint_id: str,
    experiment_contract_sha256: str,
) -> Dict[str, object]:
    return {
        "status": status,
        "error": error,
        "phase": phase,
        "protocol_role": phase,
        "split_id": split_id,
        "seed": seed,
        "train_seed": seed,
        "eval_seed": "",
        "upper_algo": upper,
        "lower_algo": lower,
        "checkpoint_id": checkpoint_id,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "experiment_contract_sha256": experiment_contract_sha256,
        "episodes": cfg.total_episodes,
        "episode_len": cfg.scenario.episode_len,
        "n_leo": cfg.scenario.n_leo,
        "requested_device": getattr(cfg, "requested_device", cfg.device),
        "actual_device": getattr(cfg, "actual_device", cfg.device),
        "device_fallback_reason": getattr(cfg, "device_fallback_reason", ""),
        "observation_mode": str(getattr(cfg.observation, "mode", "")),
        "include_oracle_cost": bool(getattr(cfg.observation, "include_oracle_cost", False)),
        "include_cost_prior_features": bool(getattr(cfg.observation, "include_cost_prior_features", False)),
        "output_dir": str(output_dir),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "selected_checkpoint": selected_checkpoint,
        "elapsed_sec": round(elapsed, 4),
        "final_normalized_system_cost": _tail_mean(history, "normalized_system_cost", 1) if history else "",
        "final_mean_deadline_exceedance": _tail_mean(history, "mean_deadline_exceedance", 1) if history else "",
        "final_mean_deadline_violation_ratio": _tail_mean(history, "mean_deadline_violation_ratio", 1) if history else "",
        "final_mean_delay_s": _tail_mean(history, "mean_delay_s", 1) if history else "",
        "final_mean_energy_j": _tail_mean(history, "mean_energy_j", 1) if history else "",
        "final_mean_queue_length_tasks": _tail_mean(history, "mean_queue_length_tasks", 1) if history else "",
        "final_mean_delay": _tail_mean(history, "mean_delay", 1) if history else "",
        "final_mean_energy": _tail_mean(history, "mean_energy", 1) if history else "",
        "final_mean_queue": _tail_mean(history, "mean_queue", 1) if history else "",
        "final_mean_system_cost": _tail_mean(history, "mean_system_cost", 1) if history else "",
        "final_mean_deadline_violation": _tail_mean(history, "mean_deadline_violation", 1) if history else "",
        "final_mean_feasibility": _tail_mean(history, "mean_feasibility", 1) if history else "",
        "tail_mean_delay": _tail_mean(history, "mean_delay", summary_window) if history else "",
        "tail_mean_energy": _tail_mean(history, "mean_energy", summary_window) if history else "",
        "tail_mean_queue": _tail_mean(history, "mean_queue", summary_window) if history else "",
        "tail_normalized_system_cost": _tail_mean(history, "normalized_system_cost", summary_window) if history else "",
        "tail_mean_system_cost": _tail_mean(history, "mean_system_cost", summary_window) if history else "",
        "tail_mean_deadline_violation": _tail_mean(history, "mean_deadline_violation", summary_window) if history else "",
        "tail_mean_feasibility": _tail_mean(history, "mean_feasibility", summary_window) if history else "",
    }


def _build_eval_row(
    *,
    status: str,
    error: str,
    phase: str,
    split_id: str,
    seed: int,
    train_seed: int,
    upper: str,
    lower: str,
    checkpoint: str,
    selected_checkpoint: str,
    checkpoint_selection_mode: str,
    checkpoint_id: str,
    observation_ablation: str,
    cfg_eval,
    experiment_contract_sha256: str,
    metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    metrics = metrics or {}
    return {
        "status": status,
        "error": error,
        "phase": phase,
        "protocol_role": phase,
        "split_id": split_id,
        "seed": int(seed),
        "train_seed": int(train_seed),
        "eval_seed": int(seed),
        "upper_algo": upper,
        "lower_algo": lower,
        "checkpoint_id": checkpoint_id,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "experiment_contract_sha256": experiment_contract_sha256,
        "checkpoint": checkpoint,
        "selected_checkpoint": selected_checkpoint,
        "observation_ablation": observation_ablation,
        "observation_mode": str(getattr(cfg_eval.observation, "mode", "")),
        "include_oracle_cost": bool(getattr(cfg_eval.observation, "include_oracle_cost", False)),
        "include_cost_prior_features": bool(getattr(cfg_eval.observation, "include_cost_prior_features", False)),
        "final_normalized_system_cost": metrics.get("normalized_system_cost", ""),
        "final_mean_deadline_exceedance": metrics.get("mean_deadline_exceedance", ""),
        "final_mean_deadline_violation_ratio": metrics.get("mean_deadline_violation_ratio", ""),
        "final_mean_delay_s": metrics.get("mean_delay_s", ""),
        "final_mean_energy_j": metrics.get("mean_energy_j", ""),
        "final_mean_queue_length_tasks": metrics.get("mean_queue_length_tasks", ""),
        "final_mean_delay": metrics.get("mean_delay", ""),
        "final_mean_energy": metrics.get("mean_energy", ""),
        "final_mean_queue": metrics.get("mean_queue", ""),
        "final_mean_system_cost": metrics.get("mean_system_cost", ""),
        "final_mean_deadline_violation": metrics.get("mean_deadline_violation", ""),
        "final_mean_feasibility": metrics.get("mean_feasibility", ""),
    }


def _resolve_split_seed_sets(base_cfg, args) -> Tuple[List[int], List[int], List[int], bool]:
    cli_train = _parse_seed_list(args.train_seeds)
    cli_val = _parse_seed_list(args.val_seeds)
    cli_test = _parse_seed_list(args.test_seeds)

    cfg_split = getattr(getattr(base_cfg, "experiment", None), "split", None)
    cfg_train = list(getattr(cfg_split, "train_seeds", []) or [])
    cfg_val = list(getattr(cfg_split, "val_seeds", []) or [])
    cfg_test = list(getattr(cfg_split, "test_seeds", []) or [])
    allow_overlap = bool(args.allow_debug_seed_overlap or bool(getattr(cfg_split, "allow_debug_seed_overlap", False)))

    if cli_train or cli_val or cli_test:
        train, val, test = cli_train, cli_val, cli_test
    elif cfg_train or cfg_val or cfg_test:
        train, val, test = cfg_train, cfg_val, cfg_test
    else:
        train = _parse_seed_list(args.seeds)
        val = []
        test = []

    if not train:
        raise ValueError("No training seeds resolved; provide --seeds or --train-seeds (or config.experiment.split.train_seeds).")

    if not allow_overlap:
        train_set, val_set, test_set = set(train), set(val), set(test)
        overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        if overlap:
            raise ValueError(
                f"train/val/test seeds overlap detected: {sorted(overlap)}. "
                "Use --allow-debug-seed-overlap to override explicitly."
            )

    return train, val, test, allow_overlap


def _pick_best_checkpoint(rows: List[Dict[str, object]], *, key: str = "final_mean_system_cost") -> str:
    candidates = [r for r in rows if r.get("status") == "ok" and r.get("checkpoint")]
    if not candidates:
        return ""
    return str(min(candidates, key=lambda r: float(r.get(key) or 1.0e18)).get("checkpoint", ""))


def _row_metric(row: Mapping[str, object], key: str = "final_normalized_system_cost") -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        raw = row.get("final_mean_system_cost", 1.0e18)
    return float(raw or 1.0e18)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriSatFlow upper/lower algorithm-combination sweep and persist CSV results.")
    parser.add_argument("--config", type=str, default="trisatflow/configs/small.yaml")
    parser.add_argument("--upper", type=str, default="mappo,ippo,iql,vdn,qmix", help="Comma list or 'all'.")
    parser.add_argument("--lower", type=str, default="maddpg,iddpg,masac,isac", help="Comma list or 'all'.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-leo", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="7", help="Comma-separated integer seeds (legacy mode or train seeds fallback).")
    parser.add_argument("--train-seeds", type=str, default="", help="Comma-separated train seeds for split protocol.")
    parser.add_argument("--val-seeds", type=str, default="", help="Comma-separated validation seeds for checkpoint selection.")
    parser.add_argument("--test-seeds", type=str, default="", help="Comma-separated test seeds for fixed-checkpoint evaluation.")
    parser.add_argument(
        "--checkpoint-selection",
        type=str,
        default="per_train_seed",
        choices=["per_train_seed", "best_val_global"],
        help="Checkpoint protocol: paper-ready inference uses per_train_seed; best_val_global is debug/deployment replay only.",
    )
    parser.add_argument("--allow-debug-seed-overlap", action="store_true", help="Allow train/val/test seed overlap for debug only.")
    parser.add_argument("--summary-window", type=int, default=5)
    parser.add_argument("--output-root", type=str, default="outputs/algorithm_sweep")
    parser.add_argument("--device", type=str, default=None, help="Override config device: cpu|cuda|cuda:0|auto")
    parser.add_argument(
        "--observation-ablation",
        type=str,
        default="no-cost-prior",
        choices=["no-cost-prior", "cost-prior-features-only", "cost-prior-regularization", "oracle-debug"],
        help="Privileged-information ablation mode.",
    )
    parser.add_argument("--print-registry", action="store_true")
    args = parser.parse_args()

    if args.print_registry:
        print(json.dumps({k: [choice.__dict__ for choice in v] for k, v in supported_algorithm_matrix().items()}, ensure_ascii=False, indent=2))
        return

    uppers = _split_algos(args.upper, upper_algorithm_names())
    lowers = _split_algos(args.lower, lower_algorithm_names())
    base_cfg = load_config(args.config)
    if args.checkpoint_selection == "best_val_global" and _is_paper_config(args.config):
        raise ValueError("--checkpoint-selection best_val_global is not allowed for paper config inference; use per_train_seed.")
    output_root = Path(args.output_root)
    summary_csv = output_root / "sweep_summary.csv"

    train_seeds, val_seeds, test_seeds, allow_overlap = _resolve_split_seed_sets(base_cfg, args)
    split_id = "legacy" if (not val_seeds and not test_seeds and args.train_seeds.strip() == "" and args.val_seeds.strip() == "" and args.test_seeds.strip() == "") else "train_val_test"

    rows: List[Dict[str, object]] = []
    is_legacy_layout = split_id == "legacy"
    combinations = [(upper, lower) for upper in uppers for lower in lowers]
    print(
        json.dumps(
            {
                "event": "SWEEP_PLAN",
                "upper_algos": uppers,
                "lower_algos": lowers,
                "combinations": [f"{upper}/{lower}" for upper, lower in combinations],
                "n_combinations": len(combinations),
                "train_seeds": [int(s) for s in train_seeds],
                "val_seeds": [int(s) for s in val_seeds],
                "test_seeds": [int(s) for s in test_seeds],
                "split_id": split_id,
                "checkpoint_selection": args.checkpoint_selection,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for upper in uppers:
        for lower in lowers:
            train_rows: List[Dict[str, object]] = []
            val_rows: List[Dict[str, object]] = []
            test_rows: List[Dict[str, object]] = []

            for seed in train_seeds:
                cfg = deepcopy(base_cfg)
                cfg.algo.upper_algo = upper
                cfg.algo.lower_algo = lower
                cfg.scenario.seed = int(seed)
                if args.episodes is not None:
                    cfg.total_episodes = args.episodes
                if args.steps is not None:
                    cfg.steps_per_episode = int(args.steps)
                    cfg.scenario.episode_len = int(args.steps)
                if args.n_leo is not None:
                    cfg.scenario.n_leo = args.n_leo
                if args.device is not None:
                    cfg.device = str(args.device)
                _apply_observation_ablation(cfg, args.observation_ablation)
                if is_legacy_layout:
                    out_dir = output_root / f"seed_{seed}" / f"upper_{upper}__lower_{lower}"
                else:
                    out_dir = output_root / "train" / f"seed_{seed}" / f"upper_{upper}__lower_{lower}"
                cfg.output_dir = str(out_dir)
                contract, contract_digest = write_contract_artifacts(out_dir, cfg, base_dir=Path(__file__).resolve().parents[1])

                start = time.time()
                status = "ok"
                error = ""
                history: List[Dict[str, float]] = []
                try:
                    trainer = HierarchicalTrainer(cfg)
                    history = trainer.train()
                    trainer.save_checkpoint(Path(cfg.output_dir) / "checkpoint.pt")
                except Exception as exc:
                    status = "failed"
                    error = repr(exc)
                elapsed = time.time() - start

                runtime_meta = _runtime_metadata(seed=int(seed))
                uses_privileged_info = bool(
                    str(getattr(cfg.observation, "mode", "")).strip().lower() == "oracle_debug"
                    or bool(getattr(cfg.observation, "include_oracle_cost", False))
                    or bool(getattr(cfg.reward, "use_oracle_cost_components", False))
                )
                runtime_meta.update(
                    {
                        "phase": "train",
                        "upper_algo": upper,
                        "lower_algo": lower,
                        "observation_ablation": args.observation_ablation,
                        "uses_privileged_info": uses_privileged_info,
                        "resolved_config": asdict(cfg),
                        "experiment_contract": contract,
                        "experiment_contract_sha256": contract_digest,
                    }
                )
                _write_json(Path(cfg.output_dir) / "run_metadata.json", runtime_meta)

                row = _build_train_row(
                    status=status,
                    error=error,
                    seed=int(seed),
                    phase="train",
                    upper=upper,
                    lower=lower,
                    cfg=cfg,
                    output_dir=out_dir,
                    elapsed=elapsed,
                    history=history,
                    summary_window=args.summary_window,
                    split_id=split_id,
                    selected_checkpoint="",
                    checkpoint_selection_mode=args.checkpoint_selection,
                    checkpoint_id=_checkpoint_id(out_dir / "checkpoint.pt", output_root=output_root),
                    experiment_contract_sha256=contract_digest,
                )
                _append_summary(summary_csv, row)
                rows.append(row)
                train_rows.append(row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))

            selected_checkpoint = ""
            selected_by_train_seed: Dict[int, str] = {}
            if args.checkpoint_selection == "per_train_seed":
                for train_row in train_rows:
                    if train_row.get("status") != "ok" or not train_row.get("checkpoint"):
                        continue
                    train_seed = int(train_row["train_seed"])
                    ckpt = Path(str(train_row["checkpoint"]))
                    if not ckpt.exists():
                        continue
                    train_digest = str(train_row.get("experiment_contract_sha256", ""))
                    checkpoint_id = str(train_row.get("checkpoint_id", _checkpoint_id(ckpt, output_root=output_root)))
                    val_metrics_for_seed: List[float] = []
                    for seed in val_seeds:
                        cfg_eval = deepcopy(base_cfg)
                        cfg_eval.algo.upper_algo = upper
                        cfg_eval.algo.lower_algo = lower
                        if args.steps is not None:
                            cfg_eval.steps_per_episode = int(args.steps)
                            cfg_eval.scenario.episode_len = int(args.steps)
                        if args.n_leo is not None:
                            cfg_eval.scenario.n_leo = int(args.n_leo)
                        if args.device is not None:
                            cfg_eval.device = str(args.device)
                        _apply_observation_ablation(cfg_eval, args.observation_ablation)
                        try:
                            metrics = _evaluate_checkpoint(
                                cfg_eval,
                                ckpt,
                                seed=int(seed),
                                episodes=int(args.episodes or 1),
                                steps=int(args.steps) if args.steps is not None else None,
                                n_leo=int(args.n_leo) if args.n_leo is not None else None,
                            )
                            val_metrics_for_seed.append(float(metrics.get("normalized_system_cost", metrics["mean_system_cost"])))
                            row = _build_eval_row(
                                status="ok",
                                error="",
                                phase="val",
                                split_id=split_id,
                                seed=int(seed),
                                train_seed=train_seed,
                                upper=upper,
                                lower=lower,
                                checkpoint=str(ckpt),
                                selected_checkpoint="",
                                checkpoint_selection_mode=args.checkpoint_selection,
                                checkpoint_id=checkpoint_id,
                                observation_ablation=args.observation_ablation,
                                cfg_eval=cfg_eval,
                                experiment_contract_sha256=train_digest,
                                metrics=metrics,
                            )
                        except Exception as exc:
                            row = _build_eval_row(
                                status="failed",
                                error=repr(exc),
                                phase="val",
                                split_id=split_id,
                                seed=int(seed),
                                train_seed=train_seed,
                                upper=upper,
                                lower=lower,
                                checkpoint=str(ckpt),
                                selected_checkpoint="",
                                checkpoint_selection_mode=args.checkpoint_selection,
                                checkpoint_id=checkpoint_id,
                                observation_ablation=args.observation_ablation,
                                cfg_eval=cfg_eval,
                                experiment_contract_sha256=train_digest,
                            )
                        _append_summary(summary_csv, row)
                        rows.append(row)
                        val_rows.append(row)
                        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    if val_seeds and not val_metrics_for_seed:
                        continue
                    selected_by_train_seed[train_seed] = str(ckpt)

            elif val_seeds:
                candidate_ckpts = [Path(str(r.get("checkpoint", ""))) for r in train_rows if r.get("status") == "ok"]
                candidate_ckpts = [p for p in candidate_ckpts if p.exists()]
                if candidate_ckpts:
                    val_score: Dict[str, List[float]] = {str(p): [] for p in candidate_ckpts}
                    for ckpt in candidate_ckpts:
                        owner = next((r for r in train_rows if str(r.get("checkpoint", "")) == str(ckpt)), {})
                        train_seed = int(owner.get("train_seed", owner.get("seed", 0)) or 0)
                        train_digest = str(owner.get("experiment_contract_sha256", ""))
                        checkpoint_id = str(owner.get("checkpoint_id", _checkpoint_id(ckpt, output_root=output_root)))
                        for seed in val_seeds:
                            cfg_eval = deepcopy(base_cfg)
                            cfg_eval.algo.upper_algo = upper
                            cfg_eval.algo.lower_algo = lower
                            if args.steps is not None:
                                cfg_eval.steps_per_episode = int(args.steps)
                                cfg_eval.scenario.episode_len = int(args.steps)
                            if args.n_leo is not None:
                                cfg_eval.scenario.n_leo = int(args.n_leo)
                            if args.device is not None:
                                cfg_eval.device = str(args.device)
                            _apply_observation_ablation(cfg_eval, args.observation_ablation)
                            try:
                                metrics = _evaluate_checkpoint(
                                    cfg_eval,
                                    ckpt,
                                    seed=int(seed),
                                    episodes=int(args.episodes or 1),
                                    steps=int(args.steps) if args.steps is not None else None,
                                    n_leo=int(args.n_leo) if args.n_leo is not None else None,
                                )
                                val_score[str(ckpt)].append(float(metrics.get("normalized_system_cost", metrics["mean_system_cost"])))
                                row = _build_eval_row(
                                    status="ok",
                                    error="",
                                    phase="val",
                                    split_id=split_id,
                                    seed=int(seed),
                                    train_seed=train_seed,
                                    upper=upper,
                                    lower=lower,
                                    checkpoint=str(ckpt),
                                    selected_checkpoint="",
                                    checkpoint_selection_mode=args.checkpoint_selection,
                                    checkpoint_id=checkpoint_id,
                                    observation_ablation=args.observation_ablation,
                                    cfg_eval=cfg_eval,
                                    experiment_contract_sha256=train_digest,
                                    metrics=metrics,
                                )
                            except Exception as exc:
                                row = _build_eval_row(
                                    status="failed",
                                    error=repr(exc),
                                    phase="val",
                                    split_id=split_id,
                                    seed=int(seed),
                                    train_seed=train_seed,
                                    upper=upper,
                                    lower=lower,
                                    checkpoint=str(ckpt),
                                    selected_checkpoint="",
                                    checkpoint_selection_mode=args.checkpoint_selection,
                                    checkpoint_id=checkpoint_id,
                                    observation_ablation=args.observation_ablation,
                                    cfg_eval=cfg_eval,
                                    experiment_contract_sha256=train_digest,
                                )
                            _append_summary(summary_csv, row)
                            rows.append(row)
                            val_rows.append(row)
                            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    successful_val_scores = {ckpt: scores for ckpt, scores in val_score.items() if scores}
                    if successful_val_scores:
                        selected_checkpoint = min(successful_val_scores.items(), key=lambda kv: mean(kv[1]))[0]
            else:
                selected_checkpoint = _pick_best_checkpoint(train_rows, key="final_normalized_system_cost")
                if selected_checkpoint:
                    owner = next((r for r in train_rows if str(r.get("checkpoint", "")) == selected_checkpoint), {})
                    if owner:
                        selected_by_train_seed[int(owner.get("train_seed", owner.get("seed", 0)) or 0)] = selected_checkpoint

            if args.checkpoint_selection == "best_val_global" and selected_checkpoint:
                owner = next((r for r in train_rows if str(r.get("checkpoint", "")) == selected_checkpoint), {})
                selected_by_train_seed = {int(owner.get("train_seed", owner.get("seed", 0)) or 0): selected_checkpoint}

            if selected_by_train_seed and test_seeds:
                for train_seed, checkpoint_for_seed in sorted(selected_by_train_seed.items()):
                    owner = next((r for r in train_rows if int(r.get("train_seed", r.get("seed", 0)) or 0) == int(train_seed)), {})
                    train_digest = str(owner.get("experiment_contract_sha256", ""))
                    checkpoint_id = str(owner.get("checkpoint_id", _checkpoint_id(checkpoint_for_seed, output_root=output_root)))
                    for seed in test_seeds:
                        cfg_eval = deepcopy(base_cfg)
                        cfg_eval.algo.upper_algo = upper
                        cfg_eval.algo.lower_algo = lower
                        if args.steps is not None:
                            cfg_eval.steps_per_episode = int(args.steps)
                            cfg_eval.scenario.episode_len = int(args.steps)
                        if args.n_leo is not None:
                            cfg_eval.scenario.n_leo = int(args.n_leo)
                        if args.device is not None:
                            cfg_eval.device = str(args.device)
                        _apply_observation_ablation(cfg_eval, args.observation_ablation)
                        try:
                            metrics = _evaluate_checkpoint(
                                cfg_eval,
                                Path(checkpoint_for_seed),
                                seed=int(seed),
                                episodes=int(args.episodes or 1),
                                steps=int(args.steps) if args.steps is not None else None,
                                n_leo=int(args.n_leo) if args.n_leo is not None else None,
                            )
                            row = _build_eval_row(
                                status="ok",
                                error="",
                                phase="test",
                                split_id=split_id,
                                seed=int(seed),
                                train_seed=int(train_seed),
                                upper=upper,
                                lower=lower,
                                checkpoint=checkpoint_for_seed,
                                selected_checkpoint=checkpoint_for_seed,
                                checkpoint_selection_mode=args.checkpoint_selection,
                                checkpoint_id=checkpoint_id,
                                observation_ablation=args.observation_ablation,
                                cfg_eval=cfg_eval,
                                experiment_contract_sha256=train_digest,
                                metrics=metrics,
                            )
                        except Exception as exc:
                            row = _build_eval_row(
                                status="failed",
                                error=repr(exc),
                                phase="test",
                                split_id=split_id,
                                seed=int(seed),
                                train_seed=int(train_seed),
                                upper=upper,
                                lower=lower,
                                checkpoint=checkpoint_for_seed,
                                selected_checkpoint=checkpoint_for_seed,
                                checkpoint_selection_mode=args.checkpoint_selection,
                                checkpoint_id=checkpoint_id,
                                observation_ablation=args.observation_ablation,
                                cfg_eval=cfg_eval,
                                experiment_contract_sha256=train_digest,
                            )
                        _append_summary(summary_csv, row)
                        rows.append(row)
                        test_rows.append(row)
                        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

            protocol_payload = {
                "split_id": split_id,
                "allow_debug_seed_overlap": bool(allow_overlap),
                "train_seeds": [int(s) for s in train_seeds],
                "val_seeds": [int(s) for s in val_seeds],
                "test_seeds": [int(s) for s in test_seeds],
                "upper_algo": upper,
                "lower_algo": lower,
                "checkpoint_selection_mode": args.checkpoint_selection,
                "selected_checkpoint": selected_checkpoint,
                "selected_checkpoints_by_train_seed": {str(k): v for k, v in sorted(selected_by_train_seed.items())},
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
            }
            protocol_path = output_root / f"protocol_{upper}_{lower}.json"
            _write_json(protocol_path, protocol_payload)

    print(f"SWEEP_OK summary_csv={summary_csv} runs={len(rows)}")


if __name__ == "__main__":
    main()
