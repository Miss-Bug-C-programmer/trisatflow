from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from typing import Mapping

import torch

from trisatflow.baselines import baseline_registry, evaluate_named_baselines
from trisatflow.config import load_config, save_config
from trisatflow.envs.physical_metrics import metric_schema_manifest
from trisatflow.experiment_contracts import write_contract_artifacts


def _split(value: str, choices):
    if value == "all":
        return list(choices)
    names = [item.strip() for item in value.split(",") if item.strip()]
    bad = [name for name in names if name not in choices]
    if bad:
        raise ValueError(f"Unknown baselines {bad}; choices={sorted(choices)}")
    return names


def _trace_size_mb(trace_path: str) -> float:
    if not trace_path:
        return 0.0
    path = Path(trace_path)
    return path.stat().st_size / (1024.0 * 1024.0) if path.exists() else 0.0


def _print_progress(payload: Mapping[str, object]) -> None:
    event = str(payload.get("event", "progress"))
    if event == "start":
        print(
            "[baseline-eval:start] "
            f"device={payload.get('device')} baselines={payload.get('baselines')} seeds={payload.get('seeds')} "
            f"episodes_per_seed={payload.get('episodes_per_seed')} steps_per_episode={payload.get('steps_per_episode')} "
            f"total_episodes={payload.get('total_episodes')} total_steps={payload.get('total_steps')}",
            flush=True,
        )
        return
    if event == "baseline_complete":
        print(
            "[baseline-eval:baseline-complete] "
            + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return
    if event == "progress":
        print(
            "[baseline-eval:progress] "
            f"baseline={payload.get('baseline')} [{payload.get('baseline_index')}/{payload.get('baseline_count')}] "
            f"seed={payload.get('seed')} [{payload.get('seed_index')}/{payload.get('seed_count')}] "
            f"episode={payload.get('episode')}/{payload.get('episodes')} "
            f"global={float(payload.get('global_progress_pct', 0.0)):.2f}% "
            f"elapsed_s={float(payload.get('global_elapsed_s', 0.0)):.1f} "
            f"steps_per_s={float(payload.get('global_steps_per_s', 0.0)):.2f}",
            flush=True,
        )
        return
    print("[baseline-eval] " + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-DRL rule baselines for TriSatFlow and write CSV results.")
    parser.add_argument("--config", type=str, default="trisatflow/configs/small.yaml")
    parser.add_argument("--baselines", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="7,11,19")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-leo", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/rule_baselines")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N completed episodes per baseline/seed. Use 0 to print only boundaries.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Atomically persist partial CSV/status files every N completed episodes globally.")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be > 0, got {args.episodes}")
    if args.steps is not None and args.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {args.steps}")
    if args.n_leo is not None and args.n_leo <= 0:
        raise ValueError(f"--n-leo must be > 0, got {args.n_leo}")
    if args.progress_every < 0:
        raise ValueError(f"--progress-every must be >= 0, got {args.progress_every}")
    if args.checkpoint_every <= 0:
        raise ValueError(f"--checkpoint-every must be > 0, got {args.checkpoint_every}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested via --device={args.device!r}, but torch.cuda.is_available() is False")

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.scenario.episode_len = args.steps
        cfg.steps_per_episode = int(args.steps)
    if args.n_leo is not None:
        cfg.scenario.n_leo = args.n_leo
    output_dir = Path(args.output_dir)
    cfg.output_dir = str(output_dir)
    save_config(cfg, output_dir / "resolved_config.yaml")
    _contract, contract_digest = write_contract_artifacts(output_dir, cfg, base_dir=Path(__file__).resolve().parents[1])
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    names = _split(args.baselines, baseline_registry().keys())
    if not seeds:
        raise ValueError("--seeds resolved to an empty list")

    total_episodes = len(names) * len(seeds) * args.episodes
    total_steps = total_episodes * int(cfg.scenario.episode_len)
    print(
        "[baseline-eval:preflight] "
        f"config={args.config} output_dir={output_dir} requested_device={args.device} "
        f"n_leo={cfg.scenario.n_leo} trace={cfg.scenario.topology_trace_path or '<analytic>'} "
        f"trace_size_mb={_trace_size_mb(cfg.scenario.topology_trace_path):.1f} "
        f"baselines={len(names)} seeds={len(seeds)} episodes_per_seed={args.episodes} "
        f"steps_per_episode={cfg.scenario.episode_len} total_episodes={total_episodes} total_steps={total_steps}",
        flush=True,
    )
    if args.device.startswith("cuda"):
        print(
            "[baseline-eval:warning] Rule baselines execute many small control-flow tensor operations. "
            "CUDA is supported, but CPU is normally faster and avoids device-synchronization overhead. "
            "Use --device cpu for the paper-ready offline baseline sweep unless profiling on your server proves otherwise.",
            flush=True,
        )

    started = time.perf_counter()
    episode_rows, summary_rows = evaluate_named_baselines(
        cfg,
        names,
        seeds,
        args.episodes,
        device=args.device,
        output_dir=output_dir,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        progress_callback=_print_progress,
    )
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    metric_schema = metric_schema_manifest(cfg)
    manifest = {
        "status": "ok",
        "entry": "scripts/evaluate_rule_baselines.py",
        "config": args.config,
        "output_dir": str(output_dir),
        "baselines": names,
        "seeds": seeds,
        "episodes": int(args.episodes),
        "steps": int(cfg.scenario.episode_len),
        "n_leo": int(cfg.scenario.n_leo),
        "artifacts": {
            "episode_csv": str(output_dir / "baseline_episode_metrics.csv"),
            "summary_csv": str(output_dir / "baseline_summary.csv"),
            "status_json": str(output_dir / "baseline_eval_status.json"),
            "resolved_config": str(output_dir / "resolved_config.yaml"),
            "experiment_contract": str(output_dir / "experiment_contract.json"),
            "experiment_contract_sha256": str(output_dir / "experiment_contract_sha256.txt"),
        },
        "experiment_contract_sha256": contract_digest,
        **metric_schema,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"BASELINE_EVAL_OK elapsed_s={time.perf_counter() - started:.1f} rows={len(episode_rows)} "
        f"summary_csv={output_dir / 'baseline_summary.csv'} "
        f"episode_csv={output_dir / 'baseline_episode_metrics.csv'} "
        f"status_json={output_dir / 'baseline_eval_status.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
