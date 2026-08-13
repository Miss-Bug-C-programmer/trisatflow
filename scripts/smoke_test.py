from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import platform
import socket
from datetime import datetime, timezone

import torch

from trisatflow.config import AlgoConfig, ModelConfig, ScenarioConfig, TemporalModelConfig, TrainConfig, load_config
from trisatflow.experiment_contracts import write_contract_artifacts
from trisatflow.envs.physical_metrics import metric_schema_manifest
from trisatflow.agents import HierarchicalTrainer
from trisatflow.algorithms import lower_algorithm_names, upper_algorithm_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--n-leo", type=int, default=4)
    parser.add_argument("--upper-algo", type=str, default="mappo", choices=upper_algorithm_names())
    parser.add_argument("--lower-algo", type=str, default="maddpg", choices=lower_algorithm_names())
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|cuda:0|auto")
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_test")
    parser.add_argument("--config", type=str, default="", help="Optional base TrainConfig YAML.")
    parser.add_argument(
        "--topology-encoder",
        type=str,
        default="static_gnn",
        choices=["no_gnn", "static_gnn", "temporal_gnn"],
        help="Encoder ablation mode.",
    )
    parser.add_argument("--history-len", type=int, default=4, help="Temporal history length when temporal_gnn is enabled.")
    parser.add_argument("--temporal-hidden-dim", type=int, default=128, help="Temporal GRU hidden size.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        cfg = load_config(args.config)
        cfg.total_episodes = int(args.episodes)
        cfg.log_interval = 1
        cfg.device = str(args.device)
        cfg.output_dir = args.output_dir
        cfg.scenario.n_leo = int(args.n_leo)
        cfg.scenario.episode_len = int(args.steps)
        cfg.steps_per_episode = int(args.steps)
        cfg.algo.upper_algo = str(args.upper_algo)
        cfg.algo.lower_algo = str(args.lower_algo)
    else:
        cfg = TrainConfig(
            total_episodes=args.episodes,
            log_interval=1,
            device=str(args.device),
            output_dir=args.output_dir,
            scenario=ScenarioConfig(n_leo=args.n_leo, episode_len=args.steps, seed=11),
            model=ModelConfig(
                topology_encoder=str(args.topology_encoder),
                temporal=TemporalModelConfig(
                    enabled=str(args.topology_encoder) == "temporal_gnn",
                    type="gnn_gru",
                    history_len=int(args.history_len),
                    hidden_dim=int(args.temporal_hidden_dim),
                ),
            ),
            algo=AlgoConfig(
                upper_algo=args.upper_algo,
                lower_algo=args.lower_algo,
                gnn_hidden_dim=32,
                policy_hidden_dim=64,
                lower_batch_size=8,
                lower_warmup=8,
                upper_batch_size=8,
                upper_warmup=8,
                exploration_noise=0.05,
            ),
        )
    trainer = HierarchicalTrainer(cfg)
    history = trainer.train()
    _contract, contract_digest = write_contract_artifacts(output_dir, cfg, base_dir=Path(__file__).resolve().parents[1])
    ckpt = Path(args.output_dir) / "smoke_checkpoint.pt"
    trainer.save_checkpoint(ckpt)
    metrics_csv = Path(args.output_dir) / "metrics.csv"
    assert len(history) == args.episodes
    assert ckpt.exists()
    assert metrics_csv.exists()
    assert history[-1]["mean_queue"] >= 0.0
    assert history[-1].get("observation_mode") == "safe_observable"
    assert float(history[-1].get("include_oracle_cost", 0.0)) == 0.0
    assert float(history[-1].get("include_cost_prior_features", 0.0)) == 0.0
    metric_schema = metric_schema_manifest(cfg)
    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "hostname": socket.gethostname(),
        "requested_device": cfg.requested_device,
        "actual_device": cfg.actual_device,
        "uses_privileged_info": False,
        "observation_mode": "safe_observable",
        "metrics_csv": str(metrics_csv),
        "checkpoint": str(ckpt),
        "resolved_config": str(output_dir / "resolved_config.yaml"),
        "experiment_contract": str(output_dir / "experiment_contract.json"),
        "experiment_contract_sha256": contract_digest,
        **metric_schema,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "ok",
        "entry": "scripts/smoke_test.py",
        "smoke_marker": "SMOKE_TEST_OK",
        "output_dir": str(output_dir),
        "artifacts": {
            "metrics_csv": str(metrics_csv),
            "checkpoint": str(ckpt),
            "run_metadata": str(output_dir / "run_metadata.json"),
            "resolved_config": str(output_dir / "resolved_config.yaml"),
            "experiment_contract": str(output_dir / "experiment_contract.json"),
            "experiment_contract_sha256": str(output_dir / "experiment_contract_sha256.txt"),
        },
        "experiment_contract_sha256": contract_digest,
        **metric_schema,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "SMOKE_TEST_OK "
        f"upper={args.upper_algo} lower={args.lower_algo} "
        f"requested_device={cfg.requested_device} actual_device={cfg.actual_device} "
        f"metrics_csv={metrics_csv} checkpoint={ckpt}"
    )


if __name__ == "__main__":
    main()
