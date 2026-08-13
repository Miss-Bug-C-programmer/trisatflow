from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from trisatflow.agents import HierarchicalTrainer
from trisatflow.algorithms import lower_algorithm_names, upper_algorithm_names
from trisatflow.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TriSatFlow hierarchical MARL prototype")
    parser.add_argument("--config", type=str, default=None, help="YAML config path. Defaults to dataclass values.")
    parser.add_argument("--upper-algo", type=str, default=None, choices=upper_algorithm_names())
    parser.add_argument("--lower-algo", type=str, default=None, choices=lower_algorithm_names())
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Override config device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.upper_algo is not None:
        cfg.algo.upper_algo = args.upper_algo
    if args.lower_algo is not None:
        cfg.algo.lower_algo = args.lower_algo
    if args.episodes is not None:
        cfg.total_episodes = args.episodes
    if args.device is not None:
        cfg.device = args.device
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    trainer = HierarchicalTrainer(cfg)
    trainer.train()
    trainer.save_checkpoint(f"{cfg.output_dir}/checkpoint.pt")
    print(f"TRAIN_OK metrics_csv={cfg.output_dir}/metrics.csv checkpoint={cfg.output_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
