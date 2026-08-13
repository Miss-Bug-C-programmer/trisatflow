from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.agents.flat_hybrid_actor_critic import FlatHybridActorCriticAgent
from trisatflow.agents.hybrid_pdqn import HybridPDQNAgent
from trisatflow.baselines.optimized_dpp import OptimizedLyapunovDppPolicy
from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
from trisatflow.oracles.small_scale_grid_oracle import SmallScaleGridOracle, compute_oracle_gap


def _mean_info(info: Dict[str, Any], key: str) -> float:
    value = info[key]
    if torch.is_tensor(value):
        return float(value.float().mean().detach().cpu().item())
    return float(value)


def _random_visible(env: GeoLeoGroundEnv, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    mask = env._upper_action_mask_at_step(env.t).detach().cpu().tolist()
    upper = []
    for row in mask:
        feasible = [idx for idx, bit in enumerate(row) if bit]
        upper.append(rng.choice(feasible) if feasible else 0)
    return torch.tensor(upper, dtype=torch.long, device=env.device), torch.ones((env.cfg.n_leo, 3), dtype=torch.float32, device=env.device)


def _dpp_action(env: GeoLeoGroundEnv, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    policy = OptimizedLyapunovDppPolicy(grid_mode="grid_low")
    contexts = env.baseline_contexts()
    upper = []
    lower = []
    for ctx in contexts:
        decision = policy.select_action(ctx["obs"], ctx["state"], ctx["mask"], ctx["candidate_info"], rng)
        upper.append(int(decision["upper_action"]))
        lower.append([float(v) for v in decision["lower_action"]])
    return torch.tensor(upper, dtype=torch.long, device=env.device), torch.tensor(lower, dtype=torch.float32, device=env.device)


def _load_agent(method: str, device: str):
    if method == "pdqn_hybrid":
        path = REPO_ROOT / "outputs" / "reviewer_repair" / "strong_baselines" / "pdqn_tiny" / "checkpoint.pt"
        if path.exists():
            return HybridPDQNAgent.load(str(path), device=device), ""
        return None, "checkpoint_missing"
    if method == "flat_hybrid_ac":
        path = REPO_ROOT / "outputs" / "reviewer_repair" / "strong_baselines" / "flat_hybrid_tiny" / "checkpoint.pt"
        if path.exists():
            return FlatHybridActorCriticAgent.load(str(path), device=device), ""
        return None, "checkpoint_missing"
    return None, ""


def evaluate_method(method: str, *, episodes: int, steps: int, n_leo: int, device: str) -> Dict[str, Any]:
    scenario = ScenarioConfig(n_leo=int(n_leo), n_geo=1, n_ground=1, episode_len=min(int(steps), 8), seed=23)
    env = GeoLeoGroundEnv(scenario, RewardWeights(), torch.device(device))
    oracle = SmallScaleGridOracle(max_exact_candidates=5000, beam_width=4)
    rng = random.Random(23)
    agent, failure = _load_agent(method, device)
    method_costs: List[float] = []
    oracle_costs: List[float] = []
    oracle_modes: List[str] = []
    if failure:
        return {
            "method": method,
            "method_cost": "",
            "oracle_cost": "",
            "oracle_gap": "",
            "oracle_mode": "",
            "failure_reason": failure,
        }
    for episode in range(min(int(episodes), 2)):
        env.cfg.seed = 23 + episode * 103
        env.generator.manual_seed(env.cfg.seed)
        obs, _, _ = env.reset(rule_baseline_observation=True)
        for _ in range(min(int(steps), 8)):
            result = oracle.solve_one_step(env)
            oracle_costs.append(float(result.oracle_cost))
            oracle_modes.append(result.oracle_mode)
            mask = env._upper_action_mask_at_step(env.t).detach()
            if method == "random_visible":
                upper, lower = _random_visible(env, rng)
            elif method == "optimized_dpp":
                upper, lower = _dpp_action(env, rng)
            else:
                assert agent is not None
                upper, lower = agent.select_action(obs, mask, epsilon=0.0) if method == "pdqn_hybrid" else agent.select_action(obs, mask)
            out = env.step(upper, lower, minimal_info=True)
            method_costs.append(_mean_info(out.info, "normalized_system_cost"))
            obs = out.obs
            if out.done:
                break
    method_mean = float(sum(method_costs) / max(1, len(method_costs)))
    oracle_mean = float(sum(oracle_costs) / max(1, len(oracle_costs)))
    return {
        "method": method,
        "method_cost": method_mean,
        "oracle_cost": oracle_mean,
        "oracle_gap": compute_oracle_gap(method_mean, oracle_mean),
        "oracle_mode": "mixed" if len(set(oracle_modes)) > 1 else (oracle_modes[0] if oracle_modes else ""),
        "failure_reason": "",
    }


def run_oracle_gap(*, episodes: int, steps: int, n_leo: int, device: str, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["random_visible", "optimized_dpp", "pdqn_hybrid", "flat_hybrid_ac"]
    rows = [evaluate_method(method, episodes=episodes, steps=steps, n_leo=n_leo, device=device) for method in methods]
    fields = ["method", "method_cost", "oracle_cost", "oracle_gap", "oracle_mode", "failure_reason"]
    with (output_dir / "oracle_gap.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": rows,
        "row_count": len(rows),
        "oracle_name": "small_scale_grid_oracle",
        "tiny_results_are_not_paper_results": True,
    }
    (output_dir / "oracle_gap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--n-leo", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_oracle_gap(episodes=args.episodes, steps=args.steps, n_leo=args.n_leo, device=args.device, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

