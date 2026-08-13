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

from scripts.train_strong_baseline_tiny import build_agent, tiny_train
from trisatflow.agents.attention_candidate_policy import AttentionCandidatePolicy
from trisatflow.baselines.strong_registry import strong_baseline_metadata
from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
from trisatflow.oracles.small_scale_grid_oracle import SmallScaleGridOracle


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


def _evaluate_policy(method: str, *, episodes: int, steps: int, n_leo: int, device: str, output_dir: Path) -> Dict[str, Any]:
    scenario = ScenarioConfig(n_leo=int(n_leo), n_geo=1, n_ground=1, episode_len=min(int(steps), 8), seed=11)
    env = GeoLeoGroundEnv(scenario, RewardWeights(), torch.device(device))
    obs, _, _ = env.reset(rule_baseline_observation=True)
    metadata = strong_baseline_metadata(method)
    smoke_training_passed = False
    failure_reason = ""
    if method in {"pdqn_hybrid", "flat_hybrid_ac"}:
        train_dir = output_dir / f"_{method}_internal_tiny_train"
        train_summary = tiny_train(baseline=method, episodes=1, steps=min(int(steps), 8), n_leo=n_leo, device=device, output_dir=train_dir)
        agent = build_agent(method, obs.shape[-1], device)
        if method == "pdqn_hybrid":
            from trisatflow.agents.hybrid_pdqn import HybridPDQNAgent

            agent = HybridPDQNAgent.load(str(train_dir / "checkpoint.pt"), device=device)
        else:
            from trisatflow.agents.flat_hybrid_actor_critic import FlatHybridActorCriticAgent

            agent = FlatHybridActorCriticAgent.load(str(train_dir / "checkpoint.pt"), device=device)
        smoke_training_passed = bool(train_summary.get("training_complete_for_smoke"))
    elif method == "attention_candidate":
        agent = AttentionCandidatePolicy(obs.shape[-1], device=device)
    else:
        agent = None
    costs: List[float] = []
    delays: List[float] = []
    energies: List[float] = []
    violations: List[float] = []
    rng = random.Random(13)
    for episode in range(min(int(episodes), 2)):
        env.cfg.seed = 11 + episode * 97
        env.generator.manual_seed(env.cfg.seed)
        obs, _, _ = env.reset(rule_baseline_observation=True)
        for _ in range(min(int(steps), 8)):
            mask = env._upper_action_mask_at_step(env.t).detach()
            if agent is None:
                upper, lower = _random_visible(env, rng)
            else:
                upper, lower = agent.select_action(obs, mask)
            out = env.step(upper, lower, minimal_info=True)
            costs.append(_mean_info(out.info, "normalized_system_cost"))
            delays.append(_mean_info(out.info, "delay"))
            energies.append(_mean_info(out.info, "energy"))
            violations.append(_mean_info(out.info, "deadline_violation_flag"))
            obs = out.obs
            if out.done:
                break
    row = {
        "method": method,
        "baseline_family": metadata["baseline_family"],
        "trainable": metadata["trainable"],
        "update_implemented": metadata["update_implemented"],
        "smoke_training_passed": smoke_training_passed,
        "paper_ready": False,
        "action_mask_supported": metadata["action_mask_supported"],
        "continuous_action_supported": metadata["continuous_action_supported"],
        "cost": float(sum(costs) / max(1, len(costs))),
        "delay": float(sum(delays) / max(1, len(delays))),
        "energy": float(sum(energies) / max(1, len(energies))),
        "violation": float(sum(violations) / max(1, len(violations))),
        "oracle_cost": "",
        "oracle_gap": "",
        "failure_reason": failure_reason,
    }
    if method == "attention_candidate":
        row["failure_reason"] = "forward_select_only_future_baseline_candidate_not_full_rl_update"
    return row


def _evaluate_oracle(*, episodes: int, steps: int, n_leo: int, device: str) -> Dict[str, Any]:
    scenario = ScenarioConfig(n_leo=int(n_leo), n_geo=1, n_ground=1, episode_len=min(int(steps), 8), seed=17)
    env = GeoLeoGroundEnv(scenario, RewardWeights(), torch.device(device))
    oracle = SmallScaleGridOracle(max_exact_candidates=5000, beam_width=4)
    costs = []
    modes = []
    for episode in range(min(int(episodes), 2)):
        env.cfg.seed = 17 + episode * 101
        env.generator.manual_seed(env.cfg.seed)
        env.reset(rule_baseline_observation=True)
        for _ in range(min(int(steps), 8)):
            result = oracle.solve_one_step(env)
            out = env.step(result.oracle_action, result.oracle_lower_action, minimal_info=True)
            costs.append(float(result.oracle_cost))
            modes.append(result.oracle_mode)
            if out.done:
                break
    metadata = oracle.metadata
    return {
        "method": "small_scale_grid_oracle",
        "baseline_family": metadata["baseline_family"],
        "trainable": False,
        "update_implemented": False,
        "smoke_training_passed": True,
        "paper_ready": False,
        "action_mask_supported": True,
        "continuous_action_supported": True,
        "cost": float(sum(costs) / max(1, len(costs))),
        "delay": "",
        "energy": "",
        "violation": "",
        "oracle_cost": float(sum(costs) / max(1, len(costs))),
        "oracle_gap": 0.0,
        "failure_reason": "" if set(modes) == {"exact_grid"} else "oracle_mode_includes_beam_grid_approx",
    }


def run_eval(*, baselines: List[str], episodes: int, steps: int, n_leo: int, device: str, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in baselines:
        if method in {"small_scale_grid_oracle", "grid_oracle"}:
            rows.append(_evaluate_oracle(episodes=episodes, steps=steps, n_leo=n_leo, device=device))
        else:
            rows.append(_evaluate_policy(method, episodes=episodes, steps=steps, n_leo=n_leo, device=device, output_dir=output_dir))
    fields = [
        "method",
        "baseline_family",
        "trainable",
        "update_implemented",
        "smoke_training_passed",
        "paper_ready",
        "action_mask_supported",
        "continuous_action_supported",
        "cost",
        "delay",
        "energy",
        "violation",
        "oracle_cost",
        "oracle_gap",
        "failure_reason",
    ]
    with (output_dir / "strong_baseline_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": rows,
        "row_count": len(rows),
        "paper_ready_any": False,
        "tiny_results_are_not_paper_results": True,
        "full_experiment_required": True,
    }
    (output_dir / "strong_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default="pdqn_hybrid,flat_hybrid_ac,small_scale_grid_oracle")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--n-leo", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    summary = run_eval(baselines=baselines, episodes=args.episodes, steps=args.steps, n_leo=args.n_leo, device=args.device, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

