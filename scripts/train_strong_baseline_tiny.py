from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.agents.flat_hybrid_actor_critic import FlatHybridActorCriticAgent
from trisatflow.agents.hybrid_pdqn import HybridPDQNAgent, HybridReplayBuffer
from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv


def _mean_tensor(info: Dict[str, Any], key: str) -> torch.Tensor:
    value = info[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return value.float()


def build_agent(name: str, obs_dim: int, device: str):
    if name == "pdqn_hybrid":
        return HybridPDQNAgent(obs_dim, device=device)
    if name == "flat_hybrid_ac":
        return FlatHybridActorCriticAgent(obs_dim, device=device)
    raise ValueError(f"Unsupported trainable strong baseline: {name}")


def tiny_train(
    *,
    baseline: str,
    episodes: int,
    steps: int,
    n_leo: int,
    device: str,
    output_dir: Path,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = ScenarioConfig(n_leo=int(n_leo), n_geo=1, n_ground=1, episode_len=min(int(steps), 8), seed=7)
    env = GeoLeoGroundEnv(scenario, RewardWeights(), torch.device(device))
    obs, _, _ = env.reset(rule_baseline_observation=True)
    agent = build_agent(baseline, obs.shape[-1], device)
    buffer = HybridReplayBuffer(capacity=2048, device=device)
    update_rows = []
    costs = []
    for episode in range(min(int(episodes), 2)):
        env.cfg.seed = 7 + episode * 1009
        env.generator.manual_seed(env.cfg.seed)
        obs, _, _ = env.reset(rule_baseline_observation=True)
        done = False
        for _ in range(min(int(steps), 8)):
            mask = env._upper_action_mask_at_step(env.t).detach()
            upper, lower = agent.select_action(obs, mask, epsilon=0.25)
            out = env.step(upper, lower, minimal_info=True)
            next_obs = out.obs.detach()
            next_mask = env._upper_action_mask_at_step(env.t).detach()
            cost_vec = _mean_tensor(out.info, "normalized_system_cost").to(env.device)
            reward = -cost_vec
            buffer.add_batch(
                obs=obs,
                mask=mask,
                action=upper,
                lower_action=lower,
                reward=reward,
                next_obs=next_obs,
                next_mask=next_mask,
                done=bool(out.done),
            )
            costs.append(float(cost_vec.mean().detach().cpu().item()))
            obs = next_obs
            done = bool(out.done)
            if len(buffer) >= 4:
                update_rows.append(agent.update(buffer.sample(batch_size=min(16, len(buffer)))))
            if done:
                break
    if not update_rows:
        update_rows.append(agent.update(buffer.sample(batch_size=max(1, len(buffer)))))
    checkpoint_path = output_dir / "checkpoint.pt"
    if save_checkpoint:
        agent.save(str(checkpoint_path))
    metadata = dict(agent.metadata)
    metadata["smoke_training_passed"] = True
    summary = {
        "baseline": baseline,
        "baseline_family": metadata["baseline_family"],
        "training_complete_for_smoke": True,
        "paper_ready": False,
        "episodes": min(int(episodes), 2),
        "steps": min(int(steps), 8),
        "n_leo": int(n_leo),
        "device": str(device),
        "updates": len(update_rows),
        "last_update": update_rows[-1],
        "mean_cost": float(sum(costs) / max(1, len(costs))),
        "checkpoint": str(checkpoint_path) if save_checkpoint else "",
        "metadata": metadata,
        "tiny_results_are_not_paper_results": True,
    }
    (output_dir / "training_smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["pdqn_hybrid", "flat_hybrid_ac"], required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--n-leo", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = tiny_train(
        baseline=args.baseline,
        episodes=args.episodes,
        steps=args.steps,
        n_leo=args.n_leo,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

