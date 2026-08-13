from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.agents.mappo_upper import UpperMAPPOAgent
from trisatflow.agents.replay import RolloutBuffer
from trisatflow.config import AlgoConfig, PolicyRegularizationConfig
from trisatflow.models import CentralValue, FeatureEncoder, UpperMAPPOPolicy, upper_action_mask_from_obs


def _build_obs(n_agents: int, *, geo_visible: bool = True) -> torch.Tensor:
    obs = torch.zeros((n_agents, 16), dtype=torch.float32)
    obs[:, 0] = 1.0
    obs[:, 1] = 1.0
    obs[:, 2] = 1.0 if geo_visible else 0.0
    obs[:, 3] = 1.0
    obs[:, 4:8] = torch.tensor([1000.0, 500.0, 260.0, 360.0], dtype=torch.float32)
    obs[:, 8:12] = torch.tensor([0.4, 0.9, 1.8, 1.2], dtype=torch.float32)
    obs[:, 12:16] = torch.tensor([3.0, 4.0, 5.0, 4.5], dtype=torch.float32)
    return obs


def _ring_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n):
        src.extend([i, i])
        dst.extend([(i - 1) % n, (i + 1) % n])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.ones((len(src), 4), dtype=torch.float32)
    return edge_index, edge_attr


def _probs(agent: UpperMAPPOAgent, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        embed = agent.encoder(obs, edge_index, edge_attr)
        mask = upper_action_mask_from_obs(obs)
        dist = agent.actor(embed, action_mask=mask)
        return dist.probs.detach().cpu()


def _make_rollout(
    agent: UpperMAPPOAgent,
    obs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    target_action: int,
    steps: int,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    alt_action = (target_action + 1) % 4
    if alt_action == 2 and not bool(upper_action_mask_from_obs(obs)[0, 2].item()):
        alt_action = 1 if target_action != 1 else 3
    for t in range(steps):
        with torch.no_grad():
            embed = agent.encoder(obs, edge_index, edge_attr)
            mask = upper_action_mask_from_obs(obs)
            dist = agent.actor(embed, action_mask=mask)
            action_idx = target_action if (t % 2 == 0) else alt_action
            action = torch.full((obs.shape[0],), action_idx, dtype=torch.long)
            log_prob = dist.log_prob(action)
            value = agent.critic(embed)
        reward_value = 2.0 if (t % 2 == 0) else -2.0
        reward = torch.full((obs.shape[0],), reward_value, dtype=torch.float32)
        buffer.obs.append(obs.clone())
        buffer.edge_index.append(edge_index.clone())
        buffer.edge_attr.append(edge_attr.clone())
        buffer.upper_action.append(action)
        buffer.log_prob.append(log_prob)
        buffer.value.append(value.detach().cpu())
        buffer.reward.append(reward)
        buffer.done.append(t == steps - 1)
    return buffer


def _run_case(case: str, target_action: int, geo_visible: bool) -> Dict[str, object]:
    torch.manual_seed(13)
    cfg = AlgoConfig(
        gnn_hidden_dim=32,
        policy_hidden_dim=64,
        upper_lr=1.0e-3,
        gamma=0.95,
        gae_lambda=0.95,
        ppo_clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epsilon_decay_episodes=100,
    )
    n_agents = 32
    obs = _build_obs(n_agents, geo_visible=geo_visible)
    edge_index, edge_attr = _ring_edges(n_agents)
    encoder = FeatureEncoder(node_dim=16, edge_dim=4, hidden_dim=cfg.gnn_hidden_dim)
    actor = UpperMAPPOPolicy(cfg.gnn_hidden_dim, cfg.policy_hidden_dim, n_actions=4)
    critic = CentralValue(cfg.gnn_hidden_dim, n_agents=n_agents, hidden_dim=cfg.policy_hidden_dim)
    agent = UpperMAPPOAgent(encoder, actor, critic, cfg, PolicyRegularizationConfig(), torch.device("cpu"))

    initial_probs = _probs(agent, obs, edge_index, edge_attr).mean(dim=0)
    rollout = _make_rollout(
        agent,
        obs,
        edge_index,
        edge_attr,
        target_action=target_action,
        steps=24,
    )
    for _ in range(20):
        agent.update(rollout)
    final_probs = _probs(agent, obs, edge_index, edge_attr).mean(dim=0)

    target_increase = float(final_probs[target_action] - initial_probs[target_action])
    masked_geo_prob = 0.0 if geo_visible else float(final_probs[2])
    passed = target_increase > 1.0e-3
    if not geo_visible:
        passed = passed and masked_geo_prob < 1.0e-6

    return {
        "case": case,
        "initial_prob": [float(x) for x in initial_probs.tolist()],
        "final_prob": [float(x) for x in final_probs.tolist()],
        "target_action": target_action,
        "target_action_prob_increase": target_increase,
        "masked_action_prob": masked_geo_prob,
        "pass": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit sanity test for MAPPO upper PPO update direction and mask handling.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cases: List[Dict[str, object]] = []
    cases.append(_run_case("case_A_local_advantage", target_action=0, geo_visible=True))
    cases.append(_run_case("case_B_geo_advantage", target_action=2, geo_visible=True))
    cases.append(_run_case("case_C_ground_advantage", target_action=3, geo_visible=True))
    cases.append(_run_case("case_D_geo_masked", target_action=0, geo_visible=False))

    any_geo_only = all(max(item["final_prob"]) == item["final_prob"][2] for item in cases)
    passed = all(bool(item["pass"]) for item in cases) and (not any_geo_only)
    output = {
        "status": "MAPPO_UPDATE_SANITY_OK" if passed else "MAPPO_UPDATE_SANITY_FAILED",
        "cases": cases,
        "all_pass": passed,
        "no_geo_only_bias": not any_geo_only,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
