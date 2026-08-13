from __future__ import annotations

import torch

from trisatflow.agents.mappo_upper import UpperMAPPOAgent
from trisatflow.agents.replay import RolloutBuffer
from trisatflow.config import AlgoConfig, PolicyRegularizationConfig
from trisatflow.models import CentralPerAgentValue, FeatureEncoder, UpperMAPPOPolicy, upper_action_mask_from_obs


def _ring_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n):
        src.extend([i, i])
        dst.extend([(i - 1) % n, (i + 1) % n])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.ones((len(src), 4), dtype=torch.float32)
    return edge_index, edge_attr


def _build_obs(n_agents: int) -> torch.Tensor:
    obs = torch.zeros((n_agents, 16), dtype=torch.float32)
    obs[:, 0:4] = 1.0
    obs[:, 4:8] = torch.tensor([800.0, 500.0, 450.0, 600.0], dtype=torch.float32)
    obs[:, 8:12] = torch.tensor([0.3, 0.7, 0.8, 0.9], dtype=torch.float32)
    obs[:, 12:16] = torch.tensor([2.0, 3.0, 3.2, 4.0], dtype=torch.float32)
    return obs


def _make_agent(cfg: AlgoConfig) -> UpperMAPPOAgent:
    encoder = FeatureEncoder(node_dim=16, edge_dim=4, hidden_dim=cfg.gnn_hidden_dim)
    actor = UpperMAPPOPolicy(cfg.gnn_hidden_dim, cfg.policy_hidden_dim, n_actions=4)
    critic = CentralPerAgentValue(cfg.gnn_hidden_dim, cfg.policy_hidden_dim)
    return UpperMAPPOAgent(encoder, actor, critic, cfg, PolicyRegularizationConfig(), torch.device("cpu"))


def _make_rollout(agent: UpperMAPPOAgent, *, steps: int, n_agents: int) -> RolloutBuffer:
    obs = _build_obs(n_agents)
    edge_index, edge_attr = _ring_edges(n_agents)
    buffer = RolloutBuffer()

    for t in range(steps):
        with torch.no_grad():
            embed = agent.encoder(obs, edge_index, edge_attr)
            mask = upper_action_mask_from_obs(obs)
            dist = agent.actor(embed, action_mask=mask, obs=obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = agent.critic(embed)
        reward = torch.linspace(-1.5, 2.0, steps=n_agents, dtype=torch.float32) + 0.15 * t
        reward = reward + 0.1 * torch.sin(torch.arange(n_agents, dtype=torch.float32))
        buffer.obs.append(obs.clone())
        buffer.edge_index.append(edge_index.clone())
        buffer.edge_attr.append(edge_attr.clone())
        buffer.upper_action.append(action.detach().cpu())
        buffer.log_prob.append(log_prob.detach().cpu())
        buffer.value.append(value.detach().cpu())
        buffer.reward.append(reward.detach().cpu())
        buffer.done.append(t == steps - 1)
    return buffer


def test_standard_ppo_uses_minibatch_updates():
    torch.manual_seed(7)
    cfg = AlgoConfig(
        gnn_hidden_dim=16,
        policy_hidden_dim=32,
        upper_lr=1.0e-3,
        ppo_update_mode="standard_ppo",
        ppo_epochs=3,
        minibatch_size=16,
        clip_param=0.2,
        value_clip_param=0.2,
        value_loss_coef=0.5,
        max_grad_norm=1.0,
        target_kl=10.0,
        advantage_normalization=True,
    )
    agent = _make_agent(cfg)
    rollout = _make_rollout(agent, steps=6, n_agents=8)
    stats = agent.update(rollout)

    assert stats["upper_num_minibatches"] >= 2
    assert 1 <= stats["upper_ppo_epochs_ran"] <= 3
    assert "upper_policy_loss" in stats
    assert "upper_approx_kl" in stats


def test_advantage_normalization_toggle_changes_advantage_scale():
    torch.manual_seed(11)
    cfg_true = AlgoConfig(
        gnn_hidden_dim=16,
        policy_hidden_dim=32,
        upper_lr=1.0e-3,
        ppo_update_mode="standard_ppo",
        ppo_epochs=2,
        minibatch_size=16,
        target_kl=10.0,
        advantage_normalization=True,
    )
    cfg_false = AlgoConfig(
        gnn_hidden_dim=16,
        policy_hidden_dim=32,
        upper_lr=1.0e-3,
        ppo_update_mode="standard_ppo",
        ppo_epochs=2,
        minibatch_size=16,
        target_kl=10.0,
        advantage_normalization=False,
    )

    agent_true = _make_agent(cfg_true)
    agent_false = _make_agent(cfg_false)
    agent_false.encoder.load_state_dict(agent_true.encoder.state_dict())
    agent_false.actor.load_state_dict(agent_true.actor.state_dict())
    agent_false.critic.load_state_dict(agent_true.critic.state_dict())

    rollout = _make_rollout(agent_true, steps=7, n_agents=6)
    stats_true = agent_true.update(rollout)
    stats_false = agent_false.update(rollout)

    assert abs(float(stats_true["upper_advantage_std"]) - 1.0) < 1.0e-3
    assert abs(float(stats_false["upper_advantage_std"]) - 1.0) > 1.0e-2
