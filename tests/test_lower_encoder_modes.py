from __future__ import annotations

import torch

from trisatflow.agents import HierarchicalTrainer
from trisatflow.agents.maddpg_lower import LowerMADDPGAgent
from trisatflow.agents.replay import ReplayBuffer
from trisatflow.config import AlgoConfig, ScenarioConfig, TrainConfig
from trisatflow.models import FeatureEncoder, LowerActor, LowerCritic


def _ring_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n):
        src.extend([i, i])
        dst.extend([(i - 1) % n, (i + 1) % n])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.ones((len(src), 4), dtype=torch.float32)
    return edge_index, edge_attr


def _build_replay(n_agents: int, node_dim: int, n_steps: int = 6) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=64)
    edge_index, edge_attr = _ring_edges(n_agents)
    for t in range(n_steps):
        obs = torch.randn(n_agents, node_dim) * 0.1 + 0.2 * t
        next_obs = obs + 0.05 * torch.randn_like(obs)
        upper_action = torch.randint(0, 4, (n_agents,), dtype=torch.long)
        lower_action = torch.rand(n_agents, 3)
        reward = torch.randn(n_agents) * 0.1 + 0.5
        replay.add(
            obs=obs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            upper_action=upper_action,
            lower_action=lower_action,
            reward=reward,
            upper_reward=reward,
            next_obs=next_obs,
            next_edge_index=edge_index,
            next_edge_attr=edge_attr,
            done=torch.tensor(float(t == n_steps - 1)),
        )
    return replay


def _build_maddpg_agent(mode: str, *, tau: float = 0.5) -> LowerMADDPGAgent:
    cfg = AlgoConfig(
        gnn_hidden_dim=8,
        policy_hidden_dim=16,
        lower_batch_size=2,
        lower_warmup=1,
        lower_lr=1.0e-3,
        critic_lr=1.0e-3,
        tau=tau,
        encoder_mode=mode,
        encoder_lr=1.0e-3,
        joint_encoder_loss_weight=0.5,
    )
    n_agents = 4
    encoder = FeatureEncoder(node_dim=16, edge_dim=4, hidden_dim=cfg.gnn_hidden_dim)
    target_encoder = FeatureEncoder(node_dim=16, edge_dim=4, hidden_dim=cfg.gnn_hidden_dim) if mode == "separate" else None
    actor = LowerActor(cfg.gnn_hidden_dim, 4, cfg.policy_hidden_dim, 3)
    critic = LowerCritic(cfg.gnn_hidden_dim, n_agents, 4, 3, cfg.policy_hidden_dim)
    target_actor = LowerActor(cfg.gnn_hidden_dim, 4, cfg.policy_hidden_dim, 3)
    target_critic = LowerCritic(cfg.gnn_hidden_dim, n_agents, 4, 3, cfg.policy_hidden_dim)
    return LowerMADDPGAgent(
        encoder,
        actor,
        critic,
        target_actor,
        target_critic,
        cfg,
        torch.device("cpu"),
        target_encoder=target_encoder,
        encoder_mode=mode,
    )


def _params_clone(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _any_param_changed(before: list[torch.Tensor], after_module: torch.nn.Module) -> bool:
    after = [p.detach() for p in after_module.parameters()]
    return any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_separate_encoder_params_update():
    torch.manual_seed(7)
    agent = _build_maddpg_agent("separate")
    replay = _build_replay(n_agents=4, node_dim=16)
    before = _params_clone(agent.encoder)
    stats = agent.update(replay)

    assert _any_param_changed(before, agent.encoder)
    assert stats["lower_encoder_grad_norm"] > 0.0


def test_shared_frozen_encoder_not_updated_by_lower_loss():
    torch.manual_seed(11)
    agent = _build_maddpg_agent("shared_frozen")
    replay = _build_replay(n_agents=4, node_dim=16)
    before = _params_clone(agent.encoder)
    stats = agent.update(replay)

    assert not _any_param_changed(before, agent.encoder)
    assert stats["lower_encoder_grad_norm"] == 0.0


def test_target_encoder_soft_update_math():
    torch.manual_seed(13)
    agent = _build_maddpg_agent("separate", tau=0.5)
    assert agent.target_encoder is not None

    with torch.no_grad():
        for p in agent.encoder.parameters():
            p.fill_(1.0)
        for p in agent.target_encoder.parameters():
            p.fill_(0.0)

    agent._soft_update(agent.target_encoder, agent.encoder)

    for p in agent.target_encoder.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5), atol=1.0e-6)


def test_three_lower_encoder_modes_trainer_bootstrap(tmp_path):
    for mode in ["shared_frozen", "shared_joint", "separate"]:
        cfg = TrainConfig(
            total_episodes=1,
            output_dir=str(tmp_path / mode),
            steps_per_episode=3,
            upper_pretrain_episodes=0,
            lower_training_enabled=True,
            lower_action_mode="learned",
            scenario=ScenarioConfig(n_leo=4, episode_len=3, seed=17, enable_gnn=False),
            algo=AlgoConfig(
                upper_algo="mappo",
                lower_algo="maddpg",
                gnn_hidden_dim=8,
                policy_hidden_dim=16,
                lower_batch_size=2,
                lower_warmup=1,
                encoder_mode=mode,
                encoder_lr=1.0e-3,
                joint_encoder_loss_weight=0.5,
            ),
        )
        history = HierarchicalTrainer(cfg).train()
        assert len(history) == 1
        assert "lower_actor_loss" in history[0]
        assert "lower_critic_loss" in history[0]
