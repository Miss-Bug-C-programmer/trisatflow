from __future__ import annotations

import torch

from trisatflow.agents.maddpg_lower import LowerMADDPGAgent
from trisatflow.agents.replay import ReplayBuffer
from trisatflow.config import AlgoConfig
from trisatflow.diagnostics.gradient_flow import build_gradient_report, lower_action_sensitivity_to_upper_action
from trisatflow.diagnostics.training_cadence import cadence_report
from trisatflow.models import FeatureEncoder, LowerActor, LowerCritic


def _ring_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n):
        src.extend([i, i])
        dst.extend([(i - 1) % n, (i + 1) % n])
    return torch.tensor([src, dst], dtype=torch.long), torch.ones((len(src), 4), dtype=torch.float32)


def _replay(n_agents: int = 4, node_dim: int = 12, n_steps: int = 8) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=32)
    edge_index, edge_attr = _ring_edges(n_agents)
    for step in range(n_steps):
        obs = torch.randn(n_agents, node_dim) * 0.1 + step * 0.01
        replay.add(
            obs=obs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            upper_action=torch.randint(0, 4, (n_agents,), dtype=torch.long),
            lower_action=torch.rand(n_agents, 3),
            reward=torch.randn(n_agents) * 0.01 + 0.2,
            upper_reward=torch.randn(n_agents) * 0.01 + 0.2,
            next_obs=obs + 0.01 * torch.randn_like(obs),
            next_edge_index=edge_index,
            next_edge_attr=edge_attr,
            done=torch.tensor(float(step == n_steps - 1)),
        )
    return replay


def _agent(mode: str, *, stop_gradient: bool, lower_observation_mode: str = "shared_embedding") -> LowerMADDPGAgent:
    torch.manual_seed(123)
    cfg = AlgoConfig(
        gnn_hidden_dim=8,
        policy_hidden_dim=16,
        lower_batch_size=2,
        lower_warmup=1,
        lower_lr=1.0e-3,
        critic_lr=1.0e-3,
        encoder_lr=1.0e-3,
        encoder_mode=mode,
        lower_observation_mode=lower_observation_mode,
        stop_gradient_to_encoder_from_lower=stop_gradient,
        joint_encoder_loss_weight=0.5,
    )
    encoder = FeatureEncoder(node_dim=12, edge_dim=4, hidden_dim=8)
    target_encoder = FeatureEncoder(node_dim=12, edge_dim=4, hidden_dim=8) if mode in {"separate", "separate_lower_encoder"} else None
    actor = LowerActor(8, 4, 16, 3)
    critic = LowerCritic(8, 4, 4, 3, 16)
    target_actor = LowerActor(8, 4, 16, 3)
    target_critic = LowerCritic(8, 4, 4, 3, 16)
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


def test_shared_upper_only_lower_does_not_update_shared_encoder() -> None:
    agent = _agent("shared_upper_only", stop_gradient=True)
    stats = agent.update(_replay())

    assert stats["lower_encoder_mode"] == "shared_upper_only"
    assert stats["lower_encoder_grad_norm"] == 0.0
    assert stats["shared_encoder_grad_norm_from_lower"] == 0.0


def test_shared_joint_lower_can_produce_shared_encoder_grad() -> None:
    agent = _agent("shared_joint", stop_gradient=False)
    stats = agent.update(_replay())

    assert stats["lower_encoder_mode"] == "shared_joint"
    assert stats["shared_encoder_grad_norm_from_lower"] > 0.0
    assert stats["lower_encoder_grad_norm"] > 0.0


def test_separate_lower_encoder_grad_exists() -> None:
    agent = _agent("separate_lower_encoder", stop_gradient=True)
    stats = agent.update(_replay())

    assert stats["lower_encoder_mode"] == "separate_lower_encoder"
    assert stats["separate_lower_encoder_grad_norm"] > 0.0


def test_action_collection_detach_is_not_training_detach() -> None:
    cfg = AlgoConfig(
        encoder_mode="shared_joint",
        stop_gradient_to_encoder_from_lower=False,
        detach_embedding_during_action_collection=True,
        lower_batch_size=2,
        lower_warmup=1,
        gnn_hidden_dim=8,
        policy_hidden_dim=16,
    )
    agent = _agent(cfg.encoder_mode, stop_gradient=cfg.stop_gradient_to_encoder_from_lower)
    stats = agent.update(_replay())

    assert cfg.detach_embedding_during_action_collection is True
    assert stats["shared_encoder_grad_norm_from_lower"] > 0.0


def test_lower_action_has_structural_sensitivity_to_upper_action() -> None:
    agent = _agent("shared_upper_only", stop_gradient=True)
    obs = torch.randn(4, 12)
    edge_index, edge_attr = _ring_edges(4)
    embed = agent.encoder(obs, edge_index, edge_attr)
    sensitivity = lower_action_sensitivity_to_upper_action(agent, embed, obs=obs, edge_index=edge_index, edge_attr=edge_attr)

    assert sensitivity["lower_action_sensitivity_to_upper_action"] > 0.0
    assert sensitivity["lower_allocator_not_conditioned_effectively"] is False


def test_update_cadence_metadata() -> None:
    report = cadence_report(
        upper_update_count=2,
        lower_update_count=4,
        env_steps_since_upper_update=0,
        env_steps_since_lower_update=3,
        replay_buffer_size=16,
        rollout_buffer_size=4,
        upper_update_every=1,
        lower_update_every=2,
        lower_updates_per_upper_update=2,
    )

    assert report["upper_update_count"] == 2
    assert report["lower_update_count"] == 4
    assert report["off_policy_lag_estimate"] == 3
    assert report["non_stationarity_warning"] is True


def test_diagnostics_missing_fields_have_reason() -> None:
    class Dummy:
        def _lower_encoder_mode(self) -> str:
            return "shared_upper_only"

    row = build_gradient_report(trainer=Dummy(), upper_losses={}, lower_losses={}, update_step=1, sensitivity={})

    assert "upper_actor_grad_norm" in row["unavailable_fields"]
    assert "upper_critic_grad_norm" in row["unavailable_fields"]
    assert "shared_encoder_grad_norm_from_upper" in row["unavailable_fields"]
