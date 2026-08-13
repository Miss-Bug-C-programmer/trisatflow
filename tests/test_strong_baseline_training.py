from __future__ import annotations

import torch

from trisatflow.agents.attention_candidate_policy import AttentionCandidatePolicy
from trisatflow.agents.flat_hybrid_actor_critic import FlatHybridActorCriticAgent
from trisatflow.agents.hybrid_pdqn import HybridBatch, HybridPDQNAgent
from trisatflow.baselines.strong_registry import strong_baseline_metadata


def _batch(obs_dim: int = 12, batch_size: int = 8) -> HybridBatch:
    obs = torch.randn(batch_size, obs_dim)
    next_obs = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, 4)
    next_mask = torch.ones(batch_size, 4)
    action = torch.randint(0, 4, (batch_size,))
    lower = torch.rand(batch_size, 3)
    reward = torch.randn(batch_size)
    done = torch.zeros(batch_size)
    return HybridBatch(obs, mask, action, lower, reward, next_obs, next_mask, done)


def _changed(before, params) -> bool:
    return any(not torch.allclose(old, new.detach().cpu()) for old, new in zip(before, params))


def test_pdqn_select_action_respects_mask_and_lower_range() -> None:
    agent = HybridPDQNAgent(12, device="cpu")
    obs = torch.randn(5, 12)
    mask = torch.zeros(5, 4, dtype=torch.bool)
    mask[:, 2] = True
    upper, lower = agent.select_action(obs, mask, epsilon=0.0)
    assert torch.equal(upper, torch.full((5,), 2, dtype=torch.long))
    assert lower.shape == (5, 3)
    assert bool((lower >= 0.0).all() and (lower <= 1.0).all())


def test_pdqn_update_changes_parameters() -> None:
    agent = HybridPDQNAgent(12, device="cpu")
    before = [p.detach().cpu().clone() for p in agent.actor.parameters()]
    metrics = agent.update(_batch())
    assert metrics["critic_loss"] >= 0.0
    assert metrics["grad_norm"] > 0.0
    assert _changed(before, agent.actor.parameters())


def test_flat_hybrid_update_changes_parameters_and_mask() -> None:
    agent = FlatHybridActorCriticAgent(12, device="cpu")
    obs = torch.randn(6, 12)
    mask = torch.zeros(6, 4, dtype=torch.bool)
    mask[:, 1] = True
    upper, lower = agent.select_action(obs, mask)
    assert torch.equal(upper, torch.ones(6, dtype=torch.long))
    assert bool((lower >= 0.0).all() and (lower <= 1.0).all())
    before = [p.detach().cpu().clone() for p in agent.actor.parameters()]
    metrics = agent.update(_batch())
    assert metrics["critic_loss"] >= 0.0
    assert metrics["grad_norm"] > 0.0
    assert _changed(before, agent.actor.parameters())


def test_metadata_skeleton_not_paper_ready() -> None:
    for name in ("pdqn_hybrid", "flat_hybrid_ac", "attention_candidate"):
        meta = strong_baseline_metadata(name)
        assert meta["paper_ready"] is False
    assert strong_baseline_metadata("attention_candidate")["update_implemented"] is False


def test_attention_candidate_forward_shape() -> None:
    policy = AttentionCandidatePolicy(12, device="cpu")
    obs = torch.randn(7, 12)
    mask = torch.ones(7, 4, dtype=torch.bool)
    logits, params = policy(obs, mask)
    assert logits.shape == (7, 4)
    assert params.shape == (7, 4, 3)
    upper, lower = policy.select_action(obs, mask)
    assert upper.shape == (7,)
    assert lower.shape == (7, 3)


def test_cpu_device_runs_without_cuda() -> None:
    agent = HybridPDQNAgent(12, device="cpu")
    assert next(agent.actor.parameters()).device.type == "cpu"
    agent.update(_batch())

