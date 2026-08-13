from __future__ import annotations

import torch

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import AlgoConfig, ModelConfig, ScenarioConfig, TemporalModelConfig, TrainConfig
from trisatflow.models import TemporalTopologyEncoder, TopologyEncoder


def _ring_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n):
        src.extend([i, i])
        dst.extend([(i - 1) % n, (i + 1) % n])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.ones((len(src), 4), dtype=torch.float32)
    return edge_index, edge_attr


def test_temporal_encoder_sequence_shape_and_update_state_behavior() -> None:
    torch.manual_seed(7)
    n_agents = 4
    base_dim = 16
    base = TopologyEncoder(node_dim=16, edge_dim=4, hidden_dim=base_dim)
    enc = TemporalTopologyEncoder(base, base_dim=base_dim, history_len=4, temporal_hidden_dim=32)

    obs = torch.randn(n_agents, 16)
    obs_next = torch.randn(n_agents, 16)
    edge_index, edge_attr = _ring_edges(n_agents)

    out = enc(obs, edge_index, edge_attr, update_state=True)
    assert out.shape == (n_agents, base_dim)
    state_1 = enc.temporal_state()
    assert state_1 is not None

    _ = enc(obs_next, edge_index, edge_attr, update_state=False)
    state_2 = enc.temporal_state()
    assert state_2 is not None
    assert torch.allclose(state_1, state_2)


def test_temporal_encoder_reset_clears_hidden_state() -> None:
    torch.manual_seed(11)
    n_agents = 4
    base_dim = 16
    base = TopologyEncoder(node_dim=16, edge_dim=4, hidden_dim=base_dim)
    enc = TemporalTopologyEncoder(base, base_dim=base_dim, history_len=3, temporal_hidden_dim=24)
    edge_index, edge_attr = _ring_edges(n_agents)
    obs = torch.randn(n_agents, 16)

    first = enc(obs, edge_index, edge_attr, update_state=True)
    assert enc.temporal_state() is not None

    _ = enc(torch.randn(n_agents, 16), edge_index, edge_attr, update_state=True)
    assert enc.temporal_state() is not None

    enc.reset_temporal_state()
    assert enc.temporal_state() is None

    first_after_reset = enc(obs, edge_index, edge_attr, update_state=True)
    assert torch.allclose(first, first_after_reset, atol=1.0e-6)


def test_trainer_temporal_reset_called_each_episode(tmp_path) -> None:
    cfg = TrainConfig(
        total_episodes=2,
        output_dir=str(tmp_path / "temporal"),
        steps_per_episode=3,
        upper_pretrain_episodes=0,
        scenario=ScenarioConfig(n_leo=4, episode_len=3, seed=17, enable_gnn=True),
        model=ModelConfig(
            topology_encoder="temporal_gnn",
            temporal=TemporalModelConfig(enabled=True, type="gnn_gru", history_len=4, hidden_dim=32),
        ),
        algo=AlgoConfig(
            upper_algo="mappo",
            lower_algo="maddpg",
            gnn_hidden_dim=16,
            policy_hidden_dim=32,
            lower_batch_size=2,
            lower_warmup=1,
        ),
    )
    trainer = HierarchicalTrainer(cfg)

    reset_counter = {"n": 0}
    original_reset = trainer.encoder.reset_temporal_state

    def wrapped_reset() -> None:
        reset_counter["n"] += 1
        original_reset()

    trainer.encoder.reset_temporal_state = wrapped_reset  # type: ignore[assignment]

    history = trainer.train()
    assert len(history) == 2
    assert reset_counter["n"] >= 2
    assert float(history[-1]["temporal_encoder_enabled"]) == 1.0
    assert int(history[-1]["history_len"]) == 4
