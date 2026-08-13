from __future__ import annotations

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import AlgoConfig, ObservationConfig, ScenarioConfig, TrainConfig
from trisatflow.envs.obs_builder import build_shared_observation
from trisatflow.envs.obs_schema import IDX_LOCAL_NORMALIZED_COST
from trisatflow.envs.obs_schema import feature_access_class


def _tiny_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig(
        total_episodes=1,
        scenario=ScenarioConfig(n_leo=4, episode_len=2, seed=23),
        algo=AlgoConfig(gnn_hidden_dim=16, policy_hidden_dim=32, lower_batch_size=4, lower_warmup=4),
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_default_mode_is_safe_observable() -> None:
    trainer = HierarchicalTrainer(_tiny_cfg())
    assert trainer.cfg.observation.mode == "safe_observable"
    assert trainer.cfg.reward.use_oracle_cost_components is False
    assert trainer.cfg.policy_regularization.enabled is False
    assert trainer.cfg.scenario.observation_include_cost_prior_features is False


def test_oracle_debug_prints_warning(capsys) -> None:
    cfg = _tiny_cfg(observation=ObservationConfig(mode="oracle_debug", include_oracle_cost=True, include_cost_prior_features=True))
    HierarchicalTrainer(cfg)
    out = capsys.readouterr().out
    assert "oracle_debug enabled" in out


def test_safe_mode_overrides_oracle_reward(capsys) -> None:
    cfg = _tiny_cfg()
    cfg.reward.mode = "oracle_aligned_cost"
    cfg.reward.use_oracle_cost_components = True
    trainer = HierarchicalTrainer(cfg)
    out = capsys.readouterr().out
    assert "safe_observable disallows oracle-aligned reward" in out
    assert trainer.cfg.reward.mode == "physical_weighted"
    assert trainer.cfg.reward.use_oracle_cost_components is False


def test_feature_access_classes() -> None:
    assert feature_access_class("local_visible") == "observable"
    assert feature_access_class("neighbor_delay") == "estimated"
    assert feature_access_class("local_normalized_cost") == "privileged"
    assert (
        feature_access_class(
            "local_normalized_cost",
            access_mode="oracle_debug",
            include_oracle_cost=True,
        )
        == "oracle/debug-only"
    )


def test_oracle_debug_observation_prefers_oracle_cost_field() -> None:
    row = {
        "local_visible": 1.0,
        "neighbor_visible": 1.0,
        "geo_visible": 1.0,
        "ground_visible": 1.0,
        "local_rate": 9.0,
        "neighbor_rate": 8.0,
        "geo_rate": 7.0,
        "ground_rate": 6.0,
        "local_delay": 0.08,
        "neighbor_delay": 0.12,
        "geo_delay": 0.22,
        "ground_delay": 0.18,
        "local_queue": 2.0,
        "neighbor_queue": 3.0,
        "geo_queue": 4.0,
        "ground_queue": 5.0,
        "local_oracle_normalized_cost": 0.123,
    }
    safe_batch = build_shared_observation(
        [row],
        node_feature_dim=20,
        access_mode="safe_observable",
        include_cost_prior_features=False,
        include_oracle_cost=False,
    )
    debug_batch = build_shared_observation(
        [row],
        node_feature_dim=20,
        access_mode="oracle_debug",
        include_cost_prior_features=True,
        include_oracle_cost=True,
    )
    assert float(safe_batch.obs[0, IDX_LOCAL_NORMALIZED_COST]) == 0.0
    assert abs(float(debug_batch.obs[0, IDX_LOCAL_NORMALIZED_COST]) - 0.123) < 1.0e-6
