import csv

import torch

from trisatflow.config import AlgoConfig, ScenarioConfig, TrainConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.agents import HierarchicalTrainer
from trisatflow.algorithms import lower_algorithm_names, upper_algorithm_names


def test_env_step_shapes():
    env = GeoLeoGroundEnv(ScenarioConfig(n_leo=4, episode_len=2, seed=5))
    obs, edge_index, edge_attr = env.reset()
    assert obs.shape == (4, env.node_feature_dim)
    assert edge_index.shape[0] == 2
    upper = torch.tensor([0, 1, 2, 3])
    lower = torch.ones(4, 3) * 0.5
    step = env.step(upper, lower)
    assert step.obs.shape == obs.shape
    assert step.upper_reward.shape == (4,)
    assert step.lower_reward.shape == (4,)


def test_trainer_smoke_one_episode_and_csv(tmp_path):
    cfg = TrainConfig(
        total_episodes=1,
        output_dir=str(tmp_path),
        scenario=ScenarioConfig(n_leo=4, episode_len=4, seed=3),
        algo=AlgoConfig(gnn_hidden_dim=16, policy_hidden_dim=32, lower_batch_size=4, lower_warmup=4),
    )
    history = HierarchicalTrainer(cfg).train()
    assert len(history) == 1
    assert history[0]["mean_queue"] >= 0
    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["upper_algo"] == "mappo"
    assert rows[0]["lower_algo"] == "maddpg"


def test_algorithm_registry_lists_multiple_choices():
    assert {"mappo", "ippo", "iql", "vdn", "qmix"}.issubset(set(upper_algorithm_names()))
    assert {"maddpg", "iddpg", "masac", "isac"}.issubset(set(lower_algorithm_names()))


def test_non_default_combination_smoke(tmp_path):
    cfg = TrainConfig(
        total_episodes=1,
        output_dir=str(tmp_path),
        scenario=ScenarioConfig(n_leo=4, episode_len=3, seed=4),
        algo=AlgoConfig(
            upper_algo="qmix",
            lower_algo="masac",
            gnn_hidden_dim=16,
            policy_hidden_dim=32,
            lower_batch_size=4,
            lower_warmup=4,
            upper_batch_size=4,
            upper_warmup=4,
        ),
    )
    history = HierarchicalTrainer(cfg).train()
    assert len(history) == 1
    assert (tmp_path / "metrics.csv").exists()


def test_ablation_switches_keep_env_runnable():
    env = GeoLeoGroundEnv(
        ScenarioConfig(
            n_leo=4,
            episode_len=2,
            seed=9,
            enable_geo=False,
            enable_ground=False,
            enable_isl=False,
            enable_lyapunov_reward=False,
            enable_cross_layer_feedback=False,
        )
    )
    obs, edge_index, edge_attr = env.reset()
    assert edge_index.shape == (2, 0)
    assert edge_attr.shape[0] == 0
    upper = torch.tensor([0, 1, 2, 3])
    lower = torch.ones(4, 3) * 0.5
    step = env.step(upper, lower)
    assert step.info["feasible"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert "virtual_delay_queue" in step.info


def test_no_gnn_trainer_ablation_smoke(tmp_path):
    cfg = TrainConfig(
        total_episodes=1,
        output_dir=str(tmp_path),
        scenario=ScenarioConfig(n_leo=4, episode_len=3, seed=12, enable_gnn=False),
        algo=AlgoConfig(gnn_hidden_dim=16, policy_hidden_dim=32, lower_batch_size=4, lower_warmup=4),
    )
    history = HierarchicalTrainer(cfg).train()
    assert len(history) == 1
    assert "mean_virtual_delay_queue" in history[0]
