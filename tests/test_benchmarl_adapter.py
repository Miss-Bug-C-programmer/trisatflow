from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml

benchmarl = pytest.importorskip("benchmarl")
_ = benchmarl  # silence linters
from trisatflow.config import ScenarioConfig

torchrl = pytest.importorskip("torchrl")
_ = torchrl  # silence linters

from benchmarl.environments import task_config_registry
from trisatflow.benchmarl_adapter import TriSatFlowBenchMARLEnv, TriSatFlowTorchRLEnv


def _ensure_trisatflow_registry_entry() -> None:
    key = "trisatflow/mixed_small"
    if key in task_config_registry:
        return
    import benchmarl.environments as env_pkg

    repo_env_dir = Path(__file__).resolve().parents[1] / "benchmarl" / "environments"
    path_value = str(repo_env_dir)
    if path_value not in list(getattr(env_pkg, "__path__", [])):
        env_pkg.__path__.append(path_value)
    importlib.invalidate_caches()
    module = importlib.import_module("benchmarl.environments.trisatflow.common")
    enum_cls = getattr(module, "TriSatFlowTask")
    task_config_registry[key] = enum_cls.MIXED_SMALL


def test_torchrl_adapter_specs_and_reset_step() -> None:
    env = TriSatFlowTorchRLEnv(
        scenario=ScenarioConfig(n_leo=4, episode_len=4, seed=13),
        device="cpu",
    )
    td0 = env.reset()
    assert "observation" in td0.keys()
    assert td0["observation"].shape == (4, env.node_feature_dim)
    assert td0["edge_index"].shape == (2, env.graph_max_edges)
    assert td0["edge_attr"].shape == (env.graph_max_edges, env.edge_feature_dim)
    assert td0["edge_mask"].shape == (env.graph_max_edges,)
    assert td0["edge_count"].shape == (1,)

    action = env.action_spec.rand()
    td_in = td0.clone()
    td_in.set("upper_action", action["upper_action"])
    td_in.set("lower_action", action["lower_action"])
    td_out = env.step(td_in)
    assert ("next", "reward") in td_out.keys(True, True)
    assert td_out["next", "reward"].shape == (4, 1)
    assert td_out["next", "done"].shape == (1,)


def test_torchrl_adapter_rollout_minimal() -> None:
    env = TriSatFlowTorchRLEnv(
        scenario=ScenarioConfig(n_leo=3, episode_len=6, seed=7),
        device="cpu",
    )
    traj = env.rollout(max_steps=3)
    assert traj.batch_size == torch.Size([3])
    assert traj["next", "reward"].shape == (3, 3, 1)
    assert traj["upper_action"].shape == (3, 3)
    assert traj["lower_action"].shape == (3, 3, 3)


def test_grouped_adapter_rollout_minimal() -> None:
    env = TriSatFlowBenchMARLEnv(
        scenario=ScenarioConfig(n_leo=3, episode_len=6, seed=9),
        device="cpu",
        group_name="leo",
        include_graph_specs=True,
    )
    traj = env.rollout(max_steps=3)
    assert traj.batch_size == torch.Size([3])
    assert traj["leo", "observation"].shape == (3, 3, env.node_feature_dim)
    assert traj["leo", "edge_index"].shape == (3, 2, env.graph_max_edges)
    assert traj["leo", "edge_attr"].shape == (3, env.graph_max_edges, env.edge_feature_dim)
    assert traj["leo", "action", "upper_action"].shape == (3, 3)
    assert traj["leo", "action", "lower_action"].shape == (3, 3, 3)


def test_benchmarl_task_registry_and_env_fun_rollout() -> None:
    _ensure_trisatflow_registry_entry()
    assert "trisatflow/mixed_small" in task_config_registry
    task_enum = task_config_registry["trisatflow/mixed_small"]
    local_yaml = Path(__file__).resolve().parents[1] / "benchmarl" / "conf" / "task" / "trisatflow" / "mixed_small.yaml"
    assert local_yaml.exists()
    cfg = yaml.safe_load(local_yaml.read_text(encoding="utf-8")) or {}
    cfg = {k: v for k, v in cfg.items() if k != "defaults"}
    task = task_enum.get_task(config=cfg)
    env = task.get_env_fun(num_envs=1, continuous_actions=True, seed=13, device="cpu")()
    td = env.rollout(max_steps=2)
    assert td["leo", "observation"].shape[0] == 2
    assert td["leo", "edge_index"].shape[-2] == 2
