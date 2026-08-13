from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from trisatflow.agents.hierarchical_trainer import HierarchicalTrainer
from trisatflow.config import load_config
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy


def _tiny_cfg(tmp_path: Path):
    cfg = load_config("trisatflow/configs/small.yaml")
    cfg.total_episodes = 1
    cfg.steps_per_episode = 2
    cfg.scenario.episode_len = 2
    cfg.scenario.n_leo = 4
    cfg.scenario.seed = 13
    cfg.device = "cpu"
    cfg.output_dir = str(tmp_path / "run")
    return cfg


def _load_ckpt_into_trainer(trainer: HierarchicalTrainer, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location=trainer.device)
    if "encoder" in payload:
        trainer.encoder.load_state_dict(payload["encoder"], strict=False)
    upper = trainer.upper_agent
    lower = trainer.lower_agent
    for name in ["actor", "critic", "value", "q_net", "mixer", "target_q_net", "target_mixer"]:
        module = getattr(upper, name, None)
        key = f"upper_{name}"
        if module is not None and key in payload and hasattr(module, "load_state_dict"):
            module.load_state_dict(payload[key], strict=False)
    for name in ["encoder", "target_encoder", "actor", "critic", "target_actor", "target_critic"]:
        module = getattr(lower, name, None)
        key = f"lower_{name}"
        if module is not None and key in payload and hasattr(module, "load_state_dict"):
            module.load_state_dict(payload[key], strict=False)


def test_checkpoint_save_load_and_resume(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    trainer_a = HierarchicalTrainer(cfg)
    trainer_a.train()
    ckpt = tmp_path / "checkpoint.pt"
    trainer_a.save_checkpoint(ckpt)
    assert ckpt.exists()

    cfg_b = deepcopy(cfg)
    cfg_b.output_dir = str(tmp_path / "run_resumed")
    trainer_b = HierarchicalTrainer(cfg_b)
    before = next(trainer_b.encoder.parameters()).detach().clone()
    _load_ckpt_into_trainer(trainer_b, ckpt)
    after = next(trainer_b.encoder.parameters()).detach().clone()
    ref = next(trainer_a.encoder.parameters()).detach().clone()
    assert not torch.allclose(before, after)
    assert torch.allclose(after, ref)

    history = trainer_b.train()
    assert len(history) >= 1


def test_eval_only_missing_checkpoint_raises(tmp_path: Path):
    missing = tmp_path / "missing_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        FrozenTriSatFlowPolicy(missing, device="cpu")
