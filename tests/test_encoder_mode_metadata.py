from __future__ import annotations

import pytest
import torch

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import AlgoConfig, ExperimentConfig, ScenarioConfig, TrainConfig, canonical_train_config_dict
from trisatflow.encoder_modes import encoder_mode_semantics, validate_checkpoint_encoder_metadata
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy


def test_canonical_encoder_modes_define_detach_and_trainability() -> None:
    expected = {
        "shared_upper_detached_lower": (True, False, True),
        "shared_joint": (False, True, True),
        "separate_lower_encoder": (False, True, True),
        "shared_frozen": (True, False, False),
    }

    for mode, fields in expected.items():
        semantics = encoder_mode_semantics(mode)
        assert (
            semantics.lower_embed_detached,
            semantics.lower_encoder_trainable,
            semantics.upper_encoder_trainable,
        ) == fields


def _trainer(mode: str) -> HierarchicalTrainer:
    cfg = TrainConfig(
        total_episodes=1,
        steps_per_episode=1,
        scenario=ScenarioConfig(n_leo=2, episode_len=1, enable_gnn=False),
        algo=AlgoConfig(
            upper_algo="mappo",
            lower_algo="maddpg",
            encoder_mode=mode,
            gnn_hidden_dim=8,
            policy_hidden_dim=16,
            lower_batch_size=2,
            lower_warmup=1,
        ),
    )
    return HierarchicalTrainer(cfg)


def test_trainer_and_checkpoint_metadata_match_actual_encoder_mode(tmp_path) -> None:
    trainer = _trainer("shared_joint")
    checkpoint = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")

    assert payload["encoder_mode_metadata"] == {
        "encoder_mode": "shared_joint",
        "lower_embed_detached": False,
        "lower_encoder_trainable": True,
        "upper_encoder_trainable": True,
        "uses_separate_lower_encoder": False,
    }
    assert trainer.lower_agent.encoder_mode == "shared_joint"
    assert trainer.lower_agent.stop_gradient_to_encoder_from_lower is False


def test_shared_frozen_freezes_upper_shared_encoder_parameters() -> None:
    trainer = _trainer("shared_frozen")

    assert trainer.cfg.algo.encoder_mode == "shared_frozen"
    assert all(not p.requires_grad for p in trainer.encoder.parameters())
    assert trainer.lower_agent.stop_gradient_to_encoder_from_lower is True


def test_formal_checkpoint_metadata_missing_fails_fast(tmp_path) -> None:
    cfg = TrainConfig(
        experiment=ExperimentConfig(paper_ready=True),
        algo=AlgoConfig(encoder_mode="shared_joint"),
    )
    checkpoint = tmp_path / "missing_encoder_metadata.pt"
    torch.save({"config": canonical_train_config_dict(cfg)}, checkpoint)

    with pytest.raises(ValueError, match="formal eval requires checkpoint encoder_mode_metadata"):
        FrozenTriSatFlowPolicy(checkpoint, device="cpu")


def test_checkpoint_metadata_conflict_fails_fast() -> None:
    algo = AlgoConfig(encoder_mode="shared_joint")
    payload = {
        "encoder_mode_metadata": {
            "encoder_mode": "shared_upper_detached_lower",
            "lower_embed_detached": True,
            "lower_encoder_trainable": False,
            "upper_encoder_trainable": True,
            "uses_separate_lower_encoder": False,
        }
    }

    with pytest.raises(ValueError, match="conflicts with config"):
        validate_checkpoint_encoder_metadata(payload, algo, formal=True)
