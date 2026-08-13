from __future__ import annotations

import random

import pytest

from trisatflow.baselines.registry import baseline_metadata, build_baseline_policy, validate_baseline_for_formal


def test_checkpoint_required_baseline_without_checkpoint_fails_formal_gate() -> None:
    meta = baseline_metadata("tri_mappo_maddpg")

    assert meta.requires_checkpoint is True
    assert meta.checkpoint_loaded is False
    assert meta.allows_formal_eval is False
    with pytest.raises(ValueError, match="requires_checkpoint=true but checkpoint_loaded=false"):
        validate_baseline_for_formal("tri_mappo_maddpg")


def test_checkpoint_required_baseline_can_pass_gate_only_with_loaded_checkpoint_metadata() -> None:
    meta = validate_baseline_for_formal("tri_mappo_maddpg", checkpoint_loaded=True)

    assert meta.requires_checkpoint is True
    assert meta.checkpoint_loaded is True
    assert meta.allows_formal_eval is True


def test_tri_mappo_facade_does_not_fallback_to_random_visible() -> None:
    policy = build_baseline_policy("tri_mappo_maddpg")

    with pytest.raises(RuntimeError, match="requires a trained checkpoint"):
        policy.select_action(
            obs=None,
            state={},
            mask=[1, 1, 1, 1],
            candidate_info={},
            rng=random.Random(13),
        )
