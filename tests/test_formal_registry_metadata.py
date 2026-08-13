from __future__ import annotations

from trisatflow.baselines.registry import (
    baseline_metadata_json,
    formal_baseline_metadata_registry,
    formal_baseline_names,
)
from trisatflow.baselines.strong_registry import list_strong_baselines, strong_baseline_metadata


REQUIRED_BASELINE_SCHEMA = {
    "baseline_name",
    "name",
    "type",
    "implemented",
    "uses_oracle",
    "uses_privileged_info",
    "trainable",
    "requires_checkpoint",
    "checkpoint_loaded",
    "paper_ready",
    "is_placeholder",
    "allows_formal_eval",
    "fallback_policy",
    "update_implemented",
}

REQUIRED_STRONG_SCHEMA = {
    "method",
    "baseline_name",
    "baseline_family",
    "implemented",
    "trainable",
    "requires_checkpoint",
    "checkpoint_loaded",
    "paper_ready",
    "is_placeholder",
    "allows_formal_eval",
    "fallback_policy",
    "update_implemented",
    "mask_supported",
    "action_mask_supported",
    "continuous_action_supported",
    "smoke_training_passed",
    "full_experiment_required",
}


def test_formal_registry_only_contains_allowed_baselines() -> None:
    registry = formal_baseline_metadata_registry()

    assert registry
    assert set(registry) == set(formal_baseline_names())
    for name, meta in registry.items():
        assert meta.allows_formal_eval is True
        assert meta.is_placeholder is False
        assert meta.paper_ready is True
        assert meta.name == name


def test_blocked_baselines_remain_visible_in_debug_metadata() -> None:
    metadata = baseline_metadata_json()

    assert REQUIRED_BASELINE_SCHEMA <= set(next(iter(metadata.values())).keys())
    assert metadata["hmadrl_maddqn_ddpg"]["allows_formal_eval"] is False
    assert metadata["hmadrl_maddqn_ddpg"]["is_placeholder"] is True
    assert metadata["tri_mappo_maddpg"]["requires_checkpoint"] is True
    assert metadata["tri_mappo_maddpg"]["checkpoint_loaded"] is False
    assert metadata["tri_mappo_maddpg"]["allows_formal_eval"] is False
    assert "hmadrl_maddqn_ddpg" not in formal_baseline_names()
    assert "tri_mappo_maddpg" not in formal_baseline_names()


def test_strong_baseline_metadata_uses_formal_gate_schema() -> None:
    all_meta = list_strong_baselines()

    assert all_meta
    for name, meta in all_meta.items():
        assert REQUIRED_STRONG_SCHEMA <= set(meta.keys())
        assert meta["baseline_name"] in {name, "small_scale_grid_oracle"}
        assert meta["paper_ready"] is False
        assert meta["allows_formal_eval"] is False
    assert strong_baseline_metadata("attention_candidate")["update_implemented"] is False
    assert strong_baseline_metadata("attention_candidate")["is_placeholder"] is True
