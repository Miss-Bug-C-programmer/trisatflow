from __future__ import annotations

import pytest

from trisatflow.baselines.registry import baseline_metadata, baseline_names, validate_baseline_for_formal


def test_placeholder_baseline_fails_in_formal_mode() -> None:
    meta = baseline_metadata("hmadrl_maddqn_ddpg")

    assert meta.is_placeholder is True
    assert meta.allows_formal_eval is False
    with pytest.raises(ValueError, match="is_placeholder=true"):
        validate_baseline_for_formal("hmadrl_maddqn_ddpg")


def test_debug_listing_can_include_placeholder_but_metadata_blocks_formal() -> None:
    names = baseline_names(paper_ready_only=False, include_placeholder=True)

    assert "hmadrl_maddqn_ddpg" in names
    meta = baseline_metadata("hmadrl_maddqn_ddpg").to_dict()
    assert meta["paper_ready"] is False
    assert meta["is_placeholder"] is True
    assert meta["allows_formal_eval"] is False
    assert meta["fallback_policy"] == "random_visible"
