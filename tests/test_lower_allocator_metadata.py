from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from scripts.evaluate_baseline_lower_fairness import _write_outputs, evaluate_one
from trisatflow.baselines.fair_wrappers import wrap_baseline_with_lower_allocator
from trisatflow.baselines.lower_allocators import (
    NeutralAllocator,
    OptimizedGreedyLowerAllocator,
    OracleLowerAllocator,
    lower_allocator_metadata,
)
from trisatflow.baselines.registry import build_baseline_policy
from trisatflow.config import ScenarioConfig, TrainConfig


def _candidate_info() -> dict[int, dict[str, object]]:
    return {
        0: {"estimated_delay": 2.0, "estimated_energy_j": 0.0, "estimated_queue": 2.0, "estimated_cost": 2.0},
        1: {"estimated_delay": 1.0, "estimated_energy_j": 0.2, "estimated_queue": 1.0, "estimated_cost": 1.0},
        2: {"estimated_delay": 1.5, "estimated_energy_j": 0.3, "estimated_queue": 1.0, "estimated_cost": 1.5},
        3: {"estimated_delay": 1.2, "estimated_energy_j": 0.4, "estimated_queue": 1.0, "estimated_cost": 1.2},
    }


def test_neutral_optimized_and_oracle_allocator_shapes() -> None:
    allocators = [
        NeutralAllocator([0.2, 0.4, 0.6]),
        OptimizedGreedyLowerAllocator(grid=[0.5, 1.0]),
        OracleLowerAllocator(grid=[0.5, 1.0]),
    ]

    for allocator in allocators:
        action = allocator.allocate(None, {"deadline_threshold": 1.0}, 1, _candidate_info())
        metadata = lower_allocator_metadata(allocator)
        assert action.shape == (3,)
        assert float(action.min()) >= 0.0
        assert float(action.max()) <= 1.0
        assert metadata["fallback_allocator"] == ""


def test_fairness_wrapper_writes_allocator_provenance_per_decision() -> None:
    wrapped = wrap_baseline_with_lower_allocator(build_baseline_policy("geo_only"), NeutralAllocator([0.3, 0.4, 0.5]))
    decision = wrapped.select_action(
        obs=None,
        state={},
        mask=[1, 1, 1, 1],
        candidate_info=_candidate_info(),
        rng=random.Random(7),
    )
    info = decision["decision_info"]

    assert info["requested_allocator"] == "neutral"
    assert info["effective_lower_allocator"] == "neutral"
    assert info["fallback_allocator"] == ""
    assert info["formal_claim_allowed"] is True
    assert np.allclose(info["lower_action_env_order_values"], [0.5, 0.3, 0.4])


def test_fairness_evaluator_summary_marks_debug_same_learned_fallback_non_formal(tmp_path: Path) -> None:
    cfg = TrainConfig(scenario=ScenarioConfig(n_leo=4, episode_len=2, seed=19))
    row = evaluate_one(
        cfg=cfg,
        baseline_name="geo_only",
        lower_allocator_name="same_learned",
        checkpoint=str(tmp_path / "missing_lower_checkpoint.pt"),
        neutral_values=None,
        episodes=1,
        steps=2,
        device="cpu",
        seed=19,
    )
    _write_outputs([row], tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert row["requested_allocator"] == "same_learned"
    assert row["effective_lower_allocator"] == "neutral"
    assert row["same_learned_lower_loaded"] is False
    assert row["fallback_allocator"] == "neutral"
    assert row["formal_claim_allowed"] is False
    assert summary["formal_collector_allowed"] is False
    assert summary["rows"][0]["fallback_allocator"] == "neutral"
