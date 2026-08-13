from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from scripts.evaluate_baseline_lower_fairness import evaluate_one, _write_outputs
from trisatflow.baselines.fair_wrappers import wrap_baseline_with_lower_allocator
from trisatflow.baselines.lower_allocators import (
    NeutralAllocator,
    OptimizedGreedyLowerAllocator,
    SameLearnedLowerAllocator,
)
from trisatflow.baselines.registry import build_baseline_policy
from trisatflow.config import ScenarioConfig, TrainConfig


def _candidate_info() -> dict[int, dict[str, object]]:
    return {
        0: {"is_available": True, "estimated_delay": 2.0, "estimated_energy_j": 0.0, "estimated_queue": 2.0, "rate_mbps": 1000.0},
        1: {"is_available": True, "estimated_delay": 1.0, "estimated_energy_j": 0.2, "estimated_queue": 1.0, "rate_mbps": 50.0},
        2: {"is_available": True, "estimated_delay": 1.5, "estimated_energy_j": 0.3, "estimated_queue": 1.0, "rate_mbps": 40.0},
        3: {"is_available": True, "estimated_delay": 1.2, "estimated_energy_j": 0.4, "estimated_queue": 1.0, "rate_mbps": 35.0},
    }


def test_neutral_allocator_returns_fixed_values() -> None:
    allocator = NeutralAllocator([0.35, 0.60, 0.45])
    action = allocator.allocate(None, {}, 0, {})

    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert np.allclose(action, [0.35, 0.60, 0.45])


def test_optimized_greedy_allocator_returns_legal_triplet() -> None:
    allocator = OptimizedGreedyLowerAllocator()
    action = allocator.allocate(None, {"deadline_threshold": 1.0}, 1, _candidate_info())

    assert action.shape == (3,)
    assert float(action.min()) >= 0.0
    assert float(action.max()) <= 1.0


def test_wrapper_preserves_upper_action_and_adds_allocator_metadata() -> None:
    baseline = build_baseline_policy("geo_only")
    wrapped = wrap_baseline_with_lower_allocator(baseline, NeutralAllocator([0.3, 0.4, 0.5]))
    decision = wrapped.select_action(
        obs=None,
        state={},
        mask=[1, 1, 1, 1],
        candidate_info=_candidate_info(),
        rng=random.Random(4),
    )

    assert decision["upper_action"] == 2
    assert np.allclose(decision["lower_action"], [0.5, 0.3, 0.4])
    assert decision["decision_info"]["lower_allocator_name"] == "neutral"


def test_same_learned_allocator_missing_checkpoint_graceful_skip() -> None:
    allocator = SameLearnedLowerAllocator(Path("outputs/reviewer_repair/lower_fairness/test_missing/missing.pt"))
    action = allocator.allocate(None, {}, 0, {})

    assert allocator.available is False
    assert allocator.skip_reason == "checkpoint_not_provided_or_missing"
    assert np.allclose(action, [1.0, 1.0, 1.0])


def test_evaluator_outputs_lower_allocator_fields() -> None:
    output_dir = Path("outputs/reviewer_repair/lower_fairness/test_evaluator")
    cfg = TrainConfig(scenario=ScenarioConfig(n_leo=4, episode_len=2, seed=31))
    row = evaluate_one(
        cfg=cfg,
        baseline_name="geo_only",
        lower_allocator_name="neutral",
        checkpoint=None,
        neutral_values=[0.35, 0.60, 0.45],
        episodes=1,
        steps=2,
        device="cpu",
        seed=31,
    )
    _write_outputs([row], output_dir)
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert row["lower_allocator"] == "neutral"
    assert row["upper_policy"] == "geo_only"
    assert np.allclose(row["lower_action_mean"], [0.45, 0.35, 0.60])
    assert payload["rows"][0]["lower_allocator"] == "neutral"
