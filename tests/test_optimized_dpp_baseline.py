from __future__ import annotations

import random

from trisatflow.baselines.offline_adapter import OfflineBaselineAdapter
from trisatflow.baselines.optimized_dpp import OptimizedLyapunovDppPolicy
from trisatflow.baselines.registry import build_baseline_policy
from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


def _candidate_info() -> dict[int, dict[str, object]]:
    return {
        0: {"is_available": True, "estimated_delay": 2.0, "estimated_queue": 4.0, "estimated_energy_j": 0.0, "mobility_risk": 0.0, "rate_mbps": 1000.0},
        1: {"is_available": True, "estimated_delay": 0.8, "estimated_queue": 1.0, "estimated_energy_j": 0.2, "mobility_risk": 0.1, "rate_mbps": 50.0},
        2: {"is_available": False, "estimated_delay": 0.1, "estimated_queue": 0.1, "estimated_energy_j": 0.1, "mobility_risk": 0.0, "rate_mbps": 100.0},
        3: {"is_available": True, "estimated_delay": 1.4, "estimated_queue": 2.0, "estimated_energy_j": 0.3, "mobility_risk": 0.2, "rate_mbps": 30.0},
    }


def test_optimized_dpp_returns_feasible_action_and_lower_bounds() -> None:
    policy = OptimizedLyapunovDppPolicy(grid_mode="grid_low")
    decision = policy.select_action(
        obs=None,
        state={"local_queue": 4.0, "deadline_threshold": 1.0},
        mask=[0, 1, 0, 0],
        candidate_info=_candidate_info(),
        rng=random.Random(1),
    )

    assert decision["upper_action"] == 1
    assert len(decision["lower_action"]) == 3
    assert all(0.0 <= value <= 1.0 for value in decision["lower_action"])
    assert decision["decision_info"]["queue_stability_claim_allowed"] is False


def test_optimized_dpp_does_not_select_infeasible_action() -> None:
    policy = OptimizedLyapunovDppPolicy(grid_mode="grid_low")
    decision = policy.select_action(
        obs=None,
        state={"local_queue": 5.0, "deadline_threshold": 1.0},
        mask=[1, 0, 1, 0],
        candidate_info=_candidate_info(),
        rng=random.Random(2),
    )

    assert decision["upper_action"] == 0


def test_registry_builds_optimized_dpp_aliases() -> None:
    assert build_baseline_policy("optimized_lyapunov_dpp").name == "optimized_lyapunov_dpp"
    assert build_baseline_policy("lyapunov_dpp_optimized").name == "lyapunov_dpp_optimized"


def test_offline_adapter_preserves_optimized_lower_action() -> None:
    env = GeoLeoGroundEnv(ScenarioConfig(n_leo=4, episode_len=1, seed=21))
    env.reset(rule_baseline_observation=True)
    adapter = OfflineBaselineAdapter(OptimizedLyapunovDppPolicy(), rng=random.Random(3))
    batch = adapter.select_actions(env)

    assert batch.lower_action.shape == (4, env.LOWER_ACTION_DIM)
    assert float(batch.lower_action.min().item()) >= 0.0
    assert float(batch.lower_action.max().item()) <= 1.0
