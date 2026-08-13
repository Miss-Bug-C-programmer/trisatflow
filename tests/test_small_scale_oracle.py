from __future__ import annotations

import math

import torch

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
from trisatflow.oracles.small_scale_grid_oracle import SmallScaleGridOracle, compute_oracle_gap


def _env(n_leo: int = 2) -> GeoLeoGroundEnv:
    scenario = ScenarioConfig(n_leo=n_leo, n_geo=1, n_ground=1, episode_len=4, seed=5)
    return GeoLeoGroundEnv(scenario, RewardWeights(), torch.device("cpu"))


def test_oracle_returns_feasible_upper_lower_and_finite_cost() -> None:
    env = _env(2)
    env.reset(rule_baseline_observation=True)
    oracle = SmallScaleGridOracle(resource_grid=(1.0,), max_exact_candidates=1000)
    result = oracle.solve_one_step(env)
    mask = env._upper_action_mask_at_step(env.t)
    assert result.oracle_action.shape == (2,)
    assert result.oracle_lower_action.shape == (2, 3)
    assert math.isfinite(result.oracle_cost)
    assert bool((result.oracle_lower_action >= 0.0).all() and (result.oracle_lower_action <= 1.0).all())
    for idx, action in enumerate(result.oracle_action.tolist()):
        assert bool(mask[idx, int(action)].item())


def test_oracle_gap_no_divide_by_zero() -> None:
    gap = compute_oracle_gap(method_cost=1.0, oracle_cost=0.0)
    assert math.isfinite(gap)


def test_exact_grid_and_beam_metadata_are_distinct() -> None:
    env = _env(1)
    env.reset(rule_baseline_observation=True)
    exact = SmallScaleGridOracle(resource_grid=(1.0,), max_exact_candidates=1000).solve_one_step(env)
    assert exact.oracle_mode == "exact_grid"
    assert exact.exact is True

    env = _env(4)
    env.reset(rule_baseline_observation=True)
    beam = SmallScaleGridOracle(resource_grid=(0.25, 0.5, 0.75, 1.0), max_exact_candidates=8, beam_width=2).solve_one_step(env)
    assert beam.oracle_mode == "beam_grid_approx"
    assert beam.exact is False


def test_oracle_uses_same_env_step_cost_estimator() -> None:
    env = _env(2)
    env.reset(rule_baseline_observation=True)
    result = SmallScaleGridOracle(resource_grid=(1.0,), max_exact_candidates=1000).solve_one_step(env)
    assert result.metadata["uses_same_env_step_cost_estimator"] is True
    assert result.metadata["oracle_name_guard"] == "grid_oracle_not_minlp"


def test_infeasible_action_not_in_candidate_set() -> None:
    scenario = ScenarioConfig(n_leo=2, n_geo=1, n_ground=1, episode_len=4, seed=5, enable_geo=False, enable_ground=False)
    env = GeoLeoGroundEnv(scenario, RewardWeights(), torch.device("cpu"))
    env.reset(rule_baseline_observation=True)
    result = SmallScaleGridOracle(resource_grid=(1.0,), max_exact_candidates=1000).solve_one_step(env)
    assert all(int(action) in {0, 1} for action in result.oracle_action.tolist())

