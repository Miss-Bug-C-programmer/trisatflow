from __future__ import annotations

import torch

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


def test_physical_mask_predictor_uses_physical_seconds_metadata() -> None:
    scenario = ScenarioConfig(
        n_leo=3,
        episode_len=1,
        seed=11,
        mask_source="predicted",
        action_mask_layer_mode="full",
        mask_prediction_horizon_s=2.0,
    )
    scenario.physical.enabled = True
    env = GeoLeoGroundEnv(scenario, RewardWeights(mode="physical_weighted"), "cpu")

    env.reset()
    details = env._upper_action_mask_details_at_step(env.t)

    assert details.predicted_safety_mask.shape == (scenario.n_leo, env.N_UPPER_ACTIONS)
    assert env._mask_predictor_units_for_step(env.t) == "physical_seconds"
    assert env._mask_predictor_units_for_step(env.t) != "legacy_normalized_debug"


def test_physical_step_reports_physical_mask_predictor_units() -> None:
    scenario = ScenarioConfig(
        n_leo=2,
        episode_len=1,
        seed=13,
        mask_source="predicted",
        action_mask_layer_mode="full",
    )
    scenario.physical.enabled = True
    env = GeoLeoGroundEnv(scenario, RewardWeights(mode="physical_weighted"), "cpu")
    env.reset()

    upper = env._upper_action_mask_at_step(env.t).float().argmax(dim=-1).long()
    lower = torch.ones((scenario.n_leo, env.LOWER_ACTION_DIM), dtype=torch.float32)
    step = env.step(upper, lower)

    assert step.info["mask_predictor_units"] == "physical_seconds"
    assert step.info["formal_claim_allowed"] is True


def test_legacy_debug_predictor_metadata_is_not_physical() -> None:
    scenario = ScenarioConfig(
        n_leo=2,
        episode_len=1,
        seed=17,
        mask_source="predicted",
        action_mask_layer_mode="full",
    )
    scenario.physical.enabled = False
    env = GeoLeoGroundEnv(scenario, RewardWeights(mode="legacy_remote_biased"), "cpu")

    env.reset()
    env._upper_action_mask_details_at_step(env.t)

    assert env._mask_predictor_units_for_step(env.t) == "legacy_normalized_debug"
