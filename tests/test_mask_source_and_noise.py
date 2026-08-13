from __future__ import annotations

import torch

from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.action_masks import build_upper_action_mask


def _mask_diag(**kwargs):
    visibility = torch.ones((2, 4), dtype=torch.bool)
    architecture = torch.ones((1, 4), dtype=torch.bool)
    return build_upper_action_mask(
        visibility_mask=visibility,
        architecture_mask=architecture,
        trace_snapshot=None,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="visible_only",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
        **kwargs,
    )


def test_oracle_trace_metadata_is_not_deployable() -> None:
    diag = _mask_diag(mask_source="oracle_trace")

    assert diag.mask_source == "oracle_trace"
    assert diag.uses_oracle_trace_mask is True
    assert diag.deployable is False


def test_predicted_metadata_is_deployable() -> None:
    diag = _mask_diag(mask_source="predicted")

    assert diag.mask_source == "predicted"
    assert diag.uses_oracle_trace_mask is False
    assert diag.deployable is True


def test_noise_and_staleness_are_recorded_in_env_info() -> None:
    env = GeoLeoGroundEnv(
        ScenarioConfig(
            n_leo=4,
            episode_len=2,
            seed=41,
            mask_source="predicted",
            action_mask_layer_mode="full",
            link_lifetime_noise_std_s=1.0,
            completion_time_noise_std_s=1.0,
            mask_false_positive_rate=1.0,
            mask_false_negative_rate=0.0,
            mask_staleness_slots=2,
        )
    )
    env.reset()
    upper = env._upper_action_mask_at_step(env.t).float().argmax(dim=-1).long()
    lower = torch.ones((env.n_agents, env.LOWER_ACTION_DIM))
    step = env.step(upper, lower, minimal_info=True)

    assert step.info["mask_source"] == "predicted"
    assert float(step.info["mask_staleness_slots"].max().item()) == 2.0
    assert float(step.info["link_lifetime_noise_std_s"].max().item()) == 1.0
    assert float(step.info["completion_time_noise_std_s"].max().item()) == 1.0
    assert float(step.info["configured_mask_false_positive_rate"].max().item()) == 1.0


def test_no_feasible_action_fallback_is_safe() -> None:
    visibility = torch.zeros((2, 4), dtype=torch.bool)
    architecture = torch.zeros((1, 4), dtype=torch.bool)
    diag = build_upper_action_mask(
        visibility_mask=visibility,
        architecture_mask=architecture,
        trace_snapshot=None,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="visible_only",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
        mask_source="measured",
    )

    assert diag.final_mask.shape == (2, 4)
    assert diag.final_mask.any(dim=-1).all()
    assert diag.fallback_due_empty_mask_count.sum().item() == 2.0


def test_false_positive_does_not_bypass_architecture_mask() -> None:
    visibility = torch.ones((1, 4), dtype=torch.bool)
    architecture = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    predicted = torch.ones((1, 4), dtype=torch.bool)
    diag = build_upper_action_mask(
        visibility_mask=visibility,
        architecture_mask=architecture,
        trace_snapshot=None,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="visible_only",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
        mask_source="predicted",
        predicted_completion_safe_mask=predicted,
        predicted_mobility_safe_mask=predicted,
        mask_false_positive_rate=1.0,
        mask_false_positive_rate_observed=torch.ones(1),
    )

    assert diag.final_mask.tolist() == [[True, False, False, False]]
    assert diag.mask_false_positive_rate == 1.0
    assert diag.mask_false_positive_rate_observed.item() == 1.0
