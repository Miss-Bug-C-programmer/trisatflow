from __future__ import annotations

import csv

import torch

from trisatflow.config import ScenarioConfig, TrainConfig, load_config
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.action_masks import build_upper_action_mask


def test_visibility_mask_blocks_invisible_targets() -> None:
    visibility = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)
    arch = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    diag = build_upper_action_mask(
        visibility_mask=visibility,
        architecture_mask=arch,
        trace_snapshot=None,
        action_mask_enabled=True,
        mode="visibility",
        legacy_mode="visible_only",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
    )
    assert diag.final_mask.tolist() == [[True, False, False, True]]
    assert float(diag.visibility_mask_ratio.item()) > 0.0


def test_local_fallback_when_everything_masked() -> None:
    visibility = torch.tensor([[0, 0, 0, 0]], dtype=torch.bool)
    arch = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    diag = build_upper_action_mask(
        visibility_mask=visibility,
        architecture_mask=arch,
        trace_snapshot=None,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="completion_safe",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
    )
    assert diag.final_mask.tolist() == [[True, False, False, False]]
    assert float(diag.final_count.item()) == 1.0


def test_environment_action_mask_modes_smoke() -> None:
    for mode in ("none", "visibility", "completion_safe", "mobility_risk", "full"):
        env = GeoLeoGroundEnv(
            ScenarioConfig(
                n_leo=4,
                episode_len=2,
                seed=5,
                action_mask_layer_mode=mode,
                action_mask_mode="completion_safe",
            )
        )
        _obs, _edge_index, _edge_attr = env.reset()
        upper = torch.tensor([0, 1, 2, 3])
        lower = torch.ones(4, 3) * 0.5
        step = env.step(upper, lower)
        assert "masked_action_ratio" in step.info
        assert "visibility_mask_ratio" in step.info
        assert "completion_mask_ratio" in step.info
        assert "mobility_mask_ratio" in step.info
        assert "action_mask_final_valid_count" in step.info


def test_full_mask_with_all_zero_trace_does_not_crash(tmp_path) -> None:
    trace_path = tmp_path / "trace.csv"
    with open(trace_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "leo_id",
                "abstract_action_mask_visible",
                "abstract_action_mask_mobility_safe",
                "abstract_action_mask_completion_safe",
            ],
        )
        writer.writeheader()
        for step in (0, 1):
            for leo in (0, 1):
                row = {
                    "step": step,
                    "leo_id": leo,
                    "abstract_action_mask_visible": "[0,0,0,0]",
                    "abstract_action_mask_mobility_safe": "[0,0,0,0]",
                    "abstract_action_mask_completion_safe": "[0,0,0,0]",
                }
                writer.writerow(row)

    env = GeoLeoGroundEnv(
        ScenarioConfig(
            n_leo=2,
            episode_len=2,
            seed=7,
            topology_mode="satedgesim_trace",
            topology_trace_path=str(trace_path),
            action_mask_layer_mode="full",
            action_mask_mode="completion_safe",
        )
    )
    _obs, _edge_index, _edge_attr = env.reset()
    upper = torch.tensor([3, 3])
    lower = torch.ones(2, 3) * 0.5
    step = env.step(upper, lower)
    assert step.obs.shape[0] == 2
    assert float(step.info["action_mask_final_valid_count"].min().item()) >= 1.0


def test_environment_action_mask_alias_from_config(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "total_episodes: 1",
                "scenario:",
                "  n_leo: 2",
                "  episode_len: 2",
                "environment:",
                "  action_mask:",
                "    mode: full",
                "    enable_visibility_mask: true",
                "    enable_completion_safe_mask: true",
                "    enable_mobility_risk_mask: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.scenario.action_mask_layer_mode == "full"
    assert cfg.scenario.enable_visibility_mask is True
    assert cfg.scenario.enable_completion_safe_mask is True
    assert cfg.scenario.enable_mobility_risk_mask is True
