from __future__ import annotations

import csv

import torch

from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.obs_schema import IDX_GEO_VISIBLE, IDX_GROUND_VISIBLE, IDX_NEIGHBOR_VISIBLE
from trisatflow.envs.topology_trace import TopologyTraceProvider


def _write_trace(path):
    fields = [
        "step",
        "leo_id",
        "abstract_action_mask_visible",
        "abstract_action_mask_completion_safe",
        "abstract_action_mask_mobility_safe",
        "abstract_action_mask_final",
        "local_rate",
        "neighbor_rate",
        "geo_rate",
        "ground_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "step": 0,
                "leo_id": 0,
                "abstract_action_mask_visible": "[1,1,1,1]",
                "abstract_action_mask_completion_safe": "[1,0,0,0]",
                "abstract_action_mask_mobility_safe": "[1,0,0,0]",
                "abstract_action_mask_final": "[1,0,0,0]",
                "local_rate": 1000,
                "neighbor_rate": 800,
                "geo_rate": 400,
                "ground_rate": 400,
            }
        )


def test_trace_snapshot_keeps_raw_visibility_separate_from_final_mask(tmp_path) -> None:
    trace_path = tmp_path / "trace.csv"
    _write_trace(trace_path)
    provider = TopologyTraceProvider(trace_path, n_leo=1, device=torch.device("cpu"), repeat=True)

    snapshot = provider.snapshot(0)

    assert snapshot.abstract_action_mask_visible.tolist() == [[True, True, True, True]]
    assert snapshot.abstract_action_mask_final.tolist() == [[True, False, False, False]]
    assert snapshot.local_visible.tolist() == [1.0]
    assert snapshot.neighbor_visible.tolist() == [1.0]
    assert snapshot.geo_visible.tolist() == [1.0]
    assert snapshot.ground_visible.tolist() == [1.0]


def test_observation_uses_raw_visibility_not_final_action_mask(tmp_path) -> None:
    trace_path = tmp_path / "trace.csv"
    _write_trace(trace_path)
    scenario = ScenarioConfig(
        n_leo=1,
        episode_len=1,
        topology_mode="satedgesim_trace",
        topology_trace_path=str(trace_path),
        topology_trace_repeat=True,
        action_mask_layer_mode="full",
        mask_source="oracle_trace",
    )
    scenario.physical.enabled = False
    env = GeoLeoGroundEnv(scenario)

    obs, _edge_index, _edge_attr = env.reset()
    mask_details = env._upper_action_mask_details_at_step(env.t)

    assert mask_details.raw_visibility_mask.tolist() == [[True, True, True, True]]
    assert mask_details.final_action_mask.tolist() == [[True, False, False, False]]
    assert obs[0, IDX_NEIGHBOR_VISIBLE].item() == 1.0
    assert obs[0, IDX_GEO_VISIBLE].item() == 1.0
    assert obs[0, IDX_GROUND_VISIBLE].item() == 1.0
