from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_builder import build_shared_observation
from trisatflow.models import upper_action_mask_from_obs


def main() -> None:
    synthetic = {
        "leo_id": 3,
        "local_visible": 1,
        "neighbor_visible": 1,
        "geo_visible": 1,
        "ground_visible": 0,
        "local_rate": 1000.0,
        "neighbor_rate": 620.0,
        "geo_rate": 240.0,
        "ground_rate": 0.0,
        "local_delay": 0.021,
        "neighbor_delay": 0.015,
        "geo_delay": 0.037,
        "ground_delay": 0.0,
        "local_queue": 4.0,
        "neighbor_queue": 7.0,
        "geo_queue": 3.0,
        "ground_queue": 0.0,
    }
    trace_obs = build_shared_observation([synthetic], source_index=0, node_feature_dim=16)
    live_obs = build_shared_observation([dict(synthetic)], source_index=0, node_feature_dim=16)
    if not torch.allclose(trace_obs.obs, live_obs.obs):
        raise SystemExit("obs tensor mismatch")
    if not torch.equal(trace_obs.mask, live_obs.mask):
        raise SystemExit("mask tensor mismatch")
    trace_mask = upper_action_mask_from_obs(trace_obs.obs)
    live_mask = upper_action_mask_from_obs(live_obs.obs)
    if not torch.equal(trace_mask, live_mask):
        raise SystemExit("upper_action_mask_from_obs mismatch")
    payload = {
        "status": "OBS_SCHEMA_CONSISTENT",
        "obs_shape": list(trace_obs.obs.shape),
        "mask": trace_mask.int().tolist(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
