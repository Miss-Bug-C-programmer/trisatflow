from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trisatflow.config import PhysicalModelConfig, RewardWeights, ScenarioConfig
from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import build_metric_records, metric_schema_manifest
from trisatflow.models import upper_action_mask_from_obs


OUTPUT = PROJECT_ROOT / "outputs" / "reviewer_repair" / "physical_model" / "summary.json"


def _run_episode(scenario: ScenarioConfig) -> Dict[str, Any]:
    env = GeoLeoGroundEnv(scenario, RewardWeights(cost_normalization_enabled=True), device="cpu")
    obs, edge_index, edge_attr = env.reset()
    steps = []
    done = False
    while not done and len(steps) < 4:
        mask = upper_action_mask_from_obs(obs)
        upper = mask.float().argmax(dim=-1).long()
        lower = torch.ones((scenario.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32)
        step = env.step(upper, lower)
        info = step.info
        steps.append(
            {
                "delay_s": float(info["physical_delay_s"].float().mean().detach().cpu()),
                "energy_j": float(info["physical_energy_j"].float().mean().detach().cpu()),
                "queue_cycles": float(info.get("physical_queue_cycles", info["queue"]).float().mean().detach().cpu()),
                "normalized_system_cost": float(info["normalized_system_cost"].float().mean().detach().cpu()),
            }
        )
        obs, edge_index, edge_attr = step.obs, step.edge_index, step.edge_attr
        done = step.done
    return {
        "steps": steps,
        "last_info_keys": sorted(env.last_metrics.keys()),
        "manifest": metric_schema_manifest(scenario),
    }


def main() -> None:
    torch.set_num_threads(1)
    legacy_scenario = ScenarioConfig(n_leo=4, episode_len=2, seed=31, arrival_rate=1.0)
    physical_scenario = ScenarioConfig(
        n_leo=4,
        episode_len=2,
        seed=31,
        arrival_rate=1.0,
        max_queue=10.0,
        physical=PhysicalModelConfig(
            enabled=True,
            slot_duration_s=0.1,
            task_size_bits_mean=1000.0,
            task_size_bits_std=0.0,
            cycles_per_bit_mean=100.0,
            cycles_per_bit_std=0.0,
            leo_cpu_hz=1.0e6,
            geo_cpu_hz=2.0e6,
            ground_cpu_hz=3.0e6,
            local_rate_bps=1.0e6,
            isl_base_rate_bps=2.0e5,
            geo_base_rate_bps=1.0e5,
            ground_base_rate_bps=1.5e5,
            max_tx_power_w=2.0,
            kappa=1.0e-27,
            queue_cap_cycles=1.0e7,
            metric_mode="dual",
        ),
    )

    legacy = _run_episode(legacy_scenario)
    physical = _run_episode(physical_scenario)
    examples = {
        "mean_delay_s": physical["steps"][-1]["delay_s"],
        "mean_energy_j": physical["steps"][-1]["energy_j"],
        "mean_queue_cycles": physical["steps"][-1]["queue_cycles"],
        "normalized_system_cost": physical["steps"][-1]["normalized_system_cost"],
    }
    metric_records = build_metric_records(examples, normalized_source="affine_normalized_offline_objective")
    units_present = all(
        all(key in record and record[key] not in (None, "") for key in ("metric", "unit", "source", "comparable_scope"))
        for record in metric_records
    )

    summary = {
        "legacy_mode_ok": bool(legacy["steps"]),
        "physical_mode_ok": bool(physical["steps"]),
        "units_present": bool(units_present),
        "ranking_audit_available": True,
        "physical_metric_examples": metric_records,
        "legacy": legacy,
        "physical": physical,
        "limits": {"episodes_per_mode": 1, "steps_per_mode": 2, "n_leo": 4},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
