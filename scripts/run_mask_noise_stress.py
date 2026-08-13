from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv


def _mean_info(info: Mapping[str, Any], key: str) -> float:
    value = info.get(key, 0.0)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return float(value.float().mean().detach().cpu().item())


def _run_setting(mask_source: str, noise_level: float, *, episodes: int, steps: int, device: str) -> dict[str, Any]:
    scenario = ScenarioConfig(
        n_leo=4,
        episode_len=int(steps),
        seed=19,
        topology_mode="analytic",
        action_mask_layer_mode="full",
        action_mask_enabled=True,
        mask_source=mask_source,
        mask_prediction_horizon_s=8.0,
        link_lifetime_noise_std_s=float(noise_level),
        completion_time_noise_std_s=float(noise_level),
        mask_false_positive_rate=0.05 * float(noise_level),
        mask_false_negative_rate=0.05 * float(noise_level),
        mask_staleness_slots=1 if float(noise_level) > 0.0 else 0,
    )
    env = GeoLeoGroundEnv(scenario, device=device)
    sums = {
        "cost": 0.0,
        "violation": 0.0,
        "infeasible_action_attempts": 0.0,
        "mask_false_positive_rate_observed": 0.0,
        "mask_false_negative_rate_observed": 0.0,
        "mobility_failure_proxy": 0.0,
    }
    count = 0
    with torch.inference_mode():
        for episode in range(int(episodes)):
            env.cfg.seed = 19 + episode
            env.generator.manual_seed(env.cfg.seed)
            env.reset()
            done = False
            while not done and count < int(episodes) * int(steps):
                mask = env._upper_action_mask_at_step(env.t)
                upper = torch.argmax(mask.float(), dim=-1).long()
                lower = torch.ones((env.n_agents, env.LOWER_ACTION_DIM), dtype=torch.float32, device=env.device)
                step = env.step(upper, lower, minimal_info=True)
                sums["cost"] += _mean_info(step.info, "normalized_system_cost")
                sums["violation"] += _mean_info(step.info, "deadline_violation_flag")
                sums["infeasible_action_attempts"] += float((1.0 - step.info["feasible"].float()).sum().detach().cpu().item())
                sums["mask_false_positive_rate_observed"] += _mean_info(step.info, "mask_false_positive_rate_observed")
                sums["mask_false_negative_rate_observed"] += _mean_info(step.info, "mask_false_negative_rate_observed")
                sums["mobility_failure_proxy"] += _mean_info(step.info, "mobility_failure_risk")
                count += 1
                done = bool(step.done)
    denom = float(max(1, count))
    return {
        "mask_source": mask_source,
        "noise_level": float(noise_level),
        "cost": sums["cost"] / denom,
        "violation": sums["violation"] / denom,
        "infeasible_action_attempts": sums["infeasible_action_attempts"],
        "mask_false_positive_rate_observed": sums["mask_false_positive_rate_observed"] / denom,
        "mask_false_negative_rate_observed": sums["mask_false_negative_rate_observed"] / denom,
        "mobility_failure_proxy": sums["mobility_failure_proxy"] / denom,
        "uses_oracle_trace_mask": mask_source == "oracle_trace",
        "deployable": mask_source != "oracle_trace",
        "episodes": int(episodes),
        "steps": int(steps),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CPU stress test for mask source/noise semantics.")
    parser.add_argument("--mask-source", default="oracle_trace,predicted,measured")
    parser.add_argument("--noise-levels", default="0,0.5,1.0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "reviewer_repair" / "mask_noise"))
    args = parser.parse_args()
    if int(args.episodes) > 2 or int(args.steps) > 8:
        raise ValueError("CPU smoke guard: use episodes<=2 and steps<=8.")
    sources = [item.strip() for item in str(args.mask_source).split(",") if item.strip()]
    noise_levels = [float(item.strip()) for item in str(args.noise_levels).split(",") if item.strip()]
    rows = [
        _run_setting(source, noise, episodes=int(args.episodes), steps=int(args.steps), device=str(args.device))
        for source in sources
        for noise in noise_levels
    ]
    summary = {
        "rows": rows,
        "row_count": len(rows),
        "mask_sources": sorted({row["mask_source"] for row in rows}),
        "oracle_trace_deployable": False,
        "note": "oracle_trace masks are diagnostic upper bounds; predicted/measured masks are deployable stress settings.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
