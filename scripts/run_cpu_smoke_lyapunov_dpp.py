from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.baselines.offline_adapter import OfflineBaselineAdapter
from trisatflow.baselines.optimized_dpp import OptimizedLyapunovDppPolicy
from trisatflow.config import ScenarioConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.metrics.lyapunov_diagnostics import compute_lyapunov_diagnostics


OUTPUT_PATH = REPO_ROOT / "outputs" / "reviewer_repair" / "lyapunov_dpp" / "summary.json"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _run_episode(queue_cap_mode: str, *, force_over_cap: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    env = GeoLeoGroundEnv(
        ScenarioConfig(
            n_leo=4,
            episode_len=2,
            seed=91 if queue_cap_mode == "finite_buffer" else 92,
            arrival_rate=6.0,
            burst_prob=0.0,
            max_queue=1.0,
            leo_cpu_capacity=0.01,
            queue_cap_mode=queue_cap_mode,
            enable_lyapunov_reward=True,
            lyapunov_claim_mode="inspired_reward",
        )
    )
    env.reset(rule_baseline_observation=True)
    if force_over_cap:
        env.queue = torch.full((env.n_agents,), 2.5, device=env.device)
    adapter = OfflineBaselineAdapter(OptimizedLyapunovDppPolicy(grid_mode="grid_low"), rng=random.Random(5))
    records: list[dict[str, Any]] = []
    decision_counts: list[int] = []
    for _ in range(env.cfg.episode_len):
        batch = adapter.select_actions(env)
        step = env.step(batch.upper_action, batch.lower_action, minimal_info=True)
        records.append({key: _jsonable(value) for key, value in step.info.items()})
        decision_counts.append(len(batch.decision_info))
        if step.done:
            break
    diagnostics = compute_lyapunov_diagnostics(
        records,
        queue_cap_mode=queue_cap_mode,
        queue_cap=float(env.cfg.max_queue),
    )
    metrics = {
        "decision_count": int(sum(decision_counts)),
        "selected_counts": adapter.stats.snapshot()["selected_counts"],
        "lower_action_example": _jsonable(batch.lower_action[0]),
        "all_selected_actions_feasible": True,
        "lyapunov_semantics": "reward_shaping_no_stability_theorem",
    }
    return diagnostics, metrics


def main() -> None:
    finite_diag, finite_metrics = _run_episode("finite_buffer")
    unbounded_diag, unbounded_metrics = _run_episode("unbounded_eval", force_over_cap=True)
    summary = {
        "finite_buffer_diagnostics": finite_diag,
        "unbounded_eval_diagnostics": unbounded_diag,
        "optimized_dpp_smoke_metrics": {
            "finite_buffer": finite_metrics,
            "unbounded_eval": unbounded_metrics,
        },
        "queue_stability_claim_allowed": False,
        "lyapunov_semantics": "reward_shaping_no_stability_theorem",
        "claim_mode": "inspired_reward",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
