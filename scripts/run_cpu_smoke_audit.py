from __future__ import annotations

import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reviewer_repair" / "audit_smoke"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def _load_script_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _assert_java_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if "class " not in text:
        raise AssertionError(f"unexpected Java file content: {path}")


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    torch.set_num_threads(1)

    imported_modules = [
        "trisatflow.envs.geo_leo_ground_env",
        "trisatflow.envs.action_masks",
        "trisatflow.envs.obs_builder",
        "trisatflow.agents.hierarchical_trainer",
        "trisatflow.agents.maddpg_lower",
        "trisatflow.baselines.registry",
        "trisatflow.baselines.static_policies",
        "trisatflow.baselines.heuristic_policies",
        "trisatflow.satedgesim_eval.action_mapper",
    ]
    for module_name in imported_modules:
        importlib.import_module(module_name)

    for script_name in (
        "replay_on_satedgesim.py",
        "replay_baseline_on_satedgesim.py",
        "summarize_satedgesim_replay.py",
    ):
        _load_script_module(PROJECT_ROOT / "scripts" / script_name)

    java_base = WORKSPACE_ROOT / "satedgeSimv2" / "SatEdgeSim" / "edu" / "weijunyong" / "satedgesim"
    java_files = [
        java_base / "server" / "RlAction.java",
        java_base / "server" / "ExecutionReceipt.java",
        java_base / "server" / "RlDecisionBridge.java",
        java_base / "TasksOrchestration" / "ExternalRLOrchestrator.java",
    ]
    for java_file in java_files:
        _assert_java_file(java_file)

    from trisatflow.baselines.registry import build_baseline_policy, extract_candidate_info
    from trisatflow.config import RewardWeights, ScenarioConfig
    from trisatflow.envs.geo_leo_ground_env import GeoLeoGroundEnv
    from trisatflow.models import upper_action_mask_from_obs

    scenario = ScenarioConfig(
        n_leo=4,
        n_geo=1,
        n_ground=1,
        episode_len=4,
        seed=123,
        arrival_rate=1.0,
        max_queue=12.0,
        node_feature_dim=12,
        topology_mode="analytic",
        enable_dynamic_skip_isl=False,
    )
    env = GeoLeoGroundEnv(scenario, RewardWeights(), device="cpu")
    obs, edge_index, edge_attr = env.reset()

    step_summaries = []
    for _ in range(4):
        mask = upper_action_mask_from_obs(obs)
        upper_action = mask.float().argmax(dim=-1).long()
        lower_action = torch.ones((scenario.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32)
        step = env.step(upper_action, lower_action)
        step_summaries.append(
            {
                "t": int(env.t),
                "done": bool(step.done),
                "upper_reward_mean": float(step.upper_reward.mean().detach().cpu()),
                "lower_reward_mean": float(step.lower_reward.mean().detach().cpu()),
                "mean_delay_s": float(step.info["physical_delay_s"].float().mean().detach().cpu()),
                "mean_energy_j": float(step.info["physical_energy_j"].float().mean().detach().cpu()),
                "mean_queue_tasks": float(step.info["physical_queue_length_tasks"].float().mean().detach().cpu()),
            }
        )
        obs, edge_index, edge_attr = step.obs, step.edge_index, step.edge_attr
        if step.done:
            break

    fake_state: Dict[str, Any] = {
        "abstractActionMask": [1, 1, 1, 0],
        "abstractActionMaskVisible": [1, 1, 1, 0],
        "candidateVms": [
            {
                "isFeasible": True,
                "abstractAction": 0,
                "vmId": 10,
                "vmIndex": 0,
                "estimatedTotalDelaySec": 0.30,
                "estimatedQueueLength": 2,
                "estimatedEnergyJ": 0.4,
                "mobilityRisk": 0.0,
            },
            {
                "isFeasible": True,
                "abstractAction": 1,
                "vmId": 11,
                "vmIndex": 1,
                "estimatedTotalDelaySec": 0.20,
                "estimatedQueueLength": 1,
                "estimatedEnergyJ": 0.5,
                "mobilityRisk": 0.1,
            },
            {
                "isFeasible": True,
                "abstractAction": 2,
                "vmId": 12,
                "vmIndex": 2,
                "estimatedTotalDelaySec": 0.80,
                "estimatedQueueLength": 0,
                "estimatedEnergyJ": 0.3,
                "mobilityRisk": 0.2,
            },
        ],
    }
    baseline = build_baseline_policy("min_delay_greedy")
    baseline_decision = baseline.select_action(
        obs=None,
        state=fake_state,
        mask=[1, 1, 1, 0],
        candidate_info=extract_candidate_info(fake_state),
        rng=random.Random(7),
    )

    summary = {
        "status": "ok",
        "device": "cpu",
        "limits": {
            "episodes": 0,
            "steps": len(step_summaries),
            "n_leo": scenario.n_leo,
            "max_decisions": 0,
        },
        "imported_modules": imported_modules,
        "imported_scripts": [
            "scripts/replay_on_satedgesim.py",
            "scripts/replay_baseline_on_satedgesim.py",
            "scripts/summarize_satedgesim_replay.py",
        ],
        "java_files_checked": [str(path) for path in java_files],
        "env": {
            "obs_shape": list(obs.shape),
            "edge_index_shape": list(edge_index.shape),
            "edge_attr_shape": list(edge_attr.shape),
            "step_summaries": step_summaries,
            "last_metric_keys": sorted(env.last_metrics.keys())[:50],
        },
        "baseline": {
            "name": baseline.name,
            "upper_action": int(baseline_decision["upper_action"]),
            "lower_action": list(baseline_decision["lower_action"]),
            "decision_info": baseline_decision["decision_info"],
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(SUMMARY_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
