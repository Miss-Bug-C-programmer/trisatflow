from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trisatflow.baselines.fair_wrappers import wrap_baseline_with_lower_allocator
from trisatflow.baselines.lower_allocators import build_lower_allocator, lower_allocator_metadata
from trisatflow.baselines.offline_adapter import OfflineBaselineAdapter, build_offline_baseline_policy
from trisatflow.baselines.registry import ACTION_NAMES
from trisatflow.config import TrainConfig, load_config
from trisatflow.envs import GeoLeoGroundEnv


DEFAULT_BASELINES = "geo_only,ground_only,random_visible"
ROOT_SUMMARY = REPO_ROOT / "outputs" / "reviewer_repair" / "lower_fairness" / "summary.json"


def _mean_info(info: Mapping[str, Any], key: str) -> float:
    value = info[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return float(value.float().mean().detach().cpu().item())


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(mean(values)), float(pstdev(values) if len(values) > 1 else 0.0)


def _json_list(values: list[float]) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _parse_values(text: str | None) -> list[float] | None:
    if not text:
        return None
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--neutral-values must contain exactly three comma-separated values")
    return values


def _load_train_config(path: str | None) -> TrainConfig:
    if path:
        return load_config(path)
    return TrainConfig()


def evaluate_one(
    *,
    cfg: TrainConfig,
    baseline_name: str,
    lower_allocator_name: str,
    checkpoint: str | None,
    neutral_values: list[float] | None,
    episodes: int,
    steps: int,
    device: str,
    seed: int,
    formal: bool = False,
) -> dict[str, Any]:
    cfg.scenario.n_leo = min(int(cfg.scenario.n_leo), 4)
    cfg.scenario.episode_len = min(int(steps), 8)
    cfg.scenario.seed = int(seed)
    allocator = build_lower_allocator(
        lower_allocator_name,
        checkpoint=checkpoint,
        neutral_values=neutral_values,
        formal=bool(formal),
        cfg=cfg,
    )
    policy = wrap_baseline_with_lower_allocator(build_offline_baseline_policy(baseline_name), allocator)
    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, torch.device(device))

    costs: list[float] = []
    delays: list[float] = []
    energies: list[float] = []
    violations: list[float] = []
    lower_rows: list[list[float]] = []
    action_counts = [0, 0, 0, 0]
    lower_allocator_fields: dict[str, Any] = lower_allocator_metadata(allocator)

    with torch.inference_mode():
        for episode in range(int(episodes)):
            env.cfg.seed = int(seed) + episode * 1009
            env.generator.manual_seed(env.cfg.seed)
            env.reset(rule_baseline_observation=True)
            adapter = OfflineBaselineAdapter(policy, rng=random.Random(env.cfg.seed))
            done = False
            step_count = 0
            while not done and step_count < int(steps):
                batch = adapter.select_actions(env)
                step = env.step(batch.upper_action, batch.lower_action, minimal_info=True)
                costs.append(_mean_info(step.info, "normalized_system_cost"))
                delays.append(_mean_info(step.info, "physical_delay_s"))
                energies.append(_mean_info(step.info, "physical_energy_j"))
                violations.append(_mean_info(step.info, "deadline_exceedance"))
                for action in batch.upper_action.detach().cpu().tolist():
                    action_counts[int(action)] += 1
                for row in batch.lower_action.detach().cpu().tolist():
                    lower_rows.append([float(v) for v in row])
                if batch.decision_info:
                    for key in (
                        "requested_allocator",
                        "effective_lower_allocator",
                        "lower_allocator_name",
                        "lower_allocator_mode",
                        "same_lower_available",
                        "same_lower_skip_reason",
                        "same_learned_lower_loaded",
                        "fallback_allocator",
                        "formal_claim_allowed",
                    ):
                        if key in batch.decision_info[0]:
                            lower_allocator_fields[key] = batch.decision_info[0][key]
                done = bool(step.done)
                step_count += 1

    lower_cols = list(zip(*lower_rows)) if lower_rows else [[], [], []]
    lower_mean = [mean(list(col)) if col else 0.0 for col in lower_cols]
    lower_std = [pstdev(list(col)) if len(col) > 1 else 0.0 for col in lower_cols]
    total_actions = float(max(1, sum(action_counts)))
    action_mix = {ACTION_NAMES[idx]: float(action_counts[idx] / total_actions) for idx in range(4)}
    cost_mean, cost_std = _stats(costs)
    delay_mean, delay_std = _stats(delays)
    energy_mean, energy_std = _stats(energies)
    violation_mean, violation_std = _stats(violations)
    return {
        "method": f"{baseline_name}+{lower_allocator_fields['lower_allocator_name']}",
        "upper_policy": baseline_name,
        "lower_allocator": lower_allocator_fields["lower_allocator_name"],
        "requested_allocator": str(lower_allocator_fields.get("requested_allocator", lower_allocator_name)),
        "effective_lower_allocator": str(lower_allocator_fields.get("effective_lower_allocator", lower_allocator_fields["lower_allocator_name"])),
        "lower_allocator_mode": lower_allocator_fields["lower_allocator_mode"],
        "same_lower_available": bool(lower_allocator_fields.get("same_lower_available", True)),
        "same_lower_skip_reason": str(lower_allocator_fields.get("same_lower_skip_reason", "")),
        "same_learned_lower_loaded": bool(lower_allocator_fields.get("same_learned_lower_loaded", True)),
        "fallback_allocator": lower_allocator_fields.get("fallback_allocator"),
        "formal_claim_allowed": bool(lower_allocator_fields.get("formal_claim_allowed", True)),
        "cost": cost_mean,
        "cost_std": cost_std,
        "delay": delay_mean,
        "delay_std": delay_std,
        "energy": energy_mean,
        "energy_std": energy_std,
        "violation": violation_mean,
        "violation_std": violation_std,
        "action_mix": action_mix,
        "lower_action_mean": lower_mean,
        "lower_action_std": lower_std,
        "lower_action_order": "cpu_share,bandwidth_share,tx_power_ratio",
        "episodes": int(episodes),
        "steps": int(steps),
        "device": str(device),
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    run_mode: str = "smoke",
    cfg: TrainConfig | None = None,
    num_training_seeds: int | None = None,
    num_eval_seeds: int | None = None,
    update_root: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    csv_fields = [
        "method",
        "upper_policy",
        "lower_allocator",
        "requested_allocator",
        "effective_lower_allocator",
        "lower_allocator_mode",
        "same_lower_available",
        "same_lower_skip_reason",
        "same_learned_lower_loaded",
        "fallback_allocator",
        "formal_claim_allowed",
        "cost",
        "delay",
        "energy",
        "violation",
        "action_mix",
        "lower_action_mean",
        "lower_action_std",
        "lower_action_order",
        "episodes",
        "steps",
        "device",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["action_mix"] = json.dumps(row["action_mix"], sort_keys=True, separators=(",", ":"))
            csv_row["lower_action_mean"] = _json_list(row["lower_action_mean"])
            csv_row["lower_action_std"] = _json_list(row["lower_action_std"])
            writer.writerow({key: csv_row.get(key, "") for key in csv_fields})
    formal_collector_allowed = all(bool(row.get("formal_claim_allowed", True)) for row in rows)
    import hashlib
    import subprocess

    cfg_sha = hashlib.sha256(repr(cfg).encode("utf-8")).hexdigest() if cfg is not None else hashlib.sha256(b"").hexdigest()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = "unknown_not_a_git_checkout"
    summary = {
        "rows": rows,
        "row_count": len(rows),
        "lower_allocators": sorted({str(row["lower_allocator"]) for row in rows}),
        "upper_policies": sorted({str(row["upper_policy"]) for row in rows}),
        "formal_collector_allowed": bool(formal_collector_allowed and run_mode == "formal"),
        "outputs_are_smoke_only": bool(run_mode != "formal"),
        "run_mode": run_mode,
        "allocator_mode": str(rows[0].get("requested_allocator", rows[0].get("lower_allocator"))) if rows else "",
        "formal_claim_allowed": bool(run_mode == "formal" and formal_collector_allowed),
        "num_training_seeds": int(num_training_seeds or 0),
        "num_eval_seeds": int(num_eval_seeds or 0),
        "config_sha256": cfg_sha,
        "git_commit": git_commit,
        "table_4b_ready_note": "Future full experiment should report rule upper x neutral/same_lower/optimized/oracle lower.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if update_root:
        _update_root_summary(rows)


def _update_root_summary(rows: list[dict[str, Any]]) -> None:
    ROOT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    if ROOT_SUMMARY.exists():
        try:
            payload = json.loads(ROOT_SUMMARY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    existing = list(payload.get("rows") or [])
    keys = {(row.get("upper_policy"), row.get("lower_allocator")) for row in rows}
    kept = [row for row in existing if (row.get("upper_policy"), row.get("lower_allocator")) not in keys]
    merged = kept + rows
    formal_collector_allowed = all(bool(row.get("formal_claim_allowed", True)) for row in merged)
    payload = {
        "rows": merged,
        "row_count": len(merged),
        "lower_allocators": sorted({str(row["lower_allocator"]) for row in merged}),
        "upper_policies": sorted({str(row["upper_policy"]) for row in merged}),
        "outputs_are_smoke_only": True,
        "formal_collector_allowed": formal_collector_allowed,
        "table_4b_ready_note": "Future full experiment should report rule upper x neutral/same_lower/optimized/oracle lower.",
    }
    ROOT_SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule baselines with fair lower allocator controls.")
    parser.add_argument("--config", default="", help="Optional TriSatFlow config YAML.")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--lower-allocator", default="neutral", choices=["neutral", "same_learned", "optimized_greedy", "oracle_grid"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neutral-values", default="", help="Allocator order: bandwidth_share,tx_power_ratio,cpu_share.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "reviewer_repair" / "lower_fairness" / "neutral"))
    args = parser.parse_args()

    if int(args.episodes) > 2 or int(args.steps) > 8:
        raise ValueError("CPU smoke guard: use episodes<=2 and steps<=8 for this reviewer repair script.")
    cfg = _load_train_config(args.config or None)
    baselines = [item.strip() for item in str(args.baselines).split(",") if item.strip()]
    neutral_values = _parse_values(args.neutral_values)
    rows = [
        evaluate_one(
            cfg=_load_train_config(args.config or None) if args.config else cfg,
            baseline_name=baseline,
            lower_allocator_name=args.lower_allocator,
            checkpoint=args.checkpoint or None,
            neutral_values=neutral_values,
            episodes=int(args.episodes),
            steps=int(args.steps),
            device=str(args.device),
            seed=int(args.seed),
        )
        for baseline in baselines
    ]
    _write_outputs(rows, Path(args.output_dir))
    print(json.dumps({"rows": rows, "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
