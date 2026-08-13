from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import csv
import json
import random
import torch

from trisatflow.config import TrainConfig
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import METRIC_SCHEMA_VERSION
from trisatflow.baselines.offline_adapter import (
    OfflineBaselineAdapter,
    build_offline_baseline_policy,
    offline_baseline_registry,
    ratio_fields,
    stats_delta,
)

ProgressCallback = Callable[[Mapping[str, object]], None]
EpisodeCallback = Callable[[Dict[str, float]], None]

# Keep this list explicit.  These are the metrics required by the paper-facing
# offline baseline table and by the statistical summary.
EVAL_METRIC_KEYS = [
    "normalized_system_cost",
    "mean_deadline_exceedance",
    "mean_deadline_violation_ratio",
    "mean_delay_s",
    "mean_energy_j",
    "mean_queue_length_tasks",
    "mean_delay",
    "mean_energy",
    "mean_queue",
    "mean_service",
    "mean_arrivals",
    "mean_system_cost",
    "mean_deadline_violation",
    "mean_feasibility",
    "mean_lyapunov_drift",
    "mean_virtual_delay_queue",
    "upper_local_ratio",
    "upper_neighbor_ratio",
    "upper_geo_ratio",
    "upper_ground_ratio",
    "requested_local_ratio",
    "requested_neighbor_ratio",
    "requested_geo_ratio",
    "requested_ground_ratio",
    "fallback_used_ratio",
    "invalid_attempt_ratio",
]

_REQUIRED_INFO_KEYS = {
    "normalized_system_cost": "normalized_system_cost",
    "mean_deadline_exceedance": "deadline_exceedance",
    "mean_deadline_violation_ratio": "deadline_violation_flag",
    "mean_delay_s": "physical_delay_s",
    "mean_energy_j": "physical_energy_j",
    "mean_queue_length_tasks": "physical_queue_length_tasks",
    "mean_delay": "delay",
    "mean_energy": "energy",
    "mean_queue": "queue",
    "mean_service": "service",
    "mean_arrivals": "arrivals",
    "mean_system_cost": "system_cost",
    "mean_deadline_violation": "deadline_violation",
    "mean_feasibility": "feasible",
    "mean_lyapunov_drift": "lyapunov_drift",
    "mean_virtual_delay_queue": "virtual_delay_queue",
}


def _seed_everything(seed: int, env: GeoLeoGroundEnv) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    env.cfg.seed = int(seed)
    env.generator.manual_seed(int(seed))


def _mean_key(info: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    value = info[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return value.float().mean()


def _to_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def evaluate_policy(
    policy,
    cfg: TrainConfig,
    episodes: int,
    seed: int,
    device: str = "cpu",
    *,
    env: GeoLeoGroundEnv | None = None,
    progress_every: int = 25,
    progress_callback: ProgressCallback | None = None,
    episode_callback: EpisodeCallback | None = None,
    baseline_name: str = "unknown",
) -> List[Dict[str, float]]:
    """Evaluate one rule policy with deterministic seed control and streaming metrics.

    The previous implementation retained every step-info dictionary until the end
    of an episode and did not emit progress.  It also instantiated a new trace
    environment for every baseline/seed pair.  This version reuses an optional
    environment, aggregates metrics on-device, and emits episode-level rows as
    soon as they are complete.
    """

    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be >= 0, got {progress_every}")

    run_cfg = deepcopy(cfg)
    run_cfg.scenario.seed = int(seed)
    run_cfg.device = str(device)
    if str(device).startswith("cpu"):
        # Tiny tensor reductions are markedly slower with a large CPU thread
        # pool.  One thread is the stable default for this simulator.
        torch.set_num_threads(1)

    if env is None:
        env = GeoLeoGroundEnv(run_cfg.scenario, run_cfg.reward, torch.device(device))
    _seed_everything(seed, env)

    rows: List[Dict[str, float]] = []
    started = perf_counter()
    completed_steps = 0

    with torch.inference_mode():
        for episode in range(1, episodes + 1):
            obs, _edge_index, _edge_attr = env.reset(rule_baseline_observation=True)
            adapter = OfflineBaselineAdapter(policy, rng=random.Random(int(seed) + episode * 1009))
            stats_before = adapter.stats.snapshot()
            scalar_sums: Dict[str, torch.Tensor] = {
                out_key: torch.zeros((), dtype=torch.float32, device=env.device)
                for out_key in _REQUIRED_INFO_KEYS
            }
            action_hist = torch.zeros(GeoLeoGroundEnv.N_UPPER_ACTIONS, dtype=torch.float32, device=env.device)
            step_count = 0
            done = False
            while not done:
                batch = adapter.select_actions(env)
                step = env.step(batch.upper_action, batch.lower_action, minimal_info=True)
                for out_key, info_key in _REQUIRED_INFO_KEYS.items():
                    scalar_sums[out_key] += _mean_key(step.info, info_key)
                actions = step.info["upper_action"].view(-1).long()
                action_hist += torch.bincount(actions, minlength=GeoLeoGroundEnv.N_UPPER_ACTIONS).float()
                step_count += 1
                completed_steps += 1
                obs, done = step.obs, step.done

            denom = float(max(1, step_count))
            hist = action_hist / action_hist.sum().clamp_min(1.0)
            decision_delta = stats_delta(adapter.stats.snapshot(), stats_before)
            decision_count = float(max(1, int(decision_delta["decision_count"])))
            row: Dict[str, float] = {
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "episode": float(episode),
                "seed": float(seed),
                **{key: _to_float(value / denom) for key, value in scalar_sums.items()},
                "upper_local_ratio": _to_float(hist[0]),
                "upper_neighbor_ratio": _to_float(hist[1]),
                "upper_geo_ratio": _to_float(hist[2]),
                "upper_ground_ratio": _to_float(hist[3]),
                **ratio_fields("requested", decision_delta["requested_counts"]),
                **ratio_fields("selected", decision_delta["selected_counts"]),
                "fallback_used_count": float(decision_delta["fallback_count"]),
                "fallback_used_ratio": float(decision_delta["fallback_count"]) / decision_count,
                "invalid_attempt_count": float(decision_delta["invalid_attempt_count"]),
                "invalid_attempt_ratio": float(decision_delta["invalid_attempt_count"]) / decision_count,
                "decision_count": float(decision_delta["decision_count"]),
            }
            rows.append(row)
            if episode_callback is not None:
                episode_callback(dict(row))

            if progress_callback is not None and (
                episode == 1 or episode == episodes or (progress_every > 0 and episode % progress_every == 0)
            ):
                elapsed = max(1.0e-9, perf_counter() - started)
                progress_callback(
                    {
                        "event": "progress",
                        "baseline": baseline_name,
                        "seed": int(seed),
                        "episode": int(episode),
                        "episodes": int(episodes),
                        "completed_steps": int(completed_steps),
                        "elapsed_s": elapsed,
                        "steps_per_s": completed_steps / elapsed,
                    }
                )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, float]], metric_keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        if not vals:
            continue
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out[f"{key}_ci95"] = 1.96 * out[f"{key}_std"] / (len(vals) ** 0.5) if vals else 0.0
    return out


def _seed_mean_row(rows: Sequence[Mapping[str, float]], *, seed: int) -> Dict[str, float]:
    out: Dict[str, float] = {"seed": float(seed)}
    for key in EVAL_METRIC_KEYS:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = mean(vals)
    return out


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically rewrite a CSV so interrupted runs never leave a torn file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def evaluate_named_baselines(
    cfg: TrainConfig,
    names: List[str],
    seeds: List[int],
    episodes: int,
    device: str = "cpu",
    *,
    output_dir: str | Path | None = None,
    progress_every: int = 25,
    checkpoint_every: int = 25,
    progress_callback: ProgressCallback | None = None,
):
    """Evaluate named baselines and persist partial results incrementally.

    Statistical summaries use per-seed episode means as independent samples.
    Pooling all episodes as independent observations, as the old implementation
    did, artificially narrows confidence intervals (pseudo-replication).
    """

    if not names:
        raise ValueError("at least one baseline is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")
    if checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {checkpoint_every}")

    registry = offline_baseline_registry(include_legacy_aliases=True)
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(registry)}")

    run_cfg = deepcopy(cfg)
    run_cfg.device = str(device)
    if str(device).startswith("cpu"):
        torch.set_num_threads(1)
    shared_env = GeoLeoGroundEnv(run_cfg.scenario, run_cfg.reward, torch.device(device))

    output_path = Path(output_dir) if output_dir is not None else None
    episode_csv = output_path / "baseline_episode_metrics.csv" if output_path is not None else None
    summary_csv = output_path / "baseline_summary.csv" if output_path is not None else None
    status_json = output_path / "baseline_eval_status.json" if output_path is not None else None

    episode_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    total_units = len(names) * len(seeds)
    total_episodes = total_units * episodes
    total_steps = total_episodes * int(run_cfg.scenario.episode_len)
    global_started = perf_counter()
    global_completed_episodes = 0

    def persist(status: str, **extra: object) -> None:
        if episode_csv is not None:
            write_csv(episode_csv, episode_rows)
        if summary_csv is not None:
            write_csv(summary_csv, summary_rows)
        if status_json is not None:
            write_json(
                status_json,
                {
                    "status": status,
                    "device": str(device),
                    "baselines": names,
                    "seeds": seeds,
                    "episodes_per_seed": episodes,
                    "steps_per_episode": int(run_cfg.scenario.episode_len),
                    "completed_episode_rows": len(episode_rows),
                    "total_episode_rows": total_episodes,
                    "elapsed_s": perf_counter() - global_started,
                    **extra,
                },
            )

    if progress_callback is not None:
        progress_callback(
            {
                "event": "start",
                "device": str(device),
                "baselines": len(names),
                "seeds": len(seeds),
                "episodes_per_seed": episodes,
                "steps_per_episode": int(run_cfg.scenario.episode_len),
                "total_episodes": total_episodes,
                "total_steps": total_steps,
            }
        )
    persist("running")

    try:
        for baseline_index, name in enumerate(names, start=1):
            seed_mean_rows: List[Dict[str, float]] = []
            for seed_index, seed in enumerate(seeds, start=1):
                current_seed_rows: List[Dict[str, float]] = []

                def on_episode(row: Dict[str, float]) -> None:
                    nonlocal global_completed_episodes
                    named_row: Dict[str, object] = {"baseline": name, **row}
                    episode_rows.append(named_row)
                    current_seed_rows.append(dict(row))
                    global_completed_episodes += 1
                    if global_completed_episodes % checkpoint_every == 0:
                        persist(
                            "running",
                            current_baseline=name,
                            current_seed=int(seed),
                            current_episode=int(row["episode"]),
                        )

                def on_progress(payload: Mapping[str, object]) -> None:
                    if progress_callback is None:
                        return
                    elapsed = max(1.0e-9, perf_counter() - global_started)
                    overall_steps = global_completed_episodes * int(run_cfg.scenario.episode_len)
                    progress_callback(
                        {
                            **dict(payload),
                            "baseline_index": baseline_index,
                            "baseline_count": len(names),
                            "seed_index": seed_index,
                            "seed_count": len(seeds),
                            "global_completed_episodes": global_completed_episodes,
                            "global_total_episodes": total_episodes,
                            "global_progress_pct": 100.0 * global_completed_episodes / max(1, total_episodes),
                            "global_elapsed_s": elapsed,
                            "global_steps_per_s": overall_steps / elapsed,
                        }
                    )

                evaluate_policy(
                    build_offline_baseline_policy(name),
                    run_cfg,
                    episodes=episodes,
                    seed=seed,
                    device=device,
                    env=shared_env,
                    progress_every=progress_every,
                    progress_callback=on_progress,
                    episode_callback=on_episode,
                    baseline_name=name,
                )
                seed_mean_rows.append(_seed_mean_row(current_seed_rows, seed=seed))
                persist("running", completed_baseline=name, completed_seed=int(seed))

            summary_row: Dict[str, object] = {
                "baseline": name,
                "seeds": ",".join(str(s) for s in seeds),
                "n_seeds": len(seeds),
                "episodes_per_seed": episodes,
                "summary_unit": "seed_mean",
                **summarize_rows(seed_mean_rows, EVAL_METRIC_KEYS),
            }
            summary_rows.append(summary_row)
            persist("running", completed_baseline=name)
            if progress_callback is not None:
                progress_callback({"event": "baseline_complete", **summary_row})
    except BaseException as exc:
        persist("interrupted", error=f"{type(exc).__name__}: {exc}")
        raise

    persist("completed")
    return episode_rows, summary_rows
