from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import shlex
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.config import load_config, save_config
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.models import upper_action_mask_from_obs


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x in (None, "", "NA"):
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _mean_info(infos: List[Dict[str, torch.Tensor]], key: str) -> float:
    vals = [info[key].float().mean().detach().cpu() for info in infos if key in info]
    if not vals:
        return float("nan")
    return float(torch.stack(vals).mean())


def _clone_env_for_lookahead(env: GeoLeoGroundEnv) -> GeoLeoGroundEnv:
    clone = object.__new__(GeoLeoGroundEnv)
    clone.__dict__ = {}
    shared_heavy_keys = {
        "cfg",
        "weights",
        "device",
        "_trace_provider",
        "_obs_norm_stats",
        "_unit_scale",
        "_trace_delay_interpretation",
    }
    for key, value in env.__dict__.items():
        if key in shared_heavy_keys:
            clone.__dict__[key] = value
            continue
        if key == "generator":
            g = torch.Generator(device=env.device)
            g.set_state(env.generator.get_state())
            clone.__dict__[key] = g
        elif torch.is_tensor(value):
            clone.__dict__[key] = value.clone()
        elif isinstance(value, dict):
            clone.__dict__[key] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in value.items()}
        elif isinstance(value, list):
            clone.__dict__[key] = list(value)
        else:
            try:
                import copy

                clone.__dict__[key] = copy.deepcopy(value)
            except Exception:
                clone.__dict__[key] = value
    return clone


def _joint_actions_from_mask(mask: torch.Tensor) -> List[Sequence[int]]:
    choices: List[List[int]] = []
    for row in mask.bool().tolist():
        idxs = [i for i, b in enumerate(row) if b]
        if not idxs:
            idxs = [0]
        choices.append(idxs)
    return list(itertools.product(*choices))


def _joint_action_count(mask: torch.Tensor) -> int:
    count = 1
    for row in mask.bool().tolist():
        k = sum(1 for b in row if b)
        count *= max(1, int(k))
    return int(count)


def _select_oracle_action_one_step(
    env: GeoLeoGroundEnv,
    obs: torch.Tensor,
    *,
    max_joint_actions: int,
) -> tuple[torch.Tensor, Dict[str, float]]:
    mask = upper_action_mask_from_obs(obs)
    candidates = _joint_actions_from_mask(mask)
    joint_count = len(candidates)
    if joint_count > max_joint_actions:
        raise ValueError(
            f"joint action space too large for brute-force oracle: {joint_count} > {max_joint_actions}; "
            "reduce n_leo/steps or increase --max-joint-actions for debug."
        )

    n_agents = int(obs.shape[0])
    lower = torch.ones((n_agents, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32, device=env.device)

    best_score = float("inf")
    best_action = None
    for cand in candidates:
        upper = torch.tensor(cand, dtype=torch.long, device=env.device)
        env_look = _clone_env_for_lookahead(env)
        step_out = env_look.step(upper, lower)
        score = float(step_out.info["system_cost"].float().mean().detach().cpu())
        if score < best_score:
            best_score = score
            best_action = upper

    assert best_action is not None
    return best_action, {
        "joint_action_count": float(joint_count),
        "best_objective": float(best_score),
        "effective_method": "bruteforce_1step",
        "search_depth": 1.0,
        "expanded_nodes": float(joint_count),
        "beam_width": 1.0,
    }


def _select_oracle_action_beam(
    env: GeoLeoGroundEnv,
    obs: torch.Tensor,
    *,
    beam_width: int,
    horizon: int,
    max_joint_actions_per_node: int,
) -> tuple[torch.Tensor, Dict[str, float]]:
    n_agents = int(obs.shape[0])
    lower = torch.ones((n_agents, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32, device=env.device)
    horizon = max(1, int(horizon))
    beam_width = max(1, int(beam_width))

    root = _clone_env_for_lookahead(env)
    beam: List[tuple[float, GeoLeoGroundEnv, torch.Tensor | None]] = [(0.0, root, None)]
    expanded = 0
    worst_joint = 0
    depth_ran = 0
    for depth in range(horizon):
        depth_ran = depth + 1
        next_beam: List[tuple[float, GeoLeoGroundEnv, torch.Tensor | None]] = []
        for cumulative_cost, state_env, first_action in beam:
            state_obs, _ei, _ea = state_env._get_obs_graph()  # noqa: SLF001
            mask = upper_action_mask_from_obs(state_obs)
            candidates = _joint_actions_from_mask(mask)
            if len(candidates) > int(max_joint_actions_per_node):
                candidates = candidates[: int(max_joint_actions_per_node)]
            worst_joint = max(worst_joint, len(candidates))
            for cand in candidates:
                expanded += 1
                upper = torch.tensor(cand, dtype=torch.long, device=env.device)
                next_env = _clone_env_for_lookahead(state_env)
                out = next_env.step(upper, lower)
                step_cost = float(out.info["system_cost"].float().mean().detach().cpu())
                next_first = upper if first_action is None else first_action
                next_beam.append((cumulative_cost + step_cost, next_env, next_first))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    if not beam:
        fallback = torch.zeros((n_agents,), dtype=torch.long, device=env.device)
        return fallback, {
            "joint_action_count": 1.0,
            "best_objective": float("inf"),
            "effective_method": "beam_search",
            "search_depth": float(depth_ran),
            "expanded_nodes": float(expanded),
            "beam_width": float(beam_width),
        }

    best_cost, _best_env, first = min(beam, key=lambda x: x[0])
    assert first is not None
    return first, {
        "joint_action_count": float(max(1, worst_joint)),
        "best_objective": float(best_cost),
        "effective_method": "beam_search",
        "search_depth": float(depth_ran),
        "expanded_nodes": float(expanded),
        "beam_width": float(beam_width),
    }


def _select_oracle_action(
    env: GeoLeoGroundEnv,
    obs: torch.Tensor,
    *,
    method: str,
    max_joint_actions: int,
    beam_width: int,
    beam_horizon: int,
) -> tuple[torch.Tensor, Dict[str, float]]:
    mask = upper_action_mask_from_obs(obs)
    joint_count = _joint_action_count(mask)
    m = str(method).strip().lower()
    if m == "auto":
        if joint_count <= int(max_joint_actions):
            return _select_oracle_action_one_step(env, obs, max_joint_actions=max_joint_actions)
        return _select_oracle_action_beam(
            env,
            obs,
            beam_width=beam_width,
            horizon=beam_horizon,
            max_joint_actions_per_node=max_joint_actions,
        )
    if m == "bruteforce_1step":
        if joint_count <= int(max_joint_actions):
            return _select_oracle_action_one_step(env, obs, max_joint_actions=max_joint_actions)
        # Degrade to beam instead of hard-failing.
        return _select_oracle_action_beam(
            env,
            obs,
            beam_width=beam_width,
            horizon=beam_horizon,
            max_joint_actions_per_node=max_joint_actions,
        )
    if m == "beam_search":
        return _select_oracle_action_beam(
            env,
            obs,
            beam_width=beam_width,
            horizon=beam_horizon,
            max_joint_actions_per_node=max_joint_actions,
        )
    raise ValueError(f"Unsupported oracle method: {method!r}")


def _read_rl_reference_cost(path: Path, *, seed: int, upper: str, lower: str) -> float:
    if not path.exists():
        return float("nan")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if str(r.get("status", "ok")).strip().lower() in {"ok", "success", ""}]
    if not rows:
        return float("nan")

    matched = []
    for r in rows:
        try:
            row_seed = int(r.get("seed", ""))
        except Exception:
            continue
        if row_seed != int(seed):
            continue
        if str(r.get("upper_algo", "")).strip() not in {"", upper}:
            continue
        if str(r.get("lower_algo", "")).strip() not in {"", lower}:
            continue
        value = _to_float(r.get("final_mean_system_cost"))
        if math.isfinite(value):
            matched.append(value)
    if not matched:
        return float("nan")
    return float(sum(matched) / len(matched))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-scale privileged oracle upper bound (not deployable policy).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-leo", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--oracle-method", type=str, default="auto", choices=["auto", "bruteforce_1step", "beam_search"])
    parser.add_argument("--max-joint-actions", type=int, default=4096)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--beam-horizon", type=int, default=3)
    parser.add_argument("--allow-large-debug", action="store_true")
    parser.add_argument("--rl-summary-csv", type=str, default="")
    parser.add_argument("--rl-reference-cost", type=float, default=float("nan"))
    args = parser.parse_args()

    small_scale_recommended = bool(args.n_leo <= 4 and args.steps <= 16)
    if (not small_scale_recommended) and not bool(args.allow_large_debug):
        print(
            "[WARN] oracle upper bound is outside recommended small scale (n_leo<=4, steps<=16). "
            "Proceeding with adaptive search; use --allow-large-debug to silence this warning."
        )

    cfg = load_config(args.config)
    cfg.scenario.n_leo = int(args.n_leo)
    cfg.scenario.episode_len = int(args.steps)
    cfg.steps_per_episode = int(args.steps)
    cfg.scenario.seed = int(args.seed)
    cfg.device = str(args.device)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_root / "oracle_resolved_config.yaml")

    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, device=args.device)

    step_rows: List[Dict[str, Any]] = []
    infos_all: List[Dict[str, torch.Tensor]] = []
    action_counts = torch.zeros((cfg.scenario.n_leo, GeoLeoGroundEnv.N_UPPER_ACTIONS), dtype=torch.float32)
    lower = torch.ones((cfg.scenario.n_leo, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32, device=env.device)

    for ep in range(1, int(args.episodes) + 1):
        obs, edge_index, edge_attr = env.reset()
        done = False
        while not done:
            t0 = time.time()
            upper, search_diag = _select_oracle_action(
                env,
                obs,
                method=str(args.oracle_method),
                max_joint_actions=int(args.max_joint_actions),
                beam_width=int(args.beam_width),
                beam_horizon=int(args.beam_horizon),
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            step = env.step(upper, lower)
            infos_all.append(step.info)
            chosen = upper.detach().cpu().long().view(-1)
            action_counts.scatter_add_(1, chosen.view(-1, 1), torch.ones((cfg.scenario.n_leo, 1), dtype=torch.float32))
            step_rows.append(
                {
                    "episode": ep,
                    "step": int(env.t - 1),
                    "oracle_method": args.oracle_method,
                    "oracle_effective_method": str(search_diag.get("effective_method", args.oracle_method)),
                    "uses_privileged_info": True,
                    "joint_action_count": int(search_diag["joint_action_count"]),
                    "search_depth": int(search_diag.get("search_depth", 1.0)),
                    "expanded_nodes": int(search_diag.get("expanded_nodes", 0.0)),
                    "search_time_ms": float(elapsed_ms),
                    "best_objective_system_cost": float(search_diag["best_objective"]),
                    "selected_mean_system_cost": float(step.info["system_cost"].float().mean().detach().cpu()),
                    "selected_mean_delay": float(step.info["delay"].float().mean().detach().cpu()),
                    "selected_mean_energy": float(step.info["energy"].float().mean().detach().cpu()),
                    "selected_mean_feasibility": float(step.info["feasible"].float().mean().detach().cpu()),
                }
            )
            obs, edge_index, edge_attr, done = step.obs, step.edge_index, step.edge_attr, step.done

    oracle_cost = _mean_info(infos_all, "system_cost")
    oracle_delay = _mean_info(infos_all, "delay")
    oracle_energy = _mean_info(infos_all, "energy")
    oracle_feasibility = _mean_info(infos_all, "feasible")

    rl_cost = _to_float(args.rl_reference_cost)
    if not math.isfinite(rl_cost):
        rl_path = Path(args.rl_summary_csv) if str(args.rl_summary_csv).strip() else (out_root.parent / "sweep_summary.csv")
        rl_cost = _read_rl_reference_cost(
            rl_path,
            seed=int(args.seed),
            upper=str(cfg.algo.upper_algo),
            lower=str(cfg.algo.lower_algo),
        )
    if math.isfinite(rl_cost):
        rl_gap_pct = float((rl_cost - oracle_cost) / max(abs(oracle_cost), 1.0e-6) * 100.0)
        rl_gap_status = "ok"
    else:
        rl_gap_pct = float("nan")
        rl_gap_status = "missing_rl_reference"

    total_actions = action_counts.sum().clamp_min(1.0)
    action_ratio = (action_counts.sum(dim=0) / total_actions).tolist()
    effective_method = str(step_rows[-1].get("oracle_effective_method", args.oracle_method)) if step_rows else str(args.oracle_method)
    baseline_name = f"oracle_upper_bound_{effective_method}"

    summary_row = {
        "status": "ok",
        "phase": "test",
        "seed": int(args.seed),
        "baseline": baseline_name,
        "upper_algo": "oracle",
        "lower_algo": "oracle",
        "observation_ablation": "",
        # Canonical aggregation field.  Keep final_mean_system_cost below as
        # a backward-compatible export for existing oracle consumers.
        "final_normalized_system_cost": float(oracle_cost),
        "final_mean_system_cost": float(oracle_cost),
        "final_mean_delay": float(oracle_delay),
        "final_mean_energy": float(oracle_energy),
        "final_mean_feasibility": float(oracle_feasibility),
        "oracle_cost": float(oracle_cost),
        "oracle_delay": float(oracle_delay),
        "oracle_energy": float(oracle_energy),
        "oracle_feasibility": float(oracle_feasibility),
        "oracle_method": str(args.oracle_method),
        "oracle_effective_method": effective_method,
        "uses_privileged_info": True,
        "deployable_policy": False,
        "n_leo": int(args.n_leo),
        "steps": int(args.steps),
        "episodes": int(args.episodes),
        "rl_reference_cost": float(rl_cost) if math.isfinite(rl_cost) else "",
        "rl_gap_percentage": float(rl_gap_pct) if math.isfinite(rl_gap_pct) else "",
        "rl_gap_status": rl_gap_status,
        "mean_joint_action_count": float(sum(r["joint_action_count"] for r in step_rows) / max(1, len(step_rows))),
        "mean_search_depth": float(sum(float(r.get("search_depth", 1.0)) for r in step_rows) / max(1, len(step_rows))),
        "mean_expanded_nodes": float(sum(float(r.get("expanded_nodes", 0.0)) for r in step_rows) / max(1, len(step_rows))),
        "mean_search_time_ms": float(sum(r["search_time_ms"] for r in step_rows) / max(1, len(step_rows))),
        "upper_local_ratio": float(action_ratio[0]),
        "upper_neighbor_ratio": float(action_ratio[1]),
        "upper_geo_ratio": float(action_ratio[2]),
        "upper_ground_ratio": float(action_ratio[3]),
        "output_dir": str(out_root),
        "metrics_csv": str(out_root / "oracle_episode_metrics.csv"),
        "checkpoint": "",
    }

    episode_rows = [
        {
            "episode": 1,
            "oracle_method": args.oracle_method,
            "oracle_effective_method": effective_method,
            "uses_privileged_info": True,
            "mean_system_cost": float(oracle_cost),
            "mean_delay": float(oracle_delay),
            "mean_energy": float(oracle_energy),
            "mean_feasibility": float(oracle_feasibility),
            "rl_reference_cost": float(rl_cost) if math.isfinite(rl_cost) else "",
            "rl_gap_percentage": float(rl_gap_pct) if math.isfinite(rl_gap_pct) else "",
            "rl_gap_status": rl_gap_status,
        }
    ]

    _write_csv(out_root / "oracle_step_metrics.csv", step_rows)
    _write_csv(out_root / "oracle_episode_metrics.csv", episode_rows)
    _write_csv(out_root / "oracle_summary.csv", [summary_row])

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", "")),
        "uses_privileged_info": True,
        "deployable_policy": False,
        "oracle_small_scale_only": True,
        "small_scale_recommended": small_scale_recommended,
        "oracle_method": args.oracle_method,
        "complexity_limit": {
            "recommended_max_n_leo": 4,
            "recommended_max_steps": 16,
            "max_joint_actions_per_node": int(args.max_joint_actions),
            "beam_width": int(args.beam_width),
            "beam_horizon": int(args.beam_horizon),
        },
        "resolved_config": asdict(cfg),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "ORACLE_UPPER_BOUND_OK "
        f"method={args.oracle_method} n_leo={args.n_leo} steps={args.steps} seed={args.seed} "
        f"effective_method={effective_method} "
        f"oracle_cost={oracle_cost:.6f} oracle_delay={oracle_delay:.6f} oracle_energy={oracle_energy:.6f} "
        f"oracle_feasibility={oracle_feasibility:.6f} rl_gap_pct={rl_gap_pct if math.isfinite(rl_gap_pct) else 'NA'} "
        f"summary_csv={out_root / 'oracle_summary.csv'}"
    )


if __name__ == "__main__":
    main()
