from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.experiment_profiles import get_profile, profile_metadata
from trisatflow.baselines.registry import baseline_metadata, baseline_metadata_json
from trisatflow.config_validation import canonicalize_train_config_path, validate_experiment_matrix_config, validate_wrapper_payload


def _run(cmd: List[str], dry_run: bool = False, env: Dict[str, str] | None = None) -> int:
    print("CMD", " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_config(path: str) -> Dict[str, Any]:
    resolved_path, deprecation_msg = canonicalize_train_config_path(path)
    if deprecation_msg:
        print(f"[DEPRECATED] {deprecation_msg}")
    payload = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8")) or {}
    wrapped, canonical = validate_wrapper_payload(payload)
    if wrapped:
        print(f"[DEPRECATED] config wrapper '{path}' -> '{canonical}'")
        return _load_config(canonical)
    validate_experiment_matrix_config(payload, source=str(resolved_path))
    return payload


def _first_profile_for_smoke(cfg: Dict[str, Any]) -> str:
    profiles = [str(x) for x in (cfg.get("profiles") or [])]
    for p in ("mobility_aware_main_v1", "mobility_aware_main"):
        if p in profiles:
            return p
    return profiles[0] if profiles else "mobility_aware_main_v1"


def _smoke_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["profiles"] = [_first_profile_for_smoke(out)]
    out["architectures"] = ["full"]
    out["baselines"] = ["random_visible", "cost_greedy", "tri_mappo_maddpg"]
    out["seeds"] = [13]
    out["max_decisions"] = 500
    out.setdefault("episodes", {})
    out["episodes"]["preflight"] = 2
    out["steps"] = 8
    return out


def _resolve_matrix_baselines(
    requested: List[str],
    *,
    allow_placeholder_baselines: bool,
    allow_non_paper_ready_baselines: bool,
) -> Dict[str, Any]:
    selected: List[str] = []
    blocked_placeholders: List[str] = []
    skipped_non_paper_ready: List[str] = []
    metadata_rows: Dict[str, Dict[str, Any]] = {}

    for raw_name in requested:
        name = str(raw_name).strip().lower()
        meta = baseline_metadata(name)
        meta_row = meta.to_dict()
        metadata_rows[name] = meta_row

        if bool(meta.paper_ready):
            selected.append(name)
            continue
        if meta.type == "placeholder":
            if allow_placeholder_baselines:
                selected.append(name)
            else:
                blocked_placeholders.append(name)
            continue
        if allow_non_paper_ready_baselines:
            selected.append(name)
        else:
            skipped_non_paper_ready.append(name)

    return {
        "selected": selected,
        "blocked_placeholders": sorted(set(blocked_placeholders)),
        "skipped_non_paper_ready": sorted(set(skipped_non_paper_ready)),
        "requested_metadata": metadata_rows,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_int(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_max_decisions(cfg: Dict[str, Any]) -> int:
    raw = cfg.get("max_decisions", 500)
    if isinstance(raw, list):
        if not raw:
            return 500
        return _as_int(raw[0], 500)
    return _as_int(raw, 500)


def _tri_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve TriSatFlow policy/checkpoint settings without using stale global checkpoints.

    v1 matrix runs are seed-safe by default: every replay seed maps to its own
    checkpoint under ``<output_root>/checkpoints/seed_<seed>/...``. A missing
    checkpoint is trained on demand before replay, unless auto training is
    explicitly disabled in the matrix YAML.
    """
    tri = dict(cfg.get("tri_mappo_maddpg") or {})
    tri.setdefault("upper", "mappo")
    tri.setdefault("lower", "maddpg")
    tri.setdefault("eval_mode", "raw_argmax")
    tri.setdefault("train_config", "trisatflow/configs/satedgesim_trace_mixed_v2_peragent_joint_logq_best.yaml")
    tri.setdefault("checkpoint_path_template", "{output_root}/checkpoints/seed_{seed}/upper_{upper}__lower_{lower}/checkpoint.pt")
    tri.setdefault("auto_train_if_missing", True)
    return tri


def _bool_cfg(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _format_checkpoint_path(template: str, *, output_root: Path, seed: int, upper: str, lower: str) -> Path:
    rendered = str(template).format(
        output_root=str(output_root),
        output_root_posix=output_root.as_posix(),
        seed=int(seed),
        upper=str(upper),
        lower=str(lower),
    )
    return Path(rendered)


def _expected_tri_checkpoint(cfg: Dict[str, Any], output_root: Path, seed: int) -> Path:
    tri = _tri_cfg(cfg)
    upper = str(tri.get("upper", "mappo"))
    lower = str(tri.get("lower", "maddpg"))
    explicit = str(tri.get("checkpoint_path", "")).strip()
    if explicit:
        # Explicit checkpoint paths are allowed only when they are seed templated.
        # This prevents a seed_13 checkpoint from silently contaminating seed_17/23 replay.
        if "{seed}" not in explicit and f"seed_{seed}" not in explicit:
            raise ValueError(
                "tri_mappo_maddpg.checkpoint_path must include {seed} or the matching "
                f"seed_{seed} segment; got {explicit!r}"
            )
        return _format_checkpoint_path(explicit, output_root=output_root, seed=seed, upper=upper, lower=lower)
    template = str(tri.get("checkpoint_path_template"))
    return _format_checkpoint_path(template, output_root=output_root, seed=seed, upper=upper, lower=lower)


def _ensure_tri_checkpoint(
    *,
    cfg: Dict[str, Any],
    output_root: Path,
    seed: int,
    device: str,
    dry_run: bool,
    env: Dict[str, str],
) -> Dict[str, Any]:
    tri = _tri_cfg(cfg)
    upper = str(tri.get("upper", "mappo"))
    lower = str(tri.get("lower", "maddpg"))
    ckpt = _expected_tri_checkpoint(cfg, output_root, seed)
    if ckpt.exists():
        return {
            "status": "existing",
            "checkpoint_path": str(ckpt),
            "train_output_root": str(ckpt.parents[2]) if len(ckpt.parents) >= 3 else str(output_root / "checkpoints"),
        }

    if not _bool_cfg(tri.get("auto_train_if_missing"), True):
        return {
            "status": "missing_auto_train_disabled",
            "checkpoint_path": str(ckpt),
            "train_output_root": str(ckpt.parents[2]) if len(ckpt.parents) >= 3 else str(output_root / "checkpoints"),
        }

    train_output_root = ckpt.parents[2] if len(ckpt.parents) >= 3 else output_root / "checkpoints"
    train_config = str(tri.get("train_config"))
    train_device = str(tri.get("train_device", device))
    if train_device.lower() == "auto":
        train_device = str(device)
    episodes_cfg = cfg.get("episodes") or {}
    train_episodes = int(tri.get("train_episodes", episodes_cfg.get("formal", cfg.get("train_episodes", 300))))
    train_steps = int(tri.get("train_steps", cfg.get("steps", 128)))
    n_leo = int(tri.get("n_leo", cfg.get("n_leo", 16)))

    train_cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--config", train_config,
        "--upper", upper,
        "--lower", lower,
        "--episodes", str(train_episodes),
        "--steps", str(train_steps),
        "--n-leo", str(n_leo),
        "--seeds", str(seed),
        "--device", train_device,
        "--output-root", str(train_output_root),
    ]
    rc = _run(train_cmd, dry_run=dry_run, env=env)
    if dry_run:
        return {
            "status": "dry_run_expected",
            "checkpoint_path": str(ckpt),
            "train_output_root": str(train_output_root),
            "train_cmd": train_cmd,
        }
    if rc != 0:
        return {
            "status": "training_failed",
            "checkpoint_path": str(ckpt),
            "train_output_root": str(train_output_root),
            "train_cmd": train_cmd,
            "return_code": int(rc),
        }
    if ckpt.exists():
        return {
            "status": "trained",
            "checkpoint_path": str(ckpt),
            "train_output_root": str(train_output_root),
            "train_cmd": train_cmd,
        }
    return {
        "status": "training_completed_checkpoint_missing",
        "checkpoint_path": str(ckpt),
        "train_output_root": str(train_output_root),
        "train_cmd": train_cmd,
    }


def _run_one(
    *,
    profile_name: str,
    architecture: str,
    baseline: str,
    seed: int,
    cfg: Dict[str, Any],
    output_root: Path,
    device: str,
    dry_run: bool,
    allow_placeholder_baselines: bool,
    allow_non_paper_ready_baselines: bool,
) -> Dict[str, Any]:
    profile = get_profile(profile_name)
    baseline_meta = baseline_metadata(baseline).to_dict()
    run_dir = output_root / f"profile_{profile_name}" / f"arch_{architecture}" / f"baseline_{baseline}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = run_dir / "replay"

    scenario_profile = str(cfg.get("scenario_profile", "mixed_cost_landscape_v2"))
    task_source_mode = str(cfg.get("task_source_mode", "round_robin_leo"))
    max_decisions = _resolve_max_decisions(cfg)

    env = dict(os.environ)
    if str(device).lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    tri_cfg = _tri_cfg(cfg)
    tri_eval_mode = str(tri_cfg.get("eval_mode", "raw_argmax")).strip()
    tri_device = str(tri_cfg.get("device", device)).strip()
    if tri_device.lower() == "auto":
        tri_device = str(device)
    tri_checkpoint = ""
    tri_checkpoint_info: Dict[str, Any] = {}

    if baseline == "tri_mappo_maddpg":
        try:
            tri_checkpoint_info = _ensure_tri_checkpoint(
                cfg=cfg,
                output_root=output_root,
                seed=seed,
                device=device,
                dry_run=dry_run,
                env=env,
            )
        except ValueError as exc:
            missing = {
                "status": "invalid_checkpoint_configuration",
                "baseline": baseline,
                "tri_mappo_maddpg": tri_cfg,
                "message": str(exc),
            }
            _write_json(run_dir / "missing_checkpoint.json", missing)
            return {"status": "missing_checkpoint", "reason": "invalid_checkpoint_configuration", "run_dir": str(run_dir)}
        tri_checkpoint = str(tri_checkpoint_info.get("checkpoint_path", "")).strip()
        if str(tri_checkpoint_info.get("status", "")).startswith("missing") or str(tri_checkpoint_info.get("status", "")).startswith("training_failed") or str(tri_checkpoint_info.get("status", "")) == "training_completed_checkpoint_missing":
            _write_json(run_dir / "missing_checkpoint.json", {
                "baseline": baseline,
                "seed": int(seed),
                "tri_mappo_maddpg": tri_cfg,
                "checkpoint_resolution": tri_checkpoint_info,
            })
            return {"status": "missing_checkpoint", "reason": str(tri_checkpoint_info.get("status", "missing_checkpoint")), "run_dir": str(run_dir)}
        cmd = [
            sys.executable,
            "scripts/replay_on_satedgesim.py",
            "--base-url", str(cfg.get("base_url", "http://127.0.0.1:8088")),
            "--checkpoint", tri_checkpoint,
            "--device", tri_device,
            "--seed", str(seed),
            "--max-decisions", str(max_decisions),
            "--eval-mode", tri_eval_mode,
            "--tie-eps", str(cfg.get("tie_eps", 0.05)),
            "--scenario-profile", scenario_profile,
            "--task-source-mode", task_source_mode,
            "--success-profile", profile.success_profile,
            "--action-mask-mode", profile.action_mask_mode,
            "--min-link-survival-margin-sec", str(profile.min_link_survival_margin_sec),
            "--architecture", architecture,
            "--profile-name", profile_name,
            "--output-dir", str(replay_dir),
        ]
        rc = _run(cmd, dry_run=dry_run, env=env)
        if rc != 0:
            return {"status": "failed", "reason": "tri_replay_failed", "run_dir": str(run_dir)}
    elif baseline == "hmadrl_maddqn_ddpg":
        _write_json(
            run_dir / "TODO_hmadrl_training_integration.json",
            {
                "status": "todo",
                "baseline": baseline,
                "message": "HMADRL baseline replay facade is available; full MADDQN+DDPG training loop integration is pending.",
            },
        )
        cmd = [
            sys.executable,
            "scripts/replay_baseline_on_satedgesim.py",
            "--baseline", baseline,
            "--architecture", architecture,
            "--profile", profile_name,
            "--min-link-survival-margin-sec", str(profile.min_link_survival_margin_sec),
            "--base-url", str(cfg.get("base_url", "http://127.0.0.1:8088")),
            "--scenario-profile", scenario_profile,
            "--task-source-mode", task_source_mode,
            "--seed", str(seed),
            "--max-decisions", str(max_decisions),
            "--output-dir", str(replay_dir),
        ]
        if allow_placeholder_baselines:
            cmd.append("--allow-placeholder-baselines")
        if allow_non_paper_ready_baselines:
            cmd.append("--allow-non-paper-ready-baselines")
        rc = _run(cmd, dry_run=dry_run, env=env)
        if rc != 0:
            return {"status": "failed", "reason": "hmadrl_baseline_replay_failed", "run_dir": str(run_dir)}
    else:
        cmd = [
            sys.executable,
            "scripts/replay_baseline_on_satedgesim.py",
            "--baseline", baseline,
            "--architecture", architecture,
            "--profile", profile_name,
            "--min-link-survival-margin-sec", str(profile.min_link_survival_margin_sec),
            "--base-url", str(cfg.get("base_url", "http://127.0.0.1:8088")),
            "--scenario-profile", scenario_profile,
            "--task-source-mode", task_source_mode,
            "--seed", str(seed),
            "--max-decisions", str(max_decisions),
            "--output-dir", str(replay_dir),
        ]
        if allow_placeholder_baselines:
            cmd.append("--allow-placeholder-baselines")
        if allow_non_paper_ready_baselines:
            cmd.append("--allow-non-paper-ready-baselines")
        rc = _run(cmd, dry_run=dry_run, env=env)
        if rc != 0:
            return {"status": "failed", "reason": "baseline_replay_failed", "run_dir": str(run_dir)}

    summarize_cmd = [
        sys.executable,
        "scripts/summarize_satedgesim_replay.py",
        "--input-dir",
        str(replay_dir),
        "--output",
        str(replay_dir / "summary_compact.json"),
    ]
    _run(summarize_cmd, dry_run=dry_run, env=env)

    regret_out = run_dir / "regret_eval.json"
    if baseline == "tri_mappo_maddpg" and str(cfg.get("trace_for_regret", "")).strip():
        regret_cmd = [
            sys.executable,
            "scripts/evaluate_policy_regret.py",
            "--checkpoint",
            tri_checkpoint,
            "--trace",
            str(cfg.get("trace_for_regret")),
            "--n-leo",
            str(cfg.get("n_leo", 16)),
            "--num-states",
            str(cfg.get("regret_num_states", 1024)),
            "--architecture",
            architecture,
            "--output",
            str(regret_out),
        ]
        _run(regret_cmd, dry_run=dry_run, env=env)
    else:
        _write_json(regret_out, {"status": "skipped", "reason": "baseline_without_checkpoint_regret"})

    compact = _read_json(replay_dir / "summary_compact.json")
    warnings = list(compact.get("warnings") or [])
    readiness = {
        "status": "pass" if not warnings else "warning",
        "warnings": warnings,
        "execution_ok": bool(
            compact.get("intent_execution_match_ratio", 0.0) >= 0.99
            and compact.get("receipt_accept_ratio", 0.0) >= 0.99
            and float((compact.get("fallback_reason_distribution") or {}).get("none", 0.0)) >= 0.99
            and int(compact.get("http_timeout_count", 1)) == 0
            and int(compact.get("http_connection_error_count", 1)) == 0
        ),
    }
    _write_json(run_dir / "readiness_check.json", readiness)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "profile": profile_metadata(profile_name),
            "architecture": architecture,
            "baseline": baseline,
            "baseline_metadata": baseline_meta,
            "seed": int(seed),
            "device": device,
            "max_decisions": max_decisions,
            "replay_dir": str(replay_dir),
            "tri_mappo_maddpg": {
                "tri_checkpoint_path": tri_checkpoint if baseline == "tri_mappo_maddpg" else "",
                "tri_checkpoint_status": tri_checkpoint_info.get("status", "") if baseline == "tri_mappo_maddpg" else "",
                "tri_checkpoint_resolution": tri_checkpoint_info if baseline == "tri_mappo_maddpg" else {},
                "tri_eval_mode": tri_eval_mode if baseline == "tri_mappo_maddpg" else "",
                "tri_requested_device": tri_device if baseline == "tri_mappo_maddpg" else "",
            },
        },
    )
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "warnings": warnings,
        "tri_checkpoint_path": tri_checkpoint if baseline == "tri_mappo_maddpg" else "",
        "tri_checkpoint_status": tri_checkpoint_info.get("status", "") if baseline == "tri_mappo_maddpg" else "",
        "tri_eval_mode": tri_eval_mode if baseline == "tri_mappo_maddpg" else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix for baselines/profiles/architectures/seeds.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output-root", type=str, default="outputs/matrix_mixed_v2")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed override, e.g. 13,17,23")
    parser.add_argument("--max-decisions", type=int, default=0, help="Optional max-decisions override for all runs")
    parser.add_argument(
        "--allow-placeholder-baselines",
        action="store_true",
        help="Allow running baselines with metadata type=placeholder (default: blocked).",
    )
    parser.add_argument(
        "--allow-non-paper-ready-baselines",
        action="store_true",
        help="Allow running non-paper-ready non-placeholder debug baselines.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.smoke:
        cfg = _smoke_overrides(cfg)
    if args.seeds.strip():
        cfg["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if int(args.max_decisions) > 0:
        cfg["max_decisions"] = int(args.max_decisions)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = list(cfg.get("profiles") or [])
    architectures = list(cfg.get("architectures") or [])
    baselines = [str(x).strip().lower() for x in (cfg.get("baselines") or [])]
    seeds = [int(s) for s in (cfg.get("seeds") or [])]
    if not profiles or not architectures or not baselines or not seeds:
        raise ValueError("config must contain non-empty profiles/architectures/baselines/seeds")

    baseline_resolution = _resolve_matrix_baselines(
        baselines,
        allow_placeholder_baselines=bool(args.allow_placeholder_baselines),
        allow_non_paper_ready_baselines=bool(args.allow_non_paper_ready_baselines),
    )
    if baseline_resolution["blocked_placeholders"]:
        raise ValueError(
            "placeholder baselines are blocked by default. "
            "Use --allow-placeholder-baselines to force run: "
            f"{baseline_resolution['blocked_placeholders']}"
        )
    if baseline_resolution["skipped_non_paper_ready"]:
        print(
            "WARN skipping non-paper-ready baselines by default: "
            f"{baseline_resolution['skipped_non_paper_ready']}"
        )
    baselines = list(baseline_resolution["selected"])
    if not baselines:
        raise ValueError("no baselines selected after paper-ready filtering")

    manifest = {
        "config": cfg,
        "smoke": bool(args.smoke),
        "dry_run": bool(args.dry_run),
        "device": args.device,
        "allow_placeholder_baselines": bool(args.allow_placeholder_baselines),
        "allow_non_paper_ready_baselines": bool(args.allow_non_paper_ready_baselines),
        "baseline_selection": baseline_resolution,
    }
    _write_json(output_root / "matrix_manifest.json", manifest)
    _write_json(output_root / "baseline_registry_metadata.json", baseline_metadata_json())

    results: List[Dict[str, Any]] = []
    for profile in profiles:
        for arch in architectures:
            for baseline in baselines:
                for seed in seeds:
                    res = _run_one(
                        profile_name=str(profile),
                        architecture=str(arch),
                        baseline=str(baseline),
                        seed=int(seed),
                        cfg=cfg,
                        output_root=output_root,
                        device=args.device,
                        dry_run=bool(args.dry_run),
                        allow_placeholder_baselines=bool(args.allow_placeholder_baselines),
                        allow_non_paper_ready_baselines=bool(args.allow_non_paper_ready_baselines),
                    )
                    res.update({"profile": profile, "architecture": arch, "baseline": baseline, "seed": int(seed)})
                    results.append(res)

    _write_json(output_root / "matrix_runs.json", {"runs": results})

    summarize_cmd = [
        sys.executable,
        "scripts/summarize_experiment_matrix.py",
        "--input-root",
        str(output_root),
        "--output-csv",
        str(output_root / "summary_matrix.csv"),
        "--output-json",
        str(output_root / "summary_matrix.json"),
    ]
    _run(summarize_cmd, dry_run=bool(args.dry_run), env=dict(os.environ))
    print(f"MATRIX_RUN_OK runs={len(results)} output_root={output_root}")


if __name__ == "__main__":
    main()
