#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml"
DEFAULT_TRACE_ROOT = "traces/paper_v3"
DEFAULT_OUTPUT_ROOT = "outputs/paper_ready_v3"

PRIOR_STAGE_GATES: Sequence[Sequence[str]] = (
    ("stage_00_gate_scaffold",),
    ("stage_01_contract",),
    ("stage_02_mask_safe_baselines",),
    ("stage_03_metric_schema",),
    ("stage_04_statistics",),
    ("stage_05_seed_protocol",),
    ("stage_06_trace_contract",),
    ("stage_06_trace_validator",),
    ("stage_07_physical_semantics",),
    ("stage8", "stage_08_trace_bank"),
    ("stage_09_policy_adaptivity",),
    ("stage_10_ablations",),
    ("stage_11_learning_baselines",),
    ("stage_12_reporting",),
)
REQUIRED_VERSION_FIELDS = {
    "simulator_version",
    "git_commit",
    "rest_api_schema_version",
    "state_schema_version",
    "candidate_cost_estimator_version",
    "lower_action_binding_version",
    "settings_root",
    "settings_sha256",
    "build_time_utc",
}


class PreflightError(RuntimeError):
    pass


def _run(
    cmd: Sequence[str],
    *,
    timeout: int,
    cwd: Path = REPO_ROOT,
    env: Dict[str, str] | None = None,
    output: Path | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "cmd": list(cmd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if result.returncode != 0 and not allow_failure:
        raise PreflightError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("OK\n", encoding="utf-8")


def _gate_ok_for_group(smoke_root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = smoke_root / name / "GATE_OK"
        if path.is_file():
            return path
    return None


def _check_prior_stage_gates(smoke_root: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    missing: List[str] = []
    for group in PRIOR_STAGE_GATES:
        found = _gate_ok_for_group(smoke_root, group)
        checks.append({"gate_group": list(group), "status": "ok" if found else "missing", "path": str(found or "")})
        if found is None:
            missing.append("|".join(group))
    if missing:
        raise PreflightError(f"missing prior stage GATE_OK artifacts: {missing}")
    return checks


def _fetch_json(url: str, *, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise PreflightError(f"REST endpoint did not return a JSON object: {url}")
    return payload


def _validate_version(version: Dict[str, Any]) -> None:
    missing = sorted(field for field in REQUIRED_VERSION_FIELDS if not version.get(field))
    if missing:
        raise PreflightError(f"/version missing provenance fields: {missing}")
    if str(version.get("git_commit", "")).strip().lower() == "unknown":
        raise PreflightError("/version git_commit is unknown")
    if str(version.get("settings_sha256", "")).startswith("MISSING:"):
        raise PreflightError("/version settings_sha256 reports missing settings")


def _satedgesim_root(raw: str) -> Path:
    text = str(raw or os.environ.get("SATEDGESIM_ROOT", "")).strip()
    if not text:
        raise PreflightError("SATEDGESIM_ROOT or --satedgesim-root is required for live preflight Maven compile")
    root = Path(text).expanduser()
    if not root.is_dir():
        raise PreflightError(f"SatEdgeSim root does not exist: {root}")
    if not (root / "scripts" / "run_rl_server.sh").is_file():
        raise PreflightError(f"SatEdgeSim root is missing scripts/run_rl_server.sh: {root}")
    return root


def run_offline(args: argparse.Namespace) -> None:
    out = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / "preflight_offline")
    checks: List[Dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)

    smoke_root = Path(args.smoke_root)
    checks.extend(_check_prior_stage_gates(smoke_root))

    config = Path(args.config)
    if not config.is_file():
        raise PreflightError(f"paper-safe config missing: {config}")
    checks.append({"name": "paper_safe_config_exists", "status": "ok", "path": str(config)})

    contract_json = out / "experiment_contract.json"
    _run(
        [
            sys.executable,
            "scripts/audit_experiment_contract.py",
            "--config",
            str(config),
            "--require-paper-safe",
            "--output-json",
            str(contract_json),
        ],
        timeout=int(args.timeout_sec),
        output=out / "audit_experiment_contract.run.json",
    )
    checks.append({"name": "paper_safe_contract", "status": "ok", "path": str(contract_json)})

    trace_root = Path(args.trace_root)
    if not trace_root.is_dir():
        raise PreflightError(f"trace bank root missing: {trace_root}")
    _run(
        [
            sys.executable,
            "scripts/audit_trace_bank.py",
            "--trace-root",
            str(trace_root),
            "--require-disjoint-splits",
            "--require-provenance",
            "--paper-strict",
        ],
        timeout=max(60, int(args.timeout_sec)),
        output=out / "audit_trace_bank.run.json",
    )
    checks.append({"name": "trace_bank_audit", "status": "ok", "path": str(trace_root)})

    stage12_root = smoke_root / "stage_12_reporting"
    _run(
        [
            sys.executable,
            "scripts/export_paper_tables.py",
            "--input-root",
            str(stage12_root),
            "--output-dir",
            str(out / "stage12_tables_probe"),
            "--allow-smoke-small-n",
        ],
        timeout=max(60, int(args.timeout_sec)),
        output=out / "export_tables_probe.run.json",
    )
    _run(
        [
            sys.executable,
            "scripts/plot_paper_results.py",
            "--input-root",
            str(stage12_root),
            "--output-dir",
            str(out / "stage12_figures_probe"),
            "--allow-smoke-small-n",
        ],
        timeout=max(60, int(args.timeout_sec)),
        output=out / "plot_figures_probe.run.json",
    )
    checks.append({"name": "reporting_input_contract", "status": "ok", "path": str(stage12_root)})

    _write_json(
        out / "preflight_offline.json",
        {
            "status": "PAPER_READY_V3_PREFLIGHT_OFFLINE_OK",
            "config": str(config),
            "trace_root": str(trace_root),
            "checks": checks,
        },
    )
    _touch(out / "GATE_OK")
    print(f"PAPER_READY_V3_PREFLIGHT_OFFLINE_OK output_dir={out}")


def run_satedgesim(args: argparse.Namespace) -> None:
    out = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / "preflight_satedgesim")
    out.mkdir(parents=True, exist_ok=True)
    root = _satedgesim_root(str(args.satedgesim_root))
    base_url = str(args.base_url).rstrip("/")
    checks: List[Dict[str, Any]] = []

    mvn = shutil.which("mvn")
    if not mvn:
        raise PreflightError("mvn is required for SatEdgeSim live preflight")
    _run([mvn, "-q", "-DskipTests", "compile"], cwd=root, timeout=int(args.compile_timeout_sec), output=out / "maven_compile.run.json")
    _touch(out / "SATEDGESIM_MAVEN_COMPILE_OK")
    checks.append({"name": "maven_compile", "status": "ok", "root": str(root)})

    health = _fetch_json(f"{base_url}/health", timeout=10.0)
    _write_json(out / "health.json", health)
    _touch(out / "SATEDGESIM_REST_HEALTH_OK")
    checks.append({"name": "rest_health", "status": "ok", "base_url": base_url})

    version = _fetch_json(f"{base_url}/version", timeout=10.0)
    _validate_version(version)
    _write_json(out / "version.json", version)
    _touch(out / "SATEDGESIM_VERSION_PROVENANCE_OK")
    checks.append({"name": "version_provenance", "status": "ok", "lower_action_binding_version": version.get("lower_action_binding_version")})

    common_env = {"SATEDGE_BASE_URL": base_url, "SATEDGESIM_ROOT": str(root)}
    _run(
        [
            sys.executable,
            "scripts/test_satedgesim_rest_contract.py",
            "--base-url",
            base_url,
            "--seed",
            str(args.seed),
            "--devices-count",
            str(args.devices_count),
            "--output",
            str(out / "rest_contract.json"),
        ],
        timeout=int(args.live_timeout_sec),
        env=common_env,
        output=out / "rest_contract.run.json",
    )
    _touch(out / "SATEDGESIM_REST_CONTRACT_OK")
    checks.append({"name": "rest_contract", "status": "ok"})

    _run(
        [
            sys.executable,
            "scripts/test_satedgesim_decision_receipt.py",
            "--base-url",
            base_url,
            "--steps",
            str(args.receipt_steps),
            "--seed",
            str(args.seed),
            "--devices-count",
            str(args.devices_count),
            "--output",
            str(out / "decision_receipt.json"),
        ],
        timeout=int(args.live_timeout_sec),
        env=common_env,
        output=out / "decision_receipt.run.json",
    )
    _touch(out / "SATEDGESIM_DECISION_RECEIPT_OK")
    checks.append({"name": "decision_receipt", "status": "ok"})

    binding_json = out / "lower_action_binding.json"
    binding = _run(
        [
            sys.executable,
            "scripts/check_satedgesim_lower_action_binding.py",
            "--base-url",
            base_url,
            "--devices-count",
            str(args.devices_count),
            "--output",
            str(binding_json),
        ],
        timeout=int(args.live_timeout_sec),
        env=common_env,
        output=out / "lower_action_binding.run.json",
        allow_failure=True,
    )
    if binding.returncode == 0:
        _touch(out / "SATEDGESIM_LOWER_ACTION_BINDING_OK")
        checks.append({"name": "lower_action_binding", "status": "ok"})
    else:
        payload = json.loads(binding_json.read_text(encoding="utf-8")) if binding_json.is_file() else {}
        if payload.get("status") != "STAGE_BLOCKED_FOR_FULL_HYBRID_CLAIM":
            raise PreflightError(f"lower action binding check failed: {binding.stderr}")
        _touch(out / "SATEDGESIM_LOWER_ACTION_BINDING_BLOCKED_FULL_HYBRID")
        checks.append({"name": "lower_action_binding", "status": "blocked_full_hybrid_claim"})

    _write_json(
        out / "preflight_satedgesim.json",
        {
            "status": "PAPER_READY_V3_PREFLIGHT_SATEDGESIM_OK",
            "base_url": base_url,
            "satedgesim_root": str(root),
            "checks": checks,
        },
    )
    _touch(out / "GATE_OK")
    print(f"PAPER_READY_V3_PREFLIGHT_SATEDGESIM_OK output_dir={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready v3 offline or SatEdgeSim live preflight checks.")
    parser.add_argument("--mode", choices=["offline", "satedgesim"], required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trace-root", default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--smoke-root", default="outputs/smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--base-url", default=os.environ.get("SATEDGE_BASE_URL", "http://127.0.0.1:8088"))
    parser.add_argument("--satedgesim-root", default=os.environ.get("SATEDGESIM_ROOT", ""))
    parser.add_argument("--compile-timeout-sec", type=int, default=180)
    parser.add_argument("--live-timeout-sec", type=int, default=180)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=12)
    parser.add_argument("--receipt-steps", type=int, default=12)
    args = parser.parse_args()

    try:
        if args.mode == "offline":
            run_offline(args)
        else:
            run_satedgesim(args)
    except (PreflightError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        out = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / ("preflight_offline" if args.mode == "offline" else "preflight_satedgesim"))
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "preflight_failed.json", {"status": "failed", "mode": args.mode, "error": str(exc)})
        raise SystemExit(f"paper-ready v3 preflight failed: {exc}") from exc


if __name__ == "__main__":
    main()
