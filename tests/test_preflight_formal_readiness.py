from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SATEDGESIM_ROOT = REPO_ROOT.parent / "satedgeSimv2"


def _write_config(path: Path, *, physical_enabled: bool = True, oracle_mask: bool = False, smoke: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_seeds = [1, 2] if smoke else [1, 2, 3, 4, 5, 6, 7, 8]
    output_dir = "outputs/preflight_smoke_case" if smoke else "outputs/preflight_formal_case"
    path.write_text(
        "\n".join(
            [
                "total_episodes: 1",
                "output_dir: " + output_dir,
                "experiment:",
                f"  paper_ready: {'false' if smoke else 'true'}",
                "  split:",
                f"    train_seeds: {train_seeds}",
                "    val_seeds: [101]",
                "    test_seeds: [201]",
                "reward:",
                "  mode: physical_weighted",
                "observation:",
                "  mode: safe_observable",
                "  include_oracle_cost: false",
                "scenario:",
                "  physical:",
                f"    enabled: {'true' if physical_enabled else 'false'}",
                "  mask_source: " + ("oracle_trace" if oracle_mask else "predicted"),
                "algo:",
                "  encoder_mode: separate_lower_encoder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": "unit-train",
        "trace_path": "traces/train/unit.jsonl",
        "source": "sgp4_generated",
        "generation_seed": 123,
        "n_leo": 6,
        "n_geo": 1,
        "n_ground": 1,
        "duration_s": 128.0,
        "slot_duration_s": 1.0,
        "scenario_profile": "unit",
        "split": "train",
        "sha256": "unitsha256",
        "content_fingerprint": "unitfingerprint",
        "contains_oracle_fields": False,
        "oracle_field_names": [],
        "safe_observable_excludes_oracle_fields": True,
        "generator_config_hash": "unitconfig",
        "created_at": "unit-test",
        "notes": "unit test manifest",
        "time_start_s": 0.0,
        "time_end_s": 128.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_preflight(config: Path, manifest: Path | None, *extra: str, run_mode: str = "formal") -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/preflight_formal_readiness.py",
        "--config",
        str(config),
        "--satedgesim-root",
        str(SATEDGESIM_ROOT),
        "--run-mode",
        run_mode,
    ]
    if manifest is not None:
        cmd.extend(["--trace-manifest", str(manifest)])
    cmd.extend(extra)
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _load_report() -> dict:
    path = REPO_ROOT / "outputs" / "preflight" / "formal_readiness_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _check_status(report: dict, name: str) -> str:
    for check in report["checks"]:
        if check["name"] == name:
            return check["status"]
    raise AssertionError(f"missing check {name}")


def test_compliant_formal_config_passes(tmp_path: Path) -> None:
    config = tmp_path / "formal_ok.yaml"
    manifest = tmp_path / "trace.manifest.json"
    _write_config(config)
    _write_manifest(manifest)

    result = _run_preflight(config, manifest)
    report = _load_report()

    assert result.returncode == 0
    assert report["run_mode"] == "formal"
    assert report["formal_ready"] is True
    assert report["formal_claim_allowed"] is True
    assert _check_status(report, "scenario_physical_enabled") == "passed"
    assert _check_status(report, "trace_manifest_split_safety") == "passed"


def test_physical_disabled_fails(tmp_path: Path) -> None:
    config = tmp_path / "physical_disabled.yaml"
    manifest = tmp_path / "trace.manifest.json"
    _write_config(config, physical_enabled=False)
    _write_manifest(manifest)

    result = _run_preflight(config, manifest)
    report = _load_report()

    assert result.returncode != 0
    assert report["formal_ready"] is False
    assert _check_status(report, "config_load") == "failed"
    assert "scenario.physical.enabled=true" in report["checks"][0]["message"]


def test_oracle_mask_formal_fails(tmp_path: Path) -> None:
    config = tmp_path / "oracle_mask.yaml"
    manifest = tmp_path / "trace.manifest.json"
    _write_config(config, oracle_mask=True)
    _write_manifest(manifest)

    result = _run_preflight(config, manifest)
    report = _load_report()

    assert result.returncode != 0
    assert report["formal_ready"] is False
    assert _check_status(report, "oracle_privileged_fields") == "failed"


def test_placeholder_baseline_formal_fails(tmp_path: Path) -> None:
    config = tmp_path / "formal_with_placeholder.yaml"
    manifest = tmp_path / "trace.manifest.json"
    _write_config(config)
    _write_manifest(manifest)

    result = _run_preflight(config, manifest, "--baseline", "hmadrl_maddqn_ddpg")
    report = _load_report()

    assert result.returncode != 0
    assert report["formal_ready"] is False
    assert _check_status(report, "baseline_formal_registry") == "failed"
    assert "is_placeholder=true" in json.dumps(report["checks"])


def test_smoke_mode_outputs_non_formal_report(tmp_path: Path) -> None:
    config = tmp_path / "smoke.yaml"
    _write_config(config, smoke=True)

    result = _run_preflight(config, None, run_mode="smoke")
    report = _load_report()

    assert result.returncode == 0
    assert report["run_mode"] == "smoke"
    assert report["formal_ready"] is False
    assert report["outputs_are_smoke_only"] is True
    assert _check_status(report, "formal_train_seed_count") == "warning"
