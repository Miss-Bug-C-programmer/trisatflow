from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_complete_gate_tree(root: Path) -> None:
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    shard_log = tests_dir / "test_action_masks.log"
    shard_log.write_text("1 passed\n", encoding="utf-8")
    (tests_dir / "shard_status.tsv").write_text(
        f"shard\tstatus\tlog\ntests/test_action_masks.py\t0\t{shard_log}\n",
        encoding="utf-8",
    )

    train_dir = root / "train"
    train_dir.mkdir()
    smoke_log = train_dir / "smoke_test.log"
    smoke_log.write_text("SMOKE_TEST_OK upper=mappo lower=maddpg\n", encoding="utf-8")
    metrics_csv = train_dir / "metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mean_delay_s", "mean_energy_j", "normalized_system_cost", "reward_mean"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mean_delay_s": "1.0",
                "mean_energy_j": "2.0",
                "normalized_system_cost": "3.0",
                "reward_mean": "-1.0",
            }
        )
    checkpoint = train_dir / "smoke_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_metadata = train_dir / "run_metadata.json"
    run_metadata.write_text(
        json.dumps(
            {
                "requested_device": "cpu",
                "actual_device": "cpu",
                "uses_privileged_info": False,
                "observation_mode": "safe_observable",
            }
        ),
        encoding="utf-8",
    )
    resolved_config = train_dir / "resolved_config.yaml"
    resolved_config.write_text("total_episodes: 2\n", encoding="utf-8")
    manifest = {
        "status": "ok",
        "artifacts": {
            "metrics_csv": str(metrics_csv),
            "checkpoint": str(checkpoint),
            "run_metadata": str(run_metadata),
            "resolved_config": str(resolved_config),
        },
    }
    (train_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_audit_stage_outputs_accepts_complete_gate_tree(tmp_path: Path) -> None:
    _write_complete_gate_tree(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_stage_outputs.py"),
            "--stage",
            "stage_00_gate_scaffold",
            "--input-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )

    assert "AUDIT_STAGE_OUTPUTS_OK" in result.stdout
    assert (tmp_path / "audit_report.json").is_file()


def test_audit_stage_outputs_rejects_missing_smoke_marker(tmp_path: Path) -> None:
    _write_complete_gate_tree(tmp_path)
    (tmp_path / "train" / "smoke_test.log").write_text("training finished\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_stage_outputs.py"),
            "--stage",
            "stage_00_gate_scaffold",
            "--input-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert "SMOKE_TEST_OK marker not found" in result.stderr


def test_stage_gate_shell_scripts_use_timeboxed_shards() -> None:
    gate_script = (REPO_ROOT / "scripts" / "run_stage_smoke_gate.sh").read_text(encoding="utf-8")
    shard_script = (REPO_ROOT / "scripts" / "test_shards.sh").read_text(encoding="utf-8")

    assert "bash scripts/test_shards.sh \"$OUT/tests\"" in gate_script
    assert "timeout \"$SMOKE_TIMEOUT\" python scripts/smoke_test.py" in gate_script
    assert "--device \"$SMOKE_DEVICE\"" in gate_script
    assert "stage_09_policy_adaptivity" in gate_script
    assert "train_stress" in gate_script
    assert "scripts/check_policy_adaptivity.py" in gate_script
    assert "python scripts/audit_stage_outputs.py" in gate_script
    assert "touch \"$OUT/GATE_OK\"" in gate_script
    assert gate_script.index("python scripts/audit_stage_outputs.py") < gate_script.index("touch \"$OUT/GATE_OK\"")
    assert gate_script.index("scripts/check_policy_adaptivity.py") < gate_script.index("touch \"$OUT/GATE_OK\"")
    assert "timeout 120s python -m pytest -q" in shard_script
    assert "tests/test_action_masks.py" in shard_script
    assert "tests/test_units_and_metrics_schema.py" in shard_script
    assert "pytest -q tests" not in shard_script
