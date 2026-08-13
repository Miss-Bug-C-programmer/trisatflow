from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from satedgesim_semantics import energy_semantics  # noqa: E402


def test_missing_final_energy_stays_null_not_zero() -> None:
    info = energy_semantics([], {}, {}, energy_source="simlog_final_wh")

    assert info["simlog_final_energy_wh"] is None
    assert info["selected_energy_value"] is None
    assert info["energy_source"] == "unavailable"
    assert info["energy_source_available"] is False
    assert info["energy_unavailable_reason"] == "simlog_final_wh_unavailable"


def test_energy_sources_do_not_mix_wh_and_j() -> None:
    rows = [
        {
            "receipt_energy_delta_wh": 2.5,
            "estimator_expected_energy_j": 10.0,
        }
    ]
    final_metrics = {"energyConsumption": 7.0, "energyCounterUnit": "Wh"}

    receipt = energy_semantics(rows, {}, final_metrics, energy_source="receipt_delta_wh")
    simlog = energy_semantics(rows, {}, final_metrics, energy_source="simlog_final_wh")
    estimator = energy_semantics(rows, {}, final_metrics, energy_source="estimator_expected_j")

    assert receipt["selected_energy_value"] == 2.5
    assert receipt["energy_unit"] == "Wh"
    assert simlog["selected_energy_value"] == 7.0
    assert simlog["energy_unit"] == "Wh"
    assert estimator["selected_energy_value"] == 10.0
    assert estimator["energy_unit"] == "J"


def _write_replay_case(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "decision_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "receipt_accepted", "actionAccepted", "executionScheduled"])
        writer.writeheader()
        writer.writerow({"step": 0, "receipt_accepted": 1, "actionAccepted": True, "executionScheduled": True})
    (path / "summary.json").write_text(json.dumps({"status": "SMOKE"}), encoding="utf-8")
    (path / "final_metrics.json").write_text(json.dumps({}), encoding="utf-8")


def test_formal_summary_missing_energy_source_fails(tmp_path: Path) -> None:
    case_dir = tmp_path / "missing_energy"
    _write_replay_case(case_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_satedgesim_replay.py",
            "--input-dir",
            str(case_dir),
            "--energy-source",
            "simlog_final_wh",
            "--formal",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "formal SatEdgeSim summary requires energy source simlog_final_wh" in (result.stdout + result.stderr)


def test_diagnostic_summary_missing_energy_source_marks_non_formal(tmp_path: Path) -> None:
    case_dir = tmp_path / "missing_energy_diagnostic"
    output = tmp_path / "diagnostic_summary.json"
    _write_replay_case(case_dir)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_satedgesim_replay.py",
            "--input-dir",
            str(case_dir),
            "--output",
            str(output),
            "--energy-source",
            "simlog_final_wh",
            "--allow-diagnostic-energy-missing",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["energy_source_available"] is False
    assert payload["energy_formal_claim_allowed"] is False
    assert payload["formal_claim_allowed"] is False
    assert payload["outputs_are_diagnostic"] is True
