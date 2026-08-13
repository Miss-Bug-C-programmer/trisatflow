from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_row(i: int) -> dict:
    phase = "phase_a" if i < 5 else "phase_b"
    task_type = "cpu_heavy" if i % 2 == 0 else "network_heavy"
    if i % 3 == 0:
        delays = {"local": 0.13 + i * 0.003, "neighbor": 0.09 + i * 0.002, "geo": 0.23 + i * 0.004, "ground": 0.17 + i * 0.003}
    elif i % 3 == 1:
        delays = {"local": 0.08 + i * 0.003, "neighbor": 0.12 + i * 0.002, "geo": 0.20 + i * 0.004, "ground": 0.16 + i * 0.003}
    else:
        delays = {"local": 0.15 + i * 0.003, "neighbor": 0.13 + i * 0.002, "geo": 0.19 + i * 0.004, "ground": 0.10 + i * 0.003}
    raw_wh = 10.0 + i * 0.2
    previous_wh = 10.0 + max(0, i - 1) * 0.2
    delta_wh = max(0.0, raw_wh - previous_wh)
    row = {
        "step": i,
        "leo_id": i % 4,
        "scenario_phase": phase,
        "task_type": task_type,
        "delay_semantic": "physical_seconds_controlled_estimate",
        "trace_semantic_class": "controlled_stress_fixture",
        "queue_estimate_source": "controlled_estimate",
        "raw_energy_counter_wh": raw_wh,
        "previous_raw_energy_counter_wh": previous_wh,
        "step_energy_delta_wh": delta_wh,
        "step_energy_delta_j": delta_wh * 3600.0,
        "energy_conversion_rule": "wh_to_j_x3600",
    }
    for tier_index, tier in enumerate(("local", "neighbor", "geo", "ground")):
        row[f"{tier}_visible"] = True
        row[f"{tier}_rate"] = 40.0 + tier_index * 20.0 + i
        row[f"{tier}_total_delay"] = delays[tier]
        row[f"{tier}_prop_delay"] = (0.0, 0.012 + i * 0.001, 0.055 + i * 0.001, 0.08 + i * 0.001)[tier_index]
        row[f"{tier}_best_queue"] = 1.0 + ((i + tier_index) % 4)
        row[f"{tier}_compute_capacity"] = 1000.0 + tier_index * 100.0
    return row


def _write_trace(path: Path, *, legacy: bool = False, bad_energy: bool = False) -> None:
    rows = [_base_row(i) for i in range(10)]
    if legacy:
        rows[0]["delay_semantic"] = "legacy_unknown"
    if bad_energy:
        rows[0]["step_energy_delta_j"] = 1.0
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _run_gate(trace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_scenario_physical_plausibility.py"),
            "--trace",
            str(trace),
            "--strict",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_physical_plausibility_gate_accepts_labeled_controlled_estimate_trace(tmp_path: Path) -> None:
    trace = tmp_path / "controlled_trace.jsonl"
    _write_trace(trace)

    result = _run_gate(trace)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PHYSICAL_PLAUSIBILITY_OK" in result.stdout
    assert "physical_seconds_controlled_estimate" in result.stdout


def test_physical_plausibility_gate_rejects_legacy_unknown_delay_semantic(tmp_path: Path) -> None:
    trace = tmp_path / "legacy_trace.jsonl"
    _write_trace(trace, legacy=True)

    result = _run_gate(trace)

    assert result.returncode != 0
    assert "paper_unsafe_delay_semantic:legacy_unknown" in result.stdout


def test_physical_plausibility_gate_rejects_wrong_wh_to_j_conversion(tmp_path: Path) -> None:
    trace = tmp_path / "bad_energy_trace.jsonl"
    _write_trace(trace, bad_energy=True)

    result = _run_gate(trace)

    assert result.returncode != 0
    assert "energy_conversion_violations" in result.stdout
