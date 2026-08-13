from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_metrics(path: Path, *, deterministic_ground: bool = False, eval_entropy: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mean_feasibility",
        "invalid_action_ratio",
        "trace_hit_ratio",
        "trace_fallback_count",
        "upper_local_ratio",
        "upper_neighbor_ratio",
        "upper_geo_ratio",
        "upper_ground_ratio",
        "eval_argmax_local_ratio",
        "eval_argmax_neighbor_ratio",
        "eval_argmax_geo_ratio",
        "eval_argmax_ground_ratio",
        "eval_policy_entropy",
        "neighbor_visible_ratio",
        "geo_visible_ratio",
        "ground_visible_ratio",
        "neighbor_selected_when_visible_ratio",
        "geo_selected_when_visible_ratio",
        "ground_selected_when_visible_ratio",
        "remote_selected_when_visible_ratio",
        "mean_delay_s",
        "mean_energy_j",
        "normalized_system_cost",
        "reward_mean",
    ]
    eval_values = ("0.00", "0.00", "0.00", "1.00") if deterministic_ground else ("0.35", "0.25", "0.20", "0.20")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _ in range(3):
            writer.writerow(
                {
                    "mean_feasibility": "1.0",
                    "invalid_action_ratio": "0.0",
                    "trace_hit_ratio": "1.0",
                    "trace_fallback_count": "0",
                    "upper_local_ratio": "0.40",
                    "upper_neighbor_ratio": "0.25",
                    "upper_geo_ratio": "0.15",
                    "upper_ground_ratio": "0.20",
                    "eval_argmax_local_ratio": eval_values[0],
                    "eval_argmax_neighbor_ratio": eval_values[1],
                    "eval_argmax_geo_ratio": eval_values[2],
                    "eval_argmax_ground_ratio": eval_values[3],
                    "eval_policy_entropy": eval_entropy,
                    "neighbor_visible_ratio": "1.0",
                    "geo_visible_ratio": "1.0",
                    "ground_visible_ratio": "1.0",
                    "neighbor_selected_when_visible_ratio": "0.25",
                    "geo_selected_when_visible_ratio": "0.15",
                    "ground_selected_when_visible_ratio": "0.20",
                    "remote_selected_when_visible_ratio": "0.60",
                    "mean_delay_s": "1.0",
                    "mean_energy_j": "1.0",
                    "normalized_system_cost": "1.0",
                    "reward_mean": "-1.0",
                }
            )


def _write_trace(root: Path, semantic: str, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    trace = root / "seed_1.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (root / "seed_1.jsonl.manifest.json").write_text(
        json.dumps({"trace_semantic_class": semantic}),
        encoding="utf-8",
    )


def _write_rollout_debug(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_phase", "selected_action"])
        writer.writeheader()
        for phase, action in rows:
            writer.writerow({"scenario_phase": phase, "selected_action": action})


def _run_check(tmp_path: Path, semantic: str, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_policy_adaptivity.py"),
        "--metrics",
        str(tmp_path / "metrics.csv"),
        "--trace-root",
        str(tmp_path / "trace"),
        "--trace-semantic-class",
        semantic,
        "--fail-on-deterministic-dominance",
    ]
    if extra:
        cmd.extend(extra)
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=20, check=False)


def test_actual_policy_adaptivity_gate_accepts_dynamic_actual_trace(tmp_path: Path) -> None:
    semantic = "actual_physical_projection"
    _write_metrics(tmp_path / "metrics.csv")
    _write_rollout_debug(tmp_path / "rollout_debug.csv", [("default_phase", 0), ("default_phase", 1), ("default_phase", 3)])
    _write_trace(
        tmp_path / "trace",
        semantic,
        [
            {"trace_semantic_class": semantic, "step": 0, "leo_id": 0, "scenario_phase": "default_phase", "abstract_action_mask_final": [1, 1, 0, 1]},
            {"trace_semantic_class": semantic, "step": 1, "leo_id": 0, "scenario_phase": "default_phase", "abstract_action_mask_final": [1, 0, 0, 1]},
        ],
    )

    result = _run_check(tmp_path, semantic)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "POLICY_ADAPTIVITY_OK" in result.stdout


def test_policy_adaptivity_gate_rejects_deterministic_ground_dominance(tmp_path: Path) -> None:
    semantic = "actual_physical_projection"
    _write_metrics(tmp_path / "metrics.csv", deterministic_ground=True)
    _write_rollout_debug(tmp_path / "rollout_debug.csv", [("default_phase", 0), ("default_phase", 1), ("default_phase", 3)])
    _write_trace(
        tmp_path / "trace",
        semantic,
        [
            {"trace_semantic_class": semantic, "step": 0, "leo_id": 0, "scenario_phase": "default_phase", "abstract_action_mask_final": [1, 1, 0, 1]},
            {"trace_semantic_class": semantic, "step": 1, "leo_id": 0, "scenario_phase": "default_phase", "abstract_action_mask_final": [1, 0, 0, 1]},
        ],
    )

    result = _run_check(tmp_path, semantic)

    assert result.returncode != 0
    assert "deterministic_single_action_dominance:ground" in result.stdout


def test_policy_adaptivity_gate_treats_high_entropy_argmax_dominance_as_warning(tmp_path: Path) -> None:
    semantic = "controlled_stress_projection"
    _write_metrics(tmp_path / "metrics.csv", deterministic_ground=True, eval_entropy="1.05")
    _write_rollout_debug(
        tmp_path / "rollout_debug.csv",
        [("local_favored", 0), ("local_favored", 1), ("ground_favored", 3), ("ground_favored", 3)],
    )
    _write_trace(
        tmp_path / "trace",
        semantic,
        [
            {"trace_semantic_class": semantic, "step": 0, "leo_id": 0, "scenario_phase": "local_favored", "abstract_action_mask_final": [1, 1, 0, 1]},
            {"trace_semantic_class": semantic, "step": 1, "leo_id": 0, "scenario_phase": "ground_favored", "abstract_action_mask_final": [1, 1, 0, 1]},
        ],
    )

    result = _run_check(tmp_path, semantic, ["--min-phase-action-divergence", "0.01"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "argmax_single_action_dominance_high_entropy:ground" in result.stdout


def test_controlled_stress_gate_requires_phase_conditioned_action_divergence(tmp_path: Path) -> None:
    semantic = "controlled_stress_projection"
    _write_metrics(tmp_path / "metrics.csv")
    _write_rollout_debug(
        tmp_path / "rollout_debug.csv",
        [("local_favored", 0), ("local_favored", 0), ("ground_favored", 0), ("ground_favored", 0)],
    )
    _write_trace(
        tmp_path / "trace",
        semantic,
        [
            {"trace_semantic_class": semantic, "step": 0, "leo_id": 0, "scenario_phase": "local_favored", "abstract_action_mask_final": [1, 1, 0, 0]},
            {"trace_semantic_class": semantic, "step": 1, "leo_id": 0, "scenario_phase": "ground_favored", "abstract_action_mask_final": [1, 0, 0, 1]},
        ],
    )

    result = _run_check(tmp_path, semantic, ["--min-phase-action-divergence", "0.01"])

    assert result.returncode != 0
    assert "phase_action_divergence=0.000000<min_phase_action_divergence=0.01" in result.stdout


def _write_gate_tree(root: Path, *, include_stress: bool) -> None:
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    shard_log = tests_dir / "test_action_masks.log"
    shard_log.write_text("1 passed\n", encoding="utf-8")
    (tests_dir / "shard_status.tsv").write_text(
        f"shard\tstatus\tlog\ntests/test_action_masks.py\t0\t{shard_log}\n",
        encoding="utf-8",
    )
    for name in (["train", "train_stress"] if include_stress else ["train"]):
        train_dir = root / name
        _write_metrics(train_dir / "metrics.csv")
        (train_dir / "smoke_test.log").write_text("SMOKE_TEST_OK upper=mappo lower=maddpg\n", encoding="utf-8")
        checkpoint = train_dir / "smoke_checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        (train_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "requested_device": "auto",
                    "actual_device": "cpu",
                    "uses_privileged_info": False,
                    "observation_mode": "safe_observable",
                }
            ),
            encoding="utf-8",
        )
        (train_dir / "resolved_config.yaml").write_text("total_episodes: 2\n", encoding="utf-8")
        (train_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "artifacts": {
                        "metrics_csv": str(train_dir / "metrics.csv"),
                        "checkpoint": str(checkpoint),
                        "run_metadata": str(train_dir / "run_metadata.json"),
                        "resolved_config": str(train_dir / "resolved_config.yaml"),
                    },
                }
            ),
            encoding="utf-8",
        )


def test_stage_09_output_audit_requires_controlled_stress_train(tmp_path: Path) -> None:
    _write_gate_tree(tmp_path, include_stress=False)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_stage_outputs.py"),
            "--stage",
            "stage_09_policy_adaptivity",
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
    assert "train_stress" in result.stderr
