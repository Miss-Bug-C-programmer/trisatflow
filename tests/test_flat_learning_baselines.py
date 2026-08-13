from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from trisatflow.agents.flat_hybrid_trainer import FlatHybridTrainer
from trisatflow.baselines.registry import baseline_metadata_registry
from trisatflow.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_CONFIG = REPO_ROOT / "trisatflow" / "configs" / "paper" / "satedgesim_trace_mixed_v3_safe.yaml"


def _tiny_flat_cfg(tmp_path: Path):
    cfg = load_config(SAFE_CONFIG)
    cfg.output_dir = str(tmp_path / "flat")
    cfg.device = "cpu"
    cfg.total_episodes = 1
    cfg.steps_per_episode = 2
    cfg.scenario.episode_len = 2
    cfg.scenario.n_leo = 4
    cfg.scenario.seed = 13
    cfg.scenario.topology_mode = "analytic"
    cfg.scenario.topology_trace_path = ""
    cfg.scenario.topology_trace_strict = False
    cfg.algo.ppo_epochs = 1
    cfg.algo.policy_hidden_dim = 32
    cfg.algo.gnn_hidden_dim = 16
    return cfg


def test_flat_policy_short_train_and_eval(tmp_path: Path) -> None:
    cfg = _tiny_flat_cfg(tmp_path)
    trainer = FlatHybridTrainer(cfg, baseline_name="flat_ppo")

    history = trainer.train()
    eval_row = trainer.evaluate(seed=101, episodes=1)
    trainer.save_checkpoint(tmp_path / "flat" / "checkpoint.pt")

    assert len(history) == 1
    assert history[-1]["baseline"] == "flat_ppo"
    assert "normalized_system_cost" in history[-1]
    assert "normalized_system_cost" in eval_row
    assert (tmp_path / "flat" / "metrics.csv").is_file()
    assert (tmp_path / "flat" / "checkpoint.pt").is_file()


def test_learning_baseline_sweep_uses_one_fairness_contract_and_budget(tmp_path: Path) -> None:
    out = tmp_path / "learning"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sweep_learning_baselines.py"),
            "--config",
            str(SAFE_CONFIG),
            "--baselines",
            "flat_ppo,flat_mappo,hierarchical_no_gnn",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--n-leo",
            "4",
            "--train-seeds",
            "13",
            "--val-seeds",
            "101",
            "--test-seeds",
            "202",
            "--device",
            "cpu",
            "--output-root",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    assert "LEARNING_BASELINES_OK" in result.stdout

    rows = list(csv.DictReader((out / "sweep_summary.csv").open("r", encoding="utf-8", newline="")))
    assert {row["baseline"] for row in rows} == {"flat_ppo", "flat_mappo", "hierarchical_no_gnn"}
    assert {row["fairness_contract_sha256"] for row in rows} == {
        json.loads((out / "learning_baseline_manifest.json").read_text(encoding="utf-8"))["fairness_contract_sha256"]
    }
    assert {row["episodes"] for row in rows} == {"1"}
    assert {row["episode_len"] for row in rows} == {"2"}
    assert {row["n_leo"] for row in rows} == {"4"}
    assert len({row["topology_trace_path"] for row in rows}) == 1
    assert all(row["status"] == "ok" for row in rows)


def test_placeholder_cannot_enter_formal_exporters(tmp_path: Path) -> None:
    metadata = baseline_metadata_registry()
    assert metadata["hmadrl_maddqn_ddpg"].type == "placeholder"
    assert metadata["hmadrl_maddqn_ddpg"].paper_ready is False

    summary_json = tmp_path / "summary_matrix.json"
    summary_json.write_text(
        json.dumps(
            [
                {
                    "profile": "paper",
                    "architecture": "full",
                    "baseline": "hmadrl_maddqn_ddpg",
                    "task_success_ratio": 0.0,
                    "mean_delay": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    table_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "v1_fix" / "export_paper_tables.py"),
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(tmp_path / "tables"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert table_result.returncode != 0
    assert "placeholder baselines" in (table_result.stderr + table_result.stdout)

    input_root = tmp_path / "aggregate_input"
    input_root.mkdir()
    with (input_root / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "phase", "seed", "baseline", "final_normalized_system_cost"])
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "phase": "test",
                "seed": "13",
                "baseline": "hmadrl_maddqn_ddpg",
                "final_normalized_system_cost": "1.0",
            }
        )
    aggregate_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_results.py"),
            "--input-root",
            str(input_root),
            "--output",
            str(tmp_path / "aggregate_out"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert aggregate_result.returncode != 0
    assert "placeholder baselines" in (aggregate_result.stderr + aggregate_result.stdout)


def test_literature_mapping_document_exists() -> None:
    doc = REPO_ROOT / "docs" / "paper_ready_v3" / "LITERATURE_BASELINE_MAPPING.md"
    text = doc.read_text(encoding="utf-8")
    assert "flat_ppo" in text
    assert "flat_mappo" in text
    assert "hierarchical_no_gnn" in text
    assert "hmadrl_maddqn_ddpg" in text
    assert "paper_ready=false" in text
