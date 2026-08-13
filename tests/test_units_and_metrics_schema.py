from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import METRIC_SCHEMA_VERSION, build_step_metric_bundle
from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import AlgoConfig, ScenarioConfig, TrainConfig
from trisatflow.envs.units import UnitScaleConfig, validate_metric_field_names

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_physical_metric_field_names_use_explicit_units() -> None:
    fields = [
        "mean_delay_s",
        "p95_delay_s",
        "mean_energy_j",
        "mean_queue_length_tasks",
        "physical_delay_s",
        "physical_energy_j",
        "physical_queue_length_tasks",
    ]
    assert validate_metric_field_names(fields) == []


def test_normalized_fields_not_named_as_physical_units() -> None:
    fields = [
        "normalized_system_cost",
        "normalized_training_cost",
        "reward_mean",
        "legacy_trace_delay_score",
        "normalized_delay_sec",  # intentionally bad
        "normalized_energy_j",  # intentionally bad
    ]
    violations = validate_metric_field_names(fields)
    assert "normalized_delay_sec" in violations
    assert "normalized_energy_j" in violations


def test_metrics_csv_contains_physical_and_normalized_columns(tmp_path) -> None:
    cfg = TrainConfig(
        total_episodes=1,
        output_dir=str(tmp_path),
        scenario=ScenarioConfig(n_leo=4, episode_len=4, seed=17),
        algo=AlgoConfig(gnn_hidden_dim=16, policy_hidden_dim=32, lower_batch_size=4, lower_warmup=4),
    )
    history = HierarchicalTrainer(cfg).train()
    assert len(history) == 1
    with open(tmp_path / "metrics.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    for field in (
        "metric_schema_version",
        "mean_deadline_exceedance",
        "mean_deadline_violation_ratio",
        "mean_delay_s",
        "p95_delay_s",
        "mean_energy_j",
        "mean_queue_length_tasks",
        "normalized_system_cost",
        "reward_mean",
    ):
        assert field in row
    assert row["metric_schema_version"] == METRIC_SCHEMA_VERSION


def test_deadline_exceedance_and_violation_ratio_are_distinct() -> None:
    cfg = TrainConfig(
        scenario=ScenarioConfig(n_leo=4, episode_len=1, seed=23, deadline_threshold=0.01),
        algo=AlgoConfig(gnn_hidden_dim=16, policy_hidden_dim=32, lower_batch_size=4, lower_warmup=4),
    )
    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, torch.device("cpu"))
    env.reset()
    upper = torch.zeros(cfg.scenario.n_leo, dtype=torch.long)
    lower = torch.ones((cfg.scenario.n_leo, 3), dtype=torch.float32) * 0.5
    step = env.step(upper, lower)

    delay = step.info["delay"].float()
    expected_exceedance = torch.relu(delay - cfg.scenario.deadline_threshold)
    expected_flag = (delay > cfg.scenario.deadline_threshold).float()
    assert torch.allclose(step.info["deadline_exceedance"].float(), expected_exceedance)
    assert torch.equal(step.info["deadline_violation_flag"].float(), expected_flag)
    assert float(step.info["deadline_violation_flag"].max()) <= 1.0


def test_trace_delay_anomaly_is_not_exported_as_physical_seconds() -> None:
    bundle = build_step_metric_bundle(
        delay_units=torch.tensor([100.0]),
        energy_units=torch.tensor([1.0]),
        queue_units=torch.tensor([2.0]),
        normalized_cost=torch.tensor([3.0]),
        reward=torch.tensor([-3.0]),
        trace_delay_anomaly_mask=torch.tensor([True]),
        units=UnitScaleConfig(),
    )
    assert float(bundle.physical_delay_s.item()) == 0.0
    assert float(bundle.legacy_trace_delay_score.item()) == 100.0
    assert float(bundle.physical_queue_length_tasks.item()) == 2.0


def test_aggregate_results_rejects_ambiguous_system_cost_alias(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "tables"
    input_root.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_results.py"),
            "--input-root",
            str(input_root),
            "--output",
            str(output_root),
            "--metric",
            "final_mean_system_cost",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert "deprecated and ambiguous" in (result.stderr + result.stdout)
