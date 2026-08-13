from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from trisatflow.reporting.input_validation import ReportingInputError, load_reporting_input


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_summary(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sweep_summary.csv"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(method: str, seed: int, *, phase: str = "test", contract: str = "contract-a", **extra):
    baseline = method if method.startswith("flat_") or method == "hierarchical_no_gnn" else ""
    upper = "" if baseline else method
    row = {
        "status": "ok",
        "phase": phase,
        "protocol_role": phase,
        "seed": seed,
        "train_seed": seed,
        "eval_seed": 202,
        "eval_seed_bank": "202,303",
        "n_eval_seeds": 2,
        "upper_algo": upper,
        "lower_algo": "maddpg" if upper else "flat_resource_head",
        "baseline": baseline,
        "experiment_contract_sha256": contract,
        "metric_schema_version": "3.0",
        "observation_mode": "safe_observable",
        "include_oracle_cost": 0,
        "include_cost_prior_features": 0,
        "final_normalized_system_cost": 1.0 + 0.1 * seed + (0.2 if method == "flat_mappo" else 0.0),
        "mean_delay_s": 2.0 + seed * 0.1,
        "mean_energy_j": 0.5 + seed * 0.01,
        "upper_local_ratio": 0.25,
        "upper_neighbor_ratio": 0.0,
        "upper_geo_ratio": 0.5,
        "upper_ground_ratio": 0.25,
        "upper_remote_ratio": 0.75,
    }
    row.update(extra)
    return row


def _formal_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ("flat_ppo", "flat_mappo", "hierarchical_no_gnn"):
        for seed in (13, 42, 57):
            rows.append(_row(method, seed))
    return rows


def test_mixed_contract_fingerprint_input_fails(tmp_path: Path) -> None:
    rows = _formal_rows()
    rows[0]["experiment_contract_sha256"] = "contract-b"
    _write_summary(tmp_path, rows)

    with pytest.raises(ReportingInputError, match="mixed experiment_contract_sha256"):
        load_reporting_input(tmp_path)


def test_debug_or_oracle_input_fails(tmp_path: Path) -> None:
    rows = _formal_rows()
    rows[0]["observation_mode"] = "oracle_debug"
    _write_summary(tmp_path, rows)

    with pytest.raises(ReportingInputError, match="oracle/debug"):
        load_reporting_input(tmp_path)


def test_placeholder_input_fails(tmp_path: Path) -> None:
    rows = _formal_rows()
    rows[0]["baseline"] = "hmadrl_maddqn_ddpg"
    rows[0]["upper_algo"] = ""
    _write_summary(tmp_path, rows)

    with pytest.raises(ReportingInputError, match="placeholder"):
        load_reporting_input(tmp_path)


def test_deprecated_mean_system_cost_input_fails(tmp_path: Path) -> None:
    row = _row("flat_ppo", 13)
    row.pop("final_normalized_system_cost")
    row["mean_system_cost"] = 1.2
    _write_summary(tmp_path, [row])

    with pytest.raises(ReportingInputError, match="deprecated metric"):
        load_reporting_input(tmp_path, allow_smoke_small_n=True)


def test_smoke_small_n_requires_explicit_opt_in(tmp_path: Path) -> None:
    _write_summary(tmp_path, [_row("flat_ppo", 13, phase="train", eval_seed="", eval_seed_bank="", n_eval_seeds=0)])

    with pytest.raises(ReportingInputError, match="too few independent train seeds"):
        load_reporting_input(tmp_path)

    report = load_reporting_input(tmp_path, allow_smoke_small_n=True)
    assert len(report.rows) == 1
    assert report.contract_sha256 == "contract-a"


def test_missing_statistical_columns_fail_when_stats_file_present(tmp_path: Path) -> None:
    _write_summary(tmp_path, _formal_rows())
    with (tmp_path / "significance_tests.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method_a", "method_b", "p_value_raw"])
        writer.writeheader()
        writer.writerow({"method_a": "a", "method_b": "b", "p_value_raw": "0.5"})

    with pytest.raises(ReportingInputError, match="missing required statistical columns"):
        load_reporting_input(tmp_path)


def test_export_tables_and_plots_generate_required_artifacts(tmp_path: Path) -> None:
    _write_summary(tmp_path, _formal_rows())
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"

    table_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_paper_tables.py"),
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(tables),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert "PAPER_TABLES_OK" in table_result.stdout

    plot_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_paper_results.py"),
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(figures),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert "PAPER_FIGURES_OK" in plot_result.stdout

    for name in (
        "table_ii_hierarchical_rl_component_selection",
        "table_iii_best_rl_vs_rule_based_baselines",
        "table_iv_learning_based_and_literature_baselines",
        "table_s1_complete_raw_seed_level_results",
        "table_s2_statistical_tests_holm",
    ):
        assert (tables / f"{name}.csv").is_file()
        assert (tables / f"{name}.tex").is_file()
        assert (tables / f"{name}.csv").stat().st_size > 0
        assert (tables / f"{name}.tex").stat().st_size > 0

    for name in (
        "fig_rl_component_selection",
        "fig_rl_delay_energy_tradeoff",
        "fig_rule_baseline_comparison",
        "fig_policy_adaptivity_by_topology_phase",
        "fig_pairwise_cost_difference_forest",
    ):
        assert (figures / f"{name}.pdf").is_file()
        assert (figures / f"{name}.png").is_file()
        assert (figures / f"{name}.pdf").stat().st_size > 0
        assert (figures / f"{name}.png").stat().st_size > 0
