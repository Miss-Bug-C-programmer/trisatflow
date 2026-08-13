from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_results  # noqa: E402
import statistical_tests  # noqa: E402
from trisatflow.analysis import statistical_tests as protocol_stats  # noqa: E402
from trisatflow.analysis.statistical_schema import StatisticalSchemaError, normalize_records  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_split_seed_overlap_is_blocked(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--config",
        "trisatflow/configs/small.yaml",
        "--upper",
        "mappo",
        "--lower",
        "maddpg",
        "--episodes",
        "1",
        "--steps",
        "2",
        "--n-leo",
        "4",
        "--train-seeds",
        "13",
        "--val-seeds",
        "13",
        "--device",
        "cpu",
        "--output-root",
        str(tmp_path / "out"),
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    combined = (p.stdout + "\n" + p.stderr).lower()
    assert "overlap" in combined


def test_print_registry_does_not_start_sweep(tmp_path: Path) -> None:
    out_root = tmp_path / "registry_out"
    cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--print-registry",
        "--output-root",
        str(out_root),
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert "upper_discrete_offloading" in p.stdout
    assert not out_root.exists()


def test_split_sweep_expands_all_upper_lower_combinations(tmp_path: Path) -> None:
    out_root = tmp_path / "split_sweep"
    cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--config",
        "trisatflow/configs/small.yaml",
        "--upper",
        "mappo,ippo",
        "--lower",
        "maddpg,masac",
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
        str(out_root),
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr
    assert '"event": "SWEEP_PLAN"' in p.stdout

    expected = {
        ("mappo", "maddpg"),
        ("mappo", "masac"),
        ("ippo", "maddpg"),
        ("ippo", "masac"),
    }
    for upper, lower in expected:
        assert (out_root / "train" / "seed_13" / f"upper_{upper}__lower_{lower}" / "checkpoint.pt").exists()
        assert (out_root / f"protocol_{upper}_{lower}.json").exists()

    rows = _read_csv(out_root / "sweep_summary.csv")
    assert {(r["upper_algo"], r["lower_algo"]) for r in rows if r["phase"] == "train"} == expected
    assert {(r["upper_algo"], r["lower_algo"]) for r in rows if r["phase"] == "val"} == expected
    assert {(r["upper_algo"], r["lower_algo"]) for r in rows if r["phase"] == "test"} == expected


def test_aggregate_results_outputs_ci_and_significance(tmp_path: Path) -> None:
    input_root = tmp_path / "debug_stats"
    input_root.mkdir(parents=True, exist_ok=True)

    summary_path = input_root / "sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "phase",
                "seed",
                "upper_algo",
                "lower_algo",
                "baseline",
                "observation_ablation",
                "final_normalized_system_cost",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "status": "ok",
                    "phase": "test",
                    "seed": "13",
                    "upper_algo": "mappo",
                    "lower_algo": "maddpg",
                    "baseline": "",
                    "observation_ablation": "",
                    "final_normalized_system_cost": "10.0",
                    "checkpoint": "ckpt_a_13.pt",
                },
                {
                    "status": "ok",
                    "phase": "test",
                    "seed": "21",
                    "upper_algo": "mappo",
                    "lower_algo": "maddpg",
                    "baseline": "",
                    "observation_ablation": "",
                    "final_normalized_system_cost": "14.0",
                    "checkpoint": "ckpt_a_21.pt",
                },
                {
                    "status": "ok",
                    "phase": "test",
                    "seed": "13",
                    "upper_algo": "ippo",
                    "lower_algo": "maddpg",
                    "baseline": "",
                    "observation_ablation": "",
                    "final_normalized_system_cost": "11.0",
                    "checkpoint": "ckpt_b_13.pt",
                },
                {
                    "status": "ok",
                    "phase": "test",
                    "seed": "21",
                    "upper_algo": "ippo",
                    "lower_algo": "maddpg",
                    "baseline": "",
                    "observation_ablation": "",
                    "final_normalized_system_cost": "15.0",
                    "checkpoint": "ckpt_b_21.pt",
                },
            ]
        )

    out_dir = input_root / "summary"
    cmd = [
        sys.executable,
        "scripts/aggregate_results.py",
        "--input-root",
        str(input_root),
        "--output",
        str(out_dir),
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr

    algo_csv = out_dir / "summary_by_algorithm.csv"
    ablation_csv = out_dir / "summary_by_ablation.csv"
    sig_csv = out_dir / "significance_tests.csv"
    assert algo_csv.exists()
    assert ablation_csv.exists()
    assert sig_csv.exists()

    algo_rows = _read_csv(algo_csv)
    assert len(algo_rows) >= 2
    row0 = algo_rows[0]
    assert "mean" in row0
    assert "std" in row0
    assert "standard_error" in row0
    assert "ci95_low" in row0
    assert "ci95_high" in row0
    assert "n_seeds" in row0

    sig_rows = _read_csv(sig_csv)
    assert len(sig_rows) >= 1
    for field in (
        "metric",
        "method_a",
        "method_b",
        "n_independent_train_seeds",
        "mean_difference",
        "relative_difference_pct",
        "ci95_low",
        "ci95_high",
        "t_statistic",
        "p_value_raw",
        "p_value_holm",
        "cohens_dz",
        "practical_threshold_pct",
        "practically_significant",
    ):
        assert field in sig_rows[0]


def test_aggregate_results_can_fallback_to_metrics_csv(tmp_path: Path) -> None:
    input_root = tmp_path / "debug_stats_fallback"
    run_dir = input_root / "seed_13" / "upper_mappo__lower_maddpg"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "normalized_system_cost", "mean_delay_s"])
        writer.writeheader()
        writer.writerow({"episode": 1, "normalized_system_cost": 12.5, "mean_delay_s": 4.0})

    out_dir = input_root / "summary"
    cmd = [
        sys.executable,
        "scripts/aggregate_results.py",
        "--input-root",
        str(input_root),
        "--output",
        str(out_dir),
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr

    algo_rows = _read_csv(out_dir / "summary_by_algorithm.csv")
    assert len(algo_rows) == 1
    assert algo_rows[0]["upper_algo"] == "mappo"
    assert algo_rows[0]["lower_algo"] == "maddpg"


def test_student_t_ci_differs_from_normal_approximation_for_three_seeds() -> None:
    values = [1.0, 2.0, 5.0]
    stats = aggregate_results._stats(values)
    mean_v = sum(values) / len(values)
    std_v = math.sqrt(sum((x - mean_v) ** 2 for x in values) / (len(values) - 1))
    stderr = std_v / math.sqrt(len(values))
    normal_low = mean_v - 1.96 * stderr

    assert not math.isclose(float(stats["ci95_low"]), normal_low, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_paired_t_test_matches_scipy_ttest_rel() -> None:
    lhs = [10.0, 14.0, 11.5, 9.0]
    rhs = [11.0, 13.0, 13.5, 12.0]
    stats = statistical_tests._paired_t_test(lhs, rhs)
    diffs = [a - b for a, b in zip(lhs, rhs)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1))
    expected_t = mean_diff / (std_diff / math.sqrt(len(diffs)))

    assert math.isclose(float(stats["t_stat"]), expected_t, rel_tol=1.0e-12)
    assert 0.0 <= float(stats["p_value"]) <= 1.0


def test_holm_adjusted_p_values_are_monotonic_and_not_below_raw() -> None:
    raw = [0.04, 0.01, 0.03]
    adjusted = statistical_tests.holm_adjust(raw)
    sorted_pairs = sorted(zip(raw, adjusted), key=lambda item: item[0])

    assert all(adj >= p for p, adj in zip(raw, adjusted))
    assert [adj for _p, adj in sorted_pairs] == sorted([adj for _p, adj in sorted_pairs])


def test_zero_variance_differences_are_handled() -> None:
    same = statistical_tests._paired_t_test([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    shifted = statistical_tests._paired_t_test([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])

    assert same["t_stat"] == 0.0
    assert same["p_value"] == 1.0
    assert same["cohens_dz"] == 0.0
    assert math.isinf(float(shifted["t_stat"]))
    assert shifted["p_value"] == 0.0


def test_small_p_value_format_uses_scientific_not_zero_decimal() -> None:
    assert statistical_tests._format_p_value(1.234e-8) == "1.234e-08"
    assert statistical_tests._format_p_value(0.0) == "<1e-300"


def test_significance_uses_seed_level_pairs_not_episode_pooling() -> None:
    rows = [
        {"phase": "train", "seed": 1, "value": 10.0, "upper_algo": "a", "lower_algo": "x"},
        {"phase": "train", "seed": 1, "value": 100.0, "upper_algo": "a", "lower_algo": "x"},
        {"phase": "train", "seed": 2, "value": 20.0, "upper_algo": "a", "lower_algo": "x"},
        {"phase": "train", "seed": 2, "value": 200.0, "upper_algo": "a", "lower_algo": "x"},
        {"phase": "train", "seed": 1, "value": 11.0, "upper_algo": "b", "lower_algo": "x"},
        {"phase": "train", "seed": 1, "value": 101.0, "upper_algo": "b", "lower_algo": "x"},
        {"phase": "train", "seed": 2, "value": 21.0, "upper_algo": "b", "lower_algo": "x"},
        {"phase": "train", "seed": 2, "value": 201.0, "upper_algo": "b", "lower_algo": "x"},
    ]
    sig_rows = statistical_tests.build_significance_rows(rows, metric="normalized_system_cost")

    assert len(sig_rows) == 1
    assert sig_rows[0]["n_independent_train_seeds"] == 2
    assert sig_rows[0]["shared_seeds"] == "1,2"


def _protocol_fixture(test_seeds: list[int] | None = None) -> list[dict[str, object]]:
    test_seeds = test_seeds or [101, 102, 103]
    rows: list[dict[str, object]] = []
    values = {
        "IPPO+MADDPG": [10.0, 11.0, 9.0, 10.5],
        "MAPPO+MADDPG": [10.2, 10.8, 9.1, 10.6],
    }
    for method, vals in values.items():
        for idx, train_seed in enumerate([11, 22, 33, 44]):
            for test_seed in test_seeds:
                rows.append(
                    {
                        "method": method,
                        "metric": "normalized_system_cost",
                        "value": vals[idx] + 0.01 * (test_seed - 102),
                        "train_seed": train_seed,
                        "checkpoint_id": f"{method}_ckpt_{train_seed}",
                        "test_seed": test_seed,
                        "split": "offline",
                        "statistical_unit": "train_seed_checkpoint",
                        "source_file": "synthetic",
                    }
                )
    return rows


def test_test_seed_expansion_does_not_change_effective_pairs() -> None:
    records = normalize_records(_protocol_fixture(test_seeds=[101, 102, 103]))
    pairwise = protocol_stats.pairwise_tests(records, n_boot=100)

    assert len(pairwise) == 1
    assert pairwise[0]["n_rows"] == 24
    assert pairwise[0]["n_effective_pairs"] == 4
    assert pairwise[0]["statistical_unit"] == "train_seed_checkpoint"


def test_protocol_holm_correction_is_monotone_and_not_below_raw() -> None:
    adjusted = protocol_stats.holm_bonferroni([0.04, 0.01, 0.03])
    sorted_pairs = sorted(zip([0.04, 0.01, 0.03], adjusted), key=lambda item: item[0])

    assert all(adj >= raw for raw, adj in zip([0.04, 0.01, 0.03], adjusted))
    assert [adj for _raw, adj in sorted_pairs] == sorted(adj for _raw, adj in sorted_pairs)


def test_no_significant_result_downgrades_claim_guard(tmp_path: Path) -> None:
    records = normalize_records(_protocol_fixture())
    result = protocol_stats.run_protocol(
        records,
        output_dir=tmp_path / "stats",
        metric="normalized_system_cost",
        n_boot=100,
    )
    guard = result["claim_guard"]

    assert guard["holm_significant_best"] is False
    assert guard["best_pair_selection_basis"] == "mean_rank_reference_only"
    assert guard["claim_allowed"] == "mean-ranked reference only; statistically comparable"
    assert "significantly outperforms" in guard["forbidden_phrases"]


def test_missing_checkpoint_id_fails_actionably() -> None:
    rows = _protocol_fixture()
    rows[0] = dict(rows[0])
    rows[0]["checkpoint_id"] = ""

    with pytest.raises(StatisticalSchemaError, match="checkpoint_id"):
        normalize_records(rows)


def test_missing_paired_seed_is_recorded_as_warning() -> None:
    rows = _protocol_fixture()
    filtered = [
        row for row in rows
        if not (row["method"] == "MAPPO+MADDPG" and row["train_seed"] == 44)
    ]
    pairwise = protocol_stats.pairwise_tests(normalize_records(filtered), n_boot=100)

    assert pairwise[0]["n_effective_pairs"] == 3
    assert "missing_paired_units" in pairwise[0]["warning"]


def test_cluster_bootstrap_outputs_valid_ci() -> None:
    low, high = protocol_stats.cluster_bootstrap_ci({"a": 0.1, "b": -0.2, "c": 0.3}, n_boot=100)

    assert math.isfinite(low)
    assert math.isfinite(high)
    assert low <= high


def test_small_n_warning_and_forbidden_phrases_present() -> None:
    pairwise = protocol_stats.pairwise_tests(normalize_records(_protocol_fixture(test_seeds=[101])), n_boot=100)
    summary = protocol_stats.method_summary(normalize_records(_protocol_fixture(test_seeds=[101])), n_boot=100)
    guard = protocol_stats.build_claim_guard(pairwise, summary, metric="normalized_system_cost")

    assert "small_n_warning" in pairwise[0]["warning"]
    assert guard["small_n_warning"] is True
    assert {"clearly best", "statistically superior", "dominates"}.issubset(set(guard["forbidden_phrases"]))


def test_online_seed_clustered_by_checkpoint_not_row_count() -> None:
    rows: list[dict[str, object]] = []
    for method, offset in [("A", 0.0), ("B", 0.2)]:
        for train_seed in [1, 2]:
            for online_seed in range(10):
                rows.append(
                    {
                        "method": method,
                        "metric": "success_rate",
                        "value": 0.8 + offset + 0.01 * train_seed + 0.001 * online_seed,
                        "train_seed": train_seed,
                        "checkpoint_id": f"{method}_ckpt_{train_seed}",
                        "online_seed": online_seed,
                        "split": "online",
                        "statistical_unit": "train_seed_checkpoint_cluster",
                        "source_file": "synthetic_online",
                    }
                )
    records = normalize_records(rows)
    pairwise = protocol_stats.pairwise_tests(records, n_boot=100)
    summary = protocol_stats.method_summary(records, n_boot=100)

    assert pairwise[0]["n_rows"] == 40
    assert pairwise[0]["n_effective_pairs"] == 2
    assert pairwise[0]["statistical_unit"] == "train_seed_checkpoint_cluster"
    assert {row["n_online_seeds"] for row in summary} == {10}
