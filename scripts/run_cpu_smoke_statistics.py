from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ["IPPO+MADDPG", "MAPPO+MADDPG", "IPPO+MASAC", "MAPPO+MASAC"]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_synthetic_fixture(path: Path) -> List[Dict[str, Any]]:
    base = {
        "IPPO+MADDPG": [10.00, 11.00, 9.00, 10.50],
        "MAPPO+MADDPG": [10.20, 10.80, 9.10, 10.60],
        "IPPO+MASAC": [10.10, 11.20, 9.30, 10.40],
        "MAPPO+MASAC": [10.40, 11.10, 9.20, 10.70],
    }
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        for train_idx, train_seed in enumerate([11, 22, 33, 44]):
            for test_seed in [101, 102, 103]:
                rows.append(
                    {
                        "method": method,
                        "metric": "normalized_system_cost",
                        "value": base[method][train_idx] + (test_seed - 102) * 0.03,
                        "train_seed": train_seed,
                        "checkpoint_id": f"{method.replace('+', '_')}_seed_{train_seed}",
                        "test_seed": test_seed,
                        "online_seed": "",
                        "scenario_id": "synthetic_offline",
                        "split": "offline",
                        "statistical_unit": "train_seed_checkpoint",
                        "source_file": str(path),
                    }
                )
    _write_csv(path, rows)
    return rows


def repository_fixture_audit() -> Dict[str, Any]:
    candidates = [
        REPO_ROOT / "outputs" / "paper_ready_v3" / "sweep_summary.csv",
        REPO_ROOT / "outputs" / "paper_ready_v3" / "summary_matrix.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            rows = _read_csv(candidate)
            return {
                "real_fixture_available": True,
                "source": str(candidate),
                "rows_scanned": min(len(rows), 25),
                "has_train_seed": any("train_seed" in row and str(row.get("train_seed", "")).strip() for row in rows[:25]),
                "has_checkpoint_id": any("checkpoint_id" in row and str(row.get("checkpoint_id", "")).strip() for row in rows[:25]),
            }
    return {"real_fixture_available": False, "source": "", "rows_scanned": 0}


def run_smoke(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "synthetic_statistical_records.csv"
    build_synthetic_fixture(fixture_path)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_statistical_tests.py"),
        "--input",
        str(fixture_path),
        "--output-dir",
        str(output_dir),
        "--metric",
        "normalized_system_cost",
        "--bootstrap-samples",
        "400",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
    pairwise = _read_csv(output_dir / "pairwise_tests.csv")
    method_summary = _read_csv(output_dir / "method_summary.csv")
    guard = json.loads((output_dir / "claim_guard.json").read_text(encoding="utf-8"))
    summary = {
        "status": "ok",
        "synthetic_fixture_rows": len(_read_csv(fixture_path)),
        "pairwise_rows": len(pairwise),
        "method_summary_rows": len(method_summary),
        "test_seed_expansion_preserved_effective_pairs": all(int(float(row["n_effective_pairs"])) == 4 for row in pairwise),
        "claim_guard": guard,
        "repository_fixture_audit": repository_fixture_audit(),
        "outputs": {
            "records": str(fixture_path),
            "pairwise_tests": str(output_dir / "pairwise_tests.csv"),
            "method_summary": str(output_dir / "method_summary.csv"),
            "claim_guard": str(output_dir / "claim_guard.json"),
            "report": str(output_dir / "statistical_protocol_report.md"),
        },
    }
    report_src = output_dir / "statistical_protocol_report.md"
    docs_report = REPO_ROOT / "docs" / "statistical_protocol_report.md"
    try:
        docs_report.parent.mkdir(parents=True, exist_ok=True)
        docs_report.write_text(report_src.read_text(encoding="utf-8"), encoding="utf-8")
        summary["docs_report_synced"] = True
        summary["docs_report_path"] = str(docs_report)
    except PermissionError as exc:
        summary["docs_report_synced"] = False
        summary["docs_report_path"] = str(docs_report)
        summary["docs_report_warning"] = f"permission_denied: {exc}"
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke for checkpoint-level statistical protocol.")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "reviewer_repair" / "statistics"))
    args = parser.parse_args()
    summary = run_smoke(Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
