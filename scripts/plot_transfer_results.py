from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def summarize_transfer_results(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_rows": len(rows),
        "transfer_claim_supported": False,
        "reason": "This helper only summarizes stress CSVs; it does not prove checkpoint transfer.",
        "available_scales": sorted({str(r.get("n_leo")) for r in rows}),
        "blocked_rows": [r for r in rows if str(r.get("transfer_blocked")).lower() == "true"],
    }
    with (output_dir / "transfer_plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "transfer_results_readme.txt").open("w", encoding="utf-8") as f:
        f.write("Use this summary as a plotting input only. Do not claim inductive transfer without checkpoint forward/evaluation on target n_leo.\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_transfer_results(args.input, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

