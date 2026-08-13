from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.analysis.statistical_schema import StatisticalSchemaError, read_records
from trisatflow.analysis.statistical_tests import run_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready checkpoint-level statistical tests.")
    parser.add_argument("--input", type=str, required=True, help="Standard long-format CSV/JSON statistical records.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    try:
        records = read_records(Path(args.input))
    except StatisticalSchemaError as exc:
        print(f"STATISTICAL_SCHEMA_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    result = run_protocol(
        records,
        output_dir=Path(args.output_dir),
        metric=args.metric or None,
        lower_is_better=not args.higher_is_better,
        n_boot=int(args.bootstrap_samples),
    )
    compact = {
        "status": "ok",
        "records": len(records),
        "pairwise_tests": len(result["pairwise_tests"]),
        "claim_allowed": result["claim_guard"].get("claim_allowed"),
        "best_pair_selection_basis": result["claim_guard"].get("best_pair_selection_basis"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
