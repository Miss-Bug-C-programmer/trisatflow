from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trisatflow.data.trace_manifest import audit_manifest_records, load_manifest_file, load_manifest_dir


def _active_manifest_files(manifest_dir: Path) -> list[str]:
    summary_path = manifest_dir / "manifest_build_summary.json"
    if not summary_path.exists():
        return []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        values = data.get("active_manifest_files")
        return [str(v) for v in values] if isinstance(values, list) else []
    except Exception:
        return []


def audit_manifest_dir(manifest_dir: Path, output_dir: Path) -> Dict[str, Any]:
    active = _active_manifest_files(manifest_dir)
    if active:
        records = [load_manifest_file(manifest_dir / name) for name in active if (manifest_dir / name).exists()]
    else:
        records = load_manifest_dir(manifest_dir)
    summary = audit_manifest_records(records)
    if active:
        summary["active_manifest_files_used"] = active
        summary["stale_manifest_files_ignored"] = [
            item.name
            for item in sorted(manifest_dir.glob("*.json"))
            if item.name not in set(active) and item.name != "manifest_build_summary.json"
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "trace_split_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "code", "trace_id", "message"])
        writer.writeheader()
        for issue in summary.get("issues", []):
            writer.writerow(issue)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_manifest_dir(args.manifest_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("audit_status") == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
