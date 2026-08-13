from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.config import load_config
from trisatflow.experiment_contracts import (
    assert_paper_safe,
    assert_same_contract,
    contract_sha256,
    resolve_contract,
    trace_sha256_for_config,
)


def _read_manifest_contract(path: str) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    digest = str(data.get("experiment_contract_sha256", "") or "").strip()
    artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    contract_path_raw = str(artifacts.get("experiment_contract", "") or "").strip()
    if not digest:
        digest_path_raw = str(artifacts.get("experiment_contract_sha256", "") or "").strip()
        if digest_path_raw:
            digest_path = Path(digest_path_raw)
            if not digest_path.is_absolute() and not digest_path.exists():
                digest_path = manifest_path.parent / digest_path
            digest = digest_path.read_text(encoding="utf-8").strip()
    if not contract_path_raw:
        raise ValueError(f"manifest missing artifacts.experiment_contract: {manifest_path}")
    contract_path = Path(contract_path_raw)
    if not contract_path.is_absolute() and not contract_path.exists():
        contract_path = manifest_path.parent / contract_path
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError(f"experiment contract must be a JSON object: {contract_path}")
    computed = contract_sha256(contract)
    if digest and digest != computed:
        raise ValueError(f"manifest digest mismatch for {manifest_path}: manifest={digest} computed={computed}")
    return contract, computed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the resolved TriSatFlow experiment contract.",
    )
    parser.add_argument("--config", default="", help="Train config YAML to audit.")
    parser.add_argument("--require-paper-safe", action="store_true", help="Reject oracle/debug/cost-prior contract fields.")
    parser.add_argument("--output-json", default="", help="Optional path for the resolved contract payload.")
    parser.add_argument("--compare", action="store_true", help="Compare two manifest experiment contracts.")
    parser.add_argument("manifests", nargs="*", help="Manifest paths used with --compare.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.compare:
        if len(args.manifests) != 2:
            raise SystemExit("--compare requires exactly two manifest paths")
        lhs, lhs_digest = _read_manifest_contract(args.manifests[0])
        rhs, rhs_digest = _read_manifest_contract(args.manifests[1])
        assert_same_contract(lhs, rhs)
        print(
            "AUDIT_EXPERIMENT_CONTRACT_COMPARE_OK "
            f"lhs={args.manifests[0]} rhs={args.manifests[1]} "
            f"experiment_contract_sha256={lhs_digest}"
        )
        return

    if not args.config:
        raise SystemExit("--config is required unless --compare is used")
    cfg = load_config(args.config)
    if args.require_paper_safe:
        assert_paper_safe(cfg)

    contract = resolve_contract(cfg, trace_sha256_for_config(cfg, base_dir=repo_root))
    digest = contract_sha256(contract)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "AUDIT_EXPERIMENT_CONTRACT_OK "
        f"config={args.config} "
        f"paper_safe={bool(args.require_paper_safe)} "
        f"experiment_contract_sha256={digest}"
    )


if __name__ == "__main__":
    main()
