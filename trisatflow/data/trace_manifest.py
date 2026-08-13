from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from trisatflow.data.trace_hashing import hash_mapping

TRACE_SOURCES = {"sgp4_generated", "satedgesim_export", "synthetic", "legacy"}
TRACE_SPLITS = {"train", "validation", "test", "satedgesim_online", "stress"}

REQUIRED_FIELDS = [
    "trace_id",
    "trace_path",
    "source",
    "generation_seed",
    "n_leo",
    "n_geo",
    "n_ground",
    "duration_s",
    "slot_duration_s",
    "scenario_profile",
    "split",
    "sha256",
    "content_fingerprint",
    "contains_oracle_fields",
    "oracle_field_names",
    "safe_observable_excludes_oracle_fields",
    "generator_config_hash",
    "created_at",
    "notes",
]

OPTIONAL_FIELDS = {
    "tle_epoch",
    "time_start_s",
    "time_end_s",
    "status",
    "same_distribution_non_overlapping",
    "split_independence_evidence",
    "non_overlap_evidence",
    "domain_shift",
    "source_file",
    "manifest_schema_version",
    "trace_semantic_class",
    "trace_generation_mode",
    "settings_sha256",
    "num_steps",
    "num_rows",
    "metadata_derivation",
}

ORACLE_FIELD_HINTS = (
    "completion_safe",
    "mobility_risk",
    "handover",
    "link_margin",
    "lifetime",
    "future",
    "oracle",
)


@dataclass
class TraceAuditIssue:
    severity: str
    code: str
    message: str
    trace_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "trace_id": self.trace_id,
        }


def normalize_manifest_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    if "generation_seed" not in out and "seed" in out:
        out["generation_seed"] = out.get("seed")
    if "sha256" not in out and "trace_sha256" in out:
        out["sha256"] = out.get("trace_sha256")
    if "source" not in out and str(out.get("trace_origin", "")).lower() == "satedgesim":
        out["source"] = "satedgesim_export"
    if "source" not in out:
        out["source"] = "synthetic" if bool(out.get("synthetic", False)) else "legacy"
    if "trace_path" not in out and "path" in out:
        out["trace_path"] = out.get("path")
    if "trace_id" not in out:
        out["trace_id"] = str(out.get("trace_path") or out.get("path") or out.get("sha256") or "unknown")
    if "scenario_profile" not in out:
        out["scenario_profile"] = out.get("profile_name") or out.get("scenario", {}).get("profile_name") or "unknown"
    if "split" not in out:
        out["split"] = infer_split(str(out.get("trace_path", "")))
    if "slot_duration_s" not in out:
        out["slot_duration_s"] = 1.0
    if "n_geo" not in out:
        out["n_geo"] = 1
    if "n_ground" not in out:
        out["n_ground"] = 1
    if "duration_s" not in out:
        steps = out.get("num_steps") or out.get("num_decision_steps")
        out["duration_s"] = float(steps) * float(out.get("slot_duration_s", 1.0)) if steps is not None else None
    if "contains_oracle_fields" not in out:
        names = list(out.get("oracle_field_names") or detect_oracle_field_names(out))
        out["oracle_field_names"] = names
        out["contains_oracle_fields"] = bool(names)
    if "oracle_field_names" not in out:
        out["oracle_field_names"] = []
    if "safe_observable_excludes_oracle_fields" not in out:
        out["safe_observable_excludes_oracle_fields"] = not bool(out.get("contains_oracle_fields", False))
    if "generator_config_hash" not in out:
        basis = {
            "source": out.get("source"),
            "generation_seed": out.get("generation_seed"),
            "scenario_profile": out.get("scenario_profile"),
            "n_leo": out.get("n_leo"),
            "trace_generation_mode": out.get("trace_generation_mode") or out.get("mode"),
        }
        out["generator_config_hash"] = hash_mapping(basis)
    if "created_at" not in out:
        out["created_at"] = "unknown"
    if "notes" not in out:
        out["notes"] = ""
    out.setdefault("manifest_schema_version", "trisatflow_trace_manifest_v1")
    return out


def infer_split(path_text: str) -> str:
    text = path_text.replace("\\", "/").lower()
    if "/validation/" in text or "/val/" in text:
        return "validation"
    if "/test/" in text:
        return "test"
    if "online" in text or "sequential_live" in text:
        return "satedgesim_online"
    if "stress" in text:
        return "stress"
    if "/train/" in text:
        return "train"
    return "stress"


def detect_oracle_field_names(obj: Any) -> List[str]:
    names: set[str] = set()

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(hint in lowered for hint in ORACLE_FIELD_HINTS):
                    names.add(key_text)
                visit(child, f"{path}.{key_text}" if path else key_text)
        elif isinstance(value, list):
            for child in value[:8]:
                visit(child, path)

    visit(obj)
    return sorted(names)


def missing_metadata(record: Mapping[str, Any]) -> List[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if value is None or value == "":
            missing.append(field)
    return missing


def validate_manifest_record(record: Mapping[str, Any]) -> List[str]:
    normalized = normalize_manifest_record(record)
    errors = missing_metadata(normalized)
    source = normalized.get("source")
    split = normalized.get("split")
    if source not in TRACE_SOURCES:
        errors.append(f"source must be one of {sorted(TRACE_SOURCES)}")
    if split not in TRACE_SPLITS:
        errors.append(f"split must be one of {sorted(TRACE_SPLITS)}")
    if bool(normalized.get("contains_oracle_fields")) and not bool(normalized.get("safe_observable_excludes_oracle_fields")):
        errors.append("safe_observable_excludes_oracle_fields must be true when oracle fields are present")
    return errors


def load_manifest_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_manifest_record(data)


def write_manifest_file(path: str | Path, record: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(normalize_manifest_record(record), f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest_dir(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [load_manifest_file(item) for item in sorted(p.glob("*.json")) if item.name != "manifest_build_summary.json"]


def audit_manifest_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = [normalize_manifest_record(r) for r in records]
    issues: List[TraceAuditIssue] = []
    evidence_rows: List[Dict[str, Any]] = []
    if not normalized:
        issues.append(TraceAuditIssue("high", "audit_failed_missing_manifest", "No trace manifest records were found."))

    for rec in normalized:
        tid = str(rec.get("trace_id", ""))
        missing = missing_metadata(rec)
        if missing:
            issues.append(TraceAuditIssue("high", "audit_status_failed_incomplete_metadata", f"Missing required metadata: {', '.join(missing)}", tid))
        if bool(rec.get("contains_oracle_fields")) and not bool(rec.get("safe_observable_excludes_oracle_fields")):
            issues.append(TraceAuditIssue("high", "oracle_leakage_risk", "Manifest says oracle fields may enter safe_observable.", tid))

    by_sha: Dict[str, List[Dict[str, Any]]] = {}
    for rec in normalized:
        sha = str(rec.get("sha256") or "")
        if sha and sha not in {"missing", "unknown"}:
            by_sha.setdefault(sha, []).append(rec)
    for sha, group in by_sha.items():
        splits = {str(r.get("split")) for r in group}
        if len(splits & {"train", "validation", "test", "satedgesim_online"}) > 1:
            ids = ", ".join(str(r.get("trace_id")) for r in group)
            issues.append(TraceAuditIssue("high", "duplicate_sha256_cross_split", f"Same sha256 appears across splits {sorted(splits)}: {ids}", sha[:12]))

    by_seed: Dict[str, List[Dict[str, Any]]] = {}
    for rec in normalized:
        seed = str(rec.get("generation_seed") or "")
        if seed:
            by_seed.setdefault(seed, []).append(rec)
    for seed, group in by_seed.items():
        splits = {str(r.get("split")) for r in group}
        if len(splits & {"train", "validation", "test", "satedgesim_online"}) > 1:
            issues.append(TraceAuditIssue("medium", "generation_seed_reused_cross_split", f"generation_seed={seed} appears in multiple splits: {sorted(splits)}", seed))

    def has_independent_split_evidence(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        if bool(a.get("same_distribution_non_overlapping")) or bool(b.get("same_distribution_non_overlapping")):
            return True
        if a.get("split_independence_evidence") or b.get("split_independence_evidence"):
            return True
        seed_a = a.get("generation_seed")
        seed_b = b.get("generation_seed")
        if seed_a is not None and seed_b is not None and str(seed_a) != str(seed_b):
            return True
        semantic_a = str(a.get("trace_semantic_class") or a.get("trace_generation_mode") or "")
        semantic_b = str(b.get("trace_semantic_class") or b.get("trace_generation_mode") or "")
        settings_a = str(a.get("settings_sha256") or a.get("generator_config_hash") or "")
        settings_b = str(b.get("settings_sha256") or b.get("generator_config_hash") or "")
        if semantic_a and semantic_b and semantic_a != semantic_b and settings_a and settings_b and settings_a != settings_b:
            return True
        return False

    def pair_id(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
        return f"{a.get('trace_id')}|{b.get('trace_id')}"

    for i, a in enumerate(normalized):
        for b in normalized[i + 1 :]:
            if str(a.get("split")) == str(b.get("split")):
                continue
            if str(a.get("scenario_profile")) != str(b.get("scenario_profile")) and not (bool(a.get("domain_shift")) or bool(b.get("domain_shift"))):
                continue
            sa = a.get("time_start_s")
            ea = a.get("time_end_s")
            sb = b.get("time_start_s")
            eb = b.get("time_end_s")
            if None in {sa, ea, sb, eb}:
                if has_independent_split_evidence(a, b):
                    evidence_rows.append(
                        {
                            "code": "independent_split_evidence_without_time_window",
                            "trace_id": pair_id(a, b),
                            "message": "No explicit time interval, but split independence is supported by seed/semantic/config evidence.",
                        }
                    )
                else:
                    issues.append(TraceAuditIssue("medium", "missing_time_nonoverlap_metadata", "Cannot prove time intervals are non-overlapping or independently generated.", pair_id(a, b)))
                continue
            if max(float(sa), float(sb)) < min(float(ea), float(eb)):
                if has_independent_split_evidence(a, b):
                    evidence_rows.append(
                        {
                            "code": "time_overlap_but_independent_generation_evidence",
                            "trace_id": pair_id(a, b),
                            "message": "Simulation time intervals overlap, but records differ by seed or semantic/config evidence; not treated as direct trace leakage.",
                        }
                    )
                else:
                    issues.append(TraceAuditIssue("high", "time_interval_overlap_cross_split", "Trace intervals overlap across splits without independent-generation evidence.", pair_id(a, b)))
            else:
                evidence_rows.append(
                    {
                        "code": "time_non_overlap_verified",
                        "trace_id": pair_id(a, b),
                        "message": "Trace intervals are explicitly non-overlapping.",
                    }
                )

    offline = [r for r in normalized if r.get("split") in {"train", "validation", "test"}]
    online = [r for r in normalized if r.get("split") == "satedgesim_online"]
    for on in online:
        for off in offline:
            if on.get("sha256") and on.get("sha256") == off.get("sha256"):
                issues.append(TraceAuditIssue("high", "satedgesim_online_duplicate_of_offline", "SatEdgeSim online replay trace duplicates an offline trace file.", str(on.get("trace_id"))))
            elif on.get("source") == off.get("source") and on.get("scenario_profile") == off.get("scenario_profile"):
                issues.append(TraceAuditIssue("low", "same_source_distribution_requires_nonoverlap_proof", "SatEdgeSim online and offline traces share source/profile; require explicit non-overlap proof.", str(on.get("trace_id"))))

    severities = {issue.severity for issue in issues}
    if "high" in severities:
        leakage_risk = "high"
    elif "medium" in severities:
        leakage_risk = "medium"
    elif "low" in severities:
        leakage_risk = "low"
    else:
        leakage_risk = "none"

    if any(i.code == "audit_failed_missing_manifest" for i in issues):
        status = "failed_missing_manifest"
    elif any(i.code == "audit_status_failed_incomplete_metadata" for i in issues):
        status = "failed_incomplete_metadata"
    elif any(i.severity == "high" for i in issues):
        status = "failed_leakage_risk"
    else:
        status = "passed"

    return {
        "audit_status": status,
        "leakage_risk": leakage_risk,
        "num_records": len(normalized),
        "issues": [i.to_dict() for i in issues],
        "evidence": evidence_rows,
        "same_distribution_non_overlapping_required": True,
        "real_trace_audit_required": True,
        "toy_manifest_is_not_evidence": True,
    }


def validate_trace_manifest_schema(record: Mapping[str, Any], *, strict_required: bool = False) -> List[str]:
    """Validate the compact formal trace manifest schema used by experiment gates.

    This is intentionally separate from the richer audit manifest above: formal
    runners only need stable split/hash/source/usage provenance, and smoke
    fixtures should not be forced to provide all exporter-only fields.
    """

    errors: List[str] = []
    if not isinstance(record, Mapping):
        return ["trace manifest record must be a mapping"]
    required = [
        "trace_id",
        "split",
        "sha256",
        "generator_version",
        "scenario_profile",
        "seed",
        "created_at",
        "source",
        "allowed_usage",
    ]
    if strict_required:
        for field in required:
            value = record.get(field)
            if value in (None, "") or (field == "allowed_usage" and not value):
                errors.append(f"{field} is required in formal trace manifest")
    split = str(record.get("split", "")).strip().lower()
    allowed_splits = {"train", "val", "validation", "test", "satedgesim_replay", "satedgesim_online", "stress"}
    if split and split not in allowed_splits:
        errors.append(f"split must be one of {sorted(allowed_splits)}, got {split!r}")
    sha = str(record.get("sha256", "")).strip().lower()
    if sha and (len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha)):
        errors.append("sha256 must be a 64-character hexadecimal digest")
    usage = record.get("allowed_usage")
    if usage is not None and (not isinstance(usage, list) or not all(isinstance(item, str) and item for item in usage)):
        errors.append("allowed_usage must be a non-empty list of usage strings")
    try:
        if "seed" in record and record.get("seed") is not None and int(record.get("seed")) < 0:
            errors.append("seed must be a non-negative integer")
    except (TypeError, ValueError):
        errors.append("seed must be a non-negative integer")
    return errors


def validate_trace_manifest_usage(
    manifests: Iterable[Mapping[str, Any]],
    *,
    usage: str,
    run_mode: str = "formal",
) -> Dict[str, Any]:
    """Validate whether trace manifests may be consumed for a requested usage."""

    rows = list(manifests)
    errors: List[str] = []
    usage = str(usage).strip().lower()
    run_mode = str(run_mode).strip().lower()
    for idx, raw in enumerate(rows):
        schema_errors = validate_trace_manifest_schema(raw, strict_required=True)
        errors.extend([f"manifest[{idx}]: {message}" for message in schema_errors])
        split = str(raw.get("split", "")).strip().lower()
        allowed_usage = {str(item).strip().lower() for item in (raw.get("allowed_usage") or [])}
        if usage not in allowed_usage:
            errors.append(f"manifest[{idx}] trace_id={raw.get('trace_id')} does not allow usage {usage!r}")
        if usage in {"eval", "replay"} and split == "train":
            message = f"{run_mode} {usage} requested train split trace: {raw.get('trace_id')}"
            if run_mode == "formal":
                raise ValueError(f"formal {usage} cannot read train split trace: {raw.get('trace_id')}")
            errors.append(message)
    if run_mode == "formal" and errors:
        raise ValueError("Invalid formal trace manifest usage:\n- " + "\n- ".join(errors))
    return {
        "trace_manifest_count": len(rows),
        "trace_manifest_errors": errors,
        "requested_usage": usage,
        "run_mode": run_mode,
        "formal_claim_allowed": bool(run_mode == "formal" and not errors),
        "outputs_are_smoke_only": bool(run_mode != "formal"),
    }
