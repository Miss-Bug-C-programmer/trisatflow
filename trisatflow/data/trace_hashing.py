from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_mapping(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(dict(obj)).encode("utf-8")).hexdigest()


def content_fingerprint(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    return f"sha256:{sha256_file(p)}|bytes:{stat.st_size}"

