from __future__ import annotations

from typing import Any, Iterable


MIN_FORMAL_TRAIN_SEEDS = 8


def validate_formal_training_seed_count(train_seeds: Iterable[Any], *, run_mode: str = "formal") -> dict[str, Any]:
    """Validate independent training seed count and return collector metadata."""

    mode = str(run_mode or "formal").strip().lower()
    if mode not in {"formal", "smoke"}:
        raise ValueError(f"run_mode must be 'formal' or 'smoke', got {run_mode!r}")
    seeds = [int(seed) for seed in train_seeds]
    unique = sorted(set(seeds))
    duplicated = len(unique) != len(seeds)
    metadata = {
        "run_mode": mode,
        "num_training_seeds": len(unique),
        "train_seeds": unique,
        "outputs_are_smoke_only": mode == "smoke",
        "formal_claim_allowed": mode == "formal" and len(unique) >= MIN_FORMAL_TRAIN_SEEDS and not duplicated,
    }
    if mode == "formal":
        if duplicated:
            raise ValueError("formal training seed gate requires independent training seeds; duplicate seeds were provided")
        if len(unique) < MIN_FORMAL_TRAIN_SEEDS:
            raise ValueError(
                f"formal mode requires independent training seeds >= {MIN_FORMAL_TRAIN_SEEDS}; "
                f"got {len(unique)}"
            )
    else:
        metadata["formal_claim_allowed"] = False
    return metadata
