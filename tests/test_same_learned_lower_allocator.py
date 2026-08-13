from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trisatflow.baselines.lower_allocators import (
    LowerAllocatorCheckpointError,
    SameLearnedLowerAllocator,
    build_lower_allocator,
    lower_allocator_metadata,
)


def test_formal_same_learned_missing_checkpoint_fails_fast(tmp_path: Path) -> None:
    missing = tmp_path / "missing_lower_checkpoint.pt"

    with pytest.raises(LowerAllocatorCheckpointError, match="same_learned lower checkpoint missing"):
        build_lower_allocator("same_learned", checkpoint=missing, formal=True)


def test_debug_same_learned_missing_checkpoint_falls_back_with_non_formal_metadata(tmp_path: Path) -> None:
    missing = tmp_path / "missing_lower_checkpoint.pt"
    allocator = SameLearnedLowerAllocator(missing, formal=False)
    action = allocator.allocate(None, {}, 0, {})
    metadata = lower_allocator_metadata(allocator)

    assert np.allclose(action, [1.0, 1.0, 1.0])
    assert metadata["requested_allocator"] == "same_learned"
    assert metadata["effective_lower_allocator"] == "neutral"
    assert metadata["same_learned_lower_loaded"] is False
    assert metadata["fallback_allocator"] == "neutral"
    assert metadata["formal_claim_allowed"] is False


def test_formal_same_learned_without_checkpoint_argument_fails_fast() -> None:
    with pytest.raises(LowerAllocatorCheckpointError, match="checkpoint_not_provided_or_missing"):
        SameLearnedLowerAllocator(formal=True)
