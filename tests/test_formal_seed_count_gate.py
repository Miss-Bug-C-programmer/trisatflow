from __future__ import annotations

import pytest

from trisatflow.formal_gates import MIN_FORMAL_TRAIN_SEEDS, validate_formal_training_seed_count


def test_formal_train_seeds_less_than_eight_fail() -> None:
    with pytest.raises(ValueError, match="independent training seeds"):
        validate_formal_training_seed_count([1, 2, 3], run_mode="formal")


def test_formal_train_seeds_gate_accepts_eight_independent_seeds() -> None:
    metadata = validate_formal_training_seed_count(range(MIN_FORMAL_TRAIN_SEEDS), run_mode="formal")

    assert metadata["num_training_seeds"] == MIN_FORMAL_TRAIN_SEEDS
    assert metadata["formal_claim_allowed"] is True
    assert metadata["outputs_are_smoke_only"] is False


def test_formal_train_seed_duplicates_fail_independence_gate() -> None:
    with pytest.raises(ValueError, match="duplicate seeds"):
        validate_formal_training_seed_count([1, 1, 2, 3, 4, 5, 6, 7], run_mode="formal")


def test_smoke_mode_allows_few_seeds_but_marks_non_formal() -> None:
    metadata = validate_formal_training_seed_count([7], run_mode="smoke")

    assert metadata["num_training_seeds"] == 1
    assert metadata["outputs_are_smoke_only"] is True
    assert metadata["formal_claim_allowed"] is False
