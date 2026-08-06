"""Tests for environment registry coverage."""

from __future__ import annotations

import pytest

from autoharness.environments.blackjack import BlackjackAdapter
from autoharness.environments.registry import get_environment_spec
from autoharness.environments.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)


@pytest.mark.parametrize("difficulty", list(DIFFICULTY_MAP))
def test_hanoi_ids_resolve_with_intrinsic_metadata(difficulty: str) -> None:
    env_id, _max_steps, optimal_steps = DIFFICULTY_MAP[difficulty]
    spec = get_environment_spec(env_id)
    assert isinstance(spec.create_adapter(), TowerOfHanoiAdapter)
    assert spec.optimal_steps == optimal_steps
    assert spec.family == "tower-of-hanoi"


def test_blackjack_resolves_with_intrinsic_metadata() -> None:
    spec = get_environment_spec("Blackjack-v0")
    assert isinstance(spec.create_adapter(), BlackjackAdapter)
    assert spec.optimal_steps == 0
    assert spec.family == "blackjack"


def test_unknown_environment_lists_valid_ids() -> None:
    with pytest.raises(ValueError, match="Blackjack-v0"):
        get_environment_spec("Unknown-v0")
