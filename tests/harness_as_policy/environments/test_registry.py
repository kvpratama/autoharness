"""Tests for environment registry coverage."""

from __future__ import annotations

import pytest

from autoharness.harness_as_policy.environments.blackjack import BlackjackAdapter
from autoharness.harness_as_policy.environments.registry import get_environment_spec
from autoharness.harness_as_policy.environments.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)


@pytest.mark.parametrize("difficulty", list(DIFFICULTY_MAP))
def test_hanoi_ids_resolve_with_one_training_rollout(difficulty: str) -> None:
    spec = get_environment_spec(DIFFICULTY_MAP[difficulty][0])
    assert isinstance(spec.create_adapter(), TowerOfHanoiAdapter)
    assert spec.default_training_rollouts == 1
    assert len(spec.evaluation_cases) == 4
    assert spec.family == "tower-of-hanoi"


def test_blackjack_resolves_with_five_training_rollouts() -> None:
    spec = get_environment_spec("Blackjack-v0")
    assert isinstance(spec.create_adapter(), BlackjackAdapter)
    assert spec.default_training_rollouts == 5
    assert len(spec.evaluation_cases) == 1
    assert spec.family == "blackjack"


def test_unknown_environment_lists_valid_ids() -> None:
    with pytest.raises(ValueError, match="Blackjack-v0"):
        get_environment_spec("Unknown-v0")
