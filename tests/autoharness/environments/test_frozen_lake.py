"""Contract tests for the FrozenLake adapter."""

from __future__ import annotations

import pytest

from autoharness.environments.base import EnvironmentAdapter
from autoharness.environments.frozen_lake import FrozenLakeAdapter


@pytest.fixture
def adapter() -> FrozenLakeAdapter:
    result = FrozenLakeAdapter()
    result.create()
    result.reset(seed=42)
    return result


def test_adapter_contract_and_metadata() -> None:
    adapter = FrozenLakeAdapter()
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.env_id == "FrozenLake-v0"
    assert "[up]" in adapter.action_format
    assert "[right]" in adapter.action_format
    assert adapter.max_steps == 100
    assert adapter.rules


def test_seed_reproduces_initial_observation() -> None:
    first, second = FrozenLakeAdapter(), FrozenLakeAdapter()
    first.create()
    second.create()
    assert first.reset(seed=42) == second.reset(seed=42)


@pytest.mark.parametrize("action", ["[down]", " [DOWN] "])
def test_safe_canonical_actions_are_accepted(adapter: FrozenLakeAdapter, action: str) -> None:
    result = adapter.step(action)
    assert result.is_legal
    assert not result.terminated
    assert result.reward == 0.0


@pytest.mark.parametrize("action", ["", "up", "[w]", "[up] [down]", "[jump]", "hello"])
def test_noncanonical_actions_are_rejected(adapter: FrozenLakeAdapter, action: str) -> None:
    result = adapter.step(action)
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "Malformed action" in result.feedback


def test_wall_collision_is_illegal_and_normalized(adapter: FrozenLakeAdapter) -> None:
    result = adapter.step("[up]")
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "invalid move" in result.feedback.lower()


def test_legal_hole_transition_preserves_native_terminal_reward(
    adapter: FrozenLakeAdapter,
) -> None:
    result = adapter.step("[right]")
    assert result.is_legal
    assert result.terminated
    assert result.reward == pytest.approx(1 / 6)


def test_legal_goal_transition_preserves_reward_and_nonterminal_zeroes(
    adapter: FrozenLakeAdapter,
) -> None:
    for action in ("[down]", "[right]", "[right]", "[down]", "[down]"):
        result = adapter.step(action)
        assert result.is_legal
        assert not result.terminated
        assert result.reward == 0.0

    result = adapter.step("[right]")
    assert result.is_legal
    assert result.terminated
    assert result.reward == 1.0


def test_use_before_create_raises() -> None:
    adapter = FrozenLakeAdapter()
    with pytest.raises(RuntimeError):
        adapter.reset()
    with pytest.raises(RuntimeError):
        adapter.step("[down]")
