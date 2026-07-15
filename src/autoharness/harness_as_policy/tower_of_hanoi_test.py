"""Tests for the Tower of Hanoi environment adapter."""

from __future__ import annotations

import pytest

from autoharness.harness_as_policy.environment import EnvironmentAdapter
from autoharness.harness_as_policy.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)


def test_adapter_is_environment_adapter() -> None:
    """TowerOfHanoiAdapter satisfies EnvironmentAdapter protocol."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter, EnvironmentAdapter)


def test_env_id_default() -> None:
    """Default env_id is TowerOfHanoi-v0."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert adapter.env_id == "TowerOfHanoi-v0"


def test_env_id_medium() -> None:
    """Medium difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="medium")
    assert adapter.env_id == "TowerOfHanoi-v0-medium"


def test_env_id_hard() -> None:
    """Hard difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="hard")
    assert adapter.env_id == "TowerOfHanoi-v0-hard"


def test_env_id_hardcore() -> None:
    """Hardcore difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="hardcore")
    assert adapter.env_id == "TowerOfHanoi-v0-hardcore"


def test_max_steps_v0() -> None:
    """Three-disk variant has 14 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert adapter.max_steps == 14


def test_max_steps_medium() -> None:
    """Four-disk variant has 30 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="medium")
    assert adapter.max_steps == 30


def test_max_steps_hard() -> None:
    """Five-disk variant has 62 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="hard")
    assert adapter.max_steps == 62


def test_max_steps_hardcore() -> None:
    """Six-disk variant has 126 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="hardcore")
    assert adapter.max_steps == 126


def test_rules_is_string() -> None:
    """Rules property returns a non-empty string."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter.rules, str)
    assert len(adapter.rules) > 0


def test_action_format_is_string() -> None:
    """Action format description is a non-empty string."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter.action_format, str)
    assert len(adapter.action_format) > 0


def test_create_and_reset_returns_string() -> None:
    """Create and reset returns a string observation."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    obs = adapter.reset(seed=42)
    assert isinstance(obs, str)
    assert len(obs) > 0


def test_legal_action_single_move() -> None:
    """A single legal bracketed move is accepted."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("[A C]")
    assert result.is_legal


def test_malformed_action_illegal() -> None:
    """Empty string is rejected before environment submission."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("")
    assert not result.is_legal


def test_multiple_bracketed_moves_illegal() -> None:
    """Multiple bracketed moves are rejected before environment submission."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("[A C] [C B]")
    assert not result.is_legal


def test_random_string_illegal() -> None:
    """A random string without brackets is rejected."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("hello world")
    assert not result.is_legal


def test_difficulty_map_correct() -> None:
    """DIFFICULTY_MAP has all four variants with correct turn limits."""
    assert DIFFICULTY_MAP["v0"] == ("TowerOfHanoi-v0", 14)
    assert DIFFICULTY_MAP["medium"] == ("TowerOfHanoi-v0-medium", 30)
    assert DIFFICULTY_MAP["hard"] == ("TowerOfHanoi-v0-hard", 62)
    assert DIFFICULTY_MAP["hardcore"] == ("TowerOfHanoi-v0-hardcore", 126)


def test_invalid_difficulty_raises() -> None:
    """Invalid difficulty raises ValueError."""
    with pytest.raises(ValueError, match="Unknown difficulty"):
        TowerOfHanoiAdapter(difficulty="nonexistent")


def test_step_before_create_raises() -> None:
    """Calling step before create raises RuntimeError."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    with pytest.raises(RuntimeError):
        adapter.step("[A C]")


def test_reset_before_create_raises() -> None:
    """Calling reset before create raises RuntimeError."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    with pytest.raises(RuntimeError):
        adapter.reset(seed=42)


def test_truncation_reward_reflects_current_partial_completion() -> None:
    """Truncation reward reports TextArena's completion fraction without another action."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    for action in ("[A C]", "[A B]", "[C B]", "[A C]"):
        result = adapter.step(action)
        assert result.is_legal
        assert not result.terminated

    assert adapter.truncation_reward() == pytest.approx(1 / 3)
