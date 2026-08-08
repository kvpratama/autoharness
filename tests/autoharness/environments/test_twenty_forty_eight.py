"""Contract tests for the 2048 adapter."""

from __future__ import annotations

import pytest

from autoharness.environments.base import EnvironmentAdapter
from autoharness.environments.twenty_forty_eight import TwentyFortyEightAdapter


@pytest.fixture
def adapter() -> TwentyFortyEightAdapter:
    result = TwentyFortyEightAdapter()
    result.create()
    result.reset(seed=42)
    return result


def test_adapter_contract_and_metadata() -> None:
    adapter = TwentyFortyEightAdapter()
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.env_id == "2048-v0"
    assert "[up]" in adapter.action_format
    assert "[right]" in adapter.action_format
    assert adapter.max_steps == 200
    assert adapter.rules


def test_seed_reproduces_initial_observation() -> None:
    first, second = TwentyFortyEightAdapter(), TwentyFortyEightAdapter()
    first.create()
    second.create()
    assert first.reset(seed=42) == second.reset(seed=42)


@pytest.mark.parametrize("action", ["[left]", " [LEFT] ", "[up]"])
def test_safe_canonical_actions_are_accepted(adapter: TwentyFortyEightAdapter, action: str) -> None:
    result = adapter.step(action)
    assert result.is_legal
    assert not result.terminated
    assert result.reward == 0.0


@pytest.mark.parametrize("action", ["", "up", "[w]", "[up] [down]", "[slide]", "hello"])
def test_noncanonical_actions_are_rejected(adapter: TwentyFortyEightAdapter, action: str) -> None:
    result = adapter.step(action)
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "Malformed action" in result.feedback


def test_textarena_invalid_move_is_illegal_and_normalized(
    adapter: TwentyFortyEightAdapter,
) -> None:
    assert adapter._state is not None
    adapter._state.observations[0] = [
        (
            -1,
            "You attempted an invalid move. Reason: Invalid action.",
            object(),
        )
    ]
    result = adapter.step("[up]")
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "invalid move" in result.feedback.lower()


def test_use_before_create_raises() -> None:
    adapter = TwentyFortyEightAdapter()
    with pytest.raises(RuntimeError):
        adapter.reset()
    with pytest.raises(RuntimeError):
        adapter.step("[left]")
