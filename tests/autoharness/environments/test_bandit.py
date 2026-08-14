"""Contract tests for the TextArena Bandit adapters."""

from __future__ import annotations

import pytest

from autoharness.environments.bandit import BANDIT_VARIANTS, BanditAdapter
from autoharness.environments.base import EnvironmentAdapter


@pytest.fixture(params=tuple(BANDIT_VARIANTS))
def adapter(request: pytest.FixtureRequest) -> BanditAdapter:
    result = BanditAdapter(request.param)
    result.create()
    result.reset(seed=42)
    return result


def test_adapter_contract_and_metadata(adapter: BanditAdapter) -> None:
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.env_id in BANDIT_VARIANTS
    assert adapter.rules
    assert adapter.action_format
    assert adapter.max_steps == BANDIT_VARIANTS[adapter.env_id].max_steps


@pytest.mark.parametrize("env_id", tuple(BANDIT_VARIANTS))
def test_seed_reproduces_initial_observation(env_id: str) -> None:
    first, second = BanditAdapter(env_id), BanditAdapter(env_id)
    first.create()
    second.create()
    assert first.reset(seed=42) == second.reset(seed=42)


def test_case_and_whitespace_are_canonicalized(adapter: BanditAdapter) -> None:
    result = adapter.step(" [RED] ")
    assert result.is_legal
    assert not result.terminated
    assert result.reward == 0.0


@pytest.mark.parametrize("action", ["", "red", "[unknown]", "[red] [blue]", "hello"])
def test_malformed_or_unknown_actions_are_rejected(adapter: BanditAdapter, action: str) -> None:
    result = adapter.step(action)
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "Malformed action" in result.feedback or "unknown" in result.feedback.lower()


def test_final_selection_returns_native_terminal_reward() -> None:
    adapter = BanditAdapter("Bandit-v0")
    adapter.create()
    adapter.reset(seed=42)

    for _ in range(20):
        result = adapter.step("[red]")
        assert result.is_legal
        assert not result.terminated
        assert result.reward == 0.0

    result = adapter.step("[red]")
    assert result.is_legal
    assert result.terminated
    assert result.reward <= 1.0


def test_textarena_invalid_move_is_normalized() -> None:
    adapter = BanditAdapter("Bandit-v0")
    adapter.create()
    adapter.reset(seed=42)
    assert adapter._state is not None
    adapter._state.observations[0] = [
        (-1, "You attempted an invalid move. Reason: invalid button", object())
    ]

    result = adapter.step("[red]")

    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0
    assert "invalid move" in result.feedback.lower()


def test_use_before_create_raises() -> None:
    adapter = BanditAdapter()
    with pytest.raises(RuntimeError):
        adapter.reset()
    with pytest.raises(RuntimeError):
        adapter.step("[red]")
