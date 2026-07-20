"""Contract tests for the Blackjack adapter."""

from __future__ import annotations

import pytest

from autoharness.harness_as_policy.blackjack import BlackjackAdapter
from autoharness.harness_as_policy.environment import EnvironmentAdapter


@pytest.fixture
def adapter() -> BlackjackAdapter:
    result = BlackjackAdapter()
    result.create()
    result.reset(seed=42)
    return result


def test_adapter_contract_and_metadata() -> None:
    adapter = BlackjackAdapter()
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.env_id == "Blackjack-v0"
    assert "[Hit]" in adapter.action_format
    assert "[Stand]" in adapter.action_format
    assert adapter.max_steps == 50


def test_seed_reproduces_initial_observation() -> None:
    first, second = BlackjackAdapter(), BlackjackAdapter()
    first.create()
    second.create()
    assert first.reset(seed=1234) == second.reset(seed=1234)


@pytest.mark.parametrize("action", ["[Hit]", "[hit]", " [Stand] ", "[STAND]"])
def test_exact_actions_are_accepted(adapter: BlackjackAdapter, action: str) -> None:
    assert adapter.step(action).is_legal


@pytest.mark.parametrize("action", ["", "Hit", "[Hit] [Stand]", "[Double]", "hello"])
def test_malformed_actions_are_rejected(adapter: BlackjackAdapter, action: str) -> None:
    result = adapter.step(action)
    assert not result.is_legal
    assert result.terminated
    assert result.reward == 0.0


def test_standing_completes_five_hands_and_propagates_reward() -> None:
    adapter = BlackjackAdapter()
    adapter.create()
    assert "Hand 1/5" in adapter.reset(seed=42)
    result = adapter.step("[Stand]")
    for _ in range(4):
        assert result.is_legal
        if result.terminated:
            break
        result = adapter.step("[Stand]")
    assert result.terminated
    assert result.reward in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def test_use_before_create_raises() -> None:
    adapter = BlackjackAdapter()
    with pytest.raises(RuntimeError):
        adapter.reset()
    with pytest.raises(RuntimeError):
        adapter.step("[Stand]")
