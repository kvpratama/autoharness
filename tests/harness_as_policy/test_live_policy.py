"""Tests for the live-policy model boundary."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoharness.harness_as_policy.live_policy import LivePolicy


class FakeChatModel(BaseChatModel):
    """A fake chat model that returns scripted responses."""

    responses: list[str]
    _call_count: int = 0

    def __init__(self, responses: list[str] | None = None) -> None:
        resp = responses or []
        super().__init__(responses=resp)
        self._call_count = 0

    def _generate(self, *args, **kwargs):
        self._call_count += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = "[A C]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake"


def test_live_policy_returns_action() -> None:
    """LivePolicy.act returns the model's response as an action."""
    model = FakeChatModel(responses=["[A C]"])
    policy = LivePolicy(model=model)
    result = policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Tower of Hanoi rules",
        action_format="[A C]",
        observation="Peg A: [3,2,1]",
    )
    assert result.success
    assert result.action == "[A C]"
    assert result.model_calls == 1


def test_live_policy_tracks_model_call_count() -> None:
    """LivePolicy tracks how many model calls were made."""
    model = FakeChatModel(responses=["[A C]", "[C B]"])
    policy = LivePolicy(model=model)
    policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="obs1",
    )
    assert policy.model_call_count == 1
    policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="obs2",
    )
    assert policy.model_call_count == 2


def test_live_policy_empty_response() -> None:
    """Empty model response returns failure."""
    model = FakeChatModel(responses=[""])
    policy = LivePolicy(model=model)
    result = policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="obs",
    )
    assert not result.success
    assert result.action is None


def test_live_policy_latency() -> None:
    """LiveActionResult includes latency measurement."""
    model = FakeChatModel(responses=["[A C]"])
    policy = LivePolicy(model=model)
    result = policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="obs",
    )
    assert result.latency >= 0
    assert isinstance(result.latency, float)
