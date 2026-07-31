"""Tests for the live-policy model boundary."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoharness.harness_as_policy.live_policy import LIVE_PROMPT, LivePolicy


class FakeChatModel(BaseChatModel):
    """A fake chat model that returns scripted responses."""

    responses: list[str | AIMessage]
    _call_count: int = 0

    def __init__(self, responses: list[str | AIMessage] | None = None) -> None:
        resp = responses or []
        super().__init__(responses=resp)
        self._call_count = 0

    def _generate(self, *args, **kwargs):
        self._call_count += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = "[A C]"
        message = response if isinstance(response, AIMessage) else AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake"


def test_live_prompt_matches_appendix_b1_contract() -> None:
    prompt = LIVE_PROMPT.format(player_id=0, observation="Current board")
    assert "You are an expert, logical, and strategic AI game player." in prompt
    assert "First, provide your step-by-step reasoning." in prompt
    assert "<move></move>" in prompt
    assert "Current board" in prompt
    assert "Do NOT include any other text" not in prompt


def test_live_policy_extracts_move_after_reasoning() -> None:
    model = FakeChatModel(responses=["I compare both legal options.\n<move>[A C]</move>"])
    result = LivePolicy(model=model).act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="Board",
    )
    assert result.success
    assert result.action == "[A C]"


@pytest.mark.parametrize(
    ("response", "error_fragment"),
    [
        ("[A C]", "exactly one"),
        ("reasoning only", "exactly one"),
        ("<move>   </move>", "empty"),
        ("<move>[A C]</move> trailing", "after"),
        ("<move>[A C]</move><move>[C B]</move>", "exactly one"),
        ("<move>bad <move>[A C]</move>", "exactly one"),
        ("reasoning </move><move>[A C]</move>", "exactly one"),
        ("</move><move>[A C]", "exactly one"),
        ("<move>[A C]", "exactly one"),
    ],
)
def test_live_policy_rejects_malformed_move_response(response: str, error_fragment: str) -> None:
    result = LivePolicy(model=FakeChatModel(responses=[response])).act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="Board",
    )
    assert not result.success
    assert result.action is None
    assert result.error_details is not None
    assert error_fragment in result.error_details


def test_malformed_response_keeps_usage_and_cost() -> None:
    message = AIMessage(
        content="untagged action",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    policy = LivePolicy(
        model=FakeChatModel(responses=[message]),
        input_price_per_million=2.0,
        output_price_per_million=8.0,
    )

    result = policy.act(
        env_name="TowerOfHanoi-v0",
        rules="Rules",
        action_format="[A C]",
        observation="Board",
    )

    assert not result.success
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.estimated_cost_usd == pytest.approx(0.00048)


def test_live_policy_returns_action() -> None:
    """LivePolicy.act returns the model's response as an action."""
    model = FakeChatModel(responses=["<move>[A C]</move>"])
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
