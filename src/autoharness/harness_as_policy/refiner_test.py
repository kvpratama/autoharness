"""Tests for the refiner model boundary."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoharness.harness_as_policy.refiner import (
    Refiner,
    build_refiner_prompt,
)


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
            response = "def propose_action(observation: str) -> str:\n    return '[A C]'"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake"


def test_refiner_returns_source() -> None:
    """Refiner extracts source from model response."""
    resp = "def propose_action(observation: str) -> str:\n    return '[A C]'"
    model = FakeChatModel(responses=[resp])
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Tower of Hanoi rules",
        action_format="[A C]",
        parent_source="def propose_action(observation: str) -> str:\n    raise NotImplementedError",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=["Initial implementation required"],
    )
    assert result.success
    assert result.source is not None
    assert "propose_action" in result.source


def test_refiner_prompt_contains_required_sections() -> None:
    """Build prompt includes rules, function contract, and parent info."""
    prompt = build_refiner_prompt(
        env_name="TowerOfHanoi-v0",
        rules="Rules here",
        action_format="[A C]",
        parent_source="source code",
        parent_heuristic=0.5,
        parent_reward=0.0,
        parent_legal_actions=5,
        parent_status="step_limit",
        feedback=["Did not solve puzzle"],
    )
    assert "TowerOfHanoi-v0" in prompt
    assert "propose_action" in prompt
    assert "source code" in prompt


def test_refiner_malformed_response() -> None:
    """Refiner handles malformed response (no source) gracefully."""
    model = FakeChatModel(responses=[""])
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[""],
    )
    assert not result.success


def test_refiner_model_call_count() -> None:
    """Refiner tracks how many model calls were made."""
    model = FakeChatModel(
        responses=[
            "def propose_action(observation: str) -> str:\n    return '[A C]'",
        ]
    )
    refiner = Refiner(model=model)
    refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[""],
    )
    assert refiner.model_call_count == 1
    assert refiner.logical_refinement_count == 1


def test_refiner_retry_on_transport_error() -> None:
    """Refiner retries once on transport failure."""

    class RetryModel(BaseChatModel):
        def __init__(self) -> None:
            super().__init__()
            self._call_count = 0

        def _generate(self, *args, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                raise ConnectionError("Transport failure")
            msg = AIMessage(
                content="def propose_action(observation: str) -> str:\n    return '[A C]'",
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "retry_fake"

    model = RetryModel()
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[""],
    )
    assert result.success
    assert model._call_count == 2
    assert refiner.model_call_count == 2
    assert refiner.logical_refinement_count == 1


def test_refiner_double_retry_failure() -> None:
    """Refiner returns failure after two transport errors."""

    class AlwaysFailsModel(BaseChatModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Always fails")

        @property
        def _llm_type(self) -> str:
            return "always_fail"

    model = AlwaysFailsModel()
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[""],
    )
    assert not result.success
