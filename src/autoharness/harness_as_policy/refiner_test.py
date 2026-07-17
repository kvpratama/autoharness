"""Tests for the refiner model boundary."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoharness.harness_as_policy.refiner import (
    Refiner,
    RefinerProtocol,
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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._call_count += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = COMPLETE_SOURCE
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake"


COMPLETE_SOURCE = """def propose_action(board: str) -> str:
    return '[A C]'

def is_legal_action(board: str, action: str) -> bool:
    return True
"""


def test_refiner_returns_source() -> None:
    """Refiner extracts source from model response."""
    resp = COMPLETE_SOURCE
    model = FakeChatModel(responses=[resp])
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Tower of Hanoi rules",
        action_format="[A C]",
        parent_source=(
            "def propose_action(board: str) -> str:\n    raise NotImplementedError\n\n"
            "def is_legal_action(board: str, action: str) -> bool:\n"
            "    raise NotImplementedError"
        ),
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=["Initial implementation required"],
        refine_legal_action=True,
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
        refine_legal_action=True,
    )
    assert "TowerOfHanoi-v0" in prompt
    assert "def propose_action(board: str) -> str:" in prompt
    assert "def is_legal_action(board: str, action: str) -> bool:" in prompt
    assert "Refine both `propose_action` and `is_legal_action`." in prompt
    assert "source code" in prompt


def test_refiner_prompt_preserves_checker_when_scope_is_action_only() -> None:
    """Prompt tells the model to preserve the checker for a policy rejection."""
    prompt = build_refiner_prompt(
        env_name="TowerOfHanoi-v0",
        rules="Rules here",
        action_format="[A C]",
        parent_source="source code",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="policy_rejected_action",
        feedback=[],
        refine_legal_action=False,
    )

    assert (
        "Refine only `propose_action`. Preserve `is_legal_action` and the helpers it depends on "
        "unchanged."
    ) in prompt


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
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_model_call_count() -> None:
    """Refiner tracks how many model calls were made."""
    model = FakeChatModel(
        responses=[
            COMPLETE_SOURCE,
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
        refine_legal_action=True,
    )
    assert refiner.model_call_count == 1
    assert refiner.logical_refinement_count == 1


def test_refiner_retry_on_transport_error() -> None:
    """Refiner retries once on transport failure."""

    class RetryModel(BaseChatModel):
        def __init__(self) -> None:
            super().__init__()
            self._call_count = 0

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            self._call_count += 1
            if self._call_count == 1:
                raise ConnectionError("Transport failure")
            msg = AIMessage(
                content=COMPLETE_SOURCE,
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
        refine_legal_action=True,
    )
    assert result.success
    assert model._call_count == 2
    assert refiner.model_call_count == 2
    assert refiner.logical_refinement_count == 1


def test_refiner_double_retry_failure() -> None:
    """Refiner returns failure after two transport errors."""

    class AlwaysFailsModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
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
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_extracts_source_from_content_blocks() -> None:
    """Refiner extracts source from thinking+text content blocks (Gemma 4 style).

    The thinking block contains a code fence (common for model reasoning),
    which should NOT confuse source extraction.
    """

    class ContentBlockModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = AIMessage(
                content=[
                    {
                        "type": "thinking",
                        "thinking": (
                            "Let me reason step by step...\n"
                            "```python\n"
                            "# Pseudo-code for algorithm\n"
                            "if solved:\n"
                            "    return '[A C]'\n"
                            "```\n"
                            "Now implementing..."
                        ),
                    },
                    {
                        "type": "text",
                        "text": (
                            "```python\n"
                            "def propose_action(board: str) -> str:\n"
                            "    return '[A C]'\n"
                            "\n"
                            "def is_legal_action(board: str, action: str) -> bool:\n"
                            "    return True\n"
                            "```"
                        ),
                    },
                ]
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "content_block_fake"

    model = ContentBlockModel()
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
        refine_legal_action=True,
    )
    assert result.success
    assert result.source is not None
    assert "propose_action" in result.source


def test_refiner_content_blocks_no_text_block() -> None:
    """Refiner returns failure when content blocks contain only thinking."""

    class ThinkingOnlyModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = AIMessage(
                content=[
                    {"type": "thinking", "thinking": "I should think about this more..."},
                ]
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "thinking_only_fake"

    model = ThinkingOnlyModel()
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
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_content_blocks_empty_list() -> None:
    """Refiner handles empty content block list gracefully."""

    class EmptyBlocksModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = AIMessage(content=[])
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "empty_blocks_fake"

    model = EmptyBlocksModel()
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
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_conforms_to_protocol() -> None:
    """Refiner satisfies RefinerProtocol structurally."""
    resp = COMPLETE_SOURCE
    model = FakeChatModel(responses=[resp])
    refiner: RefinerProtocol = Refiner(model=model)
    assert refiner.model_call_count == 0
    assert refiner.logical_refinement_count == 0


def test_refiner_rejects_response_missing_legality_checker() -> None:
    """A replacement module must contain both policy contract functions."""
    model = FakeChatModel(
        responses=["def propose_action(observation: str) -> str:\n    return '[A C]'"]
    )

    result = Refiner(model=model).refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[],
        refine_legal_action=True,
    )

    assert not result.success
    assert result.error_details == "Model response did not contain both required policy functions"


def test_refiner_propagates_programming_error() -> None:
    """Refiner propagates standard exceptions (like ValueError) immediately."""
    import pytest

    class FailModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise ValueError("Programming error")

        @property
        def _llm_type(self) -> str:
            return "fail"

    model = FailModel()
    refiner = Refiner(model=model)
    with pytest.raises(ValueError, match="Programming error"):
        refiner.refine(
            rules="Rules",
            action_format="[A C]",
            parent_source="old",
            parent_heuristic=0.0,
            parent_reward=0.0,
            parent_legal_actions=0,
            parent_status="contract_failure",
            feedback=[],
            refine_legal_action=True,
        )
    assert refiner.model_call_count == 1


def test_refiner_propagates_openai_auth_error() -> None:
    """Refiner propagates non-transient provider exceptions immediately."""
    import httpx
    import openai
    import pytest

    class AuthFailModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            req = httpx.Request("POST", "http://test")
            res = httpx.Response(401, request=req)
            raise openai.AuthenticationError("Auth failed", response=res, body=None)

        @property
        def _llm_type(self) -> str:
            return "auth_fail"

    model = AuthFailModel()
    refiner = Refiner(model=model)
    with pytest.raises(openai.AuthenticationError, match="Auth failed"):
        refiner.refine(
            rules="Rules",
            action_format="[A C]",
            parent_source="old",
            parent_heuristic=0.0,
            parent_reward=0.0,
            parent_legal_actions=0,
            parent_status="contract_failure",
            feedback=[],
            refine_legal_action=True,
        )
    assert refiner.model_call_count == 1


def test_refiner_retries_transient_openai_error() -> None:
    """Refiner retries transient provider errors (like RateLimitError)."""
    import httpx
    import openai

    class RateLimitModel(BaseChatModel):
        def __init__(self) -> None:
            super().__init__()
            self._attempts = 0

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            self._attempts += 1
            if self._attempts == 1:
                req = httpx.Request("POST", "http://test")
                res = httpx.Response(429, request=req)
                raise openai.RateLimitError("Rate limit exceeded", response=res, body=None)
            msg = AIMessage(content=COMPLETE_SOURCE)
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "rate_limit"

    model = RateLimitModel()
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=[],
        refine_legal_action=True,
    )
    assert result.success
    assert model._attempts == 2
    assert refiner.model_call_count == 2
