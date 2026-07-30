"""Tests for the refiner model boundary."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import sentinel

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoharness.harness_as_policy import refiner as refiner_module
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
            response = COMPLETE_RESPONSE
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake"


COMPLETE_SOURCE = """def propose_action(board: str) -> str:
    return '[A C]'

def is_legal_action(board: str, action: str) -> bool:
    return True
""".strip()

COMPLETE_RESPONSE = f"Analysis\n```python\n{COMPLETE_SOURCE}\n```"


def test_langfuse_uses_ids_independent_of_game_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Langfuse IDs must not use global random state seeded by TextArena."""
    client_kwargs: dict[str, object] = {}

    def fake_langfuse(**kwargs: object) -> None:
        client_kwargs.update(kwargs)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(refiner_module, "_langfuse_initialized", False)
    monkeypatch.setattr(refiner_module, "Langfuse", fake_langfuse)
    monkeypatch.setattr(refiner_module, "CallbackHandler", lambda: sentinel.handler)

    assert refiner_module._get_langfuse_handler() is sentinel.handler
    assert "id_generator" in client_kwargs


def test_langfuse_initializes_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Langfuse global initialization must run at most once per process."""
    init_count = 0

    def fake_langfuse(**kwargs: object) -> None:
        nonlocal init_count
        init_count += 1

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(refiner_module, "_langfuse_initialized", False)
    monkeypatch.setattr(refiner_module, "Langfuse", fake_langfuse)
    monkeypatch.setattr(refiner_module, "CallbackHandler", lambda: sentinel.handler)

    handler1 = refiner_module._get_langfuse_handler()
    handler2 = refiner_module._get_langfuse_handler()

    assert handler1 is sentinel.handler
    assert handler2 is sentinel.handler
    assert init_count == 1


def test_refiner_reuses_one_langfuse_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refiner initializes its tracing handler once and reuses it across calls."""
    handler_initializations = 0

    def fake_get_handler() -> None:
        nonlocal handler_initializations
        handler_initializations += 1

    monkeypatch.setattr(refiner_module, "_get_langfuse_handler", fake_get_handler)
    refiner = Refiner(model=FakeChatModel(responses=[COMPLETE_RESPONSE, COMPLETE_RESPONSE]))

    for _ in range(2):
        result = refiner.refine(
            rules="rules",
            action_format="action",
            parent_source=COMPLETE_SOURCE,
            parent_heuristic=0.0,
            parent_reward=0.0,
            parent_legal_actions=0,
            parent_status="unknown",
            trajectory="trajectory",
            refine_legal_action=True,
        )
        assert result.success

    assert handler_initializations == 1


def test_refiner_returns_source() -> None:
    """Refiner extracts source from model response."""
    resp = COMPLETE_RESPONSE
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
        trajectory="Initial implementation required",
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
        trajectory="Did not solve puzzle",
        refine_legal_action=True,
    )
    assert "TowerOfHanoi-v0" in prompt
    assert "def propose_action(board: str) -> str:" in prompt
    assert "def is_legal_action(board: str, action: str) -> bool:" in prompt
    assert "Refine both `propose_action` and `is_legal_action`." in prompt
    assert "source code" in prompt


def test_refiner_prompt_contains_appendix_b2_instructions_near_verbatim() -> None:
    prompt = build_refiner_prompt(
        env_name="TowerOfHanoi-v0",
        rules="Rules here",
        action_format="[A C]",
        parent_source="source code",
        parent_heuristic=0.5,
        parent_reward=0.0,
        parent_legal_actions=5,
        parent_status="step_limit",
        trajectory="Episode 1\nSeed: 7\nboard marker",
        refine_legal_action=True,
    )

    required = [
        "Think step by step about the code, the game boards and the error feedback.",
        "Reason about each action through the game board and write down critical failure steps.",
        "Reason about code refinements that can help fix the failure steps.",
        "Reason about the entire sequence of actions",
        "progress of the game as a value between 0 and 1",
        "code refinements that can help improve the game progress",
        "code refinements that can avoid running in loops",
        "Write down your thoughts before writing the code.",
        "new code can satisfy all the observed game boards",
        "new code can fix all the current errors",
        "Do not use any try-except blocks.",
        "python code block enclosed in ```python",
    ]
    assert all(instruction in prompt for instruction in required)
    assert "board marker" in prompt
    assert "Return ONLY" not in prompt


def test_refiner_extracts_only_fenced_source_after_visible_analysis() -> None:
    response = f"Analysis of every action and loop.\n```python\n{COMPLETE_SOURCE}\n```"
    result = Refiner(model=FakeChatModel([response])).refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="unknown",
        trajectory="Episode 1",
        refine_legal_action=True,
    )

    assert result.success
    assert result.source == COMPLETE_SOURCE


def test_extract_source_returns_final_fenced_python_block() -> None:
    response = f"```python\nfirst = True\n```\nRevised:\n```python\n{COMPLETE_SOURCE}\n```"

    assert refiner_module._extract_source(response) == COMPLETE_SOURCE


def test_extract_source_rejects_empty_final_fenced_python_block() -> None:
    response = f"```python\n{COMPLETE_SOURCE}\n```\nRevised:\n```python\n\n```"

    assert refiner_module._extract_source(response) is None


def test_refiner_rejects_unfenced_source() -> None:
    result = Refiner(model=FakeChatModel([COMPLETE_SOURCE])).refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="unknown",
        trajectory="Episode 1",
        refine_legal_action=True,
    )

    assert not result.success
    assert result.source is None


def test_refiner_trace_preserves_prompt_response_and_extracted_source() -> None:
    response = f"Visible analysis\n```python\n{COMPLETE_SOURCE}\n```"
    refiner = Refiner(model=FakeChatModel([response]))

    result = refiner.refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="unknown",
        trajectory="trajectory marker",
        refine_legal_action=True,
    )

    assert result.success
    assert refiner.last_trace is not None
    assert "trajectory marker" in refiner.last_trace.prompt
    assert refiner.last_trace.invocations[0].content == response
    assert refiner.last_trace.invocations[0].normalized_text == response
    assert refiner.last_trace.extracted_source == COMPLETE_SOURCE
    assert refiner.last_trace.outcome == "success"


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
        trajectory="",
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
        trajectory="",
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_model_call_count() -> None:
    """Refiner tracks how many model calls were made."""
    model = FakeChatModel(
        responses=[
            COMPLETE_RESPONSE,
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
        trajectory="",
        refine_legal_action=True,
    )
    assert refiner.model_call_count == 1
    assert refiner.logical_refinement_count == 1


def test_refiner_retry_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refiner retries once on transport failure."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", sleep_calls.append)

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
                content=COMPLETE_RESPONSE,
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
        trajectory="",
        refine_legal_action=True,
    )
    assert result.success
    assert model._call_count == 2
    assert refiner.model_call_count == 2
    assert refiner.logical_refinement_count == 1
    assert refiner.last_trace is not None
    assert refiner.last_trace.invocations[0].error_type == "ConnectionError"
    assert refiner.last_trace.invocations[1].normalized_text == COMPLETE_RESPONSE
    assert sleep_calls == [1.0]


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
        trajectory="",
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
        trajectory="",
        refine_legal_action=True,
    )
    assert result.success
    assert result.source is not None
    assert "propose_action" in result.source
    assert refiner.last_trace is not None
    assert isinstance(refiner.last_trace.invocations[0].content, list)
    assert "thinking" not in (refiner.last_trace.invocations[0].normalized_text or "")


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
        trajectory="",
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
        trajectory="",
        refine_legal_action=True,
    )
    assert not result.success


def test_refiner_conforms_to_protocol() -> None:
    """Refiner satisfies RefinerProtocol structurally."""
    resp = COMPLETE_RESPONSE
    model = FakeChatModel(responses=[resp])
    refiner: RefinerProtocol = Refiner(model=model)
    assert refiner.model_call_count == 0
    assert refiner.logical_refinement_count == 0


def test_refiner_rejects_response_missing_legality_checker() -> None:
    """A replacement module must contain both policy contract functions."""
    model = FakeChatModel(
        responses=[
            "```python\ndef propose_action(observation: str) -> str:\n    return '[A C]'\n```"
        ]
    )

    result = Refiner(model=model).refine(
        rules="Rules",
        action_format="[A C]",
        parent_source="old",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        trajectory="",
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
            trajectory="",
            refine_legal_action=True,
        )
    assert refiner.model_call_count == 1
    assert refiner.last_trace is not None
    assert refiner.last_trace.outcome == "provider_error"
    assert refiner.last_trace.invocations[0].error_type == "ValueError"


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
            trajectory="",
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
            msg = AIMessage(content=COMPLETE_RESPONSE)
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
        trajectory="",
        refine_legal_action=True,
    )
    assert result.success
    assert model._attempts == 2
    assert refiner.model_call_count == 2


def test_refiner_records_and_propagates_context_limit_without_shortening_prompt() -> None:
    class ContextLimitModel(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise ValueError("maximum context length exceeded")

        @property
        def _llm_type(self) -> str:
            return "context_limit"

    trajectory = "unshortened-trajectory-marker" * 100
    refiner = Refiner(model=ContextLimitModel())

    with pytest.raises(ValueError, match="maximum context length exceeded"):
        refiner.refine(
            rules="Rules",
            action_format="[A C]",
            parent_source="old",
            parent_heuristic=0.0,
            parent_reward=0.0,
            parent_legal_actions=0,
            parent_status="unknown",
            trajectory=trajectory,
            refine_legal_action=True,
        )

    assert refiner.model_call_count == 1
    assert refiner.last_trace is not None
    assert trajectory in refiner.last_trace.prompt
    assert refiner.last_trace.outcome == "provider_error"
    assert refiner.last_trace.invocations[0].error_type == "ValueError"
