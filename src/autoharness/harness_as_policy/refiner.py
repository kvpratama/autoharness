"""Refiner model boundary — provider-neutral policy synthesis via LLM."""

from __future__ import annotations

import ast
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
import httpx
import openai
import requests
from google.genai import errors as google_errors
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace.id_generator import IdGenerator


class _SystemRandomIdGenerator(IdGenerator):
    """Generate OTel IDs independently of game-controlled global randomness."""

    def generate_span_id(self) -> int:
        span_id = secrets.randbits(64)
        while span_id == 0:
            span_id = secrets.randbits(64)
        return span_id

    def generate_trace_id(self) -> int:
        trace_id = secrets.randbits(128)
        while trace_id == 0:
            trace_id = secrets.randbits(128)
        return trace_id


_LANGFUSE_ID_GENERATOR = _SystemRandomIdGenerator()


def _get_langfuse_handler() -> CallbackHandler | None:
    """Initialize Langfuse and return a handler for one refiner instance.

    Langfuse v4 uses an OpenTelemetry-based architecture where the OTel
    TracerProvider/exporter is registered by the ``Langfuse()`` constructor.
    We must call ``Langfuse()`` explicitly *before* instantiating
    ``CallbackHandler`` so the process-wide client uses IDs independent of the
    game-seeded global random state. ``Refiner`` owns and reuses the returned
    handler for all of its model calls.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    if os.environ.get("LANGFUSE_ENABLED", "").lower() not in ("1", "true", "yes"):
        return None
    # Explicitly initialize the Langfuse client so the OTel pipeline is set up
    # before CallbackHandler calls get_client() internally.
    Langfuse(id_generator=_LANGFUSE_ID_GENERATOR)
    return CallbackHandler()


def _is_transient_error(e: Exception) -> bool:
    """Determine if an exception is a transient transport/network/provider-specific error."""
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True

    if isinstance(e, httpx.RequestError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        if e.response is not None and (
            e.response.status_code in (429, 502, 503, 504) or e.response.status_code >= 500
        ):
            return True

    if isinstance(e, requests.RequestException):
        if hasattr(e, "response") and e.response is not None:
            if e.response.status_code in (429, 502, 503, 504) or e.response.status_code >= 500:
                return True
        if isinstance(e, (requests.ConnectionError, requests.Timeout)):
            return True

    if isinstance(
        e,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True

    if isinstance(
        e,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.OverloadedError,
        ),
    ):
        return True

    if isinstance(e, google_errors.ServerError):
        return True
    if isinstance(e, google_errors.APIError):
        code = getattr(e, "code", None)
        if code == 429 or (isinstance(code, int) and code >= 500):
            return True

    return False


@dataclass
class RefinerResult:
    """Result from a single refinement call."""

    success: bool
    source: str | None
    error_details: str | None = None


@dataclass
class ProviderInvocation:
    """Auditable outcome of one provider invocation."""

    content: object | None = None
    normalized_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class RefinementTrace:
    """Exact prompt and provider outcomes for one logical refinement."""

    prompt: str
    invocations: list[ProviderInvocation] = field(default_factory=list)
    extracted_source: str | None = None
    outcome: str = "in_progress"
    error_details: str | None = None


REFINER_SYSTEM_PROMPT = (
    "You are a policy-synthesis assistant. Your task is to write a Python "
    "module that solves a game by implementing two functions.\n"
    "\n"
    "Environment: {env_name}\n"
    "Rules: {rules}\n"
    "Action format: {action_format}\n"
    "\n"
    "Function contracts:\n"
    "- `def propose_action(board: str) -> str:` — propose one of the best legal actions.\n"
    "- `def is_legal_action(board: str, action: str) -> bool:` — "
    "decide whether the proposed action is\n"
    "  legal for that board.\n"
    "Both functions are required in every replacement module.\n"
    "\n"
    "Refinement scope:\n"
    "{refinement_scope}\n"
    "\n"
    "You may define private helper functions and internal data structures.\n"
    "Do NOT use filesystem, network, subprocess, or dynamic-code operations.\n"
    "\nAutoHarness-specific constraints:\n"
    "\n"
    "Parent source:\n"
    "```python\n"
    "{parent_source}\n"
    "```\n"
    "\n"
    "Parent heuristic: {parent_heuristic}\n"
    "Parent terminal reward: {parent_reward}\n"
    "Parent legal actions: {parent_legal_actions}\n"
    "Parent status: {parent_status}\n"
    "\n"
    "Complete game trajectory:\n{trajectory}\n"
    "\n"
    "Preserve working behavior and avoid a fixed move script; implement a general algorithm.\n"
    "Return one complete replacement module. If the parent solved the environment perfectly, "
    "return the same source unchanged.\n"
    "Make sure to follow these instructions.\n"
    "* Think step by step about the code, the game boards and the error feedback.\n"
    "* Reason about each action through the game board and write down critical failure steps.\n"
    "* Reason about code refinements that can help fix the failure steps.\n"
    "* Reason about the entire sequence of actions and write down the progress of the game "
    "as a value between 0 and 1.\n"
    "* Reason about code refinements that can help improve the game progress.\n"
    "* Reason about code refinements that can avoid running in loops.\n"
    "* Write down your thoughts before writing the code.\n"
    "* Make sure to follow the given function signatures.\n"
    "* Make sure the new code can satisfy all the observed game boards.\n"
    "* Make sure the new code can fix all the current errors.\n"
    "* Make sure to only produce code that is safe to execute.\n"
    "* Make sure the code is concise and precise.\n"
    "* If necessary, randomly sample one of the best legal actions and return it.\n"
    "* Do not use any try-except blocks.\n"
    "* Write your functions in a python code block enclosed in ```python and ```.\n"
)


def build_refiner_prompt(
    env_name: str,
    rules: str,
    action_format: str,
    parent_source: str,
    parent_heuristic: float,
    parent_reward: float,
    parent_legal_actions: int,
    parent_status: str,
    trajectory: str,
    *,
    refine_legal_action: bool,
) -> str:
    """Build the refiner prompt with all context."""
    refinement_scope = (
        "Refine both `propose_action` and `is_legal_action`."
        if refine_legal_action
        else (
            "Refine only `propose_action`. Preserve `is_legal_action` and the helpers it depends "
            "on unchanged."
        )
    )
    return REFINER_SYSTEM_PROMPT.format(
        env_name=env_name,
        rules=rules,
        action_format=action_format,
        parent_source=parent_source,
        parent_heuristic=parent_heuristic,
        parent_reward=parent_reward,
        parent_legal_actions=parent_legal_actions,
        parent_status=parent_status,
        trajectory=trajectory,
        refinement_scope=refinement_scope,
    )


def _extract_source(response: str) -> str | None:
    """Extract the final fenced Python module from a model response."""
    matches = re.findall(r"```python\s*\n(.*?)```", response, flags=re.DOTALL)
    if not matches:
        return None
    source = matches[-1].strip()
    return source or None


def _has_policy_contract(source: str) -> bool:
    """Return whether source defines both required top-level policy functions."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    return {"is_legal_action", "propose_action"} <= names


class MessageLike(Protocol):
    """A minimal protocol for objects exposing a content attribute."""

    content: Any


def _normalize_content(response: MessageLike) -> str:
    """Extract plain text from a model response, handling content blocks.

    Models like Gemma 4 return content as a list of blocks
    (e.g. ``{"type": "thinking", …}``, ``{"type": "text", …}``).
    Only the ``"text"`` blocks are concatenated; reasoning blocks are
    discarded so they don't interfere with source extraction.
    """
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(raw)


class RefinerProtocol(Protocol):
    """Structural protocol for a policy refiner.

    Any refiner implementation must expose these members so the
    synthesis loop can call refine() and track usage statistics
    without depending on a concrete class.
    """

    @property
    def model_call_count(self) -> int: ...
    @property
    def logical_refinement_count(self) -> int: ...
    @property
    def last_trace(self) -> RefinementTrace | None: ...
    def refine(
        self,
        rules: str,
        action_format: str,
        parent_source: str,
        parent_heuristic: float,
        parent_reward: float,
        parent_legal_actions: int,
        parent_status: str,
        trajectory: str,
        env_name: str = "",
        *,
        refine_legal_action: bool,
    ) -> RefinerResult: ...


class Refiner:
    """Synthesizes candidate policy modules using a chat model."""

    def __init__(self, model: BaseChatModel | None = None, model_id: str | None = None) -> None:
        if model is not None:
            self._model = model
        elif model_id is not None:
            self._model = init_chat_model(model_id)
        else:
            raise ValueError("Either model or model_id must be provided")
        self._langfuse_handler = _get_langfuse_handler()
        self._model_call_count: int = 0
        self._logical_refinement_count: int = 0
        self._last_trace: RefinementTrace | None = None

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def logical_refinement_count(self) -> int:
        return self._logical_refinement_count

    @property
    def last_trace(self) -> RefinementTrace | None:
        return self._last_trace

    def refine(
        self,
        rules: str,
        action_format: str,
        parent_source: str,
        parent_heuristic: float,
        parent_reward: float,
        parent_legal_actions: int,
        parent_status: str,
        trajectory: str,
        env_name: str = "",
        *,
        refine_legal_action: bool,
    ) -> RefinerResult:
        """Call the model to refine the parent policy."""
        self._logical_refinement_count += 1
        prompt = build_refiner_prompt(
            env_name=env_name,
            rules=rules,
            action_format=action_format,
            parent_source=parent_source,
            parent_heuristic=parent_heuristic,
            parent_reward=parent_reward,
            parent_legal_actions=parent_legal_actions,
            parent_status=parent_status,
            trajectory=trajectory,
            refine_legal_action=refine_legal_action,
        )
        trace = RefinementTrace(prompt=prompt)
        self._last_trace = trace
        # Attempt with one retry on transport error
        last_error: str | None = None
        config: RunnableConfig = (
            {"callbacks": [self._langfuse_handler]} if self._langfuse_handler else {}
        )
        for _ in range(2):
            try:
                response = self._model.invoke(prompt, config=config)
                self._model_call_count += 1
            except Exception as e:
                self._model_call_count += 1
                trace.invocations.append(
                    ProviderInvocation(error_type=type(e).__name__, error_message=str(e))
                )
                if _is_transient_error(e):
                    last_error = str(e)
                    continue
                trace.outcome = "provider_error"
                trace.error_details = str(e)
                raise
            content = _normalize_content(response)
            trace.invocations.append(
                ProviderInvocation(content=response.content, normalized_text=content)
            )
            source = _extract_source(content)
            trace.extracted_source = source
            if source and _has_policy_contract(source):
                trace.outcome = "success"
                return RefinerResult(success=True, source=source)
            trace.outcome = "invalid_response"
            trace.error_details = "Model response did not contain both required policy functions"
            return RefinerResult(
                success=False,
                source=None,
                error_details="Model response did not contain both required policy functions",
            )
        trace.outcome = "transport_failure"
        trace.error_details = f"Model transport failure after 2 attempts: {last_error}"
        return RefinerResult(
            success=False,
            source=None,
            error_details=trace.error_details,
        )
