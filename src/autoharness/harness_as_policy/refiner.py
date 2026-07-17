"""Refiner model boundary — provider-neutral policy synthesis via LLM."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import httpx
import openai
import requests
from google.genai import errors as google_errors
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

_langfuse_handler: CallbackHandler | None = None


def _get_langfuse_handler() -> CallbackHandler | None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    if os.environ.get("LANGFUSE_ENABLED", "").lower() not in ("1", "true", "yes"):
        return None
    global _langfuse_handler
    if _langfuse_handler is None:
        _langfuse_handler = CallbackHandler()
    return _langfuse_handler


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
    "Return ONLY complete, runnable Python source code.\n"
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
    "Feedback from previous attempt (most critical first):\n"
    "{feedback}\n"
    "\n"
    "Instructions:\n"
    "1. Preserve working behavior from the parent.\n"
    "2. Reason about failures and the feedback above.\n"
    "3. Avoid a fixed move script — implement a general algorithm.\n"
    "4. Return one COMPLETE replacement module.\n"
    "5. If the parent solved the environment perfectly, "
    "return the same source unchanged.\n"
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
    feedback: list[str],
    *,
    refine_legal_action: bool,
) -> str:
    """Build the refiner prompt with all context."""
    fb_text = "\n".join(f"- {f}" for f in feedback[:5]) if feedback else "No feedback."
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
        feedback=fb_text,
        refinement_scope=refinement_scope,
    )


def _extract_source(response: str) -> str | None:
    """Extract Python source from model response."""
    text = response.strip()
    if not text:
        return None
    # Try to extract from code fence
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) >= 2:
            code = parts[1].split("```")[0].strip()
            if code:
                return code
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
            if code:
                return code
    # Fall back to raw text
    if "def propose_action" in text:
        return text
    return None


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
    def refine(
        self,
        rules: str,
        action_format: str,
        parent_source: str,
        parent_heuristic: float,
        parent_reward: float,
        parent_legal_actions: int,
        parent_status: str,
        feedback: list[str],
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
        self._model_call_count: int = 0
        self._logical_refinement_count: int = 0

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def logical_refinement_count(self) -> int:
        return self._logical_refinement_count

    def refine(
        self,
        rules: str,
        action_format: str,
        parent_source: str,
        parent_heuristic: float,
        parent_reward: float,
        parent_legal_actions: int,
        parent_status: str,
        feedback: list[str],
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
            feedback=feedback,
            refine_legal_action=refine_legal_action,
        )
        # Attempt with one retry on transport error
        last_error: str | None = None
        handler = _get_langfuse_handler()
        config: RunnableConfig = {"callbacks": [handler]} if handler else {}
        for _ in range(2):
            try:
                response = self._model.invoke(prompt, config=config)
                self._model_call_count += 1
            except Exception as e:
                self._model_call_count += 1
                if _is_transient_error(e):
                    last_error = str(e)
                    continue
                raise
            content = _normalize_content(response)
            source = _extract_source(content)
            if source and _has_policy_contract(source):
                return RefinerResult(success=True, source=source)
            return RefinerResult(
                success=False,
                source=None,
                error_details="Model response did not contain both required policy functions",
            )
        return RefinerResult(
            success=False,
            source=None,
            error_details=f"Model transport failure after 2 attempts: {last_error}",
        )
