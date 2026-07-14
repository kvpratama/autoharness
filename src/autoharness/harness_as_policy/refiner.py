"""Refiner model boundary — provider-neutral policy synthesis via LLM."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langfuse.langchain import CallbackHandler

_langfuse_handler: CallbackHandler | None = None


def _get_langfuse_handler() -> CallbackHandler:
    global _langfuse_handler
    if _langfuse_handler is None:
        _langfuse_handler = CallbackHandler()
    return _langfuse_handler


@dataclass
class RefinerResult:
    """Result from a single refinement call."""

    success: bool
    source: str | None
    error_details: str | None = None


REFINER_SYSTEM_PROMPT = (
    "You are a policy-synthesis assistant. Your task is to write a Python "
    "module that solves a game by implementing one function.\n"
    "\n"
    "Environment: {env_name}\n"
    "Rules: {rules}\n"
    "Action format: {action_format}\n"
    "\n"
    "Function contract:\n"
    "- `def propose_action(observation: str) -> str:` — receive the "
    "current environment observation and return exactly one valid action.\n"
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
) -> str:
    """Build the refiner prompt with all context."""
    fb_text = "\n".join(f"- {f}" for f in feedback[:5]) if feedback else "No feedback."
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
        )
        # Attempt with one retry on transport error
        last_error: str | None = None
        for _ in range(2):
            try:
                response = self._model.invoke(
                    prompt, config={"callbacks": [_get_langfuse_handler()]}
                )
                self._model_call_count += 1
            except Exception as e:
                self._model_call_count += 1
                last_error = str(e)
                continue
            raw = response.content if hasattr(response, "content") else str(response)
            content = raw if isinstance(raw, str) else str(raw)
            source = _extract_source(content)
            if source and "propose_action" in source:
                return RefinerResult(success=True, source=source)
            return RefinerResult(
                success=False,
                source=None,
                error_details="Model response did not contain valid propose_action source",
            )
        return RefinerResult(
            success=False,
            source=None,
            error_details=f"Model transport failure after 2 attempts: {last_error}",
        )
