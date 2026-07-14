"""Live-policy model boundary — receives observation and returns one action."""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

LIVE_PROMPT = (
    "You are playing the game {env_name}.\n"
    "Rules: {rules}\n"
    "Action format: {action_format}\n"
    "\n"
    "Current observation:\n{observation}\n"
    "\n"
    "Return exactly one valid action matching the action format. "
    "Do NOT include any other text, explanation, or formatting."
)


@dataclass
class LiveActionResult:
    """Result from a single live-policy action call."""

    action: str | None
    success: bool
    latency: float
    model_calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    error_details: str | None = None


class LivePolicy:
    """Calls an LLM to produce a single action from an observation."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        model_id: str | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        elif model_id is not None:
            self._model = init_chat_model(model_id)
        else:
            raise ValueError("Either model or model_id must be provided")
        self._model_call_count: int = 0
        self._input_price_per_million = input_price_per_million
        self._output_price_per_million = output_price_per_million

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    def act(
        self,
        env_name: str,
        rules: str,
        action_format: str,
        observation: str,
    ) -> LiveActionResult:
        """Call the model to produce an action from the current observation."""
        prompt = LIVE_PROMPT.format(
            env_name=env_name,
            rules=rules,
            action_format=action_format,
            observation=observation,
        )
        start = time.monotonic()
        try:
            response = self._model.invoke(prompt)
            self._model_call_count += 1
            latency = time.monotonic() - start
        except Exception as e:
            return LiveActionResult(
                action=None,
                success=False,
                latency=time.monotonic() - start,
                error_details=str(e),
            )
        raw = response.content if hasattr(response, "content") else str(response)
        content = raw if isinstance(raw, str) else str(raw)
        action = content.strip()
        if not action:
            return LiveActionResult(
                action=None,
                success=False,
                latency=latency,
                error_details="Model returned empty response",
            )
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0) or 0
            output_tokens = response.usage_metadata.get("output_tokens", 0) or 0
        return LiveActionResult(
            action=action,
            success=True,
            latency=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
