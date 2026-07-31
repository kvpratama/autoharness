"""Live-policy model boundary — receives observation and returns one action."""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

LIVE_PROMPT = (
    "You are an expert, logical, and strategic AI game player. Your task is to"
    " analyze the following game information and determine the single best move"
    " to make.\n"
    "\n"
    "Read the game rules, your player role, the current game state, and all"
    " available moves carefully. Your objective is to play optimally to maximize"
    " your chances of winning the game.\n"
    "\n"
    "You are now player {player_id}.\n"
    "\n"
    "The game is {env_name}.\n"
    "The game rules are as follows: {rules}\n"
    "The required action format is as follows: {action_format}\n"
    "\n"
    "The game information is as follows: {observation}\n"
    "\n"
    "**YOUR TASK:**\n"
    "\n"
    "You must now analyze the situation and provide your move. Follow these two"
    " steps precisely.\n"
    "\n"
    "**Step 1: Think**\n"
    "\n"
    "First, provide your step-by-step reasoning. Analyze the current game state,"
    " your goal, and the available moves. Evaluate the pros and cons of the most"
    " promising options and explain why you are selecting your final move.\n"
    "\n"
    "**Step 2: Move**\n"
    "\n"
    "After your thinking block, provide only the single best move you have"
    " chosen. The move must be one of the valid moves listed in the game"
    " information. Enclose your final move in <move></move> tags. Do not add any"
    " other text, explanation, or punctuation after the closing </move> tag."
    " Example of a correct response format: <move>[Your chosen move]</move>\n"
)


def _extract_move(content: str) -> str:
    if content.count("<move>") != 1 or content.count("</move>") != 1:
        raise ValueError("Model response must contain exactly one <move>...</move> payload")
    opening_index = content.index("<move>")
    payload_start = opening_index + len("<move>")
    closing_index = content.index("</move>")
    if opening_index > closing_index:
        raise ValueError("Model response must contain exactly one <move>...</move> payload")
    action = content[payload_start:closing_index].strip()
    if not action:
        raise ValueError("Model response contains an empty <move> payload")
    if content[closing_index + len("</move>") :].strip():
        raise ValueError("Model response contains content after </move>")
    return action


@dataclass
class LiveActionResult:
    """Result from a single live-policy action call."""

    action: str | None
    success: bool
    latency: float
    model_calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
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
            player_id=0,
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
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0) or 0
            output_tokens = response.usage_metadata.get("output_tokens", 0) or 0
        estimated_cost_usd: float | None = None
        if self._input_price_per_million is not None and self._output_price_per_million is not None:
            estimated_cost_usd = (
                input_tokens * self._input_price_per_million
                + output_tokens * self._output_price_per_million
            ) / 1_000_000
        try:
            action = _extract_move(content)
        except ValueError as error:
            return LiveActionResult(
                action=None,
                success=False,
                latency=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                error_details=str(error),
            )
        return LiveActionResult(
            action=action,
            success=True,
            latency=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
