"""TextArena Blackjack environment adapter."""

from __future__ import annotations

import re
from typing import Any, Protocol, cast, runtime_checkable

import textarena as ta

from autoharness.harness_as_policy.models import StepResult


@runtime_checkable
class _BlackjackState(Protocol):
    """Structural interface for the TextArena Blackjack game-state object.

    The concrete type is an internal TextArena detail; we only rely on
    ``.rewards`` (a mapping from player-id to float) and ``.game_state``
    (a plain ``dict`` of summary statistics).
    """

    @property
    def rewards(self) -> dict[int, float]: ...

    @property
    def game_state(self) -> dict[str, Any]: ...


class _BlackjackEnv(Protocol):
    """Structural interface for the unwrapped inner TextArena environment.

    After peeling off all ``env`` wrapper layers we only need ``state``
    to read back the current game state after each step.
    """

    @property
    def state(self) -> _BlackjackState: ...


BLACKJACK_ACTION_RE = re.compile(r"\s*\[(hit|stand)\]\s*", re.IGNORECASE)
BLACKJACK_MAX_STEPS = 50


class BlackjackAdapter:
    """TextArena Blackjack adapter for the five-hand standard environment."""

    def __init__(self) -> None:
        self._env: ta.Env | None = None
        self._inner_env: _BlackjackEnv | None = None
        self._state: _BlackjackState | None = None
        self._observation = ""

    @property
    def env_id(self) -> str:
        return "Blackjack-v0"

    @property
    def rules(self) -> str:
        return (
            "Blackjack: maximize wins over five hands against a dealer that hits below 17. "
            "Card values are 2-10, face cards are 10, and aces are 1 or 11."
        )

    @property
    def action_format(self) -> str:
        return "Submit exactly one action: [Hit] or [Stand]."

    @property
    def max_steps(self) -> int:
        return BLACKJACK_MAX_STEPS

    def create(self) -> None:
        """Create the wrapped TextArena environment."""
        self._env = ta.make(self.env_id)
        environment: object = self._env
        while hasattr(environment, "env"):
            environment = getattr(environment, "env")  # noqa: B009 – `object` has no `.env`
        self._inner_env = cast(_BlackjackEnv, environment)
        self._state = None

    def reset(self, seed: int | None = None) -> str:
        """Reset Blackjack and return its initial observation."""
        if self._env is None:
            raise RuntimeError("Call create() before reset().")
        self._env.reset(num_players=1, seed=seed)
        assert self._inner_env is not None
        self._state = self._inner_env.state
        _observation_id, observation = self._env.get_observation()
        self._observation = str(observation) if observation is not None else ""
        return self._observation

    def step(self, action: str) -> StepResult:
        """Validate and submit a Blackjack action."""
        if self._env is None:
            raise RuntimeError("Call create() before step().")
        match = BLACKJACK_ACTION_RE.fullmatch(action)
        if match is None:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Malformed action: expected exactly [Hit] or [Stand]",
            )
        canonical_action = f"[{match.group(1).title()}]"
        done, _info = self._env.step(action=canonical_action)
        _observation_id, observation = self._env.get_observation()
        self._observation = str(observation) if observation is not None else ""
        if done:
            assert self._state is not None
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=True,
                reward=float(self._state.rewards.get(0, 0.0)),
                terminated=True,
                feedback="",
            )
        return StepResult(
            observation=self._observation,
            action=action,
            is_legal=True,
            reward=self._completion_reward(),
            terminated=False,
            feedback="",
        )

    def _completion_reward(self) -> float:
        """Return normalized progress across completed Blackjack hands."""
        if self._state is None:
            raise RuntimeError("Call create() and reset() before reading Blackjack progress.")
        summary = self._state.game_state["results_summary"]
        hands = int(self._state.game_state["num_hands"])
        return (float(summary["win"]) + 0.5 * float(summary["draw"])) / hands
