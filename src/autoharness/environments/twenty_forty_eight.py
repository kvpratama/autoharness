"""TextArena 2048 environment adapter."""

from __future__ import annotations

import re
from typing import Protocol, cast, runtime_checkable

import textarena as ta

from autoharness.environments.models import StepResult


@runtime_checkable
class _TwentyFortyEightState(Protocol):
    """Structural interface for the inner 2048 state."""

    @property
    def rewards(self) -> dict[int, float] | None: ...

    @property
    def observations(self) -> dict[int, list[tuple[int, str, object]]]: ...


class _TwentyFortyEightEnv(Protocol):
    """Structural interface for the unwrapped 2048 environment."""

    @property
    def state(self) -> _TwentyFortyEightState: ...


TWENTY_FORTY_EIGHT_ACTION_RE = re.compile(r"\s*\[(up|down|left|right)\]\s*", re.IGNORECASE)
TWENTY_FORTY_EIGHT_MAX_STEPS = 200
INVALID_MOVE_SIGNAL = "attempted an invalid move"


class TwentyFortyEightAdapter:
    """TextArena adapter for the standard seeded 2048 environment.

    Lifecycle::

        adapter = TwentyFortyEightAdapter()
        adapter.create()
        obs = adapter.reset(seed=42)
        result = adapter.step("[left]")
    """

    def __init__(self) -> None:
        self._env: ta.Env | None = None
        self._inner_env: _TwentyFortyEightEnv | None = None
        self._state: _TwentyFortyEightState | None = None
        self._observation = ""

    @property
    def env_id(self) -> str:
        """Return the supported TextArena environment ID."""
        return "2048-v0"

    @property
    def rules(self) -> str:
        """Return a concise description of the game objective and hazards."""
        return (
            "2048: slide tiles on a 4x4 grid using [up], [down], [left], or [right] "
            "to combine matching numbers and reach higher tile values."
        )

    @property
    def action_format(self) -> str:
        """Return the canonical action syntax accepted by this adapter."""
        return "Submit exactly one action: [up], [down], [left], or [right]."

    @property
    def max_steps(self) -> int:
        """Return the standard 2048 turn limit."""
        return TWENTY_FORTY_EIGHT_MAX_STEPS

    def create(self) -> None:
        """Create the wrapped TextArena 2048 environment."""
        self._env = ta.make(self.env_id)
        environment: object = self._env
        while hasattr(environment, "env"):
            environment = getattr(environment, "env")  # noqa: B009
        self._inner_env = cast(_TwentyFortyEightEnv, environment)
        self._state = None
        self._observation = ""

    def reset(self, seed: int | None = None) -> str:
        """Reset 2048 and return its initial observation.

        Args:
            seed: Optional RNG seed for deterministic board generation.

        Returns:
            The initial TextArena observation.

        Raises:
            RuntimeError: If ``create()`` has not been called first.
        """
        if self._env is None:
            raise RuntimeError("Call create() before reset().")
        self._env.reset(num_players=1, seed=seed)
        assert self._inner_env is not None
        self._state = self._inner_env.state
        _observation_id, observation = self._env.get_observation()
        self._observation = str(observation) if observation is not None else ""
        return self._observation

    def step(self, action: str) -> StepResult:
        """Validate, submit, and normalize one 2048 action.

        Malformed actions and TextArena-rejected moves are treated as
        terminal illegal transitions with reward ``0.0``. Legal nonterminal
        transitions return ``0.0``; terminal legal transitions preserve the
        native TextArena reward.

        Args:
            action: Raw policy output expected to contain one bracketed direction.

        Returns:
            A normalized legal, illegal, terminal, or nonterminal transition.

        Raises:
            RuntimeError: If ``create()`` has not been called first.
        """
        if self._env is None:
            raise RuntimeError("Call create() before step().")
        match = TWENTY_FORTY_EIGHT_ACTION_RE.fullmatch(action)
        if match is None:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=("Malformed action: expected exactly [up], [down], [left], or [right]"),
            )

        canonical_action = f"[{match.group(1).capitalize()}]"
        done, _info = self._env.step(action=canonical_action)
        invalid_feedback = self._invalid_move_feedback()
        _observation_id, observation = self._env.get_observation()
        self._observation = str(observation) if observation is not None else ""
        if invalid_feedback is not None:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=invalid_feedback,
            )
        if done:
            assert self._state is not None
            rewards = self._state.rewards
            reward = float(rewards.get(0, 0.0)) if rewards is not None else 0.0
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=True,
                reward=reward,
                terminated=True,
                feedback="",
            )
        return StepResult(
            observation=self._observation,
            action=action,
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        )

    def _invalid_move_feedback(self) -> str | None:
        """Return the current TextArena invalid-move message, if one was emitted.

        Returns:
            The raw TextArena message string if an invalid-move signal is present
            in the most-recent game observations, otherwise ``None``.
        """
        assert self._state is not None
        for sender_id, message, _observation_type in self._state.observations.get(0, []):
            if sender_id == ta.GAME_ID and INVALID_MOVE_SIGNAL in message.lower():
                return message
        return None
