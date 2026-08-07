"""TextArena FrozenLake environment adapter."""

from __future__ import annotations

import re
from typing import Protocol, cast, runtime_checkable

import textarena as ta

from autoharness.environments.models import StepResult


@runtime_checkable
class _FrozenLakeState(Protocol):
    """Structural interface for the inner FrozenLake state."""

    @property
    def rewards(self) -> dict[int, float] | None: ...

    @property
    def observations(self) -> dict[int, list[tuple[int, str, object]]]: ...


class _FrozenLakeEnv(Protocol):
    """Structural interface for the unwrapped FrozenLake environment."""

    @property
    def state(self) -> _FrozenLakeState: ...


FROZEN_LAKE_ACTION_RE = re.compile(r"\s*\[(up|down|left|right)\]\s*", re.IGNORECASE)
FROZEN_LAKE_MAX_STEPS = 100
INVALID_MOVE_SIGNAL = "attempted an invalid move"


class FrozenLakeAdapter:
    """TextArena adapter for the standard seeded FrozenLake environment.

    Lifecycle::

        adapter = FrozenLakeAdapter()
        adapter.create()
        obs = adapter.reset(seed=42)
        result = adapter.step("[down]")
    """

    def __init__(self) -> None:
        self._env: ta.Env | None = None
        self._inner_env: _FrozenLakeEnv | None = None
        self._state: _FrozenLakeState | None = None
        self._observation = ""

    @property
    def env_id(self) -> str:
        """Return the supported TextArena environment ID."""
        return "FrozenLake-v0"

    @property
    def rules(self) -> str:
        """Return a concise description of the game objective and hazards."""
        return (
            "FrozenLake: navigate the visible 4x4 grid from P to G without entering a hole. "
            "Moves outside the grid are illegal."
        )

    @property
    def action_format(self) -> str:
        """Return the canonical action syntax accepted by this adapter."""
        return "Submit exactly one action: [up], [down], [left], or [right]."

    @property
    def max_steps(self) -> int:
        """Return the standard FrozenLake turn limit."""
        return FROZEN_LAKE_MAX_STEPS

    def create(self) -> None:
        """Create the wrapped TextArena FrozenLake environment."""
        self._env = ta.make(self.env_id)
        environment: object = self._env
        while hasattr(environment, "env"):
            environment = getattr(environment, "env")  # noqa: B009
        self._inner_env = cast(_FrozenLakeEnv, environment)
        self._state = None
        self._observation = ""

    def reset(self, seed: int | None = None) -> str:
        """Reset FrozenLake and return its initial observation.

        Args:
            seed: Optional RNG seed for deterministic map generation.

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
        """Validate, submit, and normalize one FrozenLake action.

        Malformed actions and TextArena-rejected wall moves are treated as
        terminal illegal transitions with reward ``0.0``.  Legal nonterminal
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
        match = FROZEN_LAKE_ACTION_RE.fullmatch(action)
        if match is None:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=("Malformed action: expected exactly [up], [down], [left], or [right]"),
            )

        canonical_action = f"[{match.group(1).lower()}]"
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
