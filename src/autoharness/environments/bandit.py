"""TextArena Bandit environment adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import textarena as ta

from autoharness.environments.models import StepResult


@runtime_checkable
class _BanditState(Protocol):
    """Structural interface for the inner TextArena Bandit state."""

    @property
    def rewards(self) -> dict[int, float] | None: ...

    @property
    def observations(self) -> dict[int, list[tuple[int, str, object]]]: ...


class _BanditEnv(Protocol):
    """Structural interface for the unwrapped TextArena Bandit environment."""

    @property
    def state(self) -> _BanditState: ...


@dataclass(frozen=True)
class BanditVariant:
    """Intrinsic configuration for one TextArena Bandit variant."""

    env_id: str
    buttons: tuple[str, ...]
    exploration_steps: int

    @property
    def max_steps(self) -> int:
        """Return the exploration budget plus the final button selection."""
        return self.exploration_steps + 1


BANDIT_VARIANTS: dict[str, BanditVariant] = {
    "Bandit-v0": BanditVariant(
        env_id="Bandit-v0",
        buttons=("red", "blue", "green", "yellow", "purple"),
        exploration_steps=20,
    ),
    "Bandit-v0-hard": BanditVariant(
        env_id="Bandit-v0-hard",
        buttons=(
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
            "black",
        ),
        exploration_steps=40,
    ),
}

_INVALID_MOVE_SIGNAL = "attempted an invalid move"


class BanditAdapter:
    """TextArena adapter for the standard and hard single-player Bandits.

    Actions are bracketed button names such as ``[red]``. The adapter accepts
    surrounding whitespace and case variations, then submits canonical
    lowercase actions to TextArena.
    """

    def __init__(self, env_id: str = "Bandit-v0") -> None:
        """Initialise the adapter for the given Bandit variant.

        Args:
            env_id: TextArena environment identifier selecting the Bandit
                variant.  Must be a key present in ``BANDIT_VARIANTS``
                (e.g. ``"Bandit-v0"`` or ``"Bandit-v0-hard"``).  Defaults
                to ``"Bandit-v0"``.

        Raises:
            ValueError: If *env_id* is not a recognised Bandit variant.  The
                error message lists all valid identifiers.
        """
        try:
            self._variant = BANDIT_VARIANTS[env_id]
        except KeyError as error:
            valid_ids = ", ".join(sorted(BANDIT_VARIANTS))
            raise ValueError(
                f"Unknown Bandit environment {env_id!r}. Valid options: {valid_ids}"
            ) from error
        self._env: ta.Env | None = None
        self._inner_env: _BanditEnv | None = None
        self._state: _BanditState | None = None
        self._observation = ""
        button_pattern = "|".join(self._variant.buttons)
        self._action_re = re.compile(rf"\s*\[({button_pattern})\]\s*", re.IGNORECASE)

    @property
    def env_id(self) -> str:
        """Return the selected TextArena environment ID."""
        return self._variant.env_id

    @property
    def rules(self) -> str:
        """Return a concise description of the Bandit objective."""
        buttons = ", ".join(self._variant.buttons)
        return (
            f"Bandit: use {self._variant.exploration_steps} exploratory turns to estimate the "
            f"best Bernoulli button among {buttons}, then select the button with the highest mean."
        )

    @property
    def action_format(self) -> str:
        """Return the canonical action syntax accepted by this adapter."""
        buttons = ", ".join(f"[{button}]" for button in self._variant.buttons)
        return f"Submit exactly one action: {buttons}."

    @property
    def max_steps(self) -> int:
        """Return the exploration budget plus the final selection."""
        return self._variant.max_steps

    def create(self) -> None:
        """Create the wrapped TextArena Bandit environment."""
        self._env = ta.make(self.env_id)
        environment: object = self._env
        while hasattr(environment, "env"):
            environment = getattr(environment, "env")  # noqa: B009
        self._inner_env = cast(_BanditEnv, environment)
        self._state = None
        self._observation = ""

    def reset(self, seed: int | None = None) -> str:
        """Reset Bandit to an initial state and return the first observation.

        Args:
            seed: Optional integer seed passed through to the underlying
                TextArena environment to make episodes reproducible.  When
                ``None`` (the default) the environment uses its own default
                seeding strategy.

        Returns:
            The initial observation string produced by the TextArena
            environment after the reset, or an empty string when the
            environment returns no observation.

        Raises:
            RuntimeError: If :meth:`create` has not been called before
                this method.
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
        """Validate, submit, and normalise one Bandit action.

        The method performs three checks in order:

        1. **Format validation** — the action string must match
           ``[<button>]`` (case-insensitive, surrounding whitespace
           allowed).  A malformed action short-circuits with
           ``is_legal=False`` and ``terminated=True`` without touching
           the environment.
        2. **Invalid-move detection** — the canonical action is submitted
           to the TextArena environment; if the environment signals an
           invalid move the step is returned as illegal and terminated.
        3. **Normal progression** — on a legal, non-terminal step the
           accumulated reward is ``0.0``; on termination the final reward
           is read from the environment state.

        Args:
            action: Raw action string from the policy, e.g.
                ``"[red]"`` or ``" [Blue] "``.

        Returns:
            A :class:`~autoharness.environments.models.StepResult` whose
            fields reflect the outcome:

            * ``observation`` — latest environment observation string.
            * ``action`` — the raw *action* string as supplied.
            * ``is_legal`` — ``False`` when the format is wrong or the
              environment rejects the move; ``True`` otherwise.
            * ``reward`` — final episode reward on termination, ``0.0``
              on all non-terminal or illegal steps.
            * ``terminated`` — ``True`` when the episode has ended
              (whether legally or due to an illegal action).
            * ``feedback`` — human-readable error description on illegal
              steps; empty string otherwise.

        Raises:
            RuntimeError: If :meth:`create` has not been called before
                this method.
        """
        if self._env is None:
            raise RuntimeError("Call create() before step().")
        match = self._action_re.fullmatch(action)
        if match is None:
            buttons = ", ".join(f"[{button}]" for button in self._variant.buttons)
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=f"Malformed action: expected exactly one of {buttons}",
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
        """Return the latest TextArena invalid-move message, if any."""
        assert self._state is not None
        for sender_id, message, _observation_type in self._state.observations.get(0, []):
            if sender_id == ta.GAME_ID and _INVALID_MOVE_SIGNAL in message.lower():
                return message
        return None
