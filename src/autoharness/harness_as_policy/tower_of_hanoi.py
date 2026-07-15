"""Tower of Hanoi environment adapter using TextArena."""

from __future__ import annotations

import re
from typing import Any

import textarena as ta

from autoharness.harness_as_policy.models import StepResult

DIFFICULTY_MAP: dict[str, tuple[str, int]] = {
    "v0": ("TowerOfHanoi-v0", 14),
    "medium": ("TowerOfHanoi-v0-medium", 30),
    "hard": ("TowerOfHanoi-v0-hard", 62),
    "hardcore": ("TowerOfHanoi-v0-hardcore", 126),
}

BRACKETED_MOVE_RE = re.compile(r"\s*\[([ABC])\s*,?\s*([ABC])\]\s*", re.IGNORECASE)

INVALID_MOVE_SIGNAL = "attempted an invalid move"


class TowerOfHanoiAdapter:
    """TextArena Tower of Hanoi adapter.

    Validates actions before submission: exactly one bracketed move.
    Treats TextArena invalid-move signals as immediate illegal transitions.
    """

    def __init__(self, difficulty: str = "v0") -> None:
        if difficulty not in DIFFICULTY_MAP:
            raise ValueError(
                f"Unknown difficulty: {difficulty}. Choose from {list(DIFFICULTY_MAP)}"
            )
        self._difficulty = difficulty
        self._env_id, self._max_steps = DIFFICULTY_MAP[difficulty]
        self._env: ta.Env | None = None
        self._state: Any = None
        self._inner_env: Any = None
        self._num_disks: int | None = None
        self._observation: str = ""

    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def rules(self) -> str:
        return (
            "Tower of Hanoi: move all disks from peg A to peg C. "
            "You may only move one disk at a time, and you cannot place "
            "a larger disk on top of a smaller disk."
        )

    @property
    def action_format(self) -> str:
        return "Submit exactly one move in bracketed format, e.g. [A C] or [A, C]."

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def create(self) -> None:
        self._env = ta.make(self._env_id)
        # Find and save the inner env reference
        e: Any = self._env
        while hasattr(e, "env"):
            e = e.env
        self._inner_env = e
        self._num_disks = int(e.num_disks)
        self._state = None

    def reset(self, seed: int | None = None) -> str:
        if self._env is None:
            raise RuntimeError("Call create() before reset().")
        self._env.reset(num_players=1, seed=seed)
        # Capture the state created by reset
        self._state = self._inner_env.state
        obs_id, obs_text = self._env.get_observation()
        self._observation = str(obs_text) if obs_text is not None else ""
        return self._observation

    def step(self, action: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("Call create() before step().")
        # Pre-submission validation: exactly one bracketed move
        if not action or not action.strip():
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Malformed action: empty or whitespace-only output",
            )
        match = BRACKETED_MOVE_RE.fullmatch(action)
        if not match:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=("Malformed action: expected exactly one bracketed move, e.g. [A C]"),
            )
        # Submit to TextArena
        done, _ = self._env.step(action=action)
        obs_id, obs_text = self._env.get_observation()
        self._observation = str(obs_text) if obs_text is not None else ""
        # Check for invalid move signal
        if INVALID_MOVE_SIGNAL in self._observation.lower():
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Invalid move detected in observation",
            )
        # Determine reward and termination
        if done:
            reward = float(self._state.rewards.get(0, 0.0))
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
            reward=self._completion_fraction(),
            terminated=False,
            feedback="",
        )

    def _completion_fraction(self) -> float:
        """Fraction of disks correctly stacked from the base on peg C.

        Mirrors TextArena's completion metric without calling private APIs.
        """
        if self._state is None or self._num_disks is None:
            raise RuntimeError("Call create() and reset() before reading board progress.")
        goal = list(range(self._num_disks, 0, -1))
        tower_c = self._state.game_state["towers"]["C"]
        correct = 0
        for placed, expected in zip(tower_c, goal, strict=False):
            if placed == expected:
                correct += 1
            else:
                break
        return correct / self._num_disks
