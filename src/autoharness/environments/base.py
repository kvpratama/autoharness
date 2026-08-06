"""Generic environment adapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from autoharness.environments.models import StepResult


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Protocol for environment adapters used by the optimizer and evaluators."""

    @property
    def env_id(self) -> str:
        """Unique environment identifier."""

    @property
    def rules(self) -> str:
        """Human-readable environment rules."""

    @property
    def action_format(self) -> str:
        """Description of the expected action format."""

    @property
    def max_steps(self) -> int:
        """Maximum number of policy actions in one rollout."""

    def create(self) -> None:
        """Create the underlying environment instance."""

    def reset(self, seed: int | None = None) -> str:
        """Reset the environment and return the initial observation."""

    def step(self, action: str) -> StepResult:
        """Submit an action and return the normalized step result.

        Legal nonterminal steps must report ``0.0`` on ``reward``. Environment
        reward is exposed only on terminal steps.
        """
