"""Pydantic settings for harness-as-policy."""

from __future__ import annotations

import logging

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from autoharness.harness_as_policy.models import Profile

_VALID_LOG_LEVELS: frozenset[str] = frozenset(logging.getLevelNamesMapping().keys())
_TRAINING_ROLLOUT_DEFAULTS: dict[str, int] = {"Blackjack-v0": 5}


def _validate_log_level_value(v: object) -> object:
    """Reject log-level strings not recognised by the logging module."""
    if isinstance(v, str) and v.upper() not in _VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level {v!r}. Valid levels: {sorted(_VALID_LOG_LEVELS)}")
    return v


class Settings(BaseSettings):
    """Resolved configuration for a synthesis run."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOHARNESS_",
        env_file=".env",
        extra="ignore",
    )

    model: str
    env_id: str = "TowerOfHanoi-v0"
    profile: Profile = Profile.SMOKE
    refinements: int | None = None
    artifact_root: str = "artifacts"
    thompson_seed: int = 42
    environment_seed: int = 0
    training_rollouts: int | None = Field(default=None, gt=0)
    execution_timeout: int = 10
    max_source_size: int = 32768
    log_level: str = "WARNING"
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: object) -> object:
        return _validate_log_level_value(v)

    @property
    def effective_refinements(self) -> int:
        return self.refinements if self.refinements is not None else self.profile.refinements

    @property
    def effective_training_rollouts(self) -> int:
        """Return the explicit rollout count or the policy default for this environment."""
        if self.training_rollouts is not None:
            return self.training_rollouts
        return _TRAINING_ROLLOUT_DEFAULTS.get(self.env_id, 1)


class _LogLevelOnlySettings(BaseSettings):
    """Lightweight settings used by the CLI to read AUTOHARNESS_LOG_LEVEL early.

    All fields are optional so this can be instantiated before ``model`` is known.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTOHARNESS_",
        env_file=".env",
        extra="ignore",
    )

    log_level: str | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: object) -> object:
        return _validate_log_level_value(v)
