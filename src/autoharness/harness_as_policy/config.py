"""Pydantic settings for harness-as-policy."""

from __future__ import annotations

import logging

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from autoharness.harness_as_policy.models import Profile

_VALID_LOG_LEVELS: frozenset[str] = frozenset(logging.getLevelNamesMapping().keys())


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


class _LogLevelOnlySettings(BaseSettings):
    """Lightweight settings used by the CLI to read AUTOHARNESS_LOG_LEVEL early.

    All fields are optional so this can be instantiated before ``model`` is known.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTOHARNESS_",
        env_file=".env",
        extra="ignore",
    )

    log_level: str = "WARNING"

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: object) -> object:
        return _validate_log_level_value(v)
