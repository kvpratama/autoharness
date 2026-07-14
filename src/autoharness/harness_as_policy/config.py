"""Pydantic settings for harness-as-policy."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from autoharness.harness_as_policy.models import Profile


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

    @property
    def effective_refinements(self) -> int:
        return self.refinements if self.refinements is not None else self.profile.refinements
