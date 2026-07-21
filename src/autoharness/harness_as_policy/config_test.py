"""Tests for configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from autoharness.harness_as_policy.config import Settings, _LogLevelOnlySettings

_BASE: dict[str, Any] = {"model": "test-model"}


class _EnvOnlyLogLevelSettings(_LogLevelOnlySettings):
    """Test helper that reads log level from process env only (no .env file)."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOHARNESS_",
        env_file=None,
        extra="ignore",
    )


def _settings(**overrides: Any) -> Settings:
    kwargs = dict(_BASE)
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_default_profile_is_smoke() -> None:
    """Default profile is smoke when not set."""
    settings = _settings()
    assert settings.profile.value == "smoke"


def test_profile_from_env() -> None:
    """Profile can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_PROFILE": "low-cost"}, clear=True):
        settings = _settings()
    assert settings.profile.value == "low-cost"


def test_artifact_root_default() -> None:
    """Default artifact root is artifacts/."""
    settings = _settings()
    assert settings.artifact_root == "artifacts"


def test_model_required() -> None:
    """Model identifier must be provided."""
    settings = _settings(model="anthropic:claude-3-opus")
    assert settings.model == "anthropic:claude-3-opus"


def test_thompson_seed_default() -> None:
    """Default Thompson seed is 42."""
    settings = _settings()
    assert settings.thompson_seed == 42


def test_stochastic_training_defaults() -> None:
    settings = _settings()
    assert settings.training_rollouts is None
    assert settings.environment_seed == 0


def test_training_rollouts_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _settings(training_rollouts=0)


def test_stochastic_training_settings_from_environment() -> None:
    with patch.dict(
        os.environ,
        {"AUTOHARNESS_TRAINING_ROLLOUTS": "7", "AUTOHARNESS_ENVIRONMENT_SEED": "99"},
        clear=True,
    ):
        settings = _settings()
    assert settings.training_rollouts == 7
    assert settings.environment_seed == 99


def test_execution_timeout_default() -> None:
    """Default execution timeout is 10 seconds."""
    settings = _settings()
    assert settings.execution_timeout == 10


def test_max_source_size_default() -> None:
    """Default max source size is 32768 bytes."""
    settings = _settings()
    assert settings.max_source_size == 32768


def test_env_id_from_env() -> None:
    """Environment ID can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_ENV": "TowerOfHanoi-v0"}, clear=True):
        settings = _settings()
    assert settings.env_id == "TowerOfHanoi-v0"


def test_refinements_override() -> None:
    """Refinement budget can be overridden via env."""
    with patch.dict(os.environ, {"AUTOHARNESS_REFINEMENTS": "5"}, clear=True):
        settings = _settings()
    assert settings.refinements == 5


def test_pricing_not_configured_by_default() -> None:
    """Input/output prices are None by default."""
    settings = _settings()
    assert settings.input_price_per_million is None
    assert settings.output_price_per_million is None


def test_effective_refinements_smoke() -> None:
    """Smoke profile effective refinements is 8."""
    settings = _settings()
    assert settings.effective_refinements == 8


def test_effective_refinements_custom() -> None:
    """Custom refinements override profile default."""
    with patch.dict(os.environ, {"AUTOHARNESS_REFINEMENTS": "5"}, clear=True):
        settings = _settings()
    assert settings.effective_refinements == 5


def test_effective_refinements_full_search() -> None:
    """Full-search profile resolves to 256 refinements."""
    with patch.dict(os.environ, {"AUTOHARNESS_PROFILE": "full-search"}, clear=True):
        settings = _settings()
    assert settings.profile.value == "full-search"
    assert settings.effective_refinements == 256


def test_effective_refinements_full_search_override() -> None:
    """Full-search profile with refinements override yields the override."""
    with patch.dict(
        os.environ,
        {"AUTOHARNESS_PROFILE": "full-search", "AUTOHARNESS_REFINEMENTS": "10"},
        clear=True,
    ):
        settings = _settings()
    assert settings.profile.value == "full-search"
    assert settings.effective_refinements == 10


def test_log_level_only_settings_unset_is_none() -> None:
    """Early log-level settings leave log_level unset when env/.env omit it."""
    with patch.dict(os.environ, {}, clear=True):
        settings = _EnvOnlyLogLevelSettings()
    assert settings.log_level is None


def test_log_level_only_settings_from_process_env() -> None:
    """Early log-level settings load AUTOHARNESS_LOG_LEVEL from the process env."""
    with patch.dict(os.environ, {"AUTOHARNESS_LOG_LEVEL": "INFO"}, clear=True):
        settings = _EnvOnlyLogLevelSettings()
    assert settings.log_level == "INFO"


def test_log_level_only_settings_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Early log-level settings load AUTOHARNESS_LOG_LEVEL from a .env file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOHARNESS_LOG_LEVEL", raising=False)
    (tmp_path / ".env").write_text("AUTOHARNESS_LOG_LEVEL=DEBUG\n", encoding="utf-8")
    settings = _LogLevelOnlySettings()
    assert settings.log_level == "DEBUG"


def test_log_level_only_settings_rejects_invalid_level() -> None:
    """Early log-level settings validate AUTOHARNESS_LOG_LEVEL values."""
    with (
        patch.dict(os.environ, {"AUTOHARNESS_LOG_LEVEL": "NOTALEVEL"}, clear=True),
        pytest.raises(ValidationError),
    ):
        _EnvOnlyLogLevelSettings()
