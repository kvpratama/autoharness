"""Tests for configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from src.autoharness.harness_as_policy.config import Settings


def test_default_profile_is_smoke() -> None:
    """Default profile is smoke when not set."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.profile.value == "smoke"


def test_profile_from_env() -> None:
    """Profile can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_PROFILE": "low-cost"}, clear=True):
        settings = Settings()
    assert settings.profile.value == "low-cost"


def test_artifact_root_default() -> None:
    """Default artifact root is artifacts/."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.artifact_root == "artifacts"


def test_model_required() -> None:
    """Model identifier has a default value."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.model is not None


def test_thompson_seed_default() -> None:
    """Default Thompson seed is 42."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.thompson_seed == 42


def test_execution_timeout_default() -> None:
    """Default execution timeout is 10 seconds."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.execution_timeout == 10


def test_max_source_size_default() -> None:
    """Default max source size is 32768 bytes."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.max_source_size == 32768


def test_env_id_from_env() -> None:
    """Environment ID can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_ENV": "TowerOfHanoi-v0"}, clear=True):
        settings = Settings()
    assert settings.env_id == "TowerOfHanoi-v0"


def test_refinements_override() -> None:
    """Refinement budget can be overridden via env."""
    with patch.dict(os.environ, {"AUTOHARNESS_REFINEMENTS": "5"}, clear=True):
        settings = Settings()
    assert settings.refinements == 5


def test_pricing_not_configured_by_default() -> None:
    """Input/output prices are None by default."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.input_price_per_million is None
    assert settings.output_price_per_million is None


def test_effective_refinements_smoke() -> None:
    """Smoke profile effective refinements is 8."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
    assert settings.effective_refinements == 8


def test_effective_refinements_custom() -> None:
    """Custom refinements override profile default."""
    with patch.dict(os.environ, {"AUTOHARNESS_REFINEMENTS": "5"}, clear=True):
        settings = Settings()
    assert settings.effective_refinements == 5
