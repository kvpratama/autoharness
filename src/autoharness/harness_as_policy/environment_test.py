"""Tests for the environment adapter protocol."""

from __future__ import annotations

from typing import Protocol

from src.autoharness.harness_as_policy.environment import EnvironmentAdapter


def test_protocol_is_runtime_checkable() -> None:
    """EnvironmentAdapter is a runtime-checkable protocol."""
    assert issubclass(EnvironmentAdapter, Protocol)
