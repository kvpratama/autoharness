"""Tests for the environment adapter protocol."""

from __future__ import annotations

from typing import Protocol

from autoharness.environments.base import EnvironmentAdapter


def test_protocol_is_runtime_checkable() -> None:
    """EnvironmentAdapter is a runtime-checkable protocol."""
    assert issubclass(EnvironmentAdapter, Protocol)
