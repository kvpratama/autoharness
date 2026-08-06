"""Method-neutral environment transition models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    """Normalized result of submitting one action to an environment."""

    observation: str
    action: str | None
    is_legal: bool
    reward: float
    terminated: bool
    feedback: str
