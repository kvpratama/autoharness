"""Tests for method-neutral environment transition models."""

from autoharness.environments.models import StepResult


def test_step_result_stores_normalized_transition() -> None:
    result = StepResult(
        observation="next state",
        action="[A C]",
        is_legal=True,
        reward=1.0,
        terminated=True,
        feedback="solved",
    )

    assert result.observation == "next state"
    assert result.action == "[A C]"
    assert result.is_legal
    assert result.reward == 1.0
    assert result.terminated
    assert result.feedback == "solved"
