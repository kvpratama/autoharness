"""Tests for the rollout evaluator."""

from __future__ import annotations

from dataclasses import dataclass

from autoharness.harness_as_policy.executor import ExecutionResult
from autoharness.harness_as_policy.models import (
    StepResult,
    TerminationReason,
)
from autoharness.harness_as_policy.rollout import RolloutEvaluator


@dataclass
class FakeExecutor:
    """Fake executor that returns configured results."""

    step_results: list[str | None] | None = None

    def execute(self, source: str, observation: str) -> ExecutionResult:
        if not self.step_results:
            return ExecutionResult(
                success=False,
                output=None,
                latency=0.0,
                failure_type="execution_failure",
                error_details="fail",
            )
        result = self.step_results.pop(0) if self.step_results else None
        if result is None:
            return ExecutionResult(
                success=False,
                output=None,
                latency=0.0,
                failure_type="execution_failure",
                error_details="fail",
            )
        return ExecutionResult(
            success=True,
            output=result,
            latency=0.0,
            failure_type=None,
            error_details=None,
        )


class FakeAdapter:
    """Fake adapter that follows a scripted sequence of step results."""

    def __init__(
        self,
        step_results: list[StepResult] | None = None,
        truncation_reward: float = 0.0,
    ) -> None:
        self.env_id = "FakeEnv-v0"
        self.rules = "Fake rules"
        self.action_format = "[X Y]"
        self.max_steps = 10
        self._step_results = step_results or []
        self._step_index = -1
        self._truncation_reward = truncation_reward

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        self._step_index = -1
        return "initial observation"

    def step(self, action: str) -> StepResult:
        self._step_index += 1
        if self._step_results and self._step_index < len(self._step_results):
            return self._step_results[self._step_index]
        return StepResult(
            observation="obs",
            action=action,
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        )

    def truncation_reward(self) -> float:
        return self._truncation_reward


def test_rollout_solves_environment() -> None:
    """Rollout that reaches environment termination with reward 1.0 gets heuristic 1.0."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="obs1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs2",
                action="[C B]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs3",
                action="[A C]",
                is_legal=True,
                reward=1.0,
                terminated=True,
                feedback="",
            ),
        ]
    )
    executor = FakeExecutor(step_results=["[A C]", "[C B]", "[A C]"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 1.0
    assert result.termination_reason == TerminationReason.ENVIRONMENT_TERMINATION


def test_rollout_illegal_action_returns_zero() -> None:
    """First illegal action causes heuristic 0 and immediate stop."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="obs1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs2",
                action="invalid",
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Illegal",
            ),
        ]
    )
    executor = FakeExecutor(step_results=["[A C]", "invalid"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.0
    assert result.termination_reason == TerminationReason.ILLEGAL_ACTION


def test_rollout_step_limit() -> None:
    """Reaching adapter step limit without termination yields heuristic 0.5."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="obs1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs2",
                action="[C B]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs3",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
        ],
        truncation_reward=0.6,
    )
    adapter.max_steps = 3
    executor = FakeExecutor(step_results=["[A C]", "[C B]", "[A C]"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.8
    assert result.terminal_reward == 0.6
    assert result.termination_reason == TerminationReason.STEP_LIMIT


def test_rollout_execution_failure() -> None:
    """Executor failure on a step records execution failure."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="obs1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
        ]
    )
    executor = FakeExecutor(step_results=["[A C]", None])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.0
    assert result.termination_reason == TerminationReason.EXECUTION_FAILURE


def test_legal_action_count_tracked() -> None:
    """Legal action count is tracked correctly through the rollout."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="obs1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="obs2",
                action="invalid",
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Illegal",
            ),
        ]
    )
    executor = FakeExecutor(step_results=["[A C]", "invalid"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.legal_action_count == 1
