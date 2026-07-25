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

    step_results: list[tuple[str, bool] | None] | None = None

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
            output=result[0],
            latency=0.0,
            is_legal_action=result[1],
            failure_type=None,
            error_details=None,
        )


class FakeAdapter:
    """Fake adapter that follows a scripted sequence of step results."""

    def __init__(self, step_results: list[StepResult] | None = None) -> None:
        self.env_id = "FakeEnv-v0"
        self.rules = "Fake rules"
        self.action_format = "[X Y]"
        self.max_steps = 10
        self._step_results = step_results or []
        self._step_index = -1
        self.step_calls: list[str] = []

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        self._step_index = -1
        return "initial observation"

    def step(self, action: str) -> StepResult:
        self.step_calls.append(action)
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
    executor = FakeExecutor(step_results=[("[A C]", True), ("[C B]", True), ("[A C]", True)])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 1.0
    assert result.termination_reason == TerminationReason.ENVIRONMENT_TERMINATION


def test_rollout_illegal_action_returns_zero() -> None:
    """Environment rejection after checker approval causes zero score and immediate stop."""
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
    executor = FakeExecutor(step_results=[("[A C]", True), ("invalid", True)])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.0
    assert result.termination_reason == TerminationReason.LEGALITY_DISAGREEMENT


def test_rollout_step_limit() -> None:
    """Reaching adapter step limit uses last-step progress reward for heuristic."""
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
                reward=0.6,
                terminated=False,
                feedback="",
            ),
        ],
    )
    adapter.max_steps = 3
    executor = FakeExecutor(step_results=[("[A C]", True), ("[C B]", True), ("[A C]", True)])
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
    executor = FakeExecutor(step_results=[("[A C]", True), None])
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
    executor = FakeExecutor(step_results=[("[A C]", True), ("invalid", True)])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.legal_action_count == 1


def test_checker_rejection_stops_before_environment_step() -> None:
    """A checker-rejected action fails closed without applying an environment step."""
    adapter = FakeAdapter()
    executor = FakeExecutor(step_results=[("[A C]", False)])

    result = RolloutEvaluator(adapter=adapter, executor=executor).evaluate("dummy source")

    assert adapter.step_calls == []
    assert result.steps == []
    assert result.heuristic == 0.0
    assert result.terminal_reward == 0.0
    assert result.legal_action_count == 0
    assert result.termination_reason.value == "policy_rejected_action"
    assert result.last_observation == "initial observation"
    assert "'[A C]'" in (result.failure_summary or "")


def test_checker_environment_legality_disagreement_returns_zero() -> None:
    """Environment rejection after checker approval is reported as disagreement."""
    adapter = FakeAdapter(
        step_results=[
            StepResult(
                observation="environment observation",
                action="[A C]",
                is_legal=False,
                reward=0.7,
                terminated=True,
                feedback="Environment says illegal",
            )
        ]
    )
    executor = FakeExecutor(step_results=[("[A C]", True)])

    result = RolloutEvaluator(adapter=adapter, executor=executor).evaluate("dummy source")

    assert adapter.step_calls == ["[A C]"]
    assert result.heuristic == 0.0
    assert result.terminal_reward == 0.0
    assert result.legal_action_count == 0
    assert result.termination_reason.value == "legality_disagreement"
    assert "checker=True" in (result.failure_summary or "")
    assert "environment=False" in (result.failure_summary or "")


def test_checker_rejection_counts_one_proposed_action_attempt() -> None:
    result = RolloutEvaluator(
        FakeAdapter(), FakeExecutor(step_results=[("[A C]", False)])
    ).evaluate("source")

    assert result.action_attempt_count == 1
    assert result.legal_action_count == 0


def test_actionless_execution_failure_does_not_count_as_attempt() -> None:
    result = RolloutEvaluator(FakeAdapter(), FakeExecutor(step_results=[None])).evaluate("source")

    assert result.action_attempt_count == 0


def test_unchecked_provider_reports_environment_illegal_action() -> None:
    adapter = FakeAdapter([StepResult("next", "bad", False, 0.0, True, "Illegal move")])

    result = RolloutEvaluator(adapter).evaluate_actions(
        lambda _observation: ExecutionResult(True, "bad", 0.01),
        seed=123,
        checks_legality=False,
    )

    assert result.termination_reason == TerminationReason.ILLEGAL_ACTION
    assert result.action_attempt_count == 1
    assert result.failure_summary == "Illegal move"


class FailingSetupAdapter(FakeAdapter):
    def __init__(self, *, reset_fails: bool) -> None:
        super().__init__()
        self._reset_fails = reset_fails

    def create(self) -> None:
        if not self._reset_fails:
            raise RuntimeError("create boom")

    def reset(self, seed: int | None = None) -> str:
        if self._reset_fails:
            raise RuntimeError("reset boom")
        return super().reset(seed)


def test_environment_creation_and_reset_failures_remain_distinct() -> None:
    creation = RolloutEvaluator(FailingSetupAdapter(reset_fails=False)).evaluate_actions(
        lambda _observation: ExecutionResult(True, "action", 0.0)
    )
    reset = RolloutEvaluator(FailingSetupAdapter(reset_fails=True)).evaluate_actions(
        lambda _observation: ExecutionResult(True, "action", 0.0)
    )

    assert creation.failure_summary == "Environment creation failed: create boom"
    assert reset.failure_summary == "Environment reset failed: reset boom"
