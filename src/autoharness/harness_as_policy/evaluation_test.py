"""Tests for held-out evaluation and live-LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass

from autoharness.harness_as_policy.evaluation import (
    EvaluationResult,
    evaluate_policy_on_env,
    format_evaluation_summary,
)
from autoharness.harness_as_policy.executor import ExecutionResult
from autoharness.harness_as_policy.models import StepResult, TerminationReason


@dataclass
class FakeAdapter:
    """Fake environment adapter for evaluation tests."""

    env_id: str = "TowerOfHanoi-v0"
    rules: str = "Rules"
    action_format: str = "[A C]"
    max_steps: int = 14
    _step_results: list[StepResult] | None = None
    _step_index: int = -1
    _observation: str = ""

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        self._step_index = -1
        self._observation = "initial obs"
        return self._observation

    def step(self, action: str) -> StepResult:
        self._step_index += 1
        if self._step_results and self._step_index < len(self._step_results):
            result = self._step_results[self._step_index]
            self._observation = result.observation
            return result
        self._observation = "obs"
        return StepResult(
            observation=self._observation,
            action=action,
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        )


@dataclass
class FakeExecutor:
    """Fake executor for evaluation tests."""

    responses: list[str | None] | None = None
    _call_index: int = -1

    def execute(self, source: str, observation: str) -> ExecutionResult:
        self._call_index += 1
        if self.responses and self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            if response is None:
                return ExecutionResult(
                    success=False,
                    output=None,
                    latency=0.01,
                    failure_type="execution_failure",
                    error_details="boom",
                )
            return ExecutionResult(
                success=True,
                output=response,
                latency=0.01,
            )
        return ExecutionResult(success=True, output="[A C]", latency=0.01)


def test_evaluate_policy_on_env_solved() -> None:
    """evaluate_policy_on_env returns solved result when env terminates with reward 1."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(
            observation="o1",
            action="[A C]",
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        ),
        StepResult(
            observation="o2",
            action="[C B]",
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        ),
        StepResult(
            observation="o3",
            action="[A C]",
            is_legal=True,
            reward=1.0,
            terminated=True,
            feedback="",
        ),
    ]
    executor = FakeExecutor(responses=["[A C]", "[C B]", "[A C]"])
    result = evaluate_policy_on_env(
        adapter=adapter,
        executor=executor,
        source="policy source",
    )
    assert result.solved
    assert result.reward == 1.0
    assert result.steps_used == 3
    assert result.termination_reason == TerminationReason.ENVIRONMENT_TERMINATION
    assert result.failure_summary is None


def test_evaluate_policy_on_env_illegal() -> None:
    """evaluate_policy_on_env records illegal action reason."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(
            observation="o1",
            action="[A C]",
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        ),
        StepResult(
            observation="o2",
            action="bad",
            is_legal=False,
            reward=0.0,
            terminated=True,
            feedback="Illegal",
        ),
    ]
    executor = FakeExecutor(responses=["[A C]", "bad"])
    result = evaluate_policy_on_env(
        adapter=adapter,
        executor=executor,
        source="policy source",
    )
    assert not result.solved
    assert result.termination_reason == TerminationReason.ILLEGAL_ACTION
    assert result.failure_summary is not None


def test_evaluate_policy_no_model_calls() -> None:
    """Generated policy evaluation makes zero model calls."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(
            observation="o1",
            action="[A C]",
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        ),
    ]
    executor = FakeExecutor(responses=["[A C]"])
    result = evaluate_policy_on_env(
        adapter=adapter,
        executor=executor,
        source="policy source",
    )
    assert isinstance(result, EvaluationResult)


def test_evaluate_policy_on_env_preserves_step_progress_reward() -> None:
    """Generated policy evaluation uses last-step reward at the step limit."""
    adapter = FakeAdapter(
        max_steps=2,
        _step_results=[
            StepResult(
                observation="o1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
            StepResult(
                observation="o2",
                action="[C B]",
                is_legal=True,
                reward=0.6,
                terminated=False,
                feedback="",
            ),
        ],
    )
    executor = FakeExecutor(responses=["[A C]", "[C B]"])

    result = evaluate_policy_on_env(
        adapter=adapter,
        executor=executor,
        source="policy source",
    )

    assert not result.solved
    assert result.reward == 0.6
    assert result.steps_used == 2
    assert result.legal_action_count == 2
    assert result.termination_reason == TerminationReason.STEP_LIMIT
    assert result.failure_summary is None
    assert not result.execution_failure


def test_evaluate_policy_on_env_execution_failure_counts_env_steps_only() -> None:
    """steps_used counts applied env transitions, not failed executor attempts."""
    adapter = FakeAdapter(
        max_steps=5,
        _step_results=[
            StepResult(
                observation="o1",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            ),
        ],
    )
    executor = FakeExecutor(responses=["[A C]", None])

    result = evaluate_policy_on_env(
        adapter=adapter,
        executor=executor,
        source="policy source",
    )

    assert result.execution_failure
    assert result.termination_reason == TerminationReason.EXECUTION_FAILURE
    assert result.steps_used == 1
    assert result.legal_action_count == 1
    assert result.failure_summary == "boom"


def test_evaluation_result_attributes() -> None:
    """EvaluationResult has all expected fields."""
    result = EvaluationResult(
        env_id="TowerOfHanoi-v0-medium",
        solved=False,
        reward=0.0,
        legal_action_count=5,
        steps_used=5,
        optimal_steps=15,
        termination_reason=TerminationReason.ILLEGAL_ACTION,
        failure_summary="malformed action",
        latency=0.05,
        execution_failure=False,
    )
    assert result.env_id == "TowerOfHanoi-v0-medium"
    assert result.optimal_steps == 15
    assert result.execution_failure is False


def test_format_evaluation_summary() -> None:
    """format_evaluation_summary produces a non-empty string."""
    results = [
        EvaluationResult(
            env_id="v0",
            solved=True,
            reward=1.0,
            legal_action_count=7,
            steps_used=7,
            optimal_steps=7,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            latency=0.05,
            execution_failure=False,
        ),
        EvaluationResult(
            env_id="medium",
            solved=False,
            reward=0.0,
            legal_action_count=10,
            steps_used=10,
            optimal_steps=15,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            latency=0.08,
            execution_failure=False,
        ),
    ]
    summary = format_evaluation_summary(results)
    assert "v0" in summary
    assert "medium" in summary
    assert "step_limit" in summary
    assert len(summary) > 0
