"""Rollout evaluator: runs one episode of a policy against an environment."""

from __future__ import annotations

from typing import Protocol

from autoharness.harness_as_policy.environment import EnvironmentAdapter
from autoharness.harness_as_policy.executor import ExecutionResult, PolicyExecutor
from autoharness.harness_as_policy.models import (
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)


class ExecutorProtocol(Protocol):
    """Protocol for policy executors."""

    def execute(self, source: str, observation: str) -> ExecutionResult: ...


class RolloutEvaluator:
    """Evaluates a policy by rolling it out against an environment."""

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        executor: ExecutorProtocol | None = None,
    ) -> None:
        self._adapter = adapter
        self._executor = executor or PolicyExecutor()

    def evaluate(self, source: str, seed: int | None = None) -> RolloutResult:
        """Run one rollout and return the result."""
        try:
            self._adapter.create()
        except Exception as e:
            return RolloutResult(
                steps=[],
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.EXECUTION_FAILURE,
                failure_summary=f"Environment creation failed: {e}",
            )
        try:
            observation = self._adapter.reset(seed=seed)
        except Exception as e:
            return RolloutResult(
                steps=[],
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.EXECUTION_FAILURE,
                failure_summary=f"Environment reset failed: {e}",
            )
        steps: list[StepResult] = []
        for _ in range(self._adapter.max_steps):
            exec_result = self._executor.execute(source, observation)
            if not exec_result.success:
                ft = exec_result.failure_type or "execution_failure"
                reason = (
                    TerminationReason.CONTRACT_FAILURE
                    if ft == "contract_failure"
                    else TerminationReason.EXECUTION_FAILURE
                )
                return RolloutResult(
                    steps=steps,
                    heuristic=0.0,
                    terminal_reward=0.0,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=reason,
                    failure_summary=exec_result.error_details,
                    last_observation=steps[-1].observation if steps else None,
                )
            action = exec_result.output or ""
            step_result = self._adapter.step(action)
            steps.append(step_result)
            if not step_result.is_legal:
                return RolloutResult(
                    steps=steps,
                    heuristic=0.0,
                    terminal_reward=0.0,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=TerminationReason.ILLEGAL_ACTION,
                    failure_summary=step_result.feedback or "Illegal action",
                    last_observation=steps[-1].observation if steps else None,
                )
            if step_result.terminated:
                h = heuristic(is_legal=True, reward=step_result.reward)
                return RolloutResult(
                    steps=steps,
                    heuristic=h,
                    terminal_reward=step_result.reward,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                    failure_summary=None,
                    last_observation=steps[-1].observation if steps else None,
                )
            observation = step_result.observation
        terminal_reward = steps[-1].reward if steps else 0.0
        return RolloutResult(
            steps=steps,
            heuristic=heuristic(is_legal=True, reward=terminal_reward),
            terminal_reward=terminal_reward,
            legal_action_count=len([s for s in steps if s.is_legal]),
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            last_observation=steps[-1].observation if steps else None,
        )
