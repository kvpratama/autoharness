"""Rollout evaluator: runs one episode of a policy against an environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.executor import ExecutionResult, PolicyExecutor
from autoharness.harness_as_policy.models import (
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)

ActionProvider = Callable[[str], ExecutionResult]


class ExecutorProtocol(Protocol):
    """Protocol for policy executors."""

    def execute(self, source: str, observation: str) -> ExecutionResult: ...


class RolloutEvaluator:
    """Evaluate action providers against one environment adapter."""

    def __init__(
        self, adapter: EnvironmentAdapter, executor: ExecutorProtocol | None = None
    ) -> None:
        self._adapter = adapter
        self._executor = executor or PolicyExecutor()

    def evaluate(self, source: str, seed: int | None = None) -> RolloutResult:
        """Run one generated-policy rollout and return the result."""
        return self.evaluate_actions(
            lambda observation: self._executor.execute(source, observation),
            seed=seed,
            checks_legality=True,
        )

    def evaluate_actions(
        self, provider: ActionProvider, seed: int | None = None, checks_legality: bool = False
    ) -> RolloutResult:
        """Run one rollout using a callback that supplies actions."""
        try:
            self._adapter.create()
        except Exception as error:
            return self._result(
                [],
                0,
                TerminationReason.EXECUTION_FAILURE,
                f"Environment creation failed: {error}",
            )
        try:
            observation = self._adapter.reset(seed=seed)
        except Exception as error:
            return self._result(
                [],
                0,
                TerminationReason.EXECUTION_FAILURE,
                f"Environment reset failed: {error}",
            )

        steps: list[StepResult] = []
        attempts = 0
        for _ in range(self._adapter.max_steps):
            outcome = provider(observation)
            if not outcome.success or outcome.output is None:
                reason = (
                    TerminationReason.CONTRACT_FAILURE
                    if outcome.failure_type == "contract_failure"
                    else TerminationReason.EXECUTION_FAILURE
                )
                return self._result(steps, attempts, reason, outcome.error_details)
            action = outcome.output
            attempts += 1
            if checks_legality and outcome.is_legal_action is not True:
                return self._result(
                    steps,
                    attempts,
                    TerminationReason.POLICY_REJECTED_ACTION,
                    f"Policy legality checker rejected action {action!r} "
                    f"(checker={outcome.is_legal_action!r})",
                    observation,
                )
            step = self._adapter.step(action)
            steps.append(step)
            if not step.is_legal:
                reason = (
                    TerminationReason.LEGALITY_DISAGREEMENT
                    if checks_legality
                    else TerminationReason.ILLEGAL_ACTION
                )
                detail = step.feedback or "Illegal action"
                if checks_legality:
                    detail = (
                        "Legality disagreement: checker=True, environment=False; "
                        f"environment feedback: {detail}"
                    )
                return self._result(steps, attempts, reason, detail)
            if step.terminated:
                return self._result(
                    steps, attempts, TerminationReason.ENVIRONMENT_TERMINATION, None
                )
            observation = step.observation
        return self._result(steps, attempts, TerminationReason.STEP_LIMIT, None)

    @staticmethod
    def _result(
        steps: list[StepResult],
        attempts: int,
        reason: TerminationReason,
        failure: str | None,
        last_observation: str | None = None,
    ) -> RolloutResult:
        failed = reason in (
            TerminationReason.ILLEGAL_ACTION,
            TerminationReason.LEGALITY_DISAGREEMENT,
            TerminationReason.POLICY_REJECTED_ACTION,
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        )
        reward = 0.0 if failed else (steps[-1].reward if steps else 0.0)
        legal = sum(step.is_legal for step in steps)
        return RolloutResult(
            steps=steps,
            heuristic=0.0 if failed else heuristic(is_legal=True, reward=reward),
            terminal_reward=reward,
            legal_action_count=legal,
            termination_reason=reason,
            failure_summary=failure,
            last_observation=last_observation or (steps[-1].observation if steps else None),
            action_attempt_count=attempts,
        )
