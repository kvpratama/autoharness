"""Rollout evaluator: runs one episode of a policy against an environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.executor import ExecutionResult, PolicyExecutor
from autoharness.harness_as_policy.models import (
    ActionAttempt,
    AttemptErrorPhase,
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

    def initial_observation(self, seed: int) -> str:
        """Create and reset the environment for one seeded initial board."""
        self._adapter.create()
        return self._adapter.reset(seed=seed)

    def evaluate_actions(
        self, provider: ActionProvider, seed: int | None = None, checks_legality: bool = False
    ) -> RolloutResult:
        """Run one rollout using a callback that supplies actions."""
        try:
            self._adapter.create()
        except Exception as error:
            return self._result(
                [],
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
                [],
                0,
                TerminationReason.EXECUTION_FAILURE,
                f"Environment reset failed: {error}",
            )

        steps: list[StepResult] = []
        attempt_records: list[ActionAttempt] = []
        attempts = 0
        for _ in range(self._adapter.max_steps):
            try:
                outcome = provider(observation)
            except Exception as error:
                failure = f"Action provider failed: {error}"
                return self._record_failure(
                    steps,
                    attempt_records,
                    attempts,
                    TerminationReason.EXECUTION_FAILURE,
                    observation,
                    None,
                    None,
                    failure,
                    AttemptErrorPhase.POLICY_EXECUTION,
                )
            if not outcome.success or outcome.output is None:
                reason = (
                    TerminationReason.CONTRACT_FAILURE
                    if outcome.failure_type == "contract_failure"
                    else TerminationReason.EXECUTION_FAILURE
                )
                return self._record_failure(
                    steps,
                    attempt_records,
                    attempts,
                    reason,
                    observation,
                    None,
                    None,
                    outcome.error_details or "",
                    AttemptErrorPhase.POLICY_EXECUTION,
                    failure_summary=outcome.error_details,
                )
            action = outcome.output
            attempts += 1
            if checks_legality and outcome.is_legal_action is not True:
                failure = (
                    f"Policy legality checker rejected action {action!r} "
                    f"(checker={outcome.is_legal_action!r})"
                )
                return self._record_failure(
                    steps,
                    attempt_records,
                    attempts,
                    TerminationReason.POLICY_REJECTED_ACTION,
                    observation,
                    action,
                    False,
                    failure,
                    AttemptErrorPhase.POLICY_LEGALITY,
                )
            try:
                step = self._adapter.step(action)
            except Exception as error:
                failure = f"Environment step failed: {error}"
                return self._record_failure(
                    steps,
                    attempt_records,
                    attempts,
                    TerminationReason.EXECUTION_FAILURE,
                    observation,
                    action,
                    outcome.is_legal_action if checks_legality else None,
                    failure,
                    AttemptErrorPhase.ENVIRONMENT_STEP,
                )
            steps.append(step)
            attempt_records.append(
                ActionAttempt(
                    observation,
                    action,
                    outcome.is_legal_action if checks_legality else None,
                    step.is_legal,
                    step.observation,
                    step.reward,
                    step.terminated,
                    step.feedback,
                    None if step.is_legal else AttemptErrorPhase.ENVIRONMENT_STEP,
                )
            )
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
                return self._result(steps, attempt_records, attempts, reason, detail)
            if step.terminated:
                return self._result(
                    steps,
                    attempt_records,
                    attempts,
                    TerminationReason.ENVIRONMENT_TERMINATION,
                    None,
                )
            observation = step.observation
        return self._result(steps, attempt_records, attempts, TerminationReason.STEP_LIMIT, None)

    def _record_failure(
        self,
        steps: list[StepResult],
        attempt_records: list[ActionAttempt],
        attempts: int,
        reason: TerminationReason,
        observation: str,
        action: str | None,
        policy_legal: bool | None,
        feedback: str,
        error_phase: AttemptErrorPhase,
        failure_summary: str | None = None,
    ) -> RolloutResult:
        attempt_records.append(
            ActionAttempt(
                observation=observation,
                action=action,
                policy_legal=policy_legal,
                environment_legal=None,
                resulting_observation=None,
                reward=None,
                terminated=None,
                feedback=feedback,
                error_phase=error_phase,
            )
        )
        return self._result(
            steps,
            attempt_records,
            attempts,
            reason,
            failure_summary if failure_summary is not None else feedback,
            observation,
        )

    @staticmethod
    def _result(
        steps: list[StepResult],
        attempt_records: list[ActionAttempt],
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
            attempts=attempt_records,
        )
