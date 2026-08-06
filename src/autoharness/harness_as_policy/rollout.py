"""Rollout evaluator: runs one episode of a policy against an environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from autoharness.environments.base import EnvironmentAdapter
from autoharness.environments.models import StepResult
from autoharness.harness_as_policy.executor import (
    ExecutionResult,
    PolicyExecutor,
    derive_policy_seed,
)
from autoharness.harness_as_policy.models import (
    ActionAttempt,
    AttemptErrorPhase,
    RolloutResult,
    TerminationReason,
    heuristic,
)

ActionProvider = Callable[[str], ExecutionResult]


class ExecutorSessionProtocol(Protocol):
    """Protocol for an episode-scoped policy executor session."""

    def __enter__(self) -> ExecutorSessionProtocol: ...

    def __exit__(self, *args: object) -> None: ...

    def execute(self, observation: str, *, policy_seed: int) -> ExecutionResult: ...


class ExecutorProtocol(Protocol):
    """Protocol for policy executors that create episode-scoped sessions."""

    def begin_session(self, source: str) -> ExecutorSessionProtocol: ...


class RolloutEvaluator:
    """Evaluate action providers against one environment adapter."""

    def __init__(
        self, adapter: EnvironmentAdapter, executor: ExecutorProtocol | None = None
    ) -> None:
        self._adapter = adapter
        self._executor = executor or PolicyExecutor()

    def evaluate(self, source: str, seed: int) -> RolloutResult:
        """Run one seeded generated-policy rollout and return the result."""
        action_index = 0

        with self._executor.begin_session(source) as session:

            def execute_policy(observation: str) -> ExecutionResult:
                nonlocal action_index
                policy_seed = derive_policy_seed(seed, action_index)
                action_index += 1
                return session.execute(observation, policy_seed=policy_seed)

            return self.evaluate_actions(execute_policy, seed=seed, checks_legality=True)

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
                    steps=steps,
                    attempt_records=attempt_records,
                    attempts=attempts,
                    reason=TerminationReason.EXECUTION_FAILURE,
                    observation=observation,
                    action=None,
                    policy_legal=None,
                    feedback=failure,
                    error_phase=AttemptErrorPhase.POLICY_EXECUTION,
                )
            if not outcome.success or outcome.output is None:
                reason = (
                    TerminationReason.CONTRACT_FAILURE
                    if outcome.failure_type == "contract_failure"
                    else TerminationReason.EXECUTION_FAILURE
                )
                return self._record_failure(
                    steps=steps,
                    attempt_records=attempt_records,
                    attempts=attempts,
                    reason=reason,
                    observation=observation,
                    action=None,
                    policy_legal=None,
                    feedback=outcome.error_details or "",
                    error_phase=AttemptErrorPhase.POLICY_EXECUTION,
                    failure_summary=outcome.error_details,
                    policy_seed=outcome.policy_seed,
                )
            action = outcome.output
            attempts += 1
            if checks_legality and outcome.is_legal_action is not True:
                failure = (
                    f"Policy legality checker rejected action {action!r} "
                    f"(checker={outcome.is_legal_action!r})"
                )
                return self._record_failure(
                    steps=steps,
                    attempt_records=attempt_records,
                    attempts=attempts,
                    reason=TerminationReason.POLICY_REJECTED_ACTION,
                    observation=observation,
                    action=action,
                    policy_legal=False,
                    feedback=failure,
                    error_phase=AttemptErrorPhase.POLICY_LEGALITY,
                    policy_seed=outcome.policy_seed,
                )
            try:
                step = self._adapter.step(action)
            except Exception as error:
                failure = f"Environment step failed: {error}"
                return self._record_failure(
                    steps=steps,
                    attempt_records=attempt_records,
                    attempts=attempts,
                    reason=TerminationReason.EXECUTION_FAILURE,
                    observation=observation,
                    action=action,
                    policy_legal=outcome.is_legal_action if checks_legality else None,
                    feedback=failure,
                    error_phase=AttemptErrorPhase.ENVIRONMENT_STEP,
                    policy_seed=outcome.policy_seed,
                )
            steps.append(step)
            attempt_records.append(
                ActionAttempt(
                    observation=observation,
                    action=action,
                    policy_legal=outcome.is_legal_action if checks_legality else None,
                    environment_legal=step.is_legal,
                    resulting_observation=step.observation,
                    reward=step.reward,
                    terminated=step.terminated,
                    feedback=step.feedback,
                    error_phase=(None if step.is_legal else AttemptErrorPhase.ENVIRONMENT_STEP),
                    policy_seed=outcome.policy_seed,
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
        *,
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
        policy_seed: int | None = None,
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
                policy_seed=policy_seed,
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
        terminal_step = steps[-1] if steps and steps[-1].terminated else None
        reward = 0.0 if failed or terminal_step is None else terminal_step.reward
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
