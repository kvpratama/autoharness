"""Held-out policy evaluation and optional live-LLM baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.environments.registry import EnvironmentSpec
from autoharness.harness_as_policy.executor import PolicyExecutor
from autoharness.harness_as_policy.models import TerminationReason
from autoharness.harness_as_policy.rollout import ExecutorProtocol, RolloutEvaluator


@dataclass
class EvaluationResult:
    """Result of evaluating a policy on one environment variant."""

    env_id: str
    solved: bool
    reward: float
    legal_action_count: int
    steps_used: int
    optimal_steps: int
    termination_reason: TerminationReason | None
    failure_summary: str | None
    latency: float
    execution_failure: bool


def evaluate_policy_on_env(
    adapter: EnvironmentAdapter,
    executor: ExecutorProtocol,
    source: str,
    optimal_steps: int = 0,
) -> EvaluationResult:
    """Evaluate a generated policy on one environment without model calls.

    ``steps_used`` is the number of environment transitions applied (length of
    the rollout step list), not the number of executor attempts.
    """
    start = time.monotonic()
    rollout = RolloutEvaluator(adapter=adapter, executor=executor).evaluate(source=source)
    latency = time.monotonic() - start
    execution_failure = rollout.termination_reason in (
        TerminationReason.EXECUTION_FAILURE,
        TerminationReason.CONTRACT_FAILURE,
    )
    return EvaluationResult(
        env_id=adapter.env_id,
        solved=rollout.terminal_reward >= 1.0,
        reward=rollout.terminal_reward,
        legal_action_count=rollout.legal_action_count,
        steps_used=len(rollout.steps),
        optimal_steps=optimal_steps or adapter.max_steps,
        termination_reason=rollout.termination_reason,
        failure_summary=rollout.failure_summary,
        latency=latency,
        execution_failure=execution_failure,
    )


def evaluate_policy(
    source: str,
    spec: EnvironmentSpec,
    executor: ExecutorProtocol | None = None,
) -> list[EvaluationResult]:
    """Evaluate a generated policy across the selected registry suite.

    Zero model calls — uses PolicyExecutor directly.
    """
    policy_executor = executor or PolicyExecutor()
    return [
        evaluate_policy_on_env(
            adapter=case.create_adapter(),
            executor=policy_executor,
            source=source,
            optimal_steps=case.optimal_steps,
        )
        for case in spec.evaluation_cases
    ]


def format_evaluation_summary(
    results: list[EvaluationResult],
    family: str = "tower-of-hanoi",
) -> str:
    """Format evaluation results as a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Policy Evaluation Summary")
    lines.append("=" * 60)
    max_disk_solved = 0
    for r in results:
        status = "SOLVED" if r.solved else "FAILED"
        lines.append(f"  {r.env_id}: {status}")
        lines.append(f"    Reward: {r.reward}")
        lines.append(f"    Steps: {r.steps_used}/{r.optimal_steps}")
        lines.append(f"    Legal actions: {r.legal_action_count}")
        if r.termination_reason is not None:
            lines.append(f"    Termination: {r.termination_reason}")
        if r.failure_summary:
            lines.append(f"    Failure: {r.failure_summary}")
        if r.execution_failure:
            lines.append("    Execution failure: yes")
        lines.append(f"    Latency: {r.latency:.3f}s")
        lines.append("")
        if "hardcore" in r.env_id and r.solved:
            max_disk_solved = 6
        elif "hard" in r.env_id and r.solved:
            max_disk_solved = max(max_disk_solved, 5)
        elif "medium" in r.env_id and r.solved:
            max_disk_solved = max(max_disk_solved, 4)
        elif r.solved:
            max_disk_solved = max(max_disk_solved, 3)
    if family == "tower-of-hanoi":
        lines.append(f"  Largest disk count solved: {max_disk_solved}")
    lines.append("=" * 60)
    return "\n".join(lines)
