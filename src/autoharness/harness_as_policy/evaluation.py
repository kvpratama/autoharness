"""Held-out policy evaluation and optional live-LLM baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from autoharness.harness_as_policy.environment import EnvironmentAdapter
from autoharness.harness_as_policy.executor import PolicyExecutor
from autoharness.harness_as_policy.models import TerminationReason
from autoharness.harness_as_policy.rollout import ExecutorProtocol, RolloutEvaluator
from autoharness.harness_as_policy.tower_of_hanoi import TowerOfHanoiAdapter


@dataclass
class EvaluationResult:
    """Result of evaluating a policy on one environment variant."""

    env_id: str
    solved: bool
    reward: float
    legal_action_count: int
    steps_used: int
    optimal_steps: int
    illegal_action_reason: str | None
    latency: float
    execution_failure: bool


DIFFICULTIES = [
    ("v0", "TowerOfHanoi-v0", 14, 7),
    ("medium", "TowerOfHanoi-v0-medium", 30, 15),
    ("hard", "TowerOfHanoi-v0-hard", 62, 31),
    ("hardcore", "TowerOfHanoi-v0-hardcore", 126, 63),
]


def _optimal_steps(disks: int) -> int:
    """Optimal number of moves to solve n-disk Tower of Hanoi (2^n - 1)."""
    return (2**disks) - 1  # 3 disks -> 7, 4 -> 15, etc.


def evaluate_policy_on_env(
    adapter: EnvironmentAdapter,
    executor: ExecutorProtocol,
    source: str,
    optimal_steps: int = 0,
) -> EvaluationResult:
    """Evaluate a generated policy on one environment without model calls."""
    start = time.monotonic()
    rollout = RolloutEvaluator(adapter=adapter, executor=executor).evaluate(source=source)
    latency = time.monotonic() - start
    execution_failure = rollout.termination_reason in (
        TerminationReason.EXECUTION_FAILURE,
        TerminationReason.CONTRACT_FAILURE,
    )
    illegal_action_reason: str | None = None
    if rollout.termination_reason == TerminationReason.ILLEGAL_ACTION:
        illegal_action_reason = rollout.failure_summary or "Illegal action"
    elif rollout.termination_reason == TerminationReason.STEP_LIMIT:
        illegal_action_reason = "step_limit"

    return EvaluationResult(
        env_id=adapter.env_id,
        solved=rollout.terminal_reward >= 1.0,
        reward=rollout.terminal_reward,
        legal_action_count=rollout.legal_action_count,
        steps_used=len(rollout.steps),
        optimal_steps=optimal_steps or adapter.max_steps,
        illegal_action_reason=illegal_action_reason,
        latency=latency,
        execution_failure=execution_failure,
    )


def evaluate_policy(
    source: str,
    difficulties: list[tuple[str, str, int, int]] | None = None,
) -> list[EvaluationResult]:
    """Evaluate a generated policy across all difficulty variants.

    Zero model calls — uses PolicyExecutor directly.
    """
    if difficulties is None:
        difficulties = DIFFICULTIES
    results: list[EvaluationResult] = []
    executor = PolicyExecutor()
    for diff_key, _env_id, _max_steps_var, optimal in difficulties:
        adapter = TowerOfHanoiAdapter(difficulty=diff_key)
        result = evaluate_policy_on_env(
            adapter=adapter,
            executor=executor,
            source=source,
            optimal_steps=optimal,
        )
        results.append(result)
    return results


def format_evaluation_summary(
    results: list[EvaluationResult],
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
        if r.illegal_action_reason:
            lines.append(f"    Illegal reason: {r.illegal_action_reason}")
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
    lines.append(f"  Largest disk count solved: {max_disk_solved}")
    lines.append("=" * 60)
    return "\n".join(lines)
