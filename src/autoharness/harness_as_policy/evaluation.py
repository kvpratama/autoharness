"""Reproducible exact-environment policy evaluation."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import cast

from autoharness.harness_as_policy.assessment import generate_episode_seeds
from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.environments.registry import EnvironmentSpec
from autoharness.harness_as_policy.executor import PolicyExecutor, policy_randomness_metadata
from autoharness.harness_as_policy.models import TerminationReason
from autoharness.harness_as_policy.rollout import ActionProvider, ExecutorProtocol, RolloutEvaluator

PAPER_1P_EPISODE_COUNT = 20
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_NAME = "paper-1p"
REWARD_METRIC = "arithmetic_mean"
LEGALITY_METRIC = "legal_actions / proposed_action_attempts"


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"Evaluation protocol {name} must be a list of integers")
    return cast(list[int], value)


@dataclass(frozen=True)
class EvaluationProtocol:
    """Persisted exact-environment inputs shared by policy evaluations."""

    env_id: str
    episode_seeds: tuple[int, ...]
    training_episode_seeds: tuple[int, ...]
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    name: str = PROTOCOL_NAME

    @property
    def episode_count(self) -> int:
        """Return the fixed number of evaluation episodes."""
        return len(self.episode_seeds)

    def prefix(self, episode_count: int) -> EvaluationProtocol:
        """Return an ordered evaluation view over the first paper protocol seeds.

        Args:
            episode_count: Number of evaluation episodes to include.

        Returns:
            An EvaluationProtocol instance preserving the original seed order.

        Raises:
            ValueError: If episode_count is outside the valid range [1, self.episode_count].
        """
        if not 1 <= episode_count <= self.episode_count:
            raise ValueError(f"Evaluation episode count must be between 1 and {self.episode_count}")
        return EvaluationProtocol(
            self.env_id,
            self.episode_seeds[:episode_count],
            self.training_episode_seeds,
            self.schema_version,
            self.name,
        )

    @classmethod
    def create(
        cls, env_id: str, environment_seed: int, training_episode_seeds: Sequence[int]
    ) -> EvaluationProtocol:
        """Create deterministic seeds disjoint from synthesis episodes."""
        seeds = generate_episode_seeds(
            environment_seed, PAPER_1P_EPISODE_COUNT, training_episode_seeds
        )
        return cls(env_id, tuple(seeds), tuple(training_episode_seeds))

    @classmethod
    def from_dict(cls, data: Mapping[str, object], expected_env_id: str) -> EvaluationProtocol:
        """Validate and deserialize a persisted protocol."""
        if data.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
            raise ValueError("Evaluation protocol schema must be 1")
        if data.get("name") != PROTOCOL_NAME:
            raise ValueError("Evaluation protocol name must be paper-1p")
        if data.get("env_id") != expected_env_id:
            raise ValueError("Evaluation protocol environment does not match run")
        if data.get("episode_count") != PAPER_1P_EPISODE_COUNT:
            raise ValueError("Evaluation protocol episode_count must be 20")
        seeds = _integer_list(data.get("episode_seeds"), "episode_seeds")
        training = _integer_list(data.get("training_episode_seeds"), "training_episode_seeds")
        if len(seeds) != PAPER_1P_EPISODE_COUNT:
            raise ValueError("Evaluation protocol requires 20 episode seeds")
        if len(set(seeds)) != PAPER_1P_EPISODE_COUNT:
            raise ValueError("Evaluation protocol episode seeds must be unique")
        if set(seeds) & set(training):
            raise ValueError("Evaluation and training seeds must be disjoint")
        if data.get("metrics") != {"reward": REWARD_METRIC, "legal_action_rate": LEGALITY_METRIC}:
            raise ValueError("Evaluation protocol metrics do not match paper-1p")
        return cls(expected_env_id, tuple(seeds), tuple(training))

    def to_dict(self) -> dict[str, object]:
        """Serialize this protocol as stable JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "env_id": self.env_id,
            "episode_count": self.episode_count,
            "episode_seeds": list(self.episode_seeds),
            "training_episode_seeds": list(self.training_episode_seeds),
            "metrics": {"reward": REWARD_METRIC, "legal_action_rate": LEGALITY_METRIC},
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of one seeded exact-environment episode."""

    seed: int
    env_id: str
    solved: bool
    reward: float
    legal_action_count: int
    action_attempt_count: int
    steps_used: int
    optimal_steps: int
    termination_reason: TerminationReason
    failure_summary: str | None
    latency: float
    execution_failure: bool
    policy_invocation_count: int = 0
    policy_seeds: tuple[int, ...] = ()

    @classmethod
    def from_execution_failure(
        cls,
        seed: int,
        env_id: str,
        optimal_steps: int,
        failure_summary: str,
        latency: float = 0.0,
    ) -> EvaluationResult:
        """Create an actionless execution-failure episode result."""
        return cls(
            seed=seed,
            env_id=env_id,
            solved=False,
            reward=0.0,
            legal_action_count=0,
            action_attempt_count=0,
            steps_used=0,
            optimal_steps=optimal_steps,
            termination_reason=TerminationReason.EXECUTION_FAILURE,
            failure_summary=failure_summary,
            latency=latency,
            execution_failure=True,
            policy_invocation_count=0,
            policy_seeds=(),
        )


@dataclass(frozen=True)
class EvaluationAggregate:
    """Canonical aggregates across all protocol episodes."""

    mean_reward: float
    legal_action_count: int
    action_attempt_count: int
    legal_action_rate: float | None
    termination_counts: dict[TerminationReason, int]
    execution_failure_count: int
    total_latency: float
    mean_latency: float


@dataclass(frozen=True)
class EvaluationUsage:
    """Optional model usage for a live-policy report."""

    model_call_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None


@dataclass(frozen=True)
class EvaluationReport:
    """Validated per-episode outcomes and canonical aggregates."""

    policy_kind: str
    protocol: EvaluationProtocol
    results: list[EvaluationResult]
    aggregate: EvaluationAggregate
    usage: EvaluationUsage | None = None
    model_id: str | None = None
    policy_randomness: dict[str, object] | None = None

    @classmethod
    def create(
        cls,
        policy_kind: str,
        protocol: EvaluationProtocol,
        results: list[EvaluationResult],
        usage: EvaluationUsage | None = None,
        model_id: str | None = None,
        policy_randomness: dict[str, object] | None = None,
    ) -> EvaluationReport:
        """Validate ordered results and calculate canonical metrics."""
        if len(results) != protocol.episode_count:
            raise ValueError(
                f"Evaluation results must contain exactly {protocol.episode_count} episodes"
            )
        if [result.seed for result in results] != list(protocol.episode_seeds):
            raise ValueError("Evaluation result seeds must match protocol order")
        if any(result.env_id != protocol.env_id for result in results):
            raise ValueError("Evaluation result environment must match protocol")
        if policy_kind == "generated-policy":
            for result in results:
                if len(result.policy_seeds) != result.policy_invocation_count:
                    raise ValueError(
                        "EvaluationResult policy_seeds length must match policy_invocation_count"
                    )
        else:
            for result in results:
                if result.policy_invocation_count != 0 or result.policy_seeds:
                    raise ValueError(
                        "Live evaluation results must contain zero policy invocations and seeds"
                    )
        legal = sum(result.legal_action_count for result in results)
        attempts = sum(result.action_attempt_count for result in results)
        latency = sum(result.latency for result in results)
        aggregate = EvaluationAggregate(
            fmean(result.reward for result in results),
            legal,
            attempts,
            legal / attempts if attempts else None,
            dict(Counter(result.termination_reason for result in results)),
            sum(result.execution_failure for result in results),
            latency,
            latency / len(results),
        )
        return cls(
            policy_kind,
            protocol,
            results,
            aggregate,
            usage,
            model_id,
            policy_randomness,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete report as JSON-compatible data."""
        aggregate = asdict(self.aggregate)
        aggregate["termination_counts"] = {
            reason.value: count for reason, count in self.aggregate.termination_counts.items()
        }
        results = [
            asdict(result)
            | {
                "termination_reason": result.termination_reason.value,
                "policy_seeds": list(result.policy_seeds),
            }
            for result in self.results
        ]
        data: dict[str, object] = {
            "schema_version": 2,
            "policy_kind": self.policy_kind,
            "policy_randomness": self.policy_randomness,
            "protocol": self.protocol.to_dict(),
            "aggregate": aggregate,
            "results": results,
            "usage": asdict(self.usage) if self.usage is not None else None,
        }
        if self.model_id is not None:
            data["model_id"] = self.model_id
        return data


def evaluate_action_provider_on_env(
    adapter: EnvironmentAdapter,
    provider: ActionProvider,
    seed: int,
    optimal_steps: int = 0,
    checks_legality: bool = False,
) -> EvaluationResult:
    """Evaluate one action provider on one seeded environment episode."""
    start = time.monotonic()
    rollout = RolloutEvaluator(adapter).evaluate_actions(provider, seed, checks_legality)
    return EvaluationResult(
        seed,
        adapter.env_id,
        rollout.terminal_reward >= 1.0,
        rollout.terminal_reward,
        rollout.legal_action_count,
        rollout.action_attempt_count,
        len(rollout.steps),
        optimal_steps or adapter.max_steps,
        rollout.termination_reason,
        rollout.failure_summary,
        time.monotonic() - start,
        rollout.termination_reason
        in (TerminationReason.EXECUTION_FAILURE, TerminationReason.CONTRACT_FAILURE),
        policy_invocation_count=0,
        policy_seeds=(),
    )


def evaluate_policy_on_env(
    adapter: EnvironmentAdapter,
    executor: ExecutorProtocol,
    source: str,
    seed: int,
    optimal_steps: int = 0,
) -> EvaluationResult:
    """Evaluate generated policy code for one seeded episode."""
    start = time.monotonic()
    rollout = RolloutEvaluator(adapter, executor).evaluate(source, seed)
    return EvaluationResult(
        seed=seed,
        env_id=adapter.env_id,
        solved=rollout.terminal_reward >= 1.0,
        reward=rollout.terminal_reward,
        legal_action_count=rollout.legal_action_count,
        action_attempt_count=rollout.action_attempt_count,
        steps_used=len(rollout.steps),
        optimal_steps=optimal_steps or adapter.max_steps,
        termination_reason=rollout.termination_reason,
        failure_summary=rollout.failure_summary,
        latency=time.monotonic() - start,
        execution_failure=rollout.termination_reason
        in (TerminationReason.EXECUTION_FAILURE, TerminationReason.CONTRACT_FAILURE),
        policy_invocation_count=sum(
            attempt.policy_seed is not None for attempt in rollout.attempts
        ),
        policy_seeds=tuple(
            attempt.policy_seed for attempt in rollout.attempts if attempt.policy_seed is not None
        ),
    )


def evaluate_policy(
    source: str,
    spec: EnvironmentSpec,
    protocol: EvaluationProtocol,
    executor: ExecutorProtocol | None = None,
) -> EvaluationReport:
    """Evaluate generated policy code on 20 fresh exact-environment adapters."""
    if spec.env_id != protocol.env_id:
        raise ValueError("Evaluation protocol environment does not match specification")
    policy_executor = executor or PolicyExecutor()
    results = []
    for seed in protocol.episode_seeds:
        start = time.monotonic()
        try:
            adapter = spec.create_adapter()
        except Exception as error:
            results.append(
                EvaluationResult.from_execution_failure(
                    seed,
                    spec.env_id,
                    spec.optimal_steps,
                    f"Adapter construction failed: {error}",
                    time.monotonic() - start,
                )
            )
            continue
        results.append(
            evaluate_policy_on_env(adapter, policy_executor, source, seed, spec.optimal_steps)
        )
    return EvaluationReport.create(
        "generated-policy", protocol, results, policy_randomness=policy_randomness_metadata()
    )


def format_evaluation_summary(report: EvaluationReport) -> str:
    """Format canonical evaluation aggregates for terminal output."""
    aggregate = report.aggregate
    legality = (
        "n/a" if aggregate.legal_action_rate is None else f"{aggregate.legal_action_rate:.3f}"
    )
    reasons = ", ".join(
        f"{reason.value}={count}"
        for reason, count in sorted(
            aggregate.termination_counts.items(), key=lambda item: item[0].value
        )
    )
    return (
        "Policy Evaluation Summary\n"
        f"  Environment: {report.protocol.env_id}\n"
        f"  Episodes: {report.protocol.episode_count}\n"
        f"  Mean reward: {aggregate.mean_reward:.3f}\n"
        f"  Legal action rate: {legality} "
        f"({aggregate.legal_action_count}/{aggregate.action_attempt_count})\n"
        f"  Terminations: {reasons}\n  Execution failures: {aggregate.execution_failure_count}\n"
        f"  Latency: total={aggregate.total_latency:.3f}s mean={aggregate.mean_latency:.3f}s"
    )
