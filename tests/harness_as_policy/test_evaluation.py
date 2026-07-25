"""Tests for exact-environment evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from autoharness.harness_as_policy.assessment import generate_episode_seeds
from autoharness.harness_as_policy.environments.registry import EnvironmentSpec
from autoharness.harness_as_policy.evaluation import (
    EvaluationProtocol,
    EvaluationReport,
    EvaluationResult,
    evaluate_policy,
    evaluate_policy_on_env,
    format_evaluation_summary,
)
from autoharness.harness_as_policy.executor import ExecutionResult
from autoharness.harness_as_policy.models import StepResult, TerminationReason


@dataclass
class FakeAdapter:
    env_id: str = "Exact-v0"
    rules: str = "rules"
    action_format: str = "action"
    max_steps: int = 1
    seed: int | None = None
    step_result: StepResult | None = None

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        self.seed = seed
        return "obs"

    def step(self, action: str) -> StepResult:
        return self.step_result or StepResult("done", action, True, 1.0, True, "")


class FakeExecutor:
    def execute(self, source: str, observation: str) -> ExecutionResult:
        return ExecutionResult(True, "action", 0.0, is_legal_action=True)


def test_protocol_is_reproducible_disjoint_and_round_trips() -> None:
    training = generate_episode_seeds(17, 5)
    protocol = EvaluationProtocol.create("Exact-v0", 17, training)
    assert protocol.episode_count == 20
    assert set(protocol.episode_seeds).isdisjoint(training)
    assert EvaluationProtocol.from_dict(protocol.to_dict(), "Exact-v0") == protocol


def test_protocol_rejects_wrong_environment() -> None:
    protocol = EvaluationProtocol.create("Exact-v0", 17, [])
    with pytest.raises(ValueError, match="environment"):
        EvaluationProtocol.from_dict(protocol.to_dict(), "Other-v0")


def test_protocol_rejects_overlap() -> None:
    protocol = EvaluationProtocol.create("Exact-v0", 17, [11, 22])
    data = protocol.to_dict()
    data["training_episode_seeds"] = [protocol.episode_seeds[0]]

    with pytest.raises(ValueError, match="disjoint"):
        EvaluationProtocol.from_dict(data, "Exact-v0")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("episode_seeds", list(range(19)), "20"),
        ("episode_seeds", [1] * 20, "unique"),
        ("episode_count", 19, "20"),
        ("schema_version", 2, "schema"),
        ("name", "other", "name"),
        ("episode_seeds", [*range(19), "bad"], "integers"),
        ("metrics", {}, "metrics"),
    ],
)
def test_protocol_rejects_malformed_fields(field: str, value: object, match: str) -> None:
    data = EvaluationProtocol("Exact-v0", tuple(range(20)), ()).to_dict()
    data[field] = value

    with pytest.raises(ValueError, match=match):
        EvaluationProtocol.from_dict(data, "Exact-v0")


def test_evaluate_policy_uses_20_fresh_seeded_adapters() -> None:
    adapters: list[FakeAdapter] = []

    def factory() -> FakeAdapter:
        adapter = FakeAdapter()
        adapters.append(adapter)
        return adapter

    protocol = EvaluationProtocol.create("Exact-v0", 9, [1, 2])
    report = evaluate_policy(
        "source",
        EnvironmentSpec("Exact-v0", "fake", factory, 1, 1),
        protocol,
        FakeExecutor(),
    )
    assert len(adapters) == 20
    assert [adapter.seed for adapter in adapters] == list(protocol.episode_seeds)
    assert report.aggregate.mean_reward == 1.0


def test_report_uses_action_weighted_legality_and_excludes_actionless_failures() -> None:
    protocol = EvaluationProtocol("Exact-v0", tuple(range(20)), ())
    results = [
        EvaluationResult(
            seed, "Exact-v0", False, 0.0, 1, 2, 1, 1, TerminationReason.STEP_LIMIT, None, 0.1, False
        )
        for seed in range(19)
    ]
    results.append(
        EvaluationResult(
            19,
            "Exact-v0",
            False,
            0.0,
            0,
            0,
            0,
            1,
            TerminationReason.EXECUTION_FAILURE,
            "failed",
            0.1,
            True,
        )
    )
    report = EvaluationReport.create("test", protocol, results)
    assert report.aggregate.legal_action_rate == 0.5
    assert report.aggregate.action_attempt_count == 38
    assert report.aggregate.execution_failure_count == 1


def _result(
    seed: int,
    *,
    env_id: str = "Exact-v0",
    reward: float = 0.0,
    legal: int = 0,
    attempts: int = 0,
    reason: TerminationReason = TerminationReason.STEP_LIMIT,
    execution_failure: bool = False,
) -> EvaluationResult:
    return EvaluationResult(
        seed=seed,
        env_id=env_id,
        solved=reward >= 1.0,
        reward=reward,
        legal_action_count=legal,
        action_attempt_count=attempts,
        steps_used=legal,
        optimal_steps=1,
        termination_reason=reason,
        failure_summary="failed" if execution_failure else None,
        latency=0.1,
        execution_failure=execution_failure,
    )


def test_report_aggregates_mean_reward_and_action_weighted_legality() -> None:
    protocol = EvaluationProtocol("Exact-v0", tuple(range(20)), ())
    results = [
        *[_result(seed, reward=1.0, legal=3, attempts=4) for seed in range(10)],
        *[_result(seed, legal=1, attempts=2) for seed in range(10, 20)],
    ]

    report = EvaluationReport.create("generated-policy", protocol, results)

    assert report.aggregate.mean_reward == 0.5
    assert report.aggregate.legal_action_count == 40
    assert report.aggregate.action_attempt_count == 60
    assert report.aggregate.legal_action_rate == pytest.approx(2 / 3)


def test_report_uses_no_legality_denominator_for_actionless_failures() -> None:
    protocol = EvaluationProtocol("Exact-v0", tuple(range(20)), ())
    results = [
        _result(
            seed,
            reason=TerminationReason.EXECUTION_FAILURE,
            execution_failure=True,
        )
        for seed in range(20)
    ]

    report = EvaluationReport.create("generated-policy", protocol, results)

    assert report.aggregate.legal_action_rate is None
    assert report.aggregate.execution_failure_count == 20


@pytest.mark.parametrize(
    ("results", "match"),
    [
        ([_result(seed) for seed in range(19)], "exactly 20"),
        ([_result(seed) for seed in [1, 0, *range(2, 20)]], "seeds"),
        ([_result(seed, env_id="Other-v0") for seed in range(20)], "environment"),
    ],
)
def test_report_rejects_results_that_do_not_match_protocol(
    results: list[EvaluationResult], match: str
) -> None:
    protocol = EvaluationProtocol("Exact-v0", tuple(range(20)), ())

    with pytest.raises(ValueError, match=match):
        EvaluationReport.create("generated-policy", protocol, results)


def test_evaluate_policy_on_env_preserves_step_limit_reward() -> None:
    adapter = FakeAdapter(
        max_steps=1,
        step_result=StepResult("next", "action", True, 0.6, False, ""),
    )
    result = evaluate_policy_on_env(adapter, FakeExecutor(), "source", seed=123)

    assert result.reward == 0.6
    assert result.termination_reason == TerminationReason.STEP_LIMIT
    assert result.action_attempt_count == 1


def test_evaluation_summary_reports_canonical_aggregates() -> None:
    protocol = EvaluationProtocol("Exact-v0", tuple(range(20)), ())
    report = EvaluationReport.create(
        "generated-policy",
        protocol,
        [_result(seed, legal=1, attempts=1) for seed in range(20)],
    )

    summary = format_evaluation_summary(report)

    assert "Environment: Exact-v0" in summary
    assert "Episodes: 20" in summary
    assert "Mean reward:" in summary
    assert "Legal action rate:" in summary
    assert "Largest disk count" not in summary
