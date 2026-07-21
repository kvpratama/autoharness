"""Tests for repeated seeded candidate assessment."""

from __future__ import annotations

import pytest

from autoharness.harness_as_policy.assessment import (
    CandidateAssessor,
    build_assessment_feedback,
    generate_episode_seeds,
    should_refine_legal_action,
)
from autoharness.harness_as_policy.models import RolloutResult, TerminationReason


class ScriptedEvaluator:
    def __init__(self, results: list[RolloutResult]) -> None:
        self._results = iter(results)
        self.seeds: list[int | None] = []

    def evaluate(self, source: str, seed: int | None = None) -> RolloutResult:
        self.seeds.append(seed)
        return next(self._results)


def rollout(
    heuristic: float,
    reward: float,
    reason: TerminationReason,
    *,
    legal_actions: int = 1,
    failure: str | None = None,
) -> RolloutResult:
    return RolloutResult(
        [], heuristic, reward, legal_actions, reason, failure, f"last-{reason.value}"
    )


def test_assessor_runs_all_shared_seeds_and_aggregates() -> None:
    evaluator = ScriptedEvaluator(
        [
            rollout(1.0, 1.0, TerminationReason.ENVIRONMENT_TERMINATION, legal_actions=3),
            rollout(0.5, 0.0, TerminationReason.STEP_LIMIT, legal_actions=4),
            rollout(0.0, 0.0, TerminationReason.EXECUTION_FAILURE, legal_actions=2, failure="boom"),
        ]
    )
    assessment = CandidateAssessor(evaluator).assess("source", [11, 22, 33])
    assert evaluator.seeds == [11, 22, 33]
    assert assessment.heuristic == pytest.approx(0.5)
    assert assessment.terminal_reward == pytest.approx(1 / 3)
    assert assessment.legal_action_count == 9
    assert assessment.failure_count == 1
    assert assessment.termination_counts == {
        TerminationReason.ENVIRONMENT_TERMINATION: 1,
        TerminationReason.STEP_LIMIT: 1,
        TerminationReason.EXECUTION_FAILURE: 1,
    }
    assert assessment.representative_episode_index == 2
    assert assessment.termination_reason == TerminationReason.EXECUTION_FAILURE


def test_representative_ties_use_actionability_then_seed_order() -> None:
    evaluator = ScriptedEvaluator(
        [
            rollout(0.0, 0.0, TerminationReason.POLICY_REJECTED_ACTION),
            rollout(0.0, 0.0, TerminationReason.LEGALITY_DISAGREEMENT),
            rollout(0.0, 0.0, TerminationReason.LEGALITY_DISAGREEMENT),
        ]
    )
    assessment = CandidateAssessor(evaluator).assess("source", [10, 20, 30])
    assert assessment.representative_episode_index == 1


def test_seed_generation_is_reproducible_and_counted() -> None:
    assert generate_episode_seeds(7, 5) == generate_episode_seeds(7, 5)
    assert len(generate_episode_seeds(7, 5)) == 5
    assert generate_episode_seeds(7, 5) != generate_episode_seeds(8, 5)


def test_feedback_uses_only_representative_episode() -> None:
    evaluator = ScriptedEvaluator(
        [
            rollout(1.0, 1.0, TerminationReason.ENVIRONMENT_TERMINATION),
            rollout(0.0, 0.0, TerminationReason.LEGALITY_DISAGREEMENT, failure="bad move"),
        ]
    )
    feedback = build_assessment_feedback(CandidateAssessor(evaluator).assess("source", [101, 202]))
    assert feedback[0] == "2 episodes: mean H=0.500, mean reward=0.500"
    assert any("seed=202" in line for line in feedback)
    assert any("bad move" in line for line in feedback)
    assert all("seed=101" not in line for line in feedback)
    assert len(feedback) <= 5


def test_any_legality_disagreement_refines_both_functions() -> None:
    assessment = CandidateAssessor(
        ScriptedEvaluator(
            [
                rollout(0.0, 0.0, TerminationReason.POLICY_REJECTED_ACTION),
                rollout(0.0, 0.0, TerminationReason.LEGALITY_DISAGREEMENT),
            ]
        )
    ).assess("source", [1, 2])
    assert should_refine_legal_action(assessment)


def test_only_representative_checker_rejection_refines_action_only() -> None:
    assessment = CandidateAssessor(
        ScriptedEvaluator([rollout(0.0, 0.0, TerminationReason.POLICY_REJECTED_ACTION)])
    ).assess("source", [1])
    assert not should_refine_legal_action(assessment)
