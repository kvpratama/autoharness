"""Tests for repeated seeded candidate assessment."""

from __future__ import annotations

import pytest

from autoharness.harness_as_policy.assessment import (
    CandidateAssessor,
    build_assessment_trajectory,
    build_initial_trajectory,
    generate_episode_seeds,
    should_refine_legal_action,
)
from autoharness.harness_as_policy.models import (
    ActionAttempt,
    CandidateAssessment,
    EpisodeResult,
    RolloutResult,
    TerminationReason,
)


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


def test_seed_generation_excludes_training_seeds_without_changing_order() -> None:
    training = generate_episode_seeds(7, 3)
    held_out = generate_episode_seeds(7, 20, excluded=training)

    assert len(held_out) == 20
    assert len(set(held_out)) == 20
    assert set(training).isdisjoint(held_out)
    assert held_out == generate_episode_seeds(7, 23)[3:]


def test_initial_trajectory_contains_every_seeded_board_in_order() -> None:
    context = build_initial_trajectory([(20, "board twenty"), (10, "board ten")])

    assert context.index("Seed: 20") < context.index("Seed: 10")
    assert context.index("board twenty") < context.index("board ten")
    assert context.count("No action attempted; implement the initial policy.") == 2


def test_assessment_trajectory_contains_all_episodes_and_attempts_without_truncation() -> None:
    episodes = []
    for seed in range(7):
        attempts = [
            ActionAttempt(
                observation=f"before-{seed}-{step}",
                action=f"action-{seed}-{step}",
                policy_legal=True,
                environment_legal=True,
                resulting_observation=f"after-{seed}-{step}",
                reward=float(step),
                terminated=False,
                feedback=f"feedback-{seed}-{step}",
                error_phase=None,
            )
            for step in range(7)
        ]
        episodes.append(
            EpisodeResult(
                seed,
                RolloutResult(
                    [],
                    0.5,
                    0.0,
                    7,
                    TerminationReason.STEP_LIMIT,
                    None,
                    attempts=attempts,
                ),
            )
        )
    assessment = CandidateAssessment(
        episodes=episodes,
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=49,
        failure_count=0,
        termination_counts={TerminationReason.STEP_LIMIT: 7},
        representative_episode_index=0,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        last_observation="after-0-6",
    )

    context = build_assessment_trajectory(assessment)

    assert context.index("Seed: 0") < context.index("Seed: 6")
    assert "before-6-6" in context
    assert "action-6-6" in context
    assert "feedback-6-6" in context
    assert context.count("Attempt 7") == 7


def test_assessment_trajectory_renders_cumulative_observations_as_updates() -> None:
    attempts = [
        ActionAttempt(
            observation="initial board",
            action="first action",
            policy_legal=True,
            environment_legal=True,
            resulting_observation="initial board\nfirst result",
            reward=0.0,
            terminated=False,
            feedback="",
            error_phase=None,
        ),
        ActionAttempt(
            observation="initial board\nfirst result",
            action="second action",
            policy_legal=True,
            environment_legal=True,
            resulting_observation="initial board\nfirst result\nsecond result",
            reward=1.0,
            terminated=True,
            feedback="finished",
            error_phase=None,
        ),
    ]
    assessment = CandidateAssessment(
        episodes=[
            EpisodeResult(
                7,
                RolloutResult(
                    [],
                    1.0,
                    1.0,
                    2,
                    TerminationReason.ENVIRONMENT_TERMINATION,
                    None,
                    attempts=attempts,
                ),
            )
        ],
        heuristic=1.0,
        terminal_reward=1.0,
        legal_action_count=2,
        failure_count=0,
        termination_counts={TerminationReason.ENVIRONMENT_TERMINATION: 1},
        representative_episode_index=0,
        termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
        failure_summary=None,
        last_observation=attempts[-1].resulting_observation,
    )

    context = build_assessment_trajectory(assessment)

    assert context.count("initial board") == 1
    assert context.count("first result") == 1
    assert context.count("second result") == 1
    assert "Board before action" not in context
    assert context.count("Observation update:") == 2
    assert "Feedback:\nfinished" in context
    assert "Feedback:\n\n" not in context
    assert "Error phase: not reached" not in context


def test_assessment_trajectory_uses_snapshot_for_non_prefix_observation() -> None:
    attempt = ActionAttempt(
        observation="before",
        action="action",
        policy_legal=None,
        environment_legal=None,
        resulting_observation="replacement state",
        reward=None,
        terminated=None,
        feedback="execution failed",
        error_phase=None,
    )
    assessment = CandidateAssessment(
        episodes=[
            EpisodeResult(
                9,
                RolloutResult(
                    [],
                    0.0,
                    0.0,
                    0,
                    TerminationReason.EXECUTION_FAILURE,
                    "execution failed",
                    attempts=[attempt],
                ),
            )
        ],
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        failure_count=1,
        termination_counts={TerminationReason.EXECUTION_FAILURE: 1},
        representative_episode_index=0,
        termination_reason=TerminationReason.EXECUTION_FAILURE,
        failure_summary="execution failed",
        last_observation="replacement state",
    )

    context = build_assessment_trajectory(assessment)

    assert "Observation snapshot:\nreplacement state" in context
    assert "Policy legality check" not in context
    assert "Environment legality check" not in context
    assert "Reward:" not in context
    assert "Terminated:" not in context


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
