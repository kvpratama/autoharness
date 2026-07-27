"""Repeated seeded rollout assessment for policy candidates."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Collection
from statistics import fmean
from typing import Protocol

from autoharness.harness_as_policy.models import (
    ActionAttempt,
    CandidateAssessment,
    EpisodeResult,
    RolloutResult,
    TerminationReason,
)


class SeededRolloutEvaluator(Protocol):
    """Rollout evaluator that accepts an optional environment seed."""

    def evaluate(self, source: str, seed: int | None = None) -> RolloutResult: ...


_ACTIONABILITY = {
    TerminationReason.CONTRACT_FAILURE: 0,
    TerminationReason.EXECUTION_FAILURE: 0,
    TerminationReason.LEGALITY_DISAGREEMENT: 1,
    TerminationReason.ILLEGAL_ACTION: 1,
    TerminationReason.POLICY_REJECTED_ACTION: 2,
    TerminationReason.STEP_LIMIT: 3,
    TerminationReason.ENVIRONMENT_TERMINATION: 4,
}


def generate_episode_seeds(base_seed: int, count: int, excluded: Collection[int] = ()) -> list[int]:
    """Generate a reproducible ordered list of unique 32-bit episode seeds."""
    if count <= 0:
        raise ValueError("Episode seed count must be positive")
    rng = random.Random(base_seed)
    seeds: list[int] = []
    seen = set(excluded)
    while len(seeds) < count:
        seed = rng.getrandbits(32)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


class CandidateAssessor:
    """Assess one source over a shared ordered set of environment seeds."""

    def __init__(self, evaluator: SeededRolloutEvaluator) -> None:
        self._evaluator = evaluator

    def assess(self, source: str, seeds: list[int]) -> CandidateAssessment:
        """Run and aggregate one episode for each supplied seed."""
        if not seeds:
            raise ValueError("Candidate assessment requires at least one episode seed")
        episodes = [
            EpisodeResult(seed=seed, rollout=self._evaluator.evaluate(source, seed=seed))
            for seed in seeds
        ]
        representative_index = min(
            range(len(episodes)),
            key=lambda index: (
                episodes[index].rollout.heuristic,
                episodes[index].rollout.terminal_reward,
                _ACTIONABILITY[episodes[index].rollout.termination_reason],
                index,
            ),
        )
        representative = episodes[representative_index].rollout
        counts = Counter(episode.rollout.termination_reason for episode in episodes)
        return CandidateAssessment(
            episodes=episodes,
            heuristic=fmean(episode.rollout.heuristic for episode in episodes),
            terminal_reward=fmean(episode.rollout.terminal_reward for episode in episodes),
            legal_action_count=sum(episode.rollout.legal_action_count for episode in episodes),
            failure_count=sum(
                episode.rollout.termination_reason
                in (TerminationReason.EXECUTION_FAILURE, TerminationReason.CONTRACT_FAILURE)
                for episode in episodes
            ),
            termination_counts=dict(counts),
            representative_episode_index=representative_index,
            termination_reason=representative.termination_reason,
            failure_summary=representative.failure_summary,
            last_observation=representative.last_observation,
        )


def failed_assessment(error: str) -> CandidateAssessment:
    """Build an assessment for a refinement that failed before rollout."""
    return CandidateAssessment(
        episodes=[],
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        failure_count=1,
        termination_counts={TerminationReason.CONTRACT_FAILURE: 1},
        representative_episode_index=None,
        termination_reason=TerminationReason.CONTRACT_FAILURE,
        failure_summary=error,
        last_observation=None,
    )


def _value(value: object | None) -> str:
    if value is None:
        return "not reached"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _field(label: str, value: object | None) -> str:
    return f"{label}:\n{_value(value)}"


def build_initial_trajectory(observations: list[tuple[int, str]]) -> str:
    """Render every seeded initial board without executing a policy."""
    return "\n\n".join(
        "\n".join(
            (
                f"Episode {index}",
                f"Seed: {seed}",
                _field("Initial board", observation),
                "Action attempts: none",
                "Feedback:\nNo action attempted; implement the initial policy.",
            )
        )
        for index, (seed, observation) in enumerate(observations, start=1)
    )


def _render_attempt(index: int, attempt: ActionAttempt) -> str:
    return "\n".join(
        (
            f"Attempt {index}",
            _field("Board before action", attempt.observation),
            _field("Proposed action", attempt.action),
            f"Policy legality check: {_value(attempt.policy_legal)}",
            f"Environment legality check: {_value(attempt.environment_legal)}",
            _field("Feedback", attempt.feedback),
            _field("Board after action", attempt.resulting_observation),
            f"Reward: {_value(attempt.reward)}",
            f"Terminated: {_value(attempt.terminated)}",
            f"Error phase: {_value(attempt.error_phase)}",
        )
    )


def build_assessment_trajectory(assessment: CandidateAssessment) -> str:
    """Render every episode and action attempt in assessment order."""
    episodes: list[str] = []
    for episode_index, episode in enumerate(assessment.episodes, start=1):
        rollout = episode.rollout
        initial = rollout.attempts[0].observation if rollout.attempts else rollout.last_observation
        lines = [
            f"Episode {episode_index}",
            f"Seed: {episode.seed}",
            _field("Initial board", initial),
        ]
        lines.extend(
            _render_attempt(index, attempt)
            for index, attempt in enumerate(rollout.attempts, start=1)
        )
        lines.extend(
            (
                f"Rollout termination: {rollout.termination_reason.value}",
                _field("Rollout failure summary", rollout.failure_summary),
            )
        )
        episodes.append("\n\n".join(lines))
    return "\n\n".join(episodes)


def should_refine_legal_action(assessment: CandidateAssessment) -> bool:
    """Return whether both policy functions need refinement."""
    return assessment.termination_counts.get(TerminationReason.LEGALITY_DISAGREEMENT, 0) > 0
