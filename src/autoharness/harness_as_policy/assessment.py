"""Repeated seeded rollout assessment for policy candidates."""

from __future__ import annotations

import random
from collections import Counter
from statistics import fmean
from typing import Protocol

from autoharness.harness_as_policy.models import (
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


def generate_episode_seeds(base_seed: int, count: int) -> list[int]:
    """Generate a reproducible ordered list of unique 32-bit episode seeds."""
    if count <= 0:
        raise ValueError("Training rollout count must be positive")
    rng = random.Random(base_seed)
    seeds: list[int] = []
    seen: set[int] = set()
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


def build_assessment_feedback(assessment: CandidateAssessment) -> list[str]:
    """Build bounded deterministic feedback using the representative episode only."""
    lines = [
        f"{len(assessment.episodes)} episodes: mean H={assessment.heuristic:.3f}, "
        f"mean reward={assessment.terminal_reward:.3f}",
        "Termination counts: "
        + ", ".join(
            f"{reason.value}={count}"
            for reason, count in sorted(
                assessment.termination_counts.items(), key=lambda item: item[0].value
            )
        ),
    ]
    if assessment.representative_episode_index is not None:
        episode = assessment.episodes[assessment.representative_episode_index]
        lines.append(
            f"Representative episode: seed={episode.seed}, "
            f"termination={episode.rollout.termination_reason.value}"
        )
    if assessment.failure_summary:
        lines.append(f"Representative failure: {assessment.failure_summary}")
    if assessment.last_observation:
        lines.append(f"Representative observation: {assessment.last_observation}")
    return lines[:5]


def should_refine_legal_action(assessment: CandidateAssessment) -> bool:
    """Return whether both policy functions need refinement."""
    return assessment.termination_counts.get(TerminationReason.LEGALITY_DISAGREEMENT, 0) > 0
