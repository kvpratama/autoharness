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
    """Rollout evaluator that requires a deterministic environment seed."""

    def evaluate(self, source: str, seed: int) -> RolloutResult:
        """Evaluate a policy source against the environment using a fixed seed.

        Args:
            source: Python source code of the candidate policy module.
            seed: Required integer seed used to initialise the environment RNG
                for a deterministic, reproducible rollout.

        Returns:
            A RolloutResult summarising the episode outcome.
        """
        ...


_ACTIONABILITY = {
    TerminationReason.CONTRACT_FAILURE: 0,
    TerminationReason.EXECUTION_FAILURE: 0,
    TerminationReason.LEGALITY_DISAGREEMENT: 1,
    TerminationReason.ILLEGAL_ACTION: 1,
    TerminationReason.POLICY_REJECTED_ACTION: 2,
    TerminationReason.STEP_LIMIT: 3,
    TerminationReason.ENVIRONMENT_TERMINATION: 4,
}

_ROLLOUT_FAILURE_REASONS = {
    TerminationReason.CONTRACT_FAILURE,
    TerminationReason.EXECUTION_FAILURE,
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
                episode.rollout.termination_reason in _ROLLOUT_FAILURE_REASONS
                for episode in episodes
            ),
            termination_counts=dict(counts),
            representative_episode_index=representative_index,
            termination_reason=representative.termination_reason,
            failure_summary=representative.failure_summary,
            last_observation=representative.last_observation,
        )


def assessment_is_rollout_eligible(assessment: CandidateAssessment) -> bool:
    """Return whether at least one assessed episode completed policy execution."""
    return any(
        episode.rollout.termination_reason not in _ROLLOUT_FAILURE_REASONS
        for episode in assessment.episodes
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


def _render_observation_change(previous: str, current: str) -> str:
    """Render a diff between two consecutive observations for the refiner prompt.

    If ``current`` is a strict prefix extension of ``previous`` (i.e. ``current``
    starts with ``previous``), only the newly appended text is emitted as an
    incremental *observation update*.  Otherwise the full ``current`` text is emitted
    as an *observation snapshot*.  Returns the string ``"Observation unchanged"`` when
    both values are identical.
    """
    if current == previous:
        return "Observation unchanged"
    if current.startswith(previous):
        return _field("Observation update", current[len(previous) :])
    return _field("Observation snapshot", current)


def _render_attempt(
    index: int, attempt: ActionAttempt, current_observation: str
) -> tuple[str, str]:
    lines = [f"Attempt {index}"]
    if attempt.observation != current_observation:
        lines.append(_render_observation_change(current_observation, attempt.observation))
        current_observation = attempt.observation
    if attempt.action is not None:
        lines.append(_field("Proposed action", attempt.action))
    if attempt.policy_legal is not None:
        lines.append(f"Policy legality check: {_value(attempt.policy_legal)}")
    if attempt.environment_legal is not None:
        lines.append(f"Environment legality check: {_value(attempt.environment_legal)}")
    if attempt.feedback:
        lines.append(_field("Feedback", attempt.feedback))
    if attempt.resulting_observation is not None:
        lines.append(_render_observation_change(current_observation, attempt.resulting_observation))
        current_observation = attempt.resulting_observation
    if attempt.reward is not None:
        lines.append(f"Reward: {_value(attempt.reward)}")
    if attempt.terminated is not None:
        lines.append(f"Terminated: {_value(attempt.terminated)}")
    if attempt.error_phase is not None:
        lines.append(f"Error phase: {_value(attempt.error_phase)}")
    return "\n".join(lines), current_observation


def build_assessment_trajectory(assessment: CandidateAssessment) -> str:
    """Render every episode and action attempt in assessment order."""
    episodes: list[str] = []
    for episode_index, episode in enumerate(assessment.episodes, start=1):
        rollout = episode.rollout
        initial = rollout.attempts[0].observation if rollout.attempts else rollout.last_observation
        current_observation = initial or ""
        lines = [
            f"Episode {episode_index}",
            f"Seed: {episode.seed}",
            _field("Initial board", initial),
        ]
        for index, attempt in enumerate(rollout.attempts, start=1):
            rendered_attempt, current_observation = _render_attempt(
                index, attempt, current_observation
            )
            lines.append(rendered_attempt)
        lines.append(f"Rollout termination: {rollout.termination_reason.value}")
        if rollout.failure_summary:
            lines.append(_field("Rollout failure summary", rollout.failure_summary))
        episodes.append("\n\n".join(lines))
    return "\n\n".join(episodes)


def should_refine_legal_action(assessment: CandidateAssessment) -> bool:
    """Return whether both policy functions need refinement."""
    return assessment.termination_counts.get(TerminationReason.LEGALITY_DISAGREEMENT, 0) > 0
