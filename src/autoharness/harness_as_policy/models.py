"""Domain models for the harness-as-policy synthesis system."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import StrEnum


class TerminationReason(StrEnum):
    ILLEGAL_ACTION = "illegal_action"
    POLICY_REJECTED_ACTION = "policy_rejected_action"
    LEGALITY_DISAGREEMENT = "legality_disagreement"
    ENVIRONMENT_TERMINATION = "environment_termination"
    STEP_LIMIT = "step_limit"
    EXECUTION_FAILURE = "execution_failure"
    CONTRACT_FAILURE = "contract_failure"


class Profile(StrEnum):
    SMOKE = "smoke"
    LOW_COST = "low-cost"

    @property
    def refinements(self) -> int:
        return {"smoke": 8, "low-cost": 32}[self.value]

    @property
    def max_steps(self) -> int:
        return 14


@dataclass
class StepResult:
    """Result of a single step in a rollout."""

    observation: str
    action: str | None
    is_legal: bool
    reward: float
    terminated: bool
    feedback: str


@dataclass
class RolloutResult:
    """Result of one complete rollout."""

    steps: list[StepResult]
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    termination_reason: TerminationReason
    failure_summary: str | None
    last_observation: str | None = None


@dataclass
class Candidate:
    """A node in the program refinement tree."""

    id: str
    parent_id: str | None
    source: str
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    termination_reason: TerminationReason | None
    failure_summary: str | None
    iteration: int
    expansion_count: int = 0
    last_observation: str | None = None


@dataclass
class Event:
    """A recorded event during synthesis."""

    iteration: int
    event_type: str
    candidate_id: str | None
    parent_id: str | None
    metadata: dict = field(default_factory=dict)


@functools.total_ordering
@dataclass
class CandidateRankKey:
    """Lexicographic sort key for candidate ranking.

    Higher heuristic > higher reward > more legal actions >
    fewer failures > earlier iteration.
    """

    heuristic: float
    reward: float
    legal_actions: int
    failures: int
    iteration: int

    @classmethod
    def from_candidate(cls, c: Candidate) -> CandidateRankKey:
        failures = (
            1
            if c.termination_reason
            in (
                TerminationReason.EXECUTION_FAILURE,
                TerminationReason.CONTRACT_FAILURE,
            )
            else 0
        )
        return cls(
            heuristic=c.heuristic,
            reward=c.terminal_reward,
            legal_actions=c.legal_action_count,
            failures=failures,
            iteration=c.iteration,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateRankKey):
            return NotImplemented
        return (
            self.heuristic == other.heuristic
            and self.reward == other.reward
            and self.legal_actions == other.legal_actions
            and self.failures == other.failures
            and self.iteration == other.iteration
        )

    def __lt__(self, other: CandidateRankKey) -> bool:
        self_tuple = (
            self.heuristic,
            self.reward,
            self.legal_actions,
            -self.failures,
            -self.iteration,
        )
        other_tuple = (
            other.heuristic,
            other.reward,
            other.legal_actions,
            -other.failures,
            -other.iteration,
        )
        return self_tuple < other_tuple


def heuristic(*, is_legal: bool, reward: float) -> float:
    """Section 4.3 heuristic: 0 if illegal, else 0.5 + 0.5*r."""
    if not is_legal:
        return 0.0
    return 0.5 + 0.5 * reward
