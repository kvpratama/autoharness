"""Domain models for the harness-as-policy synthesis system."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import StrEnum


@dataclass
class ProviderInvocation:
    """Auditable outcome of one provider invocation."""

    content: object | None = None
    normalized_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class RefinementOutcome(StrEnum):
    """Allowed terminal (and initial) states for a RefinementTrace."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    TRANSPORT_FAILURE = "transport_failure"


@dataclass
class RefinementTrace:
    """Exact prompt and provider outcomes for one logical refinement."""

    prompt: str
    invocations: list[ProviderInvocation] = field(default_factory=list)
    extracted_source: str | None = None
    outcome: RefinementOutcome = RefinementOutcome.IN_PROGRESS
    error_details: str | None = None
    generation_succeeded: bool = False
    contract_valid: bool = False


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
    FULL_SEARCH = "full-search"

    @property
    def refinements(self) -> int:
        return {
            "smoke": 8,
            "low-cost": 32,
            "full-search": 256,
        }[self.value]


@dataclass
class StepResult:
    """Result of a single step in a rollout."""

    observation: str
    action: str | None
    is_legal: bool
    reward: float
    terminated: bool
    feedback: str


class AttemptErrorPhase(StrEnum):
    """Phase in which an action attempt failed."""

    POLICY_EXECUTION = "policy_execution"
    POLICY_LEGALITY = "policy_legality"
    ENVIRONMENT_STEP = "environment_step"


@dataclass(frozen=True, kw_only=True)
class ActionAttempt:
    """One policy decision with its pre-action board and resulting outcome."""

    observation: str
    action: str | None
    policy_legal: bool | None
    environment_legal: bool | None
    resulting_observation: str | None
    reward: float | None
    terminated: bool | None
    feedback: str
    error_phase: AttemptErrorPhase | None
    policy_seed: int | None = None


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
    action_attempt_count: int = 0
    attempts: list[ActionAttempt] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """One seeded episode in a candidate assessment."""

    seed: int
    rollout: RolloutResult


@dataclass
class CandidateAssessment:
    """Aggregate and individual outcomes for one policy candidate."""

    episodes: list[EpisodeResult]
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    failure_count: int
    termination_counts: dict[TerminationReason, int]
    representative_episode_index: int | None
    termination_reason: TerminationReason | None
    failure_summary: str | None
    last_observation: str | None


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
    failure_count: int = 0
    episode_count: int = 0
    assessment: CandidateAssessment | None = None
    rollout_eligible: bool = False


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
        failures = c.failure_count
        if failures == 0 and c.termination_reason in (
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        ):
            failures = 1
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
