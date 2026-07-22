"""Tests for domain models."""

from __future__ import annotations

from autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)


def test_heuristic_illegal_action() -> None:
    """Illegal action yields heuristic 0 regardless of reward."""
    assert heuristic(is_legal=False, reward=1.0) == 0.0
    assert heuristic(is_legal=False, reward=0.0) == 0.0


def test_heuristic_legal_solved() -> None:
    """Legal solved rollout yields heuristic 1.0."""
    assert heuristic(is_legal=True, reward=1.0) == 1.0


def test_heuristic_legal_no_reward() -> None:
    """Legal unsolved rollout yields heuristic 0.5."""
    assert heuristic(is_legal=True, reward=0.0) == 0.5


def test_heuristic_legal_partial() -> None:
    """Legal partial reward yields 0.5 + 0.5*r."""
    assert heuristic(is_legal=True, reward=0.6) == 0.8


def test_candidate_rank_key_primary_heuristic() -> None:
    """Rank key sorts by heuristic descending first."""
    high = CandidateRankKey(heuristic=0.9, reward=0.0, legal_actions=0, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=0, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_secondary_reward() -> None:
    """Same heuristic uses reward descending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.8, legal_actions=0, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.3, legal_actions=0, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_tertiary_legal_actions() -> None:
    """Same heuristic and reward uses legal actions descending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=10, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_quaternary_failures() -> None:
    """Same heuristic, reward, legal actions uses failures ascending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=3, iteration=0)
    assert high > low


def test_candidate_rank_key_quinary_iteration() -> None:
    """Same everything uses iteration ascending (earlier is better)."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=2)
    assert high > low


def test_candidate_rank_key_from_candidate_solved() -> None:
    """CandidateRankKey.from_candidate constructs correctly for a solved candidate."""
    cand = Candidate(
        id="003",
        parent_id="001",
        source=(
            "def propose_action(board: str) -> str: ...\n"
            "def is_legal_action(board: str, action: str) -> bool: ..."
        ),
        heuristic=1.0,
        terminal_reward=1.0,
        legal_action_count=7,
        termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
        failure_summary=None,
        iteration=3,
        expansion_count=0,
    )
    key = CandidateRankKey.from_candidate(cand)
    assert key.heuristic == 1.0
    assert key.reward == 1.0
    assert key.legal_actions == 7
    assert key.failures == 0
    assert key.iteration == 3


def test_candidate_rank_key_from_candidate_failure_counts() -> None:
    """Failure count is 1 when termination_reason is EXECUTION_FAILURE or CONTRACT_FAILURE."""
    cand = Candidate(
        id="004",
        parent_id=None,
        source="bad code",
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        termination_reason=TerminationReason.EXECUTION_FAILURE,
        failure_summary="SyntaxError",
        iteration=4,
        expansion_count=0,
    )
    key = CandidateRankKey.from_candidate(cand)
    assert key.failures == 1


def test_candidate_rank_key_uses_aggregate_failure_count() -> None:
    """Aggregate failures outrank the representative termination fallback."""
    cand = Candidate(
        id="005",
        parent_id=None,
        source="source",
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=1,
        termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
        failure_summary=None,
        iteration=5,
        failure_count=2,
    )
    assert CandidateRankKey.from_candidate(cand).failures == 2


def test_event_creation() -> None:
    """Event stores expected fields."""
    ev = Event(
        iteration=1,
        event_type="refine",
        candidate_id="001",
        parent_id="000",
        metadata={"model": "test"},
    )
    assert ev.iteration == 1
    assert ev.event_type == "refine"
    assert ev.candidate_id == "001"


def test_rollout_result_fields() -> None:
    """RolloutResult stores heuristic, reward, legal count, reason."""
    result = RolloutResult(
        steps=[],
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=5,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
    )
    assert result.heuristic == 0.5
    assert result.termination_reason == TerminationReason.STEP_LIMIT


def test_step_result_fields() -> None:
    """StepResult stores observation, action, legality, reward, feedback."""
    step = StepResult(
        observation="[A B C]",
        action="[A C]",
        is_legal=True,
        reward=0.0,
        terminated=False,
        feedback="",
    )
    assert step.is_legal
    assert step.action == "[A C]"


def test_profile_values() -> None:
    """Profile enum has correct refinement budgets."""
    assert Profile.SMOKE.refinements == 8
    assert Profile.LOW_COST.refinements == 32
    assert Profile.FULL_SEARCH.refinements == 256


def test_termination_reason_values() -> None:
    """TerminationReason has all expected members."""
    assert set(TerminationReason) == {
        TerminationReason.ILLEGAL_ACTION,
        TerminationReason.POLICY_REJECTED_ACTION,
        TerminationReason.LEGALITY_DISAGREEMENT,
        TerminationReason.ENVIRONMENT_TERMINATION,
        TerminationReason.STEP_LIMIT,
        TerminationReason.EXECUTION_FAILURE,
        TerminationReason.CONTRACT_FAILURE,
    }
