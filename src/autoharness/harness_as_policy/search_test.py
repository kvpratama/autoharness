"""Tests for the LangGraph search workflow."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from autoharness.harness_as_policy.models import (
    Candidate,
    Profile,
    StepResult,
    TerminationReason,
)
from autoharness.harness_as_policy.refiner import RefinerProtocol, RefinerResult
from autoharness.harness_as_policy.search import (
    beta_parameters,
    find_best_candidate,
    select_candidate,
    should_stop,
    synthesize,
)


def test_beta_parameters_no_children() -> None:
    """Beta parameters for candidate with no children and H=0.5."""
    a, b = beta_parameters(heuristic=0.5, children=0, weight=1.0)
    assert abs(a - 1.5) < 1e-10
    assert abs(b - 1.5) < 1e-10


def test_beta_parameters_perfect() -> None:
    """Beta parameters for perfect candidate with H=1.0."""
    a, b = beta_parameters(heuristic=1.0, children=2, weight=1.0)
    assert abs(a - 2.0) < 1e-10
    assert abs(b - 3.0) < 1e-10


def test_beta_parameters_zero() -> None:
    """Beta parameters for zero heuristic."""
    a, b = beta_parameters(heuristic=0.0, children=0, weight=1.0)
    assert abs(a - 1.0) < 1e-10
    assert abs(b - 2.0) < 1e-10


def test_select_candidate_deterministic() -> None:
    """Selection with seeded RNG is deterministic and picks the highest Beta draw."""
    candidates = {
        "000": Candidate(
            id="000",
            parent_id=None,
            source="",
            heuristic=0.0,
            terminal_reward=0.0,
            legal_action_count=0,
            termination_reason=None,
            failure_summary=None,
            iteration=0,
            expansion_count=0,
        ),
        "001": Candidate(
            id="001",
            parent_id="000",
            source="",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=7,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=1,
            expansion_count=0,
        ),
        "002": Candidate(
            id="002",
            parent_id="000",
            source="",
            heuristic=0.8,
            terminal_reward=0.5,
            legal_action_count=10,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            iteration=2,
            expansion_count=1,
        ),
    }

    # Two independent RNG instances with the same seed must produce the same selection.
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    selected_a = select_candidate(candidates, rng_a)
    selected_b = select_candidate(candidates, rng_b)
    assert selected_a == selected_b, "Same seed must yield identical selection"

    # Replay the exact Beta draws to find which candidate gets the highest draw.
    from autoharness.harness_as_policy.search import beta_parameters

    rng_ref = random.Random(42)
    draws: dict[str, float] = {}
    for cid, cand in candidates.items():
        a, b = beta_parameters(heuristic=cand.heuristic, children=cand.expansion_count)
        draws[cid] = rng_ref.betavariate(a, b)

    expected_winner = max(draws, key=lambda k: draws[k])
    winner_draw = draws[expected_winner]
    selected_draw = draws.get(selected_a, -1.0)
    assert selected_a == expected_winner, (
        f"Expected candidate with highest draw ({expected_winner}, draw={winner_draw:.4f}) "
        f"but got {selected_a} (draw={selected_draw:.4f})"
    )


def test_find_best_candidate_empty() -> None:
    """Empty candidate dict returns None."""
    assert find_best_candidate({}) is None


def test_find_best_candidate_single() -> None:
    """Single candidate is the best."""
    c = Candidate(
        id="000",
        parent_id=None,
        source="",
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=5,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        iteration=0,
        expansion_count=0,
    )
    assert find_best_candidate({"000": c}) == "000"


def test_find_best_candidate_lexicographic() -> None:
    """Best candidate follows lexicographic ranking."""
    candidates = {
        "000": Candidate(
            id="000",
            parent_id=None,
            source="",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=5,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=0,
            expansion_count=0,
        ),
        "001": Candidate(
            id="001",
            parent_id="000",
            source="",
            heuristic=0.8,
            terminal_reward=0.6,
            legal_action_count=8,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=1,
            expansion_count=0,
        ),
        "002": Candidate(
            id="002",
            parent_id="001",
            source="",
            heuristic=1.0,
            terminal_reward=1.0,
            legal_action_count=7,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            iteration=2,
            expansion_count=0,
        ),
    }
    assert find_best_candidate(candidates) == "002"


def test_should_stop_success() -> None:
    """Should stop when any candidate has H=1.0."""
    candidates = {
        "000": Candidate(
            id="000",
            parent_id=None,
            source="",
            heuristic=1.0,
            terminal_reward=1.0,
            legal_action_count=7,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            iteration=0,
            expansion_count=0,
        ),
    }
    reason = should_stop(candidates, iteration=1, max_refinements=8)
    assert reason is not None
    assert "success" in reason


def test_should_stop_budget_exhausted() -> None:
    """Should stop when iteration reaches max_refinements."""
    reason = should_stop({}, iteration=8, max_refinements=8)
    assert reason is not None
    assert "budget" in reason


def test_should_stop_not_yet() -> None:
    """Should not stop when budget remains and no success."""
    reason = should_stop({}, iteration=0, max_refinements=8)
    assert reason is None


def test_synthesize_empty_policies() -> None:
    """synthesize with only root and one failed refinement returns summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        refiner: RefinerProtocol = FakeRefiner(responses=[""])
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=refiner,
            artifact_root=Path(tmpdir),
            seed=42,
        )
    assert result["stop_reason"] is not None
    assert "iterations_used" in result
    assert "best_candidate_id" in result
    assert "total_candidates" in result


class FakeAdapter:
    """Fake environment adapter for testing synthesis."""

    def __init__(self, *, reject_actions: bool = False) -> None:
        self.env_id = "FakeEnv-v0"
        self.rules = "Fake rules"
        self.action_format = "[X Y]"
        self.max_steps = 10
        self.reject_actions = reject_actions
        self.step_calls: list[str] = []

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        return "initial observation"

    def step(self, action: str) -> StepResult:
        self.step_calls.append(action)
        return StepResult(
            observation="next observation",
            action=action,
            is_legal=not self.reject_actions,
            reward=0.0,
            terminated=False,
            feedback="",
        )


class FakeRefiner:
    """Fake refiner that returns configured responses."""

    def __init__(self, responses: list[str | None]) -> None:
        self._responses = responses
        self._call_count = 0
        self.scopes: list[bool] = []
        self.feedback: list[list[str]] = []

    @property
    def model_call_count(self) -> int:
        return self._call_count

    @property
    def logical_refinement_count(self) -> int:
        return self._call_count

    def refine(
        self,
        rules: str = "",
        action_format: str = "",
        parent_source: str = "",
        parent_heuristic: float = 0.0,
        parent_reward: float = 0.0,
        parent_legal_actions: int = 0,
        parent_status: str = "",
        feedback: list[str] | None = None,
        env_name: str = "",
        *,
        refine_legal_action: bool,
    ) -> RefinerResult:
        self._call_count += 1
        self.scopes.append(refine_legal_action)
        self.feedback.append(feedback or [])
        if self._responses:
            resp = self._responses.pop(0)
            if resp:
                return RefinerResult(success=True, source=resp)
        return RefinerResult(success=False, source=None)


REJECTED_BY_CHECKER_SOURCE = """def propose_action(observation: str) -> str:
    return '[X Y]'

def is_legal_action(observation: str, action: str) -> bool:
    return False
"""

ACCEPTED_BY_CHECKER_SOURCE = """def propose_action(observation: str) -> str:
    return '[X Y]'

def is_legal_action(observation: str, action: str) -> bool:
    return True
"""


def test_synthesize_refines_only_action_after_checker_rejection() -> None:
    """Checker rejection preserves the checker on the next refinement."""
    adapter = FakeAdapter()
    refiner = FakeRefiner([REJECTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE])
    with tempfile.TemporaryDirectory() as tmpdir:
        synthesize(
            adapter=adapter,
            profile=Profile.SMOKE,
            refiner=refiner,
            artifact_root=Path(tmpdir),
            refinements=2,
        )

    assert refiner.scopes == [True, False]
    assert refiner.feedback[1][0] == (
        "Policy legality checker rejected action '[X Y]' (checker=False)"
    )
    assert refiner.feedback[1][1] == (
        "is_legal_action rejected the proposed action; refine propose_action only"
    )
    assert adapter.step_calls == ["[X Y]"] * adapter.max_steps


def test_synthesize_refines_both_after_legality_disagreement() -> None:
    """Environment disagreement allows refining the checker and action policy."""
    adapter = FakeAdapter(reject_actions=True)
    refiner = FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE])
    with tempfile.TemporaryDirectory() as tmpdir:
        synthesize(
            adapter=adapter,
            profile=Profile.SMOKE,
            refiner=refiner,
            artifact_root=Path(tmpdir),
            refinements=2,
        )

    assert refiner.scopes == [True, True]
    assert refiner.feedback[1][0] == (
        "Legality disagreement: checker=True, environment=False; environment feedback: Illegal "
        "action"
    )
    assert refiner.feedback[1][1] == (
        "is_legal_action accepted an action that the environment rejected; refine both functions"
    )
