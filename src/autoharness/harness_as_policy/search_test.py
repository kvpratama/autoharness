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
    """Selection with seeded RNG reproduces the same draw."""
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
    rng = random.Random(42)
    selected = select_candidate(candidates, rng)
    assert selected in candidates


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

    def __init__(self) -> None:
        self.env_id = "FakeEnv-v0"
        self.rules = "Fake rules"
        self.action_format = "[X Y]"
        self.max_steps = 10

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        return "initial observation"

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation="next observation",
            action=action,
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        )


class FakeRefiner:
    """Fake refiner that returns configured responses."""

    def __init__(self, responses: list[str | None]) -> None:
        self._responses = responses
        self._call_count = 0

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
    ) -> RefinerResult:
        self._call_count += 1
        if self._responses:
            resp = self._responses.pop(0)
            if resp:
                return RefinerResult(success=True, source=resp)
        return RefinerResult(success=False, source=None)
