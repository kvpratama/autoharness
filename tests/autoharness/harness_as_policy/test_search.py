"""Tests for the LangGraph search workflow."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from autoharness.environments.models import StepResult
from autoharness.harness_as_policy.executor import ExecutionResult, policy_randomness_metadata
from autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Profile,
    ProviderInvocation,
    RefinementOutcome,
    RefinementTrace,
    TerminationReason,
)
from autoharness.harness_as_policy.refiner import (
    RefinerResult,
)
from autoharness.harness_as_policy.search import (
    RANKING_POLICY,
    _winner_explanation,
    beta_parameters,
    find_best_candidate,
    rank_candidates,
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


def test_ranking_policy_matches_candidate_rank_key_order() -> None:
    """Documented ranking precedence and directions match the executable rank key."""
    expected_policy = (
        ("heuristic", "descending"),
        ("reward", "descending"),
        ("legal_actions", "descending"),
        ("failures", "ascending"),
        ("iteration", "ascending"),
    )
    comparison_cases = (
        (
            CandidateRankKey(0.6, 0.0, 0, 1, 2),
            CandidateRankKey(0.5, 1.0, 10, 0, 1),
        ),
        (
            CandidateRankKey(0.5, 0.6, 0, 1, 2),
            CandidateRankKey(0.5, 0.5, 10, 0, 1),
        ),
        (
            CandidateRankKey(0.5, 0.5, 6, 1, 2),
            CandidateRankKey(0.5, 0.5, 5, 0, 1),
        ),
        (
            CandidateRankKey(0.5, 0.5, 5, 0, 2),
            CandidateRankKey(0.5, 0.5, 5, 1, 1),
        ),
        (
            CandidateRankKey(0.5, 0.5, 5, 0, 1),
            CandidateRankKey(0.5, 0.5, 5, 0, 2),
        ),
    )

    assert RANKING_POLICY == expected_policy
    assert len(comparison_cases) == len(expected_policy)
    for better, worse in comparison_cases:
        assert better > worse


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


def test_rank_candidates_returns_complete_lexicographic_order() -> None:
    """Candidate ordering exposes the complete best-to-worst rank."""
    candidates = {
        "reward": Candidate(
            id="reward",
            parent_id=None,
            source="policy",
            heuristic=0.5,
            terminal_reward=0.5,
            legal_action_count=1,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=4,
        ),
        "earlier": Candidate(
            id="earlier",
            parent_id=None,
            source="policy",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=3,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=1,
        ),
        "failure": Candidate(
            id="failure",
            parent_id=None,
            source="policy",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=3,
            termination_reason=TerminationReason.EXECUTION_FAILURE,
            failure_summary="failed",
            iteration=2,
        ),
        "later": Candidate(
            id="later",
            parent_id=None,
            source="policy",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=3,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=3,
        ),
    }

    assert rank_candidates(candidates) == ["reward", "earlier", "later", "failure"]
    assert find_best_candidate(candidates) == "reward"


def test_rank_candidates_preserves_input_order_for_equal_keys() -> None:
    """Exact rank-key ties retain stable candidate input order."""
    first = Candidate(
        id="first",
        parent_id=None,
        source="policy",
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=3,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        iteration=1,
    )
    second = Candidate(
        id="second",
        parent_id=None,
        source="policy",
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=3,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        iteration=1,
    )

    assert rank_candidates({"first": first, "second": second}) == ["first", "second"]


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
            rollout_eligible=True,
        ),
    }
    reason = should_stop(candidates, iteration=1, max_refinements=8)
    assert reason is not None
    assert "success" in reason


def test_should_stop_strict_equality() -> None:
    """Should not stop early when candidate heuristic is close to but not exactly 1.0."""
    candidates = {
        "000": Candidate(
            id="000",
            parent_id=None,
            source="",
            heuristic=0.999,
            terminal_reward=0.999,
            legal_action_count=7,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            iteration=0,
            expansion_count=0,
        ),
    }
    reason = should_stop(candidates, iteration=1, max_refinements=8)
    assert reason is None


def test_should_stop_ignores_ineligible_perfect_candidate() -> None:
    candidate = Candidate(
        id="001",
        parent_id="000",
        source="policy",
        heuristic=1.0,
        terminal_reward=1.0,
        legal_action_count=1,
        termination_reason=TerminationReason.CONTRACT_FAILURE,
        failure_summary="invalid",
        iteration=1,
        rollout_eligible=False,
    )

    assert should_stop({"001": candidate}, iteration=1, max_refinements=2) is None


def test_should_stop_budget_exhausted() -> None:
    """Should stop when iteration reaches max_refinements."""
    reason = should_stop({}, iteration=8, max_refinements=8)
    assert reason is not None
    assert "budget" in reason


def test_should_stop_not_yet() -> None:
    """Should not stop when budget remains and no success."""
    reason = should_stop({}, iteration=0, max_refinements=8)
    assert reason is None


def test_blank_refinement_does_not_create_node_and_search_retries_root(tmp_path: Path) -> None:
    refiner = FakeRefiner([None, ACCEPTED_BY_CHECKER_SOURCE])

    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=refiner,
        artifact_root=tmp_path,
        refinements=2,
    )

    run_dir = tmp_path / result["run_id"]
    tree = json.loads((run_dir / "tree.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    failed_refinement = next(
        event for event in events if event["event_type"] == "refine" and event["iteration"] == 1
    )

    assert set(tree["candidates"]) == {"000", "002"}
    assert tree["candidates"]["002"]["parent_id"] == "000"
    assert tree["best_candidate_id"] == "002"
    assert not (run_dir / "candidates" / "001.py").exists()
    assert not (run_dir / "rollouts" / "001.json").exists()
    assert (run_dir / "candidates" / "002.py").exists()
    assert (run_dir / "rollouts" / "002.json").exists()
    assert (run_dir / "best.py").exists()
    assert [event["candidate_id"] for event in events if event["event_type"] == "select"] == [
        "000",
        "000",
    ]
    assert failed_refinement["candidate_id"] is None
    assert failed_refinement["metadata"] == {
        "success": False,
        "generation_succeeded": False,
        "contract_valid": False,
    }
    assert result["attempted_refinements"] == 2
    assert result["successful_tree_nodes"] == 1
    assert result["provider_calls"] == 2
    assert "iterations_used" not in result
    assert "logical_refinement_count" not in result
    assert "model_call_count" not in result

    summary = json.loads((run_dir / "synthesis-summary.json").read_text())
    assert summary["attempted_refinements"] == 2
    assert summary["successful_tree_nodes"] == 1
    assert summary["provider_calls"] == 2


def test_all_failed_assessment_cannot_be_selected_ranked_or_published(tmp_path: Path) -> None:
    execution_failing_source = """def propose_action(observation: str) -> str:
    raise RuntimeError("failing policy")

def is_legal_action(observation: str, action: str) -> bool:
    return True
"""
    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=FakeRefiner([execution_failing_source, ACCEPTED_BY_CHECKER_SOURCE]),
        artifact_root=tmp_path,
        refinements=2,
    )

    run_dir = tmp_path / result["run_id"]
    tree = json.loads((run_dir / "tree.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    assert tree["candidates"]["001"]["rollout_eligible"] is False
    assert tree["candidates"]["001"]["ranking"] == {
        "eligible": False,
        "exclusion_reason": "failed_assessment",
        "components": None,
    }
    assert tree["candidates"]["002"]["parent_id"] == "000"
    assert tree["ranking"]["ordered_candidate_ids"] == ["002"]
    assert tree["best_candidate_id"] == "002"
    assert (run_dir / "best.py").read_text() == ACCEPTED_BY_CHECKER_SOURCE
    assert [event["candidate_id"] for event in events if event["event_type"] == "select"] == [
        "000",
        "000",
    ]


def test_all_failed_assessment_does_not_create_best_policy(tmp_path: Path) -> None:
    execution_failing_source = """def propose_action(observation: str) -> str:
    raise RuntimeError("failing policy")

def is_legal_action(observation: str, action: str) -> bool:
    return True
"""
    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=FakeRefiner([execution_failing_source]),
        artifact_root=tmp_path,
        refinements=1,
    )

    run_dir = tmp_path / result["run_id"]
    tree = json.loads((run_dir / "tree.json").read_text())
    assert tree["ranking"]["ordered_candidate_ids"] == []
    assert tree["best_candidate_id"] is None
    assert not (run_dir / "best.py").exists()


def test_synthesize_persists_order_matching_find_best_candidate() -> None:
    """Persisted ranking exactly matches final candidate selection."""
    failing_source = """def propose_action(observation: str) -> str:
    raise RuntimeError("failing policy")

def is_legal_action(observation: str, action: str) -> bool:
    return True
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=FakeRefiner(responses=[ACCEPTED_BY_CHECKER_SOURCE, failing_source]),
            artifact_root=Path(tmpdir),
            refinements=2,
            training_rollouts=2,
        )
        tree_path = Path(tmpdir) / result["run_id"] / "tree.json"
        tree = json.loads(tree_path.read_text())

    reconstructed_candidates = {
        candidate_id: Candidate(
            id=data["id"],
            parent_id=data["parent_id"],
            source="persisted-policy",
            heuristic=data["heuristic"],
            terminal_reward=data["terminal_reward"],
            legal_action_count=data["legal_action_count"],
            termination_reason=(
                TerminationReason(data["termination_reason"])
                if data["termination_reason"]
                else None
            ),
            failure_summary=data["failure_summary"],
            iteration=data["iteration"],
            expansion_count=data["expansion_count"],
            failure_count=data["failure_count"],
            episode_count=data["episode_count"],
            rollout_eligible=data["rollout_eligible"],
        )
        for candidate_id, data in tree["candidates"].items()
        if data["ranking"]["eligible"]
    }
    persisted_order = tree["ranking"]["ordered_candidate_ids"]

    assert persisted_order == rank_candidates(reconstructed_candidates)
    assert persisted_order[0] == find_best_candidate(reconstructed_candidates)
    assert tree["best_candidate_id"] == persisted_order[0]
    assert tree["ranking"]["strategy"] == "candidate_rank_key_v1"
    assert tree["ranking"]["policy"] == [
        {"component": "heuristic", "direction": "descending"},
        {"component": "reward", "direction": "descending"},
        {"component": "legal_actions", "direction": "descending"},
        {"component": "failures", "direction": "ascending"},
        {"component": "iteration", "direction": "ascending"},
    ]
    assert tree["candidates"]["001"]["parent_id"] == "000"
    assert tree["candidates"]["002"]["parent_id"] == "001"
    assert tree["candidates"]["001"]["ranking"]["components"] == {
        "heuristic": 0.5,
        "reward": 0.0,
        "legal_actions": 20,
        "failures": 0,
        "iteration": 1,
    }
    assert tree["candidates"]["002"]["failure_count"] == 2
    assert tree["candidates"]["002"]["episode_count"] == 2
    assert tree["candidates"]["002"]["ranking"] == {
        "eligible": False,
        "exclusion_reason": "failed_assessment",
        "components": None,
    }
    assert tree["ranking"]["winner_explanation"] == {
        "winner_id": "001",
        "runner_up_id": None,
        "outcome": "only_eligible_candidate",
        "tied_components": [],
        "decisive_component": None,
        "winner_value": None,
        "runner_up_value": None,
    }


def test_synthesize_explains_single_eligible_candidate() -> None:
    """A sole ranked policy records why no comparison was needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=FakeRefiner(responses=[ACCEPTED_BY_CHECKER_SOURCE]),
            artifact_root=Path(tmpdir),
            refinements=1,
        )
        tree = json.loads((Path(tmpdir) / result["run_id"] / "tree.json").read_text())

    assert tree["ranking"]["winner_explanation"] == {
        "winner_id": "001",
        "runner_up_id": None,
        "outcome": "only_eligible_candidate",
        "tied_components": [],
        "decisive_component": None,
        "winner_value": None,
        "runner_up_value": None,
    }


def test_winner_explanation_matches_each_ranking_component() -> None:
    """Winner explanations follow CandidateRankKey precedence for every component."""
    cases = [
        (
            "heuristic",
            Candidate(
                "winner",
                None,
                "policy",
                0.6,
                0.0,
                3,
                TerminationReason.STEP_LIMIT,
                None,
                1,
            ),
            Candidate(
                "runner",
                None,
                "policy",
                0.5,
                0.0,
                3,
                TerminationReason.STEP_LIMIT,
                None,
                1,
            ),
            [],
            0.6,
            0.5,
        ),
        (
            "reward",
            Candidate("winner", None, "policy", 0.5, 0.5, 3, TerminationReason.STEP_LIMIT, None, 1),
            Candidate("runner", None, "policy", 0.5, 0.0, 3, TerminationReason.STEP_LIMIT, None, 1),
            ["heuristic"],
            0.5,
            0.0,
        ),
        (
            "legal_actions",
            Candidate("winner", None, "policy", 0.5, 0.0, 4, TerminationReason.STEP_LIMIT, None, 1),
            Candidate("runner", None, "policy", 0.5, 0.0, 3, TerminationReason.STEP_LIMIT, None, 1),
            ["heuristic", "reward"],
            4,
            3,
        ),
        (
            "failures",
            Candidate("winner", None, "policy", 0.5, 0.0, 3, TerminationReason.STEP_LIMIT, None, 1),
            Candidate(
                "runner",
                None,
                "policy",
                0.5,
                0.0,
                3,
                TerminationReason.EXECUTION_FAILURE,
                "failed",
                1,
            ),
            ["heuristic", "reward", "legal_actions"],
            0,
            1,
        ),
        (
            "iteration",
            Candidate("winner", None, "policy", 0.5, 0.0, 3, TerminationReason.STEP_LIMIT, None, 1),
            Candidate("runner", None, "policy", 0.5, 0.0, 3, TerminationReason.STEP_LIMIT, None, 2),
            ["heuristic", "reward", "legal_actions", "failures"],
            1,
            2,
        ),
    ]

    for component, winner, runner_up, tied_components, winner_value, runner_up_value in cases:
        candidates = {"winner": winner, "runner": runner_up}
        ordered_candidate_ids = rank_candidates(candidates)
        explanation = _winner_explanation(candidates, ordered_candidate_ids)

        assert ordered_candidate_ids == ["winner", "runner"]
        assert explanation is not None
        assert explanation["outcome"] == "decisive_component"
        assert explanation["decisive_component"] == component
        assert explanation["tied_components"] == tied_components
        assert explanation["winner_value"] == winner_value
        assert explanation["runner_up_value"] == runner_up_value


def test_winner_explanation_records_exact_key_tie() -> None:
    """A complete key tie records stable input order as the deciding rule."""
    candidates = {
        candidate_id: Candidate(
            id=candidate_id,
            parent_id=None,
            source="policy",
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=3,
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
            iteration=1,
        )
        for candidate_id in ("first", "second")
    }

    assert _winner_explanation(candidates, ["first", "second"]) == {
        "winner_id": "first",
        "runner_up_id": "second",
        "outcome": "exact_key_tie",
        "tied_components": [
            "heuristic",
            "reward",
            "legal_actions",
            "failures",
            "iteration",
        ],
        "decisive_component": None,
        "winner_value": None,
        "runner_up_value": None,
    }


class FakeAdapter:
    """Fake environment adapter for testing synthesis."""

    def __init__(self, *, reject_actions: bool = False) -> None:
        self.env_id = "FakeEnv-v0"
        self.rules = "Fake rules"
        self.action_format = "[X Y]"
        self.max_steps = 10
        self.reject_actions = reject_actions
        self.step_calls: list[str] = []
        self.reset_seeds: list[int | None] = []

    def create(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> str:
        self.reset_seeds.append(seed)
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
        self.trajectories: list[str] = []
        self._last_trace: RefinementTrace | None = None

    @property
    def model_call_count(self) -> int:
        return self._call_count

    @property
    def logical_refinement_count(self) -> int:
        return self._call_count

    @property
    def last_trace(self) -> RefinementTrace | None:
        return self._last_trace

    def refine(
        self,
        rules: str = "",
        action_format: str = "",
        parent_source: str = "",
        parent_heuristic: float = 0.0,
        parent_reward: float = 0.0,
        parent_legal_actions: int = 0,
        parent_status: str = "",
        trajectory: str = "",
        env_name: str = "",
        *,
        refine_legal_action: bool,
    ) -> RefinerResult:
        self._call_count += 1
        self.scopes.append(refine_legal_action)
        self.trajectories.append(trajectory)
        self._last_trace = RefinementTrace(
            prompt=f"prompt:{trajectory}", outcome=RefinementOutcome.SUCCESS
        )
        if self._responses:
            resp = self._responses.pop(0)
            if resp:
                self._last_trace.extracted_source = resp
                self._last_trace.generation_succeeded = True
                self._last_trace.contract_valid = True
                return RefinerResult(
                    success=True,
                    source=resp,
                    generation_succeeded=True,
                    contract_valid=True,
                )
        self._last_trace.outcome = RefinementOutcome.INVALID_RESPONSE
        return RefinerResult(success=False, source=None)


class RetriedFakeRefiner(FakeRefiner):
    """Fake one internal provider retry per logical refinement."""

    @property
    def model_call_count(self) -> int:
        return self._call_count * 2


class InconsistentFakeRefiner(FakeRefiner):
    """Fake a protocol implementation with contradictory success flags."""

    def refine(
        self,
        rules: str = "",
        action_format: str = "",
        parent_source: str = "",
        parent_heuristic: float = 0.0,
        parent_reward: float = 0.0,
        parent_legal_actions: int = 0,
        parent_status: str = "",
        trajectory: str = "",
        env_name: str = "",
        *,
        refine_legal_action: bool,
    ) -> RefinerResult:
        result = super().refine(
            rules=rules,
            action_format=action_format,
            parent_source=parent_source,
            parent_heuristic=parent_heuristic,
            parent_reward=parent_reward,
            parent_legal_actions=parent_legal_actions,
            parent_status=parent_status,
            trajectory=trajectory,
            env_name=env_name,
            refine_legal_action=refine_legal_action,
        )
        return RefinerResult(
            success=True,
            source=result.source,
            generation_succeeded=True,
            contract_valid=False,
        )


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


def test_synthesis_reports_provider_retries_separately(tmp_path: Path) -> None:
    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=RetriedFakeRefiner([ACCEPTED_BY_CHECKER_SOURCE]),
        artifact_root=tmp_path,
        refinements=1,
    )

    assert result["attempted_refinements"] == 1
    assert result["successful_tree_nodes"] == 1
    assert result["provider_calls"] == 2


def test_synthesis_reports_run_local_counts_when_refiner_is_reused(tmp_path: Path) -> None:
    refiner = FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE])
    refiner.refine(refine_legal_action=True)

    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=refiner,
        artifact_root=tmp_path,
        refinements=1,
    )

    assert result["attempted_refinements"] == 1
    assert result["provider_calls"] == 1


def test_inconsistent_refiner_success_does_not_reference_candidate(tmp_path: Path) -> None:
    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=InconsistentFakeRefiner([ACCEPTED_BY_CHECKER_SOURCE]),
        artifact_root=tmp_path,
        refinements=1,
    )

    run_dir = tmp_path / result["run_id"]
    tree = json.loads((run_dir / "tree.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    refinement = next(event for event in events if event["event_type"] == "refine")

    assert set(tree["candidates"]) == {"000"}
    assert refinement["candidate_id"] is None
    assert refinement["metadata"]["success"] is False


def test_failed_refinement_logs_error_without_candidate_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailedRefiner(FakeRefiner):
        def refine(
            self,
            rules: str = "",
            action_format: str = "",
            parent_source: str = "",
            parent_heuristic: float = 0.0,
            parent_reward: float = 0.0,
            parent_legal_actions: int = 0,
            parent_status: str = "",
            trajectory: str = "",
            env_name: str = "",
            *,
            refine_legal_action: bool,
        ) -> RefinerResult:
            self._call_count += 1
            return RefinerResult(
                success=False,
                source=None,
                error_details="provider failed",
            )

    caplog.set_level("INFO", logger="autoharness.harness_as_policy.search")

    synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=FailedRefiner([]),
        artifact_root=tmp_path,
        refinements=1,
    )

    failure_logs = [
        message for message in caplog.messages if message.startswith("Refinement failed")
    ]
    assert failure_logs == ["Refinement failed — provider failed"]
    assert "candidate" not in failure_logs[0]


def test_synthesize_reuses_shared_environment_seeds_for_every_candidate(tmp_path: Path) -> None:
    """All assessed candidates receive the same ordered training seed list."""
    adapter = FakeAdapter()
    result = synthesize(
        adapter=adapter,
        profile=Profile.SMOKE,
        refiner=FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE]),
        artifact_root=tmp_path,
        refinements=2,
        environment_seed=17,
        training_rollouts=3,
    )
    config = json.loads((tmp_path / result["run_id"] / "config.json").read_text())
    seeds = config["training_episode_seeds"]
    assert len(seeds) == 3
    assert adapter.reset_seeds == [*seeds, *seeds, *seeds]
    assert config["environment_seed"] == 17
    assert config["training_rollouts"] == 3
    assert config["policy_randomness"] == policy_randomness_metadata()
    assert config["policy_randomness"]["state_model"] == "fresh-subprocess-per-action"
    assert config["policy_randomness"]["seed_inputs"] == [
        "episode_seed",
        "zero_based_policy_invocation_index",
    ]


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
    assert "Policy legality check: false" in refiner.trajectories[1]
    assert "Policy legality checker rejected action" in refiner.trajectories[1]
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
    assert "Policy legality check: true" in refiner.trajectories[1]
    assert "Environment legality check: false" in refiner.trajectories[1]
    assert "Legality disagreement" in refiner.trajectories[1]


class SeedAwareAdapter(FakeAdapter):
    def reset(self, seed: int | None = None) -> str:
        self.reset_seeds.append(seed)
        return f"initial board seed={seed}"


def test_first_refinement_receives_every_seeded_board_without_executing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_sources: list[str] = []

    class _RecordingSession:
        def __init__(self, source: str) -> None:
            self._source = source

        def __enter__(self) -> _RecordingSession:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def execute(self, observation: str, *, policy_seed: int) -> ExecutionResult:
            executed_sources.append(self._source)
            return ExecutionResult(
                success=True,
                output="[X Y]",
                latency=0.0,
                is_legal_action=True,
                policy_seed=policy_seed,
            )

    class RecordingExecutor:
        def __init__(self, timeout: int, max_source_size: int) -> None:
            self.timeout = timeout
            self.max_source_size = max_source_size

        def begin_session(self, source: str) -> _RecordingSession:
            return _RecordingSession(source)

    monkeypatch.setattr(
        "autoharness.harness_as_policy.search.PolicyExecutor",
        RecordingExecutor,
    )
    adapter = SeedAwareAdapter()
    refiner = FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE])

    result = synthesize(
        adapter=adapter,
        profile=Profile.SMOKE,
        refiner=refiner,
        artifact_root=tmp_path,
        refinements=1,
        environment_seed=17,
        training_rollouts=3,
    )

    seeds = json.loads((tmp_path / result["run_id"] / "config.json").read_text())[
        "training_episode_seeds"
    ]
    first = refiner.trajectories[0]
    assert all(f"Seed: {seed}" in first for seed in seeds)
    assert all(f"initial board seed={seed}" in first for seed in seeds)
    assert first.count("No action attempted; implement the initial policy.") == 3
    assert "Root policy — replace me" not in first
    assert executed_sources
    from autoharness.harness_as_policy.search import ROOT_SOURCE

    assert ROOT_SOURCE not in executed_sources


def test_later_refinement_receives_all_episode_attempts_without_truncation(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    adapter.max_steps = 6
    refiner = FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE])

    synthesize(
        adapter=adapter,
        profile=Profile.SMOKE,
        refiner=refiner,
        artifact_root=tmp_path,
        refinements=2,
        training_rollouts=6,
    )

    second = refiner.trajectories[1]
    assert second.count("Episode ") == 6
    assert second.count("Attempt 6") == 6
    assert second.count("Proposed action:\n[X Y]") == 36


def test_provider_error_trace_is_persisted_before_propagation(tmp_path: Path) -> None:
    class RaisingRefiner(FakeRefiner):
        def refine(
            self,
            rules: str = "",
            action_format: str = "",
            parent_source: str = "",
            parent_heuristic: float = 0.0,
            parent_reward: float = 0.0,
            parent_legal_actions: int = 0,
            parent_status: str = "",
            trajectory: str = "",
            env_name: str = "",
            *,
            refine_legal_action: bool,
        ) -> RefinerResult:
            self._call_count += 1
            self._last_trace = RefinementTrace(
                prompt=f"exact:{trajectory}",
                invocations=[
                    ProviderInvocation(error_type="ValueError", error_message="provider failed")
                ],
                outcome=RefinementOutcome.PROVIDER_ERROR,
                error_details="provider failed",
            )
            raise ValueError("provider failed")

    with pytest.raises(ValueError, match="provider failed"):
        synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=RaisingRefiner([]),
            artifact_root=tmp_path,
            refinements=1,
        )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    data = json.loads((run_dir / "refinements" / "001.json").read_text())
    assert data["outcome"] == "provider_error"
    assert data["prompt"].startswith("exact:Episode 1")
    assert data["invocations"][0] == {
        "content": None,
        "normalized_text": None,
        "error_type": "ValueError",
        "error_message": "provider failed",
    }


def test_refinement_persistence_failure_does_not_abort_successful_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_refinement(*args: object, **kwargs: object) -> None:
        raise OSError("artifact write failed")

    monkeypatch.setattr(
        "autoharness.harness_as_policy.search.ArtifactStore.write_refinement",
        fail_write_refinement,
    )

    result = synthesize(
        adapter=FakeAdapter(),
        profile=Profile.SMOKE,
        refiner=FakeRefiner([ACCEPTED_BY_CHECKER_SOURCE]),
        artifact_root=tmp_path,
        refinements=1,
    )

    assert result["attempted_refinements"] == 1


def test_refinement_persistence_failure_does_not_replace_refiner_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_refinement(*args: object, **kwargs: object) -> None:
        raise OSError("artifact write failed")

    def fail_refine(**kwargs: object) -> RefinerResult:
        raise ValueError("provider failed")

    refiner = FakeRefiner([])
    refiner._last_trace = RefinementTrace(prompt="prompt", outcome=RefinementOutcome.PROVIDER_ERROR)
    monkeypatch.setattr(refiner, "refine", fail_refine)
    monkeypatch.setattr(
        "autoharness.harness_as_policy.search.ArtifactStore.write_refinement",
        fail_write_refinement,
    )

    with pytest.raises(ValueError, match="provider failed"):
        synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=refiner,
            artifact_root=tmp_path,
            refinements=1,
        )
