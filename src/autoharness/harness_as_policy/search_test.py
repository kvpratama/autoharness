"""Tests for the LangGraph search workflow."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Profile,
    StepResult,
    TerminationReason,
)
from autoharness.harness_as_policy.refiner import RefinerProtocol, RefinerResult
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
    """Blank policies remain in the tree but are excluded from final ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        refiner: RefinerProtocol = FakeRefiner(responses=["   "])
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=refiner,
            artifact_root=Path(tmpdir),
            seed=42,
        )
        tree_path = Path(tmpdir) / result["run_id"] / "tree.json"
        tree = json.loads(tree_path.read_text())

    assert tree["candidates"]["000"]["ranking"] == {
        "eligible": False,
        "exclusion_reason": "synthetic_root",
        "components": None,
    }
    assert tree["candidates"]["001"]["ranking"] == {
        "eligible": False,
        "exclusion_reason": "empty_source",
        "components": None,
    }
    assert tree["ranking"]["ordered_candidate_ids"] == []
    assert tree["ranking"]["winner_explanation"] is None
    assert tree["best_candidate_id"] is None


def test_synthesize_persists_order_matching_find_best_candidate() -> None:
    """Persisted ranking exactly matches final candidate selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=FakeRefiner(responses=[ACCEPTED_BY_CHECKER_SOURCE, ACCEPTED_BY_CHECKER_SOURCE]),
            artifact_root=Path(tmpdir),
            refinements=2,
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
        "legal_actions": 10,
        "failures": 0,
        "iteration": 1,
    }
    assert tree["ranking"]["winner_explanation"] == {
        "winner_id": "001",
        "runner_up_id": "002",
        "outcome": "decisive_component",
        "tied_components": ["heuristic", "reward", "legal_actions", "failures"],
        "decisive_component": "iteration",
        "winner_value": 1,
        "runner_up_value": 2,
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
    assert adapter.reset_seeds == [None, *seeds, *seeds]
    assert config["environment_seed"] == 17
    assert config["training_rollouts"] == 3


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
