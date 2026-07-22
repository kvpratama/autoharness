"""LangGraph search workflow with REx Thompson selection."""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from autoharness.harness_as_policy.artifacts import ArtifactStore
from autoharness.harness_as_policy.assessment import (
    CandidateAssessor,
    build_assessment_feedback,
    failed_assessment,
    generate_episode_seeds,
    should_refine_legal_action,
)
from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.executor import PolicyExecutor
from autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    TerminationReason,
)
from autoharness.harness_as_policy.refiner import RefinerProtocol
from autoharness.harness_as_policy.rollout import RolloutEvaluator

logger = logging.getLogger(__name__)

ROOT_ID = "000"

_RankingComponent = Literal["heuristic", "reward", "legal_actions", "failures", "iteration"]
_RankingDirection = Literal["ascending", "descending"]
_RankingExclusionReason = Literal["synthetic_root", "empty_source"]
_WinnerOutcome = Literal[
    "only_eligible_candidate",
    "decisive_component",
    "exact_key_tie",
]


class _RankingComponents(TypedDict):
    heuristic: float
    reward: float
    legal_actions: int
    failures: int
    iteration: int


class _CandidateRankingArtifact(TypedDict):
    eligible: bool
    exclusion_reason: _RankingExclusionReason | None
    components: _RankingComponents | None


class _RankingPolicyEntry(TypedDict):
    component: _RankingComponent
    direction: _RankingDirection


class _WinnerExplanation(TypedDict):
    winner_id: str
    runner_up_id: str | None
    outcome: _WinnerOutcome
    tied_components: list[_RankingComponent]
    decisive_component: _RankingComponent | None
    winner_value: float | int | None
    runner_up_value: float | int | None


class _RankingArtifact(TypedDict):
    strategy: Literal["candidate_rank_key_v1"]
    policy: list[_RankingPolicyEntry]
    ordered_candidate_ids: list[str]
    winner_explanation: _WinnerExplanation | None


RANKING_POLICY: tuple[tuple[_RankingComponent, _RankingDirection], ...] = (
    ("heuristic", "descending"),
    ("reward", "descending"),
    ("legal_actions", "descending"),
    ("failures", "ascending"),
    ("iteration", "ascending"),
)


def _ranking_exclusion_reason(
    candidate_id: str,
    candidate: Candidate,
) -> _RankingExclusionReason | None:
    if candidate_id == ROOT_ID:
        return "synthetic_root"
    if not candidate.source.strip():
        return "empty_source"
    return None


def _ranking_components(candidate: Candidate) -> _RankingComponents:
    key = CandidateRankKey.from_candidate(candidate)
    return {
        "heuristic": key.heuristic,
        "reward": key.reward,
        "legal_actions": key.legal_actions,
        "failures": key.failures,
        "iteration": key.iteration,
    }


def _candidate_ranking_artifact(
    candidate_id: str,
    candidate: Candidate,
) -> _CandidateRankingArtifact:
    exclusion_reason = _ranking_exclusion_reason(candidate_id, candidate)
    return {
        "eligible": exclusion_reason is None,
        "exclusion_reason": exclusion_reason,
        "components": _ranking_components(candidate) if exclusion_reason is None else None,
    }


def _winner_explanation(
    candidates: dict[str, Candidate],
    ordered_candidate_ids: list[str],
) -> _WinnerExplanation | None:
    if not ordered_candidate_ids:
        return None

    winner_id = ordered_candidate_ids[0]
    if len(ordered_candidate_ids) == 1:
        return {
            "winner_id": winner_id,
            "runner_up_id": None,
            "outcome": "only_eligible_candidate",
            "tied_components": [],
            "decisive_component": None,
            "winner_value": None,
            "runner_up_value": None,
        }

    runner_up_id = ordered_candidate_ids[1]
    winner_components = _ranking_components(candidates[winner_id])
    runner_up_components = _ranking_components(candidates[runner_up_id])
    tied_components: list[_RankingComponent] = []
    for component, _direction in RANKING_POLICY:
        winner_value = winner_components[component]
        runner_up_value = runner_up_components[component]
        if winner_value != runner_up_value:
            return {
                "winner_id": winner_id,
                "runner_up_id": runner_up_id,
                "outcome": "decisive_component",
                "tied_components": tied_components,
                "decisive_component": component,
                "winner_value": winner_value,
                "runner_up_value": runner_up_value,
            }
        tied_components.append(component)

    return {
        "winner_id": winner_id,
        "runner_up_id": runner_up_id,
        "outcome": "exact_key_tie",
        "tied_components": tied_components,
        "decisive_component": None,
        "winner_value": None,
        "runner_up_value": None,
    }


def _ranking_artifact(
    candidates: dict[str, Candidate],
    ordered_candidate_ids: list[str],
) -> _RankingArtifact:
    return {
        "strategy": "candidate_rank_key_v1",
        "policy": [
            {"component": component, "direction": direction}
            for component, direction in RANKING_POLICY
        ],
        "ordered_candidate_ids": ordered_candidate_ids,
        "winner_explanation": _winner_explanation(candidates, ordered_candidate_ids),
    }


def beta_parameters(
    heuristic: float,
    children: int,
    weight: float = 1.0,
) -> tuple[float, float]:
    """Compute Beta distribution parameters for Thompson sampling.

    a = 1 + w * H
    b = 1 + w * (1 - H) + C
    """
    a = 1.0 + weight * heuristic
    b = 1.0 + weight * (1.0 - heuristic) + children
    return a, b


def select_candidate(
    candidates: dict[str, Candidate],
    rng: random.Random,
) -> str | None:
    """
    Candidate selection uses Thompson sampling:
    for each candidate, draw a random score from a Beta distribution shaped
    by its performance and how many times it has been expanded (direct children).
    Higher performance tends to produce higher draws; more expansions tend to produce lower draws.
    The candidate with the highest draw is selected. This balances exploiting strong candidates
    with exploring less-used ones, so a high performer is favored but not endlessly re-expanded.
    """
    if not candidates:
        return None
    best_id: str | None = None
    best_draw: float = -1.0
    for cid, cand in candidates.items():
        a, b = beta_parameters(
            heuristic=cand.heuristic,
            children=cand.expansion_count,
        )
        draw = rng.betavariate(a, b)
        if draw > best_draw:
            best_draw = draw
            best_id = cid
    return best_id


def rank_candidates(candidates: dict[str, Candidate]) -> list[str]:
    """Return candidate IDs in stable best-to-worst lexicographic order."""
    return sorted(
        candidates,
        key=lambda candidate_id: CandidateRankKey.from_candidate(candidates[candidate_id]),
        reverse=True,
    )


def find_best_candidate(
    candidates: dict[str, Candidate],
) -> str | None:
    """Find the best candidate using lexicographic ranking."""
    ordered_candidate_ids = rank_candidates(candidates)
    return ordered_candidate_ids[0] if ordered_candidate_ids else None


def should_stop(
    candidates: dict[str, Candidate],
    iteration: int,
    max_refinements: int,
) -> str | None:
    """Check termination conditions. Returns stop reason or None."""
    for cand in candidates.values():
        if cand.heuristic == 1.0:
            return f"success: candidate {cand.id} achieved H=1.0"
    if iteration >= max_refinements:
        return f"budget exhausted after {max_refinements} refinements"
    return None


ROOT_SOURCE = """def propose_action(observation: str) -> str:
    raise NotImplementedError("Root policy — replace me")

def is_legal_action(observation: str, action: str) -> bool:
    raise NotImplementedError("Root legality checker — replace me")
"""


def synthesize(
    adapter: EnvironmentAdapter,
    profile: Profile,
    refiner: RefinerProtocol,
    artifact_root: Path,
    seed: int = 42,
    refinements: int | None = None,
    execution_timeout: int = 10,
    max_source_size: int = 32768,
    model_id: str = "",
    environment_seed: int = 0,
    training_rollouts: int = 1,
) -> dict[str, Any]:
    """Run the full synthesis workflow. Returns summary dict."""
    now = datetime.now()
    run_id = now.strftime("%y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    store = ArtifactStore(root=artifact_root, run_id=run_id)
    rng = random.Random(seed)
    max_refinements = refinements if refinements is not None else profile.refinements
    policy_executor = PolicyExecutor(
        timeout=execution_timeout,
        max_source_size=max_source_size,
    )
    evaluator = RolloutEvaluator(
        adapter=adapter,
        executor=policy_executor,
    )
    episode_seeds = generate_episode_seeds(environment_seed, training_rollouts)
    assessor = CandidateAssessor(evaluator)

    try:
        adapter.create()
        # `seed` controls Thompson sampling only; this reset is an environment preflight.
        adapter.reset(seed=None)
    except Exception as e:
        raise RuntimeError(f"Environment preflight failed — cannot start synthesis: {e}") from e

    store.write_config(
        {
            "run_id": run_id,
            "profile": profile.value,
            "max_refinements": max_refinements,
            "thompson_seed": seed,
            "env_id": adapter.env_id,
            "execution_timeout": execution_timeout,
            "max_source_size": max_source_size,
            "model_id": model_id,
            "environment_seed": environment_seed,
            "training_rollouts": training_rollouts,
            "training_episode_seeds": episode_seeds,
        }
    )

    logger.info(
        "Starting synthesis run_id=%s profile=%s max_refinements=%s env=%s model=%s",
        run_id,
        profile.value,
        max_refinements,
        adapter.env_id,
        model_id,
    )

    root = Candidate(
        id=ROOT_ID,
        parent_id=None,
        source=ROOT_SOURCE,
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        termination_reason=None,
        failure_summary=None,
        iteration=0,
        expansion_count=0,
    )
    candidates: dict[str, Candidate] = {ROOT_ID: root}
    store.write_candidate(ROOT_ID, ROOT_SOURCE)

    stop_reason: str | None = None
    model_call_count = 0
    logical_refinement_count = 0

    def _evaluated_candidates() -> dict[str, Candidate]:
        """Return candidates eligible for selection and final ranking."""
        return {
            candidate_id: candidate
            for candidate_id, candidate in candidates.items()
            if _ranking_exclusion_reason(candidate_id, candidate) is None
        }

    for iteration in range(1, max_refinements + 1):
        stop_reason = should_stop(candidates, iteration - 1, max_refinements)
        if stop_reason:
            break

        pool = _evaluated_candidates()
        if not pool:
            if iteration == 1:
                pool = candidates
            else:
                stop_reason = "no evaluated candidates to select"
                break

        logger.info(
            "Iteration %d/%d — selecting from %d candidate(s)",
            iteration,
            max_refinements,
            len(pool),
        )
        parent_id = select_candidate(pool, rng)
        if parent_id is None:
            stop_reason = "no candidates to select"
            break

        parent = candidates[parent_id]
        logger.info(
            "Selected parent %s (H=%.3f, reward=%.3f, expansions=%d)",
            parent_id,
            parent.heuristic,
            parent.terminal_reward,
            parent.expansion_count,
        )
        parent.expansion_count += 1

        store.write_event(
            Event(
                iteration=iteration,
                event_type="select",
                candidate_id=parent_id,
                parent_id=parent.parent_id,
                metadata={"expansion_count": parent.expansion_count},
            )
        )

        child_id = f"{iteration:03d}"
        feedback = (
            build_assessment_feedback(parent.assessment) if parent.assessment is not None else []
        )
        descriptor = None
        if parent.termination_reason == TerminationReason.ILLEGAL_ACTION:
            descriptor = "Policy produced an illegal action"
        elif parent.termination_reason == TerminationReason.POLICY_REJECTED_ACTION:
            descriptor = "is_legal_action rejected the proposed action; refine propose_action only"
        elif parent.termination_reason == TerminationReason.LEGALITY_DISAGREEMENT:
            descriptor = (
                "is_legal_action accepted an action that the environment rejected; "
                "refine both functions"
            )
        elif parent.termination_reason == TerminationReason.STEP_LIMIT:
            descriptor = "Policy reached step limit without solving"
        elif parent.termination_reason in (
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        ):
            descriptor = "Policy execution failed at runtime"

        if descriptor:
            feedback.insert(0, descriptor)
        feedback = feedback[:5]

        logger.info(
            "Refining parent %s (iteration=%d)",
            parent_id,
            iteration,
        )
        refine_legal_action = parent.termination_reason != TerminationReason.POLICY_REJECTED_ACTION
        if parent.assessment is not None and should_refine_legal_action(parent.assessment):
            refine_legal_action = True
        refine_result = refiner.refine(
            env_name=adapter.env_id,
            rules=adapter.rules,
            action_format=adapter.action_format,
            parent_source=parent.source,
            parent_heuristic=parent.heuristic,
            parent_reward=parent.terminal_reward,
            parent_legal_actions=parent.legal_action_count,
            parent_status=(
                parent.termination_reason.value if parent.termination_reason else "unknown"
            ),
            feedback=feedback,
            refine_legal_action=refine_legal_action,
        )
        model_call_count = refiner.model_call_count
        logical_refinement_count = refiner.logical_refinement_count

        store.write_event(
            Event(
                iteration=iteration,
                event_type="refine",
                candidate_id=child_id,
                parent_id=parent_id,
                metadata={"success": refine_result.success},
            )
        )

        logger.info(
            "Refinement %s — candidate %s",
            "succeeded" if refine_result.success else "failed",
            child_id,
        )
        if not refine_result.success or not refine_result.source:
            assessment = failed_assessment(refine_result.error_details or "Refinement failed")
            child = Candidate(
                id=child_id,
                parent_id=parent_id,
                source=refine_result.source or "",
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.CONTRACT_FAILURE,
                failure_summary=assessment.failure_summary,
                last_observation=None,
                iteration=iteration,
                expansion_count=0,
                failure_count=assessment.failure_count,
                episode_count=0,
                assessment=assessment,
            )
            candidates[child_id] = child
            store.write_candidate(child_id, child.source)
            store.write_assessment(child_id, assessment)
            continue

        store.write_candidate(child_id, refine_result.source)
        assessment = assessor.assess(refine_result.source, episode_seeds)

        logger.info(
            "Evaluation: candidate %s H=%.3f reward=%.3f (%s)",
            child_id,
            assessment.heuristic,
            assessment.terminal_reward,
            assessment.termination_reason.value if assessment.termination_reason else "?",
        )

        child = Candidate(
            id=child_id,
            parent_id=parent_id,
            source=refine_result.source,
            heuristic=assessment.heuristic,
            terminal_reward=assessment.terminal_reward,
            legal_action_count=assessment.legal_action_count,
            termination_reason=assessment.termination_reason,
            failure_summary=assessment.failure_summary,
            last_observation=assessment.last_observation,
            iteration=iteration,
            expansion_count=0,
            failure_count=assessment.failure_count,
            episode_count=len(assessment.episodes),
            assessment=assessment,
        )
        candidates[child_id] = child
        store.write_assessment(child_id, assessment)

        store.write_event(
            Event(
                iteration=iteration,
                event_type="evaluate",
                candidate_id=child_id,
                parent_id=parent_id,
                metadata={
                    "heuristic": assessment.heuristic,
                    "terminal_reward": assessment.terminal_reward,
                    "legal_action_count": assessment.legal_action_count,
                    "failure_count": assessment.failure_count,
                    "episode_count": len(assessment.episodes),
                    "termination_reason": (
                        assessment.termination_reason.value
                        if assessment.termination_reason
                        else None
                    ),
                },
            )
        )

    if not stop_reason:
        stop_reason = should_stop(candidates, max_refinements, max_refinements) or "completed"

    evaluated_candidates = _evaluated_candidates()
    ordered_candidate_ids = rank_candidates(evaluated_candidates)
    best_id = ordered_candidate_ids[0] if ordered_candidate_ids else None

    logger.info("Stop reason: %s", stop_reason)
    if best_id:
        logger.info("Best candidate: %s (H=%.3f)", best_id, candidates[best_id].heuristic)

    tree_data: dict[str, Any] = {
        "candidates": {
            cid: {
                "id": c.id,
                "parent_id": c.parent_id,
                "heuristic": c.heuristic,
                "terminal_reward": c.terminal_reward,
                "legal_action_count": c.legal_action_count,
                "termination_reason": (
                    c.termination_reason.value if c.termination_reason else None
                ),
                "failure_summary": c.failure_summary,
                "iteration": c.iteration,
                "expansion_count": c.expansion_count,
                "failure_count": c.failure_count,
                "episode_count": c.episode_count,
                "ranking": _candidate_ranking_artifact(cid, c),
            }
            for cid, c in candidates.items()
        },
        "ranking": _ranking_artifact(evaluated_candidates, ordered_candidate_ids),
        "best_candidate_id": best_id,
    }
    store.write_tree(tree_data)

    summary = {
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "stop_reason": stop_reason,
        "best_candidate_id": best_id,
        "total_candidates": len(candidates),
        "iterations_used": len(candidates) - 1,
        "profile": profile.value,
        "model_call_count": model_call_count,
        "logical_refinement_count": logical_refinement_count,
    }
    store.write_synthesis_summary(summary)

    if best_id and candidates[best_id].source:
        store.write_best_policy(candidates[best_id].source)

    return summary
