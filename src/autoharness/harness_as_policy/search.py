"""LangGraph search workflow with REx Thompson selection."""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from autoharness.harness_as_policy.artifacts import ArtifactStore
from autoharness.harness_as_policy.environment import EnvironmentAdapter
from autoharness.harness_as_policy.executor import PolicyExecutor
from autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    RolloutResult,
    TerminationReason,
)
from autoharness.harness_as_policy.refiner import RefinerProtocol
from autoharness.harness_as_policy.rollout import RolloutEvaluator

logger = logging.getLogger(__name__)

RANKING_POLICY: tuple[tuple[str, str], ...] = (
    ("heuristic", "descending"),
    ("reward", "descending"),
    ("legal_actions", "descending"),
    ("failures", "ascending"),
    ("iteration", "ascending"),
)


def _ranking_exclusion_reason(candidate_id: str, candidate: Candidate) -> str | None:
    if candidate_id == "000":
        return "synthetic_root"
    if not candidate.source.strip():
        return "empty_source"
    return None


def _ranking_components(candidate: Candidate) -> dict[str, float | int]:
    key = CandidateRankKey.from_candidate(candidate)
    return {
        "heuristic": key.heuristic,
        "reward": key.reward,
        "legal_actions": key.legal_actions,
        "failures": key.failures,
        "iteration": key.iteration,
    }


def _candidate_ranking_artifact(candidate_id: str, candidate: Candidate) -> dict[str, Any]:
    exclusion_reason = _ranking_exclusion_reason(candidate_id, candidate)
    return {
        "eligible": exclusion_reason is None,
        "exclusion_reason": exclusion_reason,
        "components": _ranking_components(candidate) if exclusion_reason is None else None,
    }


def _winner_explanation(
    candidates: dict[str, Candidate],
    ordered_candidate_ids: list[str],
) -> dict[str, Any] | None:
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
    tied_components: list[str] = []
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
) -> dict[str, Any]:
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
        if cand.heuristic >= 1.0:
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
        id="000",
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
    candidates: dict[str, Candidate] = {"000": root}
    store.write_candidate("000", ROOT_SOURCE)

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
        feedback: list[str] = []
        if parent.failure_summary:
            feedback.append(parent.failure_summary)
        if parent.termination_reason == TerminationReason.ILLEGAL_ACTION:
            feedback.append("Policy produced an illegal action")
        elif parent.termination_reason == TerminationReason.POLICY_REJECTED_ACTION:
            feedback.append(
                "is_legal_action rejected the proposed action; refine propose_action only"
            )
        elif parent.termination_reason == TerminationReason.LEGALITY_DISAGREEMENT:
            feedback.append(
                "is_legal_action accepted an action that the environment rejected; refine both "
                "functions"
            )
        elif parent.termination_reason == TerminationReason.STEP_LIMIT:
            feedback.append("Policy reached step limit without solving")
        elif parent.termination_reason in (
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        ):
            feedback.append("Policy execution failed at runtime")

        if parent.last_observation:
            feedback.append(f"Last observation before termination: {parent.last_observation}")

        logger.info(
            "Refining parent %s (iteration=%d)",
            parent_id,
            iteration,
        )
        refine_legal_action = parent.termination_reason != TerminationReason.POLICY_REJECTED_ACTION
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
            child = Candidate(
                id=child_id,
                parent_id=parent_id,
                source=refine_result.source or "",
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.CONTRACT_FAILURE,
                failure_summary=(refine_result.error_details or "Refinement failed"),
                last_observation=None,
                iteration=iteration,
                expansion_count=0,
            )
            candidates[child_id] = child
            store.write_candidate(child_id, child.source)
            rollout_result = RolloutResult(
                steps=[],
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.CONTRACT_FAILURE,
                failure_summary=child.failure_summary,
            )
            store.write_rollout(child_id, rollout_result)
            continue

        store.write_candidate(child_id, refine_result.source)
        rollout_result = evaluator.evaluate(source=refine_result.source)

        logger.info(
            "Evaluation: candidate %s H=%.3f reward=%.3f (%s)",
            child_id,
            rollout_result.heuristic,
            rollout_result.terminal_reward,
            rollout_result.termination_reason.value if rollout_result.termination_reason else "?",
        )

        child = Candidate(
            id=child_id,
            parent_id=parent_id,
            source=refine_result.source,
            heuristic=rollout_result.heuristic,
            terminal_reward=rollout_result.terminal_reward,
            legal_action_count=rollout_result.legal_action_count,
            termination_reason=rollout_result.termination_reason,
            failure_summary=rollout_result.failure_summary,
            last_observation=rollout_result.last_observation,
            iteration=iteration,
            expansion_count=0,
        )
        candidates[child_id] = child
        store.write_rollout(child_id, rollout_result)

        store.write_event(
            Event(
                iteration=iteration,
                event_type="evaluate",
                candidate_id=child_id,
                parent_id=parent_id,
                metadata={
                    "heuristic": rollout_result.heuristic,
                    "terminal_reward": rollout_result.terminal_reward,
                    "legal_action_count": rollout_result.legal_action_count,
                    "termination_reason": (
                        rollout_result.termination_reason.value
                        if rollout_result.termination_reason
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
