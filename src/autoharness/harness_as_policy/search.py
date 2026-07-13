"""LangGraph search workflow with REx Thompson selection."""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any

from src.autoharness.harness_as_policy.artifacts import ArtifactStore
from src.autoharness.harness_as_policy.executor import PolicyExecutor
from src.autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    RolloutResult,
    TerminationReason,
)
from src.autoharness.harness_as_policy.rollout import RolloutEvaluator


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
    """Select a candidate using Thompson sampling (largest draw)."""
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


def find_best_candidate(
    candidates: dict[str, Candidate],
) -> str | None:
    """Find the best candidate using lexicographic ranking."""
    if not candidates:
        return None
    best_id: str | None = None
    best_key: CandidateRankKey | None = None
    for cid, cand in candidates.items():
        key = CandidateRankKey.from_candidate(cand)
        if best_key is None or key > best_key:
            best_key = key
            best_id = cid
    return best_id


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
"""


def synthesize(
    adapter: Any,
    profile: Profile,
    refiner: Any,
    artifact_root: Path,
    seed: int = 42,
    refinements: int | None = None,
) -> dict[str, Any]:
    """Run the full synthesis workflow. Returns summary dict."""
    run_id = uuid.uuid4().hex[:12]
    store = ArtifactStore(root=artifact_root, run_id=run_id)
    rng = random.Random(seed)
    max_refinements = refinements if refinements is not None else profile.refinements
    policy_executor = PolicyExecutor()
    evaluator = RolloutEvaluator(
        adapter=adapter,
        executor=policy_executor,
    )

    store.write_config(
        {
            "run_id": run_id,
            "profile": profile.value,
            "max_refinements": max_refinements,
            "thompson_seed": seed,
            "env_id": adapter.env_id,
            "model_call_count": 0,
        }
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

    best_id: str | None = None
    stop_reason: str | None = None
    model_call_count = 0
    logical_refinement_count = 0

    for iteration in range(1, max_refinements + 1):
        stop_reason = should_stop(candidates, iteration - 1, max_refinements)
        if stop_reason:
            break

        parent_id = select_candidate(candidates, rng)
        if parent_id is None:
            stop_reason = "no candidates to select"
            break

        parent = candidates[parent_id]
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
        elif parent.termination_reason == TerminationReason.STEP_LIMIT:
            feedback.append("Policy reached step limit without solving")
        elif parent.termination_reason in (
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        ):
            feedback.append("Policy execution failed at runtime")

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

        child = Candidate(
            id=child_id,
            parent_id=parent_id,
            source=refine_result.source,
            heuristic=rollout_result.heuristic,
            terminal_reward=rollout_result.terminal_reward,
            legal_action_count=rollout_result.legal_action_count,
            termination_reason=rollout_result.termination_reason,
            failure_summary=rollout_result.failure_summary,
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

        current_best = find_best_candidate(candidates)
        if current_best:
            best_id = current_best

    if not stop_reason:
        stop_reason = should_stop(candidates, max_refinements, max_refinements) or "completed"

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
            }
            for cid, c in candidates.items()
        },
        "best_candidate_id": best_id,
    }
    store.write_tree(tree_data)

    summary = {
        "run_id": run_id,
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
