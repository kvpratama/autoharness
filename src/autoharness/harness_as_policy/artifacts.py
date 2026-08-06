"""Atomic artifact persistence for synthesis runs."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from autoharness.harness_as_policy.models import (
    CandidateAssessment,
    EpisodeResult,
    Event,
    RefinementTrace,
)

logger = logging.getLogger(__name__)


class CandidateRecord(TypedDict, total=False):
    """Schema for candidate record dictionaries in synthesis trees."""

    id: str
    parent_id: str | None
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    termination_reason: str | None
    failure_summary: str | None
    iteration: int
    expansion_count: int
    failure_count: int
    episode_count: int
    rollout_eligible: bool
    ranking: Any


class SynthesisTree(TypedDict, total=False):
    """Schema for synthesis tree dictionaries."""

    candidates: Mapping[str, CandidateRecord]
    ranking: Any
    best_candidate_id: str | None


def _candidate_sort_key(
    candidate_id: str, candidates: Mapping[str, CandidateRecord]
) -> tuple[int, str]:
    return int(candidates[candidate_id]["iteration"]), candidate_id


def _candidate_status(
    candidate: CandidateRecord, candidate_id: str, best_candidate_id: str | None
) -> str:
    if candidate_id == best_candidate_id:
        return "BEST"
    if candidate["parent_id"] is None:
        return "ROOT"
    if not candidate["rollout_eligible"]:
        return "FAIL"
    if candidate["failure_count"] > 0:
        return "PARTIAL"
    return "OK"


def _candidate_diagnostic(candidate: CandidateRecord, status: str) -> str | None:
    if status not in {"FAIL", "PARTIAL"}:
        return None
    value = candidate["failure_summary"] or candidate["termination_reason"]
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if len(normalized) > 60:
        return normalized[:57] + "..."
    return normalized


def _format_candidate(
    candidate: CandidateRecord, candidate_id: str, best_candidate_id: str | None
) -> str:
    status = _candidate_status(candidate, candidate_id, best_candidate_id)
    diagnostic = _candidate_diagnostic(candidate, status)
    status_text = f"{status}: {diagnostic}" if diagnostic else status
    return (
        f"[{candidate_id} H={candidate['heuristic']:.2f} "
        f"R={candidate['terminal_reward']:.2f} {status_text}]"
    )


def render_tree_text(tree: SynthesisTree) -> str:
    """Render one synthesis tree artifact as a compact text hierarchy.

    Args:
        tree: The same in-memory tree dictionary serialized to ``tree.json``.

    Returns:
        A deterministic tree diagram ending with a newline.
    """
    candidates: Mapping[str, CandidateRecord] = tree["candidates"]
    best_candidate_id: str | None = tree.get("best_candidate_id")
    children: dict[str, list[str]] = {}
    roots: list[str] = []

    for candidate_id, candidate in candidates.items():
        parent_id = candidate["parent_id"]
        if parent_id is None or parent_id not in candidates:
            roots.append(candidate_id)
        else:
            children.setdefault(parent_id, []).append(candidate_id)

    roots.sort(key=lambda candidate_id: _candidate_sort_key(candidate_id, candidates))
    for child_ids in children.values():
        child_ids.sort(key=lambda candidate_id: _candidate_sort_key(candidate_id, candidates))

    lines = ["Synthesis tree", ""]

    def append_subtree(candidate_id: str, prefix: str, connector: str) -> None:
        candidate = candidates[candidate_id]
        lines.append(
            f"{prefix}{connector}{_format_candidate(candidate, candidate_id, best_candidate_id)}"
        )
        child_ids = children.get(candidate_id, [])
        if not connector:
            child_prefix = prefix
        else:
            child_prefix = prefix + ("    " if connector == "`-- " else "|   ")
        for index, child_id in enumerate(child_ids):
            child_connector = "`-- " if index == len(child_ids) - 1 else "|-- "
            append_subtree(child_id, child_prefix, child_connector)

    for index, root_id in enumerate(roots):
        if index > 0:
            lines.append("")
        append_subtree(root_id, "", "")

    return "\n".join(lines) + "\n"


class ArtifactStore:
    """Persists and loads synthesis run artifacts."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root
        self._run_id = run_id
        self._run_dir = root / run_id
        self._init_directories()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _init_directories(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "candidates").mkdir(exist_ok=True)
        (self._run_dir / "rollouts").mkdir(exist_ok=True)
        (self._run_dir / "refinements").mkdir(exist_ok=True)
        (self._run_dir / "evaluation").mkdir(exist_ok=True)

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(path)

    def write_config(self, config: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "config.json", config)

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content)
        tmp.replace(path)

    def write_tree(self, tree: SynthesisTree) -> None:
        """Persist canonical synthesis tree JSON and best-effort text diagram.

        Args:
            tree: The synthesis tree structure to record.
        """
        self._write_json(self._run_dir / "tree.json", tree)
        try:
            rendered = render_tree_text(tree)
            self._write_text(self._run_dir / "tree.txt", rendered)
        except Exception:
            logger.warning("Failed to render tree text artifact", exc_info=True)

    def write_event(self, event: Event) -> None:
        path = self._run_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        events = self.load_events()
        events.append(
            {
                "iteration": event.iteration,
                "event_type": event.event_type,
                "candidate_id": event.candidate_id,
                "parent_id": event.parent_id,
                "metadata": event.metadata,
            }
        )
        tmp = path.with_suffix(".tmp")
        jsonl_content = "".join(json.dumps(e, default=str) + "\n" for e in events)
        tmp.write_text(jsonl_content)
        tmp.replace(path)

    def load_events(self) -> list[dict[str, Any]]:
        """Loads events from events.jsonl, ignoring malformed/interrupted lines."""
        path = self._run_dir / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        content = path.read_text()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    def write_candidate(self, candidate_id: str, source: str) -> None:
        path = self._run_dir / "candidates" / f"{candidate_id}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source)
        tmp.rename(path)

    def write_assessment(self, candidate_id: str, assessment: CandidateAssessment) -> None:
        """Persist a version-four aggregate assessment and all episode details."""
        data = {
            "schema_version": 4,
            "aggregate": {
                "heuristic": assessment.heuristic,
                "terminal_reward": assessment.terminal_reward,
                "legal_action_count": assessment.legal_action_count,
                "failure_count": assessment.failure_count,
                "termination_counts": {
                    reason.value: count
                    for reason, count in sorted(
                        assessment.termination_counts.items(), key=lambda item: item[0].value
                    )
                },
                "termination_reason": assessment.termination_reason.value
                if assessment.termination_reason is not None
                else None,
                "failure_summary": assessment.failure_summary,
                "last_observation": assessment.last_observation,
            },
            "representative_episode_index": assessment.representative_episode_index,
            "episodes": [self._serialize_episode(episode) for episode in assessment.episodes],
        }
        self._write_json(self._run_dir / "rollouts" / f"{candidate_id}.json", data)

    @staticmethod
    def _serialize_episode(episode: EpisodeResult) -> dict[str, Any]:
        result = episode.rollout
        return {
            "seed": episode.seed,
            "heuristic": result.heuristic,
            "terminal_reward": result.terminal_reward,
            "legal_action_count": result.legal_action_count,
            "termination_reason": result.termination_reason.value,
            "failure_summary": result.failure_summary,
            "last_observation": result.last_observation,
            "steps": [
                {
                    "observation": step.observation,
                    "action": step.action,
                    "is_legal": step.is_legal,
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "feedback": step.feedback,
                }
                for step in result.steps
            ],
            "attempts": [
                {
                    "observation": attempt.observation,
                    "action": attempt.action,
                    "policy_legal": attempt.policy_legal,
                    "environment_legal": attempt.environment_legal,
                    "resulting_observation": attempt.resulting_observation,
                    "reward": attempt.reward,
                    "terminated": attempt.terminated,
                    "feedback": attempt.feedback,
                    "error_phase": attempt.error_phase.value if attempt.error_phase else None,
                    "policy_seed": attempt.policy_seed,
                }
                for attempt in result.attempts
            ],
        }

    def write_refinement(
        self,
        iteration: int,
        parent_id: str,
        refine_legal_action: bool,
        trace: RefinementTrace,
    ) -> None:
        """Persist one complete logical-refinement audit."""
        data = {
            "iteration": iteration,
            "parent_id": parent_id,
            "refine_legal_action": refine_legal_action,
            "prompt": trace.prompt,
            "invocations": [
                {
                    "content": invocation.content,
                    "normalized_text": invocation.normalized_text,
                    "error_type": invocation.error_type,
                    "error_message": invocation.error_message,
                }
                for invocation in trace.invocations
            ],
            "extracted_source": trace.extracted_source,
            "outcome": trace.outcome,
            "error_details": trace.error_details,
            "generation_succeeded": trace.generation_succeeded,
            "contract_valid": trace.contract_valid,
        }
        path = self._run_dir / "refinements" / f"{iteration:03d}.json"
        self._write_json(path, data)

    def write_best_policy(self, source: str) -> None:
        path = self._run_dir / "best.py"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source)
        tmp.rename(path)

    def write_synthesis_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "synthesis-summary.json", summary)

    def write_evaluation(self, name: str, data: Mapping[str, object]) -> None:
        self._write_json(self._run_dir / "evaluation" / f"{name}.json", data)

    def load_evaluation(self, name: str) -> dict[str, object] | None:
        """Load one named evaluation artifact when it exists."""
        path = self._run_dir / "evaluation" / f"{name}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Evaluation artifact {name!r} must contain a JSON object")
        return data

    def load_best_policy(self) -> str | None:
        path = self._run_dir / "best.py"
        if path.exists():
            return path.read_text()
        return None

    def load_config(self) -> dict[str, Any] | None:
        path = self._run_dir / "config.json"
        if path.exists():
            return json.loads(path.read_text())
        return None
