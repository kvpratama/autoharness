"""Atomic artifact persistence for synthesis runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoharness.harness_as_policy.models import Event, RolloutResult


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
        (self._run_dir / "evaluation").mkdir(exist_ok=True)

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(path)

    def write_config(self, config: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "config.json", config)

    def write_tree(self, tree: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "tree.json", tree)

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

    def write_rollout(self, candidate_id: str, result: RolloutResult) -> None:
        data = {
            "heuristic": result.heuristic,
            "terminal_reward": result.terminal_reward,
            "legal_action_count": result.legal_action_count,
            "termination_reason": (
                result.termination_reason.value if result.termination_reason else None
            ),
            "failure_summary": result.failure_summary,
            "steps": [
                {
                    "observation": s.observation,
                    "action": s.action,
                    "is_legal": s.is_legal,
                    "reward": s.reward,
                    "terminated": s.terminated,
                    "feedback": s.feedback,
                }
                for s in result.steps
            ],
        }
        self._write_json(self._run_dir / "rollouts" / f"{candidate_id}.json", data)

    def write_best_policy(self, source: str) -> None:
        path = self._run_dir / "best.py"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source)
        tmp.rename(path)

    def write_synthesis_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "synthesis-summary.json", summary)

    def write_evaluation(self, name: str, data: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "evaluation" / f"{name}.json", data)

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
