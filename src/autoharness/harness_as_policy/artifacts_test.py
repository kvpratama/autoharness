"""Tests for the artifact store."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autoharness.harness_as_policy.artifacts import ArtifactStore
from autoharness.harness_as_policy.models import (
    Event,
    RolloutResult,
    StepResult,
    TerminationReason,
)


@pytest.fixture
def store() -> ArtifactStore:
    tmpdir = tempfile.mkdtemp()
    return ArtifactStore(root=Path(tmpdir), run_id="test-run-001")


def test_artifact_store_creates_directories(store: ArtifactStore) -> None:
    """Initialization creates expected directory structure."""
    assert (store.root / store.run_id).exists()
    assert (store.root / store.run_id / "candidates").exists()
    assert (store.root / store.run_id / "rollouts").exists()
    assert (store.root / store.run_id / "evaluation").exists()


def test_write_config_json(store: ArtifactStore) -> None:
    """write_config persists config dict as JSON."""
    config = {"model": "test", "profile": "smoke", "seed": 42}
    store.write_config(config)
    path = store.root / store.run_id / "config.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["model"] == "test"


def test_write_candidate_source(store: ArtifactStore) -> None:
    """write_candidate persists source to candidates/<id>.py."""
    store.write_candidate(candidate_id="005", source="def propose_action(...): pass")
    path = store.root / store.run_id / "candidates" / "005.py"
    assert path.exists()
    assert "propose_action" in path.read_text()


def test_write_rollout(store: ArtifactStore) -> None:
    """write_rollout persists rollout result as JSON."""
    result = RolloutResult(
        steps=[
            StepResult(
                observation="obs",
                action="[A C]",
                is_legal=True,
                reward=0.0,
                terminated=False,
                feedback="",
            )
        ],
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=1,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
    )
    store.write_rollout(candidate_id="005", result=result)
    path = store.root / store.run_id / "rollouts" / "005.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["heuristic"] == 0.5


def test_write_event(store: ArtifactStore) -> None:
    """write_event appends to events.jsonl."""
    event = Event(
        iteration=1,
        event_type="refine",
        candidate_id="001",
        parent_id="000",
        metadata={},
    )
    store.write_event(event)
    path = store.root / store.run_id / "events.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["iteration"] == 1


def test_write_tree(store: ArtifactStore) -> None:
    """write_tree persists tree data as JSON."""
    tree = {"candidates": {"000": {"heuristic": 0.0}}, "best": "001"}
    store.write_tree(tree)
    path = store.root / store.run_id / "tree.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["best"] == "001"


def test_write_best_policy(store: ArtifactStore) -> None:
    """write_best_policy persists best.py."""
    source = "def propose_action(observation: str) -> str:\n    return '[A C]'"
    store.write_best_policy(source=source)
    path = store.root / store.run_id / "best.py"
    assert path.exists()
    assert "propose_action" in path.read_text()


def test_write_synthesis_summary(store: ArtifactStore) -> None:
    """write_synthesis_summary persists summary JSON."""
    summary = {"best_candidate": "003", "iterations": 5, "stop_reason": "success"}
    store.write_synthesis_summary(summary)
    path = store.root / store.run_id / "synthesis-summary.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["stop_reason"] == "success"


def test_write_evaluation(store: ArtifactStore) -> None:
    """write_evaluation persists evaluation JSON under evaluation/."""
    eval_data = {"solved": True, "max_disks": 6}
    store.write_evaluation(name="generated-policy", data=eval_data)
    path = store.root / store.run_id / "evaluation" / "generated-policy.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["solved"] is True


def test_load_best_policy(store: ArtifactStore) -> None:
    """load_best_policy reads back the best.py source."""
    source = "def propose_action(observation: str) -> str:\n    return '[A C]'"
    store.write_best_policy(source=source)
    loaded = store.load_best_policy()
    assert loaded == source


def test_load_config(store: ArtifactStore) -> None:
    """load_config reads back config.json."""
    config = {"model": "test", "profile": "smoke"}
    store.write_config(config)
    loaded = store.load_config()
    assert loaded is not None
    assert loaded["model"] == "test"
