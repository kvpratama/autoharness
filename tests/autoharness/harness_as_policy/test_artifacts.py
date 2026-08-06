"""Tests for the artifact store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoharness.environments.models import StepResult
from autoharness.harness_as_policy.artifacts import ArtifactStore, SynthesisTree, render_tree_text
from autoharness.harness_as_policy.models import (
    ActionAttempt,
    CandidateAssessment,
    EpisodeResult,
    Event,
    ProviderInvocation,
    RefinementOutcome,
    RefinementTrace,
    RolloutResult,
    TerminationReason,
)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(root=tmp_path, run_id="test-run-001")


def test_artifact_store_creates_directories(store: ArtifactStore) -> None:
    """Initialization creates expected directory structure."""
    assert (store.root / store.run_id).exists()
    assert (store.root / store.run_id / "candidates").exists()
    assert (store.root / store.run_id / "rollouts").exists()
    assert (store.root / store.run_id / "refinements").exists()
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
    source = "def propose_action(...): pass\ndef is_legal_action(...): pass"
    store.write_candidate(candidate_id="005", source=source)
    path = store.root / store.run_id / "candidates" / "005.py"
    assert path.exists()
    assert "propose_action" in path.read_text()


def test_write_assessment_preserves_aggregate_and_episodes(store: ArtifactStore) -> None:
    """write_assessment persists version-four aggregate and episode data."""
    result = RolloutResult(
        steps=[StepResult("obs", "[A C]", True, 0.0, False, "")],
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=1,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        last_observation="last",
        attempts=[
            ActionAttempt(
                observation="before",
                action="[A C]",
                policy_legal=True,
                environment_legal=True,
                resulting_observation="obs",
                reward=0.0,
                terminated=False,
                feedback="",
                error_phase=None,
                policy_seed=11891538334161795807,
            )
        ],
    )
    assessment = CandidateAssessment(
        episodes=[
            EpisodeResult(
                11,
                RolloutResult([], 1.0, 1.0, 3, TerminationReason.ENVIRONMENT_TERMINATION, None),
            ),
            EpisodeResult(22, result),
        ],
        heuristic=0.75,
        terminal_reward=0.5,
        legal_action_count=4,
        failure_count=0,
        termination_counts={
            TerminationReason.ENVIRONMENT_TERMINATION: 1,
            TerminationReason.STEP_LIMIT: 1,
        },
        representative_episode_index=1,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
        last_observation="last",
    )
    store.write_assessment(candidate_id="005", assessment=assessment)
    path = store.root / store.run_id / "rollouts" / "005.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema_version"] == 4
    assert data["aggregate"]["heuristic"] == 0.75
    assert data["aggregate"]["termination_reason"] == "step_limit"
    assert data["aggregate"]["failure_summary"] is None
    assert data["aggregate"]["last_observation"] == "last"
    assert data["representative_episode_index"] == 1
    assert [episode["seed"] for episode in data["episodes"]] == [11, 22]
    assert data["episodes"][1]["steps"][0]["observation"] == "obs"
    assert data["episodes"][1]["attempts"] == [
        {
            "observation": "before",
            "action": "[A C]",
            "policy_legal": True,
            "environment_legal": True,
            "resulting_observation": "obs",
            "reward": 0.0,
            "terminated": False,
            "feedback": "",
            "error_phase": None,
            "policy_seed": 11891538334161795807,
        }
    ]
    assert "heuristic" not in data


def test_write_refinement_preserves_prompt_provider_attempts_and_source(
    store: ArtifactStore,
) -> None:
    trace = RefinementTrace(
        prompt="exact prompt",
        invocations=[
            ProviderInvocation(error_type="ConnectionError", error_message="offline"),
            ProviderInvocation(
                content={"type": "text", "text": "raw response"},
                normalized_text="raw response",
            ),
        ],
        extracted_source="def propose_action(): pass",
        outcome=RefinementOutcome.SUCCESS,
        generation_succeeded=True,
        contract_valid=True,
    )

    store.write_refinement(1, "000", True, trace)

    path = store.run_dir / "refinements" / "001.json"
    data = json.loads(path.read_text())
    assert data["iteration"] == 1
    assert data["parent_id"] == "000"
    assert data["refine_legal_action"] is True
    assert data["prompt"] == "exact prompt"
    assert data["invocations"][0]["error_type"] == "ConnectionError"
    assert data["invocations"][1]["content"]["text"] == "raw response"
    assert data["extracted_source"] == "def propose_action(): pass"
    assert data["outcome"] == "success"
    assert data["generation_succeeded"] is True
    assert data["contract_valid"] is True


def test_write_failed_assessment_has_no_episodes(store: ArtifactStore) -> None:
    """Failed refinements retain a contract-failure aggregate without episodes."""
    assessment = CandidateAssessment(
        episodes=[],
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        failure_count=1,
        termination_counts={TerminationReason.CONTRACT_FAILURE: 1},
        representative_episode_index=None,
        termination_reason=TerminationReason.CONTRACT_FAILURE,
        failure_summary="failed",
        last_observation=None,
    )
    store.write_assessment("006", assessment)
    data = json.loads((store.run_dir / "rollouts" / "006.json").read_text())
    assert data["episodes"] == []
    assert data["representative_episode_index"] is None
    assert data["aggregate"]["termination_counts"] == {"contract_failure": 1}
    assert data["aggregate"]["termination_reason"] == "contract_failure"
    assert data["aggregate"]["failure_summary"] == "failed"
    assert data["aggregate"]["last_observation"] is None


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


def test_write_event_replaces_existing_event_file(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_event atomically replaces an existing event file."""
    path = store.run_dir / "events.jsonl"
    path.write_text('{"iteration": 1}\n')
    replaced_paths: list[Path] = []
    original_replace = Path.replace

    def recording_replace(source: Path, target: Path) -> Path:
        replaced_paths.append(target)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)

    store.write_event(
        Event(
            iteration=2,
            event_type="refine",
            candidate_id="002",
            parent_id="001",
            metadata={},
        )
    )

    assert replaced_paths == [path]
    assert [event["iteration"] for event in store.load_events()] == [1, 2]


def test_load_events_propagates_unexpected_errors(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_events does not disguise unrelated failures as empty history."""
    path = store.run_dir / "events.jsonl"
    path.write_text('{"iteration": 1}\n')

    def raise_unexpected_error(*args: object, **kwargs: object) -> str:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(Path, "read_text", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        store.load_events()


def test_load_events_propagates_file_read_errors(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_events exposes event-history read failures to callers."""
    path = store.run_dir / "events.jsonl"
    path.write_text('{"iteration": 1}\n')

    def raise_read_error(*args: object, **kwargs: object) -> str:
        raise PermissionError("event history is unreadable")

    monkeypatch.setattr(Path, "read_text", raise_read_error)

    with pytest.raises(PermissionError, match="event history is unreadable"):
        store.load_events()


def test_render_tree_text_shows_hierarchy_statuses_and_iteration_order() -> None:
    tree: SynthesisTree = {
        "candidates": {
            "003": {
                "id": "003",
                "parent_id": "001",
                "heuristic": 0.5,
                "terminal_reward": 0.25,
                "iteration": 3,
                "rollout_eligible": True,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": "step_limit",
            },
            "002": {
                "id": "002",
                "parent_id": "000",
                "heuristic": 1.0,
                "terminal_reward": 1.0,
                "iteration": 2,
                "rollout_eligible": True,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": "environment_termination",
            },
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 0,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
            "001": {
                "id": "001",
                "parent_id": "000",
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 1,
                "rollout_eligible": False,
                "failure_count": 1,
                "failure_summary": "pop from empty list",
                "termination_reason": "execution_failure",
            },
        },
        "best_candidate_id": "002",
    }

    assert render_tree_text(tree) == (
        "Synthesis tree\n"
        "\n"
        "[000 H=0.00 R=0.00 ROOT]\n"
        "|-- [001 H=0.00 R=0.00 FAIL: pop from empty list]\n"
        "|   `-- [003 H=0.50 R=0.25 OK]\n"
        "`-- [002 H=1.00 R=1.00 BEST]\n"
    )


def test_render_tree_text_prioritizes_best_over_root_for_selected_root_candidate() -> None:
    tree: SynthesisTree = {
        "candidates": {
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 1.0,
                "terminal_reward": 1.0,
                "iteration": 0,
                "rollout_eligible": True,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
        },
        "best_candidate_id": "000",
    }

    assert render_tree_text(tree) == ("Synthesis tree\n\n[000 H=1.00 R=1.00 BEST]\n")


def test_render_tree_text_normalizes_truncates_and_falls_back_for_diagnostics() -> None:
    long_summary = "x" * 61
    tree: SynthesisTree = {
        "candidates": {
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 0,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
            "001": {
                "id": "001",
                "parent_id": "000",
                "heuristic": 0.75,
                "terminal_reward": 0.5,
                "iteration": 1,
                "rollout_eligible": True,
                "failure_count": 1,
                "failure_summary": " first\n  second\tthird ",
                "termination_reason": "step_limit",
            },
            "002": {
                "id": "002",
                "parent_id": "000",
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 2,
                "rollout_eligible": False,
                "failure_count": 1,
                "failure_summary": None,
                "termination_reason": "contract_failure",
            },
            "003": {
                "id": "003",
                "parent_id": "000",
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 3,
                "rollout_eligible": False,
                "failure_count": 1,
                "failure_summary": long_summary,
                "termination_reason": "execution_failure",
            },
        },
        "best_candidate_id": None,
    }

    rendered = render_tree_text(tree)

    assert "[001 H=0.75 R=0.50 PARTIAL: first second third]" in rendered
    assert "[002 H=0.00 R=0.00 FAIL: contract_failure]" in rendered
    assert f"[003 H=0.00 R=0.00 FAIL: {'x' * 57}...]" in rendered
    assert long_summary not in rendered


def test_render_tree_text_keeps_roots_orphans_and_descendants() -> None:
    tree: SynthesisTree = {
        "candidates": {
            "011": {
                "id": "011",
                "parent_id": "010",
                "heuristic": 0.5,
                "terminal_reward": 0.25,
                "iteration": 11,
                "rollout_eligible": True,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": "step_limit",
            },
            "010": {
                "id": "010",
                "parent_id": "missing",
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 10,
                "rollout_eligible": False,
                "failure_count": 1,
                "failure_summary": "orphaned parent",
                "termination_reason": "execution_failure",
            },
            "005": {
                "id": "005",
                "parent_id": None,
                "heuristic": 0.2,
                "terminal_reward": 0.1,
                "iteration": 5,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 0,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
        },
        "best_candidate_id": None,
    }

    assert render_tree_text(tree) == (
        "Synthesis tree\n"
        "\n"
        "[000 H=0.00 R=0.00 ROOT]\n"
        "\n"
        "[005 H=0.20 R=0.10 ROOT]\n"
        "\n"
        "[010 H=0.00 R=0.00 FAIL: orphaned parent]\n"
        "`-- [011 H=0.50 R=0.25 OK]\n"
    )


def test_render_tree_text_renders_root_only() -> None:
    tree: SynthesisTree = {
        "candidates": {
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 0,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            }
        },
        "best_candidate_id": None,
    }

    assert render_tree_text(tree) == "Synthesis tree\n\n[000 H=0.00 R=0.00 ROOT]\n"


def test_write_tree_persists_unchanged_json_and_derived_text(store: ArtifactStore) -> None:
    tree: SynthesisTree = {
        "candidates": {
            "000": {
                "id": "000",
                "parent_id": None,
                "heuristic": 0.0,
                "terminal_reward": 0.0,
                "iteration": 0,
                "rollout_eligible": False,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": None,
            },
            "001": {
                "id": "001",
                "parent_id": "000",
                "heuristic": 1.0,
                "terminal_reward": 1.0,
                "iteration": 1,
                "rollout_eligible": True,
                "failure_count": 0,
                "failure_summary": None,
                "termination_reason": "environment_termination",
            },
        },
        "ranking": {"ordered_candidate_ids": ["001"]},
        "best_candidate_id": "001",
    }

    store.write_tree(tree)

    json_path = store.run_dir / "tree.json"
    text_path = store.run_dir / "tree.txt"
    assert json.loads(json_path.read_text()) == tree
    assert text_path.read_text() == (
        "Synthesis tree\n\n[000 H=0.00 R=0.00 ROOT]\n`-- [001 H=1.00 R=1.00 BEST]\n"
    )


def test_write_tree_persists_json_when_text_rendering_fails(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tree: SynthesisTree = {
        "candidates": {},
        "best_candidate_id": None,
    }

    def failing_render_tree_text(t: SynthesisTree) -> str:
        raise RecursionError("maximum recursion depth exceeded in tree rendering")

    monkeypatch.setattr(
        "autoharness.harness_as_policy.artifacts.render_tree_text", failing_render_tree_text
    )

    with caplog.at_level("WARNING"):
        store.write_tree(tree)

    json_path = store.run_dir / "tree.json"
    text_path = store.run_dir / "tree.txt"
    assert json_path.exists()
    assert json.loads(json_path.read_text()) == tree
    assert not text_path.exists()
    assert "Failed to render tree text artifact" in caplog.text


def test_write_best_policy(store: ArtifactStore) -> None:
    """write_best_policy persists best.py."""
    source = (
        "def propose_action(board: str) -> str:\n    return '[A C]'\n\n"
        "def is_legal_action(board: str, action: str) -> bool:\n    return True"
    )
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


def test_load_evaluation_round_trips_named_artifact(store: ArtifactStore) -> None:
    data = {"schema_version": 1, "name": "paper-1p"}
    store.write_evaluation("protocol", data)
    assert store.load_evaluation("protocol") == data
    assert store.load_evaluation("missing") is None


def test_load_evaluation_rejects_non_object_json(store: ArtifactStore) -> None:
    path = store.run_dir / "evaluation" / "invalid.json"
    path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(ValueError, match="must contain a JSON object"):
        store.load_evaluation("invalid")


def test_load_best_policy(store: ArtifactStore) -> None:
    """load_best_policy reads back the best.py source."""
    source = (
        "def propose_action(board: str) -> str:\n    return '[A C]'\n\n"
        "def is_legal_action(board: str, action: str) -> bool:\n    return True"
    )
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


def test_load_events_and_write_event_recovery(store: ArtifactStore) -> None:
    """load_events and write_event handle an interrupted trailing event correctly."""
    # Write a valid event
    ev1 = Event(
        iteration=1,
        event_type="select",
        candidate_id="001",
        parent_id="000",
        metadata={},
    )
    store.write_event(ev1)

    # Manually append an interrupted/partially written JSON record to the file
    path = store.run_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"iteration": 2, "event_type": "refine"\n')  # Incomplete JSON (no closing brace)

    # Verify load_events filters out/ignores the malformed trailing line
    events = store.load_events()
    assert len(events) == 1
    assert events[0]["iteration"] == 1

    # Verify writing a new event recovers and keeps only valid events
    ev3 = Event(
        iteration=3,
        event_type="rollout",
        candidate_id="003",
        parent_id="001",
        metadata={},
    )
    store.write_event(ev3)

    # Reload and confirm only the two valid events are persisted, and malformed is gone
    final_events = store.load_events()
    assert len(final_events) == 2
    assert final_events[0]["iteration"] == 1
    assert final_events[1]["iteration"] == 3
