"""Tests for the CLI."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autoharness.cli import evaluate_baseline_cmd, evaluate_cmd, main, synthesize_cmd
from autoharness.harness_as_policy.evaluation import EvaluationResult
from autoharness.harness_as_policy.live_policy import LiveActionResult
from autoharness.harness_as_policy.models import StepResult, TerminationReason


@dataclass
class FakeBaselineAdapter:
    """Small environment fake for baseline CLI tests."""

    step_result: StepResult | None = None
    setup_error: Exception | None = None
    env_id: str = "Fake-v0"
    rules: str = "Rules"
    action_format: str = "[A B]"
    max_steps: int = 1
    _observation: str = "initial observation"

    def create(self) -> None:
        """Initialize the fake environment."""
        if self.setup_error is not None:
            raise self.setup_error

    def reset(self) -> str:
        """Return the initial fake observation."""
        return self._observation

    def step(self, action: str) -> StepResult:
        """Return the configured environment outcome."""
        assert self.step_result is not None
        return self.step_result


def test_synthesize_cmd_requires_env() -> None:
    """synthesize command requires --env flag."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
    ):
        mock_synthesize.return_value = {
            "run_id": "test123",
            "stop_reason": "budget exhausted",
            "best_candidate_id": "001",
            "total_candidates": 3,
            "iterations_used": 2,
            "profile": "smoke",
            "model_call_count": 2,
            "logical_refinement_count": 2,
        }
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--profile",
                "smoke",
                "--model",
                "anthropic:claude-3-opus",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = synthesize_cmd()
    assert result is not None
    assert result["run_id"] == "test123"


def test_synthesize_cmd_creates_artifacts() -> None:
    """synthesize command creates artifact files."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
    ):
        mock_synthesize.return_value = {
            "run_id": "test",
            "stop_reason": "completed",
            "best_candidate_id": "001",
            "total_candidates": 3,
            "iterations_used": 2,
            "profile": "smoke",
            "model_call_count": 2,
            "logical_refinement_count": 2,
        }

        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--profile",
                "smoke",
                "--model",
                "anthropic:claude-3-opus",
                "--artifact-root",
                tmpdir,
            ],
        ):
            synthesize_cmd()
        artifact_dir = Path(tmpdir)
        dirs = list(artifact_dir.iterdir())
        assert len(dirs) >= 0


def test_synthesize_cmd_full_search() -> None:
    """synthesize command with --profile full-search passes refinements=256."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
    ):
        mock_synthesize.return_value = {
            "run_id": "test123",
            "stop_reason": "budget exhausted",
            "best_candidate_id": "001",
            "total_candidates": 3,
            "iterations_used": 2,
            "profile": "full-search",
            "model_call_count": 2,
            "logical_refinement_count": 2,
        }
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--profile",
                "full-search",
                "--model",
                "anthropic:claude-3-opus",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = synthesize_cmd()
    assert result is not None
    assert mock_synthesize.call_args.kwargs["refinements"] == 256


def test_synthesize_cmd_full_search_override() -> None:
    """synthesize command with --profile full-search and --refinements 10 passes refinements=10."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
    ):
        mock_synthesize.return_value = {
            "run_id": "test123",
            "stop_reason": "budget exhausted",
            "best_candidate_id": "001",
            "total_candidates": 3,
            "iterations_used": 2,
            "profile": "full-search",
            "model_call_count": 2,
            "logical_refinement_count": 2,
        }
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--profile",
                "full-search",
                "--refinements",
                "10",
                "--model",
                "anthropic:claude-3-opus",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = synthesize_cmd()
    assert result is not None
    assert mock_synthesize.call_args.kwargs["refinements"] == 10


def test_evaluate_cmd_requires_run() -> None:
    """evaluate command requires --run flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the best.py file that evaluate_cmd looks for
        (Path(tmpdir) / "best.py").write_text(
            "def propose_action(board: str) -> str: return '[A C]'\n"
            "def is_legal_action(board: str, action: str) -> bool: return True"
        )
        with patch("autoharness.cli.evaluate_policy") as mock_eval:
            mock_eval.return_value = []
            with patch("autoharness.cli.format_evaluation_summary") as mock_fmt:
                mock_fmt.return_value = "summary"
                result = evaluate_cmd(run_dir=Path(tmpdir))
    assert result is not None


def test_evaluate_cmd_persists_structured_termination_data() -> None:
    """Generated-policy artifacts retain the structured evaluation outcome."""
    result = EvaluationResult(
        env_id="TowerOfHanoi-v0",
        solved=False,
        reward=0.0,
        legal_action_count=1,
        steps_used=2,
        optimal_steps=7,
        termination_reason=TerminationReason.ILLEGAL_ACTION,
        failure_summary="Move is not legal",
        latency=0.01,
        execution_failure=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "best.py").write_text(
            "def propose_action(board: str) -> str: return '[A C]'\n"
            "def is_legal_action(board: str, action: str) -> bool: return True"
        )
        with patch("autoharness.cli.evaluate_policy", return_value=[result]):
            evaluate_cmd(run_dir=run_dir)

        data = json.loads((run_dir / "evaluation" / "generated-policy.json").read_text())

    persisted = data["results"][0]
    assert persisted["termination_reason"] == "illegal_action"
    assert persisted["failure_summary"] == "Move is not legal"
    assert "illegal_action_reason" not in persisted


@pytest.mark.parametrize(
    (
        "adapter",
        "action_result",
        "expected_reason",
        "expected_failure",
        "expected_execution_failure",
    ),
    [
        (
            FakeBaselineAdapter(setup_error=RuntimeError("setup failed")),
            None,
            TerminationReason.EXECUTION_FAILURE,
            "Environment setup failed: setup failed",
            True,
        ),
        (
            FakeBaselineAdapter(),
            LiveActionResult(
                action=None, success=False, latency=0.01, error_details="model unavailable"
            ),
            TerminationReason.EXECUTION_FAILURE,
            "model unavailable",
            True,
        ),
        (
            FakeBaselineAdapter(
                step_result=StepResult("next", "[B A]", False, 0.0, False, "Illegal move")
            ),
            LiveActionResult(action="[B A]", success=True, latency=0.01),
            TerminationReason.ILLEGAL_ACTION,
            "Illegal move",
            False,
        ),
        (
            FakeBaselineAdapter(step_result=StepResult("next", "[A C]", True, 1.0, True, "")),
            LiveActionResult(action="[A C]", success=True, latency=0.01),
            TerminationReason.ENVIRONMENT_TERMINATION,
            None,
            False,
        ),
        (
            FakeBaselineAdapter(step_result=StepResult("next", "[A C]", True, 0.0, False, "")),
            LiveActionResult(action="[A C]", success=True, latency=0.01),
            TerminationReason.STEP_LIMIT,
            None,
            False,
        ),
    ],
)
def test_evaluate_baseline_cmd_maps_each_exit_to_structured_termination_data(
    adapter: FakeBaselineAdapter,
    action_result: LiveActionResult | None,
    expected_reason: TerminationReason,
    expected_failure: str | None,
    expected_execution_failure: bool,
) -> None:
    """Baseline artifacts use the same outcome schema as generated policies."""
    live_policy = Mock()
    if action_result is not None:
        live_policy.act.return_value = action_result
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        with (
            patch("autoharness.cli.TowerOfHanoiAdapter", return_value=adapter),
            patch("autoharness.cli.LivePolicy", return_value=live_policy),
            patch(
                "autoharness.harness_as_policy.tower_of_hanoi.DIFFICULTY_MAP",
                {"v0": ("Fake-v0", 1, 1)},
            ),
        ):
            results = evaluate_baseline_cmd(run_dir=run_dir, model_id="fake:model")

        data = json.loads((run_dir / "evaluation" / "llm-baseline.json").read_text())

    assert results is not None
    result = results[0]
    persisted = data["results"][0]
    assert result.termination_reason == expected_reason
    assert result.failure_summary == expected_failure
    assert result.execution_failure is expected_execution_failure
    assert persisted["termination_reason"] == expected_reason.value
    assert persisted["failure_summary"] == expected_failure
    assert "illegal_action_reason" not in persisted


def test_main_synthesize_dispatches() -> None:
    """main dispatches synthesize command."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
        patch("autoharness.cli.evaluate_cmd") as mock_evaluate_cmd,
    ):
        mock_synthesize.return_value = {"run_id": "test", "artifact_root": tmpdir}
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--model",
                "anthropic:claude-3-opus",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = main()
    assert result == 0
    mock_evaluate_cmd.assert_called_once_with(Path(tmpdir) / "test")
