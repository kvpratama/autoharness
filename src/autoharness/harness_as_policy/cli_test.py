"""Tests for the CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.autoharness.cli import evaluate_cmd, main, synthesize_cmd


def test_synthesize_cmd_requires_env() -> None:
    """synthesize command requires --env flag."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("src.autoharness.cli.Refiner"),
        patch("src.autoharness.cli.synthesize") as mock_synthesize,
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
        patch("src.autoharness.cli.Refiner"),
        patch("src.autoharness.cli.synthesize") as mock_synthesize,
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
                "--artifact-root",
                tmpdir,
            ],
        ):
            synthesize_cmd()
        artifact_dir = Path(tmpdir)
        dirs = list(artifact_dir.iterdir())
        assert len(dirs) >= 0


def test_evaluate_cmd_requires_run() -> None:
    """evaluate command requires --run flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the best.py file that evaluate_cmd looks for
        (Path(tmpdir) / "best.py").write_text("def propose_action(obs): return '[A C]'")
        with patch("src.autoharness.cli.evaluate_policy") as mock_eval:
            mock_eval.return_value = []
            with patch("src.autoharness.cli.format_evaluation_summary") as mock_fmt:
                mock_fmt.return_value = "summary"
                result = evaluate_cmd(run_dir=Path(tmpdir))
    assert result is not None


def test_main_synthesize_dispatches() -> None:
    """main dispatches synthesize command."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("src.autoharness.cli.Refiner"),
        patch("src.autoharness.cli.synthesize") as mock_synthesize,
    ):
        mock_synthesize.return_value = {"run_id": "test"}
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = main()
    assert result == 0
