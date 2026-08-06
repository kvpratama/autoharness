"""Tests for the CLI."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autoharness.cli import (
    _baseline_artifact_name,
    _baseline_episode_count,
    evaluate_baseline_cmd,
    evaluate_cmd,
    main,
    synthesize_cmd,
)
from autoharness.environments.models import StepResult
from autoharness.environments.registry import EnvironmentSpec
from autoharness.harness_as_policy.evaluation import (
    EvaluationProtocol,
    EvaluationReport,
    EvaluationResult,
)
from autoharness.harness_as_policy.executor import policy_randomness_metadata
from autoharness.harness_as_policy.live_policy import LiveActionResult
from autoharness.harness_as_policy.models import TerminationReason


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
    reset_seed: int | None = None

    def create(self) -> None:
        """Initialize the fake environment."""
        if self.setup_error is not None:
            raise self.setup_error

    def reset(self, seed: int | None = None) -> str:
        """Return the initial fake observation."""
        self.reset_seed = seed
        return self._observation

    def step(self, action: str) -> StepResult:
        """Return the configured environment outcome."""
        assert self.step_result is not None
        return self.step_result


def test_synthesize_cmd_requires_env(capsys: pytest.CaptureFixture[str]) -> None:
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
            "attempted_refinements": 2,
            "successful_tree_nodes": 1,
            "provider_calls": 3,
            "profile": "smoke",
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
    output = capsys.readouterr().out
    assert "Attempted refinements: 2" in output
    assert "Successful tree nodes: 1" in output
    assert "Provider calls: 3" in output


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
            "attempted_refinements": 2,
            "successful_tree_nodes": 1,
            "provider_calls": 3,
            "profile": "smoke",
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
            "attempted_refinements": 2,
            "successful_tree_nodes": 1,
            "provider_calls": 3,
            "profile": "full-search",
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
            "attempted_refinements": 2,
            "successful_tree_nodes": 1,
            "provider_calls": 3,
            "profile": "full-search",
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


def test_synthesize_cmd_preserves_explicit_training_rollouts() -> None:
    """synthesize command preserves an explicitly configured training_rollouts value."""
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
            "attempted_refinements": 2,
            "successful_tree_nodes": 1,
            "provider_calls": 3,
            "profile": "smoke",
        }
        with patch(
            "sys.argv",
            [
                "autoharness",
                "synthesize",
                "--env",
                "TowerOfHanoi-v0",
                "--model",
                "anthropic:claude-3-opus",
                "--training-rollouts",
                "7",
                "--artifact-root",
                tmpdir,
            ],
        ):
            result = synthesize_cmd()
    assert result is not None
    assert mock_synthesize.call_args.kwargs["training_rollouts"] == 7


@pytest.mark.parametrize(("env_id", "expected"), [("TowerOfHanoi-v0", 1), ("Blackjack-v0", 5)])
def test_synthesize_cmd_uses_policy_training_rollout_default(env_id: str, expected: int) -> None:
    """Synthesis uses each environment's policy-owned default rollout count."""
    with (
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
        patch("sys.argv", ["autoharness", "synthesize", "--env", env_id, "--model", "test:model"]),
    ):
        mock_synthesize.return_value = {
            "run_id": "test",
            "stop_reason": "completed",
            "best_candidate_id": "001",
            "total_candidates": 1,
        }
        synthesize_cmd()

    assert mock_synthesize.call_args.kwargs["training_rollouts"] == expected


@pytest.mark.parametrize("training_rollouts", [0, -1])
def test_synthesize_cmd_reports_invalid_training_rollouts_as_cli_error(
    capsys: pytest.CaptureFixture[str],
    training_rollouts: int,
) -> None:
    """Invalid training rollouts exit through argparse instead of a traceback."""
    with patch(
        "sys.argv",
        [
            "autoharness",
            "synthesize",
            "--model",
            "anthropic:claude-3-opus",
            "--training-rollouts",
            str(training_rollouts),
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            synthesize_cmd()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "autoharness: error:" in error
    assert "training_rollouts" in error
    assert "Traceback" not in error


def _write_evaluation_run(
    run_dir: Path,
    *,
    env_id: str = "TowerOfHanoi-v0",
    include_training_seeds: bool = True,
    include_policy_randomness: bool = True,
) -> dict[str, object]:
    run_dir.mkdir()
    (run_dir / "best.py").write_text("def propose_action(observation: str) -> str: return '[A C]'")
    config: dict[str, object] = {"env_id": env_id, "environment_seed": 17}
    if include_training_seeds:
        config["training_episode_seeds"] = [11, 22]
    if include_policy_randomness:
        config["policy_randomness"] = policy_randomness_metadata()
    (run_dir / "config.json").write_text(json.dumps(config))
    return config


def _evaluation_report(protocol: EvaluationProtocol, policy_kind: str) -> EvaluationReport:
    is_generated = policy_kind == "generated-policy"
    results = [
        EvaluationResult(
            seed=seed,
            env_id=protocol.env_id,
            solved=True,
            reward=1.0,
            legal_action_count=1,
            action_attempt_count=1,
            steps_used=1,
            optimal_steps=1,
            termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
            failure_summary=None,
            latency=0.01,
            execution_failure=False,
            policy_invocation_count=1 if is_generated else 0,
            policy_seeds=(123456,) if is_generated else (),
        )
        for seed in protocol.episode_seeds
    ]
    return EvaluationReport.create(
        policy_kind,
        protocol,
        results,
        policy_randomness=policy_randomness_metadata() if is_generated else None,
    )


def test_evaluate_cmd_creates_protocol_and_structured_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _write_evaluation_run(run_dir)

    def evaluate(
        _source: str, _spec: EnvironmentSpec, protocol: EvaluationProtocol
    ) -> EvaluationReport:
        return _evaluation_report(protocol, "generated-policy")

    with patch("autoharness.cli.evaluate_policy", side_effect=evaluate):
        report = evaluate_cmd(run_dir)

    assert report is not None
    protocol_data = json.loads((run_dir / "evaluation" / "protocol.json").read_text())
    report_data = json.loads((run_dir / "evaluation" / "generated-policy.json").read_text())
    assert protocol_data["episode_count"] == 20
    training_seeds = config["training_episode_seeds"]
    assert isinstance(training_seeds, list)
    assert set(protocol_data["episode_seeds"]).isdisjoint(training_seeds)
    assert len(report_data["results"]) == 20
    assert "mean_reward" in report_data["aggregate"]


def test_evaluate_cmd_reuses_persisted_protocol(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir)
    known_protocol = EvaluationProtocol.create("TowerOfHanoi-v0", 99, [11, 22])
    (run_dir / "evaluation").mkdir()
    (run_dir / "evaluation" / "protocol.json").write_text(json.dumps(known_protocol.to_dict()))

    with patch(
        "autoharness.cli.evaluate_policy",
        return_value=_evaluation_report(known_protocol, "generated-policy"),
    ) as mock_evaluate:
        assert evaluate_cmd(run_dir) is not None

    assert mock_evaluate.call_args.args[2] == known_protocol


def test_evaluate_cmd_rejects_protocol_with_stale_training_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir)
    protocol = EvaluationProtocol.create("TowerOfHanoi-v0", 99, [1, 2])
    (run_dir / "evaluation").mkdir()
    path = run_dir / "evaluation" / "protocol.json"
    path.write_text(json.dumps(protocol.to_dict()))

    assert evaluate_cmd(run_dir) is None
    assert "training seeds do not match" in capsys.readouterr().err
    assert json.loads(path.read_text()) == protocol.to_dict()


def test_evaluate_cmd_rejects_malformed_protocol_without_overwriting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir)
    protocol = EvaluationProtocol.create("TowerOfHanoi-v0", 99, [1, 2]).to_dict()
    protocol["episode_seeds"] = list(range(19))
    protocol["episode_count"] = 19
    (run_dir / "evaluation").mkdir()
    path = run_dir / "evaluation" / "protocol.json"
    path.write_text(json.dumps(protocol))

    assert evaluate_cmd(run_dir) is None
    assert "evaluation protocol" in capsys.readouterr().err.lower()
    assert json.loads(path.read_text()) == protocol


def test_evaluate_cmd_rejects_legacy_config_without_training_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, include_training_seeds=False)

    assert evaluate_cmd(run_dir) is None
    assert "training_episode_seeds" in capsys.readouterr().err


def test_evaluate_cmd_rejects_config_without_policy_randomness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, include_policy_randomness=False)

    assert evaluate_cmd(run_dir) is None
    assert "policy_randomness" in capsys.readouterr().err


def test_evaluate_cmd_rejects_incompatible_policy_randomness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.py").write_text("def propose_action(observation: str) -> str: return '[A C]'")
    stale_metadata = {
        "seed_derivation": "autoharness-policy-seed-v1",
        "python_version": "3.99.0",
    }
    config: dict[str, object] = {
        "env_id": "TowerOfHanoi-v0",
        "environment_seed": 17,
        "training_episode_seeds": [11, 22],
        "policy_randomness": stale_metadata,
    }
    (run_dir / "config.json").write_text(json.dumps(config))

    assert evaluate_cmd(run_dir) is None
    assert "policy_randomness" in capsys.readouterr().err


def test_evaluate_baseline_cmd_succeeds_without_policy_randomness_in_config(
    tmp_path: Path,
) -> None:
    """evaluate-baseline does not require policy_randomness in config."""
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0", include_policy_randomness=False)
    adapters: list[FakeBaselineAdapter] = []
    live_policy = Mock()
    live_policy.act.return_value = LiveActionResult(
        action="[A C]", success=True, latency=0.01, model_calls=1
    )

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch("autoharness.cli.LivePolicy", return_value=live_policy),
    ):
        report = evaluate_baseline_cmd(run_dir, "fake:model")

    assert report is not None
    assert len(report.results) == 20


def _baseline_spec(adapters: list[FakeBaselineAdapter]) -> EnvironmentSpec:
    def create_adapter() -> FakeBaselineAdapter:
        adapter = FakeBaselineAdapter(step_result=StepResult("done", "[A C]", True, 1.0, True, ""))
        adapters.append(adapter)
        return adapter

    return EnvironmentSpec("Fake-v0", "fake", create_adapter, optimal_steps=1)


def test_baseline_uses_all_protocol_seeds_and_persists_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []
    live_policy = Mock()
    live_policy.act.return_value = LiveActionResult(
        action="[A C]", success=True, latency=0.01, model_calls=1
    )

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch("autoharness.cli.LivePolicy", return_value=live_policy) as live_policy_class,
    ):
        report = evaluate_baseline_cmd(run_dir, "fake:model")

    assert report is not None
    assert len(report.results) == 20
    assert [adapter.reset_seed for adapter in adapters] == list(report.protocol.episode_seeds)
    assert [result.seed for result in report.results] == list(report.protocol.episode_seeds)
    assert live_policy_class.call_count == 1
    data = json.loads((run_dir / "evaluation" / "llm-baseline.json").read_text())
    assert len(data["results"]) == 20
    assert isinstance(data["usage"], dict)
    assert isinstance(data["aggregate"], dict)


def test_baseline_actionless_failure_is_excluded_from_legality(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []
    live_policy = Mock()
    live_policy.act.side_effect = [
        LiveActionResult(None, False, 0.01, error_details="model unavailable"),
        *[LiveActionResult("[A C]", True, 0.01) for _ in range(19)],
    ]

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch("autoharness.cli.LivePolicy", return_value=live_policy),
    ):
        report = evaluate_baseline_cmd(run_dir, "fake:model")

    assert report is not None
    assert report.aggregate.execution_failure_count == 1
    assert report.aggregate.action_attempt_count == 19
    assert report.aggregate.legal_action_count == 19
    assert report.aggregate.legal_action_rate == 1.0


@pytest.mark.parametrize(
    ("model_id", "expected_count"),
    [
        ("google_genai:gemini-2.5-flash", 20),
        ("openai:gpt-5.2", 10),
        ("openai:gpt-5.2-high", 5),
    ],
)
def test_baseline_uses_paper_episode_count_and_persists_report(
    tmp_path: Path, model_id: str, expected_count: int
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []
    live_policy = Mock()
    live_policy.act.return_value = LiveActionResult(
        action="[A C]", success=True, latency=0.01, model_calls=1
    )

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch("autoharness.cli.LivePolicy", return_value=live_policy),
    ):
        report = evaluate_baseline_cmd(run_dir, model_id)

    assert report is not None
    assert report.protocol.episode_count == expected_count
    assert len(report.results) == expected_count
    assert [adapter.reset_seed for adapter in adapters] == list(report.protocol.episode_seeds)
    data = json.loads((run_dir / "evaluation" / "llm-baseline.json").read_text())
    assert len(data["results"]) == expected_count
    assert data["model_id"] == model_id
    assert data["protocol"]["episode_count"] == expected_count
    assert data["protocol"]["episode_seeds"] == list(report.protocol.episode_seeds)
    model_data = json.loads(
        (run_dir / "evaluation" / f"{_baseline_artifact_name(model_id)}.json").read_text()
    )
    assert model_data == data


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("gpt-5.2", 10),
        ("openai:gpt-5.2", 10),
        ("gpt-5.2-high", 5),
        ("openai:gpt-5.2-high", 5),
        ("openai:gpt-5.2-mini", 20),
        ("custom:my-gpt-5.2", 20),
    ],
)
def test_baseline_episode_count_uses_exact_model_name(model_id: str, expected: int) -> None:
    assert _baseline_episode_count(model_id) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai:gpt-5.2", "llm-baseline-openai-gpt-5.2-6544c8c29ef9"),
        ("Provider/Model Name", "llm-baseline-provider-model-name-56921857c4ae"),
        (":::", "llm-baseline-model-f1ae2a75ed1f"),
    ],
)
def test_baseline_artifact_name_is_filesystem_safe(model_id: str, expected: str) -> None:
    assert _baseline_artifact_name(model_id) == expected


def test_baseline_artifact_name_distinguishes_ids_with_same_sanitized_suffix() -> None:
    first = _baseline_artifact_name("provider:model/name")
    second = _baseline_artifact_name("provider:model name")

    assert first != second


def test_baseline_records_policy_construction_failure_and_continues(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch(
            "autoharness.cli.LivePolicy",
            side_effect=ImportError("policy boom"),
        ),
    ):
        report = evaluate_baseline_cmd(run_dir, "fake:model")

    assert report is not None
    assert len(report.results) == 20
    assert all(r.termination_reason == TerminationReason.EXECUTION_FAILURE for r in report.results)
    assert all(
        r.failure_summary == "Policy construction failed: policy boom" for r in report.results
    )
    generic = json.loads((run_dir / "evaluation" / "llm-baseline.json").read_text())
    model_specific = json.loads(
        (run_dir / "evaluation" / f"{_baseline_artifact_name('fake:model')}.json").read_text()
    )
    assert model_specific == generic
    assert generic["model_id"] == "fake:model"


def test_baseline_propagates_unexpected_policy_construction_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []

    with (
        patch("autoharness.cli.get_environment_spec", return_value=_baseline_spec(adapters)),
        patch("autoharness.cli.LivePolicy", side_effect=RuntimeError("policy boom")),
        pytest.raises(RuntimeError, match="policy boom"),
    ):
        evaluate_baseline_cmd(run_dir, "fake:model")


def test_generated_and_baseline_reports_reuse_identical_persisted_seeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_run(run_dir, env_id="Fake-v0")
    adapters: list[FakeBaselineAdapter] = []
    spec = _baseline_spec(adapters)

    def evaluate(
        _source: str, _spec: EnvironmentSpec, protocol: EvaluationProtocol
    ) -> EvaluationReport:
        return _evaluation_report(protocol, "generated-policy")

    live_policy = Mock()
    live_policy.act.return_value = LiveActionResult("[A C]", True, 0.01)
    with (
        patch("autoharness.cli.get_environment_spec", return_value=spec),
        patch("autoharness.cli.evaluate_policy", side_effect=evaluate),
    ):
        generated = evaluate_cmd(run_dir)
    with (
        patch("autoharness.cli.get_environment_spec", return_value=spec),
        patch("autoharness.cli.LivePolicy", return_value=live_policy),
    ):
        baseline = evaluate_baseline_cmd(run_dir, "fake:model")

    assert generated is not None
    assert baseline is not None
    assert [result.seed for result in generated.results] == [
        result.seed for result in baseline.results
    ]


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


def test_main_evaluate_missing_best_py_returns_nonzero() -> None:
    """main returns a nonzero status when evaluate is run and best.py is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        with patch(
            "sys.argv",
            [
                "autoharness",
                "evaluate",
                "--run",
                str(run_dir),
            ],
        ):
            result = main()
    assert result != 0


def test_main_evaluate_baseline_missing_config_returns_nonzero() -> None:
    """main returns a nonzero status when evaluate-baseline is run and config.json is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        with patch(
            "sys.argv",
            [
                "autoharness",
                "evaluate-baseline",
                "--run",
                str(run_dir),
                "--model",
                "fake:model",
            ],
        ):
            result = main()
    assert result not in (0, None)


def test_main_synthesize_evaluation_failure_returns_nonzero() -> None:
    """main returns a nonzero status when synthesize is run but evaluation fails."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.Refiner"),
        patch("autoharness.cli.synthesize") as mock_synthesize,
        patch("autoharness.cli.evaluate_cmd") as mock_evaluate_cmd,
    ):
        mock_synthesize.return_value = {"run_id": "test", "artifact_root": tmpdir}
        mock_evaluate_cmd.return_value = None
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
    assert result != 0


def test_main_configures_logging_from_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """main applies AUTOHARNESS_LOG_LEVEL via settings, not a raw os.environ gate."""
    monkeypatch.setenv("AUTOHARNESS_LOG_LEVEL", "DEBUG")
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.logging.basicConfig") as mock_basic_config,
        patch("autoharness.cli.evaluate_cmd", return_value=[]),
    ):
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        result = main(["evaluate", "--run", str(run_dir)])
    assert result == 0
    mock_basic_config.assert_called_once()
    assert mock_basic_config.call_args.kwargs["level"] == logging.DEBUG


def test_main_configures_logging_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main loads AUTOHARNESS_LOG_LEVEL from .env via settings even without os.environ."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOHARNESS_LOG_LEVEL", raising=False)
    (tmp_path / ".env").write_text("AUTOHARNESS_LOG_LEVEL=INFO\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (
        patch("autoharness.cli.load_dotenv"),
        patch("autoharness.cli.logging.basicConfig") as mock_basic_config,
        patch("autoharness.cli.evaluate_cmd", return_value=[]),
    ):
        result = main(["evaluate", "--run", str(run_dir)])
    assert result == 0
    mock_basic_config.assert_called_once()
    assert mock_basic_config.call_args.kwargs["level"] == logging.INFO


def test_main_skips_basic_config_when_log_level_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """main leaves logging alone when no CLI flag or AUTOHARNESS_LOG_LEVEL is set."""
    monkeypatch.delenv("AUTOHARNESS_LOG_LEVEL", raising=False)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("autoharness.cli.logging.basicConfig") as mock_basic_config,
        patch("autoharness.cli.evaluate_cmd", return_value=[]),
    ):
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        result = main(["evaluate", "--run", str(run_dir)])
    assert result == 0
    mock_basic_config.assert_not_called()
