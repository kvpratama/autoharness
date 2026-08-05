"""Top-level CLI for autoharness."""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv
from pydantic import ValidationError

from autoharness.harness_as_policy.artifacts import ArtifactStore
from autoharness.harness_as_policy.config import Settings, _LogLevelOnlySettings
from autoharness.harness_as_policy.environments.base import EnvironmentAdapter
from autoharness.harness_as_policy.environments.registry import (
    EnvironmentSpec,
    get_environment_spec,
)
from autoharness.harness_as_policy.evaluation import (
    EvaluationProtocol,
    EvaluationReport,
    EvaluationResult,
    EvaluationUsage,
    evaluate_action_provider_on_env,
    evaluate_policy,
    format_evaluation_summary,
)
from autoharness.harness_as_policy.executor import ExecutionResult, policy_randomness_metadata
from autoharness.harness_as_policy.live_policy import LivePolicy
from autoharness.harness_as_policy.models import Profile
from autoharness.harness_as_policy.refiner import Refiner
from autoharness.harness_as_policy.search import synthesize


class SettingsKwargs(TypedDict, total=False):
    """Keyword arguments to construct Settings."""

    env_id: str
    profile: str
    model: str
    refinements: int
    artifact_root: str
    thompson_seed: int
    execution_timeout: int
    max_source_size: int
    environment_seed: int
    training_rollouts: int


_PAPER_BASELINE_EPISODE_COUNTS: dict[str, int] = {
    "gpt-5.2": 10,
    "gpt-5.2-high": 5,
}


def _baseline_episode_count(model_id: str) -> int:
    """Return the Section 4.3 repetition count for a baseline model."""
    model_name = model_id.rsplit(":", maxsplit=1)[-1].lower()
    return _PAPER_BASELINE_EPISODE_COUNTS.get(model_name, 20)


def _baseline_artifact_name(model_id: str) -> str:
    """Return a filesystem-safe artifact name for one baseline model."""
    suffix = re.sub(r"[^a-z0-9._-]+", "-", model_id.lower()).strip("-._")
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return f"llm-baseline-{suffix or 'model'}-{digest}"


def _write_baseline_report(store: ArtifactStore, model_id: str, report: EvaluationReport) -> None:
    """Persist the latest and model-specific copies of a baseline report."""
    data = report.to_dict()
    store.write_evaluation("llm-baseline", data)
    store.write_evaluation(_baseline_artifact_name(model_id), data)


class SynthesisResult(TypedDict):
    """The result of a policy synthesis run."""

    run_id: str
    artifact_root: Path | str
    stop_reason: str
    best_candidate_id: str | None
    total_candidates: int
    attempted_refinements: int
    successful_tree_nodes: int
    provider_calls: int
    profile: str


def _build_parser() -> argparse.ArgumentParser:
    _shared = argparse.ArgumentParser(add_help=False)
    _shared.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    _shared.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (overrides --verbose)",
    )

    parser = argparse.ArgumentParser(
        description="AutoHarness — policy synthesis and evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    syn = subparsers.add_parser("synthesize", parents=[_shared], help="Synthesize a policy")
    syn.add_argument(
        "--env",
        default=None,
        help="Environment ID (e.g. TowerOfHanoi-v0)",
    )
    syn.add_argument(
        "--profile",
        default=None,
        choices=[p.value for p in Profile],
        help="Synthesis profile: smoke, low-cost, or full-search (default: smoke)",
    )
    syn.add_argument(
        "--model",
        default=None,
        help="Model identifier (e.g. google_genai:gemini-2.5-flash)",
    )
    syn.add_argument(
        "--refinements",
        type=int,
        default=None,
        help="Override refinement budget",
    )
    syn.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact output directory",
    )
    syn.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Thompson RNG seed",
    )
    syn.add_argument(
        "--execution-timeout",
        type=int,
        default=None,
        help="Per-action execution timeout in seconds",
    )
    syn.add_argument(
        "--max-source-size",
        type=int,
        default=None,
        help="Maximum policy source size in bytes",
    )
    syn.add_argument("--training-rollouts", type=int, default=None)
    syn.add_argument("--environment-seed", type=int, default=None)

    ev = subparsers.add_parser(
        "evaluate",
        parents=[_shared],
        help="Evaluate a synthesized policy",
    )
    ev.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run artifact directory",
    )

    evb = subparsers.add_parser(
        "evaluate-baseline",
        parents=[_shared],
        help="Evaluate a live LLM baseline",
    )
    evb.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run artifact directory",
    )
    evb.add_argument("--model", required=True, help="Model identifier")
    evb.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="Input price per million tokens (for cost estimation)",
    )
    evb.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="Output price per million tokens (for cost estimation)",
    )

    return parser


def synthesize_cmd(
    args: argparse.Namespace | None = None,
) -> SynthesisResult:
    """Run the synthesize command."""
    parser = _build_parser()
    if args is None:
        args, _ = parser.parse_known_args()

    settings_kwargs: SettingsKwargs = {}
    if args.env:
        settings_kwargs["env_id"] = args.env
    if args.profile:
        settings_kwargs["profile"] = args.profile
    if args.model:
        settings_kwargs["model"] = args.model
    if args.refinements is not None:
        settings_kwargs["refinements"] = args.refinements
    if args.artifact_root:
        settings_kwargs["artifact_root"] = args.artifact_root
    if args.seed is not None:
        settings_kwargs["thompson_seed"] = args.seed
    if args.execution_timeout is not None:
        settings_kwargs["execution_timeout"] = args.execution_timeout
    if args.max_source_size is not None:
        settings_kwargs["max_source_size"] = args.max_source_size
    if args.training_rollouts is not None:
        settings_kwargs["training_rollouts"] = args.training_rollouts
    if args.environment_seed is not None:
        settings_kwargs["environment_seed"] = args.environment_seed

    try:
        settings = Settings(**settings_kwargs)
    except ValidationError as exc:
        parser.error(str(exc))

    try:
        spec = get_environment_spec(settings.env_id)
    except ValueError as exc:
        parser.error(str(exc))
    adapter = spec.create_adapter()

    refiner = Refiner(model_id=settings.model)

    result = synthesize(
        adapter=adapter,
        profile=settings.profile,
        refiner=refiner,
        artifact_root=Path(settings.artifact_root),
        seed=settings.thompson_seed,
        refinements=settings.effective_refinements,
        execution_timeout=settings.execution_timeout,
        max_source_size=settings.max_source_size,
        model_id=settings.model,
        environment_seed=settings.environment_seed,
        training_rollouts=(
            spec.default_training_rollouts
            if settings.training_rollouts is None
            else settings.training_rollouts
        ),
    )
    print(f"Run ID: {result.get('run_id', 'unknown')}")
    print(f"Stop reason: {result.get('stop_reason', 'unknown')}")
    print(f"Best candidate: {result.get('best_candidate_id', 'none')}")
    print(f"Total candidates: {result.get('total_candidates', 0)}")
    print(f"Attempted refinements: {result.get('attempted_refinements', 0)}")
    print(f"Successful tree nodes: {result.get('successful_tree_nodes', 0)}")
    print(f"Provider calls: {result.get('provider_calls', 0)}")
    return cast(SynthesisResult, result)


def _load_or_create_evaluation_protocol(
    store: ArtifactStore, config: dict[str, object], spec: EnvironmentSpec
) -> EvaluationProtocol:
    """Load the persisted protocol, or create it strictly from run configuration."""
    training_seeds = config.get("training_episode_seeds")
    if not isinstance(training_seeds, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in training_seeds
    ):
        raise ValueError("Run config requires integer training_episode_seeds")
    typed_training_seeds = cast(list[int], training_seeds)
    existing = store.load_evaluation("protocol")
    if existing is not None:
        protocol = EvaluationProtocol.from_dict(existing, spec.env_id)
        if list(protocol.training_episode_seeds) != typed_training_seeds:
            raise ValueError("Evaluation protocol training seeds do not match run configuration")
        return protocol
    environment_seed = config.get("environment_seed")
    if not isinstance(environment_seed, int) or isinstance(environment_seed, bool):
        raise ValueError("Run config requires integer environment_seed for held-out evaluation")
    protocol = EvaluationProtocol.create(spec.env_id, environment_seed, typed_training_seeds)
    store.write_evaluation("protocol", protocol.to_dict())
    return protocol


def _validate_policy_randomness(config: Mapping[str, object]) -> None:
    """Require the current generated-policy randomness protocol and runtime."""
    actual = config.get("policy_randomness")
    expected = policy_randomness_metadata()
    if actual != expected:
        raise ValueError(
            "run policy_randomness metadata is missing or incompatible with this runtime"
        )


def evaluate_cmd(run_dir: Path) -> EvaluationReport | None:
    """Run the evaluate command."""
    best_policy_path = run_dir / "best.py"
    if not best_policy_path.exists():
        print(f"Error: no best.py found in {run_dir}", file=sys.stderr)
        return None
    store = ArtifactStore(run_dir.parent, run_dir.name)
    try:
        config = store.load_config()
        if config is None:
            raise ValueError(f"no config.json found in {run_dir}")
        _validate_policy_randomness(config)
        spec = get_environment_spec(config["env_id"])
        protocol = _load_or_create_evaluation_protocol(store, config, spec)
        report = evaluate_policy(best_policy_path.read_text(), spec, protocol)
        store.write_evaluation("generated-policy", report.to_dict())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: invalid evaluation protocol or run configuration: {exc}", file=sys.stderr)
        return None
    print(format_evaluation_summary(report))
    return report


def evaluate_baseline_cmd(
    run_dir: Path,
    model_id: str,
    input_price: float | None = None,
    output_price: float | None = None,
) -> EvaluationReport | None:
    """Run the live-LLM baseline evaluate command."""
    results = []
    total_model_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost = 0.0

    store = ArtifactStore(run_dir.parent, run_dir.name)
    try:
        config = store.load_config()
        if config is None:
            raise ValueError(f"no config.json found in {run_dir}")
        spec = get_environment_spec(config["env_id"])
        protocol = _load_or_create_evaluation_protocol(store, config, spec)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: invalid evaluation protocol or run configuration: {exc}", file=sys.stderr)
        return None

    baseline_protocol = protocol.prefix(_baseline_episode_count(model_id))

    try:
        live_policy = LivePolicy(
            model_id=model_id,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
    except ImportError as error:
        for seed in baseline_protocol.episode_seeds:
            start = time.monotonic()
            results.append(
                EvaluationResult.from_execution_failure(
                    seed,
                    spec.env_id,
                    spec.optimal_steps,
                    f"Policy construction failed: {error}",
                    time.monotonic() - start,
                )
            )
        usage = EvaluationUsage(0, 0, 0, None)
        report = EvaluationReport.create(
            "llm-baseline", baseline_protocol, results, usage, model_id=model_id
        )
        print(format_evaluation_summary(report))
        _write_baseline_report(store, model_id, report)
        return report

    for seed in baseline_protocol.episode_seeds:
        start = time.monotonic()
        try:
            adapter = spec.create_adapter()
        except Exception as error:
            results.append(
                EvaluationResult.from_execution_failure(
                    seed,
                    spec.env_id,
                    spec.optimal_steps,
                    f"Adapter construction failed: {error}",
                    time.monotonic() - start,
                )
            )
            continue

        def provide_action(
            observation: str,
            _adapter: EnvironmentAdapter = adapter,
        ) -> ExecutionResult:
            nonlocal total_model_calls, total_input_tokens, total_output_tokens
            nonlocal total_estimated_cost
            action_result = live_policy.act(
                env_name=_adapter.env_id,
                rules=_adapter.rules,
                action_format=_adapter.action_format,
                observation=observation,
            )
            total_model_calls += action_result.model_calls
            total_input_tokens += action_result.input_tokens
            total_output_tokens += action_result.output_tokens
            if action_result.estimated_cost_usd is not None:
                total_estimated_cost += action_result.estimated_cost_usd
            return ExecutionResult(
                success=action_result.success and action_result.action is not None,
                output=action_result.action,
                latency=action_result.latency,
                failure_type=None if action_result.success else "execution_failure",
                error_details=action_result.error_details,
            )

        results.append(
            evaluate_action_provider_on_env(adapter, provide_action, seed, spec.optimal_steps)
        )

    usage = EvaluationUsage(
        total_model_calls,
        total_input_tokens,
        total_output_tokens,
        total_estimated_cost if input_price is not None and output_price is not None else None,
    )
    report = EvaluationReport.create(
        "llm-baseline", baseline_protocol, results, usage, model_id=model_id
    )
    print(format_evaluation_summary(report))
    _write_baseline_report(store, model_id, report)
    return report


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    load_dotenv()
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.log_level:
        level = parsed.log_level
    elif parsed.verbose:
        level = "INFO"
    else:
        try:
            level = _LogLevelOnlySettings().log_level
        except ValidationError as exc:
            parser.error(f"Invalid AUTOHARNESS_LOG_LEVEL: {exc}")

    if level:
        level_name = level.upper()
        level_value = getattr(logging, level_name, None)
        if not isinstance(level_value, int):
            parser.error(
                f"Invalid log level {level!r}. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        logging.basicConfig(
            level=level_value,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )

    if parsed.command == "synthesize":
        summary = synthesize_cmd(parsed)
        run_id = summary["run_id"]
        artifact_root = summary["artifact_root"]
        results = evaluate_cmd(Path(f"{artifact_root}/{run_id}"))
        if results is None:
            return 1
    elif parsed.command == "evaluate":
        results = evaluate_cmd(parsed.run)
        if results is None:
            return 1
    elif parsed.command == "evaluate-baseline":
        results = evaluate_baseline_cmd(
            parsed.run, parsed.model, parsed.input_price, parsed.output_price
        )
        if results is None:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
