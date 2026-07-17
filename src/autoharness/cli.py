"""Top-level CLI for autoharness."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from autoharness.harness_as_policy.config import Settings
from autoharness.harness_as_policy.evaluation import (
    evaluate_policy,
    format_evaluation_summary,
)
from autoharness.harness_as_policy.live_policy import LivePolicy
from autoharness.harness_as_policy.models import Profile, TerminationReason
from autoharness.harness_as_policy.refiner import Refiner
from autoharness.harness_as_policy.search import synthesize
from autoharness.harness_as_policy.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)


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


_ENV_TO_DIFFICULTY: dict[str, str] = {
    env_id: diff for diff, (env_id, _, _) in DIFFICULTY_MAP.items()
}


def synthesize_cmd(
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """Run the synthesize command."""
    parser = _build_parser()
    if args is None:
        args, _ = parser.parse_known_args()

    settings_kwargs: dict[str, Any] = {}
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

    settings = Settings(**settings_kwargs)

    if settings.env_id not in _ENV_TO_DIFFICULTY:
        valid = ", ".join(sorted(_ENV_TO_DIFFICULTY))
        parser.error(f"Unknown environment ID '{settings.env_id}'. Valid options: {valid}")
    adapter = TowerOfHanoiAdapter(difficulty=_ENV_TO_DIFFICULTY[settings.env_id])

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
    )
    print(f"Run ID: {result.get('run_id', 'unknown')}")
    print(f"Stop reason: {result.get('stop_reason', 'unknown')}")
    print(f"Best candidate: {result.get('best_candidate_id', 'none')}")
    print(f"Total candidates: {result.get('total_candidates', 0)}")
    print(f"Model calls: {result.get('model_call_count', 0)}")
    return result


def evaluate_cmd(run_dir: Path) -> list[Any] | None:
    """Run the evaluate command."""
    best_policy_path = run_dir / "best.py"
    if not best_policy_path.exists():
        print(f"Error: no best.py found in {run_dir}", file=sys.stderr)
        return None
    source = best_policy_path.read_text()
    results = evaluate_policy(source=source)
    summary = format_evaluation_summary(results)
    print(summary)

    from autoharness.harness_as_policy.artifacts import ArtifactStore

    store = ArtifactStore(root=run_dir.parent, run_id=run_dir.name)
    store.write_evaluation(
        "generated-policy",
        {
            "results": [
                {
                    "env_id": r.env_id,
                    "solved": r.solved,
                    "reward": r.reward,
                    "legal_action_count": r.legal_action_count,
                    "steps_used": r.steps_used,
                    "optimal_steps": r.optimal_steps,
                    "termination_reason": (
                        r.termination_reason.value if r.termination_reason is not None else None
                    ),
                    "failure_summary": r.failure_summary,
                    "latency": r.latency,
                    "execution_failure": r.execution_failure,
                }
                for r in results
            ],
        },
    )
    return results


def evaluate_baseline_cmd(
    run_dir: Path,
    model_id: str,
    input_price: float | None = None,
    output_price: float | None = None,
) -> list[Any] | None:
    """Run the live-LLM baseline evaluate command."""
    import time

    from autoharness.harness_as_policy.evaluation import EvaluationResult
    from autoharness.harness_as_policy.tower_of_hanoi import DIFFICULTY_MAP

    results: list[EvaluationResult] = []
    total_model_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost = 0.0

    for diff_key, (env_id, _max_steps, optimal) in DIFFICULTY_MAP.items():
        adapter = TowerOfHanoiAdapter(difficulty=diff_key)
        live_policy = LivePolicy(
            model_id=model_id,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        try:
            adapter.create()
            adapter.reset()
        except Exception as e:
            results.append(
                EvaluationResult(
                    env_id=env_id,
                    solved=False,
                    reward=0.0,
                    legal_action_count=0,
                    steps_used=0,
                    optimal_steps=optimal,
                    termination_reason=TerminationReason.EXECUTION_FAILURE,
                    failure_summary=f"Environment setup failed: {e}",
                    latency=0.0,
                    execution_failure=True,
                )
            )
            continue
        start = time.monotonic()
        legal_actions = 0
        solved = False
        reward = 0.0
        steps_used = 0
        observation = adapter._observation if hasattr(adapter, "_observation") else ""
        for _ in range(adapter.max_steps):
            steps_used += 1
            action_result = live_policy.act(
                env_name=adapter.env_id,
                rules=adapter.rules,
                action_format=adapter.action_format,
                observation=observation,
            )
            total_model_calls += action_result.model_calls
            total_input_tokens += action_result.input_tokens
            total_output_tokens += action_result.output_tokens
            if action_result.estimated_cost_usd is not None:
                total_estimated_cost += action_result.estimated_cost_usd
            if not action_result.success or not action_result.action:
                results.append(
                    EvaluationResult(
                        env_id=env_id,
                        solved=False,
                        reward=0.0,
                        legal_action_count=legal_actions,
                        steps_used=steps_used,
                        optimal_steps=optimal,
                        termination_reason=TerminationReason.EXECUTION_FAILURE,
                        failure_summary=(action_result.error_details or "model_error"),
                        latency=time.monotonic() - start,
                        execution_failure=True,
                    )
                )
                break
            step_result = adapter.step(action_result.action)
            if not step_result.is_legal:
                results.append(
                    EvaluationResult(
                        env_id=env_id,
                        solved=False,
                        reward=0.0,
                        legal_action_count=legal_actions,
                        steps_used=steps_used,
                        optimal_steps=optimal,
                        termination_reason=TerminationReason.ILLEGAL_ACTION,
                        failure_summary=step_result.feedback or "Illegal",
                        latency=time.monotonic() - start,
                        execution_failure=False,
                    )
                )
                break
            legal_actions += 1
            if step_result.terminated:
                solved = step_result.reward >= 1.0
                reward = step_result.reward
                results.append(
                    EvaluationResult(
                        env_id=env_id,
                        solved=solved,
                        reward=reward,
                        legal_action_count=legal_actions,
                        steps_used=steps_used,
                        optimal_steps=optimal,
                        termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                        failure_summary=None,
                        latency=time.monotonic() - start,
                        execution_failure=False,
                    )
                )
                break
            observation = step_result.observation
        else:
            results.append(
                EvaluationResult(
                    env_id=env_id,
                    solved=False,
                    reward=0.0,
                    legal_action_count=legal_actions,
                    steps_used=steps_used,
                    optimal_steps=optimal,
                    termination_reason=TerminationReason.STEP_LIMIT,
                    failure_summary=None,
                    latency=time.monotonic() - start,
                    execution_failure=False,
                )
            )

    summary = format_evaluation_summary(results)
    summary += f"\n  Model calls: {total_model_calls}\n"
    summary += f"  Input tokens: {total_input_tokens}\n"
    summary += f"  Output tokens: {total_output_tokens}\n"
    if input_price is not None and output_price is not None:
        summary += f"  Estimated cost (USD): ${total_estimated_cost:.6f}\n"
    print(summary)

    from autoharness.harness_as_policy.artifacts import ArtifactStore

    store = ArtifactStore(root=run_dir.parent, run_id=run_dir.name)
    store.write_evaluation(
        "llm-baseline",
        {
            "results": [
                {
                    "env_id": r.env_id,
                    "solved": r.solved,
                    "reward": r.reward,
                    "legal_action_count": r.legal_action_count,
                    "steps_used": r.steps_used,
                    "optimal_steps": r.optimal_steps,
                    "termination_reason": (
                        r.termination_reason.value if r.termination_reason is not None else None
                    ),
                    "failure_summary": r.failure_summary,
                    "latency": r.latency,
                    "execution_failure": r.execution_failure,
                }
                for r in results
            ],
            "model_call_count": total_model_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost_usd": total_estimated_cost
            if input_price is not None and output_price is not None
            else None,
        },
    )
    return results


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
        level = os.environ.get("AUTOHARNESS_LOG_LEVEL")

    if level:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )

    if parsed.command == "synthesize":
        summary = synthesize_cmd(parsed)
        run_id = summary["run_id"]
        artifact_root = summary["artifact_root"]
        evaluate_cmd(Path(f"{artifact_root}/{run_id}"))
    elif parsed.command == "evaluate":
        evaluate_cmd(parsed.run)
    elif parsed.command == "evaluate-baseline":
        evaluate_baseline_cmd(parsed.run, parsed.model, parsed.input_price, parsed.output_price)
    return 0


if __name__ == "__main__":
    sys.exit(main())
