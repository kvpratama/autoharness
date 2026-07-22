#!/usr/bin/env python
"""Interactive TextArena playground for human play.

Usage:
    uv run python playground_textarena.py -e blackjack
    uv run python playground_textarena.py -e tower_of_hanoi
    uv run python playground_textarena.py -e tower_of_hanoi:medium -s 42

Extending: add an entry to ENV_REGISTRY with (adapter_class, init_kwargs).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from autoharness.harness_as_policy.environments.blackjack import BlackjackAdapter
from autoharness.harness_as_policy.environments.tower_of_hanoi import TowerOfHanoiAdapter

ENV_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "blackjack": (BlackjackAdapter, {}),
    "tower_of_hanoi": (TowerOfHanoiAdapter, {"difficulty": "v0"}),
}


def resolve_env(spec: str) -> Any:
    """Parse env spec like ``blackjack`` or ``tower_of_hanoi:medium`` and
    return an instantiated adapter."""
    if ":" in spec:
        name, difficulty = spec.split(":", 1)
    else:
        name = spec
        difficulty = None
    entry = ENV_REGISTRY.get(name)
    if entry is None:
        keys = ", ".join(sorted(ENV_REGISTRY))
        print(f"Unknown environment: {name!r}. Available: {keys}", file=sys.stderr)
        sys.exit(1)
    cls, kwargs = entry
    if difficulty is not None:
        kwargs = {**kwargs, "difficulty": difficulty}
    return cls(**kwargs)


def play_loop(adapter: Any, seed: int | None) -> None:
    """Run an interactive play session for the given adapter."""
    print(f"\n{'=' * 60}")
    print(f"  {adapter.env_id}")
    print(f"{'=' * 60}")
    print()
    print(adapter.rules)
    print()

    adapter.create()
    observation = adapter.reset(seed=seed)
    step_n = 0

    while step_n < adapter.max_steps:
        print(f"--- Step {step_n + 1} / {adapter.max_steps} ---")
        print()
        print(observation)
        print()
        print(f"Action format: {adapter.action_format}")
        print()

        try:
            raw = input("> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        result = adapter.step(raw)
        step_n += 1

        if result.is_legal:
            print(f"  [OK]      reward={result.reward}")
        else:
            print(f"  [ILLEGAL] reward={result.reward}")
        if result.feedback:
            print(f"  feedback: {result.feedback}")
        print()

        if result.terminated:
            print(f"  Game over after {step_n} steps.")
            print()
            print("Final observation:")
            print(result.observation)
            print()
            print(f"Terminal reward: {result.reward}")
            return

        observation = result.observation

    print(f"\nSession ended after {step_n} steps.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a TextArena environment interactively.")
    parser.add_argument(
        "-e",
        "--env",
        default="tower_of_hanoi",
        help="Environment spec (default: tower_of_hanoi)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None)",
    )
    args = parser.parse_args()

    adapter = resolve_env(args.env)
    play_loop(adapter, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
