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
from collections.abc import Callable

from autoharness.environments.bandit import BanditAdapter
from autoharness.environments.base import EnvironmentAdapter
from autoharness.environments.blackjack import BlackjackAdapter
from autoharness.environments.frozen_lake import FrozenLakeAdapter
from autoharness.environments.tower_of_hanoi import TowerOfHanoiAdapter
from autoharness.environments.twenty_forty_eight import TwentyFortyEightAdapter

ENV_REGISTRY: dict[str, tuple[Callable[..., EnvironmentAdapter], dict[str, str]]] = {
    "blackjack": (BlackjackAdapter, {}),
    "tower_of_hanoi": (TowerOfHanoiAdapter, {"difficulty": "v0"}),
    "frozen_lake": (FrozenLakeAdapter, {}),
    "twenty_forty_eight": (TwentyFortyEightAdapter, {}),
    "bandit": (BanditAdapter, {"env_id": "Bandit-v0"}),
    "bandit_hard": (BanditAdapter, {"env_id": "Bandit-v0-hard"}),
}


# Supported suffix aliases for the bandit base name.
_BANDIT_SUFFIX_MAP: dict[str, str] = {
    "hard": "bandit_hard",
}


def resolve_env(spec: str) -> EnvironmentAdapter:
    """Resolve an environment spec to an instantiated adapter.

    Args:
        spec: Environment selector such as ``blackjack`` or
            ``tower_of_hanoi:medium``.

    Returns:
        The instantiated :class:`EnvironmentAdapter` for the requested
        environment.

    Raises:
        ValueError: If ``spec`` uses an unsupported suffix for ``bandit``.

    Note:
        If the environment name is unknown, this function prints the available
        environment names to stderr and exits the process with status code 1.
    """
    if ":" in spec:
        name, difficulty = spec.split(":", 1)
    else:
        name = spec
        difficulty = None

    # Handle bandit suffix variants explicitly before general registry lookup.
    if name == "bandit" and difficulty is not None:
        mapped = _BANDIT_SUFFIX_MAP.get(difficulty)
        if mapped is None:
            supported = ", ".join(sorted(_BANDIT_SUFFIX_MAP))
            raise ValueError(
                f"Unsupported bandit suffix {difficulty!r}. Supported: {supported}. "
                f"Use 'bandit_hard' directly or 'bandit:hard'."
            )
        name = mapped
        difficulty = None

    entry = ENV_REGISTRY.get(name)
    if entry is None:
        keys = ", ".join(sorted(ENV_REGISTRY))
        print(f"Unknown environment: {name!r}. Available: {keys}", file=sys.stderr)
        sys.exit(1)
    cls, kwargs = entry
    if difficulty is not None and "difficulty" in kwargs:
        kwargs = {**kwargs, "difficulty": difficulty}
    return cls(**kwargs)


def play_loop(adapter: EnvironmentAdapter, seed: int | None) -> None:
    """Run an interactive play session for an environment adapter.

    Args:
        adapter: The environment adapter used to create, reset, and step
            through the interactive session.
        seed: Optional seed passed to ``adapter.reset()`` before play begins.

    The session prints the rules and observations, prompts for actions until
    the user exits, EOF or Ctrl-C is received, the maximum step count is
    reached, or the environment terminates.
    """
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
