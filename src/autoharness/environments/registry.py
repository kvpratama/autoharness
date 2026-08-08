"""Declarative registry of supported environment families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from autoharness.environments.base import EnvironmentAdapter
from autoharness.environments.blackjack import BlackjackAdapter
from autoharness.environments.frozen_lake import FrozenLakeAdapter
from autoharness.environments.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)
from autoharness.environments.twenty_forty_eight import TwentyFortyEightAdapter

AdapterFactory = Callable[[], EnvironmentAdapter]


@dataclass(frozen=True)
class EnvironmentSpec:
    """Immutable intrinsic configuration for one supported environment ID."""

    env_id: str
    family: str
    create_adapter: AdapterFactory
    optimal_steps: int = 0


ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    env_id: EnvironmentSpec(
        env_id=env_id,
        family="tower-of-hanoi",
        create_adapter=partial(TowerOfHanoiAdapter, difficulty=difficulty),
        optimal_steps=optimal_steps,
    )
    for difficulty, (env_id, _max_steps, optimal_steps) in DIFFICULTY_MAP.items()
}
ENVIRONMENTS["Blackjack-v0"] = EnvironmentSpec(
    env_id="Blackjack-v0",
    family="blackjack",
    create_adapter=BlackjackAdapter,
)
ENVIRONMENTS["FrozenLake-v0"] = EnvironmentSpec(
    env_id="FrozenLake-v0",
    family="frozen-lake",
    create_adapter=FrozenLakeAdapter,
)
ENVIRONMENTS["2048-v0"] = EnvironmentSpec(
    env_id="2048-v0",
    family="2048",
    create_adapter=TwentyFortyEightAdapter,
)


def valid_environment_ids() -> tuple[str, ...]:
    """Return all supported IDs in stable order."""
    return tuple(sorted(ENVIRONMENTS))


def get_environment_spec(env_id: str) -> EnvironmentSpec:
    """Resolve an exact environment ID or list valid alternatives."""
    try:
        return ENVIRONMENTS[env_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown environment ID {env_id!r}. "
            f"Valid options: {', '.join(valid_environment_ids())}"
        ) from error
