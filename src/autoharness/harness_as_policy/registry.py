"""Declarative registry of supported environment families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from autoharness.harness_as_policy.blackjack import BlackjackAdapter
from autoharness.harness_as_policy.environment import EnvironmentAdapter
from autoharness.harness_as_policy.tower_of_hanoi import DIFFICULTY_MAP, TowerOfHanoiAdapter

AdapterFactory = Callable[[], EnvironmentAdapter]


@dataclass(frozen=True)
class EvaluationCase:
    """One adapter factory in an environment's held-out evaluation suite."""

    create_adapter: AdapterFactory
    optimal_steps: int = 0


@dataclass(frozen=True)
class EnvironmentSpec:
    """Immutable configuration for a supported environment ID."""

    env_id: str
    family: str
    create_adapter: AdapterFactory
    default_training_rollouts: int
    evaluation_cases: tuple[EvaluationCase, ...]


HANOI_EVALUATION_CASES = tuple(
    EvaluationCase(partial(TowerOfHanoiAdapter, difficulty=difficulty), optimal_steps)
    for difficulty, (_env_id, _max_steps, optimal_steps) in DIFFICULTY_MAP.items()
)
ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    env_id: EnvironmentSpec(
        env_id=env_id,
        family="tower-of-hanoi",
        create_adapter=partial(TowerOfHanoiAdapter, difficulty=difficulty),
        default_training_rollouts=1,
        evaluation_cases=HANOI_EVALUATION_CASES,
    )
    for difficulty, (env_id, _max_steps, _optimal_steps) in DIFFICULTY_MAP.items()
}
ENVIRONMENTS["Blackjack-v0"] = EnvironmentSpec(
    env_id="Blackjack-v0",
    family="blackjack",
    create_adapter=BlackjackAdapter,
    default_training_rollouts=5,
    evaluation_cases=(EvaluationCase(BlackjackAdapter),),
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
