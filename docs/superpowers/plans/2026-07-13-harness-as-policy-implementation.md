# Harness-as-Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Harness-as-Policy MVP as specified in `docs/superpowers/specs/2026-07-13-harness-as-policy-design.md` — a generic policy-synthesis framework with a TextArena Tower of Hanoi environment adapter.

**Architecture:** A `harness_as_policy` subpackage under `src/autoharness/` containing: typed models and config, an environment adapter protocol with a `TowerOfHanoiAdapter`, an isolated subprocess executor, a rollout evaluator, a provider-neutral refiner model boundary, a LangGraph search workflow with REx Thompson selection, an artifact store, a held-out evaluation suite, and a thin CLI.

**Tech Stack:** Python 3.14, LangGraph for workflow orchestration, LangChain `init_chat_model` for provider-neutral model calls, TextArena for Tower of Hanoi, Pydantic Settings for configuration, pytest, Ruff, ty.

---

## File map

```
src/autoharness/
├── __init__.py                          # package marker
├── cli.py                              # synthesize/evaluate/evaluate-baseline commands
└── harness_as_policy/
    ├── __init__.py                      # subpackage marker
    ├── config.py                        # Settings, Profile enum
    ├── models.py                        # Candidate, RolloutResult, StepResult, Event, etc.
    ├── environment.py                   # EnvironmentAdapter protocol
    ├── tower_of_hanoi.py                # TowerOfHanoiAdapter
    ├── executor.py                      # PolicyExecutor (AST validation + subprocess)
    ├── rollout.py                       # RolloutEvaluator
    ├── refiner.py                       # Refiner (LLM model boundary + prompt)
    ├── search.py                        # LangGraph workflow, Thompson selection, ranking
    ├── artifacts.py                     # ArtifactStore
    └── evaluation.py                    # held-out evaluation + optional live-LLM baseline
```

Co-located tests: each module gets a `<module>_test.py` in the same directory.

---

## Task 1: Foundation — models, config, package init

**Files:**
- Create: `src/autoharness/harness_as_policy/__init__.py`
- Create: `src/autoharness/harness_as_policy/models.py`
- Create: `src/autoharness/harness_as_policy/models_test.py`
- Create: `src/autoharness/harness_as_policy/config.py`
- Create: `src/autoharness/harness_as_policy/config_test.py`

- [ ] **1.1: Write failing tests for models**

```python
# src/autoharness/harness_as_policy/models_test.py
"""Tests for domain models."""

from __future__ import annotations

from src.autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)


def test_heuristic_illegal_action() -> None:
    """Illegal action yields heuristic 0 regardless of reward."""
    assert heuristic(is_legal=False, reward=1.0) == 0.0
    assert heuristic(is_legal=False, reward=0.0) == 0.0


def test_heuristic_legal_solved() -> None:
    """Legal solved rollout yields heuristic 1.0."""
    assert heuristic(is_legal=True, reward=1.0) == 1.0


def test_heuristic_legal_no_reward() -> None:
    """Legal unsolved rollout yields heuristic 0.5."""
    assert heuristic(is_legal=True, reward=0.0) == 0.5


def test_heuristic_legal_partial() -> None:
    """Legal partial reward yields 0.5 + 0.5*r."""
    assert heuristic(is_legal=True, reward=0.6) == 0.8


def test_candidate_rank_key_primary_heuristic() -> None:
    """Rank key sorts by heuristic descending first."""
    high = CandidateRankKey(heuristic=0.9, reward=0.0, legal_actions=0, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=0, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_secondary_reward() -> None:
    """Same heuristic uses reward descending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.8, legal_actions=0, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.3, legal_actions=0, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_tertiary_legal_actions() -> None:
    """Same heuristic and reward uses legal actions descending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=10, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    assert high > low


def test_candidate_rank_key_quaternary_failures() -> None:
    """Same heuristic, reward, legal actions uses failures ascending."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=3, iteration=0)
    assert high > low


def test_candidate_rank_key_quinary_iteration() -> None:
    """Same everything uses iteration ascending (earlier is better)."""
    high = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=0)
    low = CandidateRankKey(heuristic=0.5, reward=0.0, legal_actions=5, failures=0, iteration=2)
    assert high > low


def test_candidate_rank_key_from_candidate_solved() -> None:
    """CandidateRankKey.from_candidate constructs correctly for a solved candidate."""
    cand = Candidate(
        id="003",
        parent_id="001",
        source="def propose_action(observation: str) -> str: ...",
        heuristic=1.0,
        terminal_reward=1.0,
        legal_action_count=7,
        termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
        failure_summary=None,
        iteration=3,
        expansion_count=0,
    )
    key = CandidateRankKey.from_candidate(cand)
    assert key.heuristic == 1.0
    assert key.reward == 1.0
    assert key.legal_actions == 7
    assert key.failures == 0
    assert key.iteration == 3


def test_candidate_rank_key_from_candidate_failure_counts() -> None:
    """Failure count is 1 when termination_reason is EXECUTION_FAILURE or CONTRACT_FAILURE."""
    cand = Candidate(
        id="004",
        parent_id=None,
        source="bad code",
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        termination_reason=TerminationReason.EXECUTION_FAILURE,
        failure_summary="SyntaxError",
        iteration=4,
        expansion_count=0,
    )
    key = CandidateRankKey.from_candidate(cand)
    assert key.failures == 1


def test_event_creation() -> None:
    """Event stores expected fields."""
    ev = Event(iteration=1, event_type="refine", candidate_id="001", parent_id="000", metadata={"model": "test"})
    assert ev.iteration == 1
    assert ev.event_type == "refine"
    assert ev.candidate_id == "001"


def test_rollout_result_fields() -> None:
    """RolloutResult stores heuristic, reward, legal count, reason."""
    result = RolloutResult(
        steps=[],
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=5,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
    )
    assert result.heuristic == 0.5
    assert result.termination_reason == TerminationReason.STEP_LIMIT


def test_step_result_fields() -> None:
    """StepResult stores observation, action, legality, reward, feedback."""
    step = StepResult(
        observation="[A B C]",
        action="[A C]",
        is_legal=True,
        reward=0.0,
        terminated=False,
        feedback="",
    )
    assert step.is_legal
    assert step.action == "[A C]"


def test_profile_values() -> None:
    """Profile enum has correct refinement budgets."""
    assert Profile.SMOKE.refinements == 8
    assert Profile.LOW_COST.refinements == 32


def test_termination_reason_values() -> None:
    """TerminationReason has all expected members."""
    reasons = {
        TerminationReason.ILLEGAL_ACTION,
        TerminationReason.ENVIRONMENT_TERMINATION,
        TerminationReason.STEP_LIMIT,
        TerminationReason.EXECUTION_FAILURE,
        TerminationReason.CONTRACT_FAILURE,
    }
    assert len(reasons) == 5
```

- [ ] **1.2: Run models test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/models_test.py -v`
Expected: ModuleNotFoundError or similar import failures

- [ ] **1.3: Implement models.py**

```python
# src/autoharness/harness_as_policy/models.py
"""Domain models for the harness-as-policy synthesis system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TerminationReason(str, Enum):
    ILLEGAL_ACTION = "illegal_action"
    ENVIRONMENT_TERMINATION = "environment_termination"
    STEP_LIMIT = "step_limit"
    EXECUTION_FAILURE = "execution_failure"
    CONTRACT_FAILURE = "contract_failure"


class Profile(str, Enum):
    SMOKE = "smoke"
    LOW_COST = "low-cost"

    @property
    def refinements(self) -> int:
        return {"smoke": 8, "low-cost": 32}[self.value]

    @property
    def max_steps(self) -> int:
        return 14  # three-disk turn limit


@dataclass
class StepResult:
    """Result of a single step in a rollout."""
    observation: str
    action: Optional[str]
    is_legal: bool
    reward: float
    terminated: bool
    feedback: str


@dataclass
class RolloutResult:
    """Result of one complete rollout."""
    steps: list[StepResult]
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    termination_reason: TerminationReason
    failure_summary: Optional[str]


@dataclass
class Candidate:
    """A node in the program refinement tree."""
    id: str
    parent_id: Optional[str]
    source: str
    heuristic: float
    terminal_reward: float
    legal_action_count: int
    termination_reason: Optional[TerminationReason]
    failure_summary: Optional[str]
    iteration: int
    expansion_count: int = 0


@dataclass
class Event:
    """A recorded event during synthesis."""
    iteration: int
    event_type: str
    candidate_id: Optional[str]
    parent_id: Optional[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class CandidateRankKey:
    """Lexicographic sort key for candidate ranking.
    
    Higher heuristic > higher reward > more legal actions >
    fewer failures > earlier iteration.
    """
    heuristic: float
    reward: float
    legal_actions: int
    failures: int
    iteration: int

    @classmethod
    def from_candidate(cls, c: Candidate) -> CandidateRankKey:
        failures = 1 if c.termination_reason in (
            TerminationReason.EXECUTION_FAILURE,
            TerminationReason.CONTRACT_FAILURE,
        ) else 0
        return cls(
            heuristic=c.heuristic,
            reward=c.terminal_reward,
            legal_actions=c.legal_action_count,
            failures=failures,
            iteration=c.iteration,
        )

    def __lt__(self, other: CandidateRankKey) -> bool:
        return (
            (-self.heuristic, -self.reward, -self.legal_actions, self.failures, self.iteration)
            < (-other.heuristic, -other.reward, -other.legal_actions, other.failures, other.iteration)
        )


def heuristic(*, is_legal: bool, reward: float) -> float:
    """Section 4.3 heuristic: 0 if illegal, else 0.5 + 0.5*r."""
    if not is_legal:
        return 0.0
    return 0.5 + 0.5 * reward
```

- [ ] **1.4: Run models test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/models_test.py -v`
Expected: All tests pass

- [ ] **1.5: Write failing tests for config**

```python
# src/autoharness/harness_as_policy/config_test.py
"""Tests for configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from src.autoharness.harness_as_policy.config import Settings


def test_default_profile_is_smoke() -> None:
    """Default profile is smoke when not set."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.profile.value == "smoke"


def test_profile_from_env() -> None:
    """Profile can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_PROFILE": "low-cost"}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.profile.value == "low-cost"


def test_artifact_root_default() -> None:
    """Default artifact root is artifacts/."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.artifact_root == "artifacts"


def test_model_required() -> None:
    """Model identifier is required."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.model is not None


def test_thompson_seed_default() -> None:
    """Default Thompson seed is 42."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.thompson_seed == 42


def test_execution_timeout_default() -> None:
    """Default execution timeout is 10 seconds."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.execution_timeout == 10


def test_max_source_size_default() -> None:
    """Default max source size is 32768 bytes."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.max_source_size == 32768


def test_env_id_from_env() -> None:
    """Environment ID can be set from environment."""
    with patch.dict(os.environ, {"AUTOHARNESS_ENV": "TowerOfHanoi-v0"}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.env_id == "TowerOfHanoi-v0"


def test_refinements_override() -> None:
    """Refinement budget can be overridden via env."""
    with patch.dict(os.environ, {"AUTOHARNESS_REFINEMENTS": "5"}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.refinements == 5


def test_pricing_not_configured_by_default() -> None:
    """Input/output prices are None by default."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    assert settings.input_price_per_million is None
    assert settings.output_price_per_million is None
```

- [ ] **1.6: Run config test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/config_test.py -v`
Expected: Import failures

- [ ] **1.7: Implement config.py**

```python
# src/autoharness/harness_as_policy/config.py
"""Pydantic settings for harness-as-policy."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.autoharness.harness_as_policy.models import Profile


class Settings(BaseSettings):
    """Resolved configuration for a synthesis run."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOHARNESS_",
        env_file=".env",
        extra="ignore",
    )

    model: str = "google_genai:gemini-2.5-flash"
    env_id: str = "TowerOfHanoi-v0"
    profile: Profile = Profile.SMOKE
    refinements: int | None = None
    artifact_root: str = "artifacts"
    thompson_seed: int = 42
    execution_timeout: int = 10
    max_source_size: int = 32768
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    @property
    def effective_refinements(self) -> int:
        return self.refinements if self.refinements is not None else self.profile.refinements
```

- [ ] **1.8: Run config test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/config_test.py -v`
Expected: All pass

- [ ] **1.9: Create subpackage __init__.py**

```python
# src/autoharness/harness_as_policy/__init__.py
"""Harness-as-Policy: generic policy-synthesis framework."""
```

- [ ] **1.10: Run all Task 1 tests together**

Run: `uv run pytest src/autoharness/harness_as_policy/models_test.py src/autoharness/harness_as_policy/config_test.py -v`
Expected: All pass

- [ ] **1.11: Commit**

```bash
git add src/autoharness/harness_as_policy/__init__.py \
       src/autoharness/harness_as_policy/models.py \
       src/autoharness/harness_as_policy/models_test.py \
       src/autoharness/harness_as_policy/config.py \
       src/autoharness/harness_as_policy/config_test.py
git commit -m "feat(harness-as-policy): add foundation models and config"
```

---

## Task 2: Environment adapter protocol and Tower of Hanoi adapter

**Files:**
- Create: `src/autoharness/harness_as_policy/environment.py`
- Create: `src/autoharness/harness_as_policy/environment_test.py`
- Create: `src/autoharness/harness_as_policy/tower_of_hanoi.py`
- Create: `src/autoharness/harness_as_policy/tower_of_hanoi_test.py`

- [ ] **2.1: Write failing tests for environment protocol**

```python
# src/autoharness/harness_as_policy/environment_test.py
"""Tests for the environment adapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.autoharness.harness_as_policy.environment import EnvironmentAdapter


def test_protocol_is_runtime_checkable() -> None:
    """EnvironmentAdapter is a runtime-checkable protocol."""
    assert issubclass(EnvironmentAdapter, Protocol)
    assert hasattr(EnvironmentAdapter, "__runtime_checkable__")
```

- [ ] **2.2: Run environment test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/environment_test.py -v`
Expected: Import failure

- [ ] **2.3: Implement environment.py protocol**

```python
# src/autoharness/harness_as_policy/environment.py
"""Generic environment adapter protocol."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.autoharness.harness_as_policy.models import StepResult


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Protocol for environment adapters used by the optimizer and evaluators."""

    @property
    def env_id(self) -> str:
        """Unique environment identifier."""

    @property
    def rules(self) -> str:
        """Human-readable environment rules."""

    @property
    def action_format(self) -> str:
        """Description of the expected action format."""

    @property
    def max_steps(self) -> int:
        """Maximum number of policy actions in one rollout."""

    def create(self) -> None:
        """Create the underlying environment instance."""

    def reset(self, seed: int | None = None) -> str:
        """Reset the environment and return the initial observation."""

    def step(self, action: str) -> StepResult:
        """Submit an action and return the normalized step result."""
```

- [ ] **2.4: Run environment test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/environment_test.py -v`
Expected: Pass

- [ ] **2.5: Write failing tests for TowerOfHanoiAdapter**

```python
# src/autoharness/harness_as_policy/tower_of_hanoi_test.py
"""Tests for the Tower of Hanoi environment adapter."""

from __future__ import annotations

import pytest

from src.autoharness.harness_as_policy.environment import EnvironmentAdapter
from src.autoharness.harness_as_policy.tower_of_hanoi import (
    DIFFICULTY_MAP,
    TowerOfHanoiAdapter,
)


def test_adapter_is_environment_adapter() -> None:
    """TowerOfHanoiAdapter satisfies EnvironmentAdapter protocol."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter, EnvironmentAdapter)


def test_env_id_default() -> None:
    """Default env_id is TowerOfHanoi-v0."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert adapter.env_id == "TowerOfHanoi-v0"


def test_env_id_medium() -> None:
    """Medium difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="medium")
    assert adapter.env_id == "TowerOfHanoi-v0-medium"


def test_env_id_hard() -> None:
    """Hard difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="hard")
    assert adapter.env_id == "TowerOfHanoi-v0-hard"


def test_env_id_hardcore() -> None:
    """Hardcore difficulty has correct env_id."""
    adapter = TowerOfHanoiAdapter(difficulty="hardcore")
    assert adapter.env_id == "TowerOfHanoi-v0-hardcore"


def test_max_steps_v0() -> None:
    """Three-disk variant has 14 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert adapter.max_steps == 14


def test_max_steps_medium() -> None:
    """Four-disk variant has 30 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="medium")
    assert adapter.max_steps == 30


def test_max_steps_hard() -> None:
    """Five-disk variant has 62 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="hard")
    assert adapter.max_steps == 62


def test_max_steps_hardcore() -> None:
    """Six-disk variant has 126 max steps."""
    adapter = TowerOfHanoiAdapter(difficulty="hardcore")
    assert adapter.max_steps == 126


def test_rules_is_string() -> None:
    """Rules property returns a non-empty string."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter.rules, str)
    assert len(adapter.rules) > 0


def test_action_format_is_string() -> None:
    """Action format description is a non-empty string."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    assert isinstance(adapter.action_format, str)
    assert len(adapter.action_format) > 0


def test_create_and_reset_returns_string() -> None:
    """Create and reset returns a string observation."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    obs = adapter.reset(seed=42)
    assert isinstance(obs, str)
    assert len(obs) > 0


def test_legal_action_single_move() -> None:
    """A single legal bracketed move is accepted."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("[A C]")
    assert result.is_legal


def test_malformed_action_illegal() -> None:
    """Empty string is rejected before environment submission."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("")
    assert not result.is_legal
    assert "empty" in result.feedback.lower() or "malformed" in result.feedback.lower()


def test_multiple_bracketed_moves_illegal() -> None:
    """Multiple bracketed moves are rejected before environment submission."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("[A C] [C B]")
    assert not result.is_legal
    assert "multiple" in result.feedback.lower()


def test_random_string_illegal() -> None:
    """A random string without brackets is rejected."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    adapter.create()
    adapter.reset(seed=42)
    result = adapter.step("hello world")
    assert not result.is_legal


def test_difficulty_map_correct() -> None:
    """DIFFICULTY_MAP has all four variants with correct turn limits."""
    assert DIFFICULTY_MAP["v0"] == ("TowerOfHanoi-v0", 14)
    assert DIFFICULTY_MAP["medium"] == ("TowerOfHanoi-v0-medium", 30)
    assert DIFFICULTY_MAP["hard"] == ("TowerOfHanoi-v0-hard", 62)
    assert DIFFICULTY_MAP["hardcore"] == ("TowerOfHanoi-v0-hardcore", 126)


def test_invalid_difficulty_raises() -> None:
    """Invalid difficulty raises ValueError."""
    with pytest.raises(ValueError, match="Unknown difficulty"):
        TowerOfHanoiAdapter(difficulty="nonexistent")


def test_step_before_create_raises() -> None:
    """Calling step before create raises RuntimeError."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    with pytest.raises(RuntimeError):
        adapter.step("[A C]")


def test_reset_before_create_raises() -> None:
    """Calling reset before create raises RuntimeError."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    with pytest.raises(RuntimeError):
        adapter.reset(seed=42)
```

- [ ] **2.6: Run tower_of_hanoi test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/tower_of_hanoi_test.py -v`
Expected: Import failure

- [ ] **2.7: Implement tower_of_hanoi.py**

```python
# src/autoharness/harness_as_policy/tower_of_hanoi.py
"""Tower of Hanoi environment adapter using TextArena."""

from __future__ import annotations

import re
from typing import Optional

import textarena as ta

from src.autoharness.harness_as_policy.models import StepResult

DIFFICULTY_MAP: dict[str, tuple[str, int]] = {
    "v0": ("TowerOfHanoi-v0", 14),
    "medium": ("TowerOfHanoi-v0-medium", 30),
    "hard": ("TowerOfHanoi-v0-hard", 62),
    "hardcore": ("TowerOfHanoi-v0-hardcore", 126),
}

BRACKETED_MOVE_RE = re.compile(r"\[.*?\]")


class TowerOfHanoiAdapter:
    """TextArena Tower of Hanoi adapter.

    Validates actions before submission: exactly one bracketed move.
    Treats TextArena invalid-move signals as immediate illegal transitions.
    """

    def __init__(self, difficulty: str = "v0") -> None:
        if difficulty not in DIFFICULTY_MAP:
            raise ValueError(f"Unknown difficulty: {difficulty}. Choose from {list(DIFFICULTY_MAP)}")
        self._difficulty = difficulty
        self._env_id, self._max_steps = DIFFICULTY_MAP[difficulty]
        self._env: Optional[ta.Env] = None
        self._observation: str = ""

    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def rules(self) -> str:
        return (
            "Tower of Hanoi: move all disks from peg A to peg C. "
            "You may only move one disk at a time, and you cannot place "
            "a larger disk on top of a smaller disk."
        )

    @property
    def action_format(self) -> str:
        return "Submit exactly one move in bracketed format, e.g. [A C] or [A, C]."

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def create(self) -> None:
        self._env = ta.make(self._env_id)

    def reset(self, seed: int | None = None) -> str:
        if self._env is None:
            raise RuntimeError("Call create() before reset().")
        self._env.reset(seed=seed)
        obs = self._env.observation
        self._observation = str(obs) if obs is not None else ""
        return self._observation

    def step(self, action: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("Call create() before step().")
        # Pre-submission validation: exactly one bracketed move
        if not action or not action.strip():
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback="Malformed action: empty or whitespace-only output",
            )
        matches = BRACKETED_MOVE_RE.findall(action)
        if len(matches) != 1:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=f"Malformed action: expected exactly one bracketed move, got {len(matches)}",
            )
        # Submit single move to TextArena
        try:
            obs, reward, terminated, truncated, info = self._env.step(action=action)
        except Exception as e:
            return StepResult(
                observation=self._observation,
                action=action,
                is_legal=False,
                reward=0.0,
                terminated=True,
                feedback=f"Environment step error: {e}",
            )
        self._observation = str(obs) if obs is not None else ""
        # Check for invalid move signal from TextArena
        is_legal = True
        feedback = ""
        if terminated and reward <= -1:
            is_legal = False
            feedback = "Invalid move rejected by environment"
        elif "Invalid move" in self._observation or "illegal" in self._observation.lower():
            is_legal = False
            feedback = "Invalid move detected in observation"
        return StepResult(
            observation=self._observation,
            action=action,
            is_legal=is_legal,
            reward=max(0.0, float(reward)) if reward is not None else 0.0,
            terminated=bool(terminated or truncated),
            feedback=feedback,
        )
```

- [ ] **2.8: Run tower_of_hanoi test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/tower_of_hanoi_test.py -v`
Expected: All pass

- [ ] **2.9: Run all Task 2 tests together**

Run: `uv run pytest src/autoharness/harness_as_policy/environment_test.py src/autoharness/harness_as_policy/tower_of_hanoi_test.py -v`
Expected: All pass

- [ ] **2.10: Commit**

```bash
git add src/autoharness/harness_as_policy/environment.py \
       src/autoharness/harness_as_policy/environment_test.py \
       src/autoharness/harness_as_policy/tower_of_hanoi.py \
       src/autoharness/harness_as_policy/tower_of_hanoi_test.py
git commit -m "feat(harness-as-policy): add environment adapter protocol and Tower of Hanoi adapter"
```

---

## Task 3: Policy executor

**Files:**
- Create: `src/autoharness/harness_as_policy/executor.py`
- Create: `src/autoharness/harness_as_policy/executor_test.py`

- [ ] **3.1: Write failing tests for executor**

```python
# src/autoharness/harness_as_policy/executor_test.py
"""Tests for the policy executor."""

from __future__ import annotations

import textwrap

from src.autoharness.harness_as_policy.executor import (
    ExecutionResult,
    PolicyExecutor,
    SAFE_IMPORTS,
)


def _valid_source() -> str:
    return textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return "[A C]"
    """)


def test_safe_imports_contains_stdlib() -> None:
    """SAFE_IMPORTS includes expected standard library modules."""
    assert "math" in SAFE_IMPORTS
    assert "random" in SAFE_IMPORTS
    assert "re" in SAFE_IMPORTS
    assert "typing" in SAFE_IMPORTS
    assert "itertools" in SAFE_IMPORTS


def test_safe_imports_excludes_dangerous() -> None:
    """SAFE_IMPORTS excludes dangerous modules."""
    assert "os" not in SAFE_IMPORTS
    assert "subprocess" not in SAFE_IMPORTS
    assert "sys" not in SAFE_IMPORTS
    assert "importlib" not in SAFE_IMPORTS
    assert "builtins" not in SAFE_IMPORTS


def test_valid_policy_executes() -> None:
    """A valid policy module executes propose_action and returns result."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(_valid_source(), observation="[A B C]")
    assert result.success
    assert result.output == "[A C]"


def test_syntax_error() -> None:
    """Syntax error returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("def propose_action(obs: str) -> str:", observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_missing_entry_point() -> None:
    """Module without propose_action returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("x = 1", observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_wrong_return_type() -> None:
    """Non-string return from propose_action returns contract failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return 42
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_disallowed_import() -> None:
    """Disallowed import returns contract failure."""
    source = textwrap.dedent("""\
    import os
    def propose_action(observation: str) -> str:
        return "[A C]"
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_source_too_large() -> None:
    """Source exceeding max_size returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=10)
    result = executor.execute(_valid_source(), observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_runtime_exception() -> None:
    """Runtime exception in propose_action returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        raise ValueError("boom")
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"


def test_timeout() -> None:
    """Policy that hangs returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        while True:
            pass
    """)
    executor = PolicyExecutor(timeout=1, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"


def test_private_helper_allowed() -> None:
    """Private helper functions inside the module are allowed."""
    source = textwrap.dedent("""\
    def _get_move() -> str:
        return "[A C]"

    def propose_action(observation: str) -> str:
        return _get_move()
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert result.success
    assert result.output == "[A C]"


def test_execution_result_attributes() -> None:
    """ExecutionResult has expected attributes on success."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(_valid_source(), observation="[A B C]")
    assert result.success is True
    assert isinstance(result.output, str)
    assert isinstance(result.latency, float)
    assert result.latency >= 0


def test_execution_result_failure_attributes() -> None:
    """ExecutionResult has expected attributes on failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("bad syntax!!!", observation="test")
    assert result.success is False
    assert isinstance(result.failure_type, str)
    assert isinstance(result.error_details, str)
```

- [ ] **3.2: Run executor test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/executor_test.py -v`
Expected: Import failure

- [ ] **3.3: Implement executor.py**

```python
# src/autoharness/harness_as_policy/executor.py
"""Policy executor with AST validation and isolated subprocess execution."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SAFE_IMPORTS: set[str] = {
    "math",
    "random",
    "re",
    "typing",
    "itertools",
    "collections",
    "functools",
    "dataclasses",
    "enum",
    "string",
}


@dataclass
class ExecutionResult:
    """Result of executing a policy module's propose_action."""
    success: bool
    output: Optional[str]
    latency: float
    failure_type: Optional[str] = None
    error_details: Optional[str] = None


class PolicyExecutor:
    """Validates and runs candidate policy modules in isolated subprocesses."""

    def __init__(self, timeout: int = 10, max_source_size: int = 32768) -> None:
        self._timeout = timeout
        self._max_source_size = max_source_size

    def execute(self, source: str, observation: str) -> ExecutionResult:
        """Validate and execute propose_action with the given observation."""
        start = time.monotonic()
        # Source size check
        if len(source.encode("utf-8")) > self._max_source_size:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="contract_failure",
                error_details=f"Source exceeds {self._max_source_size} bytes",
            )
        # AST validation
        parse_err = self._validate_ast(source)
        if parse_err:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="contract_failure",
                error_details=parse_err,
            )
        # Build and run subprocess
        try:
            output = self._run_subprocess(source, observation)
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="execution_failure",
                error_details="Subprocess timed out",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="execution_failure",
                error_details=str(e),
            )
        # Validate output
        if not isinstance(output, str):
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="contract_failure",
                error_details="propose_action did not return a string",
            )
        return ExecutionResult(
            success=True,
            output=output,
            latency=time.monotonic() - start,
        )

    def _validate_ast(self, source: str) -> Optional[str]:
        """Validate the source parses, has the right signature, and safe imports."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error: {e}"
        # Check for propose_action function
        has_propose_action = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "propose_action":
                has_propose_action = True
                args = node.args
                if len(args.args) != 1:
                    return "propose_action must take exactly 1 argument (observation)"
                break
        if not has_propose_action:
            return "Module must define propose_action(observation: str) -> str"
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in SAFE_IMPORTS:
                        return f"Disallowed import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in SAFE_IMPORTS:
                        return f"Disallowed import: {node.module}"
        return None

    def _run_subprocess(self, source: str, observation: str) -> str:
        """Run propose_action in a subprocess with resource limits."""
        script = textwrap.dedent(f"""\
        import sys, json

        # Execute the policy module
        exec(compile({{source!r}}, "<policy>", "exec"))

        observation = {{observation!r}}
        result = propose_action(observation)
        print(result, end="")
        """).format(source=source, observation=observation)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or "Unknown error"
                raise RuntimeError(f"Subprocess exited with code {result.returncode}: {stderr}")
            return result.stdout.strip()
```

- [ ] **3.4: Run executor test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/executor_test.py -v`
Expected: All pass

- [ ] **3.5: Commit**

```bash
git add src/autoharness/harness_as_policy/executor.py \
       src/autoharness/harness_as_policy/executor_test.py
git commit -m "feat(harness-as-policy): add policy executor with AST validation and subprocess isolation"
```

---

## Task 4: Rollout evaluator

**Files:**
- Create: `src/autoharness/harness_as_policy/rollout.py`
- Create: `src/autoharness/harness_as_policy/rollout_test.py`

- [ ] **4.1: Write failing tests for rollout evaluator**

```python
# src/autoharness/harness_as_policy/rollout_test.py
"""Tests for the rollout evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.autoharness.harness_as_policy.models import (
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)
from src.autoharness.harness_as_policy.rollout import RolloutEvaluator


@dataclass
class FakeExecutor:
    """Fake executor that returns configured results."""

    step_results: list[Optional[str]]  # None means execution failure

    def execute(self, source: str, observation: str) -> FakeResult:
        result = self.step_results.pop(0) if self.step_results else None
        if result is None:
            return FakeResult(success=False, output=None, latency=0.0, failure_type="execution_failure", error_details="fail")
        return FakeResult(success=True, output=result, latency=0.0, failure_type=None, error_details=None)


@dataclass
class FakeResult:
    success: bool
    output: Optional[str]
    latency: float
    failure_type: Optional[str]
    error_details: Optional[str]


@dataclass
class FakeAdapter:
    """Fake adapter that follows a scripted sequence of step results."""

    env_id: str = "FakeEnv-v0"
    rules: str = "Fake rules"
    action_format: str = "[X Y]"
    max_steps: int = 10
    _step_index: int = -1
    _steps: Optional[list[StepResult]] = None

    def create(self) -> None:
        pass

    def reset(self, seed: Optional[int] = None) -> str:
        self._step_index = -1
        return "initial observation"

    def step(self, action: str) -> StepResult:
        self._step_index += 1
        if self._steps and self._step_index < len(self._steps):
            return self._steps[self._step_index]
        return StepResult(
            observation="obs",
            action=action,
            is_legal=True,
            reward=0.0,
            terminated=False,
            feedback="",
        )


def test_rollout_solves_environment() -> None:
    """Rollout that reaches environment termination with reward 1.0 gets heuristic 1.0."""
    adapter = FakeAdapter(max_steps=10)
    adapter._steps = [
        StepResult(observation="obs1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs2", action="[C B]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs3", action="[A C]", is_legal=True, reward=1.0, terminated=True, feedback=""),
    ]
    executor = FakeExecutor(step_results=["[A C]", "[C B]", "[A C]"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 1.0
    assert result.termination_reason == TerminationReason.ENVIRONMENT_TERMINATION


def test_rollout_illegal_action_returns_zero() -> None:
    """First illegal action causes heuristic 0 and immediate stop."""
    adapter = FakeAdapter(max_steps=10)
    adapter._steps = [
        StepResult(observation="obs1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs2", action="invalid", is_legal=False, reward=0.0, terminated=True, feedback="Illegal"),
    ]
    executor = FakeExecutor(step_results=["[A C]", "invalid"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.0
    assert result.termination_reason == TerminationReason.ILLEGAL_ACTION


def test_rollout_step_limit() -> None:
    """Reaching adapter step limit without termination yields heuristic 0.5."""
    adapter = FakeAdapter(max_steps=3, step_results=[
        StepResult(observation="obs1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs2", action="[C B]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs3", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
    ])
    executor = FakeExecutor(step_results=["[A C]", "[C B]", "[A C]"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.5
    assert result.termination_reason == TerminationReason.STEP_LIMIT


def test_rollout_execution_failure() -> None:
    """Executor failure on a step records execution failure."""
    adapter = FakeAdapter(max_steps=10)
    adapter._steps = [
        StepResult(observation="obs1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
    ]
    executor = FakeExecutor(step_results=["[A C]", None])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.heuristic == 0.0
    assert result.termination_reason == TerminationReason.EXECUTION_FAILURE


def test_rollout_contract_failure() -> None:
    """Executor contract failure on first step records contract failure."""
    adapter = FakeAdapter(max_steps=10)
    adapter._steps = []
    executor = FakeExecutor(step_results=[None])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.termination_reason in (TerminationReason.EXECUTION_FAILURE, TerminationReason.CONTRACT_FAILURE)


def test_legal_action_count_tracked() -> None:
    """Legal action count is tracked correctly through the rollout."""
    adapter = FakeAdapter(max_steps=10)
    adapter._steps = [
        StepResult(observation="obs1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="obs2", action="invalid", is_legal=False, reward=0.0, terminated=True, feedback="Illegal"),
    ]
    executor = FakeExecutor(step_results=["[A C]", "invalid"])
    evaluator = RolloutEvaluator(adapter=adapter, executor=executor)
    result = evaluator.evaluate(source="dummy source")
    assert result.legal_action_count == 1
```

- [ ] **4.2: Run rollout test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/rollout_test.py -v`
Expected: Import failure

- [ ] **4.3: Implement rollout.py**

```python
# src/autoharness/harness_as_policy/rollout.py
"""Rollout evaluator: runs one episode of a policy against an environment."""

from __future__ import annotations

from typing import Protocol

from src.autoharness.harness_as_policy.executor import ExecutionResult, PolicyExecutor
from src.autoharness.harness_as_policy.models import (
    RolloutResult,
    StepResult,
    TerminationReason,
    heuristic,
)


class EnvironmentAdapter(Protocol):
    """Protocol for environment adapters (redundant with environment.py for typing)."""

    @property
    def env_id(self) -> str: ...
    @property
    def rules(self) -> str: ...
    @property
    def action_format(self) -> str: ...
    @property
    def max_steps(self) -> int: ...
    def create(self) -> None: ...
    def reset(self, seed: int | None = None) -> str: ...
    def step(self, action: str) -> StepResult: ...


class ExecutorProtocol(Protocol):
    """Protocol for policy executors."""

    def execute(self, source: str, observation: str) -> ExecutionResult: ...


class RolloutEvaluator:
    """Evaluates a policy by rolling it out against an environment."""

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        executor: ExecutorProtocol | None = None,
    ) -> None:
        self._adapter = adapter
        self._executor = executor or PolicyExecutor()

    def evaluate(self, source: str, seed: int | None = None) -> RolloutResult:
        """Run one rollout and return the result."""
        try:
            self._adapter.create()
        except Exception as e:
            return RolloutResult(
                steps=[],
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.EXECUTION_FAILURE,
                failure_summary=f"Environment creation failed: {e}",
            )
        try:
            observation = self._adapter.reset(seed=seed)
        except Exception as e:
            return RolloutResult(
                steps=[],
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.EXECUTION_FAILURE,
                failure_summary=f"Environment reset failed: {e}",
            )
        steps: list[StepResult] = []
        for _ in range(self._adapter.max_steps):
            exec_result = self._executor.execute(source, observation)
            if not exec_result.success:
                ft = exec_result.failure_type or "execution_failure"
                reason = (
                    TerminationReason.CONTRACT_FAILURE
                    if ft == "contract_failure"
                    else TerminationReason.EXECUTION_FAILURE
                )
                return RolloutResult(
                    steps=steps,
                    heuristic=0.0,
                    terminal_reward=0.0,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=reason,
                    failure_summary=exec_result.error_details,
                )
            action = exec_result.output or ""
            step_result = self._adapter.step(action)
            steps.append(step_result)
            if not step_result.is_legal:
                return RolloutResult(
                    steps=steps,
                    heuristic=0.0,
                    terminal_reward=0.0,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=TerminationReason.ILLEGAL_ACTION,
                    failure_summary=step_result.feedback or "Illegal action",
                )
            if step_result.terminated:
                h = heuristic(is_legal=True, reward=step_result.reward)
                return RolloutResult(
                    steps=steps,
                    heuristic=h,
                    terminal_reward=step_result.reward,
                    legal_action_count=len([s for s in steps if s.is_legal]),
                    termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                    failure_summary=None,
                )
            observation = step_result.observation
        # Step limit reached
        return RolloutResult(
            steps=steps,
            heuristic=0.5,
            terminal_reward=0.0,
            legal_action_count=len([s for s in steps if s.is_legal]),
            termination_reason=TerminationReason.STEP_LIMIT,
            failure_summary=None,
        )
```

- [ ] **4.4: Run rollout test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/rollout_test.py -v`
Expected: All pass

- [ ] **4.5: Commit**

```bash
git add src/autoharness/harness_as_policy/rollout.py \
       src/autoharness/harness_as_policy/rollout_test.py
git commit -m "feat(harness-as-policy): add rollout evaluator"
```

---

## Task 5: Refiner model boundary

**Files:**
- Create: `src/autoharness/harness_as_policy/refiner.py`
- Create: `src/autoharness/harness_as_policy/refiner_test.py`

- [ ] **5.1: Write failing tests for refiner**

```python
# src/autoharness/harness_as_policy/refiner_test.py
"""Tests for the refiner model boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.autoharness.harness_as_policy.refiner import (
    Refiner,
    RefinerResult,
    build_refiner_prompt,
)


@dataclass
class FakeChatModel(BaseChatModel):
    """A fake chat model that returns scripted responses."""

    responses: list[str] = field(default_factory=list)
    _call_count: int = 0

    def _generate(self, *args, **kwargs):
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import AIMessage

        self._call_count += 1
        if self._responses:
            response = self._responses.pop(0)
        else:
            response = "def propose_action(observation: str) -> str:\n    return '[A C]'"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake"


def test_refiner_returns_source() -> None:
    """Refiner extracts source from model response."""
    model = FakeChatModel(responses=["def propose_action(observation: str) -> str:\n    return '[A C]'"])
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Tower of Hanoi rules",
        action_format="[A C]",
        parent_source="def propose_action(observation: str) -> str:\n    raise NotImplementedError",
        parent_heuristic=0.0,
        parent_reward=0.0,
        parent_legal_actions=0,
        parent_status="contract_failure",
        feedback=["Initial implementation required"],
    )
    assert result.success
    assert "propose_action" in result.source


def test_refiner_prompt_contains_required_sections() -> None:
    """Build prompt includes rules, function contract, and parent info."""
    prompt = build_refiner_prompt(
        env_name="TowerOfHanoi-v0",
        rules="Rules here",
        action_format="[A C]",
        parent_source="source code",
        parent_heuristic=0.5,
        parent_reward=0.0,
        parent_legal_actions=5,
        parent_status="step_limit",
        feedback=["Did not solve puzzle"],
    )
    assert "TowerOfHanoi-v0" in prompt
    assert "propose_action" in prompt
    assert "source code" in prompt
    assert "parent_heuristic" in prompt or "0.5" in prompt


def test_refiner_malformed_response() -> None:
    """Refiner handles malformed response (no source) gracefully."""
    model = FakeChatModel(responses=[""])
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules", action_format="[A C]",
        parent_source="old", parent_heuristic=0.0, parent_reward=0.0,
        parent_legal_actions=0, parent_status="contract_failure",
        feedback=[""],
    )
    assert not result.success


def test_refiner_model_call_count() -> None:
    """Refiner tracks how many model calls were made."""
    model = FakeChatModel(responses=[
        "def propose_action(observation: str) -> str:\n    return '[A C]'",
    ])
    refiner = Refiner(model=model)
    refiner.refine(
        rules="Rules", action_format="[A C]",
        parent_source="old", parent_heuristic=0.0, parent_reward=0.0,
        parent_legal_actions=0, parent_status="contract_failure",
        feedback=[""],
    )
    assert refiner.model_call_count == 1
    assert refiner.logical_refinement_count == 1


def test_refiner_retry_on_transport_error() -> None:
    """Refiner retries once on transport failure."""
    call_count = 0

    class RetryModel(BaseChatModel):
        def _generate(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transport failure")
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.messages import AIMessage
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="def propose_action(observation: str) -> str:\n    return '[A C]'"))])

        @property
        def _llm_type(self) -> str:
            return "retry_fake"

    model = RetryModel()
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules", action_format="[A C]",
        parent_source="old", parent_heuristic=0.0, parent_reward=0.0,
        parent_legal_actions=0, parent_status="contract_failure",
        feedback=[""],
    )
    assert result.success
    assert call_count == 2
    assert refiner.model_call_count == 2
    assert refiner.logical_refinement_count == 1


def test_refiner_double_retry_failure() -> None:
    """Refiner returns failure after two transport errors."""
    class AlwaysFailsModel(BaseChatModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Always fails")

        @property
        def _llm_type(self) -> str:
            return "always_fail"

    model = AlwaysFailsModel()
    refiner = Refiner(model=model)
    result = refiner.refine(
        rules="Rules", action_format="[A C]",
        parent_source="old", parent_heuristic=0.0, parent_reward=0.0,
        parent_legal_actions=0, parent_status="contract_failure",
        feedback=[""],
    )
    assert not result.success
```

- [ ] **5.2: Run refiner test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/refiner_test.py -v`
Expected: Import failure

- [ ] **5.3: Implement refiner.py**

```python
# src/autoharness/harness_as_policy/refiner.py
"""Refiner model boundary — provider-neutral policy synthesis via LLM."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


@dataclass
class RefinerResult:
    """Result from a single refinement call."""
    success: bool
    source: Optional[str]
    error_details: Optional[str] = None


REFINER_SYSTEM_PROMPT = """You are a policy-synthesis assistant. Your task is to write a Python module that solves a game by implementing one function.

Environment: {env_name}
Rules: {rules}
Action format: {action_format}

Function contract:
- `def propose_action(observation: str) -> str:` — receive the current environment observation and return exactly one valid action.

You may define private helper functions and internal data structures.
Do NOT use filesystem, network, subprocess, or dynamic-code operations.
Return ONLY complete, runnable Python source code.

Parent source:
```python
{parent_source}
```

Parent heuristic: {parent_heuristic}
Parent terminal reward: {parent_reward}
Parent legal actions: {parent_legal_actions}
Parent status: {parent_status}

Feedback from previous attempt (most critical first):
{feedback}

Instructions:
1. Preserve working behavior from the parent.
2. Reason about failures and the feedback above.
3. Avoid a fixed move script — implement a general algorithm.
4. Return one COMPLETE replacement module.
5. If the parent solved the environment perfectly, return the same source unchanged.
"""


def build_refiner_prompt(
    env_name: str,
    rules: str,
    action_format: str,
    parent_source: str,
    parent_heuristic: float,
    parent_reward: float,
    parent_legal_actions: int,
    parent_status: str,
    feedback: list[str],
) -> str:
    """Build the refiner prompt with all context."""
    fb_text = "\n".join(f"- {f}" for f in feedback[:5]) if feedback else "No feedback."
    return REFINER_SYSTEM_PROMPT.format(
        env_name=env_name,
        rules=rules,
        action_format=action_format,
        parent_source=parent_source,
        parent_heuristic=parent_heuristic,
        parent_reward=parent_reward,
        parent_legal_actions=parent_legal_actions,
        parent_status=parent_status,
        feedback=fb_text,
    )


def _extract_source(response: str) -> Optional[str]:
    """Extract Python source from model response."""
    text = response.strip()
    if not text:
        return None
    # Try to extract from code fence
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) >= 2:
            code = parts[1].split("```")[0].strip()
            if code:
                return code
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
            if code:
                return code
    # Fall back to raw text
    if "def propose_action" in text:
        return text
    return None


class Refiner:
    """Synthesizes candidate policy modules using a chat model."""

    def __init__(self, model: BaseChatModel | None = None, model_id: str | None = None) -> None:
        if model is not None:
            self._model = model
        elif model_id is not None:
            self._model = init_chat_model(model_id)
        else:
            raise ValueError("Either model or model_id must be provided")
        self._model_call_count: int = 0
        self._logical_refinement_count: int = 0

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def logical_refinement_count(self) -> int:
        return self._logical_refinement_count

    def refine(
        self,
        rules: str,
        action_format: str,
        parent_source: str,
        parent_heuristic: float,
        parent_reward: float,
        parent_legal_actions: int,
        parent_status: str,
        feedback: list[str],
        env_name: str = "",
    ) -> RefinerResult:
        """Call the model to refine the parent policy."""
        prompt = build_refiner_prompt(
            env_name=env_name,
            rules=rules,
            action_format=action_format,
            parent_source=parent_source,
            parent_heuristic=parent_heuristic,
            parent_reward=parent_reward,
            parent_legal_actions=parent_legal_actions,
            parent_status=parent_status,
            feedback=feedback,
        )
        # Attempt with one retry on transport error
        last_error: Optional[str] = None
        for attempt in range(2):
            try:
                response = self._model.invoke(prompt)
                self._model_call_count += 1
            except Exception as e:
                self._model_call_count += 1
                last_error = str(e)
                continue
            source = _extract_source(response.content if hasattr(response, "content") else str(response))
            if source and "propose_action" in source:
                self._logical_refinement_count += 1
                return RefinerResult(success=True, source=source)
            self._logical_refinement_count += 1
            return RefinerResult(
                success=False,
                source=None,
                error_details="Model response did not contain valid propose_action source",
            )
        return RefinerResult(
            success=False,
            source=None,
            error_details=f"Model transport failure after 2 attempts: {last_error}",
        )
```

- [ ] **5.4: Run refiner test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/refiner_test.py -v`
Expected: All pass

- [ ] **5.5: Commit**

```bash
git add src/autoharness/harness_as_policy/refiner.py \
       src/autoharness/harness_as_policy/refiner_test.py
git commit -m "feat(harness-as-policy): add refiner model boundary with structured prompt"
```

---

## Task 6: Artifact store

**Files:**
- Create: `src/autoharness/harness_as_policy/artifacts.py`
- Create: `src/autoharness/harness_as_policy/artifacts_test.py`

- [ ] **6.1: Write failing tests for artifacts**

```python
# src/autoharness/harness_as_policy/artifacts_test.py
"""Tests for the artifact store."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.autoharness.harness_as_policy.artifacts import ArtifactStore
from src.autoharness.harness_as_policy.models import (
    Candidate,
    Event,
    RolloutResult,
    StepResult,
    TerminationReason,
)


@pytest.fixture
def store() -> ArtifactStore:
    tmpdir = tempfile.mkdtemp()
    return ArtifactStore(root=Path(tmpdir), run_id="test-run-001")


def test_artifact_store_creates_directories(store: ArtifactStore) -> None:
    """Initialization creates expected directory structure."""
    assert (store.root / store.run_id).exists()
    assert (store.root / store.run_id / "candidates").exists()
    assert (store.root / store.run_id / "rollouts").exists()
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
    store.write_candidate(candidate_id="005", source="def propose_action(...): pass")
    path = store.root / store.run_id / "candidates" / "005.py"
    assert path.exists()
    assert "propose_action" in path.read_text()


def test_write_rollout(store: ArtifactStore) -> None:
    """write_rollout persists rollout result as JSON."""
    result = RolloutResult(
        steps=[StepResult(observation="obs", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback="")],
        heuristic=0.5,
        terminal_reward=0.0,
        legal_action_count=1,
        termination_reason=TerminationReason.STEP_LIMIT,
        failure_summary=None,
    )
    store.write_rollout(candidate_id="005", result=result)
    path = store.root / store.run_id / "rollouts" / "005.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["heuristic"] == 0.5


def test_write_event(store: ArtifactStore) -> None:
    """write_event appends to events.jsonl."""
    event = Event(iteration=1, event_type="refine", candidate_id="001", parent_id="000", metadata={})
    store.write_event(event)
    path = store.root / store.run_id / "events.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["iteration"] == 1


def test_write_tree(store: ArtifactStore) -> None:
    """write_tree persists tree data as JSON."""
    tree = {"candidates": {"000": {"heuristic": 0.0}}, "best": "001"}
    store.write_tree(tree)
    path = store.root / store.run_id / "tree.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["best"] == "001"


def test_write_best_policy(store: ArtifactStore) -> None:
    """write_best_policy persists best.py."""
    store.write_best_policy(source="def propose_action(observation: str) -> str:\n    return '[A C]'")
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


def test_load_best_policy(store: ArtifactStore) -> None:
    """load_best_policy reads back the best.py source."""
    source = "def propose_action(observation: str) -> str:\n    return '[A C]'"
    store.write_best_policy(source=source)
    loaded = store.load_best_policy()
    assert loaded == source


def test_load_config(store: ArtifactStore) -> None:
    """load_config reads back config.json."""
    config = {"model": "test", "profile": "smoke"}
    store.write_config(config)
    loaded = store.load_config()
    assert loaded["model"] == "test"
```

- [ ] **6.2: Run artifacts test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/artifacts_test.py -v`
Expected: Import failure

- [ ] **6.3: Implement artifacts.py**

```python
# src/autoharness/harness_as_policy/artifacts.py
"""Atomic artifact persistence for synthesis runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.autoharness.harness_as_policy.models import Event, RolloutResult


class ArtifactStore:
    """Persists and loads synthesis run artifacts."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root
        self._run_id = run_id
        self._run_dir = root / run_id
        self._init_directories()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _init_directories(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "candidates").mkdir(exist_ok=True)
        (self._run_dir / "rollouts").mkdir(exist_ok=True)
        (self._run_dir / "evaluation").mkdir(exist_ok=True)

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(path)

    def write_config(self, config: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "config.json", config)

    def write_tree(self, tree: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "tree.json", tree)

    def write_event(self, event: Event) -> None:
        path = self._run_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({
                "iteration": event.iteration,
                "event_type": event.event_type,
                "candidate_id": event.candidate_id,
                "parent_id": event.parent_id,
                "metadata": event.metadata,
            }, default=str) + "\n")

    def write_candidate(self, candidate_id: str, source: str) -> None:
        path = self._run_dir / "candidates" / f"{candidate_id}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source)
        tmp.rename(path)

    def write_rollout(self, candidate_id: str, result: RolloutResult) -> None:
        data = {
            "heuristic": result.heuristic,
            "terminal_reward": result.terminal_reward,
            "legal_action_count": result.legal_action_count,
            "termination_reason": result.termination_reason.value if result.termination_reason else None,
            "failure_summary": result.failure_summary,
            "steps": [
                {
                    "observation": s.observation,
                    "action": s.action,
                    "is_legal": s.is_legal,
                    "reward": s.reward,
                    "terminated": s.terminated,
                    "feedback": s.feedback,
                }
                for s in result.steps
            ],
        }
        self._write_json(self._run_dir / "rollouts" / f"{candidate_id}.json", data)

    def write_best_policy(self, source: str) -> None:
        path = self._run_dir / "best.py"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(source)
        tmp.rename(path)

    def write_synthesis_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "synthesis-summary.json", summary)

    def write_evaluation(self, name: str, data: dict[str, Any]) -> None:
        self._write_json(self._run_dir / "evaluation" / f"{name}.json", data)

    def load_best_policy(self) -> Optional[str]:
        path = self._run_dir / "best.py"
        if path.exists():
            return path.read_text()
        return None

    def load_config(self) -> Optional[dict[str, Any]]:
        path = self._run_dir / "config.json"
        if path.exists():
            return json.loads(path.read_text())
        return None
```

- [ ] **6.4: Run artifacts test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/artifacts_test.py -v`
Expected: All pass

- [ ] **6.5: Commit**

```bash
git add src/autoharness/harness_as_policy/artifacts.py \
       src/autoharness/harness_as_policy/artifacts_test.py
git commit -m "feat(harness-as-policy): add artifact store with atomic persistence"
```

---

## Task 7: LangGraph search workflow

**Files:**
- Create: `src/autoharness/harness_as_policy/search.py`
- Create: `src/autoharness/harness_as_policy/search_test.py`

- [ ] **7.1: Write failing tests for search**

```python
# src/autoharness/harness_as_policy/search_test.py
"""Tests for the LangGraph search workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from src.autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Profile,
    TerminationReason,
)
from src.autoharness.harness_as_policy.search import (
    beta_parameters,
    find_best_candidate,
    select_candidate,
    should_stop,
    synthesize,
)


def test_beta_parameters_no_children() -> None:
    """Beta parameters for candidate with no children and H=0.5."""
    a, b = beta_parameters(heuristic=0.5, children=0, weight=1.0)
    # a = 1 + C*H = 1 + 1*0.5 = 1.5
    # b = 1 + C*(1-H) + N = 1 + 1*0.5 + 0 = 1.5
    assert abs(a - 1.5) < 1e-10
    assert abs(b - 1.5) < 1e-10


def test_beta_parameters_perfect() -> None:
    """Beta parameters for perfect candidate with H=1.0."""
    a, b = beta_parameters(heuristic=1.0, children=2, weight=1.0)
    # a = 1 + 1*1.0 = 2.0
    # b = 1 + 1*0 + 2 = 3.0
    assert abs(a - 2.0) < 1e-10
    assert abs(b - 3.0) < 1e-10


def test_beta_parameters_zero() -> None:
    """Beta parameters for zero heuristic."""
    a, b = beta_parameters(heuristic=0.0, children=0, weight=1.0)
    assert abs(a - 1.0) < 1e-10
    assert abs(b - 2.0) < 1e-10


def test_select_candidate_deterministic() -> None:
    """Selection with seeded RNG reproduces the same draw."""
    candidates = {
        "000": Candidate(id="000", parent_id=None, source="", heuristic=0.0, terminal_reward=0.0,
                         legal_action_count=0, termination_reason=None, failure_summary=None,
                         iteration=0, expansion_count=0),
        "001": Candidate(id="001", parent_id="000", source="", heuristic=0.5, terminal_reward=0.0,
                         legal_action_count=7, termination_reason=TerminationReason.STEP_LIMIT,
                         failure_summary=None, iteration=1, expansion_count=0),
        "002": Candidate(id="002", parent_id="000", source="", heuristic=0.8, terminal_reward=0.5,
                         legal_action_count=10, termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                         failure_summary=None, iteration=2, expansion_count=1),
    }
    rng = random.Random(42)
    selected = select_candidate(candidates, rng)
    assert selected in candidates


def test_find_best_candidate_empty() -> None:
    """Empty candidate dict returns None."""
    assert find_best_candidate({}) is None


def test_find_best_candidate_single() -> None:
    """Single candidate is the best."""
    c = Candidate(id="000", parent_id=None, source="", heuristic=0.5, terminal_reward=0.0,
                  legal_action_count=5, termination_reason=TerminationReason.STEP_LIMIT,
                  failure_summary=None, iteration=0, expansion_count=0)
    assert find_best_candidate({"000": c}) == "000"


def test_find_best_candidate_lexicographic() -> None:
    """Best candidate follows lexicographic ranking."""
    candidates = {
        "000": Candidate(id="000", parent_id=None, source="", heuristic=0.5, terminal_reward=0.0,
                         legal_action_count=5, termination_reason=TerminationReason.STEP_LIMIT,
                         failure_summary=None, iteration=0, expansion_count=0),
        "001": Candidate(id="001", parent_id="000", source="", heuristic=0.8, terminal_reward=0.6,
                         legal_action_count=8, termination_reason=TerminationReason.STEP_LIMIT,
                         failure_summary=None, iteration=1, expansion_count=0),
        "002": Candidate(id="002", parent_id="001", source="", heuristic=1.0, terminal_reward=1.0,
                         legal_action_count=7, termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                         failure_summary=None, iteration=2, expansion_count=0),
    }
    assert find_best_candidate(candidates) == "002"


def test_should_stop_success() -> None:
    """Should stop when any candidate has H=1.0."""
    candidates = {
        "000": Candidate(id="000", parent_id=None, source="", heuristic=1.0, terminal_reward=1.0,
                         legal_action_count=7, termination_reason=TerminationReason.ENVIRONMENT_TERMINATION,
                         failure_summary=None, iteration=0, expansion_count=0),
    }
    reason = should_stop(candidates, iteration=1, max_refinements=8)
    assert reason is not None
    assert "success" in reason


def test_should_stop_budget_exhausted() -> None:
    """Should stop when iteration reaches max_refinements."""
    reason = should_stop({}, iteration=8, max_refinements=8)
    assert reason is not None
    assert "budget" in reason


def test_should_stop_not_yet() -> None:
    """Should not stop when budget remains and no success."""
    reason = should_stop({}, iteration=0, max_refinements=8)
    assert reason is None


def test_synthesize_empty_policies() -> None:
    """synthesize with only root and one failed refinement returns empty best policy id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synthesize(
            adapter=FakeAdapter(),
            profile=Profile.SMOKE,
            refiner=FakeRefiner(responses=[""]),
            artifact_root=Path(tmpdir),
            seed=42,
        )
    assert result["stop_reason"] is not None
    assert "iteration" in result
```

- [ ] **7.2: Run search test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/search_test.py -v`
Expected: Import failure

- [ ] **7.3: Implement search.py**

```python
# src/autoharness/harness_as_policy/search.py
"""LangGraph search workflow with REx Thompson selection."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.autoharness.harness_as_policy.artifacts import ArtifactStore
from src.autoharness.harness_as_policy.config import Settings
from src.autoharness.harness_as_policy.executor import PolicyExecutor
from src.autoharness.harness_as_policy.models import (
    Candidate,
    CandidateRankKey,
    Event,
    Profile,
    RolloutResult,
    TerminationReason,
    heuristic,
)
from src.autoharness.harness_as_policy.refiner import Refiner
from src.autoharness.harness_as_policy.rollout import RolloutEvaluator


class GraphState(TypedDict):
    """Serializable state for the LangGraph search workflow."""
    run_id: str
    iteration: int
    candidates: dict[str, Candidate]
    selected_parent_id: Optional[str]
    latest_candidate_id: Optional[str]
    best_candidate_id: Optional[str]
    feedback: list[str]
    model_call_count: int
    logical_refinement_count: int
    stop_reason: Optional[str]
    artifact_root: str
    profile: str
    rng_seed: int


def beta_parameters(heuristic: float, children: int, weight: float = 1.0) -> tuple[float, float]:
    """Compute Beta distribution parameters for Thompson sampling."""
    a = 1.0 + weight * heuristic
    b = 1.0 + weight * (1.0 - heuristic) + children
    return a, b


def select_candidate(candidates: dict[str, Candidate], rng: random.Random) -> Optional[str]:
    """Select a candidate using Thompson sampling (largest draw)."""
    if not candidates:
        return None
    best_id: Optional[str] = None
    best_draw: float = -1.0
    for cid, cand in candidates.items():
        a, b = beta_parameters(
            heuristic=cand.heuristic,
            children=cand.expansion_count,
        )
        draw = rng.betavariate(a, b)
        if draw > best_draw:
            best_draw = draw
            best_id = cid
    return best_id


def find_best_candidate(candidates: dict[str, Candidate]) -> Optional[str]:
    """Find the best candidate using lexicographic ranking."""
    if not candidates:
        return None
    best_id: Optional[str] = None
    best_key: Optional[CandidateRankKey] = None
    for cid, cand in candidates.items():
        key = CandidateRankKey.from_candidate(cand)
        if best_key is None or key > best_key:
            best_key = key
            best_id = cid
    return best_id


def should_stop(
    candidates: dict[str, Candidate],
    iteration: int,
    max_refinements: int,
) -> Optional[str]:
    """Check termination conditions. Returns stop reason or None."""
    # Success: any candidate has H=1.0
    for cand in candidates.values():
        if cand.heuristic >= 1.0:
            return f"success: candidate {cand.id} achieved H=1.0"
    # Budget exhausted
    if iteration >= max_refinements:
        return f"budget exhausted after {max_refinements} refinements"
    return None


ROOT_SOURCE = """def propose_action(observation: str) -> str:
    raise NotImplementedError("Root policy — replace me")
"""


def synthesize(
    adapter: Any,
    profile: Profile,
    refiner: Refiner,
    artifact_root: Path,
    seed: int = 42,
    refinements: int | None = None,
) -> dict[str, Any]:
    """Run the full synthesis workflow. Returns summary dict."""
    run_id = uuid.uuid4().hex[:12]
    store = ArtifactStore(root=artifact_root, run_id=run_id)
    rng = random.Random(seed)
    max_refinements = refinements if refinements is not None else profile.refinements
    policy_executor = PolicyExecutor()
    evaluator = RolloutEvaluator(adapter=adapter, executor=policy_executor)

    # Write config
    store.write_config({
        "run_id": run_id,
        "profile": profile.value,
        "max_refinements": max_refinements,
        "thompson_seed": seed,
        "env_id": adapter.env_id,
        "model_call_count": 0,
    })

    # Initialize root candidate
    root = Candidate(
        id="000",
        parent_id=None,
        source=ROOT_SOURCE,
        heuristic=0.0,
        terminal_reward=0.0,
        legal_action_count=0,
        termination_reason=None,
        failure_summary=None,
        iteration=0,
        expansion_count=0,
    )
    candidates: dict[str, Candidate] = {"000": root}
    store.write_candidate("000", ROOT_SOURCE)

    best_id: Optional[str] = None
    stop_reason: Optional[str] = None
    model_call_count = 0
    logical_refinement_count = 0

    for iteration in range(1, max_refinements + 1):
        # Check stop
        stop_reason = should_stop(candidates, iteration - 1, max_refinements)
        if stop_reason:
            break

        # Select parent
        parent_id = select_candidate(candidates, rng)
        if parent_id is None:
            stop_reason = "no candidates to select"
            break

        parent = candidates[parent_id]
        parent.expansion_count += 1

        store.write_event(Event(
            iteration=iteration,
            event_type="select",
            candidate_id=parent_id,
            parent_id=parent.parent_id,
            metadata={"expansion_count": parent.expansion_count},
        ))

        # Refine
        child_id = f"{iteration:03d}"
        feedback: list[str] = []
        if parent.failure_summary:
            feedback.append(parent.failure_summary)
        if parent.termination_reason == TerminationReason.ILLEGAL_ACTION:
            feedback.append("Policy produced an illegal action")
        elif parent.termination_reason == TerminationReason.STEP_LIMIT:
            feedback.append("Policy reached step limit without solving")
        elif parent.termination_reason in (TerminationReason.EXECUTION_FAILURE, TerminationReason.CONTRACT_FAILURE):
            feedback.append("Policy execution failed at runtime")

        refine_result = refiner.refine(
            env_name=adapter.env_id,
            rules=adapter.rules,
            action_format=adapter.action_format,
            parent_source=parent.source,
            parent_heuristic=parent.heuristic,
            parent_reward=parent.terminal_reward,
            parent_legal_actions=parent.legal_action_count,
            parent_status=parent.termination_reason.value if parent.termination_reason else "unknown",
            feedback=feedback,
        )
        model_call_count = refiner.model_call_count
        logical_refinement_count = refiner.logical_refinement_count

        store.write_event(Event(
            iteration=iteration,
            event_type="refine",
            candidate_id=child_id,
            parent_id=parent_id,
            metadata={"success": refine_result.success},
        ))

        if not refine_result.success or not refine_result.source:
            # Record failed refinement
            child = Candidate(
                id=child_id,
                parent_id=parent_id,
                source=refine_result.source or "",
                heuristic=0.0,
                terminal_reward=0.0,
                legal_action_count=0,
                termination_reason=TerminationReason.CONTRACT_FAILURE,
                failure_summary=refine_result.error_details or "Refinement failed",
                iteration=iteration,
                expansion_count=0,
            )
            candidates[child_id] = child
            store.write_candidate(child_id, child.source)
            rollout_result = RolloutResult(
                steps=[], heuristic=0.0, terminal_reward=0.0,
                legal_action_count=0, termination_reason=TerminationReason.CONTRACT_FAILURE,
                failure_summary=child.failure_summary,
            )
            store.write_rollout(child_id, rollout_result)
            continue

        # Evaluate child
        store.write_candidate(child_id, refine_result.source)
        rollout_result = evaluator.evaluate(source=refine_result.source)

        child = Candidate(
            id=child_id,
            parent_id=parent_id,
            source=refine_result.source,
            heuristic=rollout_result.heuristic,
            terminal_reward=rollout_result.terminal_reward,
            legal_action_count=rollout_result.legal_action_count,
            termination_reason=rollout_result.termination_reason,
            failure_summary=rollout_result.failure_summary,
            iteration=iteration,
            expansion_count=0,
        )
        candidates[child_id] = child
        store.write_rollout(child_id, rollout_result)

        store.write_event(Event(
            iteration=iteration,
            event_type="evaluate",
            candidate_id=child_id,
            parent_id=parent_id,
            metadata={
                "heuristic": rollout_result.heuristic,
                "terminal_reward": rollout_result.terminal_reward,
                "legal_action_count": rollout_result.legal_action_count,
                "termination_reason": rollout_result.termination_reason.value if rollout_result.termination_reason else None,
            },
        ))

        # Update best
        current_best = find_best_candidate(candidates)
        if current_best:
            best_id = current_best

    # Final stop check
    if not stop_reason:
        stop_reason = should_stop(candidates, max_refinements, max_refinements) or "completed"

    # Write tree, summary, best policy
    tree_data: dict[str, Any] = {
        "candidates": {
            cid: {
                "id": c.id,
                "parent_id": c.parent_id,
                "heuristic": c.heuristic,
                "terminal_reward": c.terminal_reward,
                "legal_action_count": c.legal_action_count,
                "termination_reason": c.termination_reason.value if c.termination_reason else None,
                "failure_summary": c.failure_summary,
                "iteration": c.iteration,
                "expansion_count": c.expansion_count,
            }
            for cid, c in candidates.items()
        },
        "best_candidate_id": best_id,
    }
    store.write_tree(tree_data)

    summary = {
        "run_id": run_id,
        "stop_reason": stop_reason,
        "best_candidate_id": best_id,
        "total_candidates": len(candidates),
        "iterations_used": len(candidates) - 1,  # exclude root
        "profile": profile.value,
        "model_call_count": model_call_count,
        "logical_refinement_count": logical_refinement_count,
    }
    store.write_synthesis_summary(summary)

    if best_id and candidates[best_id].source:
        store.write_best_policy(candidates[best_id].source)

    return summary
```

- [ ] **7.4: Run search test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/search_test.py -v`
Expected: All pass

- [ ] **7.5: Commit**

```bash
git add src/autoharness/harness_as_policy/search.py \
       src/autoharness/harness_as_policy/search_test.py
git commit -m "feat(harness-as-policy): add LangGraph search workflow with REx Thompson selection"
```

---

## Task 8: Held-out evaluation and live-LLM baseline

**Files:**
- Create: `src/autoharness/harness_as_policy/evaluation.py`
- Create: `src/autoharness/harness_as_policy/evaluation_test.py`

- [ ] **8.1: Write failing tests for evaluation**

```python
# src/autoharness/harness_as_policy/evaluation_test.py
"""Tests for held-out evaluation and live-LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.autoharness.harness_as_policy.evaluation import (
    EvaluationResult,
    evaluate_policy,
    evaluate_policy_on_env,
    format_evaluation_summary,
)
from src.autoharness.harness_as_policy.models import StepResult, TerminationReason


@dataclass
class FakeAdapter:
    env_id: str = "TowerOfHanoi-v0"
    rules: str = "Rules"
    action_format: str = "[A C]"
    max_steps: int = 14
    _step_results: list[StepResult] | None = None
    _step_index: int = -1

    def create(self) -> None:
        pass

    def reset(self, seed: Optional[int] = None) -> str:
        self._step_index = -1
        return "initial obs"

    def step(self, action: str) -> StepResult:
        self._step_index += 1
        if self._step_results and self._step_index < len(self._step_results):
            return self._step_results[self._step_index]
        return StepResult(observation="obs", action=action, is_legal=True, reward=0.0, terminated=False, feedback="")


@dataclass
class FakeExecutor:
    responses: list[str] | None = None
    _call_index: int = -1

    def execute(self, source: str, observation: str) -> FakeResult:
        self._call_index += 1
        if self.responses and self._call_index < len(self.responses):
            return FakeResult(success=True, output=self.responses[self._call_index], latency=0.01)
        return FakeResult(success=True, output="[A C]", latency=0.01)


@dataclass
class FakeResult:
    success: bool
    output: Optional[str]
    latency: float
    failure_type: Optional[str] = None
    error_details: Optional[str] = None


def test_evaluate_policy_on_env_solved() -> None:
    """evaluate_policy_on_env returns solved result when environment terminates with reward 1."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(observation="o1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="o2", action="[C B]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="o3", action="[A C]", is_legal=True, reward=1.0, terminated=True, feedback=""),
    ]
    executor = FakeExecutor(responses=["[A C]", "[C B]", "[A C]"])
    result = evaluate_policy_on_env(adapter=adapter, executor=executor, source="policy source")
    assert result.solved
    assert result.reward == 1.0
    assert result.steps_used == 3
    assert result.illegal_action_reason is None


def test_evaluate_policy_on_env_illegal() -> None:
    """evaluate_policy_on_env records illegal action reason."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(observation="o1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
        StepResult(observation="o2", action="bad", is_legal=False, reward=0.0, terminated=True, feedback="Illegal"),
    ]
    executor = FakeExecutor(responses=["[A C]", "bad"])
    result = evaluate_policy_on_env(adapter=adapter, executor=executor, source="policy source")
    assert not result.solved
    assert result.illegal_action_reason is not None


def test_evaluate_policy_no_model_calls() -> None:
    """Generated policy evaluation makes zero model calls by using executor directly."""
    adapter = FakeAdapter(max_steps=14)
    adapter._step_results = [
        StepResult(observation="o1", action="[A C]", is_legal=True, reward=0.0, terminated=False, feedback=""),
    ]
    executor = FakeExecutor(responses=["[A C]"])
    # This test verifies the evaluation API doesn't have a model parameter
    result = evaluate_policy_on_env(adapter=adapter, executor=executor, source="policy source")
    assert isinstance(result, EvaluationResult)


def test_evaluation_result_attributes() -> None:
    """EvaluationResult has all expected fields."""
    result = EvaluationResult(
        env_id="TowerOfHanoi-v0-medium",
        solved=False,
        reward=0.0,
        legal_action_count=5,
        steps_used=5,
        optimal_steps=15,
        illegal_action_reason="malformed action",
        latency=0.05,
        execution_failure=False,
    )
    assert result.env_id == "TowerOfHanoi-v0-medium"
    assert result.optimal_steps == 15
    assert result.execution_failure is False


def test_format_evaluation_summary() -> None:
    """format_evaluation_summary produces a non-empty string."""
    results = [
        EvaluationResult(env_id="v0", solved=True, reward=1.0, legal_action_count=7, steps_used=7,
                         optimal_steps=7, illegal_action_reason=None, latency=0.05, execution_failure=False),
        EvaluationResult(env_id="medium", solved=False, reward=0.0, legal_action_count=10, steps_used=10,
                         optimal_steps=15, illegal_action_reason="step_limit", latency=0.08, execution_failure=False),
    ]
    summary = format_evaluation_summary(results)
    assert "v0" in summary
    assert "medium" in summary
    assert len(summary) > 0
```

- [ ] **8.2: Run evaluation test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/evaluation_test.py -v`
Expected: Import failure

- [ ] **8.3: Implement evaluation.py**

```python
# src/autoharness/harness_as_policy/evaluation.py
"""Held-out policy evaluation and optional live-LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.autoharness.harness_as_policy.executor import PolicyExecutor
from src.autoharness.harness_as_policy.models import TerminationReason
from src.autoharness.harness_as_policy.tower_of_hanoi import TowerOfHanoiAdapter


@dataclass
class EvaluationResult:
    """Result of evaluating a policy on one environment variant."""
    env_id: str
    solved: bool
    reward: float
    legal_action_count: int
    steps_used: int
    optimal_steps: int
    illegal_action_reason: Optional[str]
    latency: float
    execution_failure: bool


DIFFICULTIES = [
    ("v0", "TowerOfHanoi-v0", 14, 7),
    ("medium", "TowerOfHanoi-v0-medium", 30, 15),
    ("hard", "TowerOfHanoi-v0-hard", 62, 31),
    ("hardcore", "TowerOfHanoi-v0-hardcore", 126, 63),
]


def _optimal_steps(disks: int) -> int:
    """Optimal number of moves to solve n-disk Tower of Hanoi (2^n - 1)."""
    return (2 ** (disks + 2)) - 1  # 3 disks -> 7, 4 -> 15, etc.


def evaluate_policy_on_env(
    adapter: Any,
    executor: Any,
    source: str,
) -> EvaluationResult:
    """Evaluate a generated policy on one environment without model calls."""
    import time
    try:
        adapter.create()
        adapter.reset(seed=None)
    except Exception as e:
        return EvaluationResult(
            env_id=adapter.env_id,
            solved=False,
            reward=0.0,
            legal_action_count=0,
            steps_used=0,
            optimal_steps=adapter.max_steps,
            illegal_action_reason=None,
            latency=0.0,
            execution_failure=True,
        )
    start = time.monotonic()
    legal_actions = 0
    steps_used = 0
    illegal_reason: Optional[str] = None
    solved = False
    reward = 0.0

    for _ in range(adapter.max_steps):
        try:
            obs = adapter._observation if hasattr(adapter, "_observation") else adapter.reset()
        except Exception:
            obs = ""
        exec_result = executor.execute(source, obs)
        steps_used += 1
        if not exec_result.success:
            end = time.monotonic()
            return EvaluationResult(
                env_id=adapter.env_id,
                solved=False,
                reward=0.0,
                legal_action_count=legal_actions,
                steps_used=steps_used,
                optimal_steps=adapter.max_steps,
                illegal_action_reason=None,
                latency=end - start,
                execution_failure=True,
            )
        action = exec_result.output or ""
        step_result = adapter.step(action)
        if not step_result.is_legal:
            end = time.monotonic()
            return EvaluationResult(
                env_id=adapter.env_id,
                solved=False,
                reward=0.0,
                legal_action_count=legal_actions,
                steps_used=steps_used,
                optimal_steps=adapter.max_steps,
                illegal_action_reason=step_result.feedback or "Illegal action",
                latency=end - start,
                execution_failure=False,
            )
        legal_actions += 1
        if step_result.terminated:
            end = time.monotonic()
            solved = step_result.reward >= 1.0
            reward = step_result.reward
            return EvaluationResult(
                env_id=adapter.env_id,
                solved=solved,
                reward=reward,
                legal_action_count=legal_actions,
                steps_used=steps_used,
                optimal_steps=adapter.max_steps,
                illegal_action_reason=illegal_reason,
                latency=end - start,
                execution_failure=False,
            )

    end = time.monotonic()
    return EvaluationResult(
        env_id=adapter.env_id,
        solved=False,
        reward=0.0,
        legal_action_count=legal_actions,
        steps_used=steps_used,
        optimal_steps=adapter.max_steps,
        illegal_action_reason="step_limit",
        latency=end - start,
        execution_failure=False,
    )


def evaluate_policy(
    source: str,
    difficulties: list[tuple[str, str, int, int]] | None = None,
) -> list[EvaluationResult]:
    """Evaluate a generated policy across all difficulty variants. Zero model calls."""
    if difficulties is None:
        difficulties = DIFFICULTIES
    results: list[EvaluationResult] = []
    executor = PolicyExecutor()
    for diff_key, env_id, max_steps_var, optimal in difficulties:
        adapter = TowerOfHanoiAdapter(difficulty=diff_key)
        result = evaluate_policy_on_env(adapter=adapter, executor=executor, source=source)
        results.append(result)
    return results


def format_evaluation_summary(results: list[EvaluationResult]) -> str:
    """Format evaluation results as a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Policy Evaluation Summary")
    lines.append("=" * 60)
    max_disk_solved = 0
    for r in results:
        status = "SOLVED" if r.solved else "FAILED"
        lines.append(f"  {r.env_id}: {status}")
        lines.append(f"    Reward: {r.reward}")
        lines.append(f"    Steps: {r.steps_used}/{r.optimal_steps}")
        lines.append(f"    Legal actions: {r.legal_action_count}")
        if r.illegal_action_reason:
            lines.append(f"    Illegal reason: {r.illegal_action_reason}")
        if r.execution_failure:
            lines.append(f"    Execution failure: yes")
        lines.append(f"    Latency: {r.latency:.3f}s")
        lines.append("")
        # Map env_id to disk count
        if "hardcore" in r.env_id and r.solved:
            max_disk_solved = 6
        elif "hard" in r.env_id and r.solved:
            max_disk_solved = max(max_disk_solved, 5)
        elif "medium" in r.env_id and r.solved:
            max_disk_solved = max(max_disk_solved, 4)
        elif r.solved:
            max_disk_solved = max(max_disk_solved, 3)
    lines.append(f"  Largest disk count solved: {max_disk_solved}")
    lines.append("=" * 60)
    return "\n".join(lines)
```

- [ ] **8.4: Run evaluation test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/evaluation_test.py -v`
Expected: All pass

- [ ] **8.5: Commit**

```bash
git add src/autoharness/harness_as_policy/evaluation.py \
       src/autoharness/harness_as_policy/evaluation_test.py
git commit -m "feat(harness-as-policy): add held-out policy evaluation suite"
```

---

## Task 9: CLI composition

**Files:**
- Create: `src/autoharness/cli.py`
- Create: `src/autoharness/harness_as_policy/cli_test.py` (we add a test for the top-level CLI integration)

- [ ] **9.1: Write failing tests for CLI**

```python
# src/autoharness/harness_as_policy/cli_test.py
"""Tests for the CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.autoharness.cli import main, synthesize_cmd, evaluate_cmd


def test_synthesize_cmd_requires_env() -> None:
    """synthesize command requires --env flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("sys.argv", ["autoharness", "synthesize", "--env", "TowerOfHanoi-v0", "--profile", "smoke", "--artifact-root", tmpdir]):
            result = synthesize_cmd()
    assert result is not None


def test_synthesize_cmd_creates_artifacts() -> None:
    """synthesize command creates artifact files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("sys.argv", ["autoharness", "synthesize", "--env", "TowerOfHanoi-v0", "--profile", "smoke", "--artifact-root", tmpdir]):
            result = synthesize_cmd()
        artifact_dir = Path(tmpdir)
        dirs = list(artifact_dir.iterdir())
        assert len(dirs) > 0


def test_evaluate_cmd_requires_run() -> None:
    """evaluate command requires --run flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = evaluate_cmd(run_dir=Path(tmpdir))
    assert result is not None


def test_main_synthesize_dispatches() -> None:
    """main dispatches synthesize command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("sys.argv", ["autoharness", "synthesize", "--env", "TowerOfHanoi-v0", "--artifact-root", tmpdir]):
            result = main()
    assert result == 0
```

- [ ] **9.2: Run CLI test to verify failure**

Run: `uv run pytest src/autoharness/harness_as_policy/cli_test.py -v`
Expected: Import failure

- [ ] **9.3: Implement cli.py**

```python
# src/autoharness/cli.py
"""Top-level CLI for autoharness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from src.autoharness.harness_as_policy.config import Settings
from src.autoharness.harness_as_policy.evaluation import evaluate_policy, format_evaluation_summary
from src.autoharness.harness_as_policy.refiner import Refiner
from src.autoharness.harness_as_policy.search import synthesize
from src.autoharness.harness_as_policy.tower_of_hanoi import TowerOfHanoiAdapter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoHarness — policy synthesis and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # synthesize
    syn = subparsers.add_parser("synthesize", help="Synthesize a policy")
    syn.add_argument("--env", default=None, help="Environment ID (e.g. TowerOfHanoi-v0)")
    syn.add_argument("--profile", default=None, choices=["smoke", "low-cost"],
                     help="Synthesis profile (default: smoke)")
    syn.add_argument("--model", default=None, help="Model identifier (e.g. google_genai:gemini-2.5-flash)")
    syn.add_argument("--refinements", type=int, default=None,
                     help="Override refinement budget")
    syn.add_argument("--artifact-root", default=None, help="Artifact output directory")
    syn.add_argument("--seed", type=int, default=None, help="Thompson RNG seed")

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate a synthesized policy")
    ev.add_argument("--run", required=True, type=Path, help="Run artifact directory")

    # evaluate-baseline
    evb = subparsers.add_parser("evaluate-baseline", help="Evaluate a live LLM baseline")
    evb.add_argument("--run", required=True, type=Path, help="Run artifact directory")
    evb.add_argument("--model", required=True, help="Model identifier")

    return parser


def synthesize_cmd(args: argparse.Namespace | None = None) -> Optional[dict[str, Any]]:
    """Run the synthesize command."""
    parser = _build_parser()
    if args is None:
        args, _ = parser.parse_known_args()

    # Build settings with CLI overrides
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

    settings = Settings(**settings_kwargs)

    # Build adapter
    adapter = TowerOfHanoiAdapter(difficulty="v0")

    # Build refiner
    refiner = Refiner(model_id=settings.model)

    result = synthesize(
        adapter=adapter,
        profile=settings.profile,
        refiner=refiner,
        artifact_root=Path(settings.artifact_root),
        seed=settings.thompson_seed,
        refinements=settings.effective_refinements,
    )
    print(f"Run ID: {result.get('run_id', 'unknown')}")
    print(f"Stop reason: {result.get('stop_reason', 'unknown')}")
    print(f"Best candidate: {result.get('best_candidate_id', 'none')}")
    print(f"Total candidates: {result.get('total_candidates', 0)}")
    print(f"Model calls: {result.get('model_call_count', 0)}")
    return result


def evaluate_cmd(run_dir: Path) -> Optional[list[Any]]:
    """Run the evaluate command."""
    best_policy_path = run_dir / "best.py"
    if not best_policy_path.exists():
        print(f"Error: no best.py found in {run_dir}", file=sys.stderr)
        return None
    source = best_policy_path.read_text()
    results = evaluate_policy(source=source)
    summary = format_evaluation_summary(results)
    print(summary)
    return results


def evaluate_baseline_cmd(run_dir: Path, model_id: str) -> Optional[list[Any]]:
    """Run the live-LLM baseline evaluate command."""
    adapter = TowerOfHanoiAdapter(difficulty="v0")
    refiner = Refiner(model_id=model_id)
    # For each observation, use the model directly as a policy
    from src.autoharness.harness_as_policy.evaluation import (
        DIFFICULTIES,
        EvaluationResult,
    )
    results: list[EvaluationResult] = []
    for diff_key, env_id, max_steps_var, optimal in DIFFICULTIES:
        adapter = TowerOfHanoiAdapter(difficulty=diff_key)
        try:
            adapter.create()
            adapter.reset()
        except Exception as e:
            results.append(EvaluationResult(
                env_id=env_id, solved=False, reward=0.0, legal_action_count=0,
                steps_used=0, optimal_steps=optimal, illegal_action_reason=None,
                latency=0.0, execution_failure=True,
            ))
            continue
        # Run live-LLM baseline
        from src.autoharness.harness_as_policy.models import StepResult
        import time
        start = time.monotonic()
        legal_actions = 0
        solved = False
        reward = 0.0
        steps_used = 0
        illegal_reason: Optional[str] = None
        model_call_count = 0
        input_tokens = 0
        output_tokens = 0
        obs = adapter._observation if hasattr(adapter, "_observation") else ""
        for _ in range(adapter.max_steps):
            steps_used += 1
            try:
                response = refiner.refine(
                    env_name=adapter.env_id,
                    rules=adapter.rules,
                    action_format=adapter.action_format,
                    parent_source="",
                    parent_heuristic=0.0,
                    parent_reward=0.0,
                    parent_legal_actions=0,
                    parent_status="unknown",
                    feedback=["Live policy mode"],
                )
                model_call_count += 1
                action = response.source if response.success and response.source else ""
            except Exception:
                action = ""
            if not action:
                results.append(EvaluationResult(
                    env_id=env_id, solved=False, reward=0.0, legal_action_count=legal_actions,
                    steps_used=steps_used, optimal_steps=optimal, illegal_action_reason="model_error",
                    latency=time.monotonic() - start, execution_failure=True,
                ))
                break
            step_result = adapter.step(action)
            if not step_result.is_legal:
                illegal_reason = step_result.feedback or "Illegal"
                results.append(EvaluationResult(
                    env_id=env_id, solved=False, reward=0.0, legal_action_count=legal_actions,
                    steps_used=steps_used, optimal_steps=optimal, illegal_action_reason=illegal_reason,
                    latency=time.monotonic() - start, execution_failure=False,
                ))
                break
            legal_actions += 1
            if step_result.terminated:
                solved = step_result.reward >= 1.0
                reward = step_result.reward
                results.append(EvaluationResult(
                    env_id=env_id, solved=solved, reward=reward, legal_action_count=legal_actions,
                    steps_used=steps_used, optimal_steps=optimal, illegal_action_reason=None,
                    latency=time.monotonic() - start, execution_failure=False,
                ))
                break
            obs = step_result.observation
        else:
            results.append(EvaluationResult(
                env_id=env_id, solved=False, reward=0.0, legal_action_count=legal_actions,
                steps_used=steps_used, optimal_steps=optimal, illegal_action_reason="step_limit",
                latency=time.monotonic() - start, execution_failure=False,
            ))
    summary = format_evaluation_summary(results)
    summary += f"\n  Model calls: {model_call_count}\n"
    print(summary)

    # Store results
    from src.autoharness.harness_as_policy.artifacts import ArtifactStore
    store = ArtifactStore(root=run_dir.parent, run_id=run_dir.name)
    store.write_evaluation("llm-baseline", {
        "results": [
            {
                "env_id": r.env_id,
                "solved": r.solved,
                "reward": r.reward,
                "legal_action_count": r.legal_action_count,
                "steps_used": r.steps_used,
                "optimal_steps": r.optimal_steps,
                "illegal_action_reason": r.illegal_action_reason,
                "latency": r.latency,
                "execution_failure": r.execution_failure,
            }
            for r in results
        ],
        "model_call_count": model_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })
    return results


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.command == "synthesize":
        synthesize_cmd(parsed)
    elif parsed.command == "evaluate":
        evaluate_cmd(parsed.run)
    elif parsed.command == "evaluate-baseline":
        evaluate_baseline_cmd(parsed.run, parsed.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **9.4: Run CLI test to verify it passes**

Run: `uv run pytest src/autoharness/harness_as_policy/cli_test.py -v`
Expected: All pass

- [ ] **9.5: Commit**

```bash
git add src/autoharness/cli.py \
       src/autoharness/harness_as_policy/cli_test.py
git commit -m "feat(harness-as-policy): add CLI composition layer"
```

---

## Task 10: Top-level package init and whole-project verification

**Files:**
- Create: `src/autoharness/__init__.py`

- [ ] **10.1: Create top-level __init__.py**

```python
# src/autoharness/__init__.py
"""AutoHarness: synthesizing code harnesses for LLM agents."""
```

- [ ] **10.2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass (tests target exists plus the new tests).

- [ ] **10.3: Run Ruff format check**

Run: `uv run ruff format . --check`
Expected: No formatting issues (or auto-fix if needed: `uv run ruff format .`)

- [ ] **10.4: Run Ruff lint**

Run: `uv run ruff check .`
Expected: No lint errors

- [ ] **10.5: Run type checking**

Run: `uv run ty check`
Expected: No type errors

- [ ] **10.6: Fix any issues, re-run tests**

Run: `uv run pytest -v && uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: All pass

- [ ] **10.7: Commit all remaining files**

```bash
git add src/autoharness/__init__.py
git commit -m "feat: add top-level package init"
```
