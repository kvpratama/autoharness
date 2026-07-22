## Project Overview

AutoHarness is a Python implementation of a harness-as-policy synthesis loop for game-playing
agents. The current implementation targets TextArena Tower of Hanoi environments: it asks an LLM
to refine a Python policy module, executes the generated `propose_action(observation: str) -> str`
function in an isolated subprocess, rolls the policy out against the environment, ranks candidates,
and persists run artifacts for later evaluation.

The main package is `autoharness`; the active implementation lives under
`src/autoharness/harness_as_policy/`.

---

## Tech Stack

- **Runtime**: Python 3.14+
- **Package management**: `uv`
- **CLI**: standard-library `argparse`, exposed as the `autoharness` console script
- **Search / orchestration**: synchronous harness-as-policy search loop with REx-style Thompson
  sampling in `search.py`
- **Execution**: generated policies are AST-validated and run in isolated Python subprocesses with
  CPU, memory, process, file-size, timeout, import, and output limits
- **Environment**: built-in TextArena adapters live under
  `src/autoharness/harness_as_policy/environments/`
- **Models**: Anthropic, OpenAI, or Google GenAI via LangChain `init_chat_model`
- **Tracing**: optional Langfuse callback integration when `LANGFUSE_ENABLED` is truthy
- **Configuration**: `pydantic-settings` plus `.env` loading through `python-dotenv`
- **Tests**: `pytest` and `pytest-asyncio`
- **Lint / format**: Ruff
- **Type checking**: `ty`

---

## Project Structure

```text
.
├── AGENTS.md
├── README.md
├── playground.py                       # manual Langfuse/refiner debugging script
├── pyproject.toml                      # package metadata, scripts, Ruff, pytest config
├── artifacts/                          # generated synthesis/evaluation outputs
├── src/autoharness/
│   ├── cli.py                          # top-level CLI: synthesize, evaluate, evaluate-baseline
│   └── harness_as_policy/
│       ├── artifacts.py                # atomic artifact persistence
│       ├── config.py                   # Settings and AUTOHARNESS_* env configuration
│       ├── environments/
│       │   ├── __init__.py             # package marker
│       │   ├── base.py                 # EnvironmentAdapter protocol
│       │   ├── blackjack.py            # TextArena Blackjack adapter
│       │   ├── registry.py             # EvaluationCase, EnvironmentSpec, lookup
│       │   └── tower_of_hanoi.py       # TextArena Tower of Hanoi adapter
│       ├── evaluation.py               # held-out generated-policy evaluation
│       ├── executor.py                 # policy validation and subprocess execution sandbox
│       ├── live_policy.py              # live LLM action baseline
│       ├── models.py                   # dataclasses, enums, ranking key, heuristic
│       ├── refiner.py                  # LLM policy refinement boundary
│       ├── rollout.py                  # single-policy rollout evaluator
│       └── search.py                   # synthesis loop and candidate selection
└── tests/
    ├── test_cli.py
    └── harness_as_policy/
        ├── environments/
        │   ├── test_base.py
        │   ├── test_blackjack.py
        │   ├── test_registry.py
        │   └── test_tower_of_hanoi.py
        ├── test_artifacts.py
        ├── test_assessment.py
        ├── test_config.py
        ├── test_evaluation.py
        ├── test_executor.py
        ├── test_live_policy.py
        ├── test_models.py
        ├── test_refiner.py
        ├── test_rollout.py
        └── test_search.py
```

Tests live under `tests/`, mirror the source hierarchy, and are named `test_<module>.py`.

---

## Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a focused test file
uv run pytest tests/harness_as_policy/test_executor.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check

# Run the CLI
uv run autoharness --help
uv run autoharness synthesize --env TowerOfHanoi-v0 --model <provider:model>
uv run autoharness evaluate --run artifacts/<run-id>
uv run autoharness evaluate-baseline --run artifacts/<run-id> --model <provider:model>

# Manual Langfuse/refiner debugging
uv run python playground.py

# Add dependencies
uv add <package>
uv add --dev <package>
```

---

## Runtime Configuration

Configuration is read through `Settings` in `src/autoharness/harness_as_policy/config.py`.
Environment variables use the `AUTOHARNESS_` prefix and may be loaded from `.env`.

Important settings:

- `AUTOHARNESS_MODEL`: required for synthesis unless passed with `--model`
- `AUTOHARNESS_ENV_ID`: defaults to `TowerOfHanoi-v0`
- `AUTOHARNESS_PROFILE`: `smoke`, `low-cost`, or `full-search` (default: smoke)
- `AUTOHARNESS_REFINEMENTS`: optional override for the profile refinement budget
- `AUTOHARNESS_ARTIFACT_ROOT`: defaults to `artifacts`
- `AUTOHARNESS_THOMPSON_SEED`: defaults to `42`
- `AUTOHARNESS_EXECUTION_TIMEOUT`: defaults to `10`
- `AUTOHARNESS_MAX_SOURCE_SIZE`: defaults to `32768`
- `AUTOHARNESS_LOG_LEVEL`: optional logging level
- `AUTOHARNESS_INPUT_PRICE_PER_MILLION` and `AUTOHARNESS_OUTPUT_PRICE_PER_MILLION`: optional
  baseline cost inputs

Provider API keys are handled by the corresponding LangChain integrations. Do not hardcode secrets.

---

## Artifact Layout

Each synthesis run writes to `artifacts/<run-id>/` by default:

```text
artifacts/<run-id>/
├── best.py
├── candidates/<candidate-id>.py
├── config.json
├── events.jsonl
├── evaluation/
├── rollouts/<candidate-id>.json
├── synthesis-summary.json
└── tree.json
```

Treat `artifacts/` as generated output unless a task explicitly asks to inspect or preserve a run.

---

## Code Conventions

### General

- Python **3.14+** minimum.
- Use strict type hints on every function signature, including return types.
- Write Google-style docstrings for public functions and classes. Keep private helper docstrings when
  they clarify non-obvious behavior.
- Prefer existing dataclasses, protocols, and small module-level functions over adding broad
  abstractions.
- Keep the current synthesis, rollout, and CLI paths synchronous unless the change explicitly
  introduces an async boundary. If adding new I/O-heavy integration code, consider async only where it
  fits the surrounding call path.
- Use `pathlib.Path` for filesystem paths where practical.
- Avoid bare `Any`; use protocols or concrete types when the dependency boundary is known. Some model
  and environment boundaries currently use `Any` where third-party types are loose.
- Do not use bare `except:`. Catch specific exceptions.

### Generated Policy Contract

Generated candidate policies must define:

```python
def propose_action(observation: str) -> str:
    ...
```

The executor intentionally restricts generated code:

- allowed imports are listed in `SAFE_IMPORTS` in `executor.py`
- filesystem, network, subprocess, dynamic code, introspection, and dangerous builtins are blocked
- output and stderr are capped
- policy execution happens in a temporary working directory under an isolated Python subprocess

Preserve these constraints when changing the executor, refiner prompt, or evaluation flow.

### Environment Variables

- All secrets and configuration come from `.env` or the process environment.
- Access AutoHarness configuration through `Settings`; do not scatter `os.environ` reads for
  `AUTOHARNESS_*` settings.
- Langfuse-specific environment access is currently isolated to tracing/debug paths.

---

## Testing

- Test runner: `uv run pytest`
- Use Red-Green-Refactor TDD for behavior changes.
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` in `pyproject.toml`.
- Tests live under `tests/`, mirror the source hierarchy, and are named `test_<module>.py`.
- Mock all external model providers and tracing/network boundaries in unit tests.
- Do not hit live Anthropic, OpenAI, Google, Langfuse, or other network services in unit tests.
- Prefer small fakes like the existing fake chat models/adapters in `test_refiner.py` and
  `test_search.py`.
- Mark integration tests that require live services or long-running environments with
  `@pytest.mark.integration` and keep them skipped by default.

---

## Linting & Formatting

Ruff is the single tool for linting and formatting.

- Keep line length at 100 characters.
- Current Ruff lint families: `E`, `F`, `I`, `UP`, `B`, `C4`, `PT`.
- Run `uv run ruff check .` and `uv run ruff format .` before finishing code changes.

---

## What NOT to Do

- Do not commit `.env`.
- Do not use `pip install`; use `uv add` or `uv sync`.
- Do not hardcode API keys, connection strings, provider credentials, or local absolute paths.
- Do not let tests call live LLM providers or Langfuse.
- Do not weaken policy execution sandbox checks without adding focused tests for the risk.
- Do not treat generated artifacts as source files unless the task explicitly requires it.
