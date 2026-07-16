# AutoHarness

AutoHarness experiments with harness-as-policy synthesis for game-playing agents. The current
implementation targets TextArena Tower of Hanoi environments: an LLM refines a Python policy,
AutoHarness executes that policy in a constrained subprocess, evaluates it against the environment,
and keeps the best candidates as run artifacts.

The generated policy contract is:

```python
def propose_action(board: str) -> str:
    """Return one of the best legal actions for the current board."""


def is_legal_action(board: str, action: str) -> bool:
    """Return whether action is legal for the current board."""
```

For Tower of Hanoi, the policy must return exactly one bracketed move such as `[A C]`.

## Requirements

- Python 3.14+
- `uv`
- Provider credentials for the model backend you use through LangChain, such as Anthropic, OpenAI,
  or Google GenAI

## Setup

```bash
uv sync
```

Create a `.env` file or export environment variables for your model provider and AutoHarness
settings. At minimum, provide a model identifier:

```bash
AUTOHARNESS_MODEL=google_genai:gemini-2.5-flash
```

Provider API keys are read by the corresponding LangChain integration. For example, use the
environment variable expected by that provider package.

## Run Synthesis

```bash
uv run autoharness synthesize \
  --env TowerOfHanoi-v0 \
  --model google_genai:gemini-2.5-flash \
  --profile smoke
```

The command prints a run ID and writes artifacts under `artifacts/<run-id>/`.

Useful options:

- `--profile smoke`: short run, currently 8 refinements
- `--profile low-cost`: longer low-cost profile, currently 32 refinements
- `--refinements N`: override the profile refinement budget
- `--artifact-root DIR`: write run output somewhere other than `artifacts`
- `--seed N`: set the Thompson sampling RNG seed
- `--execution-timeout N`: set the per-action policy execution timeout in seconds
- `--max-source-size N`: cap generated policy source size in bytes

## Evaluate A Generated Policy

After synthesis, evaluate the generated `best.py` policy across Tower of Hanoi difficulty variants:

```bash
uv run autoharness evaluate --run artifacts/<run-id>
```

This writes evaluation output to `artifacts/<run-id>/evaluation/generated-policy.json`.

Evaluation requires both policy functions. Legacy runs containing only `propose_action` must be
synthesized again; evaluation reports a contract failure rather than assuming the action is legal.

## Evaluate A Live LLM Baseline

To compare the generated policy against a live model that chooses every action directly:

```bash
uv run autoharness evaluate-baseline \
  --run artifacts/<run-id> \
  --model google_genai:gemini-2.5-flash
```

Optional pricing inputs estimate baseline cost:

```bash
uv run autoharness evaluate-baseline \
  --run artifacts/<run-id> \
  --model google_genai:gemini-2.5-flash \
  --input-price 0.30 \
  --output-price 2.50
```

Prices are per million tokens.

## Configuration

AutoHarness configuration is defined in `src/autoharness/harness_as_policy/config.py` and uses the
`AUTOHARNESS_` prefix.

| Setting | Default | Description |
| --- | --- | --- |
| `AUTOHARNESS_MODEL` | required | LangChain model identifier for synthesis |
| `AUTOHARNESS_ENV_ID` | `TowerOfHanoi-v0` | Environment ID |
| `AUTOHARNESS_PROFILE` | `smoke` | Synthesis profile: `smoke` or `low-cost` |
| `AUTOHARNESS_REFINEMENTS` | profile default | Optional refinement budget override |
| `AUTOHARNESS_ARTIFACT_ROOT` | `artifacts` | Output directory for run artifacts |
| `AUTOHARNESS_THOMPSON_SEED` | `42` | RNG seed for candidate selection |
| `AUTOHARNESS_EXECUTION_TIMEOUT` | `10` | Per-action execution timeout in seconds |
| `AUTOHARNESS_MAX_SOURCE_SIZE` | `32768` | Maximum generated policy source size |
| `AUTOHARNESS_LOG_LEVEL` | unset | Logging level, such as `INFO` or `DEBUG` |
| `AUTOHARNESS_INPUT_PRICE_PER_MILLION` | unset | Optional baseline input-token price |
| `AUTOHARNESS_OUTPUT_PRICE_PER_MILLION` | unset | Optional baseline output-token price |

CLI flags override settings loaded from the environment.

## Artifacts

A synthesis run creates:

```text
artifacts/<run-id>/
├── best.py
├── candidates/
├── config.json
├── events.jsonl
├── evaluation/
├── rollouts/
├── synthesis-summary.json
└── tree.json
```

`best.py` is the best generated policy for the run. Candidate source files, rollout results, events,
and the candidate tree are kept for debugging and analysis.

## Development

```bash
# Run tests
uv run pytest

# Run a focused test file
uv run pytest src/autoharness/harness_as_policy/executor_test.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check
```

Tests are co-located with the modules they cover and named `<module>_test.py`.

## Project Layout

```text
src/autoharness/
├── cli.py
└── harness_as_policy/
    ├── artifacts.py
    ├── config.py
    ├── environment.py
    ├── evaluation.py
    ├── executor.py
    ├── live_policy.py
    ├── models.py
    ├── refiner.py
    ├── rollout.py
    ├── search.py
    └── tower_of_hanoi.py
```

See `AGENTS.md` for contributor and coding-agent guidance.
