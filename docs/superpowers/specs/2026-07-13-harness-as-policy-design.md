# Harness-as-Policy MVP Design

## Purpose

Implement the Harness-as-Policy experiment from Section 4.3 of Lou et al. (2026) as a generic policy-synthesis framework with TextArena Tower of Hanoi as its first environment adapter. The system uses an LLM only while synthesizing candidate Python policies. The selected policy then plays games as executable code with no inference-time model calls.

The MVP is designed to answer two questions at low cost:

1. Can REx-style iterative code refinement synthesize a policy that solves the deterministic three-disk Tower of Hanoi environment?
2. Does the synthesized policy encode a general algorithm that transfers unchanged to four-, five-, and six-disk variants?

An optional live-LLM baseline measures the paper's test-time cost claim without making paid baseline calls part of normal development or generated-policy evaluation.

## Scope

### Included

- A generic environment adapter protocol used by the optimizer and evaluators.
- A TextArena adapter for the registered three- through six-disk Tower of Hanoi variants.
- Synthesis of complete Python modules with one required entry point:

  ```python
  def propose_action(observation: str) -> str:
      """Return the next action to submit to the environment."""
  ```

- Arbitrary private helper functions inside generated policy modules.
- A LangGraph workflow implementing an immutable refinement tree and REx-style Thompson selection.
- Restricted local subprocess execution with validation and resource limits.
- Two built-in synthesis profiles: an eight-refinement smoke profile and a 32-refinement low-cost profile.
- Provider-neutral LangChain model configuration.
- Held-out transfer evaluation across three-, four-, five-, and six-disk variants.
- An optional live-LLM policy baseline on the same evaluation suite.
- Reproducible source, tree, rollout, event, failure, configuration, and summary artifacts.

### Excluded

- Action-verifier and action-filter harness modes.
- Multiplayer environments.
- Additional single-player game adapters.
- Remote, container, or production-grade generated-code isolation.
- Search resumption after process interruption.
- Paper-scale 256-iteration runs as a named default profile.
- Live model calls in automated tests.
- A web interface.

## Package layout

Python packages cannot contain hyphens, so the feature lives under `src/autoharness/harness_as_policy/`. Future harness modes may become sibling packages without forcing the MVP to invent shared abstractions before they are needed.

```text
src/autoharness/
├── __init__.py
├── cli.py
└── harness_as_policy/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── environment.py
    ├── tower_of_hanoi.py
    ├── executor.py
    ├── rollout.py
    ├── refiner.py
    ├── search.py
    ├── artifacts.py
    └── evaluation.py
```

All feature tests are co-located in `harness_as_policy` as `<module>_test.py`. The top-level CLI is a thin composition boundary and does not contain policy-synthesis behavior.

## Architecture

A fixed LangGraph workflow orchestrates ordinary typed Python components. LangGraph owns iterative control flow; adapters, execution, scoring, model calls, and persistence remain independently testable components. Deep Agents is not used because the optimizer follows a predetermined algorithm rather than an open-ended planning process.

```text
                                synthesis
 ┌───────────────────┐ rules + failures ┌─────────────┐
 │ EnvironmentAdapter│──────────────────▶│ Refiner LLM │
 └─────────┬─────────┘                   └──────┬──────┘
           │ observation                           │ complete policy.py
           ▼                                       ▼
 ┌───────────────────┐ action            ┌──────────────────┐
 │ RolloutEvaluator  │◀─────────────────▶│ Policy subprocess│
 └─────────┬─────────┘                   └──────────────────┘
           │ score + feedback
           ▼
 ┌───────────────────────────────────────────────────────────┐
 │ LangGraph: select parent → refine → evaluate → persist    │
 └─────────────────────────────┬─────────────────────────────┘
                               ▼
                        best policy.py

                             held-out evaluation
 ┌────────────────────┐         ┌───────────────────────────┐
 │ 3/4/5/6-disk Hanoi │◀────────│ Synthesized policy        │
 └────────────────────┘         │ or optional live LLM      │
                                └───────────────────────────┘
```

## Components

### Environment adapter

The generic adapter protocol supplies:

- an environment identifier and human-readable rules;
- an action-format description;
- environment creation and seeded reset;
- current observation retrieval;
- action submission;
- normalized legality, reward, termination reason, and feedback;
- the maximum number of steps in one rollout.

The optimizer and evaluators depend only on this protocol. `TowerOfHanoiAdapter` maps it to these TextArena registrations:

| Environment | Disks | Turn limit |
|---|---:|---:|
| `TowerOfHanoi-v0` | 3 | 14 |
| `TowerOfHanoi-v0-medium` | 4 | 30 |
| `TowerOfHanoi-v0-hard` | 5 | 62 |
| `TowerOfHanoi-v0-hardcore` | 6 | 126 |

The three-disk environment is deterministic and is the only synthesis environment. Seeds remain part of the generic adapter contract for future stochastic environments.

`TowerOfHanoiAdapter` validates policy actions before submission. A valid action contains exactly one source-target move in TextArena's bracketed format, such as `[A C]` or `[A, C]`; empty output, malformed output, or output containing multiple bracketed moves is normalized as an illegal action without calling `env.step`. This prevents a policy from exploiting TextArena's Hanoi parser, which can apply multiple bracketed moves from one submitted string.

After a submitted single move, the adapter treats any TextArena invalid-move signal or invalid-move observation as an illegal transition even if the wrapped environment has not terminated. TextArena is still the authority on rule legality, but the adapter converts its retry-oriented invalid-move behavior into the Section 4.3 rollout semantics.

### Policy executor

The executor validates and runs candidate modules. Pre-execution validation requires:

- parseable Python;
- a `propose_action(observation: str) -> str` entry point;
- source below a configured size limit;
- imports from a small safe standard-library allowlist;
- no filesystem, network, subprocess, dynamic-code, or introspection APIs.

Private helper functions and internal data structures are allowed. NumPy is not required for the MVP.

Each rollout uses a fresh temporary directory and isolated Python subprocess. The executor enforces a wall-clock timeout per call, bounded output, and Linux CPU, memory, process-count, and file-size limits where supported. The worker is terminated after the rollout.

These controls reduce accidental damage and runaway execution. They are not a security boundary, and documentation must warn users not to execute untrusted third-party policy files on a sensitive host.

### Rollout evaluator

The evaluator resets an adapter and repeatedly:

1. sends the current observation to generated `propose_action`;
2. requires a string action result;
3. asks the adapter to validate that the output is exactly one action and submit it when valid;
4. records the normalized transition;
5. stops on an illegal action, execution failure, environment termination, or adapter step limit.

There is no generated verifier or retry filter. TextArena is the authority on rule legality, while the adapter is responsible for enforcing one policy action per rollout step and converting environment retry prompts for invalid moves into immediate rollout failure.

The adapter step limit is an external truncation boundary. When the configured number of policy actions has been submitted without environment termination, the rollout stops immediately and does not make an extra TextArena call to trigger TextArena's own turn-limit outcome. The terminal status is `step_limit`, the terminal reward is `0.0`, and the rollout is still considered legal if every submitted action was legal.

For policy \(\rho\), each rollout uses the Section 4.3 heuristic:

\[
H(\rho)=
\begin{cases}
0 & \text{if }\rho\text{ produces any illegal action}\\
0.5 + 0.5r & \text{otherwise,}
\end{cases}
\]

where \(r\in[0,1]\) is terminal environment reward. A solved legal rollout receives `1.0`; a legal rollout with no reward receives `0.5`; any illegal action makes the complete rollout score `0` regardless of prior progress. For future stochastic environments, candidate score is the mean over configured rollouts.

For adapter step-limit truncation, the configured terminal reward is `0.0`, so a fully legal but unsolved rollout receives `H = 0.5`. TextArena partial-completion rewards are recorded only when TextArena itself terminates during one of the submitted policy actions.

Tower of Hanoi synthesis evaluates one rollout per candidate because its initial state and transitions are deterministic. Repeating seeds would not add evidence.

### Refiner

The refiner uses LangChain's provider-neutral model initialization. The model identifier is required through settings or a CLI argument and is recorded in artifacts. Examples may use `google_genai:gemini-2.5-flash` for paper-oriented experiments, but no provider is a code-level default.

For each refinement, the refiner receives:

- environment name, rules, observation description, and action format;
- the required function contract;
- complete selected-parent source;
- parent heuristic, terminal reward, legal-step count, and status;
- at most five prioritized failure or low-progress events;
- instructions to preserve working behavior, reason about failures, avoid a fixed move script, and return one complete replacement module.

Feedback priority is:

1. illegal environment action;
2. policy exception, timeout, malformed output, or contract failure;
3. final unsolved state or turn-limit result;
4. lowest-progress legal states if more context is needed.

The model response has one structured field containing complete source. A malformed response becomes a zero-score child rather than starting an unbounded correction dialogue. Transient model transport failures may retry once. Artifacts separately report logical refinements and actual model-call attempts so retries cannot hide cost.

### Program-tree search

Search begins with a synthetic root module whose required function raises `NotImplementedError`. The root is not treated as an evaluated policy. The first refinement generates an initial implementation; every later refinement creates an immutable child of an evaluated node.

Each candidate records its ID, parent ID, complete source artifact, heuristic, reward, legality and completion metrics, trajectory reference, failure summary, creation iteration, and expansion count. Malformed or crashing children remain in the history with score zero.

For candidate \(\rho\), selection samples:

\[
\theta_\rho \sim \operatorname{Beta}\left(
1 + C H(\rho),
1 + C(1-H(\rho)) + N_\rho
\right),
\]

where \(N_\rho\) is the number of nonterminal children already generated from that candidate and `C = 1.0`. The candidate with the largest sample is expanded. A seeded random generator makes draws reproducible. Child values are not propagated to ancestors.

This formula is the project's documented REx-style interpretation. The paper describes Thompson-guided refinement but does not specify every implementation detail needed for reproduction.

Search profiles are:

| Setting | Smoke | Low-cost |
|---|---:|---:|
| Maximum LLM refinements | 8 | 32 |
| Synthesis environment | 3 disks | 3 disks |
| Rollouts per candidate | 1 | 1 |
| Policy step limit | 14 | 14 |

Both profiles stop early when a candidate reaches `H = 1.0`. CLI overrides may raise the budget for later experiments, including 256 refinements, without adding a third built-in profile.

At termination, best-candidate ranking is lexicographic:

1. higher heuristic;
2. higher terminal reward;
3. more legal actions before termination;
4. fewer runtime failures;
5. earlier creation iteration.

Synthesis success means only that the three-disk environment was solved. Transfer performance never influences search.

### Artifact store

The artifact store writes atomically where practical and produces:

```text
artifacts/<run-id>/
├── config.json
├── tree.json
├── events.jsonl
├── candidates/
│   ├── 000.py
│   └── ...
├── rollouts/
│   └── <candidate-id>.json
├── best.py
├── synthesis-summary.json
└── evaluation/
    ├── generated-policy.json
    └── llm-baseline.json
```

The baseline file exists only when that opt-in evaluation runs. Configuration contains non-secret resolved settings, dependency versions, model identifier, profile, Thompson seed, and execution limits. Failures in artifact persistence abort the run because continuing would produce incomplete evidence.

## LangGraph workflow

Graph state contains serializable run metadata, iteration, candidate metadata, selected parent, latest result, best candidate, bounded feedback, model-call accounting, and stop reason. Candidate source and full trajectories are referenced by artifact path rather than copied through every state transition.

The graph performs:

1. initialize the synthetic root and run artifacts;
2. check success and budget termination;
3. sample and select a parent;
4. request one complete child module from the refiner;
5. validate and evaluate the child, or record a zero-score validation failure;
6. persist source, rollout, event, and updated tree state;
7. update expansion statistics and best candidate;
8. return to termination checking;
9. materialize `best.py` and the synthesis summary.

A model transport failure after one retry consumes one refinement budget unit and produces a failed refinement event without fabricated source. Environment initialization and persistence failures abort the graph. Search resumption is excluded, so the MVP does not configure a checkpointer.

## Evaluation

### Generated-policy transfer suite

The selected policy runs unchanged once on each deterministic three- through six-disk environment. Per-difficulty results include:

- legal completion;
- environment reward;
- illegal-action reason;
- steps used compared with the optimal \(2^n-1\);
- wall-clock latency;
- policy execution failure.

The summary reports the largest disk count solved. Evaluation makes no model calls and never feeds outcomes back into synthesis.

### Optional live-LLM baseline

A separate opt-in command evaluates a live policy model against the same environments and observations. On each turn, the model returns one action which is submitted directly to TextArena without verifier retries.

It reports the generated-policy gameplay metrics plus model-call count, input/output tokens when exposed by the provider, and total latency. Estimated monetary cost is reported only when the user configures explicit per-million input and output token prices; the project does not maintain a hard-coded provider pricing table.

## Configuration and CLI

Pydantic settings loaded from `.env` own model identifiers, non-secret model options, profile, artifact root, Thompson seed, execution limits, and optional pricing inputs. Provider credentials remain in their standard environment variables and are never copied into artifacts.

The CLI exposes:

```bash
uv run autoharness synthesize \
  --env TowerOfHanoi-v0 \
  --profile smoke \
  --model google_genai:gemini-2.5-flash

uv run autoharness synthesize \
  --env TowerOfHanoi-v0 \
  --profile low-cost \
  --model <provider:model>

uv run autoharness evaluate \
  --run artifacts/<run-id>

uv run autoharness evaluate-baseline \
  --run artifacts/<run-id> \
  --model <provider:model>
```

Environment, model, profile, refinement budget, artifact root, Thompson seed, and execution limits have CLI overrides. The baseline command is never invoked implicitly by synthesis or generated-policy evaluation.

## Testing strategy

Tests follow red-green-refactor TDD, live beside their target modules, and never contact external model services.

### Pure unit tests

- Section 4.3 heuristic, including illegal-action override.
- Beta parameters and seeded Thompson selection.
- Candidate ranking and early-stop behavior.
- Smoke and low-cost defaults plus CLI overrides.
- Artifact serialization and reload.
- Transfer summary and optional price calculation.

### Executor tests

- Valid policy invocation.
- Syntax, missing-function, and incorrect-contract failures.
- Disallowed imports and dangerous APIs.
- Runtime exception, timeout, oversized output, and non-string return.
- Supported Linux resource limits.

### Adapter and rollout tests

- TextArena reset and normalized observations.
- Known legal and illegal actions.
- Rejection of malformed output and multiple bracketed moves before environment submission.
- Three-disk completion and adapter step-limit truncation scoring.
- Immediate termination and zero heuristic after an illegal action.
- Exact three-, four-, five-, and six-disk variant mapping.

### Refiner and graph tests

- Structured complete-source extraction using a fake chat model.
- Malformed response and bounded transport-retry behavior.
- A scripted bad-to-good candidate sequence.
- Immutable ancestry and reproducible parent selection.
- Expansion-count updates.
- Early success plus smoke and low-cost budget termination.
- Complete event, source, rollout, and summary artifacts.

### End-to-end tests

A scripted refiner produces a policy that solves three disks and generalizes to larger variants. The CLI synthesizes, selects, persists, reloads, and evaluates that policy without a live API call. A fake baseline policy verifies model-call, token, latency, and optional cost accounting.

Whole-project verification runs:

```bash
uv run pytest
uv run ruff format .
uv run ruff check .
uv run ty check
```

## Definition of done

The MVP is complete when:

1. The optimizer and evaluators depend on the generic environment adapter protocol rather than Tower of Hanoi internals.
2. All implementation specific to this mode lives under `src/autoharness/harness_as_policy/`, except the thin top-level CLI entry point.
3. Both synthesis profiles run the same algorithm with their documented LLM refinement budgets.
4. Every successful refinement produces one complete policy module through a provider-neutral model boundary.
5. Candidate execution is isolated and resource-bounded as documented.
6. Search uses the Section 4.3 legality-adjusted heuristic and the documented reproducible REx-style selection formula.
7. The best policy is evaluated unchanged across three- through six-disk variants, and those outcomes never influence synthesis.
8. Generated-policy evaluation performs zero model calls.
9. Live-LLM comparison is optional and uses the same environment suite.
10. Artifacts account for every candidate, selection, rollout, failure, logical refinement, and actual model-call attempt.
11. The README documents commands, cost boundaries, reproducibility, and the local-execution warning.
12. Pytest, Ruff formatting and checks, and `ty check` pass without live APIs.

## Deferred extensions

- Additional environment adapters and stochastic rollout profiles.
- Action-verifier and action-filter sibling packages.
- Search resumption and durable checkpoints.
- Remote or containerized generated-code execution.
- Parallel candidate evaluation.
- Paper-scale benchmark orchestration.
- Stateful generated-policy lifecycle methods, introduced only when an environment requires memory not present in its observation.
