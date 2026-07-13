# AutoHarness Tower of Hanoi MVP Design

## Purpose

Implement a focused reproduction of the core AutoHarness method from Lou et al. (2026) against the TextArena `TowerOfHanoi-v0` single-player environment. The system will synthesize two Python functions, search over refinements with REx-style Thompson sampling, and measure whether the resulting action verifier improves a separately configured policy model.

The MVP is intended to validate the paper's central mechanism end to end. It is not a full reproduction of all 145 environments, action-filter mode, harness-as-policy mode, or paper-scale experiment budgets.

## Scope

### Included

- Direct integration with TextArena `TowerOfHanoi-v0`.
- Synthesis of complete Python modules containing:

  ```python
  def propose_action(board: str) -> str:
      """Propose a valid random action for the given board."""

  def is_legal_action(board: str, action: str) -> bool:
      """Return whether the action is legal for the given board."""
  ```

- A LangGraph state machine implementing iterative candidate selection, refinement, rollout evaluation, and termination.
- REx-style Thompson sampling over a program refinement tree.
- Local subprocess execution of generated harnesses under documented resource safeguards.
- Separate, provider-neutral policy and refiner model configuration.
- A CLI for synthesis and held-out evaluation.
- Reproducible candidate, event, failure, and summary artifacts.

### Excluded

- Action-filter and harness-as-policy modes.
- Multiplayer environments or additional single-player games.
- Modal, Docker, or another secure generated-code sandbox.
- Search resumption after process interruption.
- A web interface.
- Live model calls in automated tests.
- Full paper-scale evaluation.

## Architecture

LangGraph owns the explicit optimizer control flow. Ordinary typed Python components own environment adaptation, subprocess execution, scoring, model boundaries, and artifacts. Deep Agents is not part of the MVP because synthesis follows a fixed search algorithm rather than an open-ended planning workflow.

```text
                               synthesis
 ┌────────────┐   board   ┌─────────────────┐   action/check   ┌────────────┐
 │ TextArena  │◀─────────▶│ RolloutEvaluator│◀────────────────▶│ Subprocess │
 │ Hanoi      │           └────────┬────────┘                  │ harness    │
 └────────────┘                    │ feedback                  └────────────┘
                                   ▼
                          ┌─────────────────┐
                          │ LangGraph search│◀──── Refiner LLM
                          └─────────────────┘

                            final evaluation
 ┌────────────┐  observation  ┌────────────┐  candidate action  ┌──────────┐
 │ TextArena  │──────────────▶│ Policy LLM │───────────────────▶│ verifier │
 └────────────┘               └────────────┘                    └────┬─────┘
        ▲                                                          │
        └──────── accepted action / retry with rejection feedback ──┘
```

Generated `propose_action` and the live policy model have deliberately different roles:

- During synthesis, generated `propose_action` explores game states and tests the generated verifier.
- During final evaluation, the live policy model proposes actions and generated `is_legal_action` acts as a rejection sampler.

## Components

### TextArena environment adapter

The adapter owns `TowerOfHanoi-v0` construction, seeded reset, observation retrieval, action submission, and outcome extraction. It translates TextArena-specific state into neutral records containing observation, action, legal/invalid status, reward, completion, termination reason, and feedback.

If the selected TextArena observation wrapper exposes available-move hints, the adapter removes those hints while retaining the board and rules. This follows the paper's experiment setup and prevents the generated harness from merely copying an enumerated legal action.

### Harness executor

The executor validates and runs one candidate module in a temporary directory. Its external contract supports calls to `propose_action` and `is_legal_action` and returns structured successes or failures.

Pre-execution validation requires:

- parseable Python;
- exactly the two required public function signatures;
- a bounded source size;
- imports from a small allowlist sufficient for Tower of Hanoi, such as `re` and `random`;
- no obvious filesystem, network, child-process, or dynamic-code APIs.

Execution uses isolated Python mode (`python -I`) with wall-clock, CPU, memory, file-size, process-count, stdout, and stderr limits where Linux supports them. The worker is terminated after each candidate evaluation.

These measures reduce accidental damage and runaway execution but are not a security boundary. The MVP assumes a trusted research workstation and must warn users not to run untrusted model output on a sensitive host.

### Rollout evaluator

For each development seed, the evaluator resets Tower of Hanoi and repeats:

1. Call generated `propose_action` with the current observation.
2. Call generated `is_legal_action` with the observation and proposal.
3. If the verifier rejects the proposal, record the rejection and request another proposal.
4. Stop the rollout if five consecutive proposals are rejected.
5. Submit verifier-accepted actions to TextArena.
6. Stop the rollout if TextArena rejects an accepted action, execution fails, the puzzle completes, or the turn limit is reached.

Verifier-rejected proposals are not submitted to TextArena because doing so could mutate or terminate the episode. The paper does not fully specify false-negative instrumentation; this MVP follows inference-time rejection-sampling behavior and records retry exhaustion as refinement feedback.

The evaluator defaults to three fixed development seeds per candidate. It stores aggregate metrics and up to five representative failures for refinement.

### Refiner model

The refiner receives:

- environment name, rules, action format, and sanitized observation format;
- the two required function signatures;
- complete parent source;
- parent metrics and progress;
- up to five representative observations, actions, errors, and environment outcomes;
- instructions to reason about failures, avoid repeating unsuccessful approaches, preserve already working cases, and return one complete replacement module.

The response is structured so prose cannot be mistaken for executable source. Model-output validation failures become child candidates with execution failure scores rather than crashing the experiment.

### Program tree search

Every candidate node stores an ID, parent ID, source artifact, heuristic, completion and reward metrics, failure summary, expansion count, and creation iteration. Each refinement creates one child of the selected node; candidates are never mutated in place.

For every candidate program \(\rho\), selection samples:

\[
\theta_\rho \sim \operatorname{Beta}\left(
1 + C h(\rho),
1 + C(1-h(\rho)) + N_\rho
\right)
\]

where:

- \(h(\rho)\) is legal-action success rate in `[0, 1]`;
- \(N_\rho\) is the number of unsuccessful children generated from that candidate;
- \(C=1.0\), matching the AutoHarness paper's heuristic weight.

The candidate with the largest sample is expanded. Child values are not propagated to ancestors. Generating a child that does not meet the stopping criterion increments the selected parent's unsuccessful expansion count.

The default low-cost profile evaluates an initial seed candidate and then permits up to eight model-generated refinements. It uses a seeded random number generator for Thompson draws.

### Policy evaluator

The policy evaluator runs two conditions on evaluation seeds that are distinct from synthesis seeds:

- **Baseline:** submit each policy-model action directly to TextArena.
- **Harnessed:** check each policy-model action with the saved verifier; if rejected, prompt the policy again with an illegal-action warning and bounded retry history.

Both conditions report legal-action rate, puzzle completion, reward, invalid actions, verifier rejections, retry exhaustion, episode length, and model-call count. Evaluation does not use generated `propose_action`.

### Configuration

Pydantic settings loaded through `.env` provide separate provider-neutral LangChain model identifiers for the refiner and policy. Provider credentials remain in their standard environment variables and are never copied into experiment artifacts.

Search iterations, synthesis seeds, evaluation seeds, model identifiers, retry limits, subprocess limits, artifact root, and Thompson RNG seed have CLI overrides. The initial defaults target Gemini 2.5 Flash-compatible model identifiers while permitting any chat model supported by the installed LangChain providers.

## LangGraph workflow

The graph carries serializable experiment state: run ID, iteration, candidate metadata, selected parent, latest result, bounded failure examples, best candidate ID, and stop reason. Candidate source is referenced by artifact ID rather than repeatedly embedded in graph state.

The graph executes these nodes:

1. Initialize and evaluate a seed harness.
2. Check success and budget termination conditions.
3. Sample and select a parent using REx Thompson sampling.
4. Ask the refiner for a child module.
5. Validate and evaluate the child.
6. Persist the candidate, failures, event, and updated tree state.
7. Loop to termination checking.
8. Materialize the best candidate and run summary.

Search stops when either:

- a candidate has perfect development legal-action rate and completes at least one development rollout; or
- the configured refinement budget is exhausted.

## Scoring and failure handling

The Thompson heuristic is:

\[
h = \frac{\text{TextArena-accepted actions}}
         {\text{all generated action attempts}}
\]

Verifier rejections count as unsuccessful attempts. This prevents an always-rejecting verifier from receiving a perfect score.

Best-candidate ranking is lexicographic:

1. higher legal-action success rate;
2. more completed rollouts;
3. higher aggregate reward;
4. fewer execution and retry-exhaustion failures;
5. earlier creation iteration for deterministic final selection.

Representative feedback is classified as follows:

- verifier accepted an environment-invalid action: repair both verifier and proposer;
- repeated verifier rejection or retry exhaustion: improve the proposer and inspect verifier strictness;
- function contract, parse, timeout, or runtime failure: repair the failing code path;
- fully legal but unsolved rollout: improve proposer strategy without weakening legality.

A malformed or crashing child remains in the tree with a zero heuristic so the experiment history is complete. Failures in artifact persistence or TextArena initialization abort the run because continuing would make results incomplete or invalid. Model transport failures use bounded retries and then record a failed refinement event without fabricating candidate source.

## CLI and artifacts

The primary commands are:

```bash
uv run autoharness synthesize \
  --env TowerOfHanoi-v0 \
  --iterations 8 \
  --rollout-seeds 0,1,2

uv run autoharness evaluate \
  --run artifacts/<run-id> \
  --seeds 100,101,102
```

Each synthesis run writes:

```text
artifacts/<run-id>/
├── config.json
├── tree.json
├── events.jsonl
├── candidates/
│   ├── 000.py
│   ├── 001.py
│   └── ...
├── failures/
│   └── <candidate-id>.json
├── best.py
└── summary.json
```

`config.json` stores non-secret resolved configuration and dependency versions. `tree.json` records ancestry, scores, expansion counts, and Thompson draws. `events.jsonl` records ordered graph transitions. `summary.json` records the stopping reason and best-candidate metrics.

The MVP preserves enough state to diagnose interrupted runs but does not resume them.

## Testing strategy

Tests are co-located with target modules and mock every model call.

### Pure unit tests

- REx Beta parameters and seeded node selection.
- Lexicographic candidate ranking.
- Iteration-budget and early-stop behavior.
- Score aggregation and representative-failure sampling.
- Artifact serialization and reload.

### Subprocess tests

- Valid harness function calls.
- Syntax errors, missing functions, contract violations, crashes, oversized output, and timeout handling.
- Rejection of disallowed imports and dangerous built-ins.
- Application of supported Linux resource limits.

### TextArena adapter tests

- Seeded environment reset behavior.
- Known legal and illegal Tower of Hanoi actions through the real local TextArena package.
- Normalization of completion, reward, invalid action, and turn-limit outcomes.
- Removal of available-move hints while preserving board state.

### Graph tests

- A scripted refiner produces a known bad-to-good candidate sequence.
- Parent selection follows seeded Thompson draws.
- Refinement receives the selected parent and bounded failure feedback.
- Candidate ancestry and unsuccessful expansion counts are correct.
- The graph stops on success or after the configured budget.

### CLI end-to-end test

A tiny synthesis run with scripted model responses must produce the candidate tree and expected artifacts. A scripted policy evaluation must demonstrate baseline submission and harness rejection/retry behavior without a live API call.

## Definition of done

The MVP is complete when:

- `synthesize` runs the LangGraph, refiner, subprocess, and TextArena loop end to end;
- the search uses the REx Beta formula with `C=1.0`;
- artifacts reproduce every candidate's source, score, ancestry, expansion count, and selection draw;
- `evaluate` compares baseline and harnessed policy runs on distinct seeds and reports legality, completion, reward, retries, and model calls;
- automated tests never call a live model service;
- Ruff checks and formatting, ty type checking, and pytest pass;
- the README documents one credentialed real-model command for `TowerOfHanoi-v0` and the local-execution warning.

A credentialed synthesis run is optional operational validation because it incurs external model cost. It is not required for automated acceptance.

## Deferred extensions

The component boundaries intentionally leave room for additional TextArena adapters, secure Modal or container executors, resumable search, action-filter mode, harness-as-policy mode, and paper-scale benchmark orchestration. None of these extensions should be implemented until the focused Tower of Hanoi workflow is validated.
