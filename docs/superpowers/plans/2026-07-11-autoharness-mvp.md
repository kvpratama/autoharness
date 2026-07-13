# AutoHarness Tower of Hanoi MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the specified Tower of Hanoi harness synthesizer and baseline-versus-harness policy evaluator with reproducible artifacts and no live API calls in tests.

**Architecture:** A focused `autoharness` package separates settings, TextArena adaptation, generated-code execution, rollout and policy evaluation, REx tree search, model boundaries, artifacts, and CLI composition. A synchronous LangGraph owns only the fixed optimizer state machine; ordinary typed components perform all side effects and are injected for deterministic tests.

**Tech Stack:** Python 3.14, LangGraph, LangChain chat models, TextArena, Pydantic Settings, pytest, Ruff, ty.

---

## File map

- `src/autoharness/config.py`: validated settings and comma-separated seed parsing.
- `src/autoharness/models.py`: serializable candidate, failure, rollout, policy, and summary records.
- `src/autoharness/environment.py`: neutral Tower of Hanoi adapter over TextArena.
- `src/autoharness/executor.py`: AST validation and resource-limited isolated subprocess protocol.
- `src/autoharness/rollout.py`: generated proposer/verifier development rollouts and aggregation.
- `src/autoharness/refiner.py`: structured LangChain refiner/policy boundaries and prompts.
- `src/autoharness/search.py`: REx selection, ranking, stopping, and LangGraph workflow.
- `src/autoharness/artifacts.py`: atomic run artifact persistence and reload.
- `src/autoharness/evaluation.py`: baseline and verifier-assisted held-out policy evaluation.
- `src/autoharness/cli.py`: `synthesize` and `evaluate` command parsing/composition.
- `src/autoharness/__init__.py`, `src/autoharness/__main__.py`: package API and module entry point.
- Co-located `*_test.py` files: unit, subprocess, adapter, graph, and CLI coverage.
- `pyproject.toml`, `README.md`, `.env.example`: package command, dependencies, configuration, and safety documentation.

### Task 1: Domain records, settings, REx selection, and artifacts

- [ ] Write failing co-located tests for Beta parameters, deterministic seeded selection, lexicographic ranking, stopping conditions, settings seed parsing, and artifact round trips.
- [ ] Run the focused tests and confirm failures are caused by missing modules.
- [ ] Implement typed dataclasses/Pydantic settings, `beta_parameters(candidate, weight=1.0)`, `select_candidate(candidates, rng)`, `candidate_rank(candidate)`, stopping logic, and atomic JSON/JSONL/source persistence.
- [ ] Run the focused tests until green.

### Task 2: TextArena adapter

- [ ] Write real-package tests proving seeded reset, observation sanitization, legal `[A C]`, illegal syntax, completion, reward, invalid-action, and turn-limit normalization.
- [ ] Run the adapter tests and confirm they fail before implementation.
- [ ] Implement `TowerOfHanoiAdapter.reset(seed)`, `observation`, and `step(action)` using `textarena.make`, detecting legality from turn/error state before TextArena resets transient flags and preserving neutral outcome fields.
- [ ] Run the adapter tests until green.

### Task 3: Generated harness executor

- [ ] Write subprocess tests for a valid module plus syntax, signature, import, dangerous-call, runtime, timeout, source-size, and output-size failures.
- [ ] Run the executor tests and confirm expected failures.
- [ ] Implement AST checks requiring exactly public `propose_action(board: str) -> str` and `is_legal_action(board: str, action: str) -> bool`, allowlisted imports, blocked dynamic/host APIs, and a JSON worker launched with `python -I` and Linux `resource` limits.
- [ ] Run the executor tests until green.

### Task 4: Development rollout evaluator

- [ ] Write tests for accepted legal actions, false-positive termination, verifier rejection retries, five-rejection exhaustion, execution failures, legal-but-unsolved classification, aggregate heuristic, and five-example feedback bounding.
- [ ] Run tests red.
- [ ] Implement the evaluator so every proposal counts in the denominator, only verifier-accepted actions reach TextArena, and accepted environment actions form the numerator.
- [ ] Run tests green.

### Task 5: Refiner and policy model boundaries

- [ ] Write tests with fake chat models for complete structured source extraction, bounded failure prompt content, malformed responses, policy action extraction, and bounded model transport retries.
- [ ] Run tests red.
- [ ] Implement provider-neutral `init_chat_model` factories and injected protocols; ensure prompts contain the contract, parent source/metrics, environment rules, and no secrets.
- [ ] Run tests green.

### Task 6: LangGraph optimizer

- [ ] Write a scripted bad-to-good graph test asserting ancestry, selected parent, Thompson draw recording, unsuccessful expansion counts, malformed-child retention, success termination, and budget termination.
- [ ] Run graph tests red.
- [ ] Implement serializable search state and nodes for initialize/evaluate, terminate, select, refine, evaluate child, persist/update, and finalize; compile with explicit conditional edges and no checkpointer because resumption is excluded.
- [ ] Run graph tests green.

### Task 7: Held-out policy evaluation

- [ ] Write scripted policy tests comparing direct baseline submission with verifier rejection, feedback retry, retry exhaustion, completion, and model-call accounting.
- [ ] Run tests red.
- [ ] Implement baseline and harnessed conditions without calling generated `propose_action`; aggregate all metrics named in the design.
- [ ] Run tests green.

### Task 8: CLI and documentation

- [ ] Write CLI tests for argument parsing, a scripted tiny synthesis artifact tree, evaluation summary output, invalid seed lists, and distinct synthesis/evaluation seed enforcement.
- [ ] Run CLI tests red.
- [ ] Implement the console script and dependency-injected command handlers, update package metadata, `.env.example`, and README commands/safety warning.
- [ ] Run CLI tests green.

### Task 9: Whole-project verification

- [ ] Run `uv run pytest` and fix only failures caused by the implementation.
- [ ] Run `uv run ruff format .` followed by `uv run ruff check .`.
- [ ] Run `uv run ty check` and resolve all type errors.
- [ ] Re-run `uv run pytest` after formatting/type fixes and inspect `git diff --check`.
