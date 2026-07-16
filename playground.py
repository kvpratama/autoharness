#!/usr/bin/env python
"""Debug playground for Langfuse tracing with the Refiner.

Usage:
    uv run python playground.py
    uv run python playground.py --model anthropic:claude-sonnet-4-20250514

Checks that LANGFUSE_* env vars are loaded and a trace is sent.
After running, check http://localhost:3000 for the trace.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()

MASK_LEN = 10


def check_env() -> bool:
    """Print and validate Langfuse env vars. Return True if all present."""
    ok = True
    for key in ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL"]:
        val = os.environ.get(key)
        if val:
            masked = val[:MASK_LEN] + "..." if len(val) > MASK_LEN else val
            print(f"  ✅ {key}={masked}")
        else:
            print(f"  ❌ {key}=NOT SET")
            ok = False

    model_id = os.environ.get("AUTOHARNESS_MODEL", "NOT SET")
    print(f"  📋 AUTOHARNESS_MODEL={model_id}")
    if model_id == "NOT SET":
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Langfuse debug playground")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (default: AUTOHARNESS_MODEL from .env)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Langfuse Tracing Debug Playground")
    print("=" * 60)
    print()

    model_id = args.model or os.environ.get("AUTOHARNESS_MODEL", "")
    print("--- Environment Variables ---")
    ok = check_env()
    if not ok:
        print("\n❌ Missing required environment variables. Check your .env file.")
        return 1

    print(f"\n--- Importing Refiner (model={model_id}) ---")

    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    from autoharness.harness_as_policy.refiner import Refiner

    try:
        refiner = Refiner(model_id=model_id)
    except Exception as e:
        print(f"❌ Failed to create Refiner: {e}")
        return 1

    print("  ✅ Refiner created")

    dummy_source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return True
    """)

    print("\n--- Calling Refiner.refine() ---")
    sys.stdout.flush()

    try:
        result = refiner.refine(
            rules="Tower of Hanoi: move disks from A to C using B as auxiliary.",
            action_format="[<from> <to>]",
            parent_source=dummy_source,
            parent_heuristic=0.0,
            parent_reward=0.0,
            parent_legal_actions=3,
            parent_status="step_limit",
            feedback=["Did not solve within step limit"],
            env_name="TowerOfHanoi-v0",
            refine_legal_action=True,
        )
    except Exception as e:
        print(f"❌ refine() raised: {type(e).__name__}: {e}")
        return 1

    print("\n--- Result ---")
    print(f"  success={result.success}")
    print(f"  error_details={result.error_details}")
    print(f"  model_call_count={refiner.model_call_count}")
    print(f"  refinement_count={refiner.logical_refinement_count}")
    if result.source:
        src_preview = result.source[:200].replace("\n", "\\n")
        print(f"  source (first 200 chars)={src_preview}")
    print()
    print("--- Next steps ---")
    print("  1. Check http://localhost:3000 for the trace.")
    print("  2. If no trace appears, check refiner._get_langfuse_handler()")
    print("     is being called and the server is reachable.")
    print("  3. Run with --model <id> to test a different model.")
    print()

    return 0 if result.success else 2


def run_one_step(env_id: str = "TowerOfHanoi-v0", action: str = "[A C]") -> None:
    """
    Run one step of Tower of Hanoi and print the result.
    """
    import textarena as ta

    _env = ta.make(env_id)

    # Find and save the inner env reference
    e = _env
    while hasattr(e, "env"):
        e = e.env
    _inner_env = e

    _env.reset(num_players=1, seed=42)

    done, info = _env.step(action)

    obs_id, obs_text = _env.get_observation()
    print("done:", done)
    print("info:", info)
    print("obs_id:", obs_id)
    print("obs_text:", obs_text)


if __name__ == "__main__":
    # sys.exit(main())
    run_one_step()
