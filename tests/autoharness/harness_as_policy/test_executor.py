"""Tests for the policy executor."""

from __future__ import annotations

import random
import textwrap

import pytest

from autoharness.harness_as_policy.executor import (
    SAFE_IMPORTS,
    PolicyExecutor,
    derive_policy_seed,
    policy_randomness_metadata,
)


def _valid_source() -> str:
    return textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return action == "[A C]"
    """)


@pytest.mark.parametrize(
    ("episode_seed", "action_index", "expected"),
    [
        (0, 0, 7819754643139405723),
        (17, 0, 11891538334161795807),
        (17, 1, 6976715446583647224),
        (-1, 2, 4002281714842392574),
        (4294967295, 7, 6751179175287098193),
    ],
)
def test_policy_seed_derivation_has_stable_golden_vectors(
    episode_seed: int, action_index: int, expected: int
) -> None:
    assert derive_policy_seed(episode_seed, action_index) == expected


@pytest.mark.parametrize(
    ("episode_seed", "action_index"),
    [(True, 0), (0, True), (1.5, 0), (0, -1)],
)
def test_policy_seed_derivation_rejects_invalid_inputs(
    episode_seed: object, action_index: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_policy_seed(episode_seed, action_index)  # type: ignore


def test_execute_requires_an_assigned_policy_seed() -> None:
    with pytest.raises(TypeError):
        PolicyExecutor().execute(_valid_source(), "board")  # type: ignore


@pytest.mark.parametrize("policy_seed", [True, -1, 2**64])
def test_execute_rejects_invalid_assigned_policy_seed(policy_seed: int) -> None:
    with pytest.raises(ValueError, match="policy_seed"):
        PolicyExecutor().execute(_valid_source(), "board", policy_seed=policy_seed)


def test_policy_randomness_metadata_structure() -> None:
    meta = policy_randomness_metadata()
    assert meta["schema_version"] == 1
    assert meta["seed_derivation"] == "sha256-first-uint64-be-v1"
    assert meta["state_model"] == "fresh-subprocess-per-action"
    assert meta["system_random"] == "rejected"


def test_deterministic_public_random_behavior() -> None:
    source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            return f"{random.getrandbits(64)},{random.randint(1, 100)}"

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    res1 = PolicyExecutor().execute(source, "board", policy_seed=1234)
    res2 = PolicyExecutor().execute(source, "board", policy_seed=1234)
    assert res1.success
    assert res2.success
    assert res1.output == res2.output
    assert res1.policy_seed == 1234
    assert res2.policy_seed == 1234

    s0 = derive_policy_seed(17, 0)
    s1 = derive_policy_seed(17, 1)
    rng0 = random.Random(s0)
    rng1 = random.Random(s1)
    exp0 = f"{rng0.getrandbits(64)},{rng0.randint(1, 100)}"
    exp1 = f"{rng1.getrandbits(64)},{rng1.randint(1, 100)}"

    out0 = PolicyExecutor().execute(source, "board", policy_seed=s0).output
    out1 = PolicyExecutor().execute(source, "board", policy_seed=s1).output
    assert out0 == exp0
    assert out1 == exp1

    init_source = textwrap.dedent("""\
        import random

        INITIAL = random.getrandbits(64)

        def propose_action(board: str) -> str:
            return f"{INITIAL}:{random.getrandbits(64)}"

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    init_res1 = PolicyExecutor().execute(init_source, "board", policy_seed=5678)
    init_res2 = PolicyExecutor().execute(init_source, "board", policy_seed=5678)
    assert init_res1.success
    assert init_res2.success
    assert init_res1.output == init_res2.output

    none_source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            v1 = random.getrandbits(64)
            random.seed(None)
            v2 = random.getrandbits(64)
            r = random.Random(None)
            v3 = r.getrandbits(64)
            return f"{v1},{v2},{v3}"

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    p_seed = derive_policy_seed(42, 0)
    r_expected1 = random.Random(p_seed)
    v1_exp = r_expected1.getrandbits(64)
    r_expected1.seed(p_seed)
    v2_exp = r_expected1.getrandbits(64)
    v3_exp = random.Random(p_seed).getrandbits(64)

    none_res = PolicyExecutor().execute(none_source, "board", policy_seed=p_seed)
    assert none_res.output == f"{v1_exp},{v2_exp},{v3_exp}"

    explicit_source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            random.seed(123)
            v1 = random.getrandbits(64)
            v2 = random.Random(123).getrandbits(64)
            return f"{v1},{v2}"

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    exp_r = random.Random(123)
    e1 = exp_r.getrandbits(64)
    e2 = random.Random(123).getrandbits(64)
    explicit_res = PolicyExecutor().execute(explicit_source, "board", policy_seed=9999)
    assert explicit_res.output == f"{e1},{e2}"


@pytest.mark.parametrize(
    "expression",
    [
        "random.SystemRandom().getrandbits(64)",
        'object.__getattribute__(random, "_inner").SystemRandom().getrandbits(64)',
        "random.Random.mro()[1]().getrandbits(64)",
    ],
)
def test_random_entropy_and_proxy_escapes_are_rejected(expression: str) -> None:
    source = textwrap.dedent(f"""\
        import random

        def propose_action(board: str) -> str:
            return str({expression})

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    result = PolicyExecutor().execute(source, "board", policy_seed=1)
    assert result.success is False
    assert result.output is None


def test_from_random_import_systemrandom_is_rejected() -> None:
    source = textwrap.dedent("""\
        from random import SystemRandom

        def propose_action(board: str) -> str:
            return str(SystemRandom().getrandbits(64))

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    result = PolicyExecutor().execute(source, "board", policy_seed=1)
    assert result.success is False
    assert result.output is None


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


def test_valid_policy_executes() -> None:
    """A valid policy module executes both functions and returns their results."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(_valid_source(), observation="[A B C]", policy_seed=0)
    assert result.success
    assert result.output == "[A C]"
    assert result.is_legal_action is True
    assert result.policy_seed == 0


def test_missing_legal_action_checker() -> None:
    """A module without is_legal_action returns contract failure."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "is_legal_action" in (result.error_details or "")
    assert result.policy_seed == 0


def test_wrong_legal_action_checker_arity() -> None:
    """A checker without exactly two positional parameters fails validation."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str) -> bool:
        return True
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "exactly 2" in (result.error_details or "")
    assert result.policy_seed == 0


def test_positional_only_required_function_is_rejected() -> None:
    """A required function with positional-only parameters fails validation."""
    source = textwrap.dedent("""\
    def propose_action(board: str, /) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return True
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "exactly 1" in (result.error_details or "")
    assert result.policy_seed == 0


def test_returns_legal_action_verdict() -> None:
    """The checker receives the board and proposed action and returns its verdict."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "move:" + board

    def is_legal_action(board: str, action: str) -> bool:
        return board == "board-state" and action == "move:board-state"
    """)
    result = PolicyExecutor().execute(source, observation="board-state", policy_seed=0)
    assert result.success
    assert result.output == "move:board-state"
    assert result.is_legal_action is True
    assert result.policy_seed == 0


def test_non_bool_legal_action_verdict() -> None:
    """A checker verdict that is not a bool returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return 1
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert result.is_legal_action is None
    assert result.policy_seed == 0


def test_syntax_error() -> None:
    """Syntax error returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(
        "def propose_action(obs: str) -> str:", observation="test", policy_seed=0
    )
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_missing_entry_point() -> None:
    """Module without propose_action returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("x = 1", observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_wrong_return_type() -> None:
    """A non-string proposal fails before the legal-action checker is called."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return 42

    def is_legal_action(observation: str, action: str) -> bool:
        raise RuntimeError("CHECKER_WAS_CALLED")
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "propose_action did not return a string" in (result.error_details or "")
    assert "CHECKER_WAS_CALLED" not in (result.error_details or "")
    assert result.policy_seed == 0


def test_disallowed_import() -> None:
    """Disallowed import returns contract failure."""
    source = textwrap.dedent("""\
    import os
    def propose_action(observation: str) -> str:
        return "[A C]"
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_blocks_introspection_via_proxy_m() -> None:
    """Accessing _m on a proxied module is blocked at runtime."""
    source = textwrap.dedent("""\
    import random
    def propose_action(observation: str) -> str:
        return random._m._os.getcwd()

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert result.policy_seed == 0


def test_blocks_introspection_via_func_globals() -> None:
    """Accessing __globals__ on a module function is blocked at AST level."""
    source = textwrap.dedent("""\
    import random
    def propose_action(observation: str) -> str:
        return random.randint.__globals__["os"].getcwd()
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_blocks_introspection_via_safe_import_globals() -> None:
    """Accessing __import__ or __globals__ is blocked at AST level."""
    source = textwrap.dedent("""\
    imp = __import__
    def propose_action(observation: str) -> str:
        return imp.__globals__["os"].getcwd()
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_source_too_large() -> None:
    """Source exceeding max_size returns contract failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=10)
    result = executor.execute(_valid_source(), observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_safe_import_works_at_runtime() -> None:
    """A policy using a SAFE_IMPORTS module executes without error."""
    source = textwrap.dedent("""\
    import math
    def propose_action(observation: str) -> str:
        return str(math.pi)[:5]

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert result.success
    assert result.output == "3.141"
    assert result.policy_seed == 0


def test_runtime_exception() -> None:
    """Runtime exception in propose_action returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        raise ValueError("boom")

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert result.policy_seed == 0


def test_timeout() -> None:
    """Policy that hangs returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        while True:
            pass

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=1, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert result.policy_seed == 0


def test_private_helper_allowed() -> None:
    """Private helper functions inside the module are allowed."""
    source = textwrap.dedent("""\
    def _get_move() -> str:
        return "[A C]"

    def propose_action(observation: str) -> str:
        return _get_move()

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert result.success
    assert result.output == "[A C]"
    assert result.policy_seed == 0


def test_execution_result_attributes() -> None:
    """ExecutionResult has expected attributes on success."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(_valid_source(), observation="[A B C]", policy_seed=0)
    assert result.success is True
    assert isinstance(result.output, str)
    assert isinstance(result.latency, float)
    assert result.latency >= 0
    assert result.policy_seed == 0


def test_output_exceeds_limit() -> None:
    """Policy that produces output exceeding MAX_OUTPUT_BYTES fails."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return "X" * 200000

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test", policy_seed=0)
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "Output exceeds" in (result.error_details or "")
    assert result.policy_seed == 0


def test_execution_result_failure_attributes() -> None:
    """ExecutionResult has expected attributes on failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("bad syntax!!!", observation="test", policy_seed=0)
    assert result.success is False
    assert isinstance(result.failure_type, str)
    assert isinstance(result.error_details, str)
    assert result.policy_seed == 0


def test_missing_legality_entry_point() -> None:
    source = "def propose_action(board: str) -> str:\n    return '[A C]'"
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=0
    )
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "is_legal_action" in (result.error_details or "")
    assert result.policy_seed == 0


def test_legality_entry_point_requires_two_arguments() -> None:
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str) -> bool:
        return True
    """)
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=0
    )
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert result.policy_seed == 0


def test_executor_returns_generated_legality_verdict() -> None:
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        _valid_source(), "board", policy_seed=0
    )
    assert result.success
    assert result.output == "[A C]"
    assert result.is_legal_action is True
    assert result.policy_seed == 0


def test_legality_entry_point_must_return_bool() -> None:
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return "yes"
    """)
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=0
    )
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "is_legal_action did not return a bool" in (result.error_details or "")
    assert result.policy_seed == 0


def test_seed_keyword_a_explicit_value_module_level() -> None:
    """random.seed(a=123) must not raise TypeError and must produce deterministic output."""
    source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            random.seed(a=123)
            return str(random.getrandbits(32))

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    expected = str(random.Random(123).getrandbits(32))
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=9999
    )
    assert result.success, result.error_details
    assert result.output == expected


def test_seed_keyword_a_none_module_level() -> None:
    """random.seed(a=None) must re-seed with the policy seed, not raise TypeError."""
    source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            random.seed(a=None)
            return str(random.getrandbits(32))

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    p_seed = derive_policy_seed(7, 0)
    expected = str(random.Random(p_seed).getrandbits(32))
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=p_seed
    )
    assert result.success, result.error_details
    assert result.output == expected


def test_seed_keyword_a_explicit_value_random_instance() -> None:
    """rng.seed(a=123) on a Random instance must not raise TypeError."""
    source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            rng = random.Random()
            rng.seed(a=123)
            return str(rng.getrandbits(32))

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    expected = str(random.Random(123).getrandbits(32))
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=1
    )
    assert result.success, result.error_details
    assert result.output == expected


def test_seed_keyword_a_none_random_instance() -> None:
    """rng.seed(a=None, version=2) on a Random instance must use the policy seed."""
    source = textwrap.dedent("""\
        import random

        def propose_action(board: str) -> str:
            rng = random.Random()
            rng.seed(a=None, version=2)
            return str(rng.getrandbits(32))

        def is_legal_action(board: str, action: str) -> bool:
            return True
    """)
    p_seed = derive_policy_seed(3, 5)
    expected = str(random.Random(p_seed).getrandbits(32))
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(
        source, "board", policy_seed=p_seed
    )
    assert result.success, result.error_details
    assert result.output == expected
