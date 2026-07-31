"""Tests for the policy executor."""

from __future__ import annotations

import textwrap

from autoharness.harness_as_policy.executor import (
    SAFE_IMPORTS,
    PolicyExecutor,
)


def _valid_source() -> str:
    return textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return action == "[A C]"
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


def test_valid_policy_executes() -> None:
    """A valid policy module executes both functions and returns their results."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(_valid_source(), observation="[A B C]")
    assert result.success
    assert result.output == "[A C]"
    assert result.is_legal_action is True


def test_missing_legal_action_checker() -> None:
    """A module without is_legal_action returns contract failure."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]")
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "is_legal_action" in (result.error_details or "")


def test_wrong_legal_action_checker_arity() -> None:
    """A checker without exactly two positional parameters fails validation."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str) -> bool:
        return True
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]")
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "exactly 2" in (result.error_details or "")


def test_positional_only_required_function_is_rejected() -> None:
    """A required function with positional-only parameters fails validation."""
    source = textwrap.dedent("""\
    def propose_action(board: str, /) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return True
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]")
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "exactly 1" in (result.error_details or "")


def test_returns_legal_action_verdict() -> None:
    """The checker receives the board and proposed action and returns its verdict."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "move:" + board

    def is_legal_action(board: str, action: str) -> bool:
        return board == "board-state" and action == "move:board-state"
    """)
    result = PolicyExecutor().execute(source, observation="board-state")
    assert result.success
    assert result.output == "move:board-state"
    assert result.is_legal_action is True


def test_non_bool_legal_action_verdict() -> None:
    """A checker verdict that is not a bool returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return 1
    """)
    result = PolicyExecutor().execute(source, observation="[A B C]")
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert result.is_legal_action is None


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
    """A non-string proposal fails before the legal-action checker is called."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return 42

    def is_legal_action(observation: str, action: str) -> bool:
        raise RuntimeError("CHECKER_WAS_CALLED")
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "propose_action did not return a string" in (result.error_details or "")
    assert "CHECKER_WAS_CALLED" not in (result.error_details or "")


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
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"


def test_blocks_introspection_via_func_globals() -> None:
    """Accessing __globals__ on a module function is blocked at AST level."""
    source = textwrap.dedent("""\
    import random
    def propose_action(observation: str) -> str:
        return random.randint.__globals__["os"].getcwd()
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_blocks_introspection_via_safe_import_globals() -> None:
    """Accessing __import__ or __globals__ is blocked at AST level."""
    source = textwrap.dedent("""\
    imp = __import__
    def propose_action(observation: str) -> str:
        return imp.__globals__["os"].getcwd()
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
    result = executor.execute(source, observation="test")
    assert result.success
    assert result.output == "3.141"


def test_runtime_exception() -> None:
    """Runtime exception in propose_action returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        raise ValueError("boom")

    def is_legal_action(observation: str, action: str) -> bool:
        return True
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

    def is_legal_action(observation: str, action: str) -> bool:
        return True
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

    def is_legal_action(observation: str, action: str) -> bool:
        return True
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


def test_output_exceeds_limit() -> None:
    """Policy that produces output exceeding MAX_OUTPUT_BYTES fails."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return "X" * 200000

    def is_legal_action(observation: str, action: str) -> bool:
        return True
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "Output exceeds" in (result.error_details or "")


def test_execution_result_failure_attributes() -> None:
    """ExecutionResult has expected attributes on failure."""
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute("bad syntax!!!", observation="test")
    assert result.success is False
    assert isinstance(result.failure_type, str)
    assert isinstance(result.error_details, str)


def test_missing_legality_entry_point() -> None:
    source = "def propose_action(board: str) -> str:\n    return '[A C]'"
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(source, "board")
    assert not result.success
    assert result.failure_type == "contract_failure"
    assert "is_legal_action" in (result.error_details or "")


def test_legality_entry_point_requires_two_arguments() -> None:
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str) -> bool:
        return True
    """)
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(source, "board")
    assert not result.success
    assert result.failure_type == "contract_failure"


def test_executor_returns_generated_legality_verdict() -> None:
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(_valid_source(), "board")
    assert result.success
    assert result.output == "[A C]"
    assert result.is_legal_action is True


def test_legality_entry_point_must_return_bool() -> None:
    source = textwrap.dedent("""\
    def propose_action(board: str) -> str:
        return "[A C]"

    def is_legal_action(board: str, action: str) -> bool:
        return "yes"
    """)
    result = PolicyExecutor(timeout=5, max_source_size=65536).execute(source, "board")
    assert not result.success
    assert result.failure_type == "execution_failure"
    assert "is_legal_action did not return a bool" in (result.error_details or "")
