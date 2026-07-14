"""Tests for the policy executor."""

from __future__ import annotations

import textwrap

from autoharness.harness_as_policy.executor import (
    SAFE_IMPORTS,
    PolicyExecutor,
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
    """Non-string return from propose_action returns execution failure."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return 42
    """)
    executor = PolicyExecutor(timeout=5, max_source_size=65536)
    result = executor.execute(source, observation="test")
    assert not result.success
    assert result.failure_type == "execution_failure"


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


def test_output_exceeds_limit() -> None:
    """Policy that produces output exceeding MAX_OUTPUT_BYTES fails."""
    source = textwrap.dedent("""\
    def propose_action(observation: str) -> str:
        return "X" * 200000
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
