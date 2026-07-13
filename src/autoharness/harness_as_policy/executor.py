"""Policy executor with AST validation and isolated subprocess execution."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass

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
    output: str | None
    latency: float
    failure_type: str | None = None
    error_details: str | None = None


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

    def _validate_ast(self, source: str) -> str | None:
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
        import sys

        # Execute the policy module
        exec(compile({source!r}, "<policy>", "exec"))

        observation = {observation!r}
        result = propose_action(observation)
        if not isinstance(result, str):
            print(type(result).__name__, end="")
            sys.exit(2)
        print(result, end="")
        """)
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
