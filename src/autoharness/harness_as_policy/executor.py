"""Policy executor with AST validation and isolated subprocess execution."""

from __future__ import annotations

import ast
import os
import signal
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

DANGEROUS_BUILTINS: set[str] = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "globals",
    "locals",
}

MAX_OUTPUT_BYTES: int = 65536


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

    def __init__(
        self,
        timeout: int = 10,
        max_source_size: int = 32768,
        cpu_limit: int = 5,
        memory_limit_mb: int = 128,
    ) -> None:
        self._timeout = timeout
        self._max_source_size = max_source_size
        self._cpu_limit = cpu_limit
        self._memory_limit_bytes = memory_limit_mb * 1024 * 1024

    def execute(self, source: str, observation: str) -> ExecutionResult:
        """Validate and execute propose_action with the given observation."""
        start = time.monotonic()
        if len(source.encode("utf-8")) > self._max_source_size:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="contract_failure",
                error_details=f"Source exceeds {self._max_source_size} bytes",
            )
        parse_err = self._validate_ast(source)
        if parse_err:
            return ExecutionResult(
                success=False,
                output=None,
                latency=time.monotonic() - start,
                failure_type="contract_failure",
                error_details=parse_err,
            )
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
        """Validate source: syntax, signature, safe imports, no dangerous calls."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error: {e}"
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
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in DANGEROUS_BUILTINS:
                    return f"Disallowed builtin call: {func.id}"
                if isinstance(func, ast.Attribute) and func.attr in (
                    "__import__",
                    "__subclasshook__",
                    "__subclasses__",
                ):
                    return f"Disallowed attribute access: {func.attr}"
        return None

    def _make_script(self, source: str, observation: str) -> str:
        """Build the subprocess script with resource limits and restricted builtins."""
        cpu = self._cpu_limit
        mem = self._memory_limit_bytes
        return textwrap.dedent(f"""\
        import builtins, os, resource, signal, sys

        resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
        resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
        os.setsid()

        _allowed_builtins = {{}}
        for _key, _val in vars(builtins).items():
            if _key in (
                "open", "eval", "exec", "compile", "__import__",
                "breakpoint", "input",
                "getattr", "setattr", "delattr",
                "vars", "globals", "locals",
            ):
                continue
            _allowed_builtins[_key] = _val

        _globals = {{"__builtins__": _allowed_builtins, "__name__": "__policy__"}}
        exec(compile({source!r}, "<policy>", "exec"), _globals)

        _observation = {observation!r}
        _result = _globals["propose_action"](_observation)
        if not isinstance(_result, str):
            print(type(_result).__name__, end="")
            sys.exit(2)
        print(_result, end="")
        """)

    def _run_subprocess(self, source: str, observation: str) -> str:
        """Run propose_action in a subprocess with resource limits and process group isolation."""
        script = self._make_script(source, observation)
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self._timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError, PermissionError:
                    proc.kill()
                proc.wait()
                raise
            if proc.returncode != 0:
                err_text = (
                    stderr.decode("utf-8", errors="replace").strip() if stderr else "Unknown error"
                )
                raise RuntimeError(f"Subprocess exited with code {proc.returncode}: {err_text}")
            raw = stdout.decode("utf-8", errors="replace") if stdout else ""
            if len(raw) > MAX_OUTPUT_BYTES:
                raise RuntimeError(f"Output exceeds {MAX_OUTPUT_BYTES} bytes ({len(raw)} bytes)")
            return raw.strip()
