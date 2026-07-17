"""Policy executor with AST validation and isolated subprocess execution."""

from __future__ import annotations

import ast
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Protocol

    class _Read1able(Protocol):
        """IO[bytes] with read1 support (BufferedReader/BufferedRandom)."""

        def read1(self, size: int = -1) -> bytes: ...
        def close(self) -> None: ...
        @property
        def closed(self) -> bool: ...
        def fileno(self) -> int: ...
        def flush(self) -> None: ...
        def isatty(self) -> bool: ...
        def readable(self) -> bool: ...
        def read(self, n: int = -1) -> bytes: ...
        def readinto(self, b: bytearray) -> int: ...
        def readline(self, size: int = -1) -> bytes: ...
        def readlines(self, hint: int = -1) -> list[bytes]: ...
        def seek(self, offset: int, whence: int = 0) -> int: ...
        def seekable(self) -> bool: ...
        def tell(self) -> int: ...
        def truncate(self, size: int | None = None) -> int: ...
        def writable(self) -> bool: ...
        def write(self, s: bytes) -> int: ...
        def writelines(self, lines: Iterable[bytes]) -> None: ...
        def __enter__(self) -> _Read1able: ...
        def __exit__(self, *args: object) -> None: ...
        def __iter__(self) -> Iterable[bytes]: ...
        def __next__(self) -> bytes: ...


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

DANGEROUS_ATTRIBUTES: set[str] = {
    "__globals__",
    "__closure__",
    "__code__",
    "__class__",
    "__dict__",
    "__builtins__",
}

MAX_OUTPUT_BYTES: int = 65536
MAX_STDERR_BYTES: int = 65536
RESULT_MARKER: str = "__AUTOHARNESS_RESULT__"


@dataclass
class ExecutionResult:
    """Result of atomically executing a policy module's two-function contract."""

    success: bool
    output: str | None
    latency: float
    is_legal_action: bool | None = None
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
            output, is_legal_action = self._run_subprocess(source, observation)
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
        return ExecutionResult(
            success=True,
            output=output,
            latency=time.monotonic() - start,
            is_legal_action=is_legal_action,
        )

    def _validate_ast(self, source: str) -> str | None:
        """Validate source: syntax, signature, safe imports, no dangerous calls."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error: {e}"
        required_functions = {"propose_action": 1, "is_legal_action": 2}
        found_functions: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in required_functions:
                continue
            found_functions.add(node.name)
            args = node.args
            if (
                len(args.args) != required_functions[node.name]
                or args.posonlyargs
                or args.vararg is not None
                or args.kwarg is not None
                or args.kwonlyargs
            ):
                expected = required_functions[node.name]
                return f"{node.name} must take exactly {expected} positional argument(s)"
        if "propose_action" not in found_functions:
            return "Module must define propose_action(board: str) -> str"
        if "is_legal_action" not in found_functions:
            return "Module must define is_legal_action(board: str, action: str) -> bool"
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in DANGEROUS_ATTRIBUTES:
                return f"Disallowed attribute access: {node.attr}"
            if isinstance(node, ast.Name) and node.id == "__import__":
                return "Disallowed reference: __import__"
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
                if isinstance(func, ast.Attribute) and func.attr == "__import__":
                    return "Disallowed attribute access: __import__"
        return None

    def _make_script(self, source: str, observation: str) -> str:
        """Build the subprocess script with resource limits and restricted builtins."""
        cpu = self._cpu_limit
        mem = self._memory_limit_bytes
        return textwrap.dedent(f"""\
        import builtins, json, os, resource, sys

        resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
        resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))

        _disallowed = {{
            "open", "eval", "exec", "compile",
            "breakpoint", "input",
            "getattr", "setattr", "delattr",
            "vars", "globals", "locals",
            "__import__",
        }}
        _allowed_builtins = {{}}
        for _key, _val in vars(builtins).items():
            if _key in _disallowed:
                continue
            _allowed_builtins[_key] = _val

        _safe_import_names = {{"math", "random", "re", "typing",
            "itertools", "collections", "functools", "dataclasses",
            "enum", "string"}}

        class _ModuleProxy:
            def __init__(self, _m):
                object.__setattr__(self, "_inner", _m)
            def __getattribute__(self, _n):
                if _n.startswith("_"):
                    msg = "Access to private attribute '{{}}' is not allowed".format(_n)
                    raise AttributeError(msg)
                return getattr(object.__getattribute__(self, "_inner"), _n)

        _orig_import = builtins.__import__

        def _safe_import(_name, _g=None, _l=None, _fl=(), _lv=0):
            _top = _name.split(".")[0]
            if _top not in _safe_import_names:
                msg = "Disallowed import: {{}}".format(_name)
                raise ImportError(msg)
            _mod = _orig_import(_name, _g, _l, _fl, _lv)
            _wrap = _ModuleProxy(_mod)
            sys.modules[_top] = _wrap
            return _wrap

        _allowed_builtins["__import__"] = _safe_import

        _globals = {{"__builtins__": _allowed_builtins, "__name__": "__policy__"}}
        exec(compile({source!r}, "<policy>", "exec"), _globals)

        _observation = {observation!r}
        _action = _globals["propose_action"](_observation)
        if not isinstance(_action, str):
            print("propose_action did not return a string", file=sys.stderr)
            sys.exit(2)
        _is_legal = _globals["is_legal_action"](_observation, _action)
        if not isinstance(_is_legal, bool):
            print("is_legal_action did not return a bool", file=sys.stderr)
            sys.exit(2)
        _payload = json.dumps({{"action": _action, "is_legal_action": _is_legal}})
        print({RESULT_MARKER!r} + _payload)
        """)

    def _read_output(self, proc: subprocess.Popen[bytes]) -> str:
        """Read stdout/stderr incrementally with output cap. Raises on timeout or overflow."""
        stdout: _Read1able = cast("_Read1able", proc.stdout)
        stderr: _Read1able = cast("_Read1able", proc.stderr)
        read_fds: Iterable[_Read1able] = [stdout, stderr]

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        total_out = 0
        total_err = 0
        deadline = time.monotonic() + self._timeout

        while True:
            time_left = deadline - time.monotonic()
            if time_left <= 0:
                raise subprocess.TimeoutExpired(proc.args, self._timeout)

            rlist, _, _ = select.select(read_fds, [], [], max(0.0, time_left))

            if not rlist:
                continue

            for fd in rlist:
                try:
                    data = fd.read1()
                except ValueError, OSError:
                    data = b""
                if not data:
                    continue
                if fd is stdout:
                    stdout_chunks.append(data)
                    total_out += len(data)
                    if total_out > MAX_OUTPUT_BYTES:
                        raise RuntimeError(f"Output exceeds {MAX_OUTPUT_BYTES} bytes")
                else:
                    stderr_chunks.append(data)
                    total_err += len(data)
                    if total_err > MAX_STDERR_BYTES:
                        raise RuntimeError(f"Stderr exceeds {MAX_STDERR_BYTES} bytes")

            if proc.poll() is not None:
                for fd, dest in ((stdout, stdout_chunks), (stderr, stderr_chunks)):
                    try:
                        while True:
                            chunk = fd.read1()
                            if not chunk:
                                break
                            dest.append(chunk)
                            if fd is stdout:
                                total_out += len(chunk)
                                if total_out > MAX_OUTPUT_BYTES:
                                    raise RuntimeError(f"Output exceeds {MAX_OUTPUT_BYTES} bytes")
                            else:
                                total_err += len(chunk)
                                if total_err > MAX_STDERR_BYTES:
                                    raise RuntimeError(f"Stderr exceeds {MAX_STDERR_BYTES} bytes")
                    except ValueError, OSError:
                        pass
                break

        proc.wait()
        stdout_str = (
            b"".join(stdout_chunks).decode("utf-8", errors="replace") if stdout_chunks else ""
        )

        if proc.returncode != 0:
            err_text = (
                b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
                if stderr_chunks
                else "Unknown error"
            )
            raise RuntimeError(f"Subprocess exited with code {proc.returncode}: {err_text}")

        return stdout_str.strip()

    def _run_subprocess(self, source: str, observation: str) -> tuple[str, bool]:
        """Run both policy functions and parse the final marked protocol result."""
        script = self._make_script(source, observation)
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                start_new_session=True,
            )
            try:
                stdout = self._read_output(proc)
            except subprocess.TimeoutExpired, RuntimeError:
                self._kill_process_group(proc)
                proc.wait()
                raise
        marker_index = stdout.rfind(RESULT_MARKER)
        if marker_index < 0:
            raise RuntimeError("Missing execution result protocol marker")
        payload_text = stdout[marker_index + len(RESULT_MARKER) :].strip()
        try:
            payload: object = json.loads(payload_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed execution result protocol") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Malformed execution result protocol")
        action = payload.get("action")
        is_legal_action = payload.get("is_legal_action")
        if not isinstance(action, str):
            raise RuntimeError("propose_action did not return a string")
        if not isinstance(is_legal_action, bool):
            raise RuntimeError("is_legal_action did not return a bool")
        return action, is_legal_action

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
        """Kill the entire process group rooted at proc."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
