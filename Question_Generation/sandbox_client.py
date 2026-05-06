"""Thin synchronous client for running Python in a remote sandbox.

Mirrors the request/response shape of ``RL-Factory/envs/tools/python.py`` so
the same backend can be reused. The remote endpoint is configured by the
``SANDBOX_HOST`` constant below (or by setting the ``QG_SANDBOX_HOST`` env var).

When ``SANDBOX_HOST`` is empty *or* the ``sandbox_fusion`` package is not
installed, every call returns a structured ``sandbox_disabled`` result
without raising. The rest of the pipeline must treat that as a soft
degradation and fall back to its prior rule-only behaviour.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# CONFIGURATION — fill in once you have a sandbox endpoint.
#
# Leave SANDBOX_HOST as the empty string to disable remote execution. When
# empty, every run_code_sync(...) call returns a structured 'sandbox_disabled'
# result and validator.py preserves the prior rule-only behaviour for
# experiment_code samples.
#
# Both values can also be overridden at runtime via the QG_SANDBOX_HOST and
# QG_SANDBOX_PORT environment variables.
# ---------------------------------------------------------------------------
SANDBOX_HOST: str = os.getenv("QG_SANDBOX_HOST", "").strip()
SANDBOX_PORT: int = int(os.getenv("QG_SANDBOX_PORT", "8080"))
DEFAULT_TIMEOUT: float = float(os.getenv("QG_SANDBOX_TIMEOUT", "20.0"))


try:
    from sandbox_fusion import RunCodeRequest, run_code, set_endpoint  # type: ignore
    _HAS_SANDBOX_FUSION = True
except Exception:  # pragma: no cover - optional dependency
    run_code = None  # type: ignore
    RunCodeRequest = None  # type: ignore
    set_endpoint = None  # type: ignore
    _HAS_SANDBOX_FUSION = False


_endpoint_set: bool = False

# Eagerly add SANDBOX_HOST to NO_PROXY / no_proxy so that sandbox_fusion
# (and any underlying HTTP library) never routes sandbox traffic through a
# proxy. This runs at import time before any HTTP client is instantiated.
if SANDBOX_HOST:
    for _var in ("NO_PROXY", "no_proxy"):
        _cur = os.environ.get(_var, "")
        if SANDBOX_HOST not in {h.strip() for h in _cur.split(",") if h.strip()}:
            os.environ[_var] = f"{_cur},{SANDBOX_HOST}" if _cur else SANDBOX_HOST


def is_sandbox_available() -> bool:
    """Return True iff a sandbox host is configured AND sandbox_fusion imports."""
    return bool(SANDBOX_HOST) and _HAS_SANDBOX_FUSION


def configure_sandbox(host: str, port: int = 8080, timeout: float | None = None) -> None:
    """Override the sandbox endpoint at runtime (e.g. from a CLI flag).

    Useful when you want to point at a different sandbox without editing this
    file. After this call, ``is_sandbox_available()`` will return True iff
    ``host`` is non-empty and ``sandbox_fusion`` is installed.
    """
    global SANDBOX_HOST, SANDBOX_PORT, DEFAULT_TIMEOUT, _endpoint_set
    SANDBOX_HOST = (host or "").strip()
    SANDBOX_PORT = int(port)
    if timeout is not None:
        DEFAULT_TIMEOUT = float(timeout)
    _endpoint_set = False  # force re-binding on next call


def _add_to_no_proxy(host: str) -> None:
    """Ensure *host* appears in both NO_PROXY and no_proxy env vars.

    sandbox_fusion (and its underlying requests/httpx layer) honours these
    variables. Without this, an HTTP proxy configured in the environment will
    intercept traffic to the sandbox and either fail or hang.
    """
    if not host:
        return
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        # Check if already present (comma-separated list)
        if host in {h.strip() for h in current.split(",") if h.strip()}:
            continue
        os.environ[var] = f"{current},{host}" if current else host


def _ensure_endpoint() -> bool:
    """Bind sandbox_fusion to the configured endpoint.

    Before binding, does a fast TCP connect probe (5s timeout) so that an
    unreachable host is detected *before* ``run_code`` blocks indefinitely
    (sandbox_fusion has no built-in timeout).
    """
    global _endpoint_set
    if not SANDBOX_HOST or not _HAS_SANDBOX_FUSION:
        return False
    if _endpoint_set:
        return True
    # Ensure sandbox host bypasses any HTTP proxy
    _add_to_no_proxy(SANDBOX_HOST)
    # TCP probe — fast fail when the host is unreachable
    import socket
    try:
        sock = socket.create_connection((SANDBOX_HOST, SANDBOX_PORT), timeout=5.0)
        sock.close()
    except (OSError, socket.timeout):
        return False
    try:
        set_endpoint(f"http://{SANDBOX_HOST}:{SANDBOX_PORT}")  # type: ignore
        _endpoint_set = True
    except Exception:
        return False
    return True


def _preprocess_code(code: str) -> str:
    """Make the last top-level expression a ``print`` statement.

    Carried over from the reference python.py so the sandbox surfaces the
    value of the final expression on stdout. Our harness already ends with
    explicit print() calls, so this is a no-op for our use; we keep it for
    parity with the reference implementation.
    """
    try:
        tree = ast.parse(code)
        if tree.body:
            last_expr = tree.body[-1]
            if isinstance(last_expr, ast.Expr):
                if not (
                    isinstance(last_expr.value, ast.Call)
                    and isinstance(last_expr.value.func, ast.Name)
                    and last_expr.value.func.id == "print"
                ):
                    print_call = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[last_expr.value],
                            keywords=[],
                        )
                    )
                    tree.body[-1] = print_call
                    code = ast.unparse(tree)
    except Exception:
        pass
    return code


def _extract_error_type(text: str) -> str:
    if not text:
        return "runtime_error"
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*Error)\s*:", text)
    if match:
        return match.group(1)
    return "runtime_error"


# Heuristic for telling apart "real Python error" stderr from benign noise
# (locale warnings, DeprecationWarnings, /bin/bash setlocale notices, etc).
# Only stderr that contains a Python traceback marker or an `XxxError:` line
# is treated as a true failure when status == Finished.
_PYTHON_ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|^[A-Za-z_][A-Za-z0-9_]*Error\s*:)",
    re.MULTILINE,
)


def _stderr_indicates_failure(stderr: str) -> bool:
    if not stderr:
        return False
    return bool(_PYTHON_ERROR_RE.search(stderr))


def _reason_from_status(status: str) -> str:
    lower_status = (status or "").lower()
    if "queue" in lower_status and "timeout" in lower_status:
        return "queue_timeout"
    if "timeout" in lower_status:
        return "worker_timeout"
    if "invalid" in lower_status:
        return "invalid_input"
    return "internal_error"


def _build_result(
    *,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    reason: str = "",
    error_type: str = "",
    error_message: str = "",
    status: str = "",
) -> dict[str, Any]:
    return {
        "success": success,
        "run_success": success,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "reason": reason,
        "error_type": error_type,
        "error_message": error_message,
        "status": status,
    }


def sandbox_disabled_result(message: str = "") -> dict[str, Any]:
    detail = message or (
        "SANDBOX_HOST is empty"
        if _HAS_SANDBOX_FUSION
        else "sandbox_fusion is not installed"
    )
    return _build_result(
        success=False,
        reason="sandbox_disabled",
        error_type="sandbox_disabled",
        error_message=detail,
        stderr=detail,
        status="sandbox_disabled",
    )


def run_code_sync(
    code: str,
    language: str = "python",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run a code snippet in the configured sandbox; never raise.

    Always returns a dict with the shape produced by ``_build_result``. If
    the sandbox is disabled or unreachable, ``reason`` will be
    ``sandbox_disabled`` (or ``internal_error``) and ``success`` will be
    False — callers should treat this as a soft degradation.
    """
    if language != "python":
        msg = f"Unsupported language: {language}"
        return _build_result(
            success=False,
            reason="invalid_input",
            error_type="invalid_input",
            error_message=msg,
            stderr=msg,
            status="invalid_input",
        )

    if not _ensure_endpoint():
        return sandbox_disabled_result()

    effective_timeout = float(DEFAULT_TIMEOUT if timeout is None else timeout)

    # sandbox_fusion.run_code() has NO internal timeout — if the HTTP layer
    # hangs (sandbox accepted the request but never responds) the caller
    # blocks forever. We wrap the call in a separate thread and enforce a
    # hard wall-clock limit via concurrent.futures. On timeout we return an
    # internal_error result and abandon the background thread (it will
    # eventually die or be GC'd — this is best-effort since sandbox_fusion
    # doesn't expose cancellation).
    import concurrent.futures as _cf
    import logging as _log
    _sb_logger = _log.getLogger("question_generation.sandbox")
    _sb_logger.info("run_code_sync: sending request (timeout=%.1fs)", effective_timeout)

    def _do_run():
        return run_code(  # type: ignore[misc]
            RunCodeRequest(code=_preprocess_code(code), language=language)  # type: ignore[misc]
        )

    with _cf.ThreadPoolExecutor(max_workers=1) as _exec:
        _future = _exec.submit(_do_run)
        try:
            r = _future.result(timeout=effective_timeout)
            _sb_logger.info("run_code_sync: got response")
        except _cf.TimeoutError:
            _sb_logger.warning("run_code_sync: TIMEOUT after %.1fs — sandbox HTTP hung", effective_timeout)
            msg = f"sandbox_fusion.run_code() exceeded {effective_timeout}s"
            # Don't block shutdown on the stuck background thread
            _exec.shutdown(wait=False)
            return _build_result(
                success=False,
                reason="worker_timeout",
                error_type="worker_timeout",
                error_message=msg,
                stderr=msg,
                status="worker_timeout",
            )
        except Exception as exc:
            _sb_logger.warning("run_code_sync: exception %s", exc)
            msg = f"Code execution failed: {exc}"
            return _build_result(
                success=False,
                reason="internal_error",
                error_type="internal_error",
                error_message=msg,
                stderr=msg,
                status="internal_error",
            )

    try:
        data = json.loads(r.json())  # type: ignore[attr-defined]
    except Exception as exc:
        msg = f"Failed to parse sandbox response: {exc}"
        return _build_result(
            success=False,
            reason="internal_error",
            error_type="internal_error",
            error_message=msg,
            stderr=msg,
            status="parse_error",
        )

    run_result = data.get("run_result", {}) or {}
    status = str(run_result.get("status") or data.get("status") or "unknown")
    stdout = run_result.get("stdout", "") or ""
    stderr = run_result.get("stderr", "") or ""

    if status == "Finished":
        # status=Finished means the worker completed without crashing. We
        # only treat the run as failed if stderr contains a real Python
        # error signature; otherwise stderr noise (locale warnings, deprec
        # warnings, etc) is reported but does not flip success to False.
        if _stderr_indicates_failure(stderr):
            error_type = _extract_error_type(stderr)
            error_message = stderr.strip().splitlines()[-1]
            return _build_result(
                success=False,
                stdout=stdout,
                stderr=stderr,
                reason="",
                error_type=error_type,
                error_message=error_message,
                status=status,
            )
        return _build_result(success=True, stdout=stdout, stderr=stderr, status=status)

    reason = _reason_from_status(status)
    error_message = stderr.strip() or f"Execution failed with status: {status}"
    return _build_result(
        success=False,
        stdout=stdout,
        stderr=stderr,
        reason=reason,
        error_type=reason,
        error_message=error_message,
        status=status,
    )


# Keep effective timeout in sync with whatever the user lands on once they
# fill in SANDBOX_HOST. Read-only for callers other than configure_sandbox.
__all__ = [
    "SANDBOX_HOST",
    "SANDBOX_PORT",
    "DEFAULT_TIMEOUT",
    "configure_sandbox",
    "is_sandbox_available",
    "run_code_sync",
    "sandbox_disabled_result",
]
