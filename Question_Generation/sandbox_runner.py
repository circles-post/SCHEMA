"""Test-harness for ``experiment_code`` samples.

Bundles a blueprint's ``data_code`` (the synthetic dataset module) plus a
candidate ``main_code`` (either the reference solution or the masked
prompt) and a list of ``unit_tests`` into a single Python script, ships it
to the sandbox, and parses out per-test pass/fail.

Each unit test is a dict shaped like the ones already declared in
``experiments/blueprints/*.py``::

    {
        "name": "threshold_0_60",
        "input": {"threshold": 0.6},
        "expected_output": {"sensitivity": 0.8333, "specificity": 1.0},
        # optional: "function": "calculate_sensitivity_specificity"
    }

By default the harness routes the call to the blueprint's ``summarize_*``
orchestration function (since every blueprint exposes one). Tests that need
to target a specific helper can set ``function`` explicitly.

Comparison is recursive with a small float tolerance so the existing
expected-output literals (``0.8333``, ``1.0``, ``0.5``, ...) keep working
without re-deriving them from the synthetic data.
"""

from __future__ import annotations

import json
from typing import Any

from . import sandbox_client


FLOAT_REL_TOL = 1e-2
FLOAT_ABS_TOL = 1e-2


def _harness_template(data_code: str, main_code: str, unit_tests_json: str) -> str:
    return f'''
import json
import math
import sys
import types

# ---- bundle data_en as an in-memory module so the candidate code's
# ---- ``from data_en import ...`` import resolves without touching disk.
_data_module = types.ModuleType("data_en")
_data_source = {data_code!r}
exec(compile(_data_source, "data_en.py", "exec"), _data_module.__dict__)
sys.modules["data_en"] = _data_module

# ---- exec the candidate solution as the ``main_en`` module.
_main_module = types.ModuleType("main_en")
_main_source = {main_code!r}
_main_compile_error = ""
try:
    exec(compile(_main_source, "main_en.py", "exec"), _main_module.__dict__)
except Exception as _exc:
    _main_compile_error = f"{{type(_exc).__name__}}: {{_exc}}"

_FLOAT_REL_TOL = {FLOAT_REL_TOL!r}
_FLOAT_ABS_TOL = {FLOAT_ABS_TOL!r}


def _values_match(actual, expected):
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(
                float(actual), float(expected),
                rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL,
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            key in actual and _values_match(actual[key], val)
            for key, val in expected.items()
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(_values_match(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _pick_callable(name_hint: str):
    if name_hint:
        # Explicit function name must exist. Previously we fell through to
        # any summarize_* when the hint was wrong, which silently masked
        # test-spec bugs (e.g. a typo in ``unit_tests[i].function``).
        if hasattr(_main_module, name_hint):
            return getattr(_main_module, name_hint)
        raise AttributeError(f"explicit unit-test function not found: {{name_hint}}")
    for attr in dir(_main_module):
        if attr.startswith("summarize_"):
            return getattr(_main_module, attr)
    raise AttributeError("no summarize_* function found in main_en.py")


_unit_tests = json.loads({unit_tests_json!r})
_results = []
if _main_compile_error:
    for _test in _unit_tests:
        _results.append({{
            "name": _test.get("name", "unnamed"),
            "passed": False,
            "actual": None,
            "error": "main_compile_error: " + _main_compile_error,
        }})
else:
    for _test in _unit_tests:
        _name = _test.get("name", "unnamed")
        _fn_name = _test.get("function", "")
        _kwargs = _test.get("input") or {{}}
        _expected = _test.get("expected_output") or {{}}
        _entry = {{"name": _name, "passed": False, "actual": None, "error": ""}}
        try:
            _fn = _pick_callable(_fn_name)
            _actual = _fn(**_kwargs)
            if isinstance(_actual, (dict, list, str, int, float, bool, type(None))):
                _entry["actual"] = _actual
            else:
                _entry["actual"] = repr(_actual)
            _entry["passed"] = bool(_values_match(_actual, _expected))
        except Exception as _exc:
            _entry["error"] = f"{{type(_exc).__name__}}: {{_exc}}"
        _results.append(_entry)

print("===QG_SANDBOX_RESULT_BEGIN===")
print(json.dumps({{"results": _results, "compile_error": _main_compile_error}}, default=str))
print("===QG_SANDBOX_RESULT_END===")
'''


def _parse_harness_output(stdout: str) -> dict[str, Any] | None:
    begin_marker = "===QG_SANDBOX_RESULT_BEGIN==="
    end_marker = "===QG_SANDBOX_RESULT_END==="
    begin = stdout.find(begin_marker)
    end = stdout.find(end_marker)
    if begin == -1 or end == -1 or end <= begin:
        return None
    payload = stdout[begin + len(begin_marker) : end].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _truncate(text: str, limit: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[-limit:]


def run_unit_tests(
    *,
    data_code: str,
    main_code: str,
    unit_tests: list[dict[str, Any]],
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute the bundled (data_code, main_code, unit_tests) in the sandbox.

    The returned dict has the shape::

        {
            "sandbox_status": "ok" | "sandbox_disabled" | "execution_error" | "no_results",
            "passed": int,
            "failed": int,
            "total": int,
            "test_results": list[dict],
            "compile_error": str,
            "stdout": str,    # truncated
            "stderr": str,    # truncated
            "error_type": str,
            "error_message": str,
        }
    """
    total = len(unit_tests)
    base: dict[str, Any] = {
        "sandbox_status": "ok",
        "passed": 0,
        "failed": 0,
        "total": total,
        "test_results": [],
        "compile_error": "",
        "stdout": "",
        "stderr": "",
        "error_type": "",
        "error_message": "",
    }

    if total == 0:
        base["sandbox_status"] = "no_results"
        return base

    harness = _harness_template(
        data_code=data_code,
        main_code=main_code,
        unit_tests_json=json.dumps(unit_tests, default=str),
    )
    raw = sandbox_client.run_code_sync(harness, timeout=timeout)

    base["stdout"] = _truncate(raw.get("stdout", ""))
    base["stderr"] = _truncate(raw.get("stderr", ""))
    base["error_type"] = raw.get("error_type", "")
    base["error_message"] = raw.get("error_message", "")

    # sandbox_disabled = never tried (no host configured / no lib)
    # worker_timeout / internal_error = sandbox service unreachable or
    # hung mid-request. In all three cases we have no signal about the
    # candidate code, so treat them uniformly as "sandbox unavailable"
    # and let the caller fall back to rule_only mode instead of marking
    # every sample as a reference-solution failure.
    if raw.get("reason") in {"sandbox_disabled", "worker_timeout", "internal_error"}:
        base["sandbox_status"] = "sandbox_disabled"
        base["failed"] = total
        return base

    parsed = _parse_harness_output(raw.get("stdout", "") or "")
    if parsed is None:
        base["sandbox_status"] = "execution_error"
        base["failed"] = total
        return base

    test_results = parsed.get("results", []) or []
    base["test_results"] = test_results
    base["compile_error"] = parsed.get("compile_error", "") or ""
    base["passed"] = sum(1 for r in test_results if r.get("passed"))
    base["failed"] = sum(1 for r in test_results if not r.get("passed"))
    return base


def evaluate_experiment_sample(
    *,
    data_code: str,
    main_code: str,
    incomplete_main_code: str,
    unit_tests: list[dict[str, Any]],
    timeout: float | None = None,
) -> dict[str, Any]:
    """Self-test an ``experiment_code`` sample by running both versions in sandbox.

    1. ``reference``: ``data_code`` + ``main_code`` (the answer)
        — must pass *all* unit tests, otherwise the blueprint is buggy.
    2. ``incomplete``: ``data_code`` + ``incomplete_main_code`` (the prompt)
        — must fail at least one test, otherwise the blanks were
        unnecessary and the question is trivial.

    Returned dict::

        {
            "reference":            {... run_unit_tests output ...},
            "incomplete":           {... run_unit_tests output ...},
            "reference_passes_all": bool,
            "incomplete_fails_some": bool,
            "verdict": "passed" | "rejected",
            "rejection_reasons":   list[str],
        }

    Fail-closed: unavailable sandboxes, missing tests, timeouts, and
    harness errors are rejected because the code label cannot be trusted
    without a completed runtime check.
    """
    if not unit_tests:
        return {
            "reference": {"sandbox_status": "no_results", "passed": 0, "failed": 0, "total": 0, "test_results": []},
            "incomplete": {"sandbox_status": "no_results", "passed": 0, "failed": 0, "total": 0, "test_results": []},
            "reference_passes_all": False,
            "incomplete_fails_some": False,
            "verdict": "rejected",
            "rejection_reasons": ["missing_unit_tests"],
        }

    if not sandbox_client.is_sandbox_available():
        skipped = sandbox_client.sandbox_disabled_result()
        return {
            "reference": {
                "sandbox_status": "sandbox_disabled",
                "passed": 0,
                "failed": len(unit_tests),
                "total": len(unit_tests),
                "test_results": [],
                "compile_error": "",
                "stdout": "",
                "stderr": skipped["stderr"],
                "error_type": "sandbox_disabled",
                "error_message": skipped["error_message"],
            },
            "incomplete": {
                "sandbox_status": "sandbox_disabled",
                "passed": 0,
                "failed": len(unit_tests),
                "total": len(unit_tests),
                "test_results": [],
                "compile_error": "",
                "stdout": "",
                "stderr": skipped["stderr"],
                "error_type": "sandbox_disabled",
                "error_message": skipped["error_message"],
            },
            "reference_passes_all": False,
            "incomplete_fails_some": False,
            "verdict": "rejected",
            "rejection_reasons": ["sandbox_unavailable"],
        }

    reference = run_unit_tests(
        data_code=data_code,
        main_code=main_code,
        unit_tests=unit_tests,
        timeout=timeout,
    )
    incomplete = run_unit_tests(
        data_code=data_code,
        main_code=incomplete_main_code,
        unit_tests=unit_tests,
        timeout=timeout,
    )

    reference_passes_all = (
        reference["sandbox_status"] == "ok"
        and reference["total"] > 0
        and reference["failed"] == 0
        and not reference.get("compile_error")
    )
    # The masked prompt must have the harness complete AND then fail at least
    # one unit test (or main-code compile_error caught by the harness).
    # ``execution_error`` = harness itself crashed = we can't trust the result,
    # so reject fail-closed rather than treat as "meaningful blank".
    incomplete_meaningful_blank = (
        incomplete["sandbox_status"] == "ok"
        and (incomplete["failed"] > 0 or incomplete.get("compile_error"))
    )

    rejection_reasons: list[str] = []
    if reference["sandbox_status"] != "ok":
        rejection_reasons.append(f"reference_sandbox_status_{reference['sandbox_status']}")
    if incomplete["sandbox_status"] != "ok":
        rejection_reasons.append(f"incomplete_sandbox_status_{incomplete['sandbox_status']}")
    if not reference_passes_all:
        rejection_reasons.append("reference_solution_failed_unit_tests")
    if not incomplete_meaningful_blank:
        rejection_reasons.append("incomplete_code_already_passes_unit_tests")

    return {
        "reference": reference,
        "incomplete": incomplete,
        "reference_passes_all": reference_passes_all,
        "incomplete_fails_some": incomplete_meaningful_blank,
        "verdict": "passed" if not rejection_reasons else "rejected",
        "rejection_reasons": rejection_reasons,
    }


__all__ = [
    "evaluate_experiment_sample",
    "run_unit_tests",
]
