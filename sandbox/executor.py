"""Sandbox execution engine for candidate code.

Executes candidate solutions in isolated Python subprocesses with strict timeouts,
structured error capture, and deterministic output verification.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure interview-kb is on path for runner
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

try:
    from runner import run_cases, PRELUDE_SRC
except ImportError:
    # Fallback import if directory structure is direct
    import runner
    run_cases = runner.run_cases
    PRELUDE_SRC = runner.PRELUDE_SRC


@dataclass
class TestCaseResult:
    passed: bool
    expected: Any
    got: Any
    error: Optional[str] = None
    ms: int = 0
    tag: Optional[str] = None
    args: Optional[List[Any]] = None
    test_number: int = 1


@dataclass
class ExecutionResult:
    passed: bool
    tests_passed: int
    tests_total: int
    results: List[TestCaseResult] = field(default_factory=list)
    failing_tag: Optional[str] = None
    error_summary: Optional[str] = None
    is_syntax_error: bool = False
    is_runtime_error: bool = False
    is_timeout: bool = False
    execution_time_ms: int = 0

    @property
    def pass_ratio(self) -> str:
        return f"{self.tests_passed}/{self.tests_total}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "failing_tag": self.failing_tag,
            "error_summary": self.error_summary,
            "is_syntax_error": self.is_syntax_error,
            "is_runtime_error": self.is_runtime_error,
            "is_timeout": self.is_timeout,
            "execution_time_ms": self.execution_time_ms,
            "results": [
                {
                    "test_number": r.test_number,
                    "passed": r.passed,
                    "tag": r.tag,
                    "runtime_ms": r.ms,
                    "ms": r.ms,
                    "error": r.error,
                    "got": r.got,
                }
                for r in self.results
            ],
        }


class SandboxExecutor:
    """Encapsulates subprocess-based sandboxed execution."""

    def __init__(self, default_timeout: float = 2.0):
        self.default_timeout = default_timeout

    def run(
        self,
        code: str,
        function_name: str,
        adapter: Optional[str],
        cases: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Run a suite of test cases against candidate code in an isolated subprocess.

        Args:
            code: Candidate Python code string.
            function_name: Target function name to execute.
            adapter: Data structure transformation adapter name.
            cases: List of test cases with 'args', 'expected', and optional 'tag'.
            timeout: Per-test timeout in seconds.

        Returns:
            ExecutionResult with detailed pass/fail outcomes.
        """
        t = timeout or self.default_timeout

        if not cases:
            return ExecutionResult(
                passed=True,
                tests_passed=0,
                tests_total=0,
                results=[],
            )

        # Execute using runner.py
        raw_results = run_cases(code, function_name, adapter, cases, timeout=t)

        test_results: List[TestCaseResult] = []
        passed_count = 0
        first_failing_tag: Optional[str] = None
        error_summary: Optional[str] = None
        is_syntax = False
        is_runtime = False
        is_timeout = False
        total_ms = 0

        for i, (r, c) in enumerate(zip(raw_results, cases)):
            ok = bool(r.get("pass", False))
            err = r.get("error")
            got = r.get("got")
            ms = r.get("ms", 0)
            tag = r.get("tag") or c.get("tag")
            total_ms += ms

            if ok:
                passed_count += 1
            else:
                if first_failing_tag is None and tag:
                    first_failing_tag = tag

                if err:
                    if error_summary is None:
                        error_summary = err
                    if "SyntaxError" in err:
                        is_syntax = True
                    elif "Timeout" in err:
                        is_timeout = True
                    elif "function" in err and "not defined" in err:
                        is_syntax = True
                    else:
                        is_runtime = True

            test_results.append(
                TestCaseResult(
                    passed=ok,
                    expected=c.get("expected"),
                    got=got,
                    error=err,
                    ms=ms,
                    tag=tag,
                    args=c.get("args"),
                    test_number=i + 1,
                )
            )

        all_passed = passed_count == len(cases)

        return ExecutionResult(
            passed=all_passed,
            tests_passed=passed_count,
            tests_total=len(cases),
            results=test_results,
            failing_tag=first_failing_tag,
            error_summary=error_summary,
            is_syntax_error=is_syntax,
            is_runtime_error=is_runtime,
            is_timeout=is_timeout,
            execution_time_ms=total_ms,
        )


_default_executor = SandboxExecutor()


def run_sandboxed(
    code: str,
    function_name: str,
    adapter: Optional[str],
    cases: List[Dict[str, Any]],
    timeout: float = 2.0,
) -> ExecutionResult:
    """Convenience function to execute cases with the default sandbox."""
    return _default_executor.run(code, function_name, adapter, cases, timeout)
