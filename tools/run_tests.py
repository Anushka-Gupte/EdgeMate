"""Application-level wrapper around deterministic KB test execution."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# Ensure interview-kb is importable
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

import loader
from sandbox.executor import ExecutionResult, SandboxExecutor


def run_problem_tests(
    problem: Dict[str, Any],
    candidate_code: str,
    include_large: bool = True,
    timeout: float = 2.0,
) -> ExecutionResult:
    """Run candidate code against the full deterministic test suite of a problem.

    Args:
        problem: The problem dictionary loaded from the KB.
        candidate_code: The Python code written by the candidate.
        include_large: Whether to include large performance tests.
        timeout: Timeout per test case in seconds.

    Returns:
        ExecutionResult containing detailed pass/fail outcomes.
    """
    tests = loader.materialize_tests(problem, include_large=include_large)
    executor = SandboxExecutor(default_timeout=timeout)
    return executor.run(
        code=candidate_code,
        function_name=problem["function"],
        adapter=problem.get("adapter"),
        cases=tests,
        timeout=timeout,
    )
