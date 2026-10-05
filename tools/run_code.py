"""Quick execution tool for example / public test cases."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

from sandbox.executor import ExecutionResult, SandboxExecutor


def run_example_tests(
    problem: Dict[str, Any],
    candidate_code: str,
    max_cases: Optional[int] = None,
    timeout: float = 2.0,
) -> ExecutionResult:
    """Run candidate code against visible test cases from the KB.

    Args:
        problem: The problem dictionary loaded from KB.
        candidate_code: Candidate Python code string.
        max_cases: Optional limit on test cases (defaults to all tests in problem['tests']).
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult containing results for the test cases.
    """
    tests = problem.get("tests", [])
    if max_cases is not None:
        tests = tests[:max_cases]

    executor = SandboxExecutor(default_timeout=timeout)
    return executor.run(
        code=candidate_code,
        function_name=problem["function"],
        adapter=problem.get("adapter"),
        cases=tests,
        timeout=timeout,
    )
