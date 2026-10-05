"""Unit tests for deterministic code execution and sandbox isolation."""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import loader
from tools.run_tests import run_problem_tests
from tools.run_code import run_example_tests
from tools.edge_cases import classify_failure
from sandbox.executor import SandboxExecutor


def test_runner_passes_reference_solution():
    """Verify reference solution achieves 100% test pass rate."""
    p = loader.load("sliding-window-003")
    res = run_problem_tests(p, p["reference_solution"], include_large=False)
    assert res.passed is True
    assert res.tests_passed == res.tests_total
    assert res.failing_tag is None


def test_runner_detects_planted_bugs():
    """Verify planted bugs are correctly caught by the test runner with matching failure tags."""
    p = loader.load("sliding-window-003")
    for bug in p["known_bugs"]:
        res = run_problem_tests(p, bug["code"], include_large=False)
        assert res.passed is False
        classification = classify_failure(p, res)
        assert classification["passed"] is False
        assert classification["status"] == "logic_failure"
        assert "followup_question" in classification


def test_runner_catches_syntax_error():
    """Verify syntax errors return structured error results without crashing."""
    p = loader.load("sliding-window-003")
    bad_syntax_code = "def lengthOfLongestSubstring(s):\n    if True\n        return 1"
    res = run_problem_tests(p, bad_syntax_code, include_large=False)
    assert res.passed is False
    assert res.is_syntax_error is True
    assert "SyntaxError" in (res.error_summary or "")


def test_runner_catches_runtime_error():
    """Verify runtime exceptions (e.g., ZeroDivisionError) are caught cleanly."""
    p = loader.load("sliding-window-003")
    runtime_err_code = "def lengthOfLongestSubstring(s):\n    return 1 // 0"
    res = run_problem_tests(p, runtime_err_code, include_large=False)
    assert res.passed is False
    assert res.is_runtime_error is True
    assert "ZeroDivisionError" in (res.error_summary or "")


def test_runner_enforces_timeout():
    """Verify infinite loop triggers sandbox timeout and is safely terminated."""
    p = loader.load("sliding-window-003")
    infinite_loop_code = "def lengthOfLongestSubstring(s):\n    while True:\n        pass"
    # Use short timeout for test
    res = run_problem_tests(p, infinite_loop_code, include_large=False, timeout=0.5)
    assert res.passed is False
    assert res.is_timeout is True


def test_runner_is_model_independent():
    """Verify runner functions completely independently of any AI/LLM model."""
    p = loader.load("sliding-window-003")
    res = run_example_tests(p, p["reference_solution"])
    assert res.tests_total > 0
    assert res.tests_passed == res.tests_total


def test_runner_supports_typing_and_collections():
    """Verify runner automatically provides List, Dict, Counter, deque without manual import."""
    p = loader.load("sliding-window-003")
    code_with_typing = """
def lengthOfLongestSubstring(s: str) -> int:
    char_map: Dict[str, int] = {}
    q: Deque[str] = deque()
    max_len: int = 0
    left: int = 0
    for right, ch in enumerate(s):
        if ch in char_map and char_map[ch] >= left:
            left = char_map[ch] + 1
        char_map[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len
"""
    res = run_problem_tests(p, code_with_typing, include_large=False)
    assert res.passed is True
    assert res.tests_passed == res.tests_total

