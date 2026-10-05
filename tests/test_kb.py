"""Unit tests for Knowledge Base loading, patterns, and test materialization."""

import os
import sys
import pytest

# Ensure interview-kb is importable
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

import loader


def test_all_119_problems_load():
    """Verify that all 119 problems in the KB load without errors."""
    problems = loader.list_problems()
    assert len(problems) == 119, f"Expected 119 problems, found {len(problems)}"


def test_20_patterns_present():
    """Verify that all 20 standard DSA patterns are represented."""
    problems = loader.list_problems()
    patterns = {p["pattern"] for p in problems if "pattern" in p}
    assert len(patterns) >= 20, f"Expected at least 20 patterns, found {len(patterns)}"


def test_problem_schema_integrity():
    """Verify essential fields exist across problems."""
    problems = loader.list_problems()
    for p in problems:
        assert "id" in p, f"Missing id in {p}"
        assert "title" in p, f"Missing title in {p['id']}"
        assert "statement" in p, f"Missing statement in {p['id']}"
        assert "function" in p, f"Missing function in {p['id']}"
        assert "tests" in p and len(p["tests"]) > 0, f"Missing tests in {p['id']}"
        assert "hints" in p, f"Missing hints in {p['id']}"
        assert "complexity" in p, f"Missing complexity in {p['id']}"
        assert "reference_solution" in p, f"Missing reference_solution in {p['id']}"


def test_materialize_tests():
    """Verify test materialization including small and large generator tests."""
    p = loader.load("sliding-window-003")
    tests_small = loader.materialize_tests(p, include_large=False)
    tests_all = loader.materialize_tests(p, include_large=True)

    assert len(tests_small) == len(p["tests"])
    assert len(tests_all) > len(tests_small)
    assert any(t.get("tag") == "large" for t in tests_all)
