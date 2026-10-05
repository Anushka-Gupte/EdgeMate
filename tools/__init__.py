"""Deterministic testing and edge-case classification tools for EdgeMate."""

from .run_tests import run_problem_tests
from .run_code import run_example_tests
from .edge_cases import classify_failure, get_failure_followup

__all__ = [
    "run_problem_tests",
    "run_example_tests",
    "classify_failure",
    "get_failure_followup",
]
