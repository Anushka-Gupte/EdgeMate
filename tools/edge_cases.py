"""Edge case analysis and failure-to-question classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from sandbox.executor import ExecutionResult


def classify_failure(problem: Dict[str, Any], result: ExecutionResult) -> Dict[str, Any]:
    """Classify the failure from an execution result using the problem's KB metadata.

    Args:
        problem: The problem dictionary from KB.
        result: The ExecutionResult from running tests.

    Returns:
        Structured failure analysis dict.
    """
    if result.passed:
        return {
            "status": "passed",
            "passed": True,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
        }

    if result.is_syntax_error:
        return {
            "status": "syntax_error",
            "passed": False,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "error_summary": result.error_summary,
            "followup_question": "It looks like there's a syntax error or missing function definition. Let's take a quick look at that line.",
        }

    if result.is_timeout:
        return {
            "status": "timeout",
            "passed": False,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "error_summary": "Execution timed out",
            "followup_question": "The execution took longer than expected on a larger input. What is the time complexity of your approach, and could it be hitting an infinite loop or high polynomial runtime?",
        }

    if result.is_runtime_error and not result.failing_tag:
        return {
            "status": "runtime_error",
            "passed": False,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "error_summary": result.error_summary,
            "followup_question": f"A runtime error occurred ({result.error_summary}). What might cause that exception during execution?",
        }

    failing_tag = result.failing_tag or "unknown"

    # Search known_bugs in problem metadata
    known_bugs = problem.get("known_bugs", [])
    matched_bug = None
    followup = None

    for bug in known_bugs:
        if bug.get("fails_on") == failing_tag:
            matched_bug = bug.get("name")
            followup = bug.get("followup")
            break

    if not followup:
        # Default follow-up templates for standard edge cases
        edge_case_templates = {
            "empty": "What should your function return if the input is completely empty?",
            "single": "How does your code handle an input with just a single element?",
            "duplicates": "What happens when duplicate values appear in the input sequence?",
            "negative": "How does your logic behave when negative numbers are provided?",
            "large": "What happens when the input size grows significantly (e.g., thousands of items)?",
            "normal": "Walk me through how your code handles a standard example case step by step.",
        }
        followup = edge_case_templates.get(
            failing_tag,
            f"Walk me through what your code does on a '{failing_tag}' input."
        )

    return {
        "status": "logic_failure",
        "passed": False,
        "failing_tag": failing_tag,
        "known_bug_name": matched_bug,
        "followup_question": followup,
        "tests_passed": result.tests_passed,
        "tests_total": result.tests_total,
        "error_summary": result.error_summary,
    }


def get_failure_followup(problem: Dict[str, Any], result: ExecutionResult) -> Optional[str]:
    """Retrieve the targeted interviewer follow-up question for a failing execution result."""
    classification = classify_failure(problem, result)
    return classification.get("followup_question")
