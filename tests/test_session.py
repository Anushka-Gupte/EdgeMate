"""Unit tests for session state tracking, database persistence, and progress metrics."""

import os
import sys
import tempfile
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import loader
from interview.session import InterviewSession, InterviewPhase
from storage.database import Database
from storage.history import ProgressTracker


def test_session_init_and_starter_code():
    """Verify session initialization and starter template generation."""
    session = InterviewSession(problem_id="sliding-window-003", persona_name="friendly")
    assert session.problem_id == "sliding-window-003"
    assert session.phase == InterviewPhase.PROBLEM
    assert "lengthOfLongestSubstring" in session.candidate_code
    assert session.solved is False
    assert len(session.attempts) == 0


def test_session_persistence_and_progress():
    """Verify SQLite persistence and progress aggregation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        test_db = Database(db_path=db_path)
        session = InterviewSession(problem_id="sliding-window-003", persona_name="standard", db=test_db)

        # Record failed attempt
        session.record_attempt("def bad(): pass", tests_passed=3, tests_total=10, failure_tags=["empty"])
        # Record passing attempt
        session.record_attempt("def good(): pass", tests_passed=10, tests_total=10, failure_tags=[])
        session.transition_to(InterviewPhase.COMPLETE)

        # Verify DB contents
        stored = test_db.get_session(session.session_id)
        assert stored is not None
        assert stored["solved"] == 1
        assert len(stored["attempts"]) == 2
        assert stored["attempts"][0]["tests_passed"] == 3
        assert stored["attempts"][1]["tests_passed"] == 10

        # Verify Progress Tracker
        tracker = ProgressTracker(db=test_db)
        summary = tracker.get_summary()
        assert summary["total_attempted"] == 1
        assert summary["total_solved"] == 1
        assert summary["success_rate"] == 100.0
        assert "Sliding Window" in summary["patterns_practiced"]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
