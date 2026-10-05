"""Interview session state machine and session lifecycle management."""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

import loader
from storage.database import Database, get_db
from .personas import Persona, load_persona


class InterviewPhase(str, Enum):
    INTRO = "intro"
    UNDERSTANDING = "understanding"
    APPROACH = "approach"
    APPROACH_REVIEW = "approach_review"
    CODING = "coding"
    RUNNING_EXAMPLES = "running_examples"
    TEST_REVIEW = "test_review"
    SUBMISSION = "submission"
    DEBUGGING = "debugging"
    COMPLEXITY = "complexity"
    DEBRIEF = "debrief"
    COMPLETE = "complete"
    # Backward compatibility aliases
    PROBLEM = "understanding"
    TESTING = "submission"


class InterviewSession:
    """Maintains the full progressive state machine of an active mock technical interview."""

    def __init__(
        self,
        problem_id: str,
        persona_name: str = "friendly",
        model_name: str = "qwen2.5-coder:3b",
        session_id: Optional[str] = None,
        db: Optional[Database] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.db = db or get_db()
        self.problem_id = problem_id
        self.problem: Dict[str, Any] = loader.load(problem_id)
        self.persona_name = persona_name.lower().strip()
        self.persona: Persona = load_persona(self.persona_name)
        self.model_name = model_name

        self.phase: InterviewPhase = InterviewPhase.UNDERSTANDING
        self.status: str = "in_progress"  # "in_progress", "completed", "abandoned"
        self.understanding_done: bool = False
        self.approach_done: bool = False
        self.optimization_done: bool = False
        self.candidate_code: str = self._generate_starter_code()
        self.attempts: List[Dict[str, Any]] = []
        self.hints_used: int = 0
        self.hint_level: int = 0
        self.failures: List[Dict[str, Any]] = []
        self.complexity_step: str = "time"  # "time" or "space" or "done"
        self.complexity_answers: Dict[str, str] = {}
        self.complexity_evaluation: Optional[Dict[str, Any]] = None
        self.last_example_result: Optional[Dict[str, Any]] = None
        self.last_submission_result: Optional[Dict[str, Any]] = None
        self.solved: bool = False
        self.tests_passed: int = 0
        self.tests_total: int = len(self.problem.get("tests", [])) + len(self.problem.get("large_tests", []))
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None
        self.duration_seconds: int = 0
        self.debrief_content: Optional[str] = None

        self.conversation: List[Dict[str, Any]] = []

        # Save initial session record in DB (status: in_progress)
        self._sync_db()

    def _generate_starter_code(self) -> str:
        """Create clean initial starter template for the problem."""
        fn = self.problem.get("function", "solution")
        sig = self.problem.get("signature", f"{fn}()")
        statement = self.problem.get("statement", "")

        return f'''def {sig}:
    """
    {statement}
    """
    # Write your solution here
    pass
'''

    def _sync_db(self) -> None:
        """Save/update current session state to SQLite database."""
        try:
            self.db.save_session(
                session_id=self.session_id,
                problem_id=self.problem_id,
                problem_title=self.problem.get("title"),
                pattern=self.problem.get("pattern"),
                difficulty=self.problem.get("difficulty", "medium"),
                persona=self.persona_name,
                model=self.model_name,
                started_at=self.started_at,
                ended_at=self.ended_at,
                duration_seconds=self.duration_seconds,
                status=self.status,
                phase=self.phase.value if hasattr(self.phase, "value") else str(self.phase),
                solved=self.solved,
                hints_used=self.hints_used,
                tests_passed=self.tests_passed,
                tests_total=self.tests_total,
                attempts_count=len(self.attempts),
                final_code=self.candidate_code,
                debrief=self.debrief_content,
                messages=self.conversation,
            )
        except Exception as e:
            print(f"Warning: Database sync failed: {e}", file=sys.stderr)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation log."""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._sync_db()

    def transition_to(self, next_phase: InterviewPhase) -> None:
        """Advance the interview phase."""
        self.phase = next_phase
        if next_phase == InterviewPhase.COMPLETE:
            self.finalize_session(status="completed")
        else:
            self._sync_db()

    def finalize_session(self, status: str = "completed", debrief: Optional[str] = None) -> None:
        """Finalize the interview session, calculate duration, and mark as completed."""
        self.status = status
        self.ended_at = datetime.now(timezone.utc).isoformat()
        try:
            start_dt = datetime.fromisoformat(self.started_at)
            end_dt = datetime.fromisoformat(self.ended_at)
            self.duration_seconds = max(1, int((end_dt - start_dt).total_seconds()))
        except Exception:
            self.duration_seconds = 0

        if debrief is not None:
            self.debrief_content = debrief

        self.phase = InterviewPhase.COMPLETE
        self._sync_db()

    def record_attempt(
        self,
        code: str,
        tests_passed: int,
        tests_total: int,
        failure_tags: Optional[List[str]] = None,
    ) -> None:
        """Record an official code submission attempt."""
        self.candidate_code = code
        self.tests_passed = tests_passed
        self.tests_total = tests_total
        attempt_num = len(self.attempts) + 1

        attempt_data = {
            "attempt_number": attempt_num,
            "code": code,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "failure_tags": failure_tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.attempts.append(attempt_data)

        if tests_passed == tests_total and tests_total > 0:
            self.solved = True

        try:
            self.db.record_attempt(
                session_id=self.session_id,
                attempt_number=attempt_num,
                code=code,
                tests_passed=tests_passed,
                tests_total=tests_total,
                failure_tags=failure_tags,
                created_at=attempt_data["created_at"],
            )
        except Exception as e:
            print(f"Warning: Failed to persist attempt: {e}", file=sys.stderr)

        self._sync_db()

    def get_next_hint(self) -> Optional[Dict[str, Any]]:
        """Fetch the next level hint from the KB (1_question -> 2_nudge -> 3_big_hint)."""
        hints_dict = self.problem.get("hints", {})
        sorted_keys = sorted(hints_dict.keys())
        if not sorted_keys:
            return None

        if self.hint_level < len(sorted_keys):
            key = sorted_keys[self.hint_level]
            hint_text = hints_dict[key]
            self.hint_level += 1
            self.hints_used += 1
            self._sync_db()
            return {
                "level": self.hint_level,
                "key": key,
                "text": hint_text,
            }
        else:
            # Return the highest hint available
            key = sorted_keys[-1]
            return {
                "level": len(sorted_keys),
                "key": key,
                "text": hints_dict[key],
            }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary dict for debrief and telemetry."""
        return {
            "session_id": self.session_id,
            "problem_id": self.problem_id,
            "problem_title": self.problem.get("title"),
            "pattern": self.problem.get("pattern"),
            "difficulty": self.problem.get("difficulty"),
            "persona": self.persona_name,
            "model": self.model_name,
            "phase": self.phase.value if hasattr(self.phase, "value") else str(self.phase),
            "status": self.status,
            "solved": self.solved,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "hints_used": self.hints_used,
            "attempts_count": len(self.attempts),
            "understanding_done": self.understanding_done,
            "approach_done": self.approach_done,
            "optimization_done": self.optimization_done,
            "complexity_answers": self.complexity_answers,
            "complexity_evaluation": self.complexity_evaluation,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "debrief": self.debrief_content,
        }
