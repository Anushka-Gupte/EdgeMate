"""Unit tests for interviewer state transitions, personas, and hint progression."""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import loader
from interview.session import InterviewSession, InterviewPhase
from interview.interviewer import InterviewerAgent
from interview.personas import load_persona, get_available_personas
from interview.model import ModelClient
from storage.database import Database
import storage.database as db_module
import tempfile


@pytest.fixture(autouse=True)
def isolated_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name
    test_db = Database(db_path=temp_db_path)
    orig = db_module._db_instance
    db_module._db_instance = test_db
    yield test_db
    db_module._db_instance = orig
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass


class MockModelClient(ModelClient):
    """Deterministic mock model client for fast testing without Ollama."""
    def __init__(self):
        super().__init__(model_name="mock:test")

    def check_connection(self):
        return {"connected": True, "error": None, "models": ["mock:test"], "target_model_installed": True, "target_model": "mock:test"}

    def generate(self, prompt: str, system: str = None, history: list = None, **kwargs):
        if "HINT" in prompt:
            return "Here is a hint for you: consider tracking character positions."
        if "APPROACH" in (system or "") or "approach" in prompt.lower():
            return "That sounds like a solid sliding window approach. Go ahead and implement it!"
        if "COMPLEXITY" in prompt:
            return "Correct, the time complexity is O(n) because each character is visited at most twice."
        if "debrief" in (system or "").lower() or "SESSION METRICS" in prompt:
            return "### What went well\nGreat approach!\n\n### What to practise\nEdge cases.\n\n### Interview summary\nProblem solved: Yes\nTests passed: 10/10\nHints used: 1\nComplexity discussed: Yes\n\n### Next step\nTwo pointers."
        return "Let's check the edge case where duplicates occur."


def test_persona_loading():
    """Verify distinct rules and word limits across Friendly, Standard, and Silent personas."""
    friendly = load_persona("friendly")
    standard = load_persona("standard")
    silent = load_persona("silent")

    assert friendly.id == "friendly"
    assert friendly.max_words == 80

    assert standard.id == "standard"
    assert standard.max_words <= 50

    assert silent.id == "silent"
    assert silent.max_words <= 15


def test_hint_ladder_progression():
    """Verify hints step through Level 1 -> Level 2 -> Level 3 sequentially."""
    session = InterviewSession(problem_id="sliding-window-003", persona_name="friendly")
    h1 = session.get_next_hint()
    assert h1["level"] == 1
    assert "1_question" in h1["key"]

    h2 = session.get_next_hint()
    assert h2["level"] == 2
    assert "2_nudge" in h2["key"]

    h3 = session.get_next_hint()
    assert h3["level"] == 3
    assert "3_big_hint" in h3["key"]

    assert session.hints_used == 3


def test_interviewer_full_flow():
    """Test full progressive interview flow with mock model."""
    mock_model = MockModelClient()
    session = InterviewSession(problem_id="sliding-window-003", persona_name="friendly")
    agent = InterviewerAgent(session=session, model_client=mock_model)

    # 1. Start Interview -> UNDERSTANDING
    welcome = agent.start_interview()
    assert session.phase == InterviewPhase.UNDERSTANDING
    assert "Longest Substring Without Repeating Characters" in welcome

    # 2. Candidate explains understanding -> APPROACH
    reply1 = agent.handle_message("The problem asks for the length of the longest substring without any duplicate characters.")
    assert session.phase == InterviewPhase.APPROACH
    assert session.understanding_done is True

    # 3. Candidate explains approach -> APPROACH_REVIEW
    reply2 = agent.handle_message("I plan to use a sliding window with two pointers and a hash map to track character indices.")
    assert session.phase == InterviewPhase.APPROACH_REVIEW
    assert session.approach_done is True

    # 4. Candidate starts coding -> CODING
    start_msg = agent.start_coding()
    assert session.phase == InterviewPhase.CODING
    assert "implement it in the editor" in start_msg.lower()

    # 5. Candidate runs examples -> TEST_REVIEW
    ex_res, ex_msg = agent.run_examples(session.candidate_code)
    assert session.phase == InterviewPhase.TEST_REVIEW

    # 6. Candidate submits buggy code -> DEBUGGING
    bug_code = session.problem["known_bugs"][0]["code"]
    res, follow_up = agent.submit_solution(bug_code)
    assert res.passed is False
    assert session.phase == InterviewPhase.DEBUGGING
    assert session.attempts[0]["tests_passed"] < session.attempts[0]["tests_total"]

    # 7. Candidate requests hint
    hint = agent.request_hint()
    assert session.hints_used == 1

    # 8. Candidate submits correct solution -> COMPLEXITY
    res, comp_prompt = agent.submit_solution(session.problem["reference_solution"])
    assert res.passed is True
    assert session.phase == InterviewPhase.COMPLEXITY

    # 9. Candidate answers complexity -> DEBRIEF -> COMPLETE
    time_reply = agent.handle_message("O(n) time complexity since left and right traverse the string at most once.")
    assert session.complexity_step == "space"

    space_reply = agent.handle_message("O(min(n, alphabet)) space for the hash map.")
    assert session.phase == InterviewPhase.COMPLETE
    assert session.solved is True
    assert "What went well" in session.debrief_content
