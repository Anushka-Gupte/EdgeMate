"""Unit and integration tests for EdgeMate FastAPI REST API endpoints."""

import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import loader
from storage.database import Database
import storage.database as db_module
import ui.web_app as web_app_module
from ui.web_app import app


@pytest.fixture
def client():
    # Use temporary SQLite database for clean test isolation
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    test_db = Database(db_path=temp_db_path)
    original_db = db_module._db_instance
    db_module._db_instance = test_db

    # Re-initialize tracker to point to the test db
    from storage.history import ProgressTracker
    web_app_module._TRACKER = ProgressTracker(db=test_db)
    web_app_module._ACTIVE_SESSIONS.clear()

    with TestClient(app) as test_client:
        yield test_client

    db_module._db_instance = original_db
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass


def test_api_health(client):
    """Test /api/health endpoint returns model and server status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_connected" in data
    assert "target_model" in data


def test_api_problems_list_and_filter(client):
    """Test /api/problems with filtering by difficulty, pattern, and search query."""
    response = client.get("/api/problems")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 119
    assert len(data["problems"]) == 119

    # Filter by difficulty
    resp_easy = client.get("/api/problems?difficulty=easy")
    assert resp_easy.status_code == 200
    easy_data = resp_easy.json()
    assert all(p["difficulty"].lower() == "easy" for p in easy_data["problems"])

    # Search query
    resp_search = client.get("/api/problems?search=two sum")
    assert resp_search.status_code == 200
    search_data = resp_search.json()
    assert search_data["count"] >= 1


def test_api_problem_detail(client):
    """Test /api/problems/{id} returns full problem details."""
    response = client.get("/api/problems/sliding-window-003")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "sliding-window-003"
    assert "statement" in data
    assert "tests" in data
    assert "signature" in data


def test_api_progress_clean_zero_state(client):
    """Verify clean state produces 0s and no fake data."""
    response = client.get("/api/progress")
    assert response.status_code == 200
    data = response.json()
    assert data["total_attempted"] == 0
    assert data["total_solved"] == 0
    assert data["success_rate"] == 0.0
    assert data["avg_hints_used"] == 0.0
    assert len(data["recent_sessions"]) == 0


def test_api_interview_flow(client):
    """Test full interview lifecycle: start -> run tests -> chat -> end -> debrief -> progress."""
    # 1. Start interview
    start_resp = client.post("/api/sessions/start", json={
        "problem_id": "sliding-window-003",
        "persona_id": "silent",  # Silent persona for deterministic non-Ollama tests
        "model_id": "qwen2.5-coder:3b",
    })
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert session_id is not None
    assert start_data["problem_id"] == "sliding-window-003"

    # 2. In-progress session should not count in progress yet
    prog_resp = client.get("/api/progress")
    assert prog_resp.json()["total_attempted"] == 0

    # 3. Run and Submit reference code
    prob = loader.load("sliding-window-003")
    run_resp = client.post(f"/api/sessions/{session_id}/run", json={
        "code": prob["reference_solution"]
    })
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["passed"] is True

    submit_resp = client.post(f"/api/sessions/{session_id}/submit", json={
        "code": prob["reference_solution"]
    })
    assert submit_resp.status_code == 200
    submit_data = submit_resp.json()
    assert submit_data["passed"] is True
    assert submit_data["solved"] is True

    # 4. Request hint
    hint_resp = client.post(f"/api/sessions/{session_id}/hint")
    assert hint_resp.status_code == 200
    hint_data = hint_resp.json()
    assert "hint" in hint_data
    assert hint_data["hint_level"] == 1

    # 5. End interview
    end_resp = client.post(f"/api/sessions/{session_id}/end")
    assert end_resp.status_code == 200
    end_data = end_resp.json()
    assert "debrief" in end_data

    # 6. Retrieve completed session directly
    sess_resp = client.get(f"/api/sessions/{session_id}")
    assert sess_resp.status_code == 200
    sess_data = sess_resp.json()
    assert sess_data["solved"] is True
    assert sess_data["hints_used"] == 1

    # 7. Check Progress page now reflects the completed session
    final_prog = client.get("/api/progress")
    assert final_prog.status_code == 200
    p_data = final_prog.json()
    assert p_data["total_attempted"] == 1
    assert p_data["total_solved"] == 1
    assert p_data["success_rate"] == 100.0
    assert len(p_data["recent_sessions"]) == 1
    assert p_data["recent_sessions"][0]["session_id"] == session_id
