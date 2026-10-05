"""Automated test suite verifying deterministic test-case execution, metadata integrity, and tag accuracy from interview-kb."""

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
from tools.run_tests import run_problem_tests
from tools.run_code import run_example_tests
from tools.edge_cases import classify_failure
from storage.database import Database
import storage.database as db_module
import ui.web_app as web_app_module
from ui.web_app import app


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    test_db = Database(db_path=temp_db_path)
    original_db = db_module._db_instance
    db_module._db_instance = test_db

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


def test_two_sum_ii_kb_definition_and_tags():
    """Test A & B & C: Verify Two Sum II (two-pointers-167) loads exact KB tags without fabrication."""
    p = loader.load("two-pointers-167")
    assert p["id"] == "two-pointers-167"
    assert p["leetcode"] == 167

    kb_tags = [t.get("tag") for t in p["tests"]]
    assert kb_tags[0] == "normal"
    assert kb_tags[1] == "normal"
    assert kb_tags[2] == "negative"
    assert kb_tags[3] == "negative_target"
    assert kb_tags[4] == "duplicates"
    assert kb_tags[5] == "duplicates"

    # Verify no fabricated tags like 'no_reuse_same_element'
    assert "no_reuse_same_element" not in kb_tags

    # Run reference solution
    res = run_problem_tests(p, p["reference_solution"], include_large=True)
    assert res.passed is True
    assert res.tests_passed == len(p["tests"]) + len(p["large_tests"])

    # Verify tags on execution results match KB materialized order
    materialized = loader.materialize_tests(p, include_large=True)
    assert len(res.results) == len(materialized)
    for res_item, kb_item in zip(res.results, materialized):
        assert res_item.tag == kb_item.get("tag")


def test_two_sum_ii_known_bugs_detection():
    """Test D: Verify known buggy solutions for two-pointers-167 fail on the exact KB failure tags."""
    p = loader.load("two-pointers-167")
    known_bugs = p.get("known_bugs", [])
    assert len(known_bugs) >= 2

    for bug in known_bugs:
        res = run_problem_tests(p, bug["code"], include_large=False)
        assert res.passed is False
        assert res.failing_tag == bug["fails_on"]

        classification = classify_failure(p, res)
        assert classification["passed"] is False
        assert classification["failing_tag"] == bug["fails_on"]


def test_api_run_preserves_kb_tags_and_count(client):
    """Test E: Verify API endpoint /run returns exact KB tags and test counts for Two Sum II."""
    # Start session for Two Sum II
    start_resp = client.post("/api/sessions/start", json={
        "problem_id": "two-pointers-167",
        "persona_id": "standard",
        "model_id": "qwen2.5-coder:3b",
    })
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    p = loader.load("two-pointers-167")
    run_resp = client.post(f"/api/sessions/{session_id}/run", json={
        "code": p["reference_solution"]
    })
    assert run_resp.status_code == 200
    data = run_resp.json()

    assert data["tests_total"] == len(p["tests"])
    assert data["tests_passed"] == len(p["tests"])
    assert data["passed"] is True

    # Check tags in order
    result_tags = [r["tag"] for r in data["results"]]
    expected_tags = [t["tag"] for t in p["tests"]]
    assert result_tags == expected_tags
    assert "no_reuse_same_element" not in result_tags


def test_api_submit_includes_large_tests(client):
    """Test F: Verify API endpoint /submit executes large_tests with [large] tag."""
    start_resp = client.post("/api/sessions/start", json={
        "problem_id": "two-pointers-167",
        "persona_id": "standard",
        "model_id": "qwen2.5-coder:3b",
    })
    session_id = start_resp.json()["session_id"]

    p = loader.load("two-pointers-167")
    submit_resp = client.post(f"/api/sessions/{session_id}/submit", json={
        "code": p["reference_solution"]
    })
    assert submit_resp.status_code == 200
    data = submit_resp.json()

    materialized = loader.materialize_tests(p, include_large=True)
    assert data["tests_total"] == len(materialized)
    assert data["results"][-1]["tag"] == "large"
    assert data["results"][-1]["passed"] is True


def test_problem_switching_isolation(client):
    """Test G: Verify switching between different problems maintains strict test isolation."""
    # 1. Start Two Sum (hashing-001) which has 'no_reuse_same_element'
    resp1 = client.post("/api/sessions/start", json={
        "problem_id": "hashing-001",
        "persona_id": "standard",
        "model_id": "qwen2.5-coder:3b",
    })
    s1 = resp1.json()["session_id"]
    p1 = loader.load("hashing-001")
    run1 = client.post(f"/api/sessions/{s1}/run", json={"code": p1["reference_solution"]}).json()
    tags1 = [r["tag"] for r in run1["results"]]
    assert "no_reuse_same_element" in tags1

    # 2. Start Two Sum II (two-pointers-167) which does NOT have 'no_reuse_same_element'
    resp2 = client.post("/api/sessions/start", json={
        "problem_id": "two-pointers-167",
        "persona_id": "standard",
        "model_id": "qwen2.5-coder:3b",
    })
    s2 = resp2.json()["session_id"]
    p2 = loader.load("two-pointers-167")
    run2 = client.post(f"/api/sessions/{s2}/run", json={"code": p2["reference_solution"]}).json()
    tags2 = [r["tag"] for r in run2["results"]]
    assert "no_reuse_same_element" not in tags2
    assert "negative_target" in tags2


def test_multiple_dsa_patterns_execution():
    """Test H: Verify tree, graph, sliding window, and binary search patterns execute correctly."""
    patterns_to_test = [
        "sliding-window-003",       # Sliding Window (Longest Substring Without Repeating Characters)
        "binary-search-033",        # Binary Search in Rotated Sorted Array
        "tree-dfs-104",             # Tree DFS (Maximum Depth of Binary Tree)
        "graph-200",                # Graph BFS/DFS (Number of Islands)
    ]

    for prob_id in patterns_to_test:
        p = loader.load(prob_id)
        res = run_problem_tests(p, p["reference_solution"], include_large=True)
        assert res.passed is True
        assert res.tests_passed == res.tests_total
        assert res.tests_total > 0

        # Verify test metadata order matches
        materialized = loader.materialize_tests(p, include_large=True)
        assert len(res.results) == len(materialized)
        for r, m in zip(res.results, materialized):
            assert r.tag == m.get("tag")
