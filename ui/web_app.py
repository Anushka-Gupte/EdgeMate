"""FastAPI Web Application for EdgeMate — Local Open-Source AI Coding Interviewer.

Provides REST API endpoints and a progressive, state-driven, dark-first, zero-emoji AI Coding Interview experience.
Routes supported:
  /                        -> Landing Page
  /problems                -> Problem Catalog & Setup (119 problems)
  /problems/{id}           -> Problem Preview / Details
  /interview/{id}          -> Progressive Interview Workspace
  /interview/{id}/result   -> Interview Results / Debrief
  /results/{id}            -> Interview Results / Debrief (alias)
  /progress                -> Progress & Analytics
  /benchmark               -> Local Model Benchmark
"""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure interview-kb and base dir are importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import loader
from interview.session import InterviewSession, InterviewPhase
from interview.interviewer import InterviewerAgent
from interview.model import get_model_client, ModelClient
from interview.personas import get_available_personas, load_persona
from storage.database import get_db
from storage.history import ProgressTracker
from tools.run_tests import run_problem_tests
from tools.run_code import run_example_tests
from tools.edge_cases import classify_failure


app = FastAPI(title="EdgeMate", description="Local Open-Source AI Coding Interviewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active session store in memory
_ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}
_TRACKER = ProgressTracker()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    problem_id: str = "sliding-window-003"
    persona_id: str = "friendly"
    model_id: str = "qwen2.5-coder:3b"


class SendMessageRequest(BaseModel):
    message: str


class RunCodeRequest(BaseModel):
    code: str


class SubmitCodeRequest(BaseModel):
    code: str


class BenchmarkRequest(BaseModel):
    model_a: str = "qwen2.5-coder:3b"
    model_b: str = "llama2:latest"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    client = get_model_client()
    status = client.check_connection()
    return {
        "status": "ok",
        "ollama_connected": status["connected"],
        "target_model_installed": status["target_model_installed"],
        "target_model": status["target_model"],
        "available_models": status["models"],
    }


@app.get("/api/problems")
def list_problems(difficulty: str = "all", pattern: str = "all", search: str = ""):
    problems = loader.list_problems()
    results = []
    sq = search.strip().lower()

    for p in problems:
        p_diff = p.get("difficulty", "medium").lower()
        p_pat = p.get("pattern", "").lower()
        p_title = p.get("title", "").lower()
        p_stmt = p.get("statement", "").lower()
        p_lc = str(p.get("leetcode", ""))
        p_id = p.get("id", "").lower()

        if difficulty != "all" and p_diff != difficulty.lower():
            continue
        if pattern != "all" and p_pat != pattern.lower():
            continue
        if sq:
            if not (sq in p_title or sq in p_pat or sq in p_stmt or sq == p_lc or sq in p_id):
                continue

        results.append({
            "id": p["id"],
            "leetcode": p.get("leetcode"),
            "title": p.get("title"),
            "difficulty": p.get("difficulty", "medium"),
            "pattern": p.get("pattern", ""),
            "pattern_display": p.get("pattern", "").replace("_", " ").title(),
        })

    return {"count": len(results), "problems": results}


@app.get("/api/problems/{problem_id}")
def get_problem(problem_id: str):
    try:
        p = loader.load(problem_id)
        return {
            "id": p["id"],
            "leetcode": p.get("leetcode"),
            "title": p.get("title"),
            "difficulty": p.get("difficulty", "medium"),
            "pattern": p.get("pattern", ""),
            "pattern_display": p.get("pattern", "").replace("_", " ").title(),
            "statement": p.get("statement", ""),
            "signature": p.get("signature", ""),
            "function": p.get("function", ""),
            "tests": p.get("tests", [])[:3],
            "complexity": p.get("complexity", {"time": "O(n)", "space": "O(1)"}),
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Problem not found")


@app.get("/api/personas")
def list_personas():
    return {"personas": get_available_personas()}


@app.post("/api/sessions/start")
def start_session(req: StartSessionRequest):
    try:
        problem = loader.load(req.problem_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Problem not found")

    model_client = get_model_client(req.model_id)
    session = InterviewSession(
        problem_id=req.problem_id,
        persona_name=req.persona_id,
        model_name=req.model_id,
    )
    agent = InterviewerAgent(session=session, model_client=model_client)
    greeting = agent.start_interview()

    _ACTIVE_SESSIONS[session.session_id] = {
        "session": session,
        "agent": agent,
    }

    return {
        "session_id": session.session_id,
        "problem_id": req.problem_id,
        "problem_title": problem.get("title"),
        "difficulty": problem.get("difficulty"),
        "pattern": problem.get("pattern"),
        "persona": req.persona_id,
        "model": req.model_id,
        "starter_code": session.candidate_code,
        "greeting": greeting,
        "started_at": session.started_at,
        "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
        "status": session.status,
        "understanding_done": session.understanding_done,
        "approach_done": session.approach_done,
        "optimization_done": session.optimization_done,
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/message")
def send_message(session_id: str, req: SendMessageRequest):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        # Check DB
        db = get_db()
        db_sess = db.get_session(session_id)
        if not db_sess:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        # Re-hydrate session
        session = InterviewSession(
            problem_id=db_sess["problem_id"],
            persona_name=db_sess.get("persona", "friendly"),
            model_name=db_sess.get("model", "qwen2.5-coder:3b"),
            session_id=session_id,
        )
        session.started_at = db_sess["started_at"]
        session.status = db_sess.get("status", "in_progress")
        session.candidate_code = db_sess.get("final_code") or session.candidate_code
        session.conversation = db_sess.get("messages") or []
        agent = InterviewerAgent(session=session, model_client=get_model_client(session.model_name))
        sess_obj = {"session": session, "agent": agent}
        _ACTIVE_SESSIONS[session_id] = sess_obj

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]

    reply = agent.handle_message(req.message)

    return {
        "session_id": session_id,
        "reply": reply,
        "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
        "status": session.status,
        "solved": session.solved,
        "understanding_done": session.understanding_done,
        "approach_done": session.approach_done,
        "optimization_done": session.optimization_done,
        "complexity_answers": session.complexity_answers,
        "complexity_evaluation": session.complexity_evaluation,
        "debrief": session.debrief_content,
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/start_coding")
def start_coding_endpoint(session_id: str):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]
    reply = agent.start_coding()

    return {
        "session_id": session_id,
        "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
        "status": session.status,
        "starter_code": session.candidate_code,
        "reply": reply,
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/run")
def run_code(session_id: str, req: RunCodeRequest):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]
    result, agent_reply = agent.run_examples(req.code)

    return {
        "passed": result.passed,
        "tests_passed": result.tests_passed,
        "tests_total": result.tests_total,
        "execution_time_ms": result.execution_time_ms,
        "failing_tag": result.failing_tag,
        "error_summary": result.error_summary,
        "agent_reply": agent_reply,
        "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
        "status": session.status,
        "results": [
            {
                "test_number": r.test_number,
                "passed": r.passed,
                "tag": r.tag,
                "expected": repr(r.expected),
                "got": repr(r.got) if r.got is not None else None,
                "error": r.error,
                "runtime_ms": r.ms,
                "ms": r.ms,
            }
            for r in result.results
        ],
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/submit")
def submit_code(session_id: str, req: SubmitCodeRequest):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]

    result, agent_reply = agent.submit_solution(req.code)

    return {
        "passed": result.passed,
        "tests_passed": result.tests_passed,
        "tests_total": result.tests_total,
        "execution_time_ms": result.execution_time_ms,
        "failing_tag": result.failing_tag,
        "error_summary": result.error_summary,
        "agent_reply": agent_reply,
        "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
        "status": session.status,
        "solved": session.solved,
        "results": [
            {
                "test_number": r.test_number,
                "passed": r.passed,
                "tag": r.tag,
                "expected": repr(r.expected) if not r.passed else None,
                "got": repr(r.got) if not r.passed else None,
                "error": r.error,
                "runtime_ms": r.ms,
                "ms": r.ms,
            }
            for r in result.results
        ],
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/hint")
def request_hint(session_id: str):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]
    hint_reply = agent.request_hint()

    return {
        "hint": hint_reply,
        "hints_used": session.hints_used,
        "hint_level": session.hint_level,
        "messages": session.conversation,
    }


@app.post("/api/sessions/{session_id}/end")
def end_session(session_id: str):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if not sess_obj:
        db = get_db()
        db_sess = db.get_session(session_id)
        if not db_sess:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        return {
            "session_id": session_id,
            "debrief": db_sess.get("debrief"),
            "summary": db_sess,
            "messages": db_sess.get("messages", []),
        }

    agent: InterviewerAgent = sess_obj["agent"]
    session: InterviewSession = sess_obj["session"]
    debrief = agent.generate_debrief()

    return {
        "session_id": session_id,
        "debrief": debrief,
        "summary": session.to_summary_dict(),
        "messages": session.conversation,
    }


@app.get("/api/sessions/{session_id}")
def get_session_info(session_id: str):
    sess_obj = _ACTIVE_SESSIONS.get(session_id)
    if sess_obj:
        session = sess_obj["session"]
        p_info = loader.load(session.problem_id) if loader else {}
        return {
            "session_id": session.session_id,
            "problem_id": session.problem_id,
            "problem_title": session.problem.get("title") or p_info.get("title", session.problem_id),
            "difficulty": session.problem.get("difficulty") or p_info.get("difficulty", "medium"),
            "pattern": session.problem.get("pattern") or p_info.get("pattern", ""),
            "pattern_display": (session.problem.get("pattern") or p_info.get("pattern", "")).replace("_", " ").title(),
            "persona": session.persona_name,
            "model": session.model_name,
            "phase": session.phase.value if hasattr(session.phase, "value") else str(session.phase),
            "status": session.status,
            "solved": session.solved,
            "hints_used": session.hints_used,
            "tests_passed": session.tests_passed,
            "tests_total": session.tests_total,
            "attempts_count": len(session.attempts),
            "duration_seconds": session.duration_seconds,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "candidate_code": session.candidate_code,
            "debrief": session.debrief_content,
            "complexity_answers": session.complexity_answers,
            "complexity_evaluation": session.complexity_evaluation,
            "messages": session.conversation,
            "attempts": session.attempts,
        }

    # If not in memory, query persistent SQLite DB
    db = get_db()
    db_sess = db.get_session(session_id)
    if db_sess:
        p_info = {}
        if loader:
            try:
                p_info = loader.load(db_sess["problem_id"])
            except Exception:
                pass
        
        raw_pat = db_sess.get("pattern") or p_info.get("pattern", "")
        title = db_sess.get("problem_title") or p_info.get("title", db_sess["problem_id"])

        return {
            "session_id": db_sess["id"],
            "problem_id": db_sess["problem_id"],
            "problem_title": title,
            "difficulty": db_sess.get("difficulty") or p_info.get("difficulty", "medium"),
            "pattern": raw_pat,
            "pattern_display": raw_pat.replace("_", " ").title(),
            "persona": db_sess.get("persona", "friendly"),
            "model": db_sess.get("model", "qwen2.5-coder:3b"),
            "phase": db_sess.get("phase", "complete"),
            "status": db_sess.get("status", "completed" if db_sess.get("ended_at") else "in_progress"),
            "solved": bool(db_sess.get("solved")),
            "hints_used": db_sess.get("hints_used", 0),
            "tests_passed": db_sess.get("tests_passed", 0),
            "tests_total": db_sess.get("tests_total", 0),
            "attempts_count": db_sess.get("attempts_count", len(db_sess.get("attempts", []))),
            "duration_seconds": db_sess.get("duration_seconds", 0),
            "started_at": db_sess.get("started_at"),
            "ended_at": db_sess.get("ended_at"),
            "candidate_code": db_sess.get("final_code", ""),
            "debrief": db_sess.get("debrief"),
            "messages": db_sess.get("messages", []),
            "attempts": db_sess.get("attempts", []),
        }

    raise HTTPException(status_code=404, detail="Interview session not found")


@app.get("/api/progress")
def get_progress():
    return _TRACKER.get_summary()


@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    client = get_model_client()
    prompt = "The candidate submitted code with an edge-case bug on empty strings. Give a friendly one-sentence hint under 20 words asking about empty inputs."
    system = "You are a friendly coding interviewer. Be encouraging, concise, under 20 words."

    res1 = client.benchmark_model(req.model_a, prompt, system)
    res2 = client.benchmark_model(req.model_b, prompt, system)

    return {
        "model_a": res1,
        "model_b": res2,
    }


# ---------------------------------------------------------------------------
# Single Page Application HTML / Frontend
# ---------------------------------------------------------------------------

SPA_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeMate — Local Open-Source AI Coding Interviewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg-base: #080b14;
            --bg-surface: #0f172a;
            --bg-surface-elevated: #1e293b;
            --bg-input: #0a0f1d;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-blue: #3b82f6;
            --color-success: #10b981;
            --color-warning: #f59e0b;
            --color-danger: #ef4444;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', Consolas, Monaco, 'Courier New', monospace;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        a { color: inherit; text-decoration: none; }
        button { font-family: inherit; cursor: pointer; border: none; outline: none; transition: all 0.15s ease; }

        /* Navbar */
        .navbar {
            position: sticky;
            top: 0;
            z-index: 50;
            background-color: rgba(8, 11, 20, 0.92);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-subtle);
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }

        .brand-logo {
            width: 26px;
            height: 26px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-blue));
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 0.85rem;
        }

        .nav-links { display: flex; align-items: center; gap: 4px; }
        .nav-link {
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.15s ease;
        }
        .nav-link:hover { color: var(--text-primary); background-color: var(--bg-surface-elevated); }
        .nav-link.active { color: var(--text-primary); background-color: var(--bg-surface); border: 1px solid var(--border-subtle); }

        .nav-status { display: flex; align-items: center; gap: 8px; }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background-color: var(--color-success); }
        .status-dot.offline { background-color: var(--color-warning); }

        /* Containers */
        .main-content { flex: 1; display: flex; flex-direction: column; }
        .container { width: 100%; max-width: 1280px; margin: 0 auto; padding: 32px 24px; }
        .container-narrow { max-width: 860px; }

        .page-view { display: none; animation: fadeIn 0.15s ease; }
        .page-view.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 9px 18px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.15s ease;
            white-space: nowrap;
        }
        .btn-primary { background-color: var(--accent-primary); color: #ffffff; }
        .btn-primary:hover { background-color: var(--accent-hover); box-shadow: 0 0 12px rgba(99, 102, 241, 0.4); }
        .btn-secondary { background-color: var(--bg-surface-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); }
        .btn-secondary:hover { background-color: #334155; border-color: rgba(255, 255, 255, 0.15); }
        .btn-danger { background-color: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
        .btn-danger:hover { background-color: var(--color-danger); color: #ffffff; }
        .btn-lg { padding: 12px 24px; font-size: 1rem; border-radius: 8px; }
        .btn-sm { padding: 6px 12px; font-size: 0.8rem; }

        /* Forms */
        .form-group { margin-bottom: 16px; }
        .form-label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
        .form-select, .form-input {
            width: 100%;
            background-color: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 9px 12px;
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.15s ease;
        }
        .form-select:focus, .form-input:focus { border-color: var(--accent-primary); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2); }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .badge-easy { background-color: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-hard { background-color: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-pattern { background-color: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
        .badge-completed { background-color: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-progress { background-color: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-incomplete { background-color: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Cards */
        .card { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 20px; }

        /* Hero */
        .hero { padding: 64px 0 48px; display: grid; grid-template-columns: 1.2fr 1fr; gap: 48px; align-items: center; }
        .hero-tagline { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent-primary); margin-bottom: 12px; }
        .hero-title { font-size: 2.75rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.03em; color: var(--text-primary); margin-bottom: 18px; }
        .hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 28px; max-width: 540px; }
        .hero-ctas { display: flex; gap: 12px; align-items: center; margin-bottom: 20px; }
        .hero-positioning { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
        .hero-preview { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 10px; overflow: hidden; box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6); }
        .preview-header { background-color: var(--bg-surface-elevated); border-bottom: 1px solid var(--border-subtle); padding: 10px 14px; display: flex; align-items: center; gap: 6px; }
        .preview-dot { width: 10px; height: 10px; border-radius: 50%; background-color: rgba(255, 255, 255, 0.2); }
        .preview-content { padding: 16px; font-family: var(--font-mono); font-size: 0.8rem; color: #cbd5e1; line-height: 1.6; }

        .steps-section { padding: 64px 0; border-top: 1px solid var(--border-subtle); }
        .section-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 8px; color: var(--text-primary); }
        .section-subtitle { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 32px; }
        .steps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
        .step-card { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 24px; }
        .step-num { font-family: var(--font-mono); font-size: 0.85rem; font-weight: 700; color: var(--accent-primary); margin-bottom: 12px; }
        .step-title { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
        .step-desc { font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5; }
        .features-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 32px; }
        .feature-card { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 20px; }
        .feature-title { font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px; }
        .feature-desc { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5; }
        .cta-banner { background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-surface-elevated) 100%); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 48px; text-align: center; margin: 48px 0 64px; }

        /* Tables */
        .table-container { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden; }
        .data-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
        .data-table th { background-color: var(--bg-surface-elevated); padding: 12px 16px; font-weight: 600; color: var(--text-secondary); border-bottom: 1px solid var(--border-subtle); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .data-table td { padding: 14px 16px; border-bottom: 1px solid var(--border-subtle); color: var(--text-primary); }
        .data-table tr:last-child td { border-bottom: none; }
        .data-table tr:hover td { background-color: rgba(255, 255, 255, 0.02); }
        .table-action-link { color: var(--accent-primary); font-weight: 600; font-size: 0.85rem; }
        .table-action-link:hover { text-decoration: underline; }

        /* ===========================================================
           PROGRESSIVE INTERVIEW WORKSPACE STYLING
           =========================================================== */

        .interview-container {
            display: flex;
            flex-direction: column;
            min-height: calc(100vh - 60px);
            background-color: var(--bg-base);
        }

        /* Top Interview Header & Stage Timeline */
        .interview-header-bar {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            padding: 12px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .interview-meta-title {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .interview-timeline {
            display: flex;
            align-items: center;
            gap: 6px;
            background-color: var(--bg-surface-elevated);
            padding: 4px 8px;
            border-radius: 9999px;
            border: 1px solid var(--border-subtle);
        }

        .timeline-step {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            transition: all 0.15s ease;
        }

        .timeline-step.active {
            background-color: var(--accent-primary);
            color: #ffffff;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.4);
        }

        .timeline-step.completed {
            color: var(--color-success);
        }

        .timeline-step .step-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: currentColor;
        }

        /* Stage 1 & 2: Discussion Stage (Understanding & Approach) */
        .discussion-layout {
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            padding: 32px 24px;
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 28px;
            align-items: start;
        }

        .problem-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 24px;
        }

        .problem-statement {
            font-size: 0.95rem;
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 20px;
        }

        .example-box {
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 10px 14px;
            font-family: var(--font-mono);
            font-size: 0.85rem;
            margin-bottom: 8px;
        }

        .interviewer-panel {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            height: 600px;
            overflow: hidden;
        }

        .interviewer-panel-header {
            background-color: var(--bg-surface-elevated);
            border-bottom: 1px solid var(--border-subtle);
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .chat-card {
            max-width: 88%;
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .chat-card.interviewer {
            align-self: flex-start;
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
        }

        .chat-card.candidate {
            align-self: flex-end;
            background-color: #1e3a8a;
            border: 1px solid #2563eb;
            color: #f8fafc;
        }

        .chat-speaker-tag {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .chat-card.interviewer .chat-speaker-tag { color: var(--accent-primary); }
        .chat-card.candidate .chat-speaker-tag { color: #93c5fd; }

        .chat-input-area {
            background-color: var(--bg-surface-elevated);
            border-top: 1px solid var(--border-subtle);
            padding: 16px;
        }

        .chat-textarea {
            width: 100%;
            background-color: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 10px 12px;
            font-family: inherit;
            font-size: 0.9rem;
            resize: none;
            outline: none;
            height: 72px;
            margin-bottom: 10px;
        }

        .chat-textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        /* Stage 3: Approach Summary Card */
        .stage-center-card {
            max-width: 760px;
            margin: 48px auto;
            width: 100%;
            padding: 32px;
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
        }

        .checklist-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            font-size: 0.95rem;
            color: var(--text-primary);
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .checklist-check {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--color-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 800;
        }

        /* Stage 4 & 5: Coding Mode */
        .coding-mode-layout {
            display: grid;
            grid-template-columns: 1fr 1.25fr;
            height: calc(100vh - 120px);
            overflow: hidden;
        }

        .coding-left-panel {
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-subtle);
            background-color: var(--bg-surface);
            overflow: hidden;
        }

        .coding-right-panel {
            display: flex;
            flex-direction: column;
            background-color: var(--bg-input);
            overflow: hidden;
        }

        .collapsible-problem-header {
            padding: 12px 18px;
            background-color: var(--bg-surface-elevated);
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .coding-interviewer-card {
            background-color: var(--bg-surface-elevated);
            border-bottom: 1px solid var(--border-subtle);
            padding: 14px 18px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }

        .coding-chat-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .coding-editor-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .coding-textarea {
            flex: 1;
            width: 100%;
            background-color: transparent;
            color: #f8fafc;
            font-family: var(--font-mono);
            font-size: 0.95rem;
            line-height: 1.65;
            padding: 20px;
            border: none;
            outline: none;
            resize: none;
            tab-size: 4;
            white-space: pre;
        }

        .coding-actions-bar {
            padding: 12px 20px;
            background-color: var(--bg-surface-elevated);
            border-top: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .results-box {
            border-top: 1px solid var(--border-subtle);
            background-color: var(--bg-surface);
            padding: 14px 18px;
            max-height: 220px;
            overflow-y: auto;
            display: none;
        }

        .results-box.active { display: block; }

        .test-pill-row {
            padding: 6px 10px;
            background-color: var(--bg-surface-elevated);
            border-radius: 4px;
            font-size: 0.85rem;
            margin-top: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Stats Grid */
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
        .stat-card { background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 18px; text-align: center; }
        .stat-value { font-size: 2rem; font-weight: 800; color: var(--text-primary); line-height: 1; margin-bottom: 6px; }
        .stat-label { font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }

        .progress-bar-container { background-color: var(--bg-surface-elevated); border-radius: 4px; height: 8px; overflow: hidden; margin-top: 6px; }
        .progress-bar-fill { background-color: var(--accent-primary); height: 100%; border-radius: 4px; }

        /* Debrief Markdown */
        .debrief-body h3 { color: var(--text-primary); font-size: 1.15rem; margin-top: 18px; margin-bottom: 8px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px; }
        .debrief-body p { color: #cbd5e1; margin-bottom: 12px; line-height: 1.6; }
        .debrief-body ul, .debrief-body ol { margin-left: 20px; margin-bottom: 12px; color: #cbd5e1; }

        @media (max-width: 1024px) {
            .hero { grid-template-columns: 1fr; gap: 32px; }
            .steps-grid { grid-template-columns: 1fr; }
            .features-grid { grid-template-columns: 1fr; }
            .discussion-layout { grid-template-columns: 1fr; }
            .coding-mode-layout { grid-template-columns: 1fr; height: auto; overflow: visible; }
            .coding-left-panel { height: 450px; }
            .coding-right-panel { height: 500px; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>

    <!-- Global Navigation -->
    <nav class="navbar">
        <a href="#/" class="nav-brand" onclick="navigateTo('/'); return false;">
            <span>EdgeMate</span>
        </a>

        <div class="nav-links">
            <a href="#/problems" class="nav-link" id="nav-problems" onclick="navigateTo('/problems'); return false;">Problems</a>
            <a href="#/progress" class="nav-link" id="nav-progress" onclick="navigateTo('/progress'); return false;">Progress</a>
            <a href="#/benchmark" class="nav-link" id="nav-benchmark" onclick="navigateTo('/benchmark'); return false;">Benchmark</a>
        </div>

        <div class="nav-status">
            <div class="status-pill">
                <span class="status-dot" id="ollama-status-dot"></span>
                <span id="ollama-status-text">Local AI</span>
            </div>
            <div class="status-pill">
                <span>Private</span>
            </div>
        </div>
    </nav>

    <!-- Main Views Container -->
    <div class="main-content">

        <!-- ===========================================================
             VIEW 1: LANDING PAGE (/)
             =========================================================== -->
        <div id="view-landing" class="page-view">
            <div class="container">
                <section class="hero">
                    <div>
                        <div class="hero-tagline">Open-Source Local AI Coding Interviewer</div>
                        <h1 class="hero-title">An AI interviewer that conducts the interview. A deterministic runner that tests the code.</h1>
                        <p class="hero-subtitle">
                            EdgeMate conducts realistic, progressive mock interviews. Discuss your approach first, write code only when ready, debug verified failures with guidance, and analyze algorithmic complexity.
                        </p>
                        <div class="hero-ctas">
                            <button class="btn btn-primary btn-lg" onclick="navigateTo('/problems');">Start an Interview</button>
                            <button class="btn btn-secondary btn-lg" onclick="navigateTo('/problems');">Explore Problems</button>
                        </div>
                        <div class="hero-positioning">
                            Local AI &bull; Deterministic Code Testing &bull; Private by Design
                        </div>
                    </div>
                    <div class="hero-preview">
                        <div class="preview-header">
                            <div class="preview-dot"></div>
                            <div class="preview-dot"></div>
                            <div class="preview-dot"></div>
                            <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">progressive_interview.py</span>
                        </div>
                        <div class="preview-content">
                            <span style="color: #6366f1;"># 1. Understanding & Approach Discussion</span><br>
                            <span style="color: #38bdf8;">Interviewer:</span> "Before coding, walk me through how you think about Two Sum."<br><br>
                            <span style="color: #6366f1;"># 2. Unlocked Coding Mode</span><br>
                            <span style="color: #10b981;">Candidate:</span> [Writes two-pointer optimization]<br><br>
                            <span style="color: #6366f1;"># 3. Deterministic Feedback</span><br>
                            <span style="color: #38bdf8;">Interviewer:</span> "One edge case failed on duplicate values. What happens in your loop?"
                        </div>
                    </div>
                </section>

                <section class="steps-section">
                    <h2 class="section-title">The Progressive Interview Loop</h2>
                    <p class="section-subtitle">Real technical interviews are conversations, not simple textboxes.</p>
                    
                    <div class="steps-grid">
                        <div class="step-card">
                            <div class="step-num">01</div>
                            <h3 class="step-title">Understand & Approach</h3>
                            <p class="step-desc">Discuss inputs, outputs, and brute-force approaches before jumping into the code editor.</p>
                        </div>
                        <div class="step-card">
                            <div class="step-num">02</div>
                            <h3 class="step-title">Implement & Test</h3>
                            <p class="step-desc">Write your solution in an unlocked editor. Run examples and receive conversational feedback.</p>
                        </div>
                        <div class="step-card">
                            <div class="step-num">03</div>
                            <h3 class="step-title">Debug & Debrief</h3>
                            <p class="step-desc">Deterministic tests catch actual edge cases. Discuss time/space complexity and get a full debrief.</p>
                        </div>
                    </div>
                </section>
            </div>
        </div>

        <!-- ===========================================================
             VIEW 2: PROBLEM CATALOG & INTERVIEW SETUP (/problems)
             =========================================================== -->
        <div id="view-problems" class="page-view">
            <div class="container">
                <div style="margin-bottom: 24px;">
                    <h1 style="font-size: 1.75rem; font-weight: 700; margin-bottom: 6px;">Problem Catalog & Interview Setup</h1>
                    <p style="color: var(--text-secondary); font-size: 0.95rem;">Browse 119 curated coding interview problems across 20 DSA patterns. Select your interviewer persona and AI model to start practicing.</p>
                </div>

                <div class="card" style="margin-bottom: 24px; padding: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1.2fr 1.5fr; gap: 16px; margin-bottom: 16px;">
                        <div class="form-group">
                            <label class="form-label" style="font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">Difficulty</label>
                            <select id="catalog-diff" class="form-select" onchange="loadCatalogTable();">
                                <option value="all">All Difficulties</option>
                                <option value="easy">Easy</option>
                                <option value="medium">Medium</option>
                                <option value="hard">Hard</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label" style="font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">Pattern</label>
                            <select id="catalog-pat" class="form-select" onchange="loadCatalogTable();">
                                <option value="all">All Patterns (20 DSA Patterns)</option>
                                <option value="sliding_window">01. Sliding Window</option>
                                <option value="two_pointers">02. Two Pointers</option>
                                <option value="fast_slow_pointers">03. Fast & Slow Pointers</option>
                                <option value="binary_search_sorted">04. Binary Search (Sorted)</option>
                                <option value="binary_search_on_answer">05. Binary Search (On Answer)</option>
                                <option value="hashing_frequency">06. Hashing & Frequency</option>
                                <option value="prefix_sum">07. Prefix Sum</option>
                                <option value="difference_array">08. Difference Array</option>
                                <option value="monotonic_stack">09. Monotonic Stack</option>
                                <option value="monotonic_queue">10. Monotonic Queue</option>
                                <option value="heap_top_k">11. Heap & Top-K</option>
                                <option value="intervals">12. Intervals</option>
                                <option value="greedy">13. Greedy</option>
                                <option value="linked_list_manipulation">14. Linked List Manipulation</option>
                                <option value="tree_dfs">15. Tree DFS</option>
                                <option value="tree_bfs">16. Tree BFS</option>
                                <option value="bst">17. Binary Search Tree (BST)</option>
                                <option value="backtracking_basics">18. Backtracking (Basics)</option>
                                <option value="backtracking_constraints">19. Backtracking (Constraints)</option>
                                <option value="graph_bfs_dfs">20. Graph BFS & DFS</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label" style="font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">Search</label>
                            <input type="text" id="catalog-search" class="form-input" placeholder="Search problem title, number, or keyword..." oninput="loadCatalogTable();">
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; border-top: 1px solid var(--border-subtle); padding-top: 16px;">
                        <div class="form-group">
                            <label class="form-label" style="font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">Interviewer Persona</label>
                            <select id="catalog-persona" class="form-select">
                                <option value="friendly">Friendly (Recommended) — Encouraging & Conversational</option>
                                <option value="standard">Standard — Neutral & Direct</option>
                                <option value="silent">Silent — Minimal Feedback Stress Test</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label" style="font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">AI Model</label>
                            <select id="catalog-model" class="form-select">
                                <option value="qwen2.5-coder:3b">qwen2.5-coder:3b (Recommended)</option>
                                <option value="llama2:latest">llama2:latest</option>
                                <option value="mistral:latest">mistral:latest</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th style="width: 40%;">Problem</th>
                                <th style="width: 15%;">Difficulty</th>
                                <th style="width: 25%;">Pattern</th>
                                <th style="width: 20%; text-align: right;">Action</th>
                            </tr>
                        </thead>
                        <tbody id="catalog-tbody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ===========================================================
             VIEW 4: PROBLEM DETAILS (/problems/:id)
             =========================================================== -->
        <div id="view-problem-details" class="page-view">
            <div class="container container-narrow">
                <div style="margin-bottom: 20px;">
                    <a href="#/problems" onclick="navigateTo('/problems'); return false;" style="color: var(--text-secondary); font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px; margin-bottom: 12px;">
                        &larr; Back to Problem Catalog
                    </a>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;">
                        <div>
                            <h1 id="detail-title" style="font-size: 1.8rem; font-weight: 700; margin-bottom: 8px;">Problem Title</h1>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <span id="detail-difficulty" class="badge">Medium</span>
                                <span id="detail-pattern" class="badge badge-pattern">Pattern</span>
                            </div>
                        </div>
                        <button class="btn btn-primary" id="detail-practice-btn">Practice This Problem</button>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 24px;">
                    <h3 style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px;">Problem Statement</h3>
                    <p id="detail-statement" style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;"></p>

                    <div style="margin-top: 20px;">
                        <h4 style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px;">Examples</h4>
                        <div id="detail-examples" style="display: flex; flex-direction: column; gap: 8px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===========================================================
             VIEW 5: PROGRESSIVE AI INTERVIEW WORKSPACE (/interview/:id)
             =========================================================== -->
        <div id="view-interview" class="page-view">
            <div class="interview-container">
                
                <!-- Top Timeline & Interview State Header -->
                <div class="interview-header-bar">
                    <div class="interview-meta-title">
                        <span id="iw-title" style="font-weight: 700; font-size: 1.05rem; color: var(--text-primary);">Problem</span>
                        <span id="iw-diff-badge" class="badge">Medium</span>
                        <span id="iw-pat-badge" class="badge badge-pattern">Pattern</span>
                    </div>

                    <!-- Progressive Timeline Bar -->
                    <div class="interview-timeline">
                        <div class="timeline-step" id="step-understanding"><span class="step-dot"></span><span>1. Understanding</span></div>
                        <div class="timeline-step" id="step-approach"><span class="step-dot"></span><span>2. Approach</span></div>
                        <div class="timeline-step" id="step-coding"><span class="step-dot"></span><span>3. Coding</span></div>
                        <div class="timeline-step" id="step-testing"><span class="step-dot"></span><span>4. Testing</span></div>
                        <div class="timeline-step" id="step-debugging"><span class="step-dot"></span><span>5. Debugging</span></div>
                        <div class="timeline-step" id="step-complexity"><span class="step-dot"></span><span>6. Complexity</span></div>
                        <div class="timeline-step" id="step-debrief"><span class="step-dot"></span><span>7. Debrief</span></div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="iw-persona-badge" class="badge badge-pattern">FRIENDLY</span>
                        <button class="btn btn-danger btn-sm" id="iw-end-btn" onclick="endInterview();">End Interview</button>
                    </div>
                </div>

                <!-- STAGE 1 & 2: DISCUSSION CONTAINER (UNDERSTANDING & APPROACH) -->
                <div id="stage-discussion-container" class="discussion-layout" style="display: none;">
                    <!-- Left: Problem Description Card -->
                    <div class="problem-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <h2 style="font-size: 1.25rem; font-weight: 700;" id="disc-prob-title">Problem</h2>
                        </div>
                        <div class="problem-statement" id="disc-prob-statement"></div>
                        
                        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px;">Examples</div>
                        <div id="disc-prob-examples" style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px;"></div>

                        <div style="background-color: var(--bg-surface-elevated); padding: 12px 14px; border-radius: 6px; border: 1px solid var(--border-subtle); font-size: 0.85rem; color: var(--text-secondary);">
                            <span style="font-weight: 600; color: var(--text-primary);">Interview Note:</span> Review the requirements carefully. You will discuss your understanding and approach with the interviewer before coding.
                        </div>
                    </div>

                    <!-- Right: AI Interviewer Chat & Response Box -->
                    <div class="interviewer-panel">
                        <div class="interviewer-panel-header">
                            <div>
                                <span style="font-weight: 700; font-size: 0.9rem;">AI Interviewer</span>
                                <span style="font-size: 0.75rem; color: var(--text-secondary); margin-left: 8px;" id="disc-stage-name">Understanding Stage</span>
                            </div>
                            <span class="status-pill" style="padding: 2px 8px; font-size: 0.7rem;">Active</span>
                        </div>

                        <div id="disc-chat-messages" class="chat-scroll">
                            <!-- Populated dynamically -->
                        </div>

                        <div class="chat-input-area">
                            <textarea id="disc-chat-input" class="chat-textarea" placeholder="Explain the problem in your own words (inputs, outputs, conditions)..."></textarea>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <button class="hint-trigger" onclick="requestHint();" style="font-size: 0.8rem; color: var(--text-secondary); background: none; padding: 4px 8px; border-radius: 4px;">Need clarification?</button>
                                <button class="btn btn-primary btn-sm" id="disc-send-btn" onclick="sendDiscussionMessage();">Send Response</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- STAGE 3: APPROACH REVIEW & START CODING CONTAINER -->
                <div id="stage-approach-review-container" class="stage-center-card" style="display: none;">
                    <div style="margin-bottom: 20px;">
                        <span class="badge badge-pattern" style="margin-bottom: 8px;">Stage 2 Complete</span>
                        <h2 style="font-size: 1.5rem; font-weight: 700; margin-top: 4px;">Approach Discussion Summary</h2>
                        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px;">You have aligned with the interviewer on the problem and solution strategy.</p>
                    </div>

                    <div style="background-color: var(--bg-surface-elevated); padding: 16px 20px; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 24px;">
                        <div class="checklist-item">
                            <div class="checklist-check">&#10003;</div>
                            <div><b>Problem understood:</b> Inputs, outputs, and constraints reviewed</div>
                        </div>
                        <div class="checklist-item">
                            <div class="checklist-check">&#10003;</div>
                            <div><b>Initial approach identified:</b> Brute force / base cases considered</div>
                        </div>
                        <div class="checklist-item" style="border-bottom: none;">
                            <div class="checklist-check">&#10003;</div>
                            <div><b>Optimization discussed:</b> Efficient algorithm selected</div>
                        </div>
                    </div>

                    <div id="approach-review-interviewer-msg" class="chat-card interviewer" style="max-width: 100%; margin-bottom: 24px;">
                        <!-- Interviewer transition quote -->
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <button class="btn btn-secondary" onclick="reopenDiscussionChat();">Ask Another Question</button>
                        <button class="btn btn-primary btn-lg" onclick="startCodingNow();">Start Coding &rarr;</button>
                    </div>
                </div>

                <!-- STAGE 4 & 5: CODING & DEBUGGING CONTAINER -->
                <div id="stage-coding-container" class="coding-mode-layout" style="display: none;">
                    
                    <!-- Left: Problem Summary & Interviewer Conversation -->
                    <div class="coding-left-panel">
                        <div class="collapsible-problem-header" onclick="toggleProblemSummary();">
                            <span>Problem Statement</span>
                            <span id="problem-summary-toggle-icon" style="font-size: 0.8rem;">▲</span>
                        </div>
                        
                        <div id="coding-problem-body" style="padding: 14px 18px; border-bottom: 1px solid var(--border-subtle); max-height: 200px; overflow-y: auto; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                            <!-- Populated dynamically -->
                        </div>

                        <div class="coding-interviewer-card">
                            <div style="flex: 1;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="font-size: 0.75rem; font-weight: 700; color: var(--accent-primary); text-transform: uppercase;">AI Interviewer</span>
                                    <button class="hint-trigger" onclick="requestHint();" style="font-size: 0.75rem; color: var(--text-secondary); background: none; padding: 2px 6px; border-radius: 4px;">Need a nudge?</button>
                                </div>
                                <div id="coding-latest-ai-msg" style="font-size: 0.85rem; color: var(--text-primary); line-height: 1.5;"></div>
                            </div>
                        </div>

                        <div id="coding-chat-messages" class="coding-chat-scroll">
                            <!-- Populated dynamically -->
                        </div>

                        <div class="chat-input-area" style="padding: 10px 14px;">
                            <div style="display: flex; gap: 8px;">
                                <input type="text" id="coding-chat-input" class="form-input" style="flex: 1; padding: 7px 10px; font-size: 0.85rem;" placeholder="Ask interviewer or explain code..." onkeydown="if(event.key==='Enter') sendCodingMessage();">
                                <button class="btn btn-primary btn-sm" onclick="sendCodingMessage();">Send</button>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Python Code Editor & Execution Results -->
                    <div class="coding-right-panel">
                        <div class="panel-header" style="height: 44px; padding: 0 16px; display: flex; justify-content: space-between; align-items: center; background-color: var(--bg-surface-elevated); border-bottom: 1px solid var(--border-subtle);">
                            <span style="font-weight: 600; font-size: 0.85rem;">Python Editor</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Python 3.10+ &bull; Tab indentation enabled</span>
                        </div>

                        <div class="coding-editor-area">
                            <textarea id="coding-editor" class="coding-textarea" spellcheck="false" placeholder="# Write your Python solution here..."></textarea>
                        </div>

                        <!-- Results Drawer -->
                        <div id="coding-results-drawer" class="results-box">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-weight: 700; font-size: 0.85rem;" id="coding-results-title">Execution Results</span>
                                <span id="coding-results-pill" class="result-pill pass">Passed</span>
                            </div>
                            <div id="coding-results-list"></div>
                        </div>

                        <div class="coding-actions-bar">
                            <div style="display: flex; gap: 10px;">
                                <button class="btn btn-secondary btn-sm" id="btn-run-examples" onclick="runExamples();">Run Examples</button>
                                <button class="btn btn-primary btn-sm" id="btn-submit-solution" onclick="submitSolution();">Submit Solution</button>
                            </div>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Ctrl+Enter to run examples</span>
                        </div>
                    </div>
                </div>

                <!-- STAGE 6: COMPLEXITY DISCUSSION CONTAINER -->
                <div id="stage-complexity-container" class="stage-center-card" style="display: none;">
                    <div style="margin-bottom: 20px;">
                        <span class="badge badge-completed" style="margin-bottom: 8px;">All Tests Passed</span>
                        <h2 style="font-size: 1.5rem; font-weight: 700; margin-top: 4px;">Algorithmic Complexity Discussion</h2>
                        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px;">Your code passed all tests. Now let's evaluate its time and space efficiency.</p>
                    </div>

                    <div id="complexity-chat-messages" style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px;">
                        <!-- Interviewer question rendered here -->
                    </div>

                    <div id="complexity-input-section" class="card" style="background-color: var(--bg-surface-elevated); margin-bottom: 20px;">
                        <label class="form-label" id="complexity-input-label">What is the Time Complexity of your solution and why?</label>
                        <div style="display: flex; gap: 10px; margin-top: 8px;">
                            <input type="text" id="complexity-input" class="form-input" placeholder="e.g. O(n) because each element is visited at most twice..." onkeydown="if(event.key==='Enter') submitComplexityAnswer();">
                            <button class="btn btn-primary" id="complexity-submit-btn" onclick="submitComplexityAnswer();">Submit Answer</button>
                        </div>
                    </div>

                    <div id="complexity-review-box" class="card" style="display: none; background-color: var(--bg-surface-elevated); border: 1px solid var(--border-highlight); margin-bottom: 20px;">
                        <h4 style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: var(--accent-primary); margin-bottom: 8px;">Complexity Review</h4>
                        <div id="complexity-review-content" style="font-size: 0.9rem; line-height: 1.6;"></div>
                    </div>

                    <div id="complexity-proceed-actions" style="display: none; text-align: right;">
                        <button class="btn btn-primary btn-lg" onclick="proceedToDebrief();">View Performance Debrief &rarr;</button>
                    </div>
                </div>

                <!-- STAGE 7: DEBRIEF CONTAINER -->
                <div id="stage-debrief-container" class="stage-center-card" style="display: none;">
                    <div style="margin-bottom: 24px;">
                        <span class="badge badge-completed" style="margin-bottom: 8px;">Interview Complete</span>
                        <h2 style="font-size: 1.75rem; font-weight: 800; margin-top: 4px;">Interview Performance Debrief</h2>
                        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px;">Detailed evaluation of your problem-solving, implementation, and complexity analysis.</p>
                    </div>

                    <div class="card" style="margin-bottom: 24px;">
                        <div id="stage-debrief-content" class="debrief-body"></div>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <button class="btn btn-secondary" onclick="navigateTo('/progress');">Back to Progress</button>
                        <div style="display: flex; gap: 10px;">
                            <button class="btn btn-secondary" id="btn-view-full-results" onclick="viewFullResults();">View Full Results</button>
                            <button class="btn btn-primary" onclick="navigateTo('/problems');">Practice Another Problem</button>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <!-- ===========================================================
             VIEW 6: INTERVIEW COMPLETION / RESULTS (/interview/:id/result & /results/:id)
             =========================================================== -->
        <div id="view-debrief" class="page-view">
            <div class="container container-narrow">
                <div id="debrief-not-found" style="display: none;">
                    <div class="card" style="text-align: center; padding: 48px 24px;">
                        <h2 style="font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; color: var(--text-primary);">Interview Not Found</h2>
                        <p style="color: var(--text-secondary); margin-bottom: 24px;">This interview session could not be found or has not been recorded.</p>
                        <button class="btn btn-secondary" onclick="navigateTo('/progress');">Back to Progress</button>
                    </div>
                </div>

                <div id="debrief-main-container">
                    <div style="margin-bottom: 24px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
                            <div>
                                <h1 style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">Interview Results</h1>
                                <div id="debrief-problem-meta" style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px;"></div>
                            </div>
                            <div id="debrief-status-badge"></div>
                        </div>
                    </div>

                    <!-- Performance Summary Card -->
                    <div class="card" style="margin-bottom: 20px; background-color: var(--bg-surface-elevated);">
                        <h3 style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 12px;">Performance Summary</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px;">
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">RESULT</div>
                                <div id="debrief-stat-solved" style="font-weight: 700; font-size: 1.1rem; color: var(--color-success);">Solved</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">CHECKS PASSED</div>
                                <div id="debrief-stat-tests" style="font-weight: 700; font-size: 1.1rem;">13 / 13</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">ATTEMPTS</div>
                                <div id="debrief-stat-attempts" style="font-weight: 700; font-size: 1.1rem;">1</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">HINTS USED</div>
                                <div id="debrief-stat-hints" style="font-weight: 700; font-size: 1.1rem;">1</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">DURATION</div>
                                <div id="debrief-stat-duration" style="font-weight: 700; font-size: 1.1rem;">3m 42s</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">DATE</div>
                                <div id="debrief-stat-date" style="font-weight: 600; font-size: 0.95rem; color: var(--text-secondary);">—</div>
                            </div>
                        </div>
                    </div>

                    <!-- Debrief Markdown Body -->
                    <div class="card" style="margin-bottom: 24px;">
                        <h3 style="font-size: 0.95rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 12px;">Interviewer Report</h3>
                        <div id="debrief-content" class="debrief-body"></div>
                    </div>

                    <!-- Submitted Code View -->
                    <div class="card" style="margin-bottom: 24px;">
                        <h3 style="font-size: 0.95rem; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 12px;">Final Submitted Code</h3>
                        <pre style="background: var(--bg-input); padding: 16px; border-radius: 6px; font-family: var(--font-mono); font-size: 0.85rem; color: #f8fafc; overflow-x: auto; border: 1px solid var(--border-subtle);"><code id="debrief-code"># No code recorded</code></pre>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <button class="btn btn-secondary" onclick="navigateTo('/progress');">Back to Progress</button>
                        <button class="btn btn-primary" onclick="navigateTo('/problems');">Practice Another Problem</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===========================================================
             VIEW 7: PROGRESS & ANALYTICS (/progress)
             =========================================================== -->
        <div id="view-progress" class="page-view">
            <div class="container">
                <div style="margin-bottom: 24px;">
                    <h1 style="font-size: 1.75rem; font-weight: 700; margin-bottom: 6px;">Your Progress</h1>
                    <p style="color: var(--text-secondary); font-size: 0.95rem;">Real analytics computed strictly from your completed interview sessions.</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="prog-attempted">0</div>
                        <div class="stat-label">Problems Attempted</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="prog-solved" style="color: #34d399;">0</div>
                        <div class="stat-label">Problems Solved</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="prog-rate" style="color: #818cf8;">0%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="prog-hints" style="color: #fbbf24;">0.0</div>
                        <div class="stat-label">Average Hints</div>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 28px;">
                    <h3 style="font-size: 1rem; font-weight: 700; margin-bottom: 16px; color: var(--text-primary);">Pattern Mastery</h3>
                    <div id="prog-patterns-list" style="display: flex; flex-direction: column; gap: 12px;"></div>
                </div>

                <div style="margin-bottom: 16px;">
                    <h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 12px;">Recent Interviews</h3>
                    <div class="table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Problem</th>
                                    <th>Pattern</th>
                                    <th>Difficulty</th>
                                    <th>Result</th>
                                    <th>Hints</th>
                                    <th style="text-align: right;">Action</th>
                                </tr>
                            </thead>
                            <tbody id="prog-history-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===========================================================
             VIEW 8: MODEL BENCHMARK (/benchmark)
             =========================================================== -->
        <div id="view-benchmark" class="page-view">
            <div class="container container-narrow">
                <div style="margin-bottom: 24px;">
                    <h1 style="font-size: 1.75rem; font-weight: 700; margin-bottom: 6px;">Local Model Benchmark</h1>
                    <p style="color: var(--text-secondary); font-size: 0.95rem;">Compare local models for interview-style conversations.</p>
                </div>

                <div class="card" style="margin-bottom: 24px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
                        <div class="form-group">
                            <label class="form-label">Model A</label>
                            <select id="bench-model-a" class="form-select">
                                <option value="qwen2.5-coder:3b">qwen2.5-coder:3b</option>
                                <option value="llama2:latest">llama2:latest</option>
                                <option value="mistral:latest">mistral:latest</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Model B</label>
                            <select id="bench-model-b" class="form-select">
                                <option value="llama2:latest">llama2:latest</option>
                                <option value="mistral:latest">mistral:latest</option>
                                <option value="qwen2.5-coder:3b">qwen2.5-coder:3b</option>
                            </select>
                        </div>
                    </div>
                    <button class="btn btn-primary" id="bench-run-btn" onclick="runBenchmark();">Run Comparison</button>
                </div>

                <div id="bench-results-container" class="card" style="display: none;">
                    <h3 style="font-size: 1rem; font-weight: 700; margin-bottom: 16px;">Comparison Results</h3>
                    <div class="table-container" style="margin-bottom: 16px;">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th id="bench-th-a">Model A</th>
                                    <th id="bench-th-b">Model B</th>
                                </tr>
                            </thead>
                            <tbody id="bench-tbody"></tbody>
                        </table>
                    </div>
                    <div id="bench-preview-a" style="margin-bottom: 12px;"></div>
                    <div id="bench-preview-b"></div>
                </div>
            </div>
        </div>

    </div>

    <!-- ===============================================================
         APPLICATION LOGIC & ROUTER
         =============================================================== -->
    <script>
        // Global State
        let allProblems = [];
        let currentSessionId = null;
        let currentProblem = null;
        let currentPhase = 'understanding';
        let isSummaryCollapsed = false;

        // Health Check
        async function checkHealth() {
            try {
                const res = await fetch('/api/health');
                const data = await res.json();
                const dot = document.getElementById('ollama-status-dot');
                const text = document.getElementById('ollama-status-text');
                if (data.ollama_connected) {
                    dot.classList.remove('offline');
                    text.textContent = 'Local AI';
                } else {
                    dot.classList.add('offline');
                    text.textContent = 'Ollama Offline';
                }
            } catch (e) {
                document.getElementById('ollama-status-dot').classList.add('offline');
                document.getElementById('ollama-status-text').textContent = 'Ollama Offline';
            }
        }

        // Time formatting
        function formatLocalDateTime(isoStr) {
            if (!isoStr) return '—';
            try {
                const d = new Date(isoStr);
                if (isNaN(d.getTime())) return isoStr;
                return d.toLocaleString(undefined, {
                    day: '2-digit',
                    month: 'short',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: true
                });
            } catch (e) {
                return isoStr;
            }
        }

        function formatDuration(seconds) {
            if (!seconds || seconds <= 0) return '< 1m';
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            if (mins === 0) return `${secs}s`;
            return `${mins}m ${secs}s`;
        }

        // Router
        function navigateTo(path) {
            window.location.hash = path;
            handleRoute();
        }

        function handleRoute() {
            let hash = window.location.hash.replace('#', '') || '/';
            if (hash.startsWith('/')) hash = hash.substring(1);
            if (!hash) hash = '';

            // Deactivate all views
            document.querySelectorAll('.page-view').forEach(v => v.classList.remove('active'));
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

            const parts = hash.split('?')[0].split('/');
            const root = parts[0];

            if (root === '' || root === 'landing') {
                document.getElementById('view-landing').classList.add('active');
            } else if (root === 'practice') {
                navigateTo('/problems');
            } else if (root === 'problems') {
                if (parts[1]) {
                    document.getElementById('view-problem-details').classList.add('active');
                    document.getElementById('nav-problems').classList.add('active');
                    loadProblemDetailsView(parts[1]);
                } else {
                    document.getElementById('view-problems').classList.add('active');
                    document.getElementById('nav-problems').classList.add('active');
                    loadCatalogTable();
                }
            } else if (root === 'interview') {
                if (parts[1] && parts[2] === 'result') {
                    document.getElementById('view-debrief').classList.add('active');
                    loadDebriefView(parts[1]);
                } else if (parts[1]) {
                    document.getElementById('view-interview').classList.add('active');
                    loadInterviewWorkspace(parts[1]);
                }
            } else if (root === 'results') {
                if (parts[1]) {
                    document.getElementById('view-debrief').classList.add('active');
                    loadDebriefView(parts[1]);
                } else {
                    navigateTo('/progress');
                }
            } else if (root === 'progress') {
                document.getElementById('view-progress').classList.add('active');
                document.getElementById('nav-progress').classList.add('active');
                loadProgressView();
            } else if (root === 'benchmark') {
                document.getElementById('view-benchmark').classList.add('active');
                document.getElementById('nav-benchmark').classList.add('active');
            } else {
                document.getElementById('view-landing').classList.add('active');
            }
        }

        window.addEventListener('hashchange', handleRoute);
        window.addEventListener('DOMContentLoaded', () => {
            checkHealth();
            fetchAllProblems();
            handleRoute();
        });

        async function fetchAllProblems() {
            try {
                const res = await fetch('/api/problems');
                const data = await res.json();
                allProblems = data.problems || [];
            } catch (e) {
                console.error("Failed to load problems", e);
            }
        }

        async function startInterviewWithConfig(probId) {
            if (!probId) return alert("Please select a problem.");
            const personaEl = document.getElementById('catalog-persona');
            const modelEl = document.getElementById('catalog-model');
            const persona = personaEl ? personaEl.value : 'friendly';
            const model = modelEl ? modelEl.value : 'qwen2.5-coder:3b';

            try {
                const res = await fetch('/api/sessions/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_id: probId, persona_id: persona, model_id: model })
                });
                const data = await res.json();
                currentSessionId = data.session_id;
                navigateTo(`/interview/${probId}`);
            } catch (e) {
                alert("Failed to start session: " + e.message);
            }
        }

        // Catalog View
        async function loadCatalogTable() {
            const diff = document.getElementById('catalog-diff').value;
            const pat = document.getElementById('catalog-pat').value;
            const search = document.getElementById('catalog-search').value;

            const res = await fetch(`/api/problems?difficulty=${diff}&pattern=${pat}&search=${encodeURIComponent(search)}`);
            const data = await res.json();
            const tbody = document.getElementById('catalog-tbody');
            tbody.innerHTML = '';

            (data.problems || []).forEach(p => {
                const tr = document.createElement('tr');
                const diffBadge = `<span class="badge badge-${p.difficulty.toLowerCase()}">${p.difficulty}</span>`;
                const lc = p.leetcode ? `<span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 6px;">#${p.leetcode}</span>` : '';
                tr.innerHTML = `
                    <td>
                        <a href="#/problems/${p.id}" onclick="navigateTo('/problems/${p.id}'); return false;" style="font-weight: 600; color: var(--text-primary);">
                            ${p.title}
                        </a>
                        ${lc}
                    </td>
                    <td>${diffBadge}</td>
                    <td><span class="badge badge-pattern">${p.pattern_display}</span></td>
                    <td style="text-align: right; white-space: nowrap;">
                        <button class="btn btn-primary btn-sm" style="margin-right: 8px;" onclick="startInterviewWithConfig('${p.id}');">Start Interview</button>
                        <a href="#/problems/${p.id}" onclick="navigateTo('/problems/${p.id}'); return false;" class="table-action-link">View &rarr;</a>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Problem Details View
        async function loadProblemDetailsView(problemId) {
            try {
                const res = await fetch(`/api/problems/${problemId}`);
                const data = await res.json();
                currentProblem = data;

                document.getElementById('detail-title').textContent = data.title;
                const diffBadge = document.getElementById('detail-difficulty');
                diffBadge.className = `badge badge-${data.difficulty.toLowerCase()}`;
                diffBadge.textContent = data.difficulty;
                document.getElementById('detail-pattern').textContent = data.pattern_display;
                document.getElementById('detail-statement').textContent = data.statement;

                const exContainer = document.getElementById('detail-examples');
                exContainer.innerHTML = '';
                (data.tests || []).forEach((t, i) => {
                    const div = document.createElement('div');
                    div.style.cssText = 'background: var(--bg-surface-elevated); padding: 8px 12px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.85rem; border: 1px solid var(--border-subtle);';
                    div.innerHTML = `<b>Example ${i+1}:</b> ${data.function}(${t.args.map(a => JSON.stringify(a)).join(', ')}) &rarr; <span style="color: var(--color-success);">${JSON.stringify(t.expected)}</span>`;
                    exContainer.appendChild(div);
                });

                document.getElementById('detail-practice-btn').onclick = async () => {
                    await startInterviewWithConfig(problemId);
                };
            } catch (e) {
                console.error("Failed to load problem details", e);
            }
        }

        // ===========================================================
        // PROGRESSIVE INTERVIEW WORKSPACE LOGIC
        // ===========================================================

        async function loadInterviewWorkspace(problemId) {
            try {
                const pRes = await fetch(`/api/problems/${problemId}`);
                if (!pRes.ok) {
                    alert("Problem not found");
                    navigateTo('/problems');
                    return;
                }
                const pData = await pRes.json();
                currentProblem = pData;

                // Set Header Metadata
                document.getElementById('iw-title').textContent = pData.title;
                const diffBadge = document.getElementById('iw-diff-badge');
                diffBadge.className = `badge badge-${pData.difficulty.toLowerCase()}`;
                diffBadge.textContent = pData.difficulty;
                document.getElementById('iw-pat-badge').textContent = pData.pattern_display;

                // Check session
                let sessionData = null;
                if (currentSessionId) {
                    try {
                        const sRes = await fetch(`/api/sessions/${currentSessionId}`);
                        if (sRes.ok) {
                            const sData = await sRes.json();
                            if (sData.problem_id === problemId && sData.status === 'in_progress') {
                                sessionData = sData;
                            }
                        }
                    } catch (e) {
                        console.warn("Session check error", e);
                    }
                }

                if (!sessionData) {
                    const personaEl = document.getElementById('catalog-persona');
                    const modelEl = document.getElementById('catalog-model');
                    const persona = personaEl ? personaEl.value : 'friendly';
                    const model = modelEl ? modelEl.value : 'qwen2.5-coder:3b';
                    const sRes = await fetch('/api/sessions/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ problem_id: problemId, persona_id: persona, model_id: model })
                    });
                    sessionData = await sRes.json();
                    currentSessionId = sessionData.session_id;
                }

                // Render Problem in Discussion view & Coding view
                document.getElementById('disc-prob-title').textContent = pData.title;
                document.getElementById('disc-prob-statement').textContent = pData.statement;
                
                const discEx = document.getElementById('disc-prob-examples');
                discEx.innerHTML = '';
                (pData.tests || []).forEach((t, i) => {
                    const div = document.createElement('div');
                    div.className = 'example-box';
                    div.innerHTML = `<b>Example ${i+1}:</b> Input: <code>${JSON.stringify(t.args)}</code><br>Output: <span style="color: var(--color-success); font-weight: 600;">${JSON.stringify(t.expected)}</span>`;
                    discEx.appendChild(div);
                });

                document.getElementById('coding-problem-body').innerHTML = `
                    <div style="font-weight: 700; margin-bottom: 4px; color: var(--text-primary);">${pData.title}</div>
                    <p style="margin-bottom: 8px;">${pData.statement}</p>
                `;

                // Set Editor code
                const editor = document.getElementById('coding-editor');
                editor.value = sessionData.starter_code || sessionData.candidate_code || `def ${pData.signature}:\n    # Write your solution here\n    pass\n`;
                
                // Tab indentation
                editor.onkeydown = function(e) {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const start = this.selectionStart;
                        const end = this.selectionEnd;
                        this.value = this.value.substring(0, start) + "    " + this.value.substring(end);
                        this.selectionStart = this.selectionEnd = start + 4;
                    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        runExamples();
                    }
                };

                document.getElementById('iw-persona-badge').textContent = (sessionData.persona || 'friendly').toUpperCase();

                // Apply Current Phase
                updateInterviewPhase(sessionData.phase || 'understanding', sessionData);

            } catch (e) {
                console.error("Failed to load interview workspace", e);
            }
        }

        function updateInterviewPhase(phase, sessionData) {
            currentPhase = phase.toLowerCase();

            // Update Timeline Step Highlights
            const steps = ['understanding', 'approach', 'coding', 'testing', 'debugging', 'complexity', 'debrief'];
            steps.forEach(st => {
                const el = document.getElementById(`step-${st}`);
                if (el) {
                    el.classList.remove('active', 'completed');
                    if (st === currentPhase) {
                        el.classList.add('active');
                    } else if (steps.indexOf(st) < steps.indexOf(currentPhase)) {
                        el.classList.add('completed');
                    }
                }
            });

            // Containers
            const discCont = document.getElementById('stage-discussion-container');
            const reviewCont = document.getElementById('stage-approach-review-container');
            const codingCont = document.getElementById('stage-coding-container');
            const compCont = document.getElementById('stage-complexity-container');
            const debriefCont = document.getElementById('stage-debrief-container');

            discCont.style.display = 'none';
            reviewCont.style.display = 'none';
            codingCont.style.display = 'none';
            compCont.style.display = 'none';
            debriefCont.style.display = 'none';

            if (currentPhase === 'understanding' || currentPhase === 'intro') {
                discCont.style.display = 'grid';
                document.getElementById('disc-stage-name').textContent = 'Understanding Check';
                document.getElementById('disc-chat-input').placeholder = 'Explain the problem in your own words (inputs, outputs, constraints)...';
                renderDiscussionChat(sessionData ? sessionData.messages : []);
            } else if (currentPhase === 'approach') {
                discCont.style.display = 'grid';
                document.getElementById('disc-stage-name').textContent = 'Approach Discussion';
                document.getElementById('disc-chat-input').placeholder = 'Explain your approach (start with brute-force or high-level intuition)...';
                renderDiscussionChat(sessionData ? sessionData.messages : []);
            } else if (currentPhase === 'approach_review') {
                reviewCont.style.display = 'block';
                const lastMsg = (sessionData && sessionData.messages && sessionData.messages.length) 
                    ? sessionData.messages[sessionData.messages.length - 1].content 
                    : "Your approach plan looks solid. Let's see how you translate that into code.";
                document.getElementById('approach-review-interviewer-msg').innerHTML = `<div class="chat-speaker-tag">AI Interviewer</div><div>${marked.parse(lastMsg)}</div>`;
            } else if (currentPhase === 'coding' || currentPhase === 'running_examples' || currentPhase === 'test_review' || currentPhase === 'testing' || currentPhase === 'debugging') {
                codingCont.style.display = 'grid';
                renderCodingChat(sessionData ? sessionData.messages : []);
            } else if (currentPhase === 'complexity') {
                compCont.style.display = 'block';
                renderComplexityChat(sessionData ? sessionData.messages : []);
            } else if (currentPhase === 'debrief' || currentPhase === 'complete') {
                debriefCont.style.display = 'block';
                const debriefText = sessionData ? (sessionData.debrief || (sessionData.messages && sessionData.messages.length ? sessionData.messages[sessionData.messages.length - 1].content : '')) : '';
                document.getElementById('stage-debrief-content').innerHTML = marked.parse(debriefText || 'Interview completed.');
            }
        }

        function renderDiscussionChat(messages) {
            const container = document.getElementById('disc-chat-messages');
            container.innerHTML = '';
            (messages || []).forEach(m => {
                const div = document.createElement('div');
                const isBot = m.role === 'interviewer' || m.role === 'assistant';
                div.className = `chat-card ${isBot ? 'interviewer' : 'candidate'}`;
                const sender = isBot ? 'AI Interviewer' : 'You';
                div.innerHTML = `<div class="chat-speaker-tag">${sender}</div><div>${marked.parse(m.content)}</div>`;
                container.appendChild(div);
            });
            container.scrollTop = container.scrollHeight;
        }

        function renderCodingChat(messages) {
            const container = document.getElementById('coding-chat-messages');
            container.innerHTML = '';
            let lastAi = "Take your time writing your solution. When you're ready, run example tests to check your logic.";
            (messages || []).forEach(m => {
                const isBot = m.role === 'interviewer' || m.role === 'assistant';
                if (isBot) lastAi = m.content;
                const div = document.createElement('div');
                div.className = `chat-card ${isBot ? 'interviewer' : 'candidate'}`;
                div.style.padding = '10px 14px';
                const sender = isBot ? 'Interviewer' : 'You';
                div.innerHTML = `<div class="chat-speaker-tag" style="font-size: 0.7rem;">${sender}</div><div style="font-size: 0.85rem;">${marked.parse(m.content)}</div>`;
                container.appendChild(div);
            });
            document.getElementById('coding-latest-ai-msg').innerHTML = marked.parse(lastAi);
            container.scrollTop = container.scrollHeight;
        }

        function renderComplexityChat(messages) {
            const container = document.getElementById('complexity-chat-messages');
            container.innerHTML = '';
            (messages || []).forEach(m => {
                const isBot = m.role === 'interviewer' || m.role === 'assistant';
                const div = document.createElement('div');
                div.className = `chat-card ${isBot ? 'interviewer' : 'candidate'}`;
                const sender = isBot ? 'AI Interviewer' : 'You';
                div.innerHTML = `<div class="chat-speaker-tag">${sender}</div><div>${marked.parse(m.content)}</div>`;
                container.appendChild(div);
            });
        }

        async function sendDiscussionMessage() {
            const input = document.getElementById('disc-chat-input');
            const msg = input.value.trim();
            if (!msg || !currentSessionId) return;

            const btn = document.getElementById('disc-send-btn');
            btn.disabled = true;
            btn.textContent = 'Thinking...';
            input.value = '';

            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                updateInterviewPhase(data.phase, data);
            } catch (e) {
                console.error("Discussion message error", e);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Send Response';
            }
        }

        async function sendCodingMessage() {
            const input = document.getElementById('coding-chat-input');
            const msg = input.value.trim();
            if (!msg || !currentSessionId) return;

            input.value = '';
            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                renderCodingChat(data.messages);
            } catch (e) {
                console.error("Coding chat error", e);
            }
        }

        function reopenDiscussionChat() {
            updateInterviewPhase('approach', { messages: [] });
        }

        async function startCodingNow() {
            if (!currentSessionId) return;
            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/start_coding`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                updateInterviewPhase('coding', data);
            } catch (e) {
                console.error("Start coding error", e);
                updateInterviewPhase('coding', null);
            }
        }

        function toggleProblemSummary() {
            const body = document.getElementById('coding-problem-body');
            const icon = document.getElementById('problem-summary-toggle-icon');
            if (isSummaryCollapsed) {
                body.style.display = 'block';
                icon.textContent = '▲';
                isSummaryCollapsed = false;
            } else {
                body.style.display = 'none';
                icon.textContent = '▼';
                isSummaryCollapsed = true;
            }
        }

        async function runExamples() {
            if (!currentSessionId) return;
            const code = document.getElementById('coding-editor').value;
            const btn = document.getElementById('btn-run-examples');
            btn.disabled = true;
            btn.textContent = 'Running Examples...';

            const drawer = document.getElementById('coding-results-drawer');
            drawer.classList.add('active');

            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/run`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: code })
                });
                const data = await res.json();
                renderExecutionResults(data, "Example Results", false);
                renderCodingChat(data.messages);
            } catch (e) {
                console.error("Run error", e);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Examples';
            }
        }

        async function submitSolution() {
            if (!currentSessionId) return;
            const code = document.getElementById('coding-editor').value;
            const btn = document.getElementById('btn-submit-solution');
            btn.disabled = true;
            btn.textContent = 'Verifying Suite...';

            const drawer = document.getElementById('coding-results-drawer');
            drawer.classList.add('active');

            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/submit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: code })
                });
                const data = await res.json();
                
                if (data.passed) {
                    renderExecutionResults(data, "Solution Review", true);
                    setTimeout(() => {
                        updateInterviewPhase('complexity', data);
                    }, 1200);
                } else {
                    renderExecutionResults(data, "Solution Review", true);
                    updateInterviewPhase('debugging', data);
                }
            } catch (e) {
                console.error("Submit error", e);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Submit Solution';
            }
        }

        function renderExecutionResults(data, title, isFullSubmit) {
            document.getElementById('coding-results-title').textContent = title;
            const pill = document.getElementById('coding-results-pill');
            if (data.passed) {
                pill.className = 'result-pill pass';
                pill.textContent = `All ${data.tests_total} checks passed (${data.execution_time_ms}ms)`;
            } else {
                pill.className = 'result-pill fail';
                pill.textContent = `${data.tests_passed} / ${data.tests_total} checks passed`;
            }

            const list = document.getElementById('coding-results-list');
            list.innerHTML = '';

            if (data.error_summary) {
                const errDiv = document.createElement('div');
                errDiv.style.cssText = 'background: rgba(239, 68, 68, 0.15); border-left: 3px solid var(--color-danger); padding: 8px 10px; color: #fca5a5; font-family: var(--font-mono); font-size: 0.8rem; margin-top: 6px; border-radius: 4px;';
                errDiv.textContent = data.error_summary;
                list.appendChild(errDiv);
            }

            (data.results || []).forEach((r, i) => {
                const testNum = r.test_number || (i + 1);
                const tag = r.tag || 'normal';
                const div = document.createElement('div');
                div.className = 'test-pill-row';
                const icon = r.passed 
                    ? '<span style="color: var(--color-success); font-weight: 700; margin-right: 6px;">&#10003;</span>' 
                    : '<span style="color: var(--color-danger); font-weight: 700; margin-right: 6px;">&#10007;</span>';
                
                div.innerHTML = `
                    <div style="display: flex; align-items: center;">
                        ${icon}
                        <span style="font-weight: 600; color: var(--text-primary);">${isFullSubmit ? 'Check' : 'Example'} ${testNum}</span>
                        <span style="margin-left: 8px; font-size: 0.75rem; background: var(--bg-surface); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border-subtle); color: var(--text-secondary); font-family: var(--font-mono);">[${tag}]</span>
                    </div>
                    <div style="font-size: 0.8rem; color: ${r.passed ? 'var(--color-success)' : 'var(--color-danger)'}; font-weight: 500;">
                        ${r.passed ? 'Passed' : 'Failed'} (${r.runtime_ms ?? r.ms ?? 0}ms)
                    </div>
                `;
                list.appendChild(div);
            });
        }

        async function requestHint() {
            if (!currentSessionId) return;
            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/hint`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                if (currentPhase === 'understanding' || currentPhase === 'approach') {
                    renderDiscussionChat(data.messages);
                } else {
                    renderCodingChat(data.messages);
                }
            } catch (e) {
                console.error("Hint error", e);
            }
        }

        async function submitComplexityAnswer() {
            const input = document.getElementById('complexity-input');
            const msg = input.value.trim();
            if (!msg || !currentSessionId) return;

            input.value = '';
            const btn = document.getElementById('complexity-submit-btn');
            btn.disabled = true;
            btn.textContent = 'Evaluating...';

            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/message`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                renderComplexityChat(data.messages);

                if (data.phase === 'debrief' || data.phase === 'complete') {
                    document.getElementById('complexity-input-section').style.display = 'none';
                    if (data.complexity_evaluation) {
                        const reviewBox = document.getElementById('complexity-review-box');
                        reviewBox.style.display = 'block';
                        reviewBox.innerHTML = `
                            <h4 style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: var(--accent-primary); margin-bottom: 8px;">Complexity Review</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 8px;">
                                <div style="background: var(--bg-surface); padding: 8px 12px; border-radius: 4px;"><b>Your Time:</b> ${data.complexity_evaluation.candidate_time}</div>
                                <div style="background: var(--bg-surface); padding: 8px 12px; border-radius: 4px;"><b>Expected Time:</b> <span style="color: var(--color-success);">${data.complexity_evaluation.expected_time}</span></div>
                                <div style="background: var(--bg-surface); padding: 8px 12px; border-radius: 4px;"><b>Your Space:</b> ${data.complexity_evaluation.candidate_space}</div>
                                <div style="background: var(--bg-surface); padding: 8px 12px; border-radius: 4px;"><b>Expected Space:</b> <span style="color: var(--color-success);">${data.complexity_evaluation.expected_space}</span></div>
                            </div>
                        `;
                    }
                    document.getElementById('complexity-proceed-actions').style.display = 'block';
                } else {
                    document.getElementById('complexity-input-label').textContent = "Now, what is the Space Complexity of your implementation?";
                    document.getElementById('complexity-input').placeholder = "e.g. O(1) auxiliary space...";
                }
            } catch (e) {
                console.error("Complexity error", e);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Submit Answer';
            }
        }

        function proceedToDebrief() {
            updateInterviewPhase('debrief', null);
        }

        function viewFullResults() {
            if (currentSessionId) {
                const finishedId = currentSessionId;
                currentSessionId = null;
                navigateTo(`/results/${finishedId}`);
            } else {
                navigateTo('/progress');
            }
        }

        async function endInterview() {
            if (!currentSessionId) return;
            if (!confirm("End the interview and generate your performance debrief?")) return;

            const endBtn = document.getElementById('iw-end-btn');
            if (endBtn) {
                endBtn.disabled = true;
                endBtn.textContent = 'Compiling Debrief...';
            }

            try {
                const res = await fetch(`/api/sessions/${currentSessionId}/end`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                const finishedId = currentSessionId;
                currentSessionId = null;
                navigateTo(`/results/${finishedId}`);
            } catch (e) {
                console.error("End error", e);
                navigateTo('/progress');
            }
        }

        // ===========================================================
        // DEBRIEF / RESULTS VIEW (/results/:id & /interview/:id/result)
        // ===========================================================

        async function loadDebriefView(sessionId) {
            const notFoundBox = document.getElementById('debrief-not-found');
            const mainContainer = document.getElementById('debrief-main-container');

            try {
                const res = await fetch(`/api/sessions/${sessionId}`);
                if (!res.ok) {
                    notFoundBox.style.display = 'block';
                    mainContainer.style.display = 'none';
                    return;
                }

                const data = await res.json();
                notFoundBox.style.display = 'none';
                mainContainer.style.display = 'block';

                document.getElementById('debrief-problem-meta').textContent = `${data.problem_title || 'Coding Problem'} &bull; ${data.difficulty || 'Medium'} &bull; ${data.pattern_display || 'DSA Pattern'}`;

                const badgeBox = document.getElementById('debrief-status-badge');
                if (data.solved) {
                    badgeBox.innerHTML = '<span class="badge badge-easy" style="font-size: 0.85rem; padding: 4px 12px;">Solved</span>';
                    document.getElementById('debrief-stat-solved').textContent = 'Solved';
                    document.getElementById('debrief-stat-solved').style.color = 'var(--color-success)';
                } else {
                    badgeBox.innerHTML = '<span class="badge badge-hard" style="font-size: 0.85rem; padding: 4px 12px;">Incomplete</span>';
                    document.getElementById('debrief-stat-solved').textContent = 'Incomplete';
                    document.getElementById('debrief-stat-solved').style.color = 'var(--color-danger)';
                }

                document.getElementById('debrief-stat-tests').textContent = `${data.tests_passed || 0} / ${data.tests_total || 0}`;
                document.getElementById('debrief-stat-attempts').textContent = data.attempts_count || (data.attempts ? data.attempts.length : 1);
                document.getElementById('debrief-stat-hints').textContent = data.hints_used || 0;
                document.getElementById('debrief-stat-duration').textContent = formatDuration(data.duration_seconds);
                document.getElementById('debrief-stat-date').textContent = formatLocalDateTime(data.started_at);

                const debriefContent = document.getElementById('debrief-content');
                if (data.debrief) {
                    debriefContent.innerHTML = marked.parse(data.debrief);
                } else {
                    debriefContent.innerHTML = '<p style="color: var(--text-muted);">No debrief recorded for this session.</p>';
                }

                document.getElementById('debrief-code').textContent = data.candidate_code || '# No code submitted.';

            } catch (e) {
                notFoundBox.style.display = 'block';
                mainContainer.style.display = 'none';
            }
        }

        // ===========================================================
        // PROGRESS VIEW (/progress)
        // ===========================================================

        async function loadProgressView() {
            try {
                const res = await fetch('/api/progress');
                const data = await res.json();

                document.getElementById('prog-attempted').textContent = data.total_attempted;
                document.getElementById('prog-solved').textContent = data.total_solved;
                document.getElementById('prog-rate').textContent = `${data.success_rate}%`;
                document.getElementById('prog-hints').textContent = data.avg_hints_used;

                const patContainer = document.getElementById('prog-patterns-list');
                patContainer.innerHTML = '';
                const pats = data.patterns_practiced || {};
                const keys = Object.keys(pats);

                if (keys.length === 0) {
                    patContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No pattern mastery data yet. Complete your first interview to track DSA pattern metrics.</div>';
                } else {
                    keys.forEach(k => {
                        const st = pats[k];
                        const pct = Math.round((st.solved / st.attempted) * 100) || 0;
                        const div = document.createElement('div');
                        div.innerHTML = `
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">
                                <span>${k}</span>
                                <span style="color: var(--text-secondary);">${st.solved}/${st.attempted} solved (${pct}%)</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        `;
                        patContainer.appendChild(div);
                    });
                }

                const tbody = document.getElementById('prog-history-tbody');
                tbody.innerHTML = '';
                const recents = data.recent_sessions || [];

                if (recents.length === 0) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="7" style="text-align: center; padding: 36px 16px;">
                                <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; color: var(--text-primary);">No interviews yet</div>
                                <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 16px;">Complete your first interview to start building your progress history.</div>
                                <button class="btn btn-primary btn-sm" onclick="navigateTo('/problems');">Start an Interview</button>
                            </td>
                        </tr>
                    `;
                } else {
                    recents.forEach(s => {
                        const tr = document.createElement('tr');
                        let resBadge = '<span class="badge badge-incomplete">Incomplete</span>';
                        if (s.status === 'in_progress') {
                            resBadge = '<span class="badge badge-progress">In Progress</span>';
                        } else if (s.solved) {
                            resBadge = '<span class="badge badge-completed">Solved</span>';
                        }

                        let actionBtn = '';
                        if (s.status === 'completed' || s.ended_at) {
                            actionBtn = `<a href="#/results/${s.id}" onclick="navigateTo('/results/${s.id}'); return false;" class="table-action-link">View Results &rarr;</a>`;
                        } else {
                            actionBtn = `<a href="#/interview/${s.problem_id}" onclick="resumeSession('${s.id}', '${s.problem_id}'); return false;" class="table-action-link">Resume &rarr;</a>`;
                        }

                        tr.innerHTML = `
                            <td style="color: var(--text-secondary); font-size: 0.85rem;">${formatLocalDateTime(s.started_at)}</td>
                            <td style="font-weight: 600;">${s.title}</td>
                            <td><span class="badge badge-pattern">${s.pattern}</span></td>
                            <td><span class="badge badge-${s.difficulty.toLowerCase()}">${s.difficulty}</span></td>
                            <td>${resBadge}</td>
                            <td>${s.hints_used}</td>
                            <td style="text-align: right;">${actionBtn}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

            } catch (e) {
                console.error("Progress error", e);
            }
        }

        function resumeSession(sessionId, problemId) {
            currentSessionId = sessionId;
            navigateTo(`/interview/${problemId}`);
        }

        // ===========================================================
        // BENCHMARK VIEW (/benchmark)
        // ===========================================================

        async function runBenchmark() {
            const mA = document.getElementById('bench-model-a').value;
            const mB = document.getElementById('bench-model-b').value;
            const btn = document.getElementById('bench-run-btn');
            btn.disabled = true;
            btn.textContent = 'Benchmarking...';

            try {
                const res = await fetch('/api/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_a: mA, model_b: mB })
                });
                const data = await res.json();
                
                document.getElementById('bench-results-container').style.display = 'block';
                document.getElementById('bench-th-a').textContent = mA;
                document.getElementById('bench-th-b').textContent = mB;

                const tbody = document.getElementById('bench-tbody');
                tbody.innerHTML = `
                    <tr><td><b>Latency</b></td><td>${data.model_a.latency_ms} ms</td><td>${data.model_b.latency_ms} ms</td></tr>
                    <tr><td><b>Word Count</b></td><td>${data.model_a.word_count} words</td><td>${data.model_b.word_count} words</td></tr>
                    <tr><td><b>Status</b></td><td>${data.model_a.success ? '<span style="color: var(--color-success);">Connected</span>' : '<span style="color: var(--color-danger);">Error</span>'}</td><td>${data.model_b.success ? '<span style="color: var(--color-success);">Connected</span>' : '<span style="color: var(--color-danger);">Error</span>'}</td></tr>
                `;

                document.getElementById('bench-preview-a').innerHTML = `<div style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 4px;">Response from ${mA}:</div><div style="background: var(--bg-surface-elevated); padding: 10px; border-radius: 4px; font-size: 0.85rem;">${data.model_a.response || data.model_a.error}</div>`;
                document.getElementById('bench-preview-b').innerHTML = `<div style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 4px;">Response from ${mB}:</div><div style="background: var(--bg-surface-elevated); padding: 10px; border-radius: 4px; font-size: 0.85rem;">${data.model_b.response || data.model_b.error}</div>`;

            } catch (e) {
                alert("Benchmark error: " + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Comparison';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=SPA_HTML)


@app.get("/{full_path:path}", response_class=HTMLResponse)
def catch_all(full_path: str):
    return HTMLResponse(content=SPA_HTML)


def create_web_app():
    return app
