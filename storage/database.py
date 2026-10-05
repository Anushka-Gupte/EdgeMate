"""SQLite persistence for EdgeMate interview sessions and code attempts."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "edgemate.db")
)


class Database:
    """Manages SQLite connection and schema migrations for interview sessions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create tables and execute migrations if columns are missing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    problem_id TEXT NOT NULL,
                    problem_title TEXT,
                    pattern TEXT,
                    difficulty TEXT,
                    persona TEXT NOT NULL,
                    model TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'in_progress',
                    phase TEXT DEFAULT 'understanding',
                    solved INTEGER DEFAULT 0,
                    hints_used INTEGER DEFAULT 0,
                    tests_passed INTEGER DEFAULT 0,
                    tests_total INTEGER DEFAULT 0,
                    attempts_count INTEGER DEFAULT 0,
                    final_code TEXT,
                    debrief TEXT,
                    messages TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    tests_passed INTEGER DEFAULT 0,
                    tests_total INTEGER DEFAULT 0,
                    failure_tags TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
                """
            )

            # Migration: Ensure all newer columns exist if migrating an existing DB
            cursor.execute("PRAGMA table_info(sessions)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            migrations = [
                ("problem_title", "TEXT"),
                ("pattern", "TEXT"),
                ("model", "TEXT"),
                ("duration_seconds", "INTEGER DEFAULT 0"),
                ("status", "TEXT DEFAULT 'in_progress'"),
                ("phase", "TEXT DEFAULT 'understanding'"),
                ("attempts_count", "INTEGER DEFAULT 0"),
                ("final_code", "TEXT"),
                ("debrief", "TEXT"),
                ("messages", "TEXT"),
            ]

            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    try:
                        cursor.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass

            conn.commit()

    def save_session(
        self,
        session_id: str,
        problem_id: str,
        persona: str,
        difficulty: str = "medium",
        problem_title: Optional[str] = None,
        pattern: Optional[str] = None,
        model: Optional[str] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        duration_seconds: int = 0,
        status: str = "in_progress",
        phase: str = "understanding",
        solved: bool = False,
        hints_used: int = 0,
        tests_passed: int = 0,
        tests_total: int = 0,
        attempts_count: int = 0,
        final_code: Optional[str] = None,
        debrief: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update an interview session with full metadata."""
        start_ts = started_at or datetime.now(timezone.utc).isoformat()
        messages_json = json.dumps(messages) if messages is not None else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (
                    id, problem_id, problem_title, pattern, difficulty, persona, model,
                    started_at, ended_at, duration_seconds, status, phase,
                    solved, hints_used, tests_passed, tests_total,
                    attempts_count, final_code, debrief, messages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    problem_title=COALESCE(excluded.problem_title, sessions.problem_title),
                    pattern=COALESCE(excluded.pattern, sessions.pattern),
                    difficulty=COALESCE(excluded.difficulty, sessions.difficulty),
                    persona=excluded.persona,
                    model=COALESCE(excluded.model, sessions.model),
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    status=excluded.status,
                    phase=excluded.phase,
                    solved=excluded.solved,
                    hints_used=excluded.hints_used,
                    tests_passed=excluded.tests_passed,
                    tests_total=excluded.tests_total,
                    attempts_count=excluded.attempts_count,
                    final_code=COALESCE(excluded.final_code, sessions.final_code),
                    debrief=COALESCE(excluded.debrief, sessions.debrief),
                    messages=COALESCE(excluded.messages, sessions.messages)
                """,
                (
                    session_id,
                    problem_id,
                    problem_title,
                    pattern,
                    difficulty,
                    persona,
                    model,
                    start_ts,
                    ended_at,
                    duration_seconds,
                    status,
                    phase,
                    1 if solved else 0,
                    hints_used,
                    tests_passed,
                    tests_total,
                    attempts_count,
                    final_code,
                    debrief,
                    messages_json,
                ),
            )
            conn.commit()

    def record_attempt(
        self,
        session_id: str,
        attempt_number: int,
        code: str,
        tests_passed: int,
        tests_total: int,
        failure_tags: Optional[List[str]] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Record a single code execution attempt."""
        ts = created_at or datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(failure_tags or [])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO attempts (
                    session_id, attempt_number, code, tests_passed, tests_total,
                    failure_tags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    attempt_number,
                    code,
                    tests_passed,
                    tests_total,
                    tags_json,
                    ts,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch session by ID with its attempts and parsed messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            session = dict(row)
            
            # Parse messages JSON if present
            if session.get("messages"):
                try:
                    session["messages"] = json.loads(session["messages"])
                except Exception:
                    session["messages"] = []
            else:
                session["messages"] = []

            cursor.execute(
                "SELECT * FROM attempts WHERE session_id = ? ORDER BY attempt_number ASC",
                (session_id,),
            )
            session["attempts"] = [dict(r) for r in cursor.fetchall()]
            return session

    def get_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all sessions ordered by start time."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            sessions = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("messages"):
                    try:
                        d["messages"] = json.loads(d["messages"])
                    except Exception:
                        d["messages"] = []
                else:
                    d["messages"] = []
                sessions.append(d)
            return sessions

    def clear_all_sessions(self) -> None:
        """Clear all session and attempt records (useful for fresh resets)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attempts")
            cursor.execute("DELETE FROM sessions")
            conn.commit()


_db_instance: Optional[Database] = None


def get_db(db_path: Optional[str] = None) -> Database:
    """Get or create singleton database instance."""
    global _db_instance
    if _db_instance is None or db_path is not None:
        _db_instance = Database(db_path=db_path)
    return _db_instance
