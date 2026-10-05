"""Progress analytics and candidate performance history tracking."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from collections import defaultdict

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

try:
    import loader
except ImportError:
    loader = None

from .database import Database, get_db


class ProgressTracker:
    """Computes candidate progress metrics strictly from stored completed sessions."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
        self._problem_cache: Dict[str, Dict[str, Any]] = {}

    def _get_problem_info(self, problem_id: str) -> Optional[Dict[str, Any]]:
        if problem_id not in self._problem_cache:
            if loader:
                try:
                    self._problem_cache[problem_id] = loader.load(problem_id)
                except Exception:
                    self._problem_cache[problem_id] = None
            else:
                self._problem_cache[problem_id] = None
        return self._problem_cache[problem_id]

    def get_summary(self) -> Dict[str, Any]:
        """Generate a real progress summary computed strictly from actual stored sessions."""
        all_sessions = self.db.get_all_sessions(limit=500)
        
        # Completed sessions count as attempted interviews
        completed_sessions = [
            s for s in all_sessions 
            if s.get("status") == "completed" or (s.get("ended_at") and s.get("status") != "abandoned")
        ]

        total_attempted = len(completed_sessions)
        
        if total_attempted == 0:
            return {
                "total_attempted": 0,
                "total_solved": 0,
                "success_rate": 0.0,
                "avg_hints_used": 0.0,
                "patterns_practiced": {},
                "strongest_pattern": "N/A",
                "needs_practice_pattern": "N/A",
                "recent_sessions": self._format_recent_sessions(all_sessions[:20]),
            }

        total_solved = sum(1 for s in completed_sessions if s.get("solved"))
        success_rate = round((total_solved / total_attempted) * 100, 1) if total_attempted > 0 else 0.0
        total_hints = sum(s.get("hints_used", 0) for s in completed_sessions)
        avg_hints = round(total_hints / total_attempted, 1) if total_attempted > 0 else 0.0

        pattern_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempted": 0, "solved": 0})

        for s in completed_sessions:
            p_info = self._get_problem_info(s["problem_id"])
            raw_pat = s.get("pattern") or (p_info["pattern"] if p_info else "general")
            pattern_name = raw_pat.replace("_", " ").title()
            pattern_stats[pattern_name]["attempted"] += 1
            if s.get("solved"):
                pattern_stats[pattern_name]["solved"] += 1

        strongest_pattern = "N/A"
        needs_practice_pattern = "N/A"

        if pattern_stats:
            # Sort by win rate and volume
            sorted_patterns = sorted(
                pattern_stats.items(),
                key=lambda item: (item[1]["solved"] / item[1]["attempted"], item[1]["attempted"]),
                reverse=True,
            )
            strongest_pattern = sorted_patterns[0][0]
            
            # Pattern with lowest win rate
            needs_practice_cand = sorted(
                pattern_stats.items(),
                key=lambda item: (item[1]["solved"] / item[1]["attempted"], -item[1]["attempted"]),
            )
            needs_practice_pattern = needs_practice_cand[0][0]

        return {
            "total_attempted": total_attempted,
            "total_solved": total_solved,
            "success_rate": success_rate,
            "avg_hints_used": avg_hints,
            "patterns_practiced": dict(pattern_stats),
            "strongest_pattern": strongest_pattern,
            "needs_practice_pattern": needs_practice_pattern,
            "recent_sessions": self._format_recent_sessions(all_sessions[:25]),
        }

    def _format_recent_sessions(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for s in sessions:
            p_info = self._get_problem_info(s["problem_id"])
            title = s.get("problem_title") or (p_info["title"] if p_info else s["problem_id"])
            pattern = s.get("pattern") or (p_info["pattern"] if p_info else "general")
            status = s.get("status") or ("completed" if s.get("ended_at") else "in_progress")

            formatted.append({
                "id": s["id"],
                "session_id": s["id"],
                "problem_id": s["problem_id"],
                "title": title,
                "pattern": pattern.replace("_", " ").title(),
                "persona": s.get("persona", "friendly").capitalize(),
                "difficulty": s.get("difficulty", "medium").capitalize(),
                "status": status,
                "solved": bool(s.get("solved")),
                "hints_used": s.get("hints_used", 0),
                "tests_passed": s.get("tests_passed", 0),
                "tests_total": s.get("tests_total", 0),
                "duration_seconds": s.get("duration_seconds", 0),
                "started_at": s.get("started_at", ""),
                "ended_at": s.get("ended_at", ""),
            })
        return formatted
