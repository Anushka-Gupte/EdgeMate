"""Persona management and loader for EdgeMate interviewers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


PERSONAS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "interview-kb", "personas")
)


@dataclass
class Persona:
    id: str
    name: str
    description: str
    system_rules: str
    max_words: int
    is_default: bool = False

    def get_prompt_instructions(self) -> str:
        """Format the persona rules into explicit system instructions for LLM."""
        return f"""
INTERVIEWER PERSONA: {self.name.upper()}
{self.system_rules}

IMPORTANT CONSTRAINTS FOR THIS PERSONA:
- Adhere strictly to the tone and word budget ({self.max_words} words max).
- Never evaluate or invent code correctness; only reflect verified results.
- Ask exactly ONE clear question or comment at a time.
"""


_PERSONA_CACHE: Dict[str, Persona] = {}


def load_persona(persona_id: str = "friendly") -> Persona:
    """Load persona configuration from markdown files in interview-kb/personas/."""
    pid = persona_id.lower().strip()
    if pid in _PERSONA_CACHE:
        return _PERSONA_CACHE[pid]

    file_map = {
        "friendly": "friendly.md",
        "standard": "standard.md",
        "silent": "silent.md",
    }
    filename = file_map.get(pid, f"{pid}.md")
    path = os.path.join(PERSONAS_DIR, filename)

    if not os.path.exists(path):
        # Default fallback
        pid = "friendly"
        path = os.path.join(PERSONAS_DIR, "friendly.md")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = lines[0].replace("# Persona:", "").strip() if lines else pid.capitalize()

    # Extract max word limit heuristic from persona definition
    max_words = 80
    if "under 50 words" in content.lower():
        max_words = 50
    elif "at most 12 words" in content.lower() or "12 words" in content.lower():
        max_words = 12
    elif "under 80 words" in content.lower():
        max_words = 80

    # Clean description
    desc_lines = []
    for line in lines[1:]:
        if line.startswith("##"):
            break
        desc_lines.append(line)
    description = " ".join(desc_lines) if desc_lines else "AI Interviewer persona"

    persona = Persona(
        id=pid,
        name=title,
        description=description,
        system_rules=content,
        max_words=max_words,
        is_default=(pid == "friendly"),
    )
    _PERSONA_CACHE[pid] = persona
    return persona


def get_available_personas() -> List[Dict[str, str]]:
    """Return list of all available persona options for UI dropdowns."""
    return [
        {
            "id": "friendly",
            "name": "Friendly (Recommended)",
            "description": "Warm, encouraging, generous hints, perfect for rebuilding confidence.",
        },
        {
            "id": "standard",
            "name": "Standard",
            "description": "Professional, neutral, realistic big-tech interview simulation.",
        },
        {
            "id": "silent",
            "name": "Silent (Stress-test)",
            "description": "Minimal feedback, high pressure, long pauses, realistic stress test.",
        },
    ]
