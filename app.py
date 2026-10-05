#!/usr/bin/env python3
"""EdgeMate — Local Open-Source AI Coding Interviewer.

Main entry point for starting the application.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure paths are configured
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "interview-kb")
for path in [BASE_DIR, KB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import uvicorn
import loader
from storage.database import get_db
from interview.model import get_model_client


def validate_environment() -> None:
    """Validate KB integrity, Database setup, and Ollama connectivity."""
    print("=" * 60)
    print("  EdgeMate — Local Open-Source AI Coding Interviewer")
    print("=" * 60)

    # 1. Validate KB
    try:
        problems = loader.list_problems()
        patterns = {p.get("pattern") for p in problems if p.get("pattern")}
        print(f"[OK] Knowledge Base: {len(problems)} problems loaded across {len(patterns)} DSA patterns.")
    except Exception as e:
        print(f"[ERROR] Loading Knowledge Base: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Validate Database
    try:
        db = get_db()
        print(f"[OK] SQLite Database initialized at: {db.db_path}")
    except Exception as e:
        print(f"[ERROR] Initializing database: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Check Ollama & Model
    model_client = get_model_client()
    status = model_client.check_connection()
    if status["connected"]:
        print(f"[OK] Ollama connected at {model_client.host}.")
        if status["target_model_installed"]:
            print(f"[OK] Target model '{model_client.model_name}' is ready.")
        else:
            print(f"[WARN] Target model '{model_client.model_name}' not found locally.")
            print(f"       Available models: {status['models']}")
            print(f"       To pull the primary model, run: ollama pull {model_client.model_name}")
    else:
        print("[WARN] Ollama is not currently reachable.")
        print("       Deterministic test runner will work, but AI chat requires Ollama.")
        print(f"       Start Ollama and run: ollama pull {model_client.model_name}")
    print("-" * 60)


def main() -> None:
    validate_environment()
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"[INFO] Launching EdgeMate UI at http://{host}:{port}")
    uvicorn.run("ui.web_app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
