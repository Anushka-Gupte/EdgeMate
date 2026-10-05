"""Gradio web application for EdgeMate: Local AI Coding Interviewer.

Enhanced with:
- Dark theme UI styling (slate/charcoal dark palette with vibrant accents)
- Difficulty filtering (Easy, Medium, Hard, All)
- Pattern / Tag filtering (20 DSA patterns, All)
- Live search filtering
- Interactive Problem Catalog table (all 119 problems) with quick practice launcher
"""

from __future__ import annotations

import html
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "interview-kb"))
if KB_DIR not in sys.path:
    sys.path.insert(0, KB_DIR)

import loader
from interview.session import InterviewSession, InterviewPhase
from interview.interviewer import InterviewerAgent
from interview.model import get_model_client, ModelClient
from interview.personas import get_available_personas
from storage.database import get_db
from storage.history import ProgressTracker


# ---------------------------------------------------------------------------
# Filter Helpers
# ---------------------------------------------------------------------------

ALL_PATTERNS = [
    ("All Patterns", "all"),
    ("01. Sliding Window", "sliding_window"),
    ("02. Two Pointers", "two_pointers"),
    ("03. Fast & Slow Pointers", "fast_slow_pointers"),
    ("04. Binary Search (Sorted)", "binary_search_sorted"),
    ("05. Binary Search (On Answer)", "binary_search_on_answer"),
    ("06. Hashing & Frequency", "hashing_frequency"),
    ("07. Prefix Sum", "prefix_sum"),
    ("08. Difference Array", "difference_array"),
    ("09. Monotonic Stack", "monotonic_stack"),
    ("10. Monotonic Queue", "monotonic_queue"),
    ("11. Heap & Top-K", "heap_top_k"),
    ("12. Intervals", "intervals"),
    ("13. Greedy", "greedy"),
    ("14. Linked List Manipulation", "linked_list_manipulation"),
    ("15. Tree DFS", "tree_dfs"),
    ("16. Tree BFS", "tree_bfs"),
    ("17. Binary Search Tree (BST)", "bst"),
    ("18. Backtracking (Basics)", "backtracking_basics"),
    ("19. Backtracking (Constraints)", "backtracking_constraints"),
    ("20. Graph BFS & DFS", "graph_bfs_dfs"),
]

DIFFICULTY_CHOICES = [
    ("All Difficulties", "all"),
    ("🟢 Easy", "easy"),
    ("🟡 Medium", "medium"),
    ("🔴 Hard", "hard"),
]


def get_filtered_problems(
    difficulty: str = "all",
    pattern: str = "all",
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """Filter all KB problems by difficulty, pattern tag, and search keywords."""
    problems = loader.list_problems()
    filtered = []
    sq = (search_query or "").strip().lower()

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
            matches_title = sq in p_title
            matches_pat = sq in p_pat
            matches_stmt = sq in p_stmt
            matches_lc = sq == p_lc
            matches_id = sq in p_id
            if not (matches_title or matches_pat or matches_stmt or matches_lc or matches_id):
                continue

        filtered.append(p)

    return filtered


def get_filtered_problem_choices(
    difficulty: str = "all",
    pattern: str = "all",
    search_query: str = "",
) -> List[Tuple[str, str]]:
    """Return dropdown choices (Label, ID) matching filter criteria."""
    filtered = get_filtered_problems(difficulty, pattern, search_query)
    choices = []
    for p in filtered:
        diff = p.get("difficulty", "med").upper()[:3]
        pat = p.get("pattern", "").replace("_", " ").title()
        label = f"[{diff}] {p.get('title')} ({pat})"
        choices.append((label, p["id"]))
    if not choices:
        choices = [("No problems found matching filters", "none")]
    return choices


def get_catalog_table_data(
    difficulty: str = "all",
    pattern: str = "all",
    search_query: str = "",
) -> List[List[Any]]:
    """Format problem catalog data for Gradio Dataframe."""
    filtered = get_filtered_problems(difficulty, pattern, search_query)
    rows = []
    for p in filtered:
        diff = p.get("difficulty", "medium").capitalize()
        pat = p.get("pattern", "").replace("_", " ").title()
        lc = f"#{p.get('leetcode')}" if p.get("leetcode") else "—"
        comp = p.get("complexity", {})
        time_c = comp.get("time", "O(n)")
        space_c = comp.get("space", "O(1)")

        rows.append([
            p["id"],
            lc,
            p.get("title", ""),
            diff,
            pat,
            p.get("function", ""),
            time_c,
            space_c,
        ])
    return rows


# ---------------------------------------------------------------------------
# Markdown Formatters
# ---------------------------------------------------------------------------

def format_problem_markdown(problem: Dict[str, Any]) -> str:
    """Format problem statement, examples, and metadata for dark UI display."""
    if not problem:
        return "*Select a problem to view details.*"

    title = problem.get("title", "")
    diff = problem.get("difficulty", "medium").capitalize()
    diff_color = "#10B981" if diff == "Easy" else ("#F59E0B" if diff == "Medium" else "#EF4444")
    pattern = problem.get("pattern", "").replace("_", " ").title()
    statement = problem.get("statement", "")
    signature = problem.get("signature", "")
    lc = f"LeetCode #{problem.get('leetcode')}" if problem.get("leetcode") else "Standard DSA"

    examples_md = ""
    tests = problem.get("tests", [])[:3]
    if tests:
        examples_md = "<div style='margin-top: 14px; font-weight: 600; color: #94a3b8;'>EXAMPLES</div>\n"
        for i, t in enumerate(tests, 1):
            args_str = ", ".join(repr(a) for a in t.get("args", []))
            exp_str = repr(t.get("expected"))
            tag_str = f" <span style='color: #a855f7; font-size: 0.8em;'>[{t.get('tag')}]</span>" if t.get("tag") else ""
            examples_md += (
                f"<div style='background: #1e293b; border-left: 3px solid #6366f1; padding: 8px 12px; margin-top: 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;'>"
                f"<b>Example {i}</b>{tag_str}: <code>{problem.get('function')}({args_str})</code> &rarr; <span style='color: #10b981;'>{exp_str}</span>"
                f"</div>\n"
            )

    complexity_md = ""
    comp = problem.get("complexity", {})
    if comp:
        complexity_md = (
            f"<div style='margin-top: 12px; font-size: 0.85em; color: #94a3b8;'>"
            f"⚡ <b>Target Complexity:</b> Time: <code style='color: #38bdf8;'>{comp.get('time', 'O(n)')}</code> | "
            f"Space: <code style='color: #38bdf8;'>{comp.get('space', 'O(1)')}</code>"
            f"</div>"
        )

    return f"""<div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
    <h3 style="margin: 0; color: #f8fafc; font-size: 1.35rem;">{title}</h3>
    <div>
        <span style="background-color: {diff_color}; color: #ffffff; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em;">{diff.upper()}</span>
        <span style="background-color: #3b82f6; color: #ffffff; padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; margin-left: 4px;">{pattern}</span>
        <span style="background-color: #334155; color: #cbd5e1; padding: 3px 10px; border-radius: 9999px; font-weight: 500; font-size: 0.8rem; margin-left: 4px;">{lc}</span>
    </div>
</div>

<div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-top: 8px;">
{statement}
</div>

<div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; margin-top: 12px;">
    <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Function Signature</div>
    <code style="color: #60a5fa; font-size: 0.95rem; font-family: monospace;">def {signature}:</code>
</div>

{examples_md}
{complexity_md}
</div>
"""


def format_execution_results_markdown(result: Any, title: str = "Execution Results") -> str:
    """Format test execution results into rich readable dark-themed markdown."""
    if result is None:
        return "<div style='background: #0f172a; border: 1px dashed #334155; border-radius: 8px; padding: 16px; color: #94a3b8; text-align: center;'>No code execution results yet. Click <b>Run Examples</b> or <b>Submit Solution</b>.</div>"

    passed = result.passed
    p_count = result.tests_passed
    total = result.tests_total
    
    status_bg = "#064e3b" if passed else "#450a0a"
    status_border = "#059669" if passed else "#dc2626"
    status_text = "#6ee7b7" if passed else "#fca5a5"
    status_title = "🟢 ALL TESTS PASSED" if passed else f"🔴 FAILED ({p_count}/{total} passed)"

    md = f"""<div style="background: #0f172a; border: 1px solid {status_border}; border-radius: 8px; padding: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">{title}</span>
    <span style="background: {status_bg}; color: {status_text}; border: 1px solid {status_border}; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">{status_title}</span>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-bottom: 12px;">
    <div style="background: #1e293b; padding: 8px 12px; border-radius: 6px;">
        <div style="color: #94a3b8; font-size: 0.75rem;">PASSED</div>
        <div style="color: #f8fafc; font-weight: 700; font-size: 1.1rem;">{p_count} / {total}</div>
    </div>
    <div style="background: #1e293b; padding: 8px 12px; border-radius: 6px;">
        <div style="color: #94a3b8; font-size: 0.75rem;">EXECUTION TIME</div>
        <div style="color: #f8fafc; font-weight: 700; font-size: 1.1rem;">{result.execution_time_ms} ms</div>
    </div>
"""

    if result.failing_tag:
        md += f"""
    <div style="background: #1e293b; padding: 8px 12px; border-radius: 6px;">
        <div style="color: #f87171; font-size: 0.75rem;">FIRST FAILING TAG</div>
        <div style="color: #fca5a5; font-weight: 700; font-size: 0.95rem;">{result.failing_tag}</div>
    </div>"""

    md += "</div>"

    if result.error_summary:
        md += f"""
<div style="background: #450a0a; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 4px; color: #fca5a5; font-family: monospace; font-size: 0.85rem; margin-bottom: 12px;">
    <b>Exception:</b> {html.escape(result.error_summary)}
</div>"""

    if result.results:
        md += "<div style='color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;'>Test Case Breakdown</div>"
        md += "<div style='display: flex; flex-direction: column; gap: 4px;'>"
        for i, r in enumerate(result.results[:6], 1):
            icon = "✅" if r.passed else "❌"
            tag_str = f" <span style='color: #94a3b8; font-size: 0.8em;'>[{r.tag}]</span>" if r.tag else ""
            if r.passed:
                md += f"<div style='background: #1e293b; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; color: #cbd5e1;'>{icon} <b>Test {i}</b>{tag_str}: Passed ({r.ms}ms)</div>"
            else:
                err_msg = r.error or f"Expected {r.expected}, got {r.got}"
                md += f"<div style='background: #1e293b; border-left: 2px solid #ef4444; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; color: #fca5a5;'>{icon} <b>Test {i}</b>{tag_str}: Failed &mdash; <code>{html.escape(str(err_msg))}</code></div>"

        if len(result.results) > 6:
            md += f"<div style='color: #64748b; font-size: 0.8rem; padding: 4px 8px;'>... and {len(result.results) - 6} more hidden tests</div>"
        md += "</div>"

    md += "</div>"
    return md


# ---------------------------------------------------------------------------
# Dark Theme CSS
# ---------------------------------------------------------------------------

DARK_THEME_CSS = """
/* Dark Theme Custom Palette */
:root, body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header styling */
.edgemate-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.edgemate-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.edgemate-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 400;
}

/* Card & Panels */
.gr-panel, .gr-box, div[data-testid="block"] {
    background-color: #0f172a !important;
    border-color: #1e293b !important;
    color: #f1f5f9 !important;
}

/* Inputs, Dropdowns, Textboxes */
.gr-input, .gr-text-input, textarea, input[type="text"], .gr-dropdown {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
.gr-input:focus, textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

/* Chatbot Dark Customization */
.gr-chatbot {
    background-color: #090d16 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
.gr-chatbot .message.user, .gr-chatbot [data-testid="user"] {
    background: #1e3a8a !important;
    color: #f8fafc !important;
    border: 1px solid #2563eb !important;
    border-radius: 10px 10px 2px 10px !important;
}
.gr-chatbot .message.bot, .gr-chatbot [data-testid="bot"] {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px 10px 10px 2px !important;
}

/* Code Editor Dark Styling */
.gr-code {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Stat Cards */
.stat-card-dark {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
.stat-num-dark {
    font-size: 2.2rem;
    font-weight: 800;
    color: #60a5fa;
    line-height: 1;
}
.stat-label-dark {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
    font-weight: 600;
}

/* Tab Headers */
.tab-nav, button.tab-nav-item, .tabs button {
    color: #94a3b8 !important;
    border-color: transparent !important;
    font-weight: 600 !important;
}
.tab-nav button.selected, button.tab-nav-item.selected, .tabs button.selected {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
    background-color: transparent !important;
}

/* Dataframe Table Dark Theme */
.gr-dataframe table {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    border-collapse: collapse !important;
}
.gr-dataframe th {
    background-color: #1e293b !important;
    color: #94a3b8 !important;
    border-bottom: 1px solid #334155 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
.gr-dataframe td {
    border-bottom: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
.gr-dataframe tr:hover td {
    background-color: #1e293b !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio Application Builder
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and configure the complete Dark Theme Gradio interface for EdgeMate."""
    tracker = ProgressTracker()
    initial_problem_choices = get_filtered_problem_choices(difficulty="all", pattern="all", search_query="")
    catalog_initial_rows = get_catalog_table_data(difficulty="all", pattern="all", search_query="")

    with gr.Blocks(title="EdgeMate — Local AI Coding Interviewer") as demo:
        # Inject Dark Theme CSS
        gr.HTML(f"<style>{DARK_THEME_CSS}</style>")
        
        # Session State variable
        session_state = gr.State(value={})

        # Top Banner Header
        gr.HTML(
            """
            <div class="edgemate-header">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <h1 class="edgemate-title">🚀 EdgeMate</h1>
                        <div class="edgemate-subtitle">
                            Local Open-Source AI Coding Interviewer &bull; <i>Code checks. Model talks.</i>
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; color: #38bdf8; font-weight: 600;">
                            🧠 Ollama Local AI
                        </span>
                        <span style="background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; color: #a78bfa; font-weight: 600;">
                            🛡️ Deterministic Sandbox
                        </span>
                        <span style="background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; color: #34d399; font-weight: 600;">
                            🔒 100% Offline & Private
                        </span>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Tabs() as main_tabs:
            
            # ===============================================================
            # TAB 1: INTERVIEW PRACTICE
            # ===============================================================
            with gr.Tab("🎯 Interview Practice", id="tab_interview"):
                
                # Filters Bar for Problem Selection
                with gr.Accordion("🔍 Problem Filters & Selection (Difficulty, Pattern Tags & Search)", open=True):
                    with gr.Row():
                        filter_diff = gr.Dropdown(
                            choices=DIFFICULTY_CHOICES,
                            value="all",
                            label="Difficulty Filter",
                            scale=1,
                        )
                        filter_pattern = gr.Dropdown(
                            choices=ALL_PATTERNS,
                            value="all",
                            label="Pattern / Tag Filter (20 DSA Patterns)",
                            scale=2,
                        )
                        filter_search = gr.Textbox(
                            placeholder="Type title or keyword (e.g. substring, tree, anagram)...",
                            label="Keyword Search",
                            scale=2,
                        )

                    with gr.Row():
                        problem_dropdown = gr.Dropdown(
                            choices=initial_problem_choices,
                            value="sliding-window-003",
                            label="Problem Selector (Filtered list)",
                            scale=3,
                            interactive=True,
                        )
                        persona_dropdown = gr.Dropdown(
                            choices=[(p["name"], p["id"]) for p in get_available_personas()],
                            value="friendly",
                            label="Interviewer Persona",
                            scale=2,
                            interactive=True,
                        )
                        model_dropdown = gr.Dropdown(
                            choices=["qwen2.5-coder:3b", "llama2:latest", "mistral:latest"],
                            value="qwen2.5-coder:3b",
                            label="Ollama Model",
                            scale=2,
                            interactive=True,
                            allow_custom_value=True,
                        )
                        start_btn = gr.Button("✨ Start Interview", variant="primary", scale=2, size="lg")

                # Main 2-column split layout
                with gr.Row():
                    # Left Column: Problem Statement & Code Editor
                    with gr.Column(scale=1):
                        problem_md = gr.HTML(value="<div style='color: #94a3b8; padding: 20px;'>Select a problem and click <b>Start Interview</b> to begin.</div>")
                        
                        code_editor = gr.Code(
                            language="python",
                            label="Python Solution Editor",
                            lines=14,
                            value="# Click 'Start Interview' to load starter template",
                        )

                        with gr.Row():
                            run_btn = gr.Button("▶ Run Examples", variant="secondary", size="md")
                            submit_btn = gr.Button("⚡ Submit Solution", variant="primary", size="md")
                            hint_btn = gr.Button("💡 Request Hint", variant="secondary", size="md")
                            end_btn = gr.Button("🛑 End Interview", variant="stop", size="md")

                        test_output_md = gr.HTML(
                            value="<div style='background: #0f172a; border: 1px dashed #334155; border-radius: 8px; padding: 16px; color: #94a3b8; text-align: center;'>Execution output will appear here.</div>",
                        )

                    # Right Column: AI Interviewer Chat
                    with gr.Column(scale=1):
                        chat_history = gr.Chatbot(
                            label="AI Interviewer",
                            height=550,
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Explain your approach, ask a question, or explain complexity...",
                                label="Your Response to Interviewer",
                                scale=4,
                                lines=2,
                            )
                            send_msg_btn = gr.Button("Send 💬", variant="primary", scale=1)

            # ===============================================================
            # TAB 2: PROBLEM CATALOG (ALL 119 PROBLEMS)
            # ===============================================================
            with gr.Tab("📚 Problem Catalog (119 Problems)", id="tab_catalog"):
                gr.Markdown("### 📋 Complete Problem Catalog")
                gr.Markdown("Browse all 119 curated coding interview problems across 20 core DSA patterns. Filter by difficulty or tag, search keywords, and instantly launch into practice.")

                with gr.Row():
                    cat_filter_diff = gr.Dropdown(
                        choices=DIFFICULTY_CHOICES,
                        value="all",
                        label="Filter Difficulty",
                        scale=1,
                    )
                    cat_filter_pattern = gr.Dropdown(
                        choices=ALL_PATTERNS,
                        value="all",
                        label="Filter Pattern / Tag",
                        scale=2,
                    )
                    cat_search = gr.Textbox(
                        placeholder="Search by title, number, or keyword...",
                        label="Search Catalog",
                        scale=2,
                    )
                    cat_refresh_btn = gr.Button("🔍 Apply Filters", variant="secondary", scale=1)

                catalog_table = gr.Dataframe(
                    headers=["Problem ID", "LeetCode", "Title", "Difficulty", "Pattern", "Function", "Time", "Space"],
                    value=catalog_initial_rows,
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    catalog_select_id = gr.Textbox(
                        label="Selected Problem ID (Click or type problem ID to load)",
                        value="sliding-window-003",
                        scale=3,
                    )
                    launch_catalog_btn = gr.Button("🚀 Load Selected Problem for Practice", variant="primary", scale=2)

            # ===============================================================
            # TAB 3: PROGRESS & ANALYTICS
            # ===============================================================
            with gr.Tab("📊 Progress & Analytics", id="tab_progress"):
                with gr.Row():
                    refresh_progress_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        stat_attempted = gr.HTML(
                            """<div class="stat-card-dark"><div class="stat-num-dark">0</div><div class="stat-label-dark">Problems Attempted</div></div>"""
                        )
                    with gr.Column(scale=1):
                        stat_solved = gr.HTML(
                            """<div class="stat-card-dark"><div class="stat-num-dark" style="color: #34d399;">0</div><div class="stat-label-dark">Problems Solved</div></div>"""
                        )
                    with gr.Column(scale=1):
                        stat_rate = gr.HTML(
                            """<div class="stat-card-dark"><div class="stat-num-dark" style="color: #a78bfa;">0%</div><div class="stat-label-dark">Success Rate</div></div>"""
                        )
                    with gr.Column(scale=1):
                        stat_hints = gr.HTML(
                            """<div class="stat-card-dark"><div class="stat-num-dark" style="color: #fbbf24;">0.0</div><div class="stat-label-dark">Avg Hints Used</div></div>"""
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        patterns_md = gr.Markdown("### 📚 Pattern Mastery\n*No sessions recorded yet.*")
                    with gr.Column(scale=1):
                        insights_md = gr.Markdown("### 💡 Candidate Insights\n*Complete mock interviews to unlock strengths and focus areas.*")

                gr.Markdown("### 🕒 Recent Interview History")
                history_df = gr.Dataframe(
                    headers=["Date", "Problem", "Pattern", "Difficulty", "Persona", "Solved", "Hints"],
                    datatype=["str", "str", "str", "str", "str", "str", "number"],
                    interactive=False,
                )

            # ===============================================================
            # TAB 4: MODEL BENCHMARK & COMPARISON
            # ===============================================================
            with gr.Tab("🔬 Model Benchmark", id="tab_models"):
                gr.Markdown(
                    """
                    ### ⚖️ Local Model Benchmark
                    Compare local open-weight models running on Ollama for latency, word budget adherence, and interviewing tone.
                    """
                )
                with gr.Row():
                    bench_model1 = gr.Dropdown(
                        choices=["qwen2.5-coder:3b", "llama2:latest", "mistral:latest"],
                        value="qwen2.5-coder:3b",
                        label="Model A",
                    )
                    bench_model2 = gr.Dropdown(
                        choices=["llama2:latest", "mistral:latest", "qwen2.5-coder:3b"],
                        value="llama2:latest",
                        label="Model B",
                    )
                    bench_btn = gr.Button("⚡ Run Comparison", variant="primary")

                bench_output_md = gr.Markdown("*Click **Run Comparison** to benchmark models on an interview prompt.*")

        # -------------------------------------------------------------------
        # EVENT HANDLERS
        # -------------------------------------------------------------------

        def on_filter_change(diff: str, pat: str, search: str):
            choices = get_filtered_problem_choices(difficulty=diff, pattern=pat, search_query=search)
            first_val = choices[0][1] if choices and choices[0][1] != "none" else None
            return gr.update(choices=choices, value=first_val)

        def on_catalog_filter_change(diff: str, pat: str, search: str):
            rows = get_catalog_table_data(difficulty=diff, pattern=pat, search_query=search)
            return rows

        def on_catalog_select(evt: gr.SelectData, current_table):
            try:
                row_idx = evt.index[0]
                problem_id = current_table.iloc[row_idx, 0] if hasattr(current_table, "iloc") else current_table[row_idx][0]
                return str(problem_id)
            except Exception:
                return gr.update()

        def on_launch_from_catalog(problem_id: str, persona_id: str, model_id: str):
            if not problem_id or problem_id == "none":
                problem_id = "sliding-window-003"
            return on_start_interview(problem_id, persona_id, model_id)

        def on_start_interview(problem_id: str, persona_id: str, model_id: str):
            if not problem_id or problem_id == "none":
                problem_id = "sliding-window-003"
            problem = loader.load(problem_id)
            model_client = get_model_client(model_id)
            session = InterviewSession(problem_id=problem_id, persona_name=persona_id)
            agent = InterviewerAgent(session=session, model_client=model_client)

            greeting = agent.start_interview()
            prob_md = format_problem_markdown(problem)
            starter_code = session.candidate_code
            init_messages = [{"role": "assistant", "content": greeting}]
            res_md = "<div style='background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; color: #94a3b8;'>✨ <b>New interview started!</b> Review the problem statement and walk the interviewer through your approach in the chat.</div>"

            state_dict = {
                "session": session,
                "agent": agent,
                "problem_id": problem_id,
            }
            return prob_md, starter_code, init_messages, res_md, state_dict

        def on_send_message(user_msg: str, messages: List[Dict[str, str]], state: Dict[str, Any]):
            if not state or "agent" not in state:
                return messages, "", "<div style='color: #f87171; padding: 10px;'>Please click <b>Start Interview</b> first.</div>"
            if not user_msg or not user_msg.strip():
                return messages, "", gr.update()

            agent: InterviewerAgent = state["agent"]
            curr_messages = list(messages or [])
            curr_messages.append({"role": "user", "content": user_msg})

            reply = agent.handle_message(user_msg)
            curr_messages.append({"role": "assistant", "content": reply})

            return curr_messages, "", gr.update()

        def on_run_examples(code: str, state: Dict[str, Any]):
            if not state or "agent" not in state:
                return "<div style='color: #f87171; padding: 10px;'>Please click <b>Start Interview</b> first.</div>"
            agent: InterviewerAgent = state["agent"]
            result, _ = agent.run_examples(code)
            return format_execution_results_markdown(result, title="Example Test Results")

        def on_submit_solution(code: str, messages: List[Dict[str, str]], state: Dict[str, Any]):
            if not state or "agent" not in state:
                return "<div style='color: #f87171; padding: 10px;'>Please click <b>Start Interview</b> first.</div>", messages
            agent: InterviewerAgent = state["agent"]
            curr_messages = list(messages or [])

            result, agent_reply = agent.submit_solution(code)
            curr_messages.append({"role": "assistant", "content": agent_reply})

            result_md = format_execution_results_markdown(result, title="Full Hidden Test Suite Results")
            return result_md, curr_messages

        def on_request_hint(messages: List[Dict[str, str]], state: Dict[str, Any]):
            if not state or "agent" not in state:
                return messages
            agent: InterviewerAgent = state["agent"]
            curr_messages = list(messages or [])
            hint_reply = agent.request_hint()
            curr_messages.append({"role": "assistant", "content": hint_reply})
            return curr_messages

        def on_end_interview(messages: List[Dict[str, str]], state: Dict[str, Any]):
            if not state or "agent" not in state:
                return messages
            agent: InterviewerAgent = state["agent"]
            curr_messages = list(messages or [])
            debrief_reply = agent.generate_debrief()
            curr_messages.append({"role": "assistant", "content": debrief_reply})
            return curr_messages

        def on_refresh_progress():
            summary = tracker.get_summary()
            att = summary["total_attempted"]
            sol = summary["total_solved"]
            rate = summary["success_rate"]
            avg_h = summary["avg_hints_used"]

            c_att = f"""<div class="stat-card-dark"><div class="stat-num-dark">{att}</div><div class="stat-label-dark">Problems Attempted</div></div>"""
            c_sol = f"""<div class="stat-card-dark"><div class="stat-num-dark" style="color: #34d399;">{sol}</div><div class="stat-label-dark">Problems Solved</div></div>"""
            c_rate = f"""<div class="stat-card-dark"><div class="stat-num-dark" style="color: #a78bfa;">{rate}%</div><div class="stat-label-dark">Success Rate</div></div>"""
            c_hints = f"""<div class="stat-card-dark"><div class="stat-num-dark" style="color: #fbbf24;">{avg_h}</div><div class="stat-label-dark">Avg Hints Used</div></div>"""

            # Pattern breakdown
            pat_md = "### 📚 Pattern Mastery Breakdown\n\n"
            if summary["patterns_practiced"]:
                for pat, stats in summary["patterns_practiced"].items():
                    pct = round((stats["solved"] / stats["attempted"]) * 100) if stats["attempted"] else 0
                    pat_md += f"- **{pat}**: `{stats['solved']}/{stats['attempted']} solved` ({pct}%)\n"
            else:
                pat_md += "*No patterns practiced yet.*"

            insights = f"""### 💡 Candidate Insights
- **Strongest Pattern**: `{summary['strongest_pattern']}`
- **Focus / Needs Practice**: `{summary['needs_practice_pattern']}`
- **Recommendation**: Focus on reasoning through edge cases and mastering time/space complexity invariants.
"""

            rows = []
            for s in summary["recent_sessions"]:
                rows.append([
                    s["started_at"],
                    s["title"],
                    s["pattern"],
                    s["difficulty"],
                    s["persona"],
                    "✅ Yes" if s["solved"] else "❌ No",
                    s["hints_used"],
                ])

            return c_att, c_sol, c_rate, c_hints, pat_md, insights, rows

        def on_run_benchmark(m1: str, m2: str):
            client = get_model_client()
            prompt = "The candidate submitted code with an edge-case bug on empty strings. Give a friendly one-sentence hint under 20 words asking about empty inputs."
            system = "You are a friendly coding interviewer. Be encouraging, concise, under 20 words."

            res1 = client.benchmark_model(m1, prompt, system)
            res2 = client.benchmark_model(m2, prompt, system)

            md = f"""### 📊 Model Benchmark Results

| Metric | **{m1}** | **{m2}** |
|---|---|---|
| **Latency** | `{res1['latency_ms']} ms` | `{res2['latency_ms']} ms` |
| **Word Count** | `{res1['word_count']} words` | `{res2['word_count']} words` |
| **Status** | {'🟢 Connected' if res1['success'] else '🔴 Error'} | {'🟢 Connected' if res2['success'] else '🔴 Error'} |

#### Generated Response from `{m1}`:
> {res1['response'] or res1['error']}

#### Generated Response from `{m2}`:
> {res2['response'] or res2['error']}
"""
            return md

        # Filter listeners in Practice Tab
        filter_diff.change(
            fn=on_filter_change,
            inputs=[filter_diff, filter_pattern, filter_search],
            outputs=[problem_dropdown],
        )
        filter_pattern.change(
            fn=on_filter_change,
            inputs=[filter_diff, filter_pattern, filter_search],
            outputs=[problem_dropdown],
        )
        filter_search.input(
            fn=on_filter_change,
            inputs=[filter_diff, filter_pattern, filter_search],
            outputs=[problem_dropdown],
        )

        # Catalog Filter listeners
        cat_filter_diff.change(
            fn=on_catalog_filter_change,
            inputs=[cat_filter_diff, cat_filter_pattern, cat_search],
            outputs=[catalog_table],
        )
        cat_filter_pattern.change(
            fn=on_catalog_filter_change,
            inputs=[cat_filter_diff, cat_filter_pattern, cat_search],
            outputs=[catalog_table],
        )
        cat_search.input(
            fn=on_catalog_filter_change,
            inputs=[cat_filter_diff, cat_filter_pattern, cat_search],
            outputs=[catalog_table],
        )
        cat_refresh_btn.click(
            fn=on_catalog_filter_change,
            inputs=[cat_filter_diff, cat_filter_pattern, cat_search],
            outputs=[catalog_table],
        )

        catalog_table.select(
            fn=on_catalog_select,
            inputs=[catalog_table],
            outputs=[catalog_select_id],
        )

        launch_catalog_btn.click(
            fn=on_launch_from_catalog,
            inputs=[catalog_select_id, persona_dropdown, model_dropdown],
            outputs=[problem_md, code_editor, chat_history, test_output_md, session_state],
        )

        # Practice action buttons
        start_btn.click(
            fn=on_start_interview,
            inputs=[problem_dropdown, persona_dropdown, model_dropdown],
            outputs=[problem_md, code_editor, chat_history, test_output_md, session_state],
        )

        send_msg_btn.click(
            fn=on_send_message,
            inputs=[chat_input, chat_history, session_state],
            outputs=[chat_history, chat_input, test_output_md],
        )
        chat_input.submit(
            fn=on_send_message,
            inputs=[chat_input, chat_history, session_state],
            outputs=[chat_history, chat_input, test_output_md],
        )

        run_btn.click(
            fn=on_run_examples,
            inputs=[code_editor, session_state],
            outputs=[test_output_md],
        )

        submit_btn.click(
            fn=on_submit_solution,
            inputs=[code_editor, chat_history, session_state],
            outputs=[test_output_md, chat_history],
        )

        hint_btn.click(
            fn=on_request_hint,
            inputs=[chat_history, session_state],
            outputs=[chat_history],
        )

        end_btn.click(
            fn=on_end_interview,
            inputs=[chat_history, session_state],
            outputs=[chat_history],
        )

        refresh_progress_btn.click(
            fn=on_refresh_progress,
            inputs=[],
            outputs=[stat_attempted, stat_solved, stat_rate, stat_hints, patterns_md, insights_md, history_df],
        )

        bench_btn.click(
            fn=on_run_benchmark,
            inputs=[bench_model1, bench_model2],
            outputs=[bench_output_md],
        )

        # Initialize with problem 1 on page load
        demo.load(
            fn=on_start_interview,
            inputs=[problem_dropdown, persona_dropdown, model_dropdown],
            outputs=[problem_md, code_editor, chat_history, test_output_md, session_state],
        )

    return demo
