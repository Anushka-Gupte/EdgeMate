"""Prompt engineering and structured templates for EdgeMate interview phases."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .personas import Persona


SYSTEM_CORE_PROMPT = """You are EdgeMate, a specialized AI Coding Interviewer conducting a mock technical interview.
You are running locally on the candidate's machine to provide private, low-pressure, realistic practice.

CORE ARCHITECTURAL RULE:
- CODE CHECKS. MODEL TALKS.
- You are NEVER the judge of code correctness. The deterministic test runner already executed the code.
- You must NEVER say "I think your code is correct" or "This looks wrong" on your own authority.
- Only talk about the verified execution results and pre-matched follow-up questions provided to you.
- NEVER write out the full working solution or code replacement for the candidate.
- Ask exactly ONE question or make ONE short point at a time.
- Stay strictly in character according to the persona instructions.
"""


def build_system_prompt(persona: Persona, problem: Dict[str, Any], phase: str) -> str:
    """Construct the minimal, context-controlled system prompt for the current phase."""
    return f"""{SYSTEM_CORE_PROMPT}

{persona.get_prompt_instructions()}

CURRENT PROBLEM CONTEXT:
- Title: {problem.get('title')} ({problem.get('difficulty', 'medium').capitalize()})
- Pattern: {problem.get('pattern', '').replace('_', ' ').title()}
- Function Signature: {problem.get('signature')}
- Current Interview Phase: {phase.upper()}
"""


def build_understanding_prompt(
    candidate_message: str,
    problem: Dict[str, Any],
    persona: Persona,
) -> str:
    """Prompt to evaluate the candidate's understanding of the problem statement, inputs, and constraints."""
    examples_str = ""
    for i, ex in enumerate(problem.get("tests", [])[:2], 1):
        examples_str += f"Example {i}: Input {ex.get('args')}, Output {ex.get('expected')}\n"

    return f"""The candidate is explaining their understanding of "{problem.get('title')}" before writing any code.

CANDIDATE'S EXPLANATION:
\"\"\"{candidate_message}\"\"\"

PROBLEM CONTEXT:
- Title: {problem.get('title')} ({problem.get('difficulty', 'medium')})
- Statement: {problem.get('statement')}
{examples_str}

Interviewer task:
- Evaluate whether the candidate understands the problem inputs, outputs, and constraints.
- If their understanding is solid, acknowledge it warmly in character ({persona.name}) and invite them to explain their initial approach: "Good. Now walk me through how you would approach it. Start with the simplest solution you can think of."
- If they misunderstood anything (e.g. 1-indexed vs 0-indexed, ordering, constraints), gently clarify without giving away the algorithm.
- Keep response under {persona.max_words} words.
"""


def build_approach_prompt(
    candidate_message: str,
    problem: Dict[str, Any],
    persona: Persona,
) -> str:
    """Prompt to evaluate the candidate's proposed approach before coding."""
    pattern = problem.get("pattern", "").replace("_", " ")
    return f"""The candidate is explaining their approach to solving "{problem.get('title')}".

Candidate's explanation:
\"\"\"{candidate_message}\"\"\"

Problem statement:
{problem.get('statement')}
Target pattern: {pattern}

Instructions for the interviewer:
- Acknowledge their proposed idea in character ({persona.name}).
- If they mentioned a brute-force approach (e.g. O(n^2) nested loops), ask what its time complexity would be and whether they can optimize it using {pattern}.
- If they proposed a sound approach or optimization, confirm it and tell them they are ready to write code in the editor!
- DO NOT write the code for them. Keep your response under {persona.max_words} words.
"""


def build_example_review_prompt(
    results: List[Dict[str, Any]],
    persona: Persona,
    problem: Dict[str, Any],
) -> str:
    """Prompt to review example test execution outcomes conversationally."""
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    all_passed = (passed == total and total > 0)

    return f"""The candidate just ran the visible EXAMPLE test cases in the editor.
RESULTS: {passed}/{total} example cases passed.

Interviewer task ({persona.name}):
{'- Congratulate them on passing the examples and suggest submitting the solution for the complete test suite when ready.' if all_passed else '- Note that an example case did not pass. Before changing code, ask them what they think might be happening in that case.'}
- Keep response under {persona.max_words} words.
"""


def build_failure_followup_prompt(
    verified_result: Dict[str, Any],
    followup_question: str,
    persona: Persona,
    candidate_code: str,
) -> str:
    """Prompt to ask the verified follow-up question when code tests fail."""
    passed = verified_result.get("tests_passed", 0)
    total = verified_result.get("tests_total", 0)
    tag = verified_result.get("failing_tag", "edge case")
    error = verified_result.get("error_summary", "")

    return f"""The deterministic test runner just finished running the candidate's code.
VERIFIED TEST RESULTS:
- Tests Passed: {passed} / {total}
- Failing Case Tag: {tag}
- Exception / Details: {error or 'Logical mismatch'}

KB VERIFIED FOLLOW-UP QUESTION:
\"{followup_question}\"

Interviewer task:
- State high-level test outcome briefly ({passed}/{total} tests passed).
- Ask the KB follow-up question naturally in your persona's tone ({persona.name}).
- DO NOT give away the exact failing input or the code fix.
- Length limit: under {persona.max_words} words.
"""


def build_hint_prompt(
    hint_text: str,
    hint_level: int,
    persona: Persona,
) -> str:
    """Prompt to deliver a progressive hint ladder item."""
    level_names = {1: "Level 1 (Clarifying Question)", 2: "Level 2 (Nudge / Direction)", 3: "Level 3 (Concrete Hint)"}
    lvl_name = level_names.get(hint_level, f"Level {hint_level}")

    return f"""The candidate requested a hint.
HINT TIER: {lvl_name}
KB STORED HINT:
\"{hint_text}\"

Interviewer task:
- Deliver this hint in your persona's voice ({persona.name}).
- DO NOT write code for them.
- Keep response under {persona.max_words} words.
"""


def build_complexity_prompt(
    candidate_answer: str,
    expected_complexity: Dict[str, Any],
    asking_for: str,  # "time" or "space"
    persona: Persona,
) -> str:
    """Prompt to evaluate candidate's time or space complexity explanation."""
    expected = expected_complexity.get(asking_for, "optimal")
    followups = expected_complexity.get("followups", [])
    sample_followup = followups[0] if followups else f"Can you explain why the {asking_for} complexity is {expected}?"

    return f"""The candidate has passed all deterministic unit tests!
We are now discussing {asking_for.upper()} COMPLEXITY.

Candidate's answer:
\"\"\"{candidate_answer}\"\"\"

KB Expected {asking_for.upper()} Complexity: {expected}
KB Follow-up Question: {sample_followup}

Interviewer task:
- Check if the candidate's stated complexity matches {expected}.
- If correct, acknowledge briefly and either ask the next complexity question or the follow-up.
- If incorrect, gently nudge them to re-count operations / auxiliary data structures.
- Tone: {persona.name}. Keep under {persona.max_words} words.
"""


def build_debrief_prompt(
    problem: Dict[str, Any],
    session_summary: Dict[str, Any],
) -> str:
    """Prompt to generate the final personalized interview debrief."""
    solved = session_summary.get("solved", False)
    tests_passed = session_summary.get("tests_passed", 0)
    tests_total = session_summary.get("tests_total", 0)
    attempts_count = session_summary.get("attempts_count", 0)
    hints_used = session_summary.get("hints_used", 0)
    complexity_discussed = session_summary.get("complexity_discussed", False)
    failure_tags = session_summary.get("failure_tags", [])

    status_str = "Solved (100% tests passed)" if solved else f"Incomplete ({tests_passed}/{tests_total} tests passed)"
    complexity_str = "Yes" if complexity_discussed else "No"
    failure_tags_str = ", ".join(failure_tags) if failure_tags else "boundary conditions / logic mismatch"

    if solved:
        outcome_instructions = f"""- The candidate SOLVED the problem ({tests_passed}/{tests_total} tests passed).
- Highlight their successful implementation and logical structure.
- Commend their handling of the {problem.get('pattern', '').replace('_', ' ')} pattern."""
    else:
        outcome_instructions = f"""- The candidate did NOT solve the problem ({tests_passed}/{tests_total} tests passed across {attempts_count} attempt(s)).
- CRITICAL: DO NOT claim their solution worked or that they solved the problem.
- Acknowledge their debugging attempt.
- Explicitly advise them to practice handling these failing edge cases: {failure_tags_str}."""

    return f"""The mock interview session is complete. Generate a concise, strictly honest, and encouraging debrief.

SESSION METRICS:
- Problem: {problem.get('title')} ({problem.get('difficulty', 'medium').capitalize()})
- Pattern: {problem.get('pattern', '').replace('_', ' ').title()}
- Status: {status_str}
- Tests Passed: {tests_passed}/{tests_total}
- Hints Used: {hints_used}
- Attempts Count: {attempts_count}
- Complexity Discussed: {complexity_str}

GUIDELINES:
{outcome_instructions}
- Be concise (keep under 180 words).

Format your debrief EXACTLY with these 4 sections:

### What went well
(1-2 concise sentences reflecting their actual performance)

### What to practise
(1-2 actionable sentences addressing their actual mistakes or failed edge cases)

### Interview summary
Problem solved: {'Yes' if solved else 'No'}
Tests passed: {tests_passed}/{tests_total}
Hints used: {hints_used}
Complexity discussed: {complexity_str}

### Next step
(One specific DSA pattern or problem to practice next)
"""
