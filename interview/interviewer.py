"""Interviewer Agent orchestrator implementing 'Code checks, model talks'."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

from sandbox.executor import ExecutionResult
from tools.run_tests import run_problem_tests
from tools.run_code import run_example_tests
from tools.edge_cases import classify_failure, get_failure_followup

from .session import InterviewSession, InterviewPhase
from .model import ModelClient, get_model_client
from .personas import Persona, load_persona
from .prompts import (
    build_system_prompt,
    build_understanding_prompt,
    build_approach_prompt,
    build_example_review_prompt,
    build_failure_followup_prompt,
    build_hint_prompt,
    build_complexity_prompt,
    build_debrief_prompt,
)


class InterviewerAgent:
    """Interviewer agent coordinating deterministic tools, Ollama LLM, and session state."""

    def __init__(
        self,
        session: InterviewSession,
        model_client: Optional[ModelClient] = None,
    ):
        self.session = session
        self.model = model_client or get_model_client()

    @property
    def persona(self) -> Persona:
        return self.session.persona

    @property
    def problem(self) -> Dict[str, Any]:
        return self.session.problem

    def start_interview(self) -> str:
        """Initialize the interview, greet the candidate, and ask for problem understanding."""
        self.session.transition_to(InterviewPhase.UNDERSTANDING)
        title = self.problem.get("title", "Problem")
        difficulty = self.problem.get("difficulty", "medium").capitalize()
        pattern = self.problem.get("pattern", "").replace("_", " ").title()

        if self.persona.id == "friendly":
            welcome = (
                f"Welcome! Let's work through this problem together in a low-pressure environment.\n\n"
                f"Today we're looking at **{title}** ({difficulty} • {pattern}).\n\n"
                f"Before writing any code, I'd like to understand how you think about the problem. "
                f"Can you explain the problem back to me in your own words, and identify any key constraints or conditions?"
            )
        elif self.persona.id == "silent":
            welcome = f"Problem: **{title}** ({difficulty}). State your understanding of inputs, outputs, and constraints."
        else:  # Standard
            welcome = (
                f"Hello. We will be working on **{title}** ({difficulty} • {pattern}).\n\n"
                f"Please take a minute to read the problem description, then explain the problem back to me in your own words before we discuss solutions."
            )

        self.session.add_message("interviewer", welcome)
        return welcome

    def handle_message(self, user_message: str) -> str:
        """Handle candidate conversation messages based on current progressive phase."""
        self.session.add_message("candidate", user_message)

        current_phase = self.session.phase

        if current_phase == InterviewPhase.UNDERSTANDING:
            return self._handle_understanding_response(user_message)

        elif current_phase == InterviewPhase.APPROACH:
            return self._handle_approach_response(user_message)

        elif current_phase == InterviewPhase.APPROACH_REVIEW:
            return self._handle_approach_review_chat(user_message)

        elif current_phase in (InterviewPhase.CODING, InterviewPhase.RUNNING_EXAMPLES, InterviewPhase.TEST_REVIEW, InterviewPhase.DEBUGGING):
            return self._handle_coding_chat(user_message)

        elif current_phase == InterviewPhase.COMPLEXITY:
            return self._handle_complexity_response(user_message)

        elif current_phase in (InterviewPhase.DEBRIEF, InterviewPhase.COMPLETE):
            closing = "The interview is complete! Feel free to review the debrief above or view your full results."
            self.session.add_message("interviewer", closing)
            return closing

        else:
            return self._generic_chat_response(user_message)

    def _handle_understanding_response(self, candidate_message: str) -> str:
        """Evaluate candidate's understanding of the problem and advance to APPROACH."""
        self.session.understanding_done = True
        system_prompt = build_system_prompt(self.persona, self.problem, "understanding")
        prompt = build_understanding_prompt(candidate_message, self.problem, self.persona)

        response = self.model.generate(
            prompt=prompt,
            system=system_prompt,
            history=self.session.conversation[:-1],
            temperature=0.5,
        )

        # Transition to APPROACH
        self.session.transition_to(InterviewPhase.APPROACH)
        self.session.add_message("interviewer", response)
        return response

    def _handle_approach_response(self, candidate_message: str) -> str:
        """Evaluate approach and advance to APPROACH_REVIEW phase."""
        self.session.approach_done = True
        self.session.optimization_done = True
        system_prompt = build_system_prompt(self.persona, self.problem, "approach")
        prompt = build_approach_prompt(candidate_message, self.problem, self.persona)

        response = self.model.generate(
            prompt=prompt,
            system=system_prompt,
            history=self.session.conversation[:-1],
            temperature=0.5,
        )

        # Transition to APPROACH_REVIEW where Start Coding CTA is shown
        self.session.transition_to(InterviewPhase.APPROACH_REVIEW)
        self.session.add_message("interviewer", response)
        return response

    def _handle_approach_review_chat(self, candidate_message: str) -> str:
        """Handle chat during the approach review stage."""
        system_prompt = build_system_prompt(self.persona, self.problem, "approach_review")
        prompt = f"""The candidate sent a message while reviewing the approach plan:
\"{candidate_message}\"

Interviewer task:
- Answer briefly in character ({self.persona.name}).
- If they are ready to code, invite them to click 'Start Coding'.
- Keep response under {self.persona.max_words} words.
"""
        response = self.model.generate(prompt=prompt, system=system_prompt, history=self.session.conversation[:-1], temperature=0.5)
        self.session.add_message("interviewer", response)
        return response

    def start_coding(self) -> str:
        """Transition from approach review into coding mode."""
        self.session.transition_to(InterviewPhase.CODING)
        msg = (
            "You've described your approach. Go ahead and implement it in the editor.\n\n"
            "Take your time. When you're ready, run the examples to check your logic, or submit when you're confident."
        )
        self.session.add_message("interviewer", msg)
        return msg

    def _handle_coding_chat(self, candidate_message: str) -> str:
        """Respond to in-progress questions or comments while candidate writes or debugs code."""
        system_prompt = build_system_prompt(self.persona, self.problem, self.session.phase.value)
        prompt = f"""The candidate sent a message while working on their code:
\"\"\"{candidate_message}\"\"\"

Interviewer task:
- Answer briefly according to your persona ({self.persona.name}).
- Do NOT write the code for them.
- If they are clarifying problem inputs/constraints, answer clearly based on the problem statement:
  {self.problem.get('statement')}
- Keep response under {self.persona.max_words} words.
"""
        response = self.model.generate(
            prompt=prompt,
            system=system_prompt,
            history=self.session.conversation[:-1],
            temperature=0.5,
        )
        self.session.add_message("interviewer", response)
        return response

    def run_examples(self, candidate_code: str) -> Tuple[ExecutionResult, str]:
        """Run candidate code against visible example cases (deterministic) and review results."""
        self.session.transition_to(InterviewPhase.TEST_REVIEW)
        result = run_example_tests(self.problem, candidate_code)
        self.session.last_example_result = result.to_dict()

        if result.passed:
            if self.persona.id == "silent":
                ai_review = f"All {result.tests_passed}/{result.tests_total} examples passed."
            elif self.persona.id == "standard":
                ai_review = f"Your first {result.tests_passed}/{result.tests_total} example cases passed. When you are ready, submit your solution for the complete test suite."
            else:  # Friendly
                ai_review = f"Great start! All {result.tests_passed} example cases pass. Take a quick look for any edge cases, and click **Submit Solution** when you feel confident."
        else:
            if self.persona.id == "silent":
                ai_review = f"Example check failed: {result.failing_tag or 'logic error'}."
            elif self.persona.id == "standard":
                ai_review = f"I noticed an example case did not pass ({result.tests_passed}/{result.tests_total} passed). What do you think is causing that discrepancy before changing the code?"
            else:  # Friendly
                ai_review = f"Your first examples ran: {result.tests_passed}/{result.tests_total} passed. One case failed ({result.failing_tag or 'mismatch'}). Before changing your code, what do you think is happening in that case?"

        self.session.add_message("interviewer", ai_review)
        return result, ai_review

    def submit_solution(self, candidate_code: str) -> Tuple[ExecutionResult, str]:
        """Submit code against the complete deterministic hidden test suite."""
        self.session.transition_to(InterviewPhase.SUBMISSION)

        # DETERMINISTIC RUNNER IS THE SOLE JUDGE
        exec_result = run_problem_tests(self.problem, candidate_code, include_large=True)
        failing_tags = [exec_result.failing_tag] if exec_result.failing_tag else []
        self.session.last_submission_result = exec_result.to_dict()

        # Record attempt
        self.session.record_attempt(
            code=candidate_code,
            tests_passed=exec_result.tests_passed,
            tests_total=exec_result.tests_total,
            failure_tags=failing_tags,
        )

        if exec_result.passed:
            # ALL TESTS PASSED
            self.session.solved = True
            self.session.transition_to(InterviewPhase.COMPLEXITY)
            self.session.complexity_step = "time"

            if self.persona.id == "silent":
                interviewer_response = "All tests passed. What is the time complexity?"
            elif self.persona.id == "standard":
                interviewer_response = f"All {exec_result.tests_total}/{exec_result.tests_total} tests passed. What is the time complexity of your solution?"
            else:  # Friendly
                interviewer_response = (
                    f"Excellent work! All {exec_result.tests_total} required checks passed smoothly!\n\n"
                    f"Let's discuss efficiency: what is the **time complexity** of your solution, and why?"
                )

            self.session.add_message("interviewer", interviewer_response)
            return exec_result, interviewer_response

        else:
            # DETERMINISTIC FAILURE DETECTED -> DEBUGGING
            self.session.transition_to(InterviewPhase.DEBUGGING)

            failure_info = classify_failure(self.problem, exec_result)
            followup_q = failure_info.get("followup_question", "Walk me through this case.")

            # Silent persona uses verbatim follow-up directly
            if self.persona.id == "silent":
                interviewer_response = followup_q
            else:
                system_prompt = build_system_prompt(self.persona, self.problem, "debugging")
                prompt = build_failure_followup_prompt(
                    verified_result=exec_result.to_dict(),
                    followup_question=followup_q,
                    persona=self.persona,
                    candidate_code=candidate_code,
                )
                interviewer_response = self.model.generate(
                    prompt=prompt,
                    system=system_prompt,
                    history=self.session.conversation,
                    temperature=0.5,
                )

            self.session.add_message("interviewer", interviewer_response)
            return exec_result, interviewer_response

    def request_hint(self) -> str:
        """Provide the next level hint from KB (Level 1 -> Level 2 -> Level 3)."""
        hint_data = self.session.get_next_hint()
        if not hint_data:
            msg = "You're already on the right track. Try tracing an example input step-by-step!"
            self.session.add_message("interviewer", msg)
            return msg

        level = hint_data["level"]
        raw_hint = hint_data["text"]

        if self.persona.id == "silent" and level > 1:
            msg = "Think through the invariant."
        else:
            system_prompt = build_system_prompt(self.persona, self.problem, "hint")
            prompt = build_hint_prompt(raw_hint, level, self.persona)
            msg = self.model.generate(
                prompt=prompt,
                system=system_prompt,
                history=self.session.conversation,
                temperature=0.4,
            )

        self.session.add_message("interviewer", msg)
        return msg

    def _handle_complexity_response(self, candidate_message: str) -> str:
        """Handle candidate explanations for time and space complexity."""
        complexity_meta = self.problem.get("complexity", {"time": "O(n)", "space": "O(1)"})

        if self.session.complexity_step == "time":
            self.session.complexity_answers["time"] = candidate_message
            self.session.complexity_step = "space"

            system_prompt = build_system_prompt(self.persona, self.problem, "complexity")
            prompt = build_complexity_prompt(
                candidate_answer=candidate_message,
                expected_complexity=complexity_meta,
                asking_for="time",
                persona=self.persona,
            )
            eval_resp = self.model.generate(
                prompt=prompt,
                system=system_prompt,
                history=self.session.conversation,
                temperature=0.4,
            )

            # Follow-up asking for space complexity
            if self.persona.id == "silent":
                resp = "And what is the space complexity?"
            else:
                resp = f"{eval_resp}\n\nNow, what is the **space complexity** of your implementation?"

            self.session.add_message("interviewer", resp)
            return resp

        else:
            # Space complexity answered -> record evaluation and proceed to debrief
            self.session.complexity_answers["space"] = candidate_message
            self.session.complexity_step = "done"
            self.session.complexity_evaluation = {
                "candidate_time": self.session.complexity_answers.get("time", ""),
                "candidate_space": candidate_message,
                "expected_time": complexity_meta.get("time", "O(n)"),
                "expected_space": complexity_meta.get("space", "O(1)"),
            }
            self.session.transition_to(InterviewPhase.DEBRIEF)
            return self.generate_debrief()

    def generate_debrief(self) -> str:
        """Generate final personalized debrief report."""
        summary = self.session.to_summary_dict()
        attempts_count = len(self.session.attempts)
        tests_passed = self.session.tests_passed
        tests_total = self.session.tests_total or len(self.problem.get("tests", []))
        hints_used = self.session.hints_used
        solved = self.session.solved
        title = self.problem.get("title", "Coding Problem")
        pattern = self.problem.get("pattern", "").replace("_", " ").title()
        difficulty = self.problem.get("difficulty", "medium").capitalize()

        complexity_discussed = bool(
            self.session.complexity_answers.get("time")
            or self.session.complexity_answers.get("space")
            or self.session.phase in (InterviewPhase.COMPLEXITY, InterviewPhase.DEBRIEF)
        )

        all_failure_tags = []
        for a in self.session.attempts:
            all_failure_tags.extend(a.get("failure_tags", []))
        summary["failure_tags"] = list(dict.fromkeys(all_failure_tags))
        summary["complexity_discussed"] = complexity_discussed
        summary["tests_total"] = tests_total
        summary["tests_passed"] = tests_passed
        summary["attempts_count"] = attempts_count

        # FAST ZERO-LATENCY PATH: If no code was attempted / submitted, return instant accurate debrief
        if attempts_count == 0 and not solved:
            debrief = f"""### What went well
You reviewed the problem statement and constraints for **{title}** ({difficulty}). Taking time to read requirements carefully and identifying the underlying {pattern} pattern is an important first step.

### What to practise
Make an active coding attempt in the editor. In live technical interviews, writing out an initial brute-force or partial solution and running example tests provides immediate execution feedback and helps you uncover logic gaps early.

### Interview summary
Problem solved: No (Session concluded before code submission)
Tests passed: 0 / {tests_total}
Hints used: {hints_used}
Complexity discussed: No

### Next step
Re-attempt **{title}** by writing your implementation in Python, or explore an introductory starter problem in the **{pattern}** pattern.
"""
            self.session.finalize_session(status="completed", debrief=debrief)
            self.session.add_message("interviewer", debrief)
            return debrief

        # Standard AI Generation for actual code submissions
        prompt = build_debrief_prompt(self.problem, summary)
        friendly_persona = load_persona("friendly")
        system_prompt = build_system_prompt(friendly_persona, self.problem, "debrief")

        try:
            debrief = self.model.generate(
                prompt=prompt,
                system=system_prompt,
                history=[],
                temperature=0.3,
                max_tokens=220,
            )
        except Exception:
            status_text = "Yes" if solved else "No"
            comp_text = "Yes" if complexity_discussed else "No"
            debrief = f"""### What went well
{'You successfully passed all test cases with a working solution.' if solved else f'You actively attempted the problem with {attempts_count} submission(s).'}

### What to practise
{'Review optimal time and space complexity tradeoffs.' if solved else f'Focus on edge-case handling and debugging: {summary["failure_tags"] or "boundary conditions"}.'}

### Interview summary
Problem solved: {status_text}
Tests passed: {tests_passed}/{tests_total}
Hints used: {hints_used}
Complexity discussed: {comp_text}

### Next step
Practice more problems in the **{pattern}** pattern.
"""

        self.session.finalize_session(status="completed", debrief=debrief)
        self.session.add_message("interviewer", debrief)
        return debrief

    def _generic_chat_response(self, candidate_message: str) -> str:
        system_prompt = build_system_prompt(self.persona, self.problem, self.session.phase.value)
        prompt = f"Candidate says: \"{candidate_message}\"\nRespond appropriately as {self.persona.name} interviewer."
        response = self.model.generate(prompt=prompt, system=system_prompt, history=self.session.conversation)
        self.session.add_message("interviewer", response)
        return response
