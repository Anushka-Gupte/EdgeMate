"""Loads KB problems and runs a candidate solution against them.

Design rule: code does the checking, the model does the talking.
check() gives pass/fail per test; followup_for() turns the first failing
test's tag into a pre-written interviewer question (no model needed to find the bug).
"""
import copy
import glob
import json
import os

from runner import PRELUDE_SRC, run_cases

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM_DIR = os.path.join(HERE, "problems")


def list_problems(pattern=None):
    out = []
    for path in sorted(glob.glob(os.path.join(PROBLEM_DIR, "**", "*.json"), recursive=True)):
        with open(path) as f:
            p = json.load(f)
        if pattern is None or p["pattern"] == pattern:
            out.append(p)
    return out


def load(problem_id):
    for p in list_problems():
        if p["id"] == problem_id or str(p["leetcode"]) == str(problem_id):
            return p
    raise KeyError(problem_id)


def _ns(code):
    ns = {}
    exec(PRELUDE_SRC, ns)
    exec(code, ns)
    return ns


def materialize_tests(p, include_large=True):
    """Small tests are stored inline. Large inputs are generated from stored code
    (keeps the repo small); their expected output comes from the reference solution."""
    tests = [dict(t) for t in p["tests"]]
    if include_large and p["large_tests"]:
        ref = _ns(p["reference_solution"])
        for lt in p["large_tests"]:
            g = {}
            exec(lt["gen_code"], g)
            args = g["gen"]()
            exp = ref["call"](ref[p["function"]], p["adapter"], copy.deepcopy(args))
            tests.append({"args": args, "expected": exp, "tag": lt["tag"]})
    return tests


def check(p, user_code, include_large=True, timeout=2.0):
    return run_cases(user_code, p["function"], p["adapter"],
                     materialize_tests(p, include_large), timeout)


def followup_for(p, results):
    """Pick the interviewer's question from the first failing test's tag."""
    for r in results:
        if not r["pass"]:
            for bug in p["known_bugs"]:
                if bug["fails_on"] == r["tag"]:
                    return bug["followup"]
            return "Walk me through what your code does on a %s input." % r["tag"]
    return None


def hint(p, level):
    keys = sorted(p["hints"])
    return p["hints"][keys[min(level, len(keys) - 1)]]
