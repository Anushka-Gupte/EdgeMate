# Interview KB - local "interviewer that runs your code"

119 problems across 20 patterns (every problem from both of your lists; 347 appears under both Hashing and Heap and is stored once). Everything is plain files: edit a JSON entry, no retraining.

```
interview-kb/
  problems/01_sliding_window ... 20_graph_bfs_dfs/   one JSON per problem (119)
  index.json        problem list + roadmap (all marked done)
  runner.py         sandboxed test runner (subprocess, timeout, CPU/memory caps)
  loader.py         load problems, run a solution, pick the follow-up question, get hints
  verify.py         proves the KB is trustworthy (see below)
  personas/         friendly.md, standard.md, silent.md
  authoring/        defs_a..defs_j.py (problem sources), treegen.py (random trees), build_kb.py
  demo.py           buggy solution -> failing test -> interviewer question
```

## What is in each problem JSON
`statement` (original wording), `signature`, `reference_solution`, `brute_force_solution`,
`tests` (tagged: normal / empty / single / duplicates / negative / ...), `large_tests` (generated from
stored code so the repo stays small), 3-step `hints`, `known_bugs` (buggy code + the test tag it fails on +
the follow-up question to ask), and `complexity` with follow-up questions.

## Design rule: code checks, the model talks
`loader.check()` runs the candidate's code. `loader.followup_for()` maps the first failing test's tag to a
pre-written question. A small 3B model only has to phrase it kindly - it never has to find the bug.

## Trust: two independent checks
1. `authoring/build_kb.py`: reference vs. an independent brute-force solution on every hand test plus 300
   random inputs per problem (small-test expected values come from the brute force, not the reference).
2. `verify.py`: in the real sandbox, the reference passes every test (incl. large), and every planted bug
   fails at least one test carrying the tag it claims. Last run: 119 problems, 0 issues.

## Usage
    python3 demo.py
    python3 verify.py                  # all, or: python3 verify.py hashing-001
    python3 authoring/build_kb.py      # regenerate JSON after editing authoring/defs_*.py

```python
import loader
p = loader.load(3)                      # by LeetCode number or id
results = loader.check(p, user_code)    # [{pass, got, error, ms, tag}, ...]
print(loader.followup_for(p, results), loader.hint(p, 0))
```

## Honest limits
- The sandbox is a subprocess with timeouts and rlimits, not a security boundary (no network isolation).
  Fine for a friend's practice code; don't run untrusted code with it.
- Problem statements are restated, not copied from LeetCode. Reference problems by number in public repos.
- Premium problems (370, 774, 1943) are restated from memory of the public descriptions; double check them
  against your own account before relying on edge-case wording.
- Checker adapters: unordered answers are sorted before comparing; trees are level-order lists with null;
  linked lists/trees are built and read back by the runner; any-valid-answer problems (peak element, BST delete)
  are checked by validity, not by exact output.
- Large tests for recursion-heavy problems use iterative reference solutions on purpose, so a recursive
  candidate hits Python's recursion limit exactly like it would in a real interview.
