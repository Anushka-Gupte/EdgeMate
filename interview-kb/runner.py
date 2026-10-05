"""Sandboxed test runner for the interview KB.

Runs candidate code in a separate Python process with:
  * a per-test timeout (SIGALRM on Linux/Mac; one process per test elsewhere)
  * CPU-time and memory caps (resource.setrlimit on Linux/Mac)
  * isolated mode (-I) so it ignores env vars and user site-packages

This is NOT a security boundary against hostile code (no network namespace).
It is meant for a friend's own practice solutions and for the model's tools.
"""
import json
import os
import subprocess
import sys

PRELUDE_SRC = '''
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Deque, DefaultDict, Iterator, Callable, Iterable
from collections import deque, defaultdict, Counter, OrderedDict
import heapq
import math
import bisect
import itertools
import functools
import re
import copy

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head

def build_cycle(vals, pos):
    head = build_list(vals)
    if pos >= 0 and head is not None:
        nodes, cur = [], head
        while cur is not None:
            nodes.append(cur)
            cur = cur.next
        nodes[-1].next = nodes[pos]
    return head

def to_list(node):
    out = []
    while node is not None:
        out.append(node.val)
        node = node.next
    return out

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = None

class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def build_tree(vals):
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q, qi, i = [root], 0, 1
    while qi < len(q) and i < len(vals):
        n = q[qi]; qi += 1
        if i < len(vals) and vals[i] is not None:
            n.left = TreeNode(vals[i]); q.append(n.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            n.right = TreeNode(vals[i]); q.append(n.right)
        i += 1
    return root

def tree_to_list(root):
    out, q, i = [], [root], 0
    while i < len(q):
        n = q[i]; i += 1
        if n is None:
            out.append(None); continue
        out.append(n.val); q.append(n.left); q.append(n.right)
    while out and out[-1] is None:
        out.pop()
    return out

def inorder_vals(root):
    out, st, cur = [], [], root
    while st or cur:
        while cur:
            st.append(cur); cur = cur.left
        cur = st.pop(); out.append(cur.val); cur = cur.right
    return out

def find_node(root, v):
    q = [root] if root else []
    for n in q:
        if n.val == v:
            return n
        if n.left: q.append(n.left)
        if n.right: q.append(n.right)
    return None

def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(a - b) <= 1e-5
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def call(fn, adapter, args):
    if adapter == "list_to_list":
        return to_list(fn(build_list(args[0])))
    if adapter == "list_args_to_list":
        return to_list(fn(build_list(args[0]), *args[1:]))
    if adapter == "list_to_value":
        return fn(build_list(args[0]))
    if adapter == "cycle":
        return fn(build_cycle(args[0], args[1]))
    if adapter == "cycle_start":
        head = build_cycle(args[0], args[1])
        nodes, cur, seen = [], head, set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur)); nodes.append(cur); cur = cur.next
        r = fn(head)
        if r is None:
            return -1
        for i, n in enumerate(nodes):
            if n is r:
                return i
        return -2
    if adapter == "intersection":
        a, b, c = args
        common = build_list(c)
        cn, cur = [], common
        while cur is not None:
            cn.append(cur); cur = cur.next
        def attach(vals):
            head = common
            for v in reversed(vals):
                head = ListNode(v, head)
            return head
        r = fn(attach(a), attach(b))
        if r is None:
            return -1
        for i, n in enumerate(cn):
            if n is r:
                return i
        return -2
    if adapter == "sort_flat":
        return sorted(fn(*args))
    if adapter == "sort_nested":
        return sorted(sorted(x) for x in fn(*args))
    if adapter == "peak":
        i = fn(*args)
        n = args[0]
        if not isinstance(i, int) or i < 0 or i >= len(n):
            return False
        return (i == 0 or n[i] > n[i-1]) and (i == len(n)-1 or n[i] > n[i+1])
    if adapter == "two_lists":
        return to_list(fn(build_list(args[0]), build_list(args[1])))
    if adapter == "k_lists":
        return to_list(fn([build_list(x) for x in args[0]]))
    if adapter == "sort_outer":
        return sorted(fn(*args))
    if adapter == "tree_value":
        return fn(build_tree(args[0]), *args[1:])
    if adapter == "tree_tree":
        return tree_to_list(fn(build_tree(args[0]), *args[1:]))
    if adapter == "tree_inplace":
        root = build_tree(args[0])
        fn(root)
        return tree_to_list(root)
    if adapter == "tree_paths":
        return sorted(fn(build_tree(args[0]), *args[1:]))
    if adapter == "tree_next":
        root = build_tree(args[0])
        fn(root)
        rows, left = [], root
        while left is not None:
            row, n = [], left
            while n is not None:
                row.append(n.val); n = n.next
            rows.append(row); left = left.left
        return rows
    if adapter == "tree_lca":
        root = build_tree(args[0])
        r = fn(root, find_node(root, args[1]), find_node(root, args[2]))
        return None if r is None else r.val
    if adapter == "bst_delete":
        root = build_tree(args[0])
        return inorder_vals(fn(root, args[1]))
    if adapter == "random_list":
        pairs = args[0]
        nodes = [Node(v) for v, _ in pairs]
        for i, n in enumerate(nodes):
            if i + 1 < len(nodes): n.next = nodes[i + 1]
            if pairs[i][1] is not None: n.random = nodes[pairs[i][1]]
        orig = {id(n) for n in nodes}
        r = fn(nodes[0] if nodes else None)
        seq, cur = [], r
        while cur is not None:
            seq.append(cur); cur = cur.next
        if any(id(n) in orig for n in seq):
            return "shared_node_with_original"
        idx = {id(n): i for i, n in enumerate(seq)}
        out = []
        for n in seq:
            if n.random is not None and id(n.random) not in idx:
                return "random_points_outside_copy"
            out.append([n.val, None if n.random is None else idx[id(n.random)]])
        return out
    return fn(*args)
'''

HARNESS_BODY = r'''
import sys, json, time, signal
class _TO(BaseException):
    pass
def _alarm(*a):
    raise _TO()
payload = json.loads(sys.stdin.read())
limit = payload["timeout"]
use_alarm = hasattr(signal, "setitimer")
if use_alarm:
    signal.signal(signal.SIGALRM, _alarm)
def arm():
    if use_alarm:
        signal.setitimer(signal.ITIMER_REAL, limit)
def disarm():
    if use_alarm:
        signal.setitimer(signal.ITIMER_REAL, 0)
ns = {}
exec(PRELUDE_SRC, ns)
load_err = None
fn = None
try:
    arm()
    exec(payload["code"], ns)
    disarm()
    if payload["fn"] not in ns:
        load_err = "function '%s' is not defined" % payload["fn"]
    else:
        fn = ns[payload["fn"]]
except _TO:
    load_err = "Timeout while loading code"
except BaseException as e:
    disarm()
    load_err = type(e).__name__ + ": " + str(e)[:150]
out = []
for case in payload["cases"]:
    if load_err:
        out.append({"pass": False, "got": None, "error": load_err, "ms": 0})
        continue
    t = time.time()
    try:
        arm()
        got = ns["call"](fn, payload["adapter"], case["args"])
        disarm()
        ok = ns["same"](got, case["expected"])
        out.append({"pass": ok, "got": None if ok else repr(got)[:200],
                    "error": None, "ms": int((time.time() - t) * 1000)})
    except _TO:
        out.append({"pass": False, "got": None, "error": "Timeout", "ms": int(limit * 1000)})
    except BaseException as e:
        disarm()
        out.append({"pass": False, "got": None,
                    "error": type(e).__name__ + ": " + str(e)[:150], "ms": 0})
sys.stdout.write(json.dumps(out))
'''

HARNESS_SRC = "PRELUDE_SRC = " + repr(PRELUDE_SRC) + "\n" + HARNESS_BODY


def _limits(seconds):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        except Exception:
            pass
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GB
        except Exception:
            pass
    return apply


def _run_batch(code, fn, adapter, cases, timeout):
    payload = json.dumps({
        "code": code, "fn": fn, "adapter": adapter, "timeout": timeout,
        "cases": [{"args": c["args"], "expected": c["expected"]} for c in cases],
    })
    total = timeout * len(cases) + (1.0 if len(cases) <= 2 else 3.0)
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-B", "-c", HARNESS_SRC],
            input=payload, capture_output=True, text=True, timeout=total,
            preexec_fn=_limits(int(total) + 2) if os.name == "posix" else None,
        )
        res = json.loads(p.stdout)
    except subprocess.TimeoutExpired:
        res = [{"pass": False, "got": None, "error": "Timeout", "ms": int(timeout * 1000)}
               for _ in cases]
    except Exception:
        res = [{"pass": False, "got": None,
                "error": "Crashed (out of memory or killed)", "ms": 0} for _ in cases]
    for r, c in zip(res, cases):
        r["tag"] = c.get("tag")
    return res


def run_cases(code, fn, adapter, cases, timeout=2.0):
    """cases: list of {"args": [...], "expected": ..., "tag": str}
    Returns one result dict per case: pass, got, error, ms, tag."""
    import signal
    if not hasattr(signal, "setitimer"):  # Windows: one process per test
        out = []
        for i, c in enumerate(cases):
            res = _run_batch(code, fn, adapter, [c], timeout)
            out.extend(res)
            # If load error or timeout occurs on code level, remaining cases will fail identically
            if res and res[0].get("error") and ("Timeout" in str(res[0]["error"]) or "SyntaxError" in str(res[0]["error"]) or "is not defined" in str(res[0]["error"])):
                for rem in cases[i + 1:]:
                    out.append({"pass": False, "got": None, "error": res[0]["error"], "ms": res[0]["ms"], "tag": rem.get("tag")})
                break
        return out
    return _run_batch(code, fn, adapter, cases, timeout)
