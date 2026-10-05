"""Step 2 of trust: run everything through the real sandbox.
  * reference solution passes every test (incl. large) within the time limit
  * every planted bug FAILS at least one test carrying the tag it claims
  * brute force passes all small tests
"""
import sys
import loader
from runner import run_cases


def main(only=None):
    bad = 0
    probs = loader.list_problems()
    for p in probs:
        if only and p["id"] not in only:
            continue
        tests = loader.materialize_tests(p)
        fn, ad = p["function"], p["adapter"]
        res = run_cases(p["reference_solution"], fn, ad, tests)
        fails = [(r["tag"], r["error"] or r["got"]) for r in res if not r["pass"]]
        status = "ok" if not fails else "REF FAILS %s" % fails[:3]
        if fails:
            bad += 1
        small = [t for t in tests if t["tag"] != "large"]
        bres = run_cases(p["brute_force_solution"], fn, ad, small)
        if not all(r["pass"] for r in bres):
            bad += 1
            status += " | BRUTE FAILS"
        lines = []
        for b in p["known_bugs"]:
            r = run_cases(b["code"], fn, ad, tests)
            failed_tags = {x["tag"] for x in r if not x["pass"]}
            good = b["fails_on"] in failed_tags
            if not good:
                bad += 1
            lines.append("   %s bug '%s' -> fails %s" % ("OK " if good else "XX ", b["name"], sorted(failed_tags)))
        print("%-32s %s" % (p["id"], status))
        for l in lines:
            print(l)
    print("\nPROBLEMS WITH ISSUES:", bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:] or None) else 0)
