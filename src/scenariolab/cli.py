"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import analysis, report
from .planners import PLANNERS
from .runner import run_all
from .scenarios import KINDS, sample


def cmd_run(a: argparse.Namespace) -> int:
    df = run_all(sample(a.n, a.seed), a.planner)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(json.dumps(analysis.summary(df), indent=2))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    res = analysis.gate(pd.read_csv(a.baseline), pd.read_csv(a.candidate), a.alpha)
    print(f"failures {res['base_failures']} -> {res['candidate_failures']}, "
          f"new {len(res['new_failures'])}, fixed {len(res['fixed'])}, p={res['p_value']:.4f}")
    for sid in res["new_failures"][:20]:
        print(f"  new failure: {sid}")
    print("GATE PASSED" if res["passed"] else "GATE FAILED")
    return 0 if res["passed"] else 1


def compare_search(planner: str, budget: int, seed: int) -> dict:
    """Spend the same budget on targeted search and random sampling, on every kind."""
    each = budget // len(KINDS)
    random_rows = run_all(sample(each, seed + 1), planner)
    found = sum(len(analysis.falsify(planner, kind, each, seed)["failures"])
                for kind in KINDS)
    return {"planner": planner, "budget": each * len(KINDS), "search_failures": found,
            "random_failures": int((random_rows.outcome == analysis.FAIL).sum())}


def cmd_falsify(a: argparse.Namespace) -> int:
    print(json.dumps(compare_search(a.planner, a.budget, a.seed), indent=2))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    scenarios = sample(a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in PLANNERS:
        results[name] = run_all(scenarios, name)
        results[name].to_csv(a.out / f"{name}.csv", index=False)
        s = analysis.summary(results[name])
        print(f"{name:8s} crashes {s['failures']:5d}  {s['failure_rate']:.2%} "
              f"[{s['ci_low']:.2%}, {s['ci_high']:.2%}]  unavoidable {s['unavoidable']}")
    best = min(results, key=lambda k: analysis.summary(results[k])["failures"])
    found = compare_search(best, a.search_budget, a.seed) if a.search_budget else None
    if found:
        print(json.dumps(found))
    report.build(results, a.out / "report.html", a.seed, found)
    print(f"report: {a.out / 'report.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scenariolab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="test one planner and write a CSV")
    run.add_argument("--planner", choices=PLANNERS, required=True)
    run.add_argument("--n", type=int, default=500, help="scenarios per kind")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(fn=cmd_run)

    g = sub.add_parser("gate", help="exit 1 if the candidate is significantly worse")
    g.add_argument("baseline", type=Path)
    g.add_argument("candidate", type=Path)
    g.add_argument("--alpha", type=float, default=0.05)
    g.set_defaults(fn=cmd_gate)

    f = sub.add_parser("falsify", help="compare targeted failure search with random sampling")
    f.add_argument("--planner", choices=PLANNERS, required=True)
    f.add_argument("--budget", type=int, default=1200)
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(fn=cmd_falsify)

    b = sub.add_parser("bench", help="test every planner and build the HTML report")
    b.add_argument("--n", type=int, default=2000, help="scenarios per kind")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--search-budget", type=int, default=1200)
    b.add_argument("--out", type=Path, default=Path("out"))
    b.set_defaults(fn=cmd_bench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
