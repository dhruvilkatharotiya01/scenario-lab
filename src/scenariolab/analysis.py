"""Statistics on results: failure rates, failure regions, regression gating, falsification."""
from __future__ import annotations

from math import comb, sqrt

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .runner import evaluate
from .scenarios import SPACE, from_unit

FAIL = "failure"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Stays sensible when k is 0 or close to n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    k = int((df.outcome == FAIL).sum())
    lo, hi = wilson(k, n)
    return {
        "n": n, "failures": k, "failure_rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi,
        "unavoidable": int((df.outcome == "unavoidable").sum()),
        "near_miss": int((df.outcome == "near_miss").sum()),
        "passed": int((df.outcome == "pass").sum()),
    }


def failure_rules(df: pd.DataFrame, depth: int = 3, min_share: float = 0.03) -> list[dict]:
    """Fit a small decision tree to turn "what fails" into readable rules.

    The tree explains this batch of results. It is a description, not a safety
    model, so rules are reported with their sample counts.
    """
    data = df[df.outcome != "unavoidable"]
    y = (data.outcome == FAIL).to_numpy()
    if y.sum() < 5 or (~y).sum() < 5:
        return []
    names = list(SPACE)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(10, int(len(data) * min_share)),
                                  class_weight=None, random_state=0)
    tree.fit(data[names].to_numpy(), y)
    t = tree.tree_
    rules: list[dict] = []

    def walk(node: int, conds: list[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            rate = float(counts[1] / counts.sum())
            if rate >= 0.25:
                rules.append({"rule": " and ".join(conds) or "all scenarios", "rate": rate,
                              "samples": int(t.n_node_samples[node])})
            return
        name, cut = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [f"{name} <= {_fmt(name, cut)}"])
        walk(t.children_right[node], conds + [f"{name} > {_fmt(name, cut)}"])

    walk(0, [])
    return sorted(rules, key=lambda r: -r["rate"] * r["samples"])


def _fmt(name: str, value: float) -> str:
    return f"{value * 3.6:.0f} km/h" if name == "ego_speed" else f"{value:.2f}"


def importance(df: pd.DataFrame) -> dict[str, float]:
    """Which parameters best separate failures from non-failures."""
    data = df[df.outcome != "unavoidable"]
    y = (data.outcome == FAIL).to_numpy()
    if y.sum() < 5 or (~y).sum() < 5:
        return {}
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0)
    tree.fit(data[list(SPACE)].to_numpy(), y)
    pairs = sorted(zip(SPACE, tree.feature_importances_), key=lambda x: -x[1])
    return {k: float(v) for k, v in pairs if v >= 0.01}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact test on discordant pairs: is b vs c more lopsided than a fair coin?"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def gate(base: pd.DataFrame, cand: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Compare two planners on identical scenarios.

    Pairing matters: each scenario is its own control, so the test detects
    smaller changes than comparing two overall rates would.
    """
    if set(base.id) != set(cand.id):
        raise ValueError("gate needs both runs to use the same scenarios (same n and seed)")
    a = base.set_index("id").outcome == FAIL
    b = cand.set_index("id").outcome == FAIL
    b = b.reindex(a.index)
    new = sorted(a.index[~a & b])
    fixed = sorted(a.index[a & ~b])
    p = mcnemar_exact(len(new), len(fixed))
    return {
        "base_failures": int(a.sum()), "candidate_failures": int(b.sum()),
        "new_failures": new, "fixed": fixed, "p_value": p,
        "passed": not (len(new) > len(fixed) and p < alpha),
    }


def robustness(row: dict) -> float:
    """Lower is closer to a crash. A planner-caused crash is the minimum."""
    if row["outcome"] == FAIL:
        return -1.0
    if row["outcome"] == "unavoidable":
        return float("inf")  # no credit for finding crashes nobody could avoid
    return min(row["min_ttc"], 10.0)


def falsify(planner: str, kind: str, budget: int, seed: int, batch: int = 40) -> dict:
    """Search for avoidable failures, focusing samples near the most dangerous runs.

    Cross-entropy style: keep the most critical quarter of runs, fit a Gaussian
    in normalised parameter space, sample the next batch from it.
    """
    rng = np.random.default_rng(seed)
    dims = len(SPACE)
    mean, std = np.full(dims, 0.5), np.full(dims, 0.35)
    used, found, seen = 0, [], []
    while used < budget:
        size = min(batch, budget - used)
        us = np.clip(rng.normal(mean, std, (size, dims)), 0, 0.999)
        rows = [evaluate(from_unit(kind, f"search-{used + i}", u), planner) for i, u in enumerate(us)]
        used += size
        found += [r for r in rows if r["outcome"] == FAIL]
        seen += [(robustness(r), u) for r, u in zip(rows, us)]
        seen.sort(key=lambda x: x[0])
        elite = np.array([u for _, u in seen[:max(5, len(seen) // 4)]])
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.03)
    return {"evaluated": used, "failures": found}
