"""Run scenarios in parallel and classify each outcome."""
from __future__ import annotations

from multiprocessing import Pool

import pandas as pd

from .planners import PLANNERS, Oracle
from .scenarios import SPACE, Scenario
from .sim import simulate

NEAR_MISS_TTC = 1.0

# outcome meanings:
#   failure     crashed, but ideal in-lane braking would have avoided it
#   unavoidable crashed, and ideal in-lane braking crashes too
#   near_miss   no crash, but time-to-collision fell below NEAR_MISS_TTC
#   pass        none of the above


def evaluate(sc: Scenario, planner_name: str) -> dict:
    res = simulate(sc, PLANNERS[planner_name]())
    if res.collision:
        outcome = "unavoidable" if simulate(sc, Oracle()).collision else "failure"
    elif res.min_ttc < NEAR_MISS_TTC:
        outcome = "near_miss"
    else:
        outcome = "pass"
    return {
        "id": sc.id, "kind": sc.kind, "planner": planner_name, "outcome": outcome,
        "collision_speed": res.collision_speed, "collision_with": res.collision_with,
        "min_ttc": res.min_ttc, "min_gap": res.min_gap, "max_brake": res.max_brake,
        **sc.params(),
    }


def _job(args: tuple[Scenario, str]) -> dict:
    return evaluate(*args)


def run_all(scenarios: list[Scenario], planner_name: str, workers: int | None = None) -> pd.DataFrame:
    jobs = [(sc, planner_name) for sc in scenarios]
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=32)
    return pd.DataFrame(rows)


def scenario_from_row(row) -> Scenario:
    return Scenario(id=row["id"], kind=row["kind"], **{k: float(row[k]) for k in SPACE})
