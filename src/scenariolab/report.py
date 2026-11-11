"""Build a single self-contained HTML report from result tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import FAIL, failure_rules, importance, summary, wilson
from .planners import PLANNERS
from .runner import scenario_from_row
from .sim import simulate

SPEED_BINS = [30, 50, 70, 90, 110, 130]
FRICTION_BINS = [0.35, 0.48, 0.61, 0.74, 0.87, 1.0]
TEMPLATE = Path(__file__).with_name("report_template.html")


def by_speed(df: pd.DataFrame) -> list[dict]:
    kmh = df.ego_speed * 3.6
    out = []
    for lo, hi in zip(SPEED_BINS, SPEED_BINS[1:]):
        part = df[(kmh >= lo) & (kmh < hi if hi != SPEED_BINS[-1] else kmh <= hi)]
        k, n = int((part.outcome == FAIL).sum()), len(part)
        low, high = wilson(k, n)
        out.append({"label": f"{lo}–{hi}", "rate": k / n if n else 0, "k": k, "n": n, "low": low, "high": high})
    return out


def heatmap(df: pd.DataFrame) -> list[list[dict]]:
    kmh = df.ego_speed.to_numpy() * 3.6
    mu = df.friction.to_numpy()
    fail = (df.outcome == FAIL).to_numpy()
    grid = []
    for m_lo, m_hi in zip(FRICTION_BINS, FRICTION_BINS[1:]):
        row = []
        for s_lo, s_hi in zip(SPEED_BINS, SPEED_BINS[1:]):
            mask = (mu >= m_lo) & (mu <= m_hi) & (kmh >= s_lo) & (kmh <= s_hi)
            row.append({"k": int(fail[mask].sum()), "n": int(mask.sum())})
        grid.append(row)
    return grid


def worst_replays(df: pd.DataFrame, planner: str, count: int = 6) -> list[dict]:
    worst = df[df.outcome == FAIL].sort_values("collision_speed", ascending=False).head(count)
    replays = []
    for _, row in worst.iterrows():
        res = simulate(scenario_from_row(row), PLANNERS[planner](), record=True)
        replays.append({
            "id": row["id"], "kind": row["kind"], "planner": planner,
            "speed": float(row.ego_speed * 3.6), "friction": float(row.friction),
            "impact": float(row.collision_speed * 3.6), "with": row["collision_with"],
            "frames": res.frames,
        })
    return replays


def build(results: dict[str, pd.DataFrame], path: Path, seed: int, falsified: dict | None = None) -> None:
    planners = []
    for name, df in results.items():
        planners.append({
            "name": name, "summary": summary(df),
            "kinds": {k: summary(g) for k, g in df.groupby("kind")},
            "bySpeed": by_speed(df), "heatmap": heatmap(df),
            "rules": failure_rules(df)[:4], "importance": importance(df),
            "neighbour": int((df.collision_with == "neighbour").sum()),
            "meanBrake": float(np.mean(df.max_brake)),
        })
    replays = [r for name, df in results.items() for r in worst_replays(df, name, 3)]
    data = {"planners": planners, "replays": replays, "seed": seed, "speedBins": SPEED_BINS,
            "frictionBins": FRICTION_BINS, "falsified": falsified}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text().replace("/*DATA*/null", blob))
