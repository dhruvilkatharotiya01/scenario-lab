"""Scenario definitions and sampling.

A scenario is a kind of traffic situation plus a point in a parameter space.
Every parameter has a physical range, so "how much of the space fails?" is a
well-defined question.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

KMH = 1 / 3.6

# New kinds go at the end: sampling draws kinds in order from one generator,
# so appending keeps earlier kinds identical for the same seed.
KINDS = ("lead_brake", "cut_in", "cut_out", "merge")

# name: (low, high). Speeds are m/s, times are s, distances are m.
SPACE: dict[str, tuple[float, float]] = {
    "ego_speed": (30 * KMH, 130 * KMH),
    "gap": (15.0, 80.0),            # bumper-to-bumper distance to the actor (merge: see sim.py)
    "actor_speed_ratio": (0.5, 1.0),  # actor speed as a fraction of ego speed
    "threat_time": (1.0, 4.0),      # when the actor starts its manoeuvre (merge: ramp ends)
    "actor_brake": (2.0, 9.0),      # lead_brake: actor deceleration (merge: ramp acceleration x3)
    "manoeuvre_time": (1.0, 4.0),   # cut_in / cut_out / merge: lane change duration
    "reveal_gap": (20.0, 120.0),    # cut_out: stopped car beyond the lead
    "friction": (0.35, 1.0),        # road grip; limits braking to friction * g
    "sensor_range": (60.0, 200.0),  # perception range
    "latency": (0.0, 0.5),          # perception delay
    "neighbour_gap": (-40.0, 60.0),  # car in the left lane, relative to ego
}


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    ego_speed: float
    gap: float
    actor_speed_ratio: float
    threat_time: float
    actor_brake: float
    manoeuvre_time: float
    reveal_gap: float
    friction: float
    sensor_range: float
    latency: float
    neighbour_gap: float

    def params(self) -> dict[str, float]:
        d = asdict(self)
        d.pop("id")
        d.pop("kind")
        return d

    def clipped(self) -> "Scenario":
        """Return a copy with every parameter forced back inside SPACE."""
        p = {k: float(np.clip(v, *SPACE[k])) for k, v in self.params().items()}
        return Scenario(id=self.id, kind=self.kind, **p)


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^dims with exactly one point in each 1/n slice per axis."""
    points = (np.arange(n)[:, None] + rng.random((n, dims))) / n
    for d in range(dims):
        points[:, d] = rng.permutation(points[:, d])
    return points


def from_unit(kind: str, id_: str, u: np.ndarray) -> Scenario:
    p = {k: lo + float(x) * (hi - lo) for x, (k, (lo, hi)) in zip(u, SPACE.items())}
    return Scenario(id=id_, kind=kind, **p)


def sample(n_per_kind: int, seed: int, kinds: tuple[str, ...] = KINDS) -> list[Scenario]:
    """Latin hypercube sample of each kind. Same seed, same scenarios."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for kind in kinds:
        for i, u in enumerate(latin_hypercube(n_per_kind, len(SPACE), rng)):
            scenarios.append(from_unit(kind, f"{kind}-{i:05d}", u))
    return scenarios
