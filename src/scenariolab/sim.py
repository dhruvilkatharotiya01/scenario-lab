"""Fixed-step simulation of one scenario.

The ego is a point-mass with friction-limited acceleration and a smooth lateral
lane change. Scenario cars replay scripted motion. The left-lane car also
brakes for the ego after a human reaction delay, so a swerve is not punished
by a driver who never responds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .planners import (CAR_LENGTH, CAR_WIDTH, LANE_WIDTH, Command, Observation, Oracle, Planner,
                       Track, idm_accel)
from .scenarios import Scenario

DT = 0.05
DURATION = 25.0
G = 9.81
MAX_JERK = 25.0  # brakes need time to build pressure
REACTION_TIME = 1.0  # typical human delay before braking for a car ahead


def smoothstep(u: float) -> float:
    """Cubic with zero slope at both ends, so lateral motion starts and stops smoothly."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Actor:
    s: float
    v: float
    d: float = 0.0
    brake: float = 0.0
    brake_at: float = float("inf")
    stop_speed: float = 0.0
    lane_from: float = 0.0
    lane_to: float = 0.0
    lane_at: float = float("inf")
    lane_time: float = 1.0
    vd: float = 0.0
    reactive: bool = False
    noticed_at: float = float("inf")

    def react(self, t: float, ego_s: float, ego_d: float, ego_v: float, limit: float) -> None:
        """Follow the ego with IDM once it has been ahead in this lane for REACTION_TIME."""
        if ego_s > self.s and abs(ego_d - self.d) < CAR_WIDTH:
            self.noticed_at = min(self.noticed_at, t)
        else:
            self.noticed_at = float("inf")
        if t - self.noticed_at >= REACTION_TIME:
            wanted = idm_accel(self.v, self.cruise_speed, Track(ego_s, ego_d, ego_v, 0.0), self.s)
            self.v = max(0.0, self.v + min(max(wanted, -limit), limit) * DT)

    def __post_init__(self) -> None:
        self.cruise_speed = self.v

    def step(self, t: float) -> None:
        if t >= self.brake_at and self.v > self.stop_speed:
            self.v = max(self.stop_speed, self.v - self.brake * DT)
        self.s += self.v * DT
        new_d = self.lane_from + (self.lane_to - self.lane_from) * smoothstep(
            (t - self.lane_at) / self.lane_time)
        self.vd = (new_d - self.d) / DT
        self.d = new_d


def build_actors(sc: Scenario) -> list[Actor]:
    first = sc.gap + CAR_LENGTH
    actor_v = sc.ego_speed * sc.actor_speed_ratio
    # A cut-in car starts further ahead, so the gap equals sc.gap when it moves.
    cut_in_start = first + (sc.ego_speed - actor_v) * sc.threat_time
    actors = [Actor(s=sc.neighbour_gap, v=sc.ego_speed, d=LANE_WIDTH,
                    lane_from=LANE_WIDTH, lane_to=LANE_WIDTH, reactive=True)]
    if sc.kind == "lead_brake":
        actors.append(Actor(s=first, v=sc.ego_speed, brake=sc.actor_brake,
                            brake_at=sc.threat_time, stop_speed=actor_v * 0.2))
    elif sc.kind == "cut_in":
        actors.append(Actor(s=cut_in_start, v=actor_v, d=-LANE_WIDTH, lane_from=-LANE_WIDTH,
                            lane_to=0.0, lane_at=sc.threat_time, lane_time=sc.manoeuvre_time))
    elif sc.kind == "cut_out":
        # The lead hides a stopped car, then leaves the lane at the last moment.
        actors.append(Actor(s=first, v=sc.ego_speed, lane_to=-LANE_WIDTH,
                            lane_at=sc.threat_time, lane_time=sc.manoeuvre_time))
        clear_after = sc.threat_time + sc.manoeuvre_time
        stopped_at = first + sc.ego_speed * clear_after + sc.reveal_gap
        actors.append(Actor(s=stopped_at, v=0.0))
    else:
        raise ValueError(f"unknown scenario kind: {sc.kind}")
    return actors


def overlaps(s1: float, d1: float, s2: float, d2: float) -> bool:
    return abs(s1 - s2) < CAR_LENGTH and abs(d1 - d2) < CAR_WIDTH


@dataclass
class Result:
    collision: bool = False
    collision_speed: float = 0.0
    collision_with: str = ""
    min_ttc: float = float("inf")
    min_gap: float = float("inf")
    max_brake: float = 0.0
    frames: list = field(default_factory=list)


def visible(actors: list[Actor], ego_s: float, ego_d: float, sensor_range: float) -> tuple[Track, ...]:
    """Tracks within range and not hidden behind a nearer car in the same lane."""
    tracks = []
    for a in actors:
        if abs(a.s - ego_s) > sensor_range:
            continue
        hidden = a.s > ego_s and any(
            ego_s < b.s < a.s and abs(b.d - a.d) < CAR_WIDTH / 2 and abs(b.d - ego_d) < CAR_WIDTH / 2
            for b in actors)
        if not hidden:
            tracks.append(Track(a.s, a.d, a.v, a.vd))
    return tuple(tracks)


def simulate(sc: Scenario, planner: Planner, record: bool = False) -> Result:
    actors = build_actors(sc)
    s, d, v, accel = 0.0, 0.0, sc.ego_speed, 0.0
    target_lane = 0
    lane_from, lane_start, lane_time = 0.0, -1.0, 1.0
    planner.reset(sc.ego_speed)
    if isinstance(planner, Oracle):
        planner.threat_time = sc.threat_time

    limit = sc.friction * G
    delay = max(1, round(sc.latency / DT))
    queue: deque[tuple[Track, ...]] = deque(maxlen=delay)
    res = Result()

    for step in range(int(DURATION / DT)):
        t = step * DT
        queue.append(visible(actors, s, d, sc.sensor_range))
        obs = Observation(t, s, d, v, target_lane, queue[0])
        cmd: Command = planner.plan(obs)

        if cmd.target_lane != target_lane:
            target_lane, lane_from, lane_start = cmd.target_lane, d, t
            lane_time = max(2.0, 3.0 - v / 40)

        wanted = min(max(cmd.accel, -limit), min(limit, 2.0))
        change = min(max(wanted - accel, -MAX_JERK * DT), MAX_JERK * DT)
        accel += change
        v = max(0.0, v + accel * DT)
        s += v * DT
        if lane_start >= 0 and v > 0.1:
            goal = target_lane * LANE_WIDTH
            d = lane_from + (goal - lane_from) * smoothstep((t - lane_start) / lane_time)
        for a in actors:
            if a.reactive:
                a.react(t, s, d, v, limit)
            a.step(t)
        res.max_brake = max(res.max_brake, -accel)

        for i, a in enumerate(actors):
            if abs(a.d - d) < CAR_WIDTH:
                gap = abs(a.s - s) - CAR_LENGTH
                res.min_gap = min(res.min_gap, max(gap, 0.0))
                if a.s > s and v > a.v:
                    res.min_ttc = min(res.min_ttc, max(gap, 0.0) / (v - a.v))
            if overlaps(s, d, a.s, a.d) and not res.collision:
                res.collision = True
                res.collision_speed = abs(v - a.v)
                res.collision_with = "neighbour" if i == 0 else "scenario_actor"

        if record and step % 2 == 0:
            res.frames.append([round(t, 2), round(s, 2), round(d, 2), round(v, 2),
                               [[round(a.s, 2), round(a.d, 2)] for a in actors]])
        if res.collision or (v < 0.05 and t > sc.threat_time + 3):
            break
    return res
