"""Driving planners under test.

A planner sees only an Observation, which is limited by sensor range and
delayed by latency. It never sees the scenario parameters, including friction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LANE_WIDTH = 3.5
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8


@dataclass(frozen=True)
class Track:
    s: float
    d: float
    v: float
    vd: float  # lateral speed


@dataclass(frozen=True)
class Observation:
    t: float
    s: float
    d: float
    v: float
    target_lane: int
    tracks: tuple[Track, ...]
    # Map knowledge: where the on-ramp to the right ends, if there is one. Real
    # vehicles get this from map data, not from their sensors.
    ramp_end: float | None = None


@dataclass(frozen=True)
class Command:
    accel: float
    target_lane: int


def lead_in_corridor(obs: Observation, d: float, horizon: float = 0.0) -> Track | None:
    """Nearest track ahead whose position, projected by horizon seconds, overlaps lateral d."""
    best = None
    for tr in obs.tracks:
        if tr.s <= obs.s:
            continue
        if abs(tr.d + tr.vd * horizon - d) < CAR_WIDTH:
            if best is None or tr.s < best.s:
                best = tr
    return best


def idm_accel(v: float, v0: float, lead: Track | None, ego_s: float,
              a_max: float = 1.5, b: float = 2.0, headway: float = 1.5,
              min_gap: float = 2.0) -> float:
    """Intelligent Driver Model (Treiber et al., 2000)."""
    free = 1 - (v / max(v0, 0.1)) ** 4
    if lead is None:
        return a_max * free
    gap = max(lead.s - ego_s - CAR_LENGTH, 0.1)
    dv = v - lead.v
    desired = min_gap + max(0.0, v * headway + v * dv / (2 * (a_max * b) ** 0.5))
    return a_max * (free - (desired / gap) ** 2)


def ttc(obs: Observation, lead: Track | None) -> float:
    if lead is None or obs.v <= lead.v:
        return float("inf")
    return max(lead.s - obs.s - CAR_LENGTH, 0.0) / (obs.v - lead.v)


class Planner:
    name = "base"

    def reset(self, cruise_speed: float) -> None:
        self.cruise_speed = cruise_speed

    def plan(self, obs: Observation) -> Command:
        raise NotImplementedError


class IDM(Planner):
    """Car following only. Reacts once a car is in its lane."""
    name = "idm"

    def plan(self, obs: Observation) -> Command:
        lead = lead_in_corridor(obs, obs.d)
        return Command(idm_accel(obs.v, self.cruise_speed, lead, obs.s), obs.target_lane)


@dataclass
class IDMAEB(Planner):
    """IDM plus emergency braking on time-to-collision, with cut-in prediction."""
    name = "idm_aeb"
    ttc_brake: float = 2.2
    predict_horizon: float = 1.0
    full_brake: float = 10.0

    def plan(self, obs: Observation) -> Command:
        now = lead_in_corridor(obs, obs.d)
        soon = lead_in_corridor(obs, obs.d, self.predict_horizon)
        lead = min((x for x in (now, soon) if x), key=lambda x: x.s, default=None)
        if ttc(obs, lead) < self.ttc_brake:
            return Command(-self.full_brake, obs.target_lane)
        return Command(idm_accel(obs.v, self.cruise_speed, lead, obs.s), obs.target_lane)


@dataclass
class Evasive(IDMAEB):
    """IDMAEB plus a swerve into the left lane if that lane looks clear."""
    name = "evasive"
    ttc_swerve: float = 2.8
    clear_ahead: float = 25.0
    clear_behind: float = 12.0

    def plan(self, obs: Observation) -> Command:
        cmd = super().plan(obs)
        lead = lead_in_corridor(obs, obs.d, self.predict_horizon)
        if obs.target_lane == 0 and ttc(obs, lead) < self.ttc_swerve and self._left_clear(obs):
            return Command(cmd.accel, 1)
        return cmd

    def _left_clear(self, obs: Observation) -> bool:
        left = LANE_WIDTH
        for tr in obs.tracks:
            if abs(tr.d - left) < CAR_WIDTH and -self.clear_behind < tr.s - obs.s < self.clear_ahead:
                return False
        return True


@dataclass
class Guarded(Evasive):
    """Evasive with two fixes found by testing it:

    1. Brake on the deceleration needed to stop in time, assuming only
       comfort-level grip. A fixed TTC threshold ignores speed and fires too
       late at high speed on a slippery road.
    2. Check the left lane using closing speed over the whole lane change,
       because a fixed distance misses a car arriving fast from behind.
    3. Yield to a car beside it that starts moving into its lane. Corridor
       checks only look ahead, so a merger level with the ego was ignored
       until it was already overlapping.
    4. Use the map to see a merge conflict coming. If a car on the ramp will
       reach the ramp end level with or just ahead of the ego, ease off early
       instead of braking hard once it moves across.
    """
    name = "guarded"
    brake_budget: float = 3.0
    lane_change_time: float = 3.0
    safety_gap: float = 8.0
    merge_speed: float = 0.3  # lateral speed towards the ego that counts as merging
    merge_window: float = CAR_LENGTH + 3.0
    ramp_horizon: float = 6.0   # s, how far ahead to look for merge conflicts
    ramp_margin: float = 10.0   # m, wanted spacing to the merger at the ramp end
    ramp_aware: bool = True

    def plan(self, obs: Observation) -> Command:
        lead = lead_in_corridor(obs, obs.d, self.predict_horizon)
        needed = 0.0
        if lead is not None and obs.v > lead.v:
            gap = max(lead.s - obs.s - CAR_LENGTH - 2.0, 0.1)
            needed = (obs.v - lead.v) ** 2 / (2 * gap)
        cmd = super().plan(obs)
        # Braking is added on top of the parent's command, so a swerve still happens.
        if needed > self.brake_budget or self._merging_beside(obs):
            return Command(-self.full_brake, cmd.target_lane)
        if self.ramp_aware and self._ramp_conflict(obs):
            return Command(min(cmd.accel, -self.brake_budget), cmd.target_lane)
        return cmd

    def _ramp_conflict(self, obs: Observation) -> bool:
        """True if a ramp car is predicted to reach the ramp end level with or just
        ahead of the ego, assuming both keep their current speed. Speeding up to
        pass a merger was tried too, and caused rear-end crashes once the ego
        slowed back down, so the planner only ever eases off."""
        if obs.ramp_end is None:
            return False
        for tr in obs.tracks:
            on_ramp = abs(tr.d + LANE_WIDTH) < CAR_WIDTH / 2 and tr.s < obs.ramp_end
            if not on_ramp:
                continue
            t_end = (obs.ramp_end - tr.s) / max(tr.v, 1.0)
            if t_end > self.ramp_horizon:
                continue
            ahead = obs.ramp_end - (obs.s + obs.v * t_end)  # merger minus ego at the ramp end
            if -CAR_LENGTH / 2 < ahead < CAR_LENGTH + self.ramp_margin:
                return True
        return False

    def _merging_beside(self, obs: Observation) -> bool:
        for tr in obs.tracks:
            offset = obs.d - tr.d
            towards = tr.vd * (1 if offset > 0 else -1)
            beside = -1.0 < tr.s - obs.s < self.merge_window
            if CAR_WIDTH <= abs(offset) < LANE_WIDTH + 0.5 and towards > self.merge_speed and beside:
                return True
        return False

    def _left_clear(self, obs: Observation) -> bool:
        left = LANE_WIDTH
        # Ego is braking while it swerves, so assume speed lost during the change.
        future_v = max(obs.v - self.brake_budget * self.lane_change_time, 0.0)
        for tr in obs.tracks:
            if abs(tr.d - left) >= CAR_WIDTH:
                continue
            now = tr.s - obs.s
            later = now + (tr.v - (obs.v + future_v) / 2) * self.lane_change_time
            if min(abs(now), abs(later)) < CAR_LENGTH + self.safety_gap or now * later < 0:
                return False
        return True


@dataclass
class Oracle(Planner):
    """Reference, not a real planner: brakes as hard as physics allows the moment
    the threat begins, with perfect information. If this still crashes, no
    in-lane braking strategy could have avoided the collision."""
    name = "oracle"
    threat_time: float = field(default=0.0)

    def plan(self, obs: Observation) -> Command:
        accel = -100.0 if obs.t >= self.threat_time else 0.0
        return Command(accel, obs.target_lane)


PLANNERS = {"idm": IDM, "idm_aeb": IDMAEB, "evasive": Evasive, "guarded": Guarded}
