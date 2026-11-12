from dataclasses import replace

import pytest

from scenariolab.planners import IDM, Evasive, Guarded, IDMAEB, Oracle
from scenariolab.scenarios import KMH, Scenario, sample
from scenariolab.sim import (DT, MERGE_OFFSET, REACTION_TIME, Actor, build_actors, simulate,
                             smoothstep, visible)

BASE = Scenario(id="t", kind="lead_brake", ego_speed=100 * KMH, gap=40.0, actor_speed_ratio=0.8,
                threat_time=2.0, actor_brake=4.0, manoeuvre_time=2.0, reveal_gap=60.0,
                friction=0.9, sensor_range=150.0, latency=0.1, neighbour_gap=-30.0)


def test_smoothstep_is_flat_at_both_ends():
    assert smoothstep(0) == 0 and smoothstep(1) == 1
    assert smoothstep(0.001) < 1e-4 and 1 - smoothstep(0.999) < 1e-4


def test_simulation_is_deterministic():
    a, b = simulate(BASE, IDM()), simulate(BASE, IDM())
    assert (a.collision, a.min_gap, a.max_brake) == (b.collision, b.min_gap, b.max_brake)


def test_gentle_lead_braking_is_handled():
    assert not simulate(BASE, IDM()).collision


def test_braking_is_limited_by_friction():
    sc = replace(BASE, friction=0.4, gap=15.0, actor_brake=9.0)
    assert simulate(sc, IDMAEB()).max_brake <= 0.4 * 9.81 + 1e-9


def test_physically_impossible_case_beats_the_oracle():
    sc = replace(BASE, kind="cut_in", ego_speed=130 * KMH, actor_speed_ratio=0.5, gap=15.0,
                 manoeuvre_time=1.0, friction=0.35)
    assert simulate(sc, Oracle()).collision


def test_stopped_car_stays_hidden_until_the_lead_moves():
    lead = Actor(s=30.0, v=20.0)
    stopped = Actor(s=100.0, v=0.0)
    assert [t.s for t in visible([lead, stopped], 0.0, 0.0, 200.0)] == [30.0]
    lead.d = -3.5
    assert sorted(t.s for t in visible([lead, stopped], 0.0, 0.0, 200.0)) == [30.0, 100.0]


def test_sensor_range_limits_visibility():
    assert visible([Actor(s=100.0, v=0.0)], 0.0, 0.0, 60.0) == ()


def test_plain_evasive_swerves_into_fast_car_but_guarded_does_not():
    # Taken from a benchmark run: this exact case exposed the Evasive bug.
    sc = replace(BASE, kind="cut_out", ego_speed=15.717, gap=21.833, actor_speed_ratio=0.996,
                 threat_time=2.151, actor_brake=8.268, manoeuvre_time=1.008, reveal_gap=90.388,
                 friction=0.817, sensor_range=60.825, latency=0.188, neighbour_gap=-31.3)
    assert simulate(sc, Evasive()).collision_with == "neighbour"
    assert simulate(sc, Guarded()).collision_with != "neighbour"


def test_neighbour_waits_for_reaction_time_then_brakes():
    car = Actor(s=0.0, v=30.0, d=3.5, reactive=True)
    car.react(0.0, 20.0, 3.5, 20.0, 8.0)
    car.react(REACTION_TIME - DT, 20.0, 3.5, 20.0, 8.0)
    assert car.v == 30.0
    car.react(REACTION_TIME, 20.0, 3.5, 20.0, 8.0)
    assert car.v < 30.0


def test_neighbour_braking_respects_road_grip():
    car = Actor(s=0.0, v=30.0, d=3.5, reactive=True)
    car.react(0.0, 6.0, 3.5, 0.0, 3.0)
    car.react(REACTION_TIME, 6.0, 3.5, 0.0, 3.0)
    assert car.v == pytest.approx(30.0 - 3.0 * DT)


def test_neighbour_ignores_ego_in_another_lane_and_forgets_it():
    car = Actor(s=0.0, v=30.0, d=3.5, reactive=True)
    car.react(0.0, 20.0, 3.5, 20.0, 8.0)
    car.react(0.5, 20.0, 0.0, 20.0, 8.0)
    car.react(1.0, 20.0, 3.5, 20.0, 8.0)
    car.react(1.5, 20.0, 3.5, 20.0, 8.0)
    assert car.v == 30.0


def test_time_step_is_small_enough_for_car_sized_collisions():
    assert 130 * KMH * DT < 4.5 / 2


def test_sampling_is_reproducible_and_stays_in_range():
    a, b = sample(20, 7), sample(20, 7)
    assert a == b
    assert all(s == s.clipped() for s in a)
    assert sample(20, 8) != a


def test_adding_a_kind_keeps_earlier_kinds_identical():
    old = sample(20, 3, kinds=("lead_brake", "cut_in", "cut_out"))
    assert sample(20, 3)[:len(old)] == old


def test_merger_reaches_the_planned_spot_when_the_ramp_ends():
    sc = replace(BASE, kind="merge", gap=MERGE_OFFSET + 10.0, actor_speed_ratio=0.6,
                 actor_brake=6.0, threat_time=3.0)
    car = build_actors(sc)[1]
    t = 0.0
    while t < sc.threat_time - 1e-9:
        car.step(t)
        t += DT
    assert car.d == -3.5
    assert car.s - sc.ego_speed * sc.threat_time == pytest.approx(10.0, abs=0.5)


def test_merger_beside_the_ego_is_unavoidable_by_braking_alone():
    # Speed matched and alongside at the ramp end: braking cannot clear it in time.
    sc = replace(BASE, kind="merge", gap=MERGE_OFFSET, actor_speed_ratio=1.0,
                 threat_time=2.0, manoeuvre_time=1.0)
    assert simulate(sc, Oracle()).collision


def test_merger_behind_the_ego_is_harmless():
    sc = replace(BASE, kind="merge", gap=15.0, actor_speed_ratio=0.5, actor_brake=2.0)
    assert not simulate(sc, IDM()).collision


def test_guarded_yields_to_a_merger_beside_it():
    # Taken from a seed 42 benchmark run: the merger is level with the ego when the ramp ends.
    sc = replace(BASE, kind="merge", ego_speed=19.839, gap=36.857, actor_speed_ratio=0.729,
                 threat_time=3.741, actor_brake=5.945, manoeuvre_time=3.981, reveal_gap=79.832,
                 friction=0.784, sensor_range=169.07, latency=0.009, neighbour_gap=-29.394)
    assert simulate(sc, IDMAEB()).collision
    assert simulate(sc, Guarded(merge_speed=float("inf"))).collision  # yielding switched off
    assert not simulate(sc, Guarded()).collision
