# Scenario Lab

Scenario-based safety testing for highway driving planners. It generates thousands of traffic
scenarios, runs each planner through the same set, separates avoidable crashes from
physically unavoidable ones, explains where each planner fails, and blocks regressions in CI.

A full run produces a self-contained HTML report with charts, failure rules and crash replays.
A sample is in [docs/sample-report.html](docs/sample-report.html).

## Main result

Adding a swerve to a working emergency-braking planner made it less safe, not more. Testing found
out why, and the fixes cut avoidable crashes by almost half compared with braking alone.

8,000 scenarios per planner (2,000 of each kind), seed 42, with a left-lane driver who reacts to
the ego:

| Planner | What it does | Avoidable crashes | Rate (95% CI) | Crashes into left-lane car |
|---|---|---:|---:|---:|
| IDM | Car following only | 420 | 5.2% (4.8–5.8) | 0 |
| IDM + AEB | Adds emergency braking and cut-in prediction | 324 | 4.0% (3.6–4.5) | 0 |
| Evasive | Adds a swerve when the left lane looks clear | 377 | 4.7% (4.3–5.2) | 196 |
| Guarded | Fixes Evasive's lane check, braking trigger and merge handling | 179 | 2.2% (1.9–2.6) | 12 |

**The swerve bug:** Evasive checked a fixed distance behind it. All 194 of its avoidable left-lane
crashes involved a car that started behind the ego. That car was not faster; the ego was braking
while it swerved, so the car behind caught up. Guarded predicts the gap across the whole lane
change using the speed difference.

**The braking trigger:** a fixed time-to-collision threshold ignores how far the car needs to stop.
Guarded also brakes when the deceleration needed exceeds a conservative 3 m/s², because the planner
cannot measure road grip.

**The merge gap:** every planner looked for threats only ahead of the ego, so a car merging from
an on-ramp level with the ego was ignored until it was already overlapping. Guarded now brakes
when a car beside it starts moving into its lane, and still swerves if the left lane is clear. On
merges this cut its avoidable crashes from 132 to 108.

**The gate:** Evasive is blocked against IDM + AEB (170 new crashes, 117 fixed, p = 0.002).
Guarded passes, but it is not free: it fixes 155 crashes and introduces 10 new ones. An overall
rate hides that, which is why the gate reports both lists.

### Check on other data

The merge fix was developed on seed 42, then rerun on five other seeds with 3,000 scenarios each.
"Before" is Guarded without the merge fix, on the same scenarios.

| Seed | IDM + AEB | Evasive | Guarded before | Guarded after | Gate p |
|---:|---:|---:|---:|---:|---:|
| 1 | 126 | 135 | 72 | 74 | 0.77 |
| 2 | 123 | 140 | 75 | 63 | 0.008 |
| 3 | 128 | 140 | 82 | 73 | 0.035 |
| 4 | 132 | 143 | 86 | 76 | 0.013 |
| 5 | 121 | 140 | 72 | 62 | 0.006 |

The fix helped significantly on four seeds and made no real difference on seed 1. It also has a
cost: Guarded's crashes into the left-lane car rose from 30 to 40 across these five seeds, because
braking for a merger sometimes coincides with a swerve.

**Still unsolved:** when the merger is within about 4 m of level with the ego as the ramp ends,
Guarded still has an avoidable crash in 53% of cases (138 scenarios). Braking alone is often too
late there, and a better answer probably needs earlier yielding or speeding up, which in turn needs
a planner that knows where the ramp ends.

### How traffic realism changed the conclusion

The first version used a left-lane car that never reacted. Evasive then looked clearly worse than
IDM + AEB in every seed (322 against 182 on seed 42, before merges were added). Adding a 1 second
reaction made the gap much smaller, while IDM and IDM + AEB stayed exactly the same because they
never leave their lane. A result can depend on one modelling assumption, so that assumption should
be tested.

## How it works

1. **Generate:** Latin hypercube sampling across 11 parameters, including speed (30–130 km/h),
   road grip, sensor range and latency, for four situations: lead car braking, cut-in, cut-out
   that reveals a stopped car, and a car that must merge from an on-ramp when the ramp ends.
2. **Simulate:** a 20 Hz simulation with friction-limited braking, brake build-up, sensor range,
   occlusion and latency. Lane changes follow a smooth cubic curve.
3. **Classify:** every crash is rerun with an oracle that brakes as hard as physics allows from
   the moment the threat begins. If the oracle also crashes, the case is `unavoidable` and is not
   counted against the planner.
4. **Explain:** a shallow decision tree turns results into readable rules, such as
   `ego_speed > 106 km/h and sensor_range <= 96.77 and friction <= 0.68` for IDM + AEB (29%
   avoidable crash rate in 239 scenarios).
5. **Gate:** both planners run on identical scenarios and an exact McNemar test checks whether the
   candidate is significantly worse. Evasive is blocked on seed 42 (p = 0.002, exit code 1).
6. **Search:** a cross-entropy search concentrates on dangerous regions. With 1,200 runs on
   Guarded it found 120 failures against 22 from random sampling. Found failures cluster, so this
   measures search efficiency, not 120 separate problems.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

```bash
scenariolab bench --out out
```

```bash
scenariolab run --planner idm_aeb --n 500 --out out/baseline.csv
scenariolab run --planner guarded --n 500 --out out/candidate.csv
scenariolab gate out/baseline.csv out/candidate.csv
```

```bash
scenariolab falsify --planner guarded --budget 1200
```

To test your own planner, subclass `Planner` in `planners.py`, implement `plan()` and add it to
`PLANNERS`.

## Limitations

- Numbers come from a simplified simulator and say nothing about real-world safety. They are only
  meaningful for comparing planners within this setup.
- Only the left-lane car reacts, with a fixed 1 second delay and no steering. Scenario cars are
  scripted, and drivers do not vary. The merging car never gives up or waits for a gap.
- A merging car that hits the ego from beside or behind is usually classed `unavoidable`, because
  the oracle only brakes. Speeding up might avoid some of those.
- `unavoidable` means ideal in-lane braking fails. A perfect swerve might still avoid the crash.
- Ego physics is a point-mass model with no tyre or suspension dynamics, on a straight road.
- Sensor noise and false detections are not modelled.
- The Dockerfile and GitHub Actions workflow have not been run yet.

## Layout

| File | Purpose |
|---|---|
| `src/scenariolab/scenarios.py` | Parameter space and sampling |
| `src/scenariolab/sim.py` | Simulator and perception model |
| `src/scenariolab/planners.py` | Planners and the oracle |
| `src/scenariolab/runner.py` | Parallel runs and outcome classification |
| `src/scenariolab/analysis.py` | Statistics, rules, gate and search |
| `src/scenariolab/report.py` | HTML report |
