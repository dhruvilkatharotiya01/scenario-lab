# Scenario Lab

Scenario-based safety testing for highway driving planners. It generates thousands of traffic
scenarios, runs each planner through the same set, separates avoidable crashes from
physically unavoidable ones, explains where each planner fails, and blocks regressions in CI.

A full run produces a self-contained HTML report with charts, failure rules and crash replays.
A sample is in [docs/sample-report.html](docs/sample-report.html).

## Main result

Adding a swerve to a working emergency-braking planner brought no reliable safety gain. Testing
found out why, and the fix cut avoidable crashes by more than half on scenario sets it was not
developed on.

6,000 scenarios per planner, seed 42, with a left-lane driver who reacts to the ego:

| Planner | What it does | Avoidable crashes | Rate (95% CI) | Crashes into left-lane car |
|---|---|---:|---:|---:|
| IDM | Car following only | 247 | 4.1% (3.6–4.6) | 0 |
| IDM + AEB | Adds emergency braking and cut-in prediction | 182 | 3.0% (2.6–3.5) | 0 |
| Evasive | Adds a swerve when the left lane looks clear | 219 | 3.7% (3.2–4.2) | 169 |
| Guarded | Fixes Evasive's lane check and braking trigger | 72 | 1.2% (1.0–1.5) | 6 |

Left-lane crashes count every collision with that car, including 2 classed as unavoidable for Evasive.

**The bug:** Evasive checked a fixed distance behind it. All 167 of its avoidable left-lane
crashes involved a car that started behind the ego. That car was not faster; the ego was braking
while it swerved, so the car behind caught up. Guarded predicts the gap across the whole lane
change using the speed difference.

**Second fix:** a fixed time-to-collision trigger ignores how far the car needs to stop. Guarded
also brakes when the deceleration needed exceeds a conservative 3 m/s², because the planner
cannot measure road grip.

**Check on other data:** the planners were developed on seed 42, then rerun on five other seeds
with 3,000 scenarios each. Guarded had the fewest avoidable crashes every time. Evasive was
sometimes better and sometimes worse than IDM + AEB, so seed 42 alone would overstate its problem.

| Seed | IDM + AEB | Evasive | Guarded |
|---:|---:|---:|---:|
| 1 | 106 | 119 | 45 |
| 2 | 94 | 89 | 30 |
| 3 | 105 | 96 | 39 |
| 4 | 84 | 118 | 35 |
| 5 | 95 | 95 | 33 |

**The improvement is not free:** compared with IDM + AEB, Guarded fixes 116 crashes but introduces
6 new ones. An overall rate hides that, which is why the gate reports both lists.

**How traffic realism changed the conclusion:** the first version used a left-lane car that never
reacted. Evasive then looked clearly worse than IDM + AEB in every seed (322 against 182 on seed 42).
Adding a 1 second reaction changed that verdict to "no reliable difference", while IDM and IDM + AEB
stayed exactly the same because they never leave their lane. A result can depend on one modelling
assumption, so that assumption should be tested.

## How it works

1. **Generate:** Latin hypercube sampling across 11 parameters, including speed (30–130 km/h),
   road grip, sensor range and latency, for three situations: lead car braking, cut-in, and
   cut-out that reveals a stopped car.
2. **Simulate:** a 20 Hz simulation with friction-limited braking, brake build-up, sensor range,
   occlusion and latency. Lane changes follow a smooth cubic curve.
3. **Classify:** every crash is rerun with an oracle that brakes as hard as physics allows from
   the moment the threat begins. If the oracle also crashes, the case is `unavoidable` and is not
   counted against the planner.
4. **Explain:** a shallow decision tree turns results into readable rules, such as
   `neighbour_gap <= -4.91 and sensor_range <= 94.50 and ego_speed > 86 km/h` (26% avoidable crash
   rate in 200 scenarios).
5. **Gate:** both planners run on identical scenarios and an exact McNemar test checks whether the
   candidate is significantly worse. Evasive is blocked on seed 42 (p = 0.023, exit code 1).
6. **Search:** a cross-entropy search concentrates on dangerous regions. With 1,200 runs on
   Guarded it found 213 failures against 13 from random sampling. Found failures cluster, so this
   measures search efficiency, not 213 separate problems.

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
  scripted, and drivers do not vary.
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
