import pandas as pd
import pytest

from scenariolab.analysis import gate, mcnemar_exact, robustness, summary, wilson
from scenariolab.cli import main


def frame(outcomes):
    return pd.DataFrame({"id": [f"s{i}" for i in range(len(outcomes))], "outcome": outcomes})


def test_wilson_handles_zero_failures():
    low, high = wilson(0, 100)
    assert low == 0 and 0.03 < high < 0.04


def test_wilson_contains_the_point_estimate():
    low, high = wilson(30, 900)
    assert low < 30 / 900 < high


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_gate_blocks_a_clear_regression():
    base = frame(["pass"] * 30)
    worse = frame(["failure"] * 12 + ["pass"] * 18)
    res = gate(base, worse)
    assert not res["passed"] and len(res["new_failures"]) == 12


def test_gate_allows_noise_and_improvement():
    base = frame(["failure"] * 3 + ["pass"] * 27)
    noisy = frame(["pass"] * 3 + ["failure"] * 4 + ["pass"] * 23)
    assert gate(base, noisy)["passed"]
    assert gate(base, frame(["pass"] * 30))["passed"]


def test_gate_refuses_mismatched_scenarios():
    with pytest.raises(ValueError):
        gate(frame(["pass"] * 3), frame(["pass"] * 4))


def test_unavoidable_crashes_do_not_attract_the_search():
    assert robustness({"outcome": "unavoidable", "min_ttc": 0.0}) == float("inf")
    assert robustness({"outcome": "failure", "min_ttc": 0.0}) == -1.0


def test_summary_counts():
    s = summary(frame(["failure", "unavoidable", "near_miss", "pass"]))
    assert (s["failures"], s["unavoidable"], s["near_miss"], s["passed"]) == (1, 1, 1, 1)


def test_cli_end_to_end(tmp_path):
    a, b = tmp_path / "idm.csv", tmp_path / "guarded.csv"
    assert main(["run", "--planner", "idm", "--n", "20", "--out", str(a)]) == 0
    assert main(["run", "--planner", "guarded", "--n", "20", "--out", str(b)]) == 0
    assert main(["gate", str(a), str(b)]) == 0
    assert main(["bench", "--n", "20", "--search-budget", "0", "--out", str(tmp_path / "r")]) == 0
    assert "Scenario Lab Report" in (tmp_path / "r" / "report.html").read_text()
