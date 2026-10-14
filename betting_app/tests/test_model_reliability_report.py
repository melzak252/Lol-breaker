"""A wide interval containing zero cannot certify reliable probabilities."""
from src.analysis.model_reliability import calibration_gate


CONFIG = {"calibration_min_rows": 200, "calibration_min_months": 6,
          "calibration_gap_tolerance": .05}


def test_wide_interval_containing_zero_fails_equivalence_tolerance():
    interval = {"n": 600, "blocks": 12, "ci_lower": -.08, "ci_upper": .02}
    assert calibration_gate(interval, CONFIG) == "FAILED_CALIBRATION_TOLERANCE"
    supported = {**interval, "ci_lower": -.02, "ci_upper": .03}
    assert calibration_gate(supported, CONFIG) == "SUPPORTED_IN_DEVELOPMENT"


def test_insufficient_support_cannot_pass_even_with_zero_observed_error():
    interval = {"n": 199, "blocks": 12, "ci_lower": 0., "ci_upper": 0.}
    assert calibration_gate(interval, CONFIG) == "INSUFFICIENT_EVIDENCE"
    assert calibration_gate({**interval, "n": 1000, "blocks": 5}, CONFIG) == "INSUFFICIENT_EVIDENCE"
    assert calibration_gate({**interval, "n": 1000, "ci_lower": None}, CONFIG) == "INSUFFICIENT_EVIDENCE"


def test_favorite_calibration_is_side_symmetric_and_excludes_neutral_rows():
    import pandas as pd
    import pytest
    from src.analysis.model_reliability import _calibration

    rows = []
    for month in range(1, 7):
        for index in range(40):
            rows.append({"date": f"2025-{month:02d}-01", "p": .3, "y": int(index < 12)})
        rows.extend({"date": f"2025-{month:02d}-01", "p": .5000000000000001, "y": 1} for _ in range(10))
    frame = pd.DataFrame(rows)
    config = {**CONFIG, "resamples": 5000, "seed": 20261001}
    first = _calibration(frame, "p", config)
    assert first["n"] == 240
    assert first["neutral_excluded"] == 60
    assert first["mean_confidence"] == pytest.approx(.7)
    assert first["favorite_win_rate"] == pytest.approx(.7)
    assert first["status"] == "SUPPORTED_IN_DEVELOPMENT"
    frame["p"], frame["y"] = 1-frame.p, 1-frame.y
    reversed_result = _calibration(frame, "p", config)
    assert reversed_result["gap_monthly"]["delta"] == pytest.approx(first["gap_monthly"]["delta"])
    assert reversed_result["status"] == first["status"]
