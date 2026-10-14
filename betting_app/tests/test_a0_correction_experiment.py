import numpy as np
import pandas as pd

from src.models.a0_correction_experiment import select_penalty, training_indices
from src.models.a0_sports_correction import AnchoredSportsCorrection


def test_delayed_outcomes_and_future_forecasts_cannot_change_earlier_fit():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2025-01-01"],
        "head_label_release_day": ["2024-01-02", "2024-02-02", "2025-01-01", "2025-01-02"],
        "p_a0": [.7, .3, .9, .1], "p_specialist": [.6, .4, .1, .9],
        "best_of": [3, 3, 3, 3], "y": [1, 0, 1, 0],
    })
    before = training_indices(frame, "2025-01-01")
    np.testing.assert_array_equal(before, [0, 1])
    changed = frame.copy()
    changed.loc[[2, 3], "y"] = 1 - changed.loc[[2, 3], "y"]
    changed.loc[[2, 3], ["p_a0", "p_specialist"]] = [[.01, .99], [.99, .01]]
    fitted = []
    for data in (frame, changed):
        subset = data.iloc[training_indices(data, "2025-01-01")]
        model = AnchoredSportsCorrection(True, .01).fit(
            subset.p_a0, subset.p_specialist, subset.best_of, subset.y,
        )
        fitted.append(model.coefficients)
    np.testing.assert_array_equal(*fitted)


def test_selection_uses_only_closed_earlier_origin_validation():
    frame = pd.DataFrame({
        "golgg_match_id": ["old", "delayed", "future"],
        "date": ["2024-07-02", "2024-10-02", "2025-01-02"],
        "head_label_release_day": ["2024-07-03", "2025-01-01", "2025-01-03"],
        "y": [1, 0, 0],
    })
    cv = pd.DataFrame({
        "cv_origin": ["2024-07-01", "2024-10-01", "2025-01-01"],
        "candidate__0.01": [.6, .001, .001],
        "candidate__0.1": [.8, .99, .99],
    })
    selected, receipt = select_penalty(frame, cv, "2025-01-01", "candidate", [.01, .1])
    assert selected == .1
    assert receipt["validation_ids"] == ["old"]
    frame.loc[[1, 2], "y"] = 1
    selected_changed, receipt_changed = select_penalty(frame, cv, "2025-01-01", "candidate", [.01, .1])
    assert selected_changed == selected
    assert receipt_changed == receipt
