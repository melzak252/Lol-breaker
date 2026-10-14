from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experiment_full_graph_glicko2 import (
    annual_graph_masks,
    fit_graph_relations,
)


def test_annual_graph_masks_hold_out_stop_calibration_and_test_chronologically():
    frame = pd.DataFrame(
        {
            "date": [
                "2017-01-01",
                "2018-06-30",
                "2018-07-01",
                "2018-12-31",
                "2019-01-01",
                "2019-12-31",
                "2020-01-01",
            ],
            "result_day": [
                "2017-01-01",
                "2018-06-30",
                "2018-07-01",
                "2018-12-31",
                "2019-01-01",
                "2019-12-31",
                "2020-01-01",
            ],
        }
    )

    masks = annual_graph_masks(frame, 2020)

    assert masks["train"].tolist() == [True, True, False, False, False, False, False]
    assert masks["stop"].tolist() == [False, False, True, True, False, False, False]
    assert masks["calibration"].tolist() == [False, False, False, False, True, True, False]
    assert masks["test"].tolist() == [False, False, False, False, False, False, True]
    assert not (masks["train"] & masks["stop"]).any()
    assert not (masks["stop"] & masks["calibration"]).any()
    assert not (masks["calibration"] & masks["test"]).any()


def test_graph_relation_logit_is_antisymmetric_under_team_swap():
    direct = np.array(
        [
            [-1.0, -0.5],
            [1.0, 0.5],
            [-0.4, -1.2],
            [0.4, 1.2],
            [-1.4, -0.1],
            [1.4, 0.1],
            [-0.2, -0.8],
            [0.2, 0.8],
        ]
    )
    baseline_logits = np.array([-0.7, 0.7, -0.3, 0.3, -1.0, 1.0, -0.2, 0.2])
    labels = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    masks = {
        "train": np.array([True, True, True, True, False, False, False, False]),
        "stop": np.array([False, False, False, False, True, True, True, True]),
    }

    logits, _, _ = fit_graph_relations(direct, baseline_logits, labels, masks)

    assert np.allclose(logits[::2] + logits[1::2], 0.0, atol=1e-12)
