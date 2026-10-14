"""Serving contracts at the controlled C0 boundary."""


import pytest

from betting_app.core.models import C0_NATIVE, PredictionEngine
from betting_app.services import c0_inference


def _features():
    return {
        "canonical_match_id": 17,
        "canonical": {
            "id": 17,
            "best_of": 3,
            "team_a_name": "Alpha",
            "team_b_name": "Beta",
        },
        "c0_request": {"team1_id": "1", "team2_id": "2"},
    }




def test_native_c0_boundary_error_does_not_become_a_ratings_fallback(monkeypatch):
    def unavailable(_features):
        raise FileNotFoundError("native history/artifact unavailable")

    monkeypatch.setattr(c0_inference, "predict_c0", unavailable)
    with pytest.raises(FileNotFoundError, match="native history"):
        PredictionEngine.predict_from_features(_features(), C0_NATIVE)




