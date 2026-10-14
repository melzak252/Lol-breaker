"""Regression contracts for trained, masked, side-symmetric A1 inference."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from betting_app.core.models import A1_CONSOLIDATED, PredictionEngine
from scripts.a1.dataset_loader import A1Dataset
from scripts.a1.train_a1 import train_a1_model
from src.models.a1.architecture import A1Config, A1Model
from src.models.a1.architecture_dynamic import A1DynamicModel
from src.models.a1.engine import A1PredictorEngine
from src.models.a1.predictor import A1Predictor


def _inputs(n=1):
    return {
        "context": np.zeros((n, A1Config().context_dim), dtype=np.float32),
        "player_ratings": np.zeros((n, 2, 5, 21), dtype=np.float32),
        "player_history": np.zeros((n, 2, 5, 16, 33), dtype=np.float32),
        "player_mask": np.zeros((n, 2, 5, 16), dtype=np.uint8),
        "player_champions": np.zeros((n, 2, 5, 16), dtype=np.int64),
        "meta_champions": np.zeros((n, 2, 5, 32), dtype=np.float32),
        "best_of": np.ones(n, dtype=np.int64),
    }


def test_consolidated_a1_requires_explicit_macro_weights():
    with pytest.raises(ValueError):
        A1PredictorEngine()


def test_operational_a1_does_not_label_rating_consensus_as_causal_a0():
    features = {
        "canonical": {"id": 1, "best_of": 3, "league": "LEC"},
        "player_ratings": {"probabilities": {"consensus": 0.65}},
        "ratings": {"probabilities": {"consensus": 0.60}},
    }
    with pytest.raises(ValueError):
        PredictionEngine.predict_from_features(features, A1_CONSOLIDATED)


def test_a1_rejects_side_biased_calibration():
    with torch.random.fork_rng():
        torch.manual_seed(20260926)
        model = A1Model(A1Config(dropout=0.0))
    with pytest.raises(ValueError):
        A1Predictor(model, {"platt_a": 1.01, "platt_b": 0.055})


def test_masked_champion_padding_cannot_change_match_probability():
    with torch.random.fork_rng():
        torch.manual_seed(20260926)
        model = A1Model(A1Config(dropout=0.0))
    predictor = A1Predictor(model)
    inputs = _inputs()
    original = predictor.predict_batch(**inputs)
    inputs["player_champions"][:, 0, :, -1] = 17
    changed = predictor.predict_batch(**inputs)
    np.testing.assert_allclose(changed, original, rtol=0.0, atol=1e-7)


def test_masked_nonfinite_history_cannot_change_match_probability():
    with torch.random.fork_rng():
        torch.manual_seed(20260926)
        model = A1Model(A1Config(dropout=0.0))
    predictor = A1Predictor(model)
    inputs = _inputs()
    original = predictor.predict_batch(**inputs)
    inputs["player_history"][:, 0] = np.nan
    changed = predictor.predict_batch(**inputs)
    np.testing.assert_allclose(changed, original, rtol=0.0, atol=1e-7)


class _ScalarForecast(nn.Module):
    """One-parameter learner whose validation optimum precedes its final epoch."""

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, context, *args, **kwargs):
        return torch.sigmoid(self.logit * context[:, 0])


def test_calibration_matches_the_restored_best_checkpoint():
    inputs = _inputs(4)
    inputs["context"][:, 0] = 1.0
    train = A1Dataset(**inputs, labels=np.ones(4, dtype=np.float32))
    validation = A1Dataset(
        **inputs, labels=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    )
    model, calibration = train_a1_model(
        _ScalarForecast(), DataLoader(train, batch_size=4, shuffle=False),
        validation, epochs=3, lr=1.0, weight_decay=0.0,
    )
    predictions = A1Predictor(model, calibration).predict_batch(**inputs)
    np.testing.assert_allclose(predictions, 0.75, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("model_class", [A1Model, A1DynamicModel])
def test_unsupported_series_cannot_silently_select_another_head(model_class):
    with torch.random.fork_rng():
        torch.manual_seed(20260926)
        model = model_class().eval()
    inputs = {key: torch.from_numpy(value) for key, value in _inputs().items()}
    inputs["best_of"] = torch.tensor([2])
    if model_class is A1DynamicModel:
        inputs.pop("player_champions")
        inputs["dynamic_champions"] = inputs.pop("meta_champions")
    with pytest.raises(ValueError):
        model(**inputs)
