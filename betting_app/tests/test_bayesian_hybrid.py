import pytest
from betting_app.core.models.engine import bayesian_logit_shrinkage


def test_bayesian_logit_shrinkage_symmetry() -> None:
    # Strict side symmetry: blend(p, q, w) + blend(1-p, 1-q, w) must equal 1.0
    for p, q, w in [
        (0.60, 0.55, 0.65),
        (0.40, 0.20, 0.65),
        (0.75, 0.85, 0.50),
        (0.15, 0.10, 0.35),
    ]:
        p_ab = bayesian_logit_shrinkage(p, q, model_weight=w)
        p_ba = bayesian_logit_shrinkage(1.0 - p, 1.0 - q, model_weight=w)
        assert abs(p_ab + p_ba - 1.0) < 1e-12


def test_bayesian_logit_shrinkage_identity() -> None:
    # When model and market agree, blended probability equals that probability
    for p in [0.20, 0.50, 0.80]:
        assert abs(bayesian_logit_shrinkage(p, p, model_weight=0.65) - p) < 1e-6


def test_bayesian_logit_shrinkage_weights() -> None:
    p = 0.70
    q = 0.40
    # Full model weight
    assert abs(bayesian_logit_shrinkage(p, q, model_weight=1.0) - p) < 1e-6
    # Full market weight
    assert abs(bayesian_logit_shrinkage(p, q, model_weight=0.0) - q) < 1e-6

    # Intermediate weight pulls between the two
    blended = bayesian_logit_shrinkage(p, q, model_weight=0.65)
    assert q < blended < p


def test_bayesian_logit_shrinkage_invalid_weight() -> None:
    with pytest.raises(ValueError):
        bayesian_logit_shrinkage(0.5, 0.5, model_weight=-0.1)

    with pytest.raises(ValueError):
        bayesian_logit_shrinkage(0.5, 0.5, model_weight=1.1)
