"""Regression tests for model hybrid contracts and blending weights."""

from betting_app.core.models.engine import bayesian_logit_shrinkage




def test_bayesian_logit_shrinkage_weight_direction() -> None:
    """An explicit 75% model weight must favor model log odds over market log odds."""
    p_model = 0.80  # logit ~ 1.386
    p_market = 0.50  # logit = 0.0

    # With model_weight = 0.75, logit_blend = 0.75 * 1.386 + 0.25 * 0 = 1.0395 -> p ~ 0.7388
    blended = bayesian_logit_shrinkage(p_model, p_market, model_weight=0.75)
    assert 0.70 < blended < 0.76
    assert abs(blended - p_model) < abs(blended - p_market)


def test_declared_blend_preserves_logit_math_and_temperature() -> None:
    import math
    import pytest
    from betting_app.core.models.engine import blend_market_probability

    value = blend_market_probability(
        .8, .4, alpha=.5, temperature=1., blending_mode="logit_shrinkage",
    )
    assert value == pytest.approx(.6202041028867288, abs=1e-12)
    assert .88 * 3 * value - 1 == pytest.approx(.637338831620964, abs=1e-12)
    scaled = 1 / (1 + math.exp(-math.log(4) / 2))
    assert blend_market_probability(
        .8, .4, alpha=.3, temperature=2., blending_mode="linear",
    ) == pytest.approx(.3 * scaled + .7 * .4, abs=1e-12)
    with pytest.raises(ValueError):
        blend_market_probability(.8, .4, alpha=.5, temperature=1., blending_mode="unknown")

