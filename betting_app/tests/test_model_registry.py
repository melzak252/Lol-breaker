"""Unit tests for the unified model registry and prediction engine."""

import pytest

from betting_app.core.models import (
    A1_CONSOLIDATED,
    ACTIVE_MODEL_NAME,
    BAYESIAN_SHRUNK_HYBRID,
    C0_NATIVE,
    EXP039_THESIS,
    EXP078_LINEAR,
    HybridSpec,
    PredictionEngine,
    UnifiedPredictionResult,
    get_active_hybrid,
    get_active_model,
    get_model,
    set_active_hybrid,
    set_active_model,
)


def test_registry_lookups_keep_legacy_models_distinct():
    assert get_model(ACTIVE_MODEL_NAME) is C0_NATIVE
    assert get_model("Causal-C0") is C0_NATIVE
    assert get_model("native_c0") is C0_NATIVE
    assert get_model(BAYESIAN_SHRUNK_HYBRID.name) is BAYESIAN_SHRUNK_HYBRID
    assert get_model("shrunk_hybrid") is BAYESIAN_SHRUNK_HYBRID
    assert get_model("exp081") is None
    assert get_model("Symmetrized-Siamese-Series-EXP081") is None
    assert get_model("Consolidated-A1") is A1_CONSOLIDATED
    assert get_model("a1") is A1_CONSOLIDATED
    assert get_model("Operational-PlayerTeamRatings-W20") is EXP078_LINEAR
    assert get_model("exp078") is EXP078_LINEAR
    assert get_model("Sym-Cal LR-ElasticNet-W20-Binomial") is EXP039_THESIS
    assert get_model("exp039") is EXP039_THESIS
    assert get_model("non_existent") is None

def test_registry_switching_keeps_shrinkage_on_selected_base():
    try:
        assert get_active_model() is C0_NATIVE
        assert get_active_hybrid().base_model is C0_NATIVE
        set_active_model("exp078")
        assert get_active_model() is EXP078_LINEAR
        assert get_active_hybrid().base_model is EXP078_LINEAR
        assert get_active_hybrid().hybrid_model_version.startswith(EXP078_LINEAR.version)
        set_active_model(C0_NATIVE)
        assert get_active_model() is C0_NATIVE
        assert get_active_hybrid().base_model is C0_NATIVE
        set_active_model(BAYESIAN_SHRUNK_HYBRID)
        assert get_active_model() is BAYESIAN_SHRUNK_HYBRID
        assert get_active_hybrid().base_model is C0_NATIVE
    finally:
        set_active_model(C0_NATIVE)

def test_custom_hybrid_policy_has_a_derived_version():
    original = get_active_hybrid()
    try:
        custom = HybridSpec(
            base_model=C0_NATIVE,
            alpha=0.25,
            temperature=1.4,
            blending_mode="logit_shrinkage",
            custom_version=BAYESIAN_SHRUNK_HYBRID.version,
        )
        set_active_hybrid(custom)
        active_hybrid = get_active_hybrid()
        assert active_hybrid.custom_version is None
        assert active_hybrid.hybrid_model_version == f"{C0_NATIVE.version}-a0.25-t1.40"
        assert get_active_model() is C0_NATIVE
    finally:
        set_active_hybrid(original)



def test_unified_prediction_result_validation():
    """Verify invariant validation in UnifiedPredictionResult."""
    valid = UnifiedPredictionResult(
        canonical_match_id=1,
        prob_a=0.65,
        prob_b=0.35,
        model_name="test",
        model_version="v1",
    )
    assert valid.prob_a == 0.65
    assert valid.prob_b == 0.35

    with pytest.raises(ValueError, match="prob_a must be in"):
        UnifiedPredictionResult(canonical_match_id=1, prob_a=1.2, prob_b=-0.2)

    with pytest.raises(ValueError, match="prob_a \\+ prob_b must equal 1.0"):
        UnifiedPredictionResult(canonical_match_id=1, prob_a=0.6, prob_b=0.6)


def test_blend_with_market():
    """Verify market blending using both logit shrinkage and linear modes."""
    # Equal 50-50
    p_blend = PredictionEngine.blend_with_market(0.50, 0.50)
    assert p_blend == pytest.approx(0.50, abs=1e-3)

    # Model says 0.70, market says 0.60
    p_blend_shrink = PredictionEngine.blend_with_market(0.70, 0.60)
    assert 0.50 < p_blend_shrink < 0.70

    # Linear blending test
    lin_spec = HybridSpec(base_model=C0_NATIVE, alpha=0.40, temperature=1.0, blending_mode="linear")
    p_linear = PredictionEngine.blend_with_market(0.80, 0.50, hybrid_spec=lin_spec)
    assert p_linear == pytest.approx(0.40 * 0.80 + 0.60 * 0.50, abs=1e-3)


def test_model_alias_normalization_keeps_frozen_research_models_distinct():
    assert get_model("CAUSAL_C0") is C0_NATIVE
    assert get_model("EXP-078") is EXP078_LINEAR
    assert get_model("exp078") is EXP078_LINEAR
    assert get_model("EXP-039") is EXP039_THESIS
    assert get_model("nonexistent-model") is None


def test_blend_with_market_rejects_non_finite():
    """Verify blend_with_market raises ValueError on NaN or Inf."""
    with pytest.raises(ValueError, match="finite"):
        PredictionEngine.blend_with_market(float("nan"), 0.50)
    with pytest.raises(ValueError, match="finite"):
        PredictionEngine.blend_with_market(0.50, float("inf"))


def test_native_c0_engine_returns_the_shared_predictor_result(monkeypatch):
    from betting_app.services import c0_inference

    features = {"canonical": {"id": 23, "best_of": 3}}
    expected = UnifiedPredictionResult(
        canonical_match_id=23,
        prob_a=0.63,
        prob_b=0.37,
        best_of=3,
        model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version,
        feature_version=C0_NATIVE.feature_version,
    )
    calls = []
    monkeypatch.setattr(c0_inference, "predict_c0", lambda payload: calls.append(payload) or expected)

    result = PredictionEngine.predict_from_features(features, C0_NATIVE)

    assert result is expected
    assert calls == [features]



@pytest.mark.parametrize("mode", ["linear", "logit_shrinkage"])
@pytest.mark.parametrize(
    "parameter,value",
    [("alpha", float("nan")), ("alpha", 1.1), ("temperature", float("inf")),
     ("temperature", 0.0), ("temperature", -1.0)],
)
def test_hybrid_rejects_invalid_transform_parameters(mode, parameter, value):
    spec = HybridSpec(base_model=C0_NATIVE, blending_mode=mode, **{parameter: value})
    with pytest.raises(ValueError):
        PredictionEngine.blend_with_market(0.7, 0.6, spec)







def test_unavailable_model_selection_preserves_active_state():
    before = get_active_model(), get_active_hybrid()
    with pytest.raises(ValueError):
        set_active_model(A1_CONSOLIDATED)
    assert (get_active_model(), get_active_hybrid()) == before


def test_environment_hybrid_override_cannot_reuse_another_base(monkeypatch):
    try:
        set_active_model(EXP078_LINEAR)
        monkeypatch.setenv("ACTIVE_PREDICTION_MODEL", BAYESIAN_SHRUNK_HYBRID.name)
        assert get_active_hybrid().base_model.name == get_active_model().metadata["base_model_name"]
    finally:
        set_active_model(C0_NATIVE)


def test_unknown_environment_model_cannot_fall_back(monkeypatch):
    monkeypatch.setenv("ACTIVE_PREDICTION_MODEL", "nonexistent-model")
    with pytest.raises(ValueError):
        get_active_model()
