"""Central model registry providing the Single Source of Truth for models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from betting_app.core.models.contract import HybridSpec, ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------------------------------------------------------------
# Known Model Specifications
# -----------------------------------------------------------------------------


C0_NATIVE = ModelSpec(
    name="Causal-C0",
    version="c0-native-2026-w32-e12-v1",
    feature_version="native-c0-history16-v1",
    artifact_path=PROJECT_ROOT / "data" / "06_models" / "a0_data_focus" / "a0_data_focus_20260929_192443",
    description="Frozen native C0 series predictor using native pre-match history features.",
    metadata={
        "operational": True,
        "symmetric": True,
        "artifact_release": "data/06_models/a0_data_focus/a0_data_focus_20260929_192443",
        "artifact_kind": "native_control_trio_shared_calibration",
    },
)

EXP078_LINEAR = ModelSpec(
    name="Operational-PlayerTeamRatings-W20",
    version="exp078-symmetric-series-v1",
    feature_version="ratings-w20-symmetric-series-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="linear_symmetric",
    artifact_path=PROJECT_ROOT / "betting_app" / "models" / "exp078_symmetric_series_v1.json",
    description="Symmetric series-level linear regression on ratings-w20-symmetric-series-v1 features.",
    metadata={"symmetric": True},
)

EXP039_THESIS = ModelSpec(
    name="Sym-Cal LR-ElasticNet-W20-Binomial",
    version="exp-039",
    feature_version="thesis-exp039",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="map",
    has_uncertainty=False,
    family="elastic_net",
    artifact_path=PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib",
    description="Frozen thesis reference model: symmetric calibrated ElasticNet with binomial series simulation.",
    metadata={"frozen": True, "academic_baseline": True},
)

A1_CONSOLIDATED = ModelSpec(
    name="Consolidated-A1",
    version="a1-consolidated-v1",
    feature_version="causal-a0-research-bank-residual-v1",
    ratings_version="latest-full",
    w20_version="w20-latest",
    prediction_target="series",
    has_uncertainty=False,
    family="a1_engine",
    description="Research-only residual over Causal A0; no validated live A0 feature adapter or macro artifact is configured.",
    metadata={
        "operational": False,
        "unavailable_reason": "Consolidated A1 requires a validated Causal A0 adapter and out-of-sample macro artifact; rating consensus is not Causal A0.",
        "tier_scale": 0.94,
        "symmetric": True,
    },
)

ACTIVE_MODEL_NAME = C0_NATIVE.name
ACTIVE_MODEL_VERSION = C0_NATIVE.version
ACTIVE_MODEL_FAMILY = C0_NATIVE.family

BAYESIAN_SHRUNK_HYBRID = ModelSpec(
    name="Hybrid-Operational-Market",
    version="c0-native-2026-w32-e12-v1-a0.50-t1.00",
    feature_version=C0_NATIVE.feature_version,
    ratings_version=C0_NATIVE.ratings_version,
    w20_version=C0_NATIVE.w20_version,
    prediction_target="series",
    has_uncertainty=False,
    family="bayesian_market_hybrid",
    description="Native C0 series predictions blended with no-vig series market odds.",
    metadata={
        "base_model_name": C0_NATIVE.name,
        "alpha": 0.50,
        "temperature": 1.0,
        "blending_mode": "logit_shrinkage",
        "symmetric": True,
    },
)

REGISTERED_MODELS: dict[str, ModelSpec] = {
    C0_NATIVE.name: C0_NATIVE,
    BAYESIAN_SHRUNK_HYBRID.name: BAYESIAN_SHRUNK_HYBRID,
    A1_CONSOLIDATED.name: A1_CONSOLIDATED,
    EXP078_LINEAR.name: EXP078_LINEAR,
    EXP039_THESIS.name: EXP039_THESIS,
    "c0": C0_NATIVE,
    "native_c0": C0_NATIVE,
    "shrunk_hybrid": BAYESIAN_SHRUNK_HYBRID,
    "bayesian_hybrid": BAYESIAN_SHRUNK_HYBRID,
    "a1": A1_CONSOLIDATED,
    "a1_consolidated": A1_CONSOLIDATED,
    "exp078": EXP078_LINEAR,
    "exp039": EXP039_THESIS,
}

# -----------------------------------------------------------------------------
# Active Model and Hybrid State
# -----------------------------------------------------------------------------
_ACTIVE_MODEL: ModelSpec = C0_NATIVE

# The separately selectable market-shrinkage policy uses native C0 once.
_ACTIVE_HYBRID: HybridSpec = HybridSpec(
    base_model=C0_NATIVE,
    hybrid_model_name=BAYESIAN_SHRUNK_HYBRID.name,
    alpha=0.50,
    temperature=1.0,
    blending_mode="logit_shrinkage",
    custom_version=BAYESIAN_SHRUNK_HYBRID.version,
)
# Thesis hybrid reference
THESIS_HYBRID: HybridSpec = HybridSpec(
    base_model=EXP039_THESIS,
    hybrid_model_name="Hybrid-Thesis-Market",
    alpha=0.35,
    temperature=0.80,
    blending_mode="linear",
    custom_version="a0.35-t0.80",
)


def get_hybrid_spec(model: ModelSpec) -> HybridSpec:
    """Resolve the pure base and blending policy declared by one model version."""
    if model.family != "bayesian_market_hybrid":
        raise ValueError("Expected a hybrid ModelSpec.")
    base = get_model(model.metadata.get("base_model_name", ""))
    if base is None or base.family == "bayesian_market_hybrid":
        raise ValueError("A hybrid must declare a registered pure sports model.")
    if base.metadata.get("operational") is False:
        raise ValueError(base.metadata["unavailable_reason"])
    return HybridSpec(
        base_model=base,
        hybrid_model_name=model.name,
        alpha=float(model.metadata["alpha"]),
        temperature=float(model.metadata["temperature"]),
        blending_mode=model.metadata.get("blending_mode", "logit_shrinkage"),
        custom_version=model.version,
    )


def get_active_model() -> ModelSpec:
    """Return the configured model, rejecting unknown or unavailable overrides."""
    override = os.getenv("ACTIVE_PREDICTION_MODEL")
    spec = get_model(override) if override else _ACTIVE_MODEL
    if spec is None:
        raise ValueError(f"Unknown ACTIVE_PREDICTION_MODEL: {override}")
    if spec.metadata.get("operational") is False:
        raise ValueError(spec.metadata["unavailable_reason"])
    return spec


def get_active_model_spec() -> ModelSpec:
    """Return the active operational model specification."""
    return get_active_model()


def set_active_model(model_or_name: ModelSpec | str) -> None:
    """Switch the active sports model or explicitly selected hybrid model."""
    global _ACTIVE_MODEL, _ACTIVE_HYBRID
    if isinstance(model_or_name, str):
        spec = get_model(model_or_name)
        if spec is None:
            raise KeyError(f"Unknown model: {model_or_name}. Registered: {list(REGISTERED_MODELS.keys())}")
    else:
        spec = model_or_name
    if spec.metadata.get("operational") is False:
        raise ValueError(spec.metadata["unavailable_reason"])
    if spec.family == "bayesian_market_hybrid":
        hybrid = get_hybrid_spec(spec)
    else:
        hybrid = HybridSpec(
            base_model=spec,
            hybrid_model_name=_ACTIVE_HYBRID.hybrid_model_name,
            alpha=_ACTIVE_HYBRID.alpha,
            temperature=_ACTIVE_HYBRID.temperature,
            blending_mode=_ACTIVE_HYBRID.blending_mode,
            custom_version=_ACTIVE_HYBRID.custom_version if _ACTIVE_HYBRID.base_model == spec else None,
        )
    _ACTIVE_MODEL = spec
    _ACTIVE_HYBRID = hybrid


def get_active_hybrid() -> HybridSpec:
    """Return market shrinkage synchronized with the currently selected sports model."""
    current_active = get_active_model()
    if current_active.family == "bayesian_market_hybrid":
        return get_hybrid_spec(current_active)
    if _ACTIVE_HYBRID.base_model != current_active:
        return HybridSpec(
            base_model=current_active,
            hybrid_model_name=_ACTIVE_HYBRID.hybrid_model_name,
            alpha=_ACTIVE_HYBRID.alpha,
            temperature=_ACTIVE_HYBRID.temperature,
            blending_mode=_ACTIVE_HYBRID.blending_mode,
        )
    return _ACTIVE_HYBRID

def set_active_hybrid(hybrid_spec: HybridSpec) -> None:
    """Set the separate market-shrinkage policy for the active sports model."""
    global _ACTIVE_HYBRID
    if hybrid_spec.base_model.family == "bayesian_market_hybrid":
        raise ValueError("A hybrid must use a pure sports model, not another hybrid.")
    if hybrid_spec.base_model.metadata.get("operational") is False:
        raise ValueError(hybrid_spec.base_model.metadata["unavailable_reason"])
    active = get_active_model()
    if active.family == "bayesian_market_hybrid":
        if hybrid_spec != get_hybrid_spec(active):
            raise ValueError("Select a separately versioned ModelSpec before changing its hybrid policy.")
    elif hybrid_spec.base_model != active:
        raise ValueError("Hybrid base must match the active sports model.")
    if (
        hybrid_spec.base_model == C0_NATIVE
        and hybrid_spec.hybrid_model_name == BAYESIAN_SHRUNK_HYBRID.name
        and hybrid_spec.alpha == BAYESIAN_SHRUNK_HYBRID.metadata["alpha"]
        and hybrid_spec.temperature == BAYESIAN_SHRUNK_HYBRID.metadata["temperature"]
        and hybrid_spec.blending_mode == BAYESIAN_SHRUNK_HYBRID.metadata["blending_mode"]
        and hybrid_spec.custom_version == BAYESIAN_SHRUNK_HYBRID.version
    ):
        _ACTIVE_HYBRID = hybrid_spec
    else:
        _ACTIVE_HYBRID = HybridSpec(
            base_model=hybrid_spec.base_model,
            hybrid_model_name=hybrid_spec.hybrid_model_name,
            alpha=hybrid_spec.alpha,
            temperature=hybrid_spec.temperature,
            blending_mode=hybrid_spec.blending_mode,
        )

def get_thesis_model() -> ModelSpec:
    """Return the frozen thesis baseline model specification."""
    return EXP039_THESIS


def get_thesis_hybrid() -> HybridSpec:
    """Return the frozen thesis hybrid specification."""
    return THESIS_HYBRID


def get_model(name_or_alias: str) -> ModelSpec | None:
    """Retrieve a model by key, alias, name, or version (case/dash insensitive)."""
    if name_or_alias in REGISTERED_MODELS:
        return REGISTERED_MODELS[name_or_alias]
    # Normalize model names, versions, and supported aliases.
    normalized = name_or_alias.lower().replace("-", "").replace("_", "").strip()
    for key, spec in REGISTERED_MODELS.items():
        key_norm = key.lower().replace("-", "").replace("_", "")
        name_norm = spec.name.lower().replace("-", "").replace("_", "")
        ver_norm = spec.version.lower().replace("-", "").replace("_", "")
        if normalized in (key_norm, name_norm, ver_norm):
            return spec
    return None


def list_registered_models() -> list[dict[str, Any]]:
    """List all registered models with their metadata for API responses."""
    active = get_active_model()
    seen = set()
    result = []
    for model in (C0_NATIVE, BAYESIAN_SHRUNK_HYBRID, EXP078_LINEAR, EXP039_THESIS):
        if model.name in seen:
            continue
        seen.add(model.name)
        result.append({
            "name": model.name,
            "version": model.version,
            "feature_version": model.feature_version,
            "ratings_version": model.ratings_version,
            "w20_version": model.w20_version,
            "prediction_target": model.prediction_target,
            "target": model.prediction_target,
            "has_uncertainty": model.has_uncertainty,
            "has_epistemic_uncertainty": model.has_uncertainty,
            "family": model.family,
            "is_active_operational": model.name == active.name and model.version == active.version,
            "description": model.description,
            "metadata": model.metadata,
        })
    return result

def get_active_prediction_db_params() -> dict[str, str]:
    """Return SQL bind params for filtering active operational & hybrid predictions."""
    hybrid = get_active_hybrid()
    base_sports = hybrid.base_model
    return {
        "sn": base_sports.name,
        "sv": base_sports.version,
        "hn": hybrid.hybrid_model_name,
        "hv": hybrid.hybrid_model_version,
        "operational_name": base_sports.name,
        "operational_version": base_sports.version,
        "hybrid_name": hybrid.hybrid_model_name,
        "hybrid_version": hybrid.hybrid_model_version,
    }
