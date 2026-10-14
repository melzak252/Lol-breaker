"""Registry-backed inference for shadow/production ML models."""

from betting_app.ml.inference.registry_inference import run_registry_shadow_inference
from betting_app.ml.inference.types import InferencePrediction, RegisteredModelArtifact, ShadowInferenceResult

__all__ = [
    "InferencePrediction",
    "RegisteredModelArtifact",
    "ShadowInferenceResult",
    "run_registry_shadow_inference",
]
