"""Candidate model factory for regular retraining."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betting_app.ml.training.types import ModelCandidateSpec


def default_candidate_specs() -> list[ModelCandidateSpec]:
    return [
        ModelCandidateSpec("logreg_l2", "logistic_regression", {"C": 1.0, "max_iter": 1000}),
        ModelCandidateSpec("logreg_l1", "logistic_regression", {"C": 0.5, "penalty": "l1", "solver": "liblinear", "max_iter": 1000}),
        ModelCandidateSpec("hist_gbdt", "hist_gradient_boosting", {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15}),
        ModelCandidateSpec("random_forest", "random_forest", {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
    ]


def build_estimator(spec: ModelCandidateSpec):
    if spec.estimator_type == "logistic_regression":
        params = {"max_iter": 1000, **spec.params}
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
    if spec.estimator_type == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=42, **spec.params)),
        ])
    if spec.estimator_type == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**spec.params)),
        ])
    raise ValueError(f"Unsupported estimator type: {spec.estimator_type}")
