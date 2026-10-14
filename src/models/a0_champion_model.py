"""Champion-augmented specialist architecture and residual connector.

Implements:
1. ChampionSpecialist: LightGBM classifier trained on 30 differential champion features
   + 12 player rating differences (Elo, Glicko, and team means).
   Enforces exact side-symmetry via reflection training and symmetric inference:
       p_sym(X) = 0.5 * (p_raw(X) + 1.0 - p_raw(-X))
   which guarantees p_sym(-X) = 1.0 - p_sym(X) and p_sym(0) = 0.5 (neutral cold start).

2. ChampionResidualConnector: Logit-space residual connector combining frozen baseline
   C0 probabilities with champion specialist predictions:
       logit(p_cand) = logit(p_C0) + gamma * (logit(p_champ) - offset)
   where offset = 0.0 (exact symmetry) and gamma in [0.0, 0.5] is fitted strictly on CAL
   (1,328 series in 2025H2) using bounded L-BFGS-B / Brent scalar minimization.
   When gamma = 0, p_cand = p_C0 exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from src.features.champion_pool_features import DIFF_FEATURE_NAMES

RATING_DIFF_NAMES: list[str] = [
    "diff_top_elo",
    "diff_jungle_elo",
    "diff_mid_elo",
    "diff_adc_elo",
    "diff_support_elo",
    "diff_top_gl_mu",
    "diff_jungle_gl_mu",
    "diff_mid_gl_mu",
    "diff_adc_gl_mu",
    "diff_support_gl_mu",
    "diff_mean_player_elo",
    "diff_mean_player_gl_mu",
]

SPECIALIST_FEATURE_NAMES: list[str] = DIFF_FEATURE_NAMES + RATING_DIFF_NAMES
assert len(SPECIALIST_FEATURE_NAMES) == 42, f"Expected 42 features, got {len(SPECIALIST_FEATURE_NAMES)}"


def build_specialist_features(
    diff_champ_features: np.ndarray,
    players_array: np.ndarray,
) -> np.ndarray:
    """Combine 30 differential champion features with 12 player rating differences.

    Args:
        diff_champ_features: Array of shape (N, 30), dtype float32.
        players_array: Array of shape (N, 2, 5, 21), dtype float32 from arrays.npz['players'].
            Column 0 is Elo, column 1 is Glicko mu.

    Returns:
        Array of shape (N, 42), dtype float32.
    """
    if diff_champ_features.ndim != 2 or diff_champ_features.shape[1] != 30:
        raise ValueError(f"Expected diff_champ_features of shape (N, 30), got {diff_champ_features.shape}")
    if players_array.ndim != 4 or players_array.shape[1:3] != (2, 5) or players_array.shape[3] < 2:
        raise ValueError(f"Expected players_array of shape (N, 2, 5, >=2), got {players_array.shape}")
    if diff_champ_features.shape[0] != players_array.shape[0]:
        raise ValueError(
            f"Length mismatch: champ features {diff_champ_features.shape[0]} vs players {players_array.shape[0]}"
        )

    # Elo diff per role: shape (N, 5)
    elo_diff = (players_array[:, 0, :, 0] - players_array[:, 1, :, 0]).astype(np.float32)
    # Glicko mu diff per role: shape (N, 5)
    gl_diff = (players_array[:, 0, :, 1] - players_array[:, 1, :, 1]).astype(np.float32)
    # Team mean rating diffs: shape (N, 1)
    mean_elo_diff = elo_diff.mean(axis=1, keepdims=True).astype(np.float32)
    mean_gl_diff = gl_diff.mean(axis=1, keepdims=True).astype(np.float32)

    rating_diffs = np.concatenate([elo_diff, gl_diff, mean_elo_diff, mean_gl_diff], axis=1)
    return np.concatenate([diff_champ_features.astype(np.float32), rating_diffs], axis=1)


class ChampionSpecialist:
    """LightGBM classifier on differential champion + rating features with exact symmetry."""

    def __init__(
        self,
        num_leaves: int = 15,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        min_child_samples: int = 50,
        colsample_bytree: float = 0.8,
        subsample: float = 0.8,
        n_estimators: int = 150,
        random_state: int = 20261005,
    ) -> None:
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_: lgb.LGBMClassifier | None = None
        self.feature_names_: list[str] = list(SPECIALIST_FEATURE_NAMES)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ChampionSpecialist:
        """Fit LightGBM with symmetric reflection augmentation."""
        X_clean = np.asarray(X_train, dtype=np.float32)
        y_clean = np.asarray(y_train, dtype=np.float32)

        # Reflection augmentation: (X, y) and (-X, 1 - y)
        X_sym = np.concatenate([X_clean, -X_clean], axis=0)
        y_sym = np.concatenate([y_clean, 1.0 - y_clean], axis=0)
        df_sym = pd.DataFrame(X_sym, columns=self.feature_names_)

        clf = lgb.LGBMClassifier(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            n_estimators=self.n_estimators,
            objective="binary",
            random_state=self.random_state,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(df_sym, y_sym)
        self.model_ = clf
        return self

    def predict_proba_sym(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with guaranteed side-swap symmetry.

        p_sym(X) = 0.5 * (p_raw(X) + 1.0 - p_raw(-X))
        Returns:
            Array of shape (N,), probability that side A (team 1) wins.
        """
        if self.model_ is None:
            raise RuntimeError("ChampionSpecialist must be fitted before predict_proba_sym")
        X_clean = np.asarray(X, dtype=np.float32)
        df_pos = pd.DataFrame(X_clean, columns=self.feature_names_)
        df_neg = pd.DataFrame(-X_clean, columns=self.feature_names_)
        p_raw = self.model_.predict_proba(df_pos)[:, 1]
        p_neg = self.model_.predict_proba(df_neg)[:, 1]
        p_sym = 0.5 * (p_raw + (1.0 - p_neg))
        return np.clip(p_sym, 1.0e-7, 1.0 - 1.0e-7)
    def save(self, output_path: Path | str) -> None:
        """Persist fitted model and hyperparameters."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "params": {
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_child_samples": self.min_child_samples,
                "colsample_bytree": self.colsample_bytree,
                "subsample": self.subsample,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names_,
            "model": self.model_,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path | str) -> ChampionSpecialist:
        """Load persisted model bundle."""
        bundle = joblib.load(path)
        instance = cls(**bundle["params"])
        instance.model_ = bundle["model"]
        instance.feature_names_ = bundle.get("feature_names", list(SPECIALIST_FEATURE_NAMES))
        return instance


class ChampionResidualConnector:
    """Logit-space residual connector combining baseline C0 with champion specialist."""

    MODES: tuple[str, ...] = ("full", "no_w20", "no_organization")

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.gammas_: dict[str, float] = {mode: 0.0 for mode in self.MODES}
        self.cal_losses_: dict[str, float] = {}
        self.baseline_cal_losses_: dict[str, float] = {}

    def fit_cal(
        self,
        p_c0_cal: dict[str, np.ndarray],
        p_champ_cal: np.ndarray,
        y_cal: np.ndarray,
        gamma_bounds: tuple[float, float] = (0.0, 0.5),
    ) -> ChampionResidualConnector:
        """Optimize gamma on CAL for each evaluation mode."""
        y_true = np.asarray(y_cal, dtype=np.float64)
        p_champ = np.clip(np.asarray(p_champ_cal, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
        logit_champ = logit(p_champ) - self.offset

        for mode in self.MODES:
            if mode not in p_c0_cal:
                raise KeyError(f"Missing mode {mode} in p_c0_cal")
            p_c0 = np.clip(np.asarray(p_c0_cal[mode], dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
            logit_c0 = logit(p_c0)

            # Baseline log loss with gamma = 0
            base_loss = float(
                -(y_true * np.log(p_c0) + (1.0 - y_true) * np.log(1.0 - p_c0)).mean()
            )
            self.baseline_cal_losses_[mode] = base_loss

            def objective(g: float) -> float:
                p_comb = expit(logit_c0 + g * logit_champ)
                p_comb = np.clip(p_comb, 1.0e-7, 1.0 - 1.0e-7)
                return float(-(y_true * np.log(p_comb) + (1.0 - y_true) * np.log(1.0 - p_comb)).mean())

            res = minimize_scalar(objective, bounds=gamma_bounds, method="bounded")
            opt_gamma = float(res.x)
            opt_loss = float(res.fun)

            # If optimization did not strictly improve, keep baseline gamma = 0.0
            if opt_loss >= base_loss:
                self.gammas_[mode] = 0.0
                self.cal_losses_[mode] = base_loss
            else:
                self.gammas_[mode] = opt_gamma
                self.cal_losses_[mode] = opt_loss

        return self

    def predict(
        self,
        p_c0: dict[str, np.ndarray],
        p_champ: np.ndarray,
        bo: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Produce blended candidate predictions for all modes."""
        p_champ_clean = np.clip(np.asarray(p_champ, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
        logit_champ = logit(p_champ_clean) - self.offset

        blended: dict[str, np.ndarray] = {}
        for mode in self.MODES:
            if mode not in p_c0:
                raise KeyError(f"Missing mode {mode} in p_c0")
            p_c0_mode = np.clip(np.asarray(p_c0[mode], dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
            logit_c0 = logit(p_c0_mode)
            gamma = self.gammas_.get(mode, 0.0)

            # Blended logit
            logit_cand = logit_c0 + gamma * logit_champ
            p_cand = expit(logit_cand)
            blended[mode] = np.clip(p_cand, 1.0e-7, 1.0 - 1.0e-7).astype(np.float64)

        return blended

    def save(self, output_path: Path | str) -> None:
        """Persist fitted connector parameters."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "offset": self.offset,
            "gammas": self.gammas_,
            "cal_losses": self.cal_losses_,
            "baseline_cal_losses": self.baseline_cal_losses_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ChampionResidualConnector:
        """Load persisted connector configuration."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        instance = cls(offset=payload.get("offset", 0.0))
        instance.gammas_ = payload.get("gammas", {m: 0.0 for m in cls.MODES})
        instance.cal_losses_ = payload.get("cal_losses", {})
        instance.baseline_cal_losses_ = payload.get("baseline_cal_losses", {})
        return instance
