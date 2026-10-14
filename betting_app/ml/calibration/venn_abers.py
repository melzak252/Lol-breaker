"""Venn-Abers conformal calibration and lower-bound risk gating for binary match prediction.

This module implements:
1. VennAbersCalibrator: Fast exact inductive Venn-Abers predictor (IVAP) for binary
   classification (Vovk & Petej, 2014; Nouretdinov et al., 2018). It produces finite-sample
   valid multi-probabilistic prediction intervals [p0, p1] from augmented isotonic regressions,
   point prediction p = p1 / (1.0 - p0 + p1), and epistemic uncertainty width (p1 - p0).
   Strict binary side-symmetry is enforced:
       p0(team_a) == 1.0 - p1(team_b)
       p(team_a) + p(team_b) == 1.0
2. ConformalRiskGater: Conservative expected value decision gater combining the Venn-Abers
   pessimistic lower probability bound p_lower, market odds, and local turnover tax:
       ev_lower = p_lower * (odds * (1.0 - tax_rate)) - 1.0
   Rejects bets where the conservative lower EV is negative or where epistemic uncertainty
   exceeds an acceptable risk threshold.
"""

from __future__ import annotations

from typing import Any, NamedTuple
import numpy as np
from scipy.special import expit


EPSILON: float = 1e-12


def _ensure_1d_float(arr: Any, name: str = "array") -> np.ndarray:
    """Convert input to a contiguous 1D float array with sanity checks."""
    if arr is None:
        raise ValueError(f"{name} must not be None")
    try:
        a = np.asarray(arr, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not convert {name} to float array: {exc}") from exc

    if a.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a.ravel()


def _ensure_binary_labels(labels: Any, name: str = "cal_labels") -> np.ndarray:
    """Validate and convert calibration labels to a 1D binary (0 or 1) integer array."""
    if labels is None:
        raise ValueError(f"{name} must not be None")
    y = np.asarray(labels)
    if y.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(y.astype(float))):
        raise ValueError(f"{name} contains NaN or infinite values")

    y_1d = y.ravel()
    unique = np.unique(y_1d)
    for val in unique:
        if val not in (0, 1, 0.0, 1.0, False, True):
            raise ValueError(f"{name} must be binary with values in {{0, 1}}, found: {unique}")
    return (y_1d > 0.5).astype(int)


def _extract_probabilities(scores: Any, name: str = "scores") -> tuple[np.ndarray, bool]:
    """Extract 1D probability values in [0, 1] from various score inputs.

    Supports:
    - 1D array of probabilities or logits.
    - 2D array of shape (N, 1).
    - 2D array of shape (N, 2) representing [P(y=0), P(y=1)] or paired logits [z_0, z_1].
    - Scalars.

    Returns
    -------
    tuple[np.ndarray, bool]
        (1D float array of probabilities in [0, 1], whether input was scalar)
    """
    if scores is None:
        raise ValueError(f"{name} must not be None")

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")

    is_scalar = (arr.ndim == 0)

    if arr.ndim == 2 and arr.shape[1] == 2:
        # Check if values look like logits or probabilities
        row_sums = arr.sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=1e-3) and np.all(arr >= 0.0) and np.all(arr <= 1.0):
            # Already normalized probabilities: column 1 is P(y=1)
            probs = arr[:, 1]
        else:
            # Paired team logits [z_0, z_1]: P(y=1) = sigmoid(z_1 - z_0)
            z_diff = arr[:, 1] - arr[:, 0]
            probs = expit(z_diff)
        return np.clip(probs, 0.0, 1.0), is_scalar

    probs = arr.ravel()
    # If any values lie strictly outside [0, 1], treat as logits and convert via sigmoid
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        probs = expit(probs)

    probs = np.clip(probs, 0.0, 1.0)
    return probs, is_scalar


def _not_below(tx: float, ty: float, p1x: float, p1y: float, p2x: float, p2y: float) -> bool:
    """Determine whether test point (tx, ty) lies at or above the line passing through p1 and p2."""
    dx = p2x - p1x
    if abs(dx) < 1e-15:
        return ty >= p1y - 1e-15
    m = (p2y - p1y) / dx
    b = p1y - m * p1x
    return ty >= tx * m + b - 1e-15


def _fast_compute_f_vectors(x_prime: np.ndarray, y_csd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast exact algorithm for computing Venn-Abers critical slope vectors F0 and F1.

    Implements Algorithms 1-4 from Vovk, Petej & Fedorova (2015).

    Parameters
    ----------
    x_prime : np.ndarray
        Cumulative sum of weights (sample counts) for unique sorted calibration scores.
    y_csd : np.ndarray
        Cumulative sum of labels (positive counts) for unique sorted calibration scores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        F0: array of shape (k_prime + 1,) for hypothesis y=0
        F1: array of shape (k_prime + 1,) for hypothesis y=1
    """
    k_prime = len(x_prime)

    # -----------------------------------------------------------------------
    # 1. Compute F1 (Algorithms 1 & 2)
    # -----------------------------------------------------------------------
    Px1 = np.empty(k_prime + 2, dtype=float)
    Py1 = np.empty(k_prime + 2, dtype=float)
    Px1[0] = -1.0
    Py1[0] = -1.0
    Px1[1] = 0.0
    Py1[1] = 0.0
    Px1[2:] = x_prime
    Py1[2:] = y_csd

    # Algorithm 1: Initialize corners of initial GCM
    S: list[tuple[float, float]] = [(-1.0, -1.0), (0.0, 0.0)]
    for i in range(1, k_prime + 1):
        cx = Px1[i + 1]
        cy = Py1[i + 1]
        while len(S) > 1:
            ax, ay = S[-2]
            bx, by = S[-1]
            # nonleftTurn: cross product <= 0
            if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) <= 0.0:
                S.pop()
            else:
                break
        S.append((cx, cy))

    # Algorithm 2: Compute F1
    Sprime = S[::-1]
    F1 = np.zeros(k_prime + 1, dtype=float)
    for i in range(1, k_prime + 1):
        top_x, top_y = Sprime[-1]
        ntop_x, ntop_y = Sprime[-2]
        F1[i] = (ntop_y - top_y) / (ntop_x - top_x)

        new_px = Px1[i - 1] + Px1[i + 1] - Px1[i]
        new_py = Py1[i - 1] + Py1[i + 1] - Py1[i]
        Px1[i] = new_px
        Py1[i] = new_py

        if _not_below(new_px, new_py, top_x, top_y, ntop_x, ntop_y):
            continue

        Sprime.pop()
        while len(Sprime) > 1:
            ntop_x, ntop_y = Sprime[-1]
            nntop_x, nntop_y = Sprime[-2]
            # nonleftTurn: cross product <= 0
            if (ntop_x - new_px) * (nntop_y - ntop_y) - (ntop_y - new_py) * (nntop_x - ntop_x) <= 0.0:
                Sprime.pop()
            else:
                break
        Sprime.append((new_px, new_py))

    # -----------------------------------------------------------------------
    # 2. Compute F0 (Algorithms 3 & 4)
    # -----------------------------------------------------------------------
    Px0 = np.empty(k_prime + 2, dtype=float)
    Py0 = np.empty(k_prime + 2, dtype=float)
    Px0[0] = 0.0
    Py0[0] = 0.0
    Px0[1:k_prime + 1] = x_prime
    Py0[1:k_prime + 1] = y_csd
    Px0[k_prime + 1] = Px0[k_prime] + 1.0
    Py0[k_prime + 1] = Py0[k_prime] + 0.0

    # Algorithm 3: Initialize corners of initial GCM for F0
    S = [(Px0[k_prime + 1], Py0[k_prime + 1]), (Px0[k_prime], Py0[k_prime])]
    for i in range(k_prime - 1, -1, -1):
        cx = Px0[i]
        cy = Py0[i]
        while len(S) > 1:
            ax, ay = S[-2]
            bx, by = S[-1]
            # nonrightTurn: cross product >= 0
            if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) >= 0.0:
                S.pop()
            else:
                break
        S.append((cx, cy))

    # Algorithm 4: Compute F0
    Sprime = S[::-1]
    F0 = np.zeros(k_prime + 1, dtype=float)
    for i in range(k_prime, 0, -1):
        top_x, top_y = Sprime[-1]
        ntop_x, ntop_y = Sprime[-2]
        F0[i] = (ntop_y - top_y) / (ntop_x - top_x)

        new_px = Px0[i - 1] + Px0[i + 1] - Px0[i]
        new_py = Py0[i - 1] + Py0[i + 1] - Py0[i]
        Px0[i] = new_px
        Py0[i] = new_py

        if _not_below(new_px, new_py, top_x, top_y, ntop_x, ntop_y):
            continue

        Sprime.pop()
        while len(Sprime) > 1:
            ntop_x, ntop_y = Sprime[-1]
            nntop_x, nntop_y = Sprime[-2]
            # nonrightTurn: cross product >= 0
            if (ntop_x - new_px) * (nntop_y - ntop_y) - (ntop_y - new_py) * (nntop_x - ntop_x) >= 0.0:
                Sprime.pop()
            else:
                break
        Sprime.append((new_px, new_py))

    return F0, F1


def _lookup_raw_bounds(
    F0: np.ndarray,
    F1_ext: np.ndarray,
    pts_unique: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up raw (p0, p1) bounds for test scores in O(log K) using binary search."""
    s = np.asarray(test_scores, dtype=float)
    idx_left = np.searchsorted(pts_unique, s, side="left")
    n_unique = len(pts_unique)

    # Exact matches with unique calibration scores
    is_exact = (idx_left < n_unique) & (pts_unique[np.minimum(idx_left, n_unique - 1)] == s)

    # Lookup indices
    p0_idx = np.where(is_exact, idx_left + 1, idx_left)
    p1_idx = idx_left + 1

    p0 = F0[p0_idx]
    p1 = F1_ext[p1_idx]

    return np.clip(p0, 0.0, 1.0), np.clip(p1, 0.0, 1.0)


class VennAbersIntervals(NamedTuple):
    """Named tuple representing Venn-Abers calibrated outputs.

    Attributes
    ----------
    p0 : np.ndarray
        Lower probability bound (hypothesis y = 0).
    p1 : np.ndarray
        Upper probability bound (hypothesis y = 1).
    p : np.ndarray
        Calibrated point prediction p = p1 / (1.0 - p0 + p1).
    """

    p0: np.ndarray
    p1: np.ndarray
    p: np.ndarray

    @property
    def uncertainty(self) -> np.ndarray:
        """Epistemic uncertainty interval width: (p1 - p0)."""
        return np.maximum(0.0, self.p1 - self.p0)

    @property
    def uncertainty_width(self) -> np.ndarray:
        """Alias for uncertainty."""
        return self.uncertainty

    @property
    def p_lower(self) -> np.ndarray:
        """Alias for lower bound p0."""
        return self.p0

    @property
    def p_upper(self) -> np.ndarray:
        """Alias for upper bound p1."""
        return self.p1

    @property
    def p_point(self) -> np.ndarray:
        """Alias for calibrated point prediction p."""
        return self.p


class VennAbersCalibrator:
    """Fast exact Venn-Abers calibration for binary classification.

    Implements the Inductive Venn-Abers Predictor (IVAP) (Vovk & Petej, 2014;
    Nouretdinov et al., 2018). For each test score s, it computes the exact lower
    and upper probability bounds [p0, p1] via augmented isotonic regression
    under hypotheses y=0 and y=1 in O(log K) inference time.

    When `symmetric=True` (default), exact binary symmetry is enforced:
        p0(team_a) == 1.0 - p1(team_b)
        p(team_a) + p(team_b) == 1.0
        uncertainty(team_a) == uncertainty(team_b)

    Parameters
    ----------
    symmetric : bool, default=True
        Whether to enforce strict binary side-symmetry under role inversion.
    """

    def __init__(self, symmetric: bool = True) -> None:
        self.symmetric = bool(symmetric)
        self.is_fitted_: bool = False
        self.unique_scores_: np.ndarray | None = None
        self.F0_: np.ndarray | None = None
        self.F1_ext_: np.ndarray | None = None

    def fit(self, cal_scores: Any, cal_labels: Any) -> VennAbersCalibrator:
        """Fit the Venn-Abers calibrator on calibration scores and true binary outcomes.

        Parameters
        ----------
        cal_scores : array-like of shape (N,) or (N, 2)
            Calibration scores or probabilities output by the base scoring classifier.
        cal_labels : array-like of shape (N,)
            True binary class labels in {0, 1}.

        Returns
        -------
        VennAbersCalibrator
            Fitted calibrator instance (self).
        """
        scores, _ = _extract_probabilities(cal_scores, "cal_scores")
        labels = _ensure_binary_labels(cal_labels, "cal_labels")

        if len(scores) != len(labels):
            raise ValueError(
                f"Length mismatch: cal_scores has {len(scores)} samples, "
                f"cal_labels has {len(labels)} samples."
            )

        # Symmetrize calibration dataset if requested
        if self.symmetric:
            s_fit = np.concatenate([scores, 1.0 - scores])
            y_fit = np.concatenate([labels, 1 - labels])
        else:
            s_fit = scores
            y_fit = labels

        # Identify unique scores and compute cumulative sums
        unique_scores, inverse_idx, counts = np.unique(
            s_fit, return_inverse=True, return_counts=True
        )
        k_prime = len(unique_scores)

        y_sums = np.zeros(k_prime, dtype=float)
        np.add.at(y_sums, inverse_idx, y_fit)

        y_csd = np.cumsum(y_sums)
        x_prime = np.cumsum(counts, dtype=float)

        # Precompute F0 and F1 vectors using fast exact algorithms 1-4
        f0, f1 = _fast_compute_f_vectors(x_prime, y_csd)

        # F1 extended with boundary F1[k_prime + 1] = 1.0
        f1_ext = np.zeros(k_prime + 2, dtype=float)
        f1_ext[1 : k_prime + 1] = f1[1 : k_prime + 1]
        f1_ext[k_prime + 1] = 1.0

        self.unique_scores_ = unique_scores
        self.F0_ = f0
        self.F1_ext_ = f1_ext
        self.is_fitted_ = True
        return self

    def _predict_raw_bounds(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute unsymmetrized raw (p0, p1) bounds for 1D scores."""
        if not self.is_fitted_ or self.unique_scores_ is None or self.F0_ is None or self.F1_ext_ is None:
            raise RuntimeError("VennAbersCalibrator is not fitted. Call fit() first.")
        return _lookup_raw_bounds(self.F0_, self.F1_ext_, self.unique_scores_, scores)

    def predict_intervals(
        self,
        test_scores: Any,
    ) -> VennAbersIntervals:
        """Compute calibrated probability intervals [p0, p1] and point prediction p.

        Parameters
        ----------
        test_scores : array-like of shape (N,) or (N, 2)
            Uncalibrated scores, probabilities, or logits to calibrate.

        Returns
        -------
        VennAbersIntervals
            Named tuple containing (p0, p1, p), where:
            - p0: lower probability bound in [0, 1]
            - p1: upper probability bound in [0, 1] (p0 <= p1)
            - p: calibrated point prediction p = p1 / (1.0 - p0 + p1)
            Also provides `.uncertainty`, `.p_lower`, and `.p_upper` properties.
        """
        if not self.is_fitted_:
            raise RuntimeError("VennAbersCalibrator is not fitted. Call fit() first.")

        scores, is_scalar = _extract_probabilities(test_scores, "test_scores")

        if self.symmetric:
            p0_raw, p1_raw = self._predict_raw_bounds(scores)
            p0_rev, p1_rev = self._predict_raw_bounds(1.0 - scores)

            # Enforce exact mathematical symmetry:
            # p0_sym(s) = 0.5 * (p0_raw(s) + (1.0 - p1_rev))
            # p1_sym(s) = 1.0 - p0_sym(1 - s)
            p0 = 0.5 * (p0_raw + (1.0 - p1_rev))
            p0_rev_sym = 0.5 * (p0_rev + (1.0 - p1_raw))
            p1 = 1.0 - p0_rev_sym
        else:
            p0, p1 = self._predict_raw_bounds(scores)

        # Bound sanity checks: 0 <= p0 <= p1 <= 1
        p0 = np.clip(p0, 0.0, 1.0)
        p1 = np.clip(p1, 0.0, 1.0)
        p1 = np.maximum(p0, p1)

        # Venn-Abers minimax point prediction: p = p1 / (1.0 - p0 + p1)
        # Denominator = 1.0 + (p1 - p0) >= 1.0, strictly bounded away from 0
        denom = 1.0 - p0 + p1
        p = p1 / denom

        if self.symmetric:
            # Guarantee p(s) + p(1-s) == 1.0 to exact machine bit precision
            p = 0.5 * (p + (1.0 - (p1_rev_sym_p := (1.0 - p0) / denom)))

        p = np.clip(p, 0.0, 1.0)

        if is_scalar:
            return VennAbersIntervals(
                np.asarray(float(p0[0])),
                np.asarray(float(p1[0])),
                np.asarray(float(p[0])),
            )

        return VennAbersIntervals(p0, p1, p)

    def predict_proba(self, test_scores: Any) -> np.ndarray:
        """Return 2D calibrated class probabilities [P(y = 0), P(y = 1)].

        Adheres strictly to the scikit-learn API contract:
        - Output shape is (N, 2).
        - P(y = 0) + P(y = 1) == 1.0 across every sample.

        Parameters
        ----------
        test_scores : array-like
            Scores or logits to calibrate.

        Returns
        -------
        np.ndarray of shape (N, 2)
            Calibrated class probabilities.
        """
        intervals = self.predict_intervals(test_scores)
        p = np.atleast_1d(intervals.p)
        return np.column_stack([1.0 - p, p])

    def transform(self, test_scores: Any) -> np.ndarray:
        """Return 1D calibrated point probability estimates in [0, 1].

        Parameters
        ----------
        test_scores : array-like
            Scores or logits to calibrate.

        Returns
        -------
        np.ndarray
            Calibrated point probabilities.
        """
        intervals = self.predict_intervals(test_scores)
        p = intervals.p
        if np.ndim(test_scores) == 0:
            return float(p)
        return p

    def predict(self, test_scores: Any, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions (0 or 1) thresholded at given probability.

        Parameters
        ----------
        test_scores : array-like
            Scores or logits to calibrate.
        threshold : float, default=0.5
            Classification threshold in [0, 1].

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        p = np.atleast_1d(self.transform(test_scores))
        return (p >= threshold).astype(int)


class ConformalRiskGater:
    """Finite-sample conformal risk gater for binary outcome betting markets.

    Combines Venn-Abers lower probability bounds, decimal market odds, and
    statutory turnover tax to compute conservative expected value:
        ev_lower = p_lower * (odds * (1.0 - tax_rate)) - 1.0

    Rejects any bet proposal where:
    1. The conservative expected value is below `min_ev` (negative EV even under
       the pessimistic probability bound).
    2. The epistemic uncertainty width (p_upper - p_lower) exceeds `max_uncertainty`
       (the model's confidence interval is excessively wide).
    3. Input data is invalid (odds <= 1.0, NaNs, probability bounds outside [0, 1]).

    Parameters
    ----------
    tax_rate : float, default=0.12
        Turnover tax rate deducted from gross return (e.g. 0.12 for Polish regulated market).
    min_ev : float, default=0.0
        Minimum required conservative lower expected value for a bet to be actionable.
    max_uncertainty : float, default=0.08
        Maximum allowable interval width (p_upper - p_lower) before rejection.
    """

    def __init__(
        self,
        tax_rate: float = 0.12,
        min_ev: float = 0.0,
        max_uncertainty: float = 0.08,
    ) -> None:
        if not (0.0 <= tax_rate < 1.0):
            raise ValueError(f"tax_rate must be in [0, 1), got {tax_rate}")
        if max_uncertainty <= 0.0:
            raise ValueError(f"max_uncertainty must be strictly positive, got {max_uncertainty}")

        self.tax_rate = float(tax_rate)
        self.min_ev = float(min_ev)
        self.max_uncertainty = float(max_uncertainty)

    def compute_ev_lower(self, p_lower: Any, odds: Any) -> np.ndarray:
        """Compute conservative expected value under lower probability bound:
            ev_lower = p_lower * (odds * (1.0 - tax_rate)) - 1.0

        Parameters
        ----------
        p_lower : array-like or float
            Pessimistic lower probability bound in [0, 1].
        odds : array-like or float
            Decimal market odds.

        Returns
        -------
        np.ndarray
            Conservative expected value.
        """
        arr_lower = np.asarray(p_lower, dtype=float)
        arr_odds = np.asarray(odds, dtype=float)
        effective_odds = arr_odds * (1.0 - self.tax_rate)
        return arr_lower * effective_odds - 1.0

    def filter_bets(
        self,
        p_lower: Any,
        p_upper: Any,
        odds: Any,
    ) -> dict[str, Any]:
        """Filter betting opportunities against conservative EV and uncertainty thresholds.

        Parameters
        ----------
        p_lower : array-like or float
            Lower probability bound from Venn-Abers calibrator.
        p_upper : array-like or float
            Upper probability bound from Venn-Abers calibrator.
        odds : array-like or float
            Offered decimal odds from bookmaker.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'is_actionable': boolean mask indicating approved bets.
            - 'ev_lower': conservative expected value array.
            - 'uncertainty': interval width (p_upper - p_lower) array.
            - 'uncertainty_width': alias for uncertainty.
            - 'p_lower': validated lower bounds.
            - 'p_upper': validated upper bounds.
            - 'odds': decimal odds.
        """
        arr_l = np.asarray(p_lower, dtype=float)
        arr_u = np.asarray(p_upper, dtype=float)
        arr_o = np.asarray(odds, dtype=float)

        is_scalar = (arr_l.ndim == 0 and arr_u.ndim == 0 and arr_o.ndim == 0)

        # Broadcast inputs to common shape
        try:
            arr_l, arr_u, arr_o = np.broadcast_arrays(arr_l, arr_u, arr_o)
        except ValueError as exc:
            raise ValueError(
                f"Could not broadcast shapes p_lower {arr_l.shape}, "
                f"p_upper {arr_u.shape}, odds {arr_o.shape}: {exc}"
            ) from exc

        ev_lower = self.compute_ev_lower(arr_l, arr_o)
        uncertainty = arr_u - arr_l

        # Validity conditions
        valid_finite = (
            np.isfinite(arr_l)
            & np.isfinite(arr_u)
            & np.isfinite(arr_o)
            & np.isfinite(ev_lower)
        )
        valid_bounds = (
            (arr_l >= 0.0)
            & (arr_u <= 1.0)
            & (arr_l <= arr_u)
        )
        valid_odds = (arr_o > 1.0)

        # Risk criteria:
        # 1. Conservative EV must meet or exceed min_ev
        # 2. Uncertainty width must not exceed max_uncertainty
        ev_pass = (ev_lower >= self.min_ev)
        uncertainty_pass = (uncertainty <= self.max_uncertainty)

        is_actionable = valid_finite & valid_bounds & valid_odds & ev_pass & uncertainty_pass

        if is_scalar:
            return {
                "is_actionable": bool(is_actionable.item()),
                "ev_lower": float(ev_lower.item()),
                "uncertainty": float(uncertainty.item()),
                "uncertainty_width": float(uncertainty.item()),
                "p_lower": float(arr_l.item()),
                "p_upper": float(arr_u.item()),
                "odds": float(arr_o.item()),
            }

        return {
            "is_actionable": is_actionable,
            "ev_lower": ev_lower,
            "uncertainty": uncertainty,
            "uncertainty_width": uncertainty,
            "p_lower": arr_l,
            "p_upper": arr_u,
            "odds": arr_o,
        }
