"""Series scoreline laws preserving the supplied series probability.

The fixed Markov law is an explicit heuristic control. The learned law fits
TRAIN-only conditional losing-map distributions, independently of winner modeling.
Guarantees:
1. Exact probability mass conservation: sum(P(w_A, w_B)) == 1.0.
2. Side anti-symmetry: P(w_B, w_A | B, A) == P(w_A, w_B | A, B).
3. Exact marginal preservation: sum_{w_B} P(K, w_B) == p and sum_{w_A} P(w_A, K) == 1.0 - p.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax


DEFAULT_MOMENTUM_BETA: float = 0.35
FORMAT_SLOPES: dict[int, float] = {1: 1.0, 3: 1.34, 5: 1.62}

def _best_of(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (1, 3, 5):
        raise ValueError(f"best_of must be 1, 3, or 5, got {value!r}")
    return int(value)


def _probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"probability must be finite and in [0, 1], got {value!r}")
    return float(value)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("missing date")
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date {value!r}") from exc


def compute_scoreline_pmf(
    p_series: float,
    best_of: int,
    beta_momentum: float = DEFAULT_MOMENTUM_BETA,
) -> dict[tuple[int, int], float]:
    """Compute exact, mass-conserving, side-symmetric scoreline PMF.

    Parameters
    ----------
    p_series : float
        Series-win probability for Team A vs Team B, in [0, 1].
    best_of : int
        Best-of series length (1, 3, or 5).
    beta_momentum : float
        Logit shift per game win differential (w_1 - w_2).

    Returns
    -------
    dict[tuple[int, int], float]
        Dictionary mapping (wins_team_a, wins_team_b) to probability.
    """
    best_of = _best_of(best_of)
    p = _probability(p_series)
    if not math.isfinite(beta_momentum):
        raise ValueError("beta_momentum must be finite")

    if best_of == 1:
        return {(1, 0): p, (0, 1): 1.0 - p}

    needed = best_of // 2 + 1
    if p in (0.0, 1.0):
        # The limiting Markov paths sweep; retain all legal terminal categories.
        return {
            **{(needed, lost): p if lost == 0 else 0.0 for lost in range(needed)},
            **{(lost, needed): 1.0 - p if lost == 0 else 0.0 for lost in range(needed)},
        }
    slope = FORMAT_SLOPES.get(best_of, 1.0)
    l0 = logit(p) / slope

    # Expand the state graph: state is (w1, w2) -> path probability
    paths: dict[tuple[int, int], float] = {(0, 0): 1.0}
    terminal: dict[tuple[int, int], float] = {}

    while paths:
        next_paths: dict[tuple[int, int], float] = {}
        for (w1, w2), prob in paths.items():
            if w1 == needed or w2 == needed:
                terminal[(w1, w2)] = terminal.get((w1, w2), 0.0) + prob
                continue
            log_k = l0 + beta_momentum * (w1 - w2)
            p_win = float(expit(log_k))
            next_paths[(w1 + 1, w2)] = next_paths.get((w1 + 1, w2), 0.0) + prob * p_win
            next_paths[(w1, w2 + 1)] = next_paths.get((w1, w2 + 1), 0.0) + prob * (1.0 - p_win)
        paths = next_paths

    s_a = sum(prob for (w1, w2), prob in terminal.items() if w1 == needed)
    s_b = sum(prob for (w1, w2), prob in terminal.items() if w2 == needed)

    pmf: dict[tuple[int, int], float] = {}
    for (w1, w2), prob in terminal.items():
        if w1 == needed:
            val = p * (prob / s_a) if s_a > 0.0 else 0.0
        else:
            val = (1.0 - p) * (prob / s_b) if s_b > 0.0 else 0.0
        pmf[(w1, w2)] = float(val)

    # Clean renormalization to ensure sum is 1.0 to within 1e-15
    total = sum(pmf.values())
    if total > 0.0 and abs(total - 1.0) > 1e-12:
        pmf = {k: v / total for k, v in pmf.items()}

    return pmf


def compute_losing_map_pmf(
    p_series: float,
    best_of: int,
    beta_momentum: float = DEFAULT_MOMENTUM_BETA,
) -> list[float]:
    """Compute losing-team map count PMF conditional on the series winner.

    Returns a list of probabilities [P(loser wins 0 maps), P(loser wins 1 map), ...].
    For BO3: length 2 ([2-0, 2-1]).
    For BO5: length 3 ([3-0, 3-1, 3-2]).
    """
    best_of = _best_of(best_of)
    p = _probability(p_series)
    if not math.isfinite(beta_momentum):
        raise ValueError("beta_momentum must be finite")
    if best_of == 1:
        return [1.0]

    pmf = compute_scoreline_pmf(p_series, best_of, beta_momentum=beta_momentum)
    needed = best_of // 2 + 1
    if p == 1.0:
        return [1.0] + [0.0] * (needed - 1)
    if p == 0.0:
        # As the rare winner's map probability tends to zero, the opponent
        # takes every permitted map before the winner completes the series.
        return [0.0] * (needed - 1) + [1.0]

    # P(loser gets k maps | winner wins) = P(needed, k) / p
    losing_probs = []
    for k in range(needed):
        val = pmf.get((needed, k), 0.0) / p if p > 0.0 else 0.0
        losing_probs.append(float(val))

    total = sum(losing_probs)
    if total > 0.0:
        losing_probs = [v / total for v in losing_probs]
    return losing_probs


def compute_series_length_pmf(
    p_series: float,
    best_of: int,
    beta_momentum: float = DEFAULT_MOMENTUM_BETA,
) -> dict[int, float]:
    """Compute PMF over the total number of games played in the series.

    For BO3: keys in {2, 3}.
    For BO5: keys in {3, 4, 5}.
    """
    best_of = _best_of(best_of)
    _probability(p_series)
    if best_of == 1:
        return {1: 1.0}

    pmf = compute_scoreline_pmf(p_series, best_of, beta_momentum=beta_momentum)
    lengths: dict[int, float] = {}
    for (w1, w2), prob in pmf.items():
        total_games = w1 + w2
        lengths[total_games] = lengths.get(total_games, 0.0) + prob
    return lengths


def build_score_distributions(
    p_series: float = 0.5,
    best_ofs: Sequence[int] = (3, 5),
    beta_momentum: float = DEFAULT_MOMENTUM_BETA,
) -> dict[int, list[float]]:
    """Build pooled score_distributions mapping for simulate_tournament."""
    return {
        bo: compute_losing_map_pmf(p_series, bo, beta_momentum=beta_momentum)
        for bo in best_ofs
        if bo in (3, 5)
    }


def build_conditional_score_distributions(
    bins: Sequence[float] = (0.0, 0.5, 0.65, 0.8, 1.0),
    best_ofs: Sequence[int] = (3, 5),
    beta_momentum: float = DEFAULT_MOMENTUM_BETA,
) -> dict[str, Any]:
    """Build conditional_score_distributions structure for simulate_tournament.

    Parameters
    ----------
    bins : Sequence[float]
        Monotonically increasing bin thresholds in [0, 1].
    best_ofs : Sequence[int]
        Best-of formats to include (3, 5).
    beta_momentum : float
        Momentum beta parameter.
    """
    bin_list = list(bins)
    if len(bin_list) < 2 or bin_list[0] != 0.0 or bin_list[-1] != 1.0:
        raise ValueError("bins must start at 0.0 and end at 1.0 with at least 2 points")

    distributions: dict[int, list[list[float]]] = {}
    for bo in best_ofs:
        if bo not in (3, 5):
            continue
        bo_dists = []
        for i in range(len(bin_list) - 1):
            p_mid = 0.5 * (bin_list[i] + bin_list[i + 1])
            bo_dists.append(compute_losing_map_pmf(p_mid, bo, beta_momentum=beta_momentum))
        distributions[bo] = bo_dists

    return {
        "probability_bins": bin_list,
        "distributions": distributions,
    }


@dataclass(frozen=True, slots=True)
class ConditionalScorelineModel:
    """Fixed Markov-momentum heuristic control; not a native or learned model."""

    beta_momentum: float = DEFAULT_MOMENTUM_BETA

    def scoreline_pmf(self, p_series: float, best_of: int) -> dict[tuple[int, int], float]:
        return compute_scoreline_pmf(p_series, best_of, self.beta_momentum)

    def losing_map_pmf(self, p_series: float, best_of: int) -> list[float]:
        return compute_losing_map_pmf(p_series, best_of, self.beta_momentum)

    def series_length_pmf(self, p_series: float, best_of: int) -> dict[int, float]:
        return compute_series_length_pmf(p_series, best_of, self.beta_momentum)

    def score_distributions(
        self, p_series: float = 0.5, best_ofs: Sequence[int] = (3, 5)
    ) -> dict[int, list[float]]:
        return build_score_distributions(p_series, best_ofs, self.beta_momentum)

    def conditional_distributions(
        self,
        bins: Sequence[float] = (0.0, 0.5, 0.65, 0.8, 1.0),
        best_ofs: Sequence[int] = (3, 5),
    ) -> dict[str, Any]:
        return build_conditional_score_distributions(bins, best_ofs, self.beta_momentum)


class LearnedConditionalScorelineModel:
    """TRAIN-only winner-confidence multinomial laws for stopped BO3/BO5 scores.

    Losing-map count is the category. Its logits are intercept + slope times
    logit(probability of the sampled winner), with the last category fixed at
    zero. The summed TRAIN negative log likelihood penalizes slopes by L2=1;
    intercepts are unpenalized. This extension never changes the series winner
    marginal and cannot establish improvement in tournament-winner forecasts.
    """

    SCHEMA = "learned_conditional_scoreline_v1"
    MODEL_KIND = "train_only_conditional_multinomial_logit"
    L2_SLOPES = 1.0
    TRAIN_START = "2024-01-01"
    TRAIN_END = "2025-01-01"

    def __init__(
        self,
        coefficients: Mapping[int, Mapping[str, Sequence[float]]],
        pooled_distributions: Mapping[int, Sequence[float]],
        provenance: Mapping[str, Any],
    ) -> None:
        self.coefficients = {
            int(bo): {name: list(values) for name, values in law.items()}
            for bo, law in coefficients.items()
        }
        self.pooled_distributions = {
            int(bo): list(pmf) for bo, pmf in pooled_distributions.items()
        }
        self.provenance = json.loads(json.dumps(provenance, allow_nan=False))

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        probability_column: str = "p_c0",
        cutoff: Any = "2025-01-01",
    ) -> LearnedConditionalScorelineModel:
        required = {probability_column, "y", "best_of", "wins_a", "wins_b", "date", "result_day"}
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
            raise ValueError(f"TRAIN frame requires columns {sorted(required)}")
        cutoff_ts = _utc_timestamp(cutoff)
        start, end = _utc_timestamp(cls.TRAIN_START), _utc_timestamp(cls.TRAIN_END)
        dates = frame["date"].map(_utc_timestamp)
        releases = frame["result_day"].map(_utc_timestamp)
        if dates.isna().any() or releases.isna().any():
            raise ValueError("TRAIN chronology cannot contain missing dates")
        selected = (dates >= start) & (dates < end) & (dates < cutoff_ts) & (releases < min(cutoff_ts, end))
        train = frame.loc[selected].copy()
        if train.empty:
            raise ValueError("No released TRAIN2024 scoreline observations before cutoff")
        train["_date"] = dates.loc[selected]
        train["_release"] = releases.loc[selected]
        records = []
        for row in train.to_dict("records"):
            numeric = {}
            for field in ("best_of", "wins_a", "wins_b", "y"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or int(value) != value:
                    raise ValueError(f"{field} must be an integer")
                numeric[field] = int(value)
            bo = _best_of(numeric["best_of"])
            y, a, b = numeric["y"], numeric["wins_a"], numeric["wins_b"]
            target = bo // 2 + 1
            if y not in (0, 1) or not (
                (a == target and 0 <= b < target and y == 1) or
                (b == target and 0 <= a < target and y == 0)
            ):
                raise ValueError(f"Illegal stopped BO{bo} score or inconsistent winner: {(a, b, y)}")
            p = _probability(row[probability_column])
            if row["_release"] < row["_date"]:
                raise ValueError("result_day cannot precede match date")
            p_winner = p if y else 1.0 - p
            if bo != 1 and p_winner in (0.0, 1.0):
                # A finite-logit training design is required by this optimizer.
                # Do not silently clip, exclude, or manufacture boundary data.
                raise ValueError("BO3/BO5 TRAIN winner probability has infinite logit; finite-logit fit unavailable")
            record = {
                "best_of": bo, "losing_maps": min(a, b), "p_winner": p_winner,
                "p": p, "y": y, "wins_a": a, "wins_b": b,
                "date": row["_date"].isoformat(), "result_day": row["_release"].isoformat(),
            }
            if "match_id" in row:
                record["match_id"] = str(row["match_id"])
            records.append(record)
        records.sort(key=lambda row: json.dumps(row, sort_keys=True))
        coefficients, optimizer_results = {}, {}
        pooled = {1: [1.0]}
        counts_by_format = {}
        for bo in (1, 3, 5):
            rows = [row for row in records if row["best_of"] == bo]
            counts_by_format[str(bo)] = len(rows)
            if not rows or bo == 1:
                continue
            categories = bo // 2 + 1
            target = np.asarray([row["losing_maps"] for row in rows], dtype=int)
            counts = np.bincount(target, minlength=categories)
            pooled[bo] = (counts / len(rows)).tolist()
            if np.any(counts == 0):
                raise ValueError(f"BO{bo} TRAIN lacks losing-map categories; finite unpenalized-intercept fit unavailable")
            x = np.asarray([logit(row["p_winner"]) for row in rows], dtype=float)
            free = categories - 1
            initial = np.zeros((2, free))
            initial[0] = np.log(counts[:-1] / counts[-1])
            row_indices = np.arange(len(rows))

            def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
                intercepts, slopes = parameters.reshape(2, free)
                logits = np.zeros((len(rows), categories))
                logits[:, :-1] = intercepts + x[:, None] * slopes
                probabilities = softmax(logits, axis=1)
                loss = np.sum(logsumexp(logits, axis=1) - logits[row_indices, target])
                loss += 0.5 * cls.L2_SLOPES * np.dot(slopes, slopes)
                probabilities[row_indices, target] -= 1.0
                residual = probabilities[:, :-1]
                gradient = np.stack((
                    residual.sum(axis=0),
                    (residual * x[:, None]).sum(axis=0) + cls.L2_SLOPES * slopes,
                ))
                return float(loss), gradient.ravel()

            result = minimize(
                objective, initial.ravel(), method="L-BFGS-B", jac=True,
                options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
            )
            loss, gradient = objective(result.x)
            gradient_max = float(np.max(np.abs(gradient)))
            if (not result.success or not np.isfinite(result.x).all() or
                    not math.isfinite(loss) or not np.isfinite(gradient).all() or gradient_max > 1e-4):
                raise RuntimeError(
                    f"BO{bo} scoreline optimizer did not converge: {result.message}; gradient={gradient_max}"
                )
            intercepts, slopes = result.x.reshape(2, free)
            coefficients[bo] = {
                "intercepts": intercepts.tolist() + [0.0],
                "slopes": slopes.tolist() + [0.0],
            }
            optimizer_results[str(bo)] = {
                "method": "L-BFGS-B", "converged": True,
                "message": str(result.message), "iterations": int(result.nit),
                "objective": loss, "gradient_max_abs": gradient_max,
                "gradient_acceptance_threshold": 1e-4,
                "reference_losing_maps": categories - 1,
            }
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
        provenance = {
            "train_start": start.isoformat(), "train_end_exclusive": end.isoformat(),
            "cutoff": cutoff_ts.isoformat(),
            "selection": "TRAIN2024 date < cutoff and result_day < cutoff (strict UTC)",
            "probability_column": probability_column,
            "feature": "logit(probability_of_observed_winner)",
            "rows_input": len(frame), "rows_train": len(records),
            "rows_excluded": len(frame) - len(records), "rows_by_best_of": counts_by_format,
            "min_date": min(row["date"] for row in records),
            "min_result_day": min(row["result_day"] for row in records),
            "max_date": max(row["date"] for row in records),
            "max_result_day": max(row["result_day"] for row in records),
            "train_rows_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "optimizer": optimizer_results,
            "l2_slopes": cls.L2_SLOPES, "l2_intercepts": 0.0,
            "purpose": "Exact-score/series-length diagnostic only; does not change winner probabilities",
        }
        return cls(coefficients, pooled, provenance)

    def losing_map_pmf(self, p_winner: float, best_of: int) -> list[float]:
        p, bo = _probability(p_winner), _best_of(best_of)
        if bo == 1:
            return [1.0]
        if bo not in self.coefficients:
            raise ValueError(f"Learned BO{bo} law unavailable: no fitted TRAIN support")
        law = self.coefficients[bo]
        intercepts = np.asarray(law["intercepts"], dtype=float)
        slopes = np.asarray(law["slopes"], dtype=float)
        if p in (0.0, 1.0):
            # At +/-infinity only extremal slopes survive. Equal slopes retain
            # relative intercept mass; there is no epsilon input/output floor.
            extremum = slopes.max() if p == 1.0 else slopes.min()
            active = slopes == extremum
            probabilities = np.zeros(len(slopes))
            probabilities[active] = softmax(intercepts[active])
        else:
            probabilities = softmax(intercepts + slopes * logit(p))
        return probabilities.tolist()

    def scoreline_pmf(self, p: float, best_of: int) -> dict[tuple[int, int], float]:
        p, bo = _probability(p), _best_of(best_of)
        target = bo // 2 + 1
        winner_a = self.losing_map_pmf(p, bo)
        winner_b = self.losing_map_pmf(1.0 - p, bo)
        return {
            **{(target, lost): p * winner_a[lost] for lost in range(target)},
            **{(lost, target): (1.0 - p) * winner_b[lost] for lost in range(target)},
        }

    def series_length_pmf(self, p: float, best_of: int) -> dict[int, float]:
        pmf = self.scoreline_pmf(p, best_of)
        target = int(best_of) // 2 + 1
        return {
            target + lost: pmf[(target, lost)] + pmf[(lost, target)]
            for lost in range(target)
        }

    @staticmethod
    def _probability_bins(bins: Sequence[float] | None) -> list[float]:
        values = [i / 100 for i in range(101)] if bins is None else list(bins)
        values = [_probability(value) for value in values]
        if len(values) < 2 or values[0] != 0.0 or values[-1] != 1.0 or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("Probability bins must strictly partition [0, 1]")
        return values

    def conditional_distributions(
        self, best_ofs: Sequence[int] = (3, 5), bins: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Midpoint-bin approximation for the engine; scoring stays continuous."""
        values = self._probability_bins(bins)
        formats = [_best_of(bo) for bo in best_ofs]
        if len(set(formats)) != len(formats):
            raise ValueError("Duplicate best_of conditional distributions")
        return {
            "probability_bins": values,
            "distributions": {
                bo: [self.losing_map_pmf((lo + hi) / 2, bo) for lo, hi in zip(values, values[1:])]
                for bo in formats
            },
        }

    @property
    def engine_approximation(self) -> dict[str, Any]:
        return {
            "method": "piecewise_constant_at_probability_bin_midpoint",
            "probability_variable": "sampled_winner_series_probability",
            "default_bin_width": 0.01, "default_bin_count": 100,
            "engine_intervals": "left-inclusive; last right-inclusive",
            "scoring": "exact_continuous_law_not_binned",
            "boundary_note": "Engine bins approximate endpoints; continuous law uses exact limiting softmax",
        }

    def to_dict(self) -> dict[str, Any]:
        """Finite JSON-ready fitted state, including TRAIN and optimizer evidence."""
        return json.loads(json.dumps({
            "schema": self.SCHEMA, "model_kind": self.MODEL_KIND,
            "coefficients": {str(bo): law for bo, law in self.coefficients.items()},
            "pooled_distributions": {str(bo): pmf for bo, pmf in self.pooled_distributions.items()},
            "provenance": self.provenance, "engine_approximation": self.engine_approximation,
        }, allow_nan=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearnedConditionalScorelineModel:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA or payload.get("model_kind") != cls.MODEL_KIND:
            raise ValueError("Unsupported learned scoreline state")
        try:
            state = json.loads(json.dumps(dict(payload), allow_nan=False))
            coefficients, pooled = {}, {}
            for key, law in state["coefficients"].items():
                if key not in ("3", "5") or set(law) != {"intercepts", "slopes"}:
                    raise ValueError("Invalid learned scoreline coefficient format")
                bo, target = int(key), int(key) // 2 + 1
                for values in law.values():
                    if not isinstance(values, list) or len(values) != target or any(
                        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                        for value in values
                    ) or values[-1] != 0:
                        raise ValueError("Coefficients must be finite with an identifiable zero reference category")
                coefficients[bo] = law
            for key, values in state["pooled_distributions"].items():
                if key not in ("1", "3", "5") or not isinstance(values, list):
                    raise ValueError("Invalid pooled scoreline format")
                bo = int(key)
                pmf = [_probability(value) for value in values]
                if len(pmf) != bo // 2 + 1 or not math.isclose(sum(pmf), 1.0, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("Invalid pooled losing-map distribution")
                pooled[bo] = pmf
            if set(pooled) != {1, *coefficients} or pooled[1] != [1.0]:
                raise ValueError("Pooled and learned support must agree")
            provenance = state["provenance"]
            if not isinstance(provenance, dict) or provenance.get("l2_slopes") != cls.L2_SLOPES:
                raise ValueError("Invalid fixed-L2 fitted provenance")
            if provenance.get("l2_intercepts") != 0.0:
                raise ValueError("Invalid intercept penalty")
            cutoff = _utc_timestamp(provenance["cutoff"])
            boundary = min(cutoff, _utc_timestamp(cls.TRAIN_END))
            if _utc_timestamp(provenance["max_result_day"]) >= boundary or _utc_timestamp(provenance["max_date"]) >= boundary:
                raise ValueError("Fitted scoreline chronology crosses TRAIN cutoff")
            if _utc_timestamp(provenance["min_date"]) < _utc_timestamp(cls.TRAIN_START):
                raise ValueError("Fitted scoreline chronology precedes TRAIN")
            for bo in coefficients:
                optimizer = provenance["optimizer"][str(bo)]
                if (optimizer.get("converged") is not True or
                        not isinstance(optimizer.get("gradient_max_abs"), Real) or
                        not 0 <= optimizer["gradient_max_abs"] <= 1e-4):
                    raise ValueError("Fitted scoreline optimizer did not converge")
            return cls(coefficients, pooled, provenance)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Malformed learned scoreline state") from exc
