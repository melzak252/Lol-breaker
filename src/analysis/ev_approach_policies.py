"""Frozen EV policy selection and event-ordered portfolio replay.

This module is a research replay only; it never reads or writes a wallet. Policy
probabilities are side-oriented inputs, while opportunity ``p`` is A-oriented.
Money is computed with :class:`Decimal`, rounded to cents with ROUND_HALF_UP at
stake placement and per-bet PnL settlement.

P0 and admitted P5 use the mean without a conservative bound. P1--P4 scores
are explicit research haircuts, not guaranteed probability lower bounds.
Selected rows and replay ledgers retain ``prob_for_ev``, ``prob_conservative``,
and the uncertainty basis/reason. A reducing caller risk bound is measured
only when the caller declares ``uncertainty_basis="measured_bounds"``; an
unlabelled bound remains a research assumption. A research-policy score stays
an assumption even when combined with a declared measured risk bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
import re
from typing import Any

import pandas as pd

from betting_app.services.bet_qualification_service import (
    is_bet_eligible,
    qualification_tier,
)

CENT = Decimal("0.01")
_POLICY_KAPPAS = {0.5, 0.75, 1.0, 1.25}
_POLICY_LINEAR_KAPPAS = {1.0, 1.5, 2.0}
_APPLICATION_CAPS = {
    "max_tier1_market_gap": 0.08,
    "max_ev_longshot": 0.20,
    "longshot_odds_threshold": 3.50,
    "max_bo1_market_gap": 0.12,
    "max_coinflip_market_gap": 0.08,
    "coinflip_min_odds": 1.80,
    "coinflip_max_odds": 2.50,
}
_SELECTED_COLUMNS = [
    "bet_id", "row_id", "side", "p", "p_for_ev", "odds", "bookmaker",
    "quote_id", "quote_at", "decision_at", "settle_at", "won", "ev",
    "unit_return", "close_odds", "close_available_at", "close_same_book",
    "league", "best_of", "tax_rate", "p_mean_a", "p_mean_b", "policy",
    "qualification", "qualification_reason", "competition_tier", "p5_threshold",
    "p5_threshold_n", "p5_threshold_source", "p5_threshold_source_ids",
    "p5_cal_variant", "p5_cal_model_year", "risk_p_low", "risk_allow",
    "source_probability_column", "source_probability_a", "source_input_p",
    "prob_for_ev", "prob_conservative", "uncertainty_basis", "uncertainty_reason",
]
_LEDGER_COLUMNS = [
    "placed", "skip_reason", "status", "stake", "pnl", "free_capital_before",
    "free_capital_after", "reserved_stake_after", "equity_after",
    "settled_at", "matured",
]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except (TypeError, ValueError):
        return False


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _optional_number(value: Any, name: str, *, minimum: float | None = None,
                     maximum: float | None = None) -> float | None:
    if _missing(value):
        return None
    return _number(value, name, minimum=minimum, maximum=maximum)


def _policy_spec(policy: str) -> tuple[str, float | None]:
    if not isinstance(policy, str):
        raise ValueError("policy must be a string")
    policy_id = policy.strip().upper()
    if policy_id in {"P0", "P1", "P4", "P5"}:
        return policy_id, None
    if policy_id == "P2":
        return "P2", 0.75
    if policy_id == "P3":
        return "P3", 1.5
    match = re.fullmatch(r"P([23])_(0\.5|0\.75|1\.0|1\.25|1\.5|2\.0)", policy_id)
    if match is None:
        raise ValueError(f"unknown EV policy: {policy}")
    family, raw_kappa = match.groups()
    kappa = float(raw_kappa)
    allowed = _POLICY_KAPPAS if family == "2" else _POLICY_LINEAR_KAPPAS
    if kappa not in allowed:
        raise ValueError(f"unsupported kappa for P{family}: {raw_kappa}")
    return f"P{family}", kappa


def _logit_lower(p: float, scale: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return p
    z = math.log(p) - math.log1p(-p) - scale
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def policy_probability(
    p: float,
    policy: str,
    side: str = "A",
    sigma_z: float | None = None,
    p_std: float | None = None,
    p_q10: float | None = None,
    p_q90: float | None = None,
    threshold: float | None = None,
) -> float | None:
    """Return the policy's EV score for an already-oriented side.

    For P4, ``p_q10``/``p_q90`` are always A-oriented. The caller supplies
    ``p`` as the chosen side's coherent mean; B therefore uses ``1 - p_q90``.
    Missing inputs required by a policy return ``None`` rather than a fallback.
    P0/P5 return the admitted mean; P1--P4 apply research assumptions rather
    than certified probability lower bounds.
    """
    mean = _number(p, "p", minimum=0.0, maximum=1.0)
    side_id = side.strip().upper() if isinstance(side, str) else ""
    if side_id not in {"A", "B"}:
        raise ValueError("side must be 'A' or 'B'")
    family, kappa = _policy_spec(policy)
    if family == "P0":
        return mean
    if family == "P1":
        return _logit_lower(mean, 0.75 * 0.15)
    if family == "P2":
        sigma = _optional_number(sigma_z, "sigma_z", minimum=0.0)
        return None if sigma is None else _logit_lower(mean, float(kappa) * sigma)
    if family == "P3":
        spread = _optional_number(p_std, "p_std", minimum=0.0)
        return None if spread is None else max(0.0, mean - float(kappa) * spread)
    if family == "P4":
        q10 = _optional_number(p_q10, "p_q10", minimum=0.0, maximum=1.0)
        q90 = _optional_number(p_q90, "p_q90", minimum=0.0, maximum=1.0)
        if q10 is None or q90 is None:
            return None
        side_quantile = q10 if side_id == "A" else 1.0 - q90
        return min(mean, side_quantile)
    sigma = _optional_number(sigma_z, "sigma_z", minimum=0.0)
    cal_threshold = _optional_number(threshold, "threshold", minimum=0.0)
    if sigma is None or cal_threshold is None:
        return None
    return None if sigma > cal_threshold else mean


def _model_year_key(value: Any) -> str:
    year = _number(value, "model_year", minimum=1.0)
    if not year.is_integer():
        raise ValueError("model_year must be an integer")
    return str(int(year))


def _cal_threshold_table(
    cal_thresholds: Any,
) -> dict[tuple[str, str], tuple[float, int | None, list[Any]]]:
    """Normalize caller-supplied prior-CAL q75 values or sigma_z references."""
    if cal_thresholds is None:
        return {}
    if isinstance(cal_thresholds, pd.DataFrame):
        required = {"variant", "model_year"}
        if not required.issubset(cal_thresholds.columns):
            raise ValueError("cal_thresholds requires variant and model_year columns")
        if "threshold" in cal_thresholds.columns:
            value_col = "threshold"
        elif "sigma_z_q75" in cal_thresholds.columns:
            value_col = "sigma_z_q75"
        elif "cal_sigma_z_q75" in cal_thresholds.columns:
            value_col = "cal_sigma_z_q75"
        elif "sigma_z" in cal_thresholds.columns:
            value_col = "sigma_z"
        else:
            raise ValueError("cal_thresholds requires threshold, sigma_z_q75, or CAL sigma_z")
        rows: dict[tuple[str, str], tuple[float, int | None, list[Any]]] = {}
        for (variant, year), group in cal_thresholds.groupby(["variant", "model_year"], dropna=False):
            if _missing(variant) or _missing(year):
                continue
            valid_values = [
                _number(value, "CAL sigma_z", minimum=0.0)
                for value in group[value_col]
                if not _missing(value)
            ]
            if not valid_values:
                continue
            if value_col == "sigma_z":
                threshold = float(pd.Series(valid_values).quantile(0.75))
                count = len(valid_values)
            else:
                unique_thresholds = set(valid_values)
                if len(unique_thresholds) != 1:
                    raise ValueError("one prior-CAL threshold is required per variant/model_year")
                threshold = valid_values[0]
                count = None
            id_column = "row_id" if "row_id" in group.columns else (
                "golgg_match_id" if "golgg_match_id" in group.columns else None
            )
            if "cal_row_ids" in group.columns:
                source_ids = []
                for raw_ids in group["cal_row_ids"]:
                    if isinstance(raw_ids, (list, tuple, set)):
                        source_ids.extend(raw_ids)
                    elif not _missing(raw_ids):
                        source_ids.append(raw_ids)
                source_ids = list(dict.fromkeys(source_ids))
            else:
                source_ids = [] if id_column is None else group[id_column].dropna().drop_duplicates().tolist()
            if "cal_rows" in group.columns:
                counts = {
                    int(_number(value, "CAL row count", minimum=0.0))
                    for value in group["cal_rows"] if not _missing(value)
                }
                if len(counts) > 1:
                    raise ValueError("one prior-CAL row count is required per variant/model_year")
                if counts:
                    count = counts.pop()
            rows[(str(variant), _model_year_key(year))] = (threshold, count, source_ids)
        return rows
    if not isinstance(cal_thresholds, Mapping):
        raise ValueError("cal_thresholds must be a mapping or prior-CAL DataFrame")
    result: dict[tuple[str, str], tuple[float, int | None, list[Any]]] = {}
    for key, value in cal_thresholds.items():
        if isinstance(key, tuple) and len(key) == 2:
            variant, year = key
            result[(str(variant), _model_year_key(year))] = (
                _threshold_value(value), _threshold_count(value), _threshold_ids(value),
            )
        elif isinstance(value, Mapping):
            variant = key
            for year, nested_value in value.items():
                result[(str(variant), _model_year_key(year))] = (
                    _threshold_value(nested_value), _threshold_count(nested_value),
                    _threshold_ids(nested_value),
                )
    return result


def _threshold_value(value: Any) -> float:
    if isinstance(value, Mapping):
        for key in ("threshold", "sigma_z_q75", "cal_sigma_z_q75"):
            if key in value:
                value = value[key]
                break
    return _number(value, "CAL sigma_z q75", minimum=0.0)


def _threshold_count(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("cal_rows", "count", "n"):
        if key in value and not _missing(value[key]):
            return int(_number(value[key], "CAL row count", minimum=0.0))
    return None


def _threshold_ids(value: Any) -> list[Any]:
    if not isinstance(value, Mapping):
        return []
    for key in ("cal_row_ids", "calibration_ids", "source_ids", "row_ids", "cal_ids"):
        if key in value:
            ids = value[key]
            return list(ids) if isinstance(ids, (list, tuple, set)) else [ids]
    return []


def _row_threshold(
    row: Mapping[str, Any],
    table: Mapping[tuple[str, str], tuple[float, int | None, list[Any]]],
) -> tuple[float | None, int | None, list[Any]]:
    variant = row.get("variant")
    year = row.get("model_year")
    if _missing(variant) or _missing(year):
        return None, None, []
    try:
        return table.get((str(variant), _model_year_key(year)), (None, None, []))
    except TypeError:
        return None, None, []


def _effective_settle_time(row: Mapping[str, Any]) -> tuple[Any, str | None, Any]:
    for column in ("settle_at", "label_available_at"):
        value = row.get(column)
        if not _missing(value):
            return value, "actual", None
    for column in ("scenario_label_available_at", "scenario_result_available_at"):
        value = row.get(column)
        assumption = row.get("scenario_timestamp_assumption")
        if not _missing(value) and not _missing(assumption) and str(assumption).strip():
            return value, "scenario", assumption
    return pd.NaT, None, None


def _scenario_assumption(row: Mapping[str, Any], fallback: Any = None) -> Any:
    assumption = row.get("scenario_timestamp_assumption")
    if not _missing(assumption) and str(assumption).strip():
        return assumption
    return fallback


def _same_book_close(row: Mapping[str, Any], suffix: str, bookmaker: Any) -> tuple[float | None, Any, bool]:
    close_odds = _optional_number(row.get(f"close_odds_{suffix}"), f"close_odds_{suffix}", minimum=1.0)
    if close_odds is None:
        close_odds = _optional_number(row.get("close_odds"), "close_odds", minimum=1.0)
    close_book = row.get(f"close_bookmaker_{suffix}")
    if _missing(close_book):
        close_book = row.get("close_bookmaker")
    explicit_same = row.get(f"close_same_book_{suffix}", row.get("close_same_book"))
    if not _missing(explicit_same):
        same_book = _risk_allowed(explicit_same)
    else:
        same_book = (
            not _missing(close_book) and not _missing(bookmaker)
            and str(close_book) == str(bookmaker)
        )
    close_time = row.get("close_available_at")
    if close_odds is not None and close_odds <= 1.0:
        raise ValueError("close odds must be > 1")
    if _missing(close_time):
        close_time = row.get(f"close_available_at_{suffix}")
    if not same_book or _missing(close_time):
        return None, close_time, False
    return close_odds, close_time, close_odds is not None


def _market_probability(q: Any, side: str) -> float | None:
    if _missing(q):
        return None
    market = _number(q, "q", minimum=0.0, maximum=1.0)
    if market in (0.0, 1.0):
        raise ValueError("q must be strictly between 0 and 1")
    return market if side == "A" else 1.0 - market


def _risk_allowed(value: Any) -> bool:
    if _missing(value):
        return True
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    if not isinstance(value, (str, bytes)) and value in (0, 1):
        return bool(value)
    raise ValueError("risk allow_a/allow_b values must be boolean or missing")


def _side_probability(row: Mapping[str, Any], probability_col: str) -> tuple[float, float]:
    p_a = _number(row.get(probability_col), probability_col, minimum=0.0, maximum=1.0)
    return p_a, 1.0 - p_a


def _candidate_for_side(
    row: Mapping[str, Any], side: str, p_mean: float, policy: str,
    tax_rate: float, min_ev: float, qualification: str,
    cal_threshold: float | None,
) -> dict[str, Any]:
    suffix = side.lower()
    sigma = row.get("sigma_z")
    spread = row.get("p_std")
    kwargs: dict[str, Any] = {
        "sigma_z": sigma,
        "p_std": spread,
        "p_q10": row.get("p_q10"),
        "p_q90": row.get("p_q90"),
        "threshold": cal_threshold,
    }
    policy_p = policy_probability(p_mean, policy, side=side, **kwargs)
    family, _ = _policy_spec(policy)
    if policy_p is None:
        basis = "missing_bounds"
        uncertainty_reason = (
            "P5_sigma_abstain"
            if family == "P5" and not _missing(sigma) and cal_threshold is not None
            else "missing_policy_input"
        )
    elif family in {"P0", "P5"}:
        basis = "mean"
        uncertainty_reason = (
            "P0_mean_policy" if family == "P0" else "P5_sigma_admitted_mean"
        )
    else:
        basis = "assumed_research_haircut"
        uncertainty_reason = {
            "P1": "P1_fixed_sigma_0.15_research_haircut",
            "P2": "P2_configured_sigma_research_haircut",
            "P3": "P3_configured_spread_research_haircut",
            "P4": "P4_quantile_research_score",
        }[family]
    allow = _risk_allowed(row.get(f"allow_{suffix}"))
    bound_value = row.get(f"p_low_{suffix}")
    bound = _optional_number(bound_value, f"p_low_{suffix}", minimum=0.0, maximum=1.0)
    bound_error = False
    if bound is not None and bound > p_mean:
        bound_error = True
    if bound is not None and policy_p is not None:
        if bound < p_mean:
            declared_basis = row.get("uncertainty_basis")
            bound_basis = (
                "measured_bounds"
                if not _missing(declared_basis) and declared_basis == "measured_bounds"
                else "assumed_research_haircut"
            )
            declared_reason = row.get("uncertainty_reason")
            bound_reason = (
                str(declared_reason) if not _missing(declared_reason) and str(declared_reason).strip()
                else "caller_declared_risk_bound" if bound_basis == "measured_bounds"
                else "provided_research_risk_bound_unverified"
            )
            if basis == "mean":
                basis, uncertainty_reason = bound_basis, bound_reason
            else:
                uncertainty_reason = f"{uncertainty_reason}; risk_bound:{bound_reason}"
        policy_p = min(policy_p, bound)
    conservative = None if basis in {"mean", "missing_bounds"} else policy_p
    odds = _number(row.get(f"odds_{suffix}"), f"odds_{suffix}", minimum=1.0)
    if odds <= 1.0:
        raise ValueError(f"odds_{suffix} must be > 1")
    bookmaker = row.get(f"bookmaker_{suffix}")
    quote_id = row.get(f"quote_id_{suffix}")
    quote_at = row.get(f"quote_at_{suffix}")
    ev = None if policy_p is None else policy_p * odds * (1.0 - tax_rate) - 1.0
    eligible = False
    reason = "missing_policy_input" if policy_p is None else "insufficient_ev"
    tier = qualification_tier(row.get("league"), row.get("date"))
    if not allow:
        reason = "risk_abstain"
    elif bound_error:
        reason = "invalid_risk_bound"
    elif policy_p is not None:
        if qualification == "raw":
            eligible = ev > min_ev
            reason = "eligible" if eligible else "insufficient_ev"
        else:
            if not 0.0 < p_mean < 1.0:
                reason = "invalid_probability"
            else:
                q_side = _market_probability(row.get("q"), side)
                eligible, reason, _ = is_bet_eligible(
                    prob_model=p_mean,
                    prob_conservative=conservative,
                    odds=odds,
                    prob_market_novig=q_side,
                    tax_rate=tax_rate,
                    min_ev_net=min_ev,
                    dynamic_hurdle=True,
                    competition_tier=tier,
                    max_tier1_market_gap=_APPLICATION_CAPS["max_tier1_market_gap"],
                    max_ev_net=None,
                    best_of=None if _missing(row.get("best_of")) else int(row["best_of"]),
                    max_ev_longshot=_APPLICATION_CAPS["max_ev_longshot"],
                    longshot_odds_threshold=_APPLICATION_CAPS["longshot_odds_threshold"],
                    max_bo1_market_gap=_APPLICATION_CAPS["max_bo1_market_gap"],
                    bo1_max_odds=None,
                    prob_market_close_novig=None,
                    max_negative_clv_drift=None,
                    max_coinflip_market_gap=_APPLICATION_CAPS["max_coinflip_market_gap"],
                    coinflip_min_odds=_APPLICATION_CAPS["coinflip_min_odds"],
                    coinflip_max_odds=_APPLICATION_CAPS["coinflip_max_odds"],
                    max_rating_disagreement=None,
                    uncertainty_required=False,
                    uncertainty_basis=basis,
                    uncertainty_reason=uncertainty_reason,
                )
    close_odds, close_time, close_same_book = _same_book_close(row, suffix, bookmaker)
    y = row.get("y")
    if _missing(y):
        won = None
    else:
        y_int = _number(y, "y", minimum=0.0, maximum=1.0)
        if y_int not in (0.0, 1.0):
            raise ValueError("y must be binary")
        won = bool(y_int == (1.0 if side == "A" else 0.0))
    settle_at, settle_source, settle_assumption = _effective_settle_time(row)
    unit_return = None if won is None else (odds * (1.0 - tax_rate) - 1.0 if won else -1.0)
    return {
        "side": side,
        "p": p_mean,
        "p_for_ev": policy_p,
        "prob_for_ev": policy_p,
        "prob_conservative": conservative,
        "uncertainty_basis": basis,
        "uncertainty_reason": uncertainty_reason,
        "odds": odds,
        "bookmaker": bookmaker,
        "quote_id": quote_id,
        "quote_at": quote_at,
        "decision_at": row.get("decision_at"),
        "settle_at": settle_at,
        "settlement_time_source": settle_source,
        "scenario_timestamp_assumption": _scenario_assumption(row, settle_assumption),
        "won": won,
        "ev": ev,
        "unit_return": unit_return,
        "close_odds": close_odds,
        "close_available_at": close_time,
        "close_same_book": close_same_book,
        "league": row.get("league"),
        "best_of": row.get("best_of"),
        "tax_rate": tax_rate,
        "p_mean_a": p_mean if side == "A" else 1.0 - p_mean,
        "p_mean_b": 1.0 - p_mean if side == "A" else p_mean,
        "policy": policy,
        "qualification": qualification,
        "qualification_reason": reason,
        "competition_tier": tier,
        "eligible": eligible,
        "risk_p_low": bound,
        "risk_allow": allow,
        "p5_threshold": cal_threshold,
        "p5_threshold_n": None,
    }


def select_policy_bets(
    frame: pd.DataFrame,
    probability_col: str = "p",
    policy: str = "P0",
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    qualification: str = "raw",
    cal_thresholds: Any = None,
) -> pd.DataFrame:
    """Select at most one highest-net-EV side per opportunity.

    ``cal_thresholds`` must be supplied from prior CAL only: either a mapping of
    ``(variant, model_year)`` to a q75 threshold, or a DataFrame with those
    identifiers and either a precomputed threshold column or CAL ``sigma_z``
    references. The TEST opportunity frame is never used to estimate P5 q75.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    tax = _number(tax_rate, "tax_rate", minimum=0.0)
    if tax >= 1.0:
        raise ValueError("tax_rate must be < 1")
    hurdle = _number(min_ev, "min_ev", minimum=0.0)
    if qualification not in {"raw", "application"}:
        raise ValueError("qualification must be 'raw' or 'application'")
    family, _ = _policy_spec(policy)
    thresholds = _cal_threshold_table(cal_thresholds) if family == "P5" else {}
    required = {probability_col, "odds_a", "odds_b"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity frame missing required columns: {missing}")
    if family in {"P2", "P3", "P4"}:
        needed = {"sigma_z"} if family == "P2" else ({"p_std"} if family == "P3" else {"p_q10", "p_q90"})
        missing = sorted(needed.difference(frame.columns))
        if missing:
            raise ValueError(f"{policy} requires columns: {missing}")
    results: list[dict[str, Any]] = []
    for _, series in frame.iterrows():
        row = series.to_dict()
        p_a, p_b = _side_probability(row, probability_col)
        threshold, threshold_n, threshold_ids = (
            _row_threshold(row, thresholds) if family == "P5" else (None, None, [])
        )
        candidates = [
            _candidate_for_side(row, "A", p_a, policy, tax, hurdle, qualification, threshold),
            _candidate_for_side(row, "B", p_b, policy, tax, hurdle, qualification, threshold),
        ]
        for candidate in candidates:
            candidate["p5_threshold_source_ids"] = threshold_ids
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            continue
        # Only an exact full-precision EV tie falls back to side A.
        chosen = max(eligible, key=lambda candidate: (
            candidate["ev"], candidate["side"] == "A",
        ))
        settle_at, settle_source, settle_assumption = _effective_settle_time(row)
        bet_id = row.get("bet_id")
        if _missing(bet_id) or not str(bet_id).strip():
            base_id = row.get("row_id", row.get("golgg_match_id"))
            if _missing(base_id):
                base_id = len(results)
            bet_id = f"{base_id}:{chosen['side']}"
        record = dict(row)
        record.update(chosen)
        record.update({
            "bet_id": bet_id,
            "row_id": row.get("row_id"),
            "p_mean_a": p_a,
            "p_mean_b": p_b,
            "source_probability_column": probability_col,
            "source_probability_a": p_a,
            "source_input_p": row.get("p") if probability_col != "p" else None,
            "scenario_timestamp_assumption": _scenario_assumption(row, settle_assumption),
            "policy": policy,
            "qualification": qualification,
            "p5_threshold": threshold,
            "p5_threshold_n": threshold_n,
            "p5_threshold_source": "caller_prior_CAL" if threshold is not None else None,
            "p5_threshold_source_ids": threshold_ids,
            "p5_cal_variant": row.get("variant") if threshold is not None else None,
            "p5_cal_model_year": row.get("model_year") if threshold is not None else None,
        })
        results.append(record)
    columns = list(dict.fromkeys([*frame.columns, *_SELECTED_COLUMNS,
                                  "settlement_time_source", "scenario_timestamp_assumption",
                                  "eligible", "p5_threshold_n"]))
    return pd.DataFrame(results, columns=columns)


def _decimal(value: Any, name: str, *, minimum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool) or _missing(value):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if (
        not number.is_finite()
        or not math.isfinite(float(number))
        or (minimum is not None and number < minimum)
    ):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def _money(value: Decimal) -> Decimal:
    try:
        result = value.quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError("money value cannot be represented as finite cents") from exc
    if not math.isfinite(float(result)):
        raise ValueError("money value must remain finite")
    return result

def _finite_float(value: Decimal, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must remain finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must remain finite")
    return result


def _utc_times(values: pd.Series, column: str) -> pd.Series:
    times = pd.to_datetime(values, utc=True, errors="coerce")
    invalid = values.notna() & times.isna()
    if bool(invalid.any()):
        raise ValueError(f"{column} contains invalid timestamps")
    return times


def _won_value(value: Any) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    number = _number(value, "won", minimum=0.0, maximum=1.0)
    if number not in (0.0, 1.0):
        raise ValueError("won must be boolean or binary")
    return bool(number)


def _stake_for_bet(
    strategy: str, free_capital: Decimal, row: Mapping[str, Any], flat_stake: Decimal,
    tax: Decimal, kelly_fraction: Decimal, max_fraction: Decimal,
) -> tuple[Decimal, str | None]:
    if strategy == "flat":
        stake = _money(flat_stake)
        if stake <= 0:
            return Decimal("0.00"), "nonpositive_stake"
        if stake > free_capital:
            return Decimal("0.00"), "insufficient_cash"
        return stake, None
    if strategy == "provided_fraction":
        action_fraction = _decimal(row.get("action_fraction"), "action_fraction", minimum=Decimal("0"))
        if action_fraction > max_fraction:
            raise ValueError("action_fraction must not exceed max_fraction")
        if action_fraction == 0:
            return Decimal("0.00"), "zero_action_fraction"
        stake = _money(free_capital * action_fraction)
        if stake <= 0:
            return Decimal("0.00"), "stake_below_cent"
        return stake, None
    p_value = row.get("p_for_ev", row.get("p"))
    if _missing(p_value):
        return Decimal("0.00"), "missing_probability"
    p = _decimal(p_value, "p_for_ev", minimum=Decimal("0"))
    if p > 1:
        raise ValueError("p_for_ev must be in [0, 1]")
    odds = _decimal(row.get("odds"), "odds", minimum=Decimal("1"))
    if odds <= 1:
        raise ValueError("odds must be > 1")
    net_odds = odds * (Decimal("1") - tax)
    net_profit = net_odds - Decimal("1")
    if net_profit <= 0:
        return Decimal("0.00"), "nonpositive_kelly"
    full_kelly = (p * net_odds - Decimal("1")) / net_profit
    if full_kelly <= 0:
        return Decimal("0.00"), "nonpositive_kelly"
    fraction = min(max_fraction, kelly_fraction * full_kelly)
    stake = _money(free_capital * fraction)
    if stake <= 0:
        return Decimal("0.00"), "stake_below_cent"
    return stake, None


def simulate_portfolio(
    bets: pd.DataFrame,
    strategy: str = "flat",
    initial_bankroll: float = 10000.0,
    flat_stake: float = 100.0,
    tax_rate: float = 0.12,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.025,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay decisions and released outcomes without future-PnL funding.

    Actual ``decision_at`` remains UTC or NaT in the ledger. An unknown actual
    time may be ordered using ``scenario_decision_at`` only when an explicit
    ``scenario_timestamp_assumption`` is supplied; that effective time and its
    source are recorded separately.
    """
    if not isinstance(bets, pd.DataFrame):
        raise TypeError("bets must be a pandas DataFrame")
    if strategy not in {"flat", "quarter_kelly", "provided_fraction"}:
        raise ValueError("strategy must be 'flat', 'quarter_kelly', or 'provided_fraction'")
    required = {"bet_id", "odds"}
    if strategy == "provided_fraction":
        required.update({"action_side", "action_fraction"})
    missing = sorted(required.difference(bets.columns))
    if missing:
        raise ValueError(f"bets missing required columns: {missing}")
    if not {"decision_at", "scenario_decision_at"}.intersection(bets.columns):
        raise ValueError("bets require decision_at or scenario_decision_at")
    if not bets.empty and bets["bet_id"].isna().any():
        raise ValueError("bet_id values must be unique and non-null")
    if bets["bet_id"].duplicated().any():
        raise ValueError("bet_id values must be unique; duplicate bet_id found")

    bankroll = _money(_decimal(initial_bankroll, "initial_bankroll", minimum=Decimal("0")))
    flat = _money(_decimal(flat_stake, "flat_stake", minimum=Decimal("0")))
    tax = _decimal(tax_rate, "tax_rate", minimum=Decimal("0"))
    if tax >= 1:
        raise ValueError("tax_rate must be < 1")
    kelly = _decimal(kelly_fraction, "kelly_fraction", minimum=Decimal("0"))
    cap = _decimal(max_fraction, "max_fraction", minimum=Decimal("0"))
    if kelly > 1 or cap > 1:
        raise ValueError("kelly_fraction and max_fraction must be <= 1")

    assumptions = bets.get(
        "scenario_timestamp_assumption", pd.Series("", index=bets.index, dtype=object)
    )
    has_assumption = assumptions.notna() & assumptions.astype(str).str.strip().ne("")
    actual_decisions = (
        bets["decision_at"].copy()
        if "decision_at" in bets.columns
        else pd.Series(pd.NaT, index=bets.index)
    )
    effective_decisions = actual_decisions.copy()
    if "scenario_decision_at" in bets.columns:
        effective_decisions = effective_decisions.where(
            effective_decisions.notna(),
            bets["scenario_decision_at"].where(has_assumption),
        )
    actual_decision_times = _utc_times(actual_decisions, "decision_at")
    decision_times = _utc_times(effective_decisions, "effective_decision_at")
    if bool(decision_times.isna().any()):
        raise ValueError("decision_at or explicit scenario_decision_at is required for every bet")

    actual_settle_column = (
        "settle_at" if "settle_at" in bets.columns
        else "label_available_at" if "label_available_at" in bets.columns
        else None
    )
    actual_settle_values = (
        bets[actual_settle_column].copy()
        if actual_settle_column is not None
        else pd.Series(pd.NaT, index=bets.index)
    )
    if actual_settle_column == "settle_at" and "label_available_at" in bets.columns:
        actual_settle_values = actual_settle_values.where(
            actual_settle_values.notna(), bets["label_available_at"]
        )
    effective_settle_values = actual_settle_values.copy()
    if "scenario_label_available_at" in bets.columns:
        effective_settle_values = effective_settle_values.where(
            effective_settle_values.notna(),
            bets["scenario_label_available_at"].where(has_assumption),
        )
    actual_settle_times = _utc_times(actual_settle_values, "actual_settle_at")
    settle_times = _utc_times(effective_settle_values, "settle_at")

    records: list[dict[str, Any]] = []
    for ordinal, (_, source) in enumerate(bets.iterrows()):
        row = source.to_dict()
        decision = decision_times.iloc[ordinal]
        settle = settle_times.iloc[ordinal]
        if not pd.isna(settle) and settle <= decision:
            raise ValueError("settle_at must be later than effective decision time")
        odds = _decimal(row.get("odds"), "odds", minimum=Decimal("1"))
        if odds <= 1 or not math.isfinite(float(odds)):
            raise ValueError("odds must be > 1 and finite")
        row["odds"] = odds
        row["decision_at"] = actual_decision_times.iloc[ordinal]
        row["decision_effective_at"] = decision
        row["decision_time_source"] = (
            "actual" if not pd.isna(actual_decision_times.iloc[ordinal]) else "scenario"
        )
        row["_event_decision_at"] = decision
        row["settle_at"] = settle
        if _missing(row.get("settlement_time_source")):
            if not pd.isna(actual_settle_times.iloc[ordinal]):
                row["settlement_time_source"] = "actual"
            elif not pd.isna(settle):
                row["settlement_time_source"] = "scenario"
            else:
                row["settlement_time_source"] = None
        if strategy == "provided_fraction":
            action_side = str(row.get("action_side", "")).strip().upper()
            if action_side not in {"A", "B"}:
                raise ValueError("action_side must be 'A' or 'B'")
            action_fraction = _decimal(
                row.get("action_fraction"), "action_fraction", minimum=Decimal("0")
            )
            if action_fraction > cap:
                raise ValueError("action_fraction must not exceed max_fraction")
        row["won"] = _won_value(row.get("won"))
        row["_ordinal"] = ordinal
        records.append(row)

    ordered = sorted(
        records, key=lambda row: (row["_event_decision_at"], row["_ordinal"])
    )
    free = bankroll
    reserved: dict[int, Decimal] = {}
    open_bets: dict[int, Decimal] = {}
    placed: dict[int, bool] = {}
    skip_reasons: dict[int, str | None] = {}
    stake_by_ordinal: dict[int, Decimal] = {}
    pnl_by_ordinal: dict[int, Decimal] = {}
    settled_at_by_ordinal: dict[int, Any] = {}
    free_before: dict[int, Decimal] = {}
    free_after: dict[int, Decimal] = {}
    reserved_after: dict[int, Decimal] = {}
    equity_after: dict[int, Decimal] = {}
    equity_peak = bankroll
    max_drawdown = Decimal("0.00")
    max_drawdown_pct = 0.0
    settlement_events = sorted(
        [
            row for row in records
            if not pd.isna(row["settle_at"]) and row["won"] is not None
        ],
        key=lambda row: (row["settle_at"], row["_ordinal"]),
    )
    settlement_cursor = 0
    def mark_event() -> Decimal:
        nonlocal equity_peak, max_drawdown, max_drawdown_pct
        equity = free + sum(open_bets.values(), Decimal("0.00"))
        if equity > equity_peak:
            equity_peak = equity
        drawdown = equity_peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if equity_peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, float(drawdown / equity_peak))
        return equity

    def release_until(cutoff: Any) -> None:
        nonlocal settlement_cursor, free
        while settlement_cursor < len(settlement_events):
            event_time = settlement_events[settlement_cursor]["settle_at"]
            if event_time > cutoff:
                break
            changed = False
            cash_delta = Decimal("0.00")
            while (
                settlement_cursor < len(settlement_events)
                and settlement_events[settlement_cursor]["settle_at"] == event_time
            ):
                event = settlement_events[settlement_cursor]
                settlement_cursor += 1
                ordinal = event["_ordinal"]
                stake = reserved.get(ordinal)
                if stake is None:
                    continue
                net_return = event["odds"] * (Decimal("1") - tax) - Decimal("1")
                pnl = _money(stake * net_return) if event["won"] else -stake
                pnl_by_ordinal[ordinal] = pnl
                settled_at_by_ordinal[ordinal] = event_time
                cash_delta += stake + pnl
                del reserved[ordinal]
                del open_bets[ordinal]
                changed = True
            if changed:
                free += cash_delta
                if free < 0:
                    raise ArithmeticError("settlement produced negative free capital")
                mark_event()

    for row in ordered:
        ordinal = row["_ordinal"]
        decision = row["_event_decision_at"]
        # Result releases at an equal instant are available before this decision.
        release_until(decision)
        free_before[ordinal] = free
        stake, skip_reason = _stake_for_bet(
            strategy, free, row, flat, tax, kelly, cap,
        )
        if stake > free:
            stake, skip_reason = Decimal("0.00"), "insufficient_cash"
        placed[ordinal] = skip_reason is None
        skip_reasons[ordinal] = skip_reason
        stake_by_ordinal[ordinal] = stake if skip_reason is None else Decimal("0.00")
        if skip_reason is None:
            free -= stake
            reserved[ordinal] = stake
            open_bets[ordinal] = stake
        free_after[ordinal] = free
        reserved_after[ordinal] = sum(open_bets.values(), Decimal("0.00"))
        equity_after[ordinal] = mark_event()
    release_until(pd.Timestamp.max.tz_localize("UTC"))

    ledger_records = []
    for row in ordered:
        ordinal = row["_ordinal"]
        record = {
            key: value for key, value in row.items()
            if key not in {"_ordinal", "_event_decision_at"}
        }
        record.update({
            "placed": placed[ordinal],
            "skip_reason": skip_reasons[ordinal],
            "status": (
                "skipped" if not placed[ordinal]
                else "settled" if ordinal in pnl_by_ordinal
                else "unsettled"
            ),
            "stake": _finite_float(stake_by_ordinal[ordinal], "stake"),
            "pnl": None if ordinal not in pnl_by_ordinal else _finite_float(pnl_by_ordinal[ordinal], "pnl"),
            "free_capital_before": _finite_float(free_before[ordinal], "free capital"),
            "free_capital_after": _finite_float(free_after[ordinal], "free capital"),
            "reserved_stake_after": _finite_float(reserved_after[ordinal], "reserved stake"),
            "equity_after": _finite_float(equity_after[ordinal], "equity"),
            "settled_at": settled_at_by_ordinal.get(ordinal, pd.NaT),
            "matured": ordinal in pnl_by_ordinal,
        })
        ledger_records.append(record)
    ledger_columns = list(dict.fromkeys([
        *bets.columns, *_LEDGER_COLUMNS, "decision_at", "settle_at",
        "decision_effective_at", "decision_time_source", "settlement_time_source",
    ]))
    ledger = pd.DataFrame(ledger_records, columns=ledger_columns)
    ledger["placed"] = pd.Series(
        [bool(record["placed"]) for record in ledger_records],
        index=ledger.index, dtype=object,
    )

    total_staked = sum(
        (stake for ordinal, stake in stake_by_ordinal.items() if placed[ordinal]),
        Decimal("0.00"),
    )
    matured_stake = sum(
        (stake_by_ordinal[ordinal] for ordinal in pnl_by_ordinal), Decimal("0.00")
    )
    total_pnl = sum(pnl_by_ordinal.values(), Decimal("0.00"))
    final_reserved = sum(open_bets.values(), Decimal("0.00"))
    final_capital = free + final_reserved
    roi = _finite_float(total_pnl / matured_stake, "ROI") if matured_stake else None
    settled_rows = [row for row in ordered if row["_ordinal"] in pnl_by_ordinal]
    probability_errors: list[float] = []
    ev_errors: list[float] = []
    if strategy != "provided_fraction":
        for row in settled_rows:
            ordinal = row["_ordinal"]
            actual_win = 1.0 if row["won"] else 0.0
            probability = row.get("p")
            if not _missing(probability):
                probability_errors.append(
                    actual_win - _number(probability, "p", minimum=0.0, maximum=1.0)
                )
            p_for_ev = row.get("p_for_ev", probability)
            if not _missing(p_for_ev):
                p_ev = _number(p_for_ev, "p_for_ev", minimum=0.0, maximum=1.0)
                expected_ev = p_ev * float(row["odds"]) * (1.0 - float(tax)) - 1.0
                if stake_by_ordinal[ordinal]:
                    realized_return = float(
                        pnl_by_ordinal[ordinal] / stake_by_ordinal[ordinal]
                    )
                    ev_errors.append(realized_return - expected_ev)

    clv_values: list[float] = []
    for row in ordered:
        ordinal = row["_ordinal"]
        if not placed[ordinal]:
            continue
        close_odds = row.get("close_odds")
        close_time = row.get("close_available_at")
        same_book_value = row.get("close_same_book")
        if _missing(same_book_value):
            close_book = row.get("close_bookmaker")
            same_book = (
                not _missing(close_book) and not _missing(row.get("bookmaker"))
                and str(close_book) == str(row.get("bookmaker"))
            )
        else:
            same_book = _risk_allowed(same_book_value)
        if _missing(close_odds) or _missing(close_time) or not same_book:
            continue
        close_time_utc = pd.to_datetime(close_time, utc=True, errors="coerce")
        if pd.isna(close_time_utc):
            continue
        decision_time_utc = pd.to_datetime(
            row.get("decision_effective_at", row.get("decision_at")),
            utc=True, errors="coerce",
        )
        if pd.isna(decision_time_utc) or close_time_utc < decision_time_utc:
            continue
        start_at = row.get("start_at")
        if not _missing(start_at):
            start_time_utc = pd.to_datetime(start_at, utc=True, errors="coerce")
            if pd.isna(start_time_utc) or close_time_utc >= start_time_utc:
                continue
        close = _decimal(close_odds, "close_odds", minimum=Decimal("1"))
        if close <= 1:
            raise ValueError("close_odds must be > 1")
        clv_values.append((float(row["odds"]) / float(close) - 1.0) * 100.0)

    placed_count = int(sum(placed.values()))
    settled_count = len(pnl_by_ordinal)
    clv_count = len(clv_values)
    summary = {
        "strategy": strategy,
        "selected_count": len(bets),
        "placed_count": placed_count,
        "skipped_count": len(bets) - placed_count,
        "settled_count": settled_count,
        "unsettled_count": placed_count - settled_count,
        "total_staked": _finite_float(total_staked, "total_staked"),
        "matured_stake": _finite_float(matured_stake, "matured_stake"),
        "total_pnl": _finite_float(total_pnl, "total_pnl"),
        "roi": roi,
        "settled_roi": roi,
        "final_capital": _finite_float(final_capital, "final_capital"),
        "final_free_capital": _finite_float(free, "final_free_capital"),
        "reserved_stake": _finite_float(final_reserved, "reserved_stake"),
        "max_drawdown": _finite_float(max_drawdown, "max_drawdown"),
        "max_drawdown_pct": max_drawdown_pct,
        "conditional_probability_error": (
            sum(probability_errors) / len(probability_errors) if probability_errors else None
        ),
        "conditional_ev_error": sum(ev_errors) / len(ev_errors) if ev_errors else None,
        "observed_clv_count": clv_count,
        "observed_clv_available_count": clv_count,
        "observed_clv_unavailable_count": placed_count - clv_count,
        "observed_clv_available": clv_count > 0,
        "observed_clv_coverage": clv_count / placed_count if placed_count else None,
        "mean_observed_clv_pct": sum(clv_values) / clv_count if clv_count else None,
        "money_rounding": "Decimal cents, ROUND_HALF_UP on stakes and each settled bet PnL",
    }
    return ledger, summary
