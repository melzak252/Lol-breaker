"""Locked historical DEVELOPMENT assessment, never a deployment decision.

The unchanged C0 evaluator supplies recipe ledgers and own-policy metrics.  This
module adds genuinely matched H-price contrasts and diagnostics on the original
S selections; changing a probability on a fixed price/result cannot change cash.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import xlogy

from src.analysis.ev_approach_evaluation import evaluate_c0_market_heads


_RECIPES = ("S", "S-cal", "M", "M-cal", "M-shape", "Fixed50", "H", "M-price", "H-price")
_PRIMARY = ("M-price", "M-shape")
_SECONDARY = ("S", "Fixed50", "M-cal", "H")
_NET_MULTIPLIER = .88


class _NativeAdmittedHead:
    """Apply the same evaluator-level native admission rule to the new hybrid."""

    def __init__(self, head):
        self.head = head

    def predict_details(self, frame):
        if not frame["native_admitted"].fillna(False).all():
            raise ValueError("native_not_admitted")
        return self.head.predict_details(frame)


def _probability(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and 0 <= value <= 1)


def _ratio_summary(numerators, denominators, months, resamples, confidence=.95):
    """Full-universe paired block ratio; one undefined draw invalidates its CI."""
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    total = float(den.sum())
    contributing = int(np.count_nonzero(den > 0))
    sampled_den = den[resamples].sum(axis=1)
    sampled_num = num[resamples].sum(axis=1)
    defined = sampled_den > 0
    undefined = int(np.count_nonzero(~defined))
    alpha = (1 - confidence) / 2
    interval = None
    if contributing >= 2 and undefined == 0:
        interval = np.quantile(sampled_num / sampled_den, [alpha, 1-alpha]).tolist()
    equal_defined = (den[resamples] > 0).all(axis=1) if len(months) else np.zeros(len(resamples), bool)
    equal_undefined = int(np.count_nonzero(~equal_defined))
    equal_point = float(np.mean(num / den)) if len(months) and np.all(den > 0) else None
    equal_interval = None
    if contributing >= 2 and equal_point is not None and equal_undefined == 0:
        equal_interval = np.quantile(
            (num[resamples] / den[resamples]).mean(axis=1), [alpha, 1-alpha]
        ).tolist()
    return {
        "point": float(num.sum() / total) if total else None,
        "interval": interval,
        "equal_month_point": equal_point,
        "equal_month_interval": equal_interval,
        "undefined_draws": undefined,
        "equal_month_undefined_draws": equal_undefined,
        "contributing_blocks": contributing,
        "count": int(total),
        "confidence_level": confidence,
        "status": "ESTIMABLE" if interval is not None else "NOT_ESTIMABLE",
        "equal_month_status": "ESTIMABLE" if equal_interval is not None else "NOT_ESTIMABLE",
        "monthly": {month: {"numerator": float(n), "denominator": int(d),
                             "point": float(n/d) if d else None}
                    for month, n, d in zip(months, num, den)},
    }


def _value_summary(values, months, resamples, confidence=.95):
    numerators = np.zeros(len(months))
    denominators = np.zeros(len(months), dtype=int)
    positions = {month: index for index, month in enumerate(months)}
    for month, value in values:
        if value is not None:
            index = positions[month]
            numerators[index] += value
            denominators[index] += 1
    return _ratio_summary(numerators, denominators, months, resamples, confidence)


def _paired_contrast(frame, ledgers, control, months, resamples, confidence):
    candidate = ledgers["H-price"]
    baseline = ledgers[control]
    observations = []
    for _, row in frame.iterrows():
        row_id = row["row_id"]
        p = candidate[row_id]["probability"]
        q = baseline[row_id]["probability"]
        if (not row["primary_decision"] or not row["native_admitted"]
                or not _probability(row["p_C0"]) or not _probability(row["q"])
                or not _probability(row["y"]) or row["y"] not in (0, 1)
                or not _probability(p) or not _probability(q)):
            continue
        y = float(row["y"])
        # Head probabilities already obey the bound epsilon scoring policy.
        ll_delta = float(-(xlogy(y, p) + xlogy(1-y, 1-p))
                         + xlogy(y, q) + xlogy(1-y, 1-q))
        observations.append((row_id, row["month"], ll_delta, float((p-y)**2 - (q-y)**2)))
    ll = _value_summary([(month, loss) for _, month, loss, _ in observations],
                        months, resamples, confidence)
    brier = _value_summary([(month, loss) for _, month, _, loss in observations],
                           months, resamples, confidence)
    return {
        "candidate": "H-price", "control": control,
        "delta_LL": ll["point"], "LL_interval": ll["interval"],
        "delta_Brier": brier["point"], "Brier_interval": brier["interval"],
        "equal_month_delta_LL": ll["equal_month_point"],
        "equal_month_LL_interval": ll["equal_month_interval"],
        "equal_month_delta_Brier": brier["equal_month_point"],
        "equal_month_Brier_interval": brier["equal_month_interval"],
        "common_count": len(observations),
        "row_ids": [row_id for row_id, _, _, _ in observations],
        "contributing_blocks": ll["contributing_blocks"],
        "undefined_draws": ll["undefined_draws"],
        "equal_month_undefined_draws": ll["equal_month_undefined_draws"],
        "confidence_level": confidence, "status": ll["status"],
        "LL": ll, "Brier": brier,
    }


def _fixed_selection(fixed, frame_by_id, ledger, months, resamples):
    values = {name: [] for name in (
        "overstatement", "model_market_component", "market_residual", "paired_error_change")}
    rows = []
    available = 0
    for original in fixed:
        row_id = original["row_id"]
        row = frame_by_id.loc[row_id]
        prediction = ledger[row_id]
        p = prediction["probability"]
        q = row["q"]
        is_a = original["side"] == "A"
        y = float(row["y"]) if is_a else 1-float(row["y"])
        odds = original["odds"]
        error = gap = residual = change = None
        absence = prediction["absence"]
        if _probability(p):
            available += 1
            p = float(p) if is_a else 1-float(p)
            error = _NET_MULTIPLIER * odds * (p-y)
            change = error - original["overstatement"]
            if _probability(q):
                q = float(q) if is_a else 1-float(q)
                gap = _NET_MULTIPLIER * odds * (p-q)
                residual = _NET_MULTIPLIER * odds * (q-y)
        else:
            absence = absence or "probability_unavailable"
        row_values = {"overstatement": error, "model_market_component": gap,
                      "market_residual": residual, "paired_error_change": change}
        for name, value in row_values.items():
            values[name].append((original["month"], value))
        rows.append({"row_id": row_id, "month": original["month"],
                     "side": original["side"], "odds": odds,
                     "quote_id": original["quote_id"], "bookmaker": original["bookmaker"],
                     "S_predicted_ev": original["predicted_ev"],
                     "fixed_profit": original["profit"],
                     "probability_selected_side": float(p) if _probability(p) else None,
                     "absence": absence, **row_values})
    return {
        "selection_count": len(fixed), "count": available,
        "absence_count": len(fixed)-available,
        "fixed_realized_return": float(np.mean([row["profit"] for row in fixed])) if fixed else None,
        "fixed_cash_profit": float(sum(row["profit"] for row in fixed)),
        "rows": rows,
        **{name: _value_summary(entries, months, resamples) for name, entries in values.items()},
    }


def _cash_increment(frame, ledgers, months, resamples):
    positions = {month: index for index, month in enumerate(months)}
    num = np.zeros(len(months))
    den = np.zeros(len(months), dtype=int)
    for _, row in frame.iterrows():
        if not row["financial_eligible"]:
            continue
        index = positions[row["month"]]
        row_id = row["row_id"]
        # Both policies retain absent predictions as abstentions, not exclusions.
        num[index] += ledgers["H-price"][row_id]["profit"] - ledgers["M-price"][row_id]["profit"]
        den[index] += 1
    return {"candidate": "H-price", "control": "M-price",
            "estimand": "eligible_opportunity_weighted",
            **_ratio_summary(num, den, months, resamples)}


def evaluate_development(panel, heads, draws=5000, seed=20261006):
    """Evaluate all nine sealed heads without fitting or selecting parameters."""
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, (bool, np.bool_)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(heads, Mapping) or set(heads) != set(_RECIPES):
        raise ValueError("heads must contain exactly the nine locked development recipes")
    required = {"row_id", "month", "p_C0", "q", "y", "native_admitted",
                "financial_eligible", "odds_a", "odds_b", "quotes"}
    if not required.issubset(panel.columns):
        raise ValueError("panel lacks required development columns: " + ", ".join(sorted(required-set(panel.columns))))
    if panel["row_id"].isna().any() or panel["row_id"].duplicated().any():
        raise ValueError("row_id must be unique and nonmissing")
    frame = panel.reset_index(drop=True).copy()
    if "primary_decision" not in frame:
        frame["primary_decision"] = True
    for name in ("primary_decision", "native_admitted", "financial_eligible"):
        if not frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(name + " must contain explicit booleans")
    frame["financial_eligible"] &= frame["primary_decision"]
    months = sorted(set(frame.attrs["monthly_universe"])) if "monthly_universe" in frame.attrs else sorted(set(frame["month"]))
    if any(not isinstance(month, str) or len(month) != 7 or month[4] != "-"
           or not month[:4].isdigit() or not month[5:].isdigit()
           or not 1 <= int(month[5:]) <= 12 for month in months):
        raise ValueError("monthly universe must use YYYY-MM calendar blocks")
    if not set(frame["month"]).issubset(months):
        raise ValueError("every panel row must belong to the declared monthly universe")
    frame.attrs["monthly_universe"] = months
    evaluated_heads = {recipe: heads[recipe] for recipe in _RECIPES}
    evaluated_heads["H-price"] = _NativeAdmittedHead(heads["H-price"])
    result = evaluate_c0_market_heads(frame, evaluated_heads, draws=int(draws), seed=int(seed))
    # Discard the forward evaluator's H-vs-M-cal decision and contrasts entirely.
    result.pop("contrasts")
    result.pop("profit_increment_interval")
    resamples = np.random.default_rng(seed).integers(0, len(months), (draws, len(months))) if months else np.empty((draws, 0), int)
    # The frozen helper marks every draw undefined when fewer than two blocks
    # contribute. Keep its null CI, but report the actual empty-draw count and
    # the full settled-opportunity denominator for this historical study.
    for metric in result["metrics"].values():
        blocks = [metric["monthly"][month] for month in months]
        for name, numerator, denominator in (
                ("ROI", "profit", "bets"),
                ("profit_per_opportunity", "profit", "eligible"),
                ("overstatement", "error", "bets"),
                ("near_overstatement", "near_error", "near_bets")):
            metric[name] = _ratio_summary([block[numerator] for block in blocks],
                [block[denominator] for block in blocks], months, resamples)
    ledgers = {recipe: {} for recipe in _RECIPES}
    for row in result["rows"]:
        ledgers[row["recipe"]][row["row_id"]] = row
    primary = {control: _paired_contrast(frame, ledgers, control, months, resamples, .975)
               for control in _PRIMARY}
    secondary = {control: _paired_contrast(frame, ledgers, control, months, resamples, .95)
                 for control in _SECONDARY}
    original = [row for row in ledgers["S"].values() if row["selected"]]
    near = [row for row in original if row["near_band"]]
    frame_by_id = frame.set_index("row_id")
    fixed = {}
    for recipe in _RECIPES:
        fixed[recipe] = _fixed_selection(original, frame_by_id, ledgers[recipe], months, resamples)
        fixed[recipe]["near_band"] = _fixed_selection(near, frame_by_id, ledgers[recipe], months, resamples)
    increment = _cash_increment(frame, ledgers, months, resamples)
    candidate = result["metrics"]["H-price"]

    def wholly_bounded(metric):
        interval = metric["interval"]
        return interval is not None and interval[0] >= -.02 and interval[1] <= .02

    gates = {}
    for control, contrast in primary.items():
        interval = contrast["LL_interval"]
        gates["LL_vs_"+control] = interval is not None and interval[1] < 0
        gates["Brier_vs_"+control] = contrast["delta_Brier"] is not None and contrast["delta_Brier"] <= 0
        gates["coverage_vs_"+control] = candidate["coverage_count"] >= result["metrics"][control]["coverage_count"]
    gates["selected_overstatement_within_2pp"] = wholly_bounded(candidate["overstatement"])
    gates["near_band_overstatement_within_2pp"] = wholly_bounded(candidate["near_overstatement"])
    profit_interval = candidate["profit_per_opportunity"]["interval"]
    gates["positive_profit_per_opportunity"] = profit_interval is not None and profit_interval[0] > 0
    gates["positive_increment_vs_M_price"] = increment["interval"] is not None and increment["interval"][0] > 0
    gates["selected_ROI_estimable"] = candidate["ROI"]["interval"] is not None
    passed = all(gates.values())
    result.update(
        schema="c0_historical_development_assessment_v1",
        primary_candidate="H-price", primary_contrasts=primary,
        secondary_contrasts=secondary, fixed_S_selection=fixed,
        own_selections={recipe: [row for row in ledgers[recipe].values() if row["selected"]]
                        for recipe in _RECIPES},
        profit_increment_vs_M_price=increment,
        decision="DEVELOPMENT_CANDIDATE_RETAINED" if passed else "NOT_ADOPTED",
        gates=gates, reasons=[name for name, value in gates.items() if not value],
        evidence_class="exposed_historical_development",
        limits=[
            "All 2026 outcomes were exposed; this is not a protected holdout or authentic forward G0 evidence.",
            "Assessment has three calendar blocks including partial September; monthly uncertainty is sparse.",
            "Fixed S sets and their near-band are diagnostics, not future selection policies; fixed cash is identical across heads.",
            "A retained development candidate is not deployment, promotion, or certification.",
        ],
        bootstrap={"draws": int(draws), "seed": int(seed), "months": months,
                   "primary_confidence_level": .975, "secondary_and_cash_confidence_level": .95,
                   "primary_comparators": 2, "adjustment": "Bonferroni",
                   "paired_block_basis": "full_declared_preselection_month_universe",
                   "undefined_policy": "any_undefined_draw_nulls_interval_no_finite_subset"},
    )
    result["estimands"].update(
        primary_contrasts="H-price_minus_control_match_weighted_exact_matched_native_market_population",
        fixed_S_selection="original_S_side_price_result_bet_weighted_probability_error",
        fixed_S_error_change="model_minus_S_on_exact_available_fixed_selection_pairs",
        profit_increment="H-price_minus_M-price_eligible_opportunity_weighted",
        equal_month="diagnostic_full_declared_universe_any_undefined_constituent_nulls_point",
    )
    return result
