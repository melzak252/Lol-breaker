"""Forensic market and source diagnostics for the frozen repaired A0 audit."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import evaluate_binary_probabilities

DRAWS = 5000
SEED = 20260930
TAX = 0.12


def _truth(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _mean(series: pd.Series):
    values = _finite(series).dropna()
    return float(values.mean()) if len(values) else None


def _normal_name(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    return normalized or None


def _ratio_ci(values, mask, blocks, *, seed=SEED) -> dict:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    blocks = np.asarray(blocks, dtype=object)
    numerator = np.where(mask, values, 0.0)
    denominator = mask.astype(float)
    return block_ratio_interval(numerator, denominator, blocks, draws=DRAWS, seed=seed)


def _paired_difference_ci(values, left, right, blocks) -> dict:
    """Paired monthly block CI for two disjoint row subsets, preserving zero months."""
    value = np.asarray(values, dtype=float)
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    block = np.asarray(blocks, dtype=object)
    labels, codes = np.unique(block, return_inverse=True)
    n_blocks = len(labels)
    sums_l = np.bincount(codes, weights=np.where(left, value, 0.0), minlength=n_blocks)
    sums_r = np.bincount(codes, weights=np.where(right, value, 0.0), minlength=n_blocks)
    counts_l = np.bincount(codes, weights=left.astype(float), minlength=n_blocks)
    counts_r = np.bincount(codes, weights=right.astype(float), minlength=n_blocks)
    total_l, total_r = counts_l.sum(), counts_r.sum()
    point_l = sums_l.sum() / total_l if total_l else None
    point_r = sums_r.sum() / total_r if total_r else None
    point = point_l - point_r if point_l is not None and point_r is not None else None
    active = (counts_l > 0) & (counts_r > 0)
    equal = float(np.mean(sums_l[active] / counts_l[active] - sums_r[active] / counts_r[active])) if active.any() else None
    valid = 0
    low = high = None
    if n_blocks and total_l and total_r:
        rng = np.random.default_rng(SEED)
        sampled = rng.integers(0, n_blocks, size=(DRAWS, n_blocks))
        sl, sr = sums_l[sampled].sum(axis=1), sums_r[sampled].sum(axis=1)
        cl, cr = counts_l[sampled].sum(axis=1), counts_r[sampled].sum(axis=1)
        defined = (cl > 0) & (cr > 0)
        valid = int(defined.sum())
        if n_blocks >= 2 and valid == DRAWS:
            delta = sl[defined] / cl[defined] - sr[defined] / cr[defined]
            low, high = (float(x) for x in np.quantile(delta, [0.025, 0.975]))
    status = "ESTIMATED" if n_blocks >= 2 and valid == DRAWS else ("UNDEFINED_REPLICATES" if valid != DRAWS else "INSUFFICIENT_BLOCKS")
    if not total_l or not total_r:
        status = "EMPTY_SELECTION"
    return {
        "left_n": int(total_l), "right_n": int(total_r), "blocks": n_blocks,
        "active_blocks_both": int(active.sum()), "difference": point,
        "equal_block_difference_active_both": equal, "draws": DRAWS,
        "valid_replicates": valid, "empty_replicates": DRAWS - valid,
        "ci_lower": low, "ci_upper": high, "status": status,
    }


def _decomposition(frame: pd.DataFrame, tax: float) -> pd.DataFrame:
    result = frame.copy()
    result["return_error"] = (1.0 - tax) * result.odds * (result.p - result.y)
    result["sports_vs_market"] = (1.0 - tax) * result.odds * (result.p - result.q)
    result["market_vs_result"] = (1.0 - tax) * result.odds * (result.q - result.y)
    result["identity_residual"] = result.return_error - result.sports_vs_market - result.market_vs_result
    return result


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "winners": 0, "losers": 0}
    return {
        "n": int(len(frame)),
        "winners": int(frame.y.sum()),
        "losers": int(len(frame) - frame.y.sum()),
        "mean_odds": _mean(frame.odds),
        "mean_p": _mean(frame.p),
        "mean_q": _mean(frame.q),
        "mean_model_market_gap": _mean(frame.p - frame.q),
        "mean_return_error": _mean(frame.return_error),
        "mean_sports_vs_market": _mean(frame.sports_vs_market),
        "mean_market_vs_result": _mean(frame.market_vs_result),
        "sum_return_error": float(frame.return_error.sum()),
        "sum_sports_vs_market": float(frame.sports_vs_market.sum()),
        "sum_market_vs_result": float(frame.market_vs_result.sum()),
        "max_abs_identity_residual": float(frame.identity_residual.abs().max()),
    }


def _probability_scores(frame: pd.DataFrame, epsilon: float) -> dict:
    if frame.empty:
        return {"n": 0}
    y = _finite(frame.y)
    valid = y.notna()
    y_values = y.loc[valid].astype(int).to_numpy()
    output = {"n": int(valid.sum()), "epsilon": float(epsilon)}
    if len(np.unique(y_values)) < 2:
        return {**output, "status": "AUC_UNDEFINED_SINGLE_CLASS"}
    for column, name in (("p", "repaired_A0"), ("q", "no_vig_consensus")):
        output[name] = evaluate_binary_probabilities(
            y_values,
            _finite(frame.loc[valid, column]).to_numpy(dtype=float),
            epsilon=epsilon,
        )
    output["model_minus_market_logloss"] = output["repaired_A0"]["logloss"] - output["no_vig_consensus"]["logloss"]
    return output


def _block_component_evidence(frame: pd.DataFrame, mask: pd.Series, blocks: pd.Series, prefix: str) -> dict:
    output = {}
    for component in ("sports_vs_market", "market_vs_result", "return_error"):
        output[component] = _ratio_ci(frame[component].to_numpy(), mask.to_numpy(), blocks.to_numpy())
    output["scope"] = prefix
    return output


def _selection_rows(bets: pd.DataFrame, quality: str, policy: str, source: str | None = None) -> pd.DataFrame:
    keep = (
        bets.cohort.eq("oddspapi_common") & bets.horizon.eq("T-1h")
        & bets.quality.eq(quality) & bets.policy.eq(policy) & _truth(bets.placed)
    )
    if source is not None:
        keep &= bets.source.eq(source)
    return bets.loc[keep].copy()


def _side_frame(opportunities: pd.DataFrame, selected: pd.DataFrame, tax: float) -> pd.DataFrame:
    base = opportunities.copy()
    selected_key = selected[["fixture_id", "horizon", "side"]].drop_duplicates().copy()
    if len(selected_key) != len(selected):
        raise ValueError("Frozen selected ledger has duplicate fixture/horizon/side keys")
    parts = []
    for side, suffix in (("A", "a"), ("B", "b")):
        part = base.copy()
        part["side"] = side
        part["p"] = part.p_event if side == "A" else 1.0 - part.p_event
        part["q"] = part.q_consensus_a if side == "A" else 1.0 - part.q_consensus_a
        part["y"] = part.y_event if side == "A" else 1.0 - part.y_event
        part["odds"] = part[f"odds_{suffix}"]
        part["selected"] = False
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    selected_keys = set(map(tuple, selected_key[["fixture_id", "horizon", "side"]].astype(str).to_numpy()))
    keys = list(map(tuple, frame[["fixture_id", "horizon", "side"]].astype(str).to_numpy()))
    frame["selected"] = np.fromiter((key in selected_keys for key in keys), dtype=bool, count=len(keys))
    frame = _decomposition(frame, tax)
    frame["unselected"] = ~frame.selected
    frame["selected_winner"] = frame.selected & frame.y.eq(1)
    frame["selected_loser"] = frame.selected & frame.y.eq(0)
    return frame


def _row_json(row: pd.Series, fields: list[str]) -> dict:
    result = {}
    for key in fields:
        value = row.get(key)
        if pd.isna(value) if not isinstance(value, (list, dict, np.ndarray)) else False:
            result[key] = None
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, (pd.Timestamp,)):
            result[key] = value.isoformat()
        elif isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        else:
            result[key] = value
    return result


def _cut_table(frame: pd.DataFrame, column: str, bins, labels, *, selected_only=False) -> list[dict]:
    work = frame.loc[frame.selected].copy() if selected_only else frame.copy()
    work["bin"] = pd.cut(work[column], bins=bins, labels=labels, right=False, include_lowest=True)
    out = []
    for label in labels:
        group = work.loc[work.bin.eq(label)]
        item = {"bin": str(label), **_summary(group)}
        out.append(item)
    missing = work.loc[work.bin.isna()]
    out.append({"bin": "missing", **_summary(missing)})
    return out


def _book_dispersion(opps: pd.DataFrame) -> pd.DataFrame:
    books = opps.loc[opps.source.ne("best_price") & opps.audit_primary_eligible].copy()
    stats = books.groupby(["fixture_id", "horizon"], dropna=False).q_book_a.agg(
        book_count="count", dispersion_sd="std", dispersion_range=lambda s: s.max() - s.min()
    ).reset_index()
    return stats


def _source_count_table(bets: pd.DataFrame) -> pd.DataFrame:
    selected = bets.loc[
        bets.cohort.eq("oddspapi_common") & bets.horizon.eq("T-1h")
        & bets.policy.isin(["raw_ev_gt_5", "current_application_filters"])
        & bets.quality.isin(["as_supplied_diagnostic", "exclude_q25_conflicts"])
    ].copy()
    rows = []
    for (quality, policy, source), group in selected.groupby(["quality", "policy", "source"], dropna=False, sort=True):
        rows.append({
            "quality": quality, "policy": policy, "source": source,
            "saved_rows": int(len(group)), "placed_n": int(_truth(group.placed).sum()),
            "winners": int(group.loc[_truth(group.placed), "won"].sum()),
            "losers": int(_truth(group.placed).sum() - group.loc[_truth(group.placed), "won"].sum()),
        })
    return pd.DataFrame(rows)


def _market_source_integrity(frame: pd.DataFrame, selected: pd.DataFrame, opportunities: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    available = frame
    selected_rows = selected.copy()
    selected_rows["selected_bookmaker"] = np.where(
        selected_rows.side.eq("A"), selected_rows.bookmaker_a, selected_rows.bookmaker_b
    )
    individual = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.horizon.eq("T-1h")
        & opportunities.source.ne("best_price") & opportunities.audit_primary_eligible
    ].copy()
    individual["source_bookmaker"] = individual.bookmaker.astype(str)
    lookup = individual.set_index(["fixture_id", "horizon", "source_bookmaker"], drop=False)
    chosen_records = []
    for row in selected_rows.itertuples(index=False):
        key = (row.fixture_id, row.horizon, str(row.selected_bookmaker))
        if key not in lookup.index:
            chosen_records.append({"fixture_id": row.fixture_id, "side": row.side, "selected_bookmaker": row.selected_bookmaker, "individual_quote_found": False})
            continue
        quote = lookup.loc[key]
        if isinstance(quote, pd.DataFrame):
            quote = quote.iloc[0]
        side = row.side
        side_suffix = "a" if side == "A" else "b"
        best_price = _num(getattr(row, "odds"), None)
        individual_price = _num(quote[f"odds_{side_suffix}"], None)
        qbook_a = _num(quote.q_book_a, None)
        qbook = qbook_a if side == "A" else (1.0 - qbook_a if qbook_a is not None else None)
        qcons_a = _num(quote.q_consensus_a, None)
        qcons = qcons_a if side == "A" else (1.0 - qcons_a if qcons_a is not None else None)
        chosen_records.append({
            "fixture_id": row.fixture_id, "golgg_match_id": row.golgg_match_id,
            "side": side, "selected_bookmaker": row.selected_bookmaker,
            "best_price_odds": best_price, "book_quote_odds": individual_price,
            "price_matches_source_row": bool(best_price is not None and individual_price is not None and np.isclose(best_price, individual_price, atol=1e-12, rtol=0)),
            "q_chosen_book": qbook, "q_consensus": qcons,
            "q_chosen_minus_consensus": (qbook - qcons) if qbook is not None and qcons is not None else None,
            "individual_quote_found": True,
            "participant_binding_a": quote.proof_a_participant_binding_status,
            "participant_binding_b": quote.proof_b_participant_binding_status,
            "bookmaker_binding_verified": bool(quote.bookmaker_binding_verified),
        })
    chosen = pd.DataFrame(chosen_records)
    chosen_found = chosen.loc[chosen.individual_quote_found] if len(chosen) else chosen
    # Leave-one-book-out q is the mean of per-book no-vig probabilities after excluding the book pricing the selected side.
    q_by_event = {}
    for (fixture_id, horizon), group in individual.groupby(["fixture_id", "horizon"], sort=False):
        q_by_event[(fixture_id, horizon)] = dict(zip(group.source_bookmaker.astype(str), _finite(group.q_book_a)))
    loo_values = []
    for row in chosen.itertuples(index=False):
        values = q_by_event.get((row.fixture_id, "T-1h"), {})
        others = [_num(value, None) for name, value in values.items() if name != str(row.selected_bookmaker)]
        others = [value for value in others if value is not None]
        q_loo_a = float(np.mean(others)) if len(others) >= 2 else None
        q_loo = q_loo_a if row.side == "A" else (1.0 - q_loo_a if q_loo_a is not None else None)
        loo_values.append(q_loo)
    if len(chosen):
        chosen["q_leave_chosen_book_out"] = loo_values
        chosen["q_book_minus_leave_one_out"] = chosen.q_chosen_book - chosen.q_leave_chosen_book_out
    source_summary = {
        "selected_best_price_n": int(len(selected_rows)),
        "selected_underlying_book_label_n": int(selected_rows.selected_bookmaker.notna().sum()),
        "individual_book_quote_found_n": int(chosen.individual_quote_found.sum()),
        "best_price_side_price_matches_individual_source_n": int(chosen.price_matches_source_row.sum()),
        "bookmaker_binding_verified_true_n": int(_truth(available.bookmaker_binding_verified).sum()),
        "available_quote_rows_n": int(len(available)),
        "available_quote_contract_ok_n": int(_truth(available.quote_contract_ok).sum()),
        "participant_binding_status_a": available.proof_a_participant_binding_status.value_counts(dropna=False).to_dict(),
        "participant_binding_status_b": available.proof_b_participant_binding_status.value_counts(dropna=False).to_dict(),
        "loo_with_other_book_quotes_n": int(chosen.q_leave_chosen_book_out.notna().sum()) if len(chosen) else 0,
        "loo_minimum_remaining_books": 2,
        "mean_chosen_book_minus_leave_one_out": _mean(chosen.q_book_minus_leave_one_out) if len(chosen) else None,
        "mean_chosen_book_minus_published_consensus": _mean(chosen.q_chosen_minus_consensus) if len(chosen) else None,
    }
    return source_summary, chosen, individual


def _fixed_selection_bookmaker_comparison(selected: pd.DataFrame, opportunities: pd.DataFrame, tax: float) -> tuple[pd.DataFrame, list[dict]]:
    fixed = selected[["fixture_id", "horizon", "side", "p_model_bet", "q_bet", "won"]].rename(
        columns={"p_model_bet": "p_fixed", "q_bet": "q_consensus_fixed", "won": "y_fixed"}
    )
    books = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.horizon.eq("T-1h")
        & opportunities.source.ne("best_price") & opportunities.audit_primary_eligible
    ].copy()
    paired = books.merge(fixed, on=["fixture_id", "horizon"], how="inner", validate="many_to_one")
    paired["odds_fixed_side"] = np.where(paired.side.eq("A"), paired.odds_a, paired.odds_b)
    paired["q_book_fixed_side"] = np.where(paired.side.eq("A"), paired.q_book_a, 1.0-paired.q_book_a)
    paired["q_consensus_source_side"] = np.where(paired.side.eq("A"), paired.q_consensus_a, 1.0-paired.q_consensus_a)
    paired["y_fixed_side"] = np.where(paired.side.eq("A"), paired.y_fixed, 1.0-paired.y_fixed)
    if not np.allclose(paired.q_consensus_source_side, paired.q_consensus_fixed, atol=1e-12, rtol=0):
        raise ValueError("Selected-side consensus differs across the fixed-source opportunity rows")
    paired["sports_vs_book"] = (1-tax)*paired.odds_fixed_side*(paired.p_fixed-paired.q_book_fixed_side)
    paired["book_vs_result"] = (1-tax)*paired.odds_fixed_side*(paired.q_book_fixed_side-paired.y_fixed_side)
    paired["return_error"] = (1-tax)*paired.odds_fixed_side*(paired.p_fixed-paired.y_fixed_side)
    paired["book_vs_consensus"] = (1-tax)*paired.odds_fixed_side*(paired.q_book_fixed_side-paired.q_consensus_fixed)
    paired["consensus_vs_result"] = (1-tax)*paired.odds_fixed_side*(paired.q_consensus_fixed-paired.y_fixed_side)
    paired["decomposition_residual"] = paired.return_error-paired.sports_vs_book-paired.book_vs_result
    summaries = []
    for source, group in paired.groupby("source", sort=True):
        score_frame = group[["p_fixed", "q_book_fixed_side", "y_fixed_side"]].rename(
            columns={"p_fixed": "p", "q_book_fixed_side": "q", "y_fixed_side": "y"}
        )
        score_frame = score_frame.dropna()
        book_scores = _probability_scores(score_frame, 1e-12)
        summaries.append({
            "bookmaker_source_label": source, "fixed_selected_ids_matched_n": int(len(group)),
            "winners": int(group.y_fixed_side.sum()), "losers": int(len(group)-group.y_fixed_side.sum()),
            "mean_odds_on_same_selected_sides": _mean(group.odds_fixed_side),
            "mean_q_book_minus_consensus": _mean(group.q_book_fixed_side-group.q_consensus_fixed),
            "mean_model_minus_book_probability": _mean(group.p_fixed-group.q_book_fixed_side),
            "mean_sports_vs_book": _mean(group.sports_vs_book),
            "mean_book_vs_result": _mean(group.book_vs_result),
            "mean_return_error": _mean(group.return_error),
            "mean_quote_age_max_hours": _mean(group.quote_age_max_hours),
            "quote_freshness_true_n": int(_truth(group.quote_freshness_ok).sum()),
            "probability_scores_repaired_vs_this_book": book_scores,
            "max_abs_decomposition_residual": float(group.decomposition_residual.abs().max()),
        })
    return paired, summaries


def _fixed_price_comparisons(selected: pd.DataFrame, opportunities: pd.DataFrame, tax: float) -> tuple[pd.DataFrame, dict]:
    chosen = selected.copy()
    chosen["selected_odds"] = _finite(chosen.odds)
    chosen["selected_p"] = _finite(chosen.p_model_bet)
    chosen["selected_q"] = _finite(chosen.q_bet)
    chosen["selected_y"] = _truth(chosen.won).astype(float)
    keys = ["fixture_id", "horizon", "side"]
    price_frames = {}
    for source in ("best_price", "pinnacle"):
        raw = opportunities.loc[
            opportunities.cohort.eq("oddspapi_common") & opportunities.horizon.eq("T-1h")
            & opportunities.source.eq(source) & opportunities.audit_primary_eligible
        ].copy()
        raw["side"] = "A"
        raw["odds_A"] = raw.odds_a
        raw["odds_B"] = raw.odds_b
        raw["q_A"] = raw.q_consensus_a
        raw["q_B"] = 1.0 - raw.q_consensus_a
        raw["qbook_A"] = raw.q_book_a
        raw["qbook_B"] = 1.0 - raw.q_book_a
        raw["p_A"] = raw.p_event
        raw["p_B"] = 1.0 - raw.p_event
        raw["y_A"] = raw.y_event
        raw["y_B"] = 1.0 - raw.y_event
        part_a = raw[["fixture_id", "horizon", "odds_A", "q_A", "qbook_A", "p_A", "y_A", "bookmaker_a", "quote_age_max_hours", "quote_freshness_ok"]].rename(columns={"odds_A":"odds","q_A":"q","qbook_A":"q_book","p_A":"p","y_A":"y","bookmaker_a":"source_book"})
        part_a["side"] = "A"
        part_b = raw[["fixture_id", "horizon", "odds_B", "q_B", "qbook_B", "p_B", "y_B", "bookmaker_b", "quote_age_max_hours", "quote_freshness_ok"]].rename(columns={"odds_B":"odds","q_B":"q","qbook_B":"q_book","p_B":"p","y_B":"y","bookmaker_b":"source_book"})
        part_b["side"] = "B"
        price_frames[source] = pd.concat([part_a, part_b], ignore_index=True)
    selected_cols = chosen[["fixture_id", "horizon", "side", "selected_odds", "selected_p", "selected_q", "selected_y", "won", "month", "edition_proxy"]].copy()
    selected_cols["best_quote_age_hours"] = _finite(chosen.quote_age_max_hours).to_numpy()
    selected_cols["best_price_book_a"] = chosen.bookmaker_a.to_numpy()
    selected_cols["best_price_book_b"] = chosen.bookmaker_b.to_numpy()
    best = selected_cols.merge(price_frames["best_price"], on=keys, how="left", validate="one_to_one", suffixes=("", "_op"))
    pinnacle = selected_cols.merge(price_frames["pinnacle"], on=keys, how="left", validate="one_to_one", suffixes=("", "_pin"))
    best = best.rename(columns={"odds":"best_price_odds", "q":"q_consensus", "q_book":"q_best_price_constructed", "p":"p_best", "y":"y_best", "source_book":"best_price_selected_side_book"})
    pinnacle = pinnacle.rename(columns={"odds":"pinnacle_odds", "q":"q_consensus_pin", "q_book":"q_pinnacle", "p":"p_pin", "y":"y_pin", "source_book":"pinnacle_book"})
    paired = best.merge(
        pinnacle[[*keys, "pinnacle_odds", "q_consensus_pin", "q_pinnacle", "p_pin", "y_pin", "pinnacle_book", "quote_age_max_hours", "quote_freshness_ok"]],
        on=keys, how="left", validate="one_to_one", suffixes=("", "_pin")
    )
    paired["pinnacle_matched"] = _finite(paired.pinnacle_odds).notna()
    paired["consensus_fair_odds"] = np.where(paired.selected_q.gt(0), 1.0 / paired.selected_q, np.nan)
    for prefix, odds_col, q_col in (
        ("best_price", "best_price_odds", "selected_q"),
        ("pinnacle", "pinnacle_odds", "selected_q"),
        ("consensus_fair", "consensus_fair_odds", "selected_q"),
    ):
        odds = _finite(paired[odds_col])
        q = _finite(paired[q_col])
        p = paired.selected_p
        y = paired.selected_y
        paired[f"{prefix}_sports_vs_market"] = (1-tax) * odds * (p-q)
        paired[f"{prefix}_market_vs_result"] = (1-tax) * odds * (q-y)
        paired[f"{prefix}_return_error"] = (1-tax) * odds * (p-y)
        paired[f"{prefix}_actual_net_return"] = odds * (1-tax) * y - 1.0
    paired["best_minus_pinnacle_odds"] = paired.best_price_odds - paired.pinnacle_odds
    paired["best_minus_pinnacle_return_error"] = paired.best_price_return_error - paired.pinnacle_return_error
    paired["best_minus_pinnacle_sports_vs_market"] = paired.best_price_sports_vs_market - paired.pinnacle_sports_vs_market
    paired["best_minus_pinnacle_market_vs_result"] = paired.best_price_market_vs_result - paired.pinnacle_market_vs_result
    source_rows = []
    for variant, prefix, odds_col, qbook_col in (
        ("best_price", "best_price", "best_price_odds", None),
        ("pinnacle", "pinnacle", "pinnacle_odds", "q_pinnacle"),
        ("consensus_no_vig_fair_price_scenario", "consensus_fair", "consensus_fair_odds", None),
    ):
        valid = _finite(paired[odds_col]).notna()
        part = paired.loc[valid]
        source_rows.append({
            "variant": variant, "fixed_selected_identity_universe": True, "quoted_rows_available_n": int(len(part)),
            "wins": int(part.selected_y.sum()), "losses": int(len(part)-part.selected_y.sum()),
            "mean_odds": _mean(part[odds_col]),
            "mean_actual_net_return_per_stake": _mean(part[f"{prefix}_actual_net_return"]),
            "mean_return_error": _mean(part[f"{prefix}_return_error"]),
            "mean_sports_vs_market": _mean(part[f"{prefix}_sports_vs_market"]),
            "mean_market_vs_result": _mean(part[f"{prefix}_market_vs_result"]),
            "mean_bookmaker_q_minus_consensus": _mean(part[qbook_col] - part.selected_q) if qbook_col else None,
            "price_interpretation": "Observed quote" if variant != "consensus_no_vig_fair_price_scenario" else "Hypothetical no-vig fair-price scenario; not an offered quote",
        })
    pinned = paired.loc[paired.pinnacle_matched]
    matched_variant_rows = []
    for variant, prefix, odds_col, qbook_col in (
        ("best_price", "best_price", "best_price_odds", None),
        ("pinnacle", "pinnacle", "pinnacle_odds", "q_pinnacle"),
        ("consensus_no_vig_fair_price_scenario", "consensus_fair", "consensus_fair_odds", None),
    ):
        matched_variant_rows.append({
            "variant": variant, "n_identical_selected_ids_and_sides": int(len(pinned)),
            "wins": int(pinned.selected_y.sum()), "losses": int(len(pinned)-pinned.selected_y.sum()),
            "mean_odds": _mean(pinned[odds_col]),
            "mean_actual_net_return_per_stake": _mean(pinned[f"{prefix}_actual_net_return"]),
            "mean_return_error": _mean(pinned[f"{prefix}_return_error"]),
            "mean_sports_vs_market": _mean(pinned[f"{prefix}_sports_vs_market"]),
            "mean_market_vs_result": _mean(pinned[f"{prefix}_market_vs_result"]),
            "mean_bookmaker_q_minus_consensus": _mean(pinned[qbook_col]-pinned.selected_q) if qbook_col else None,
            "price_interpretation": "Observed quote" if variant != "consensus_no_vig_fair_price_scenario" else "Hypothetical no-vig fair-price scenario; not an offered quote",
        })
    paired_summary = {
        "selected_best_price_n": int(len(paired)),
        "pinnacle_price_matched_same_selected_ids_n": int(len(pinned)),
        "pinnacle_match_fraction": float(len(pinned)/len(paired)) if len(paired) else None,
        "on_pinnacle_matched_ids": {
            "n": int(len(pinned)),
            "best_price_mean_odds": _mean(pinned.best_price_odds),
            "pinnacle_mean_odds": _mean(pinned.pinnacle_odds),
            "mean_best_minus_pinnacle_odds": _mean(pinned.best_minus_pinnacle_odds),
            "mean_best_price_actual_net_return_per_stake": _mean(pinned.best_price_actual_net_return),
            "mean_pinnacle_actual_net_return_per_stake": _mean(pinned.pinnacle_actual_net_return),
            "mean_best_price_return_error": _mean(pinned.best_price_return_error),
            "mean_pinnacle_return_error": _mean(pinned.pinnacle_return_error),
            "mean_best_minus_pinnacle_return_error": _mean(pinned.best_minus_pinnacle_return_error),
            "mean_price_delta_sports_component": _mean(pinned.best_minus_pinnacle_sports_vs_market),
            "mean_price_delta_market_component": _mean(pinned.best_minus_pinnacle_market_vs_result),
            "max_abs_paired_price_decomposition_residual": float((pinned.best_minus_pinnacle_return_error-pinned.best_minus_pinnacle_sports_vs_market-pinned.best_minus_pinnacle_market_vs_result).abs().max()),
        },
        "coverage_variants_on_fixed_selection_universe": source_rows,
        "strict_same_matched_ids_and_sides_variants": matched_variant_rows,
    }
    return paired, paired_summary


def _paired_horizon(selected: pd.DataFrame, opportunities: pd.DataFrame, tax: float) -> tuple[pd.DataFrame, dict]:
    keys = ["fixture_id", "side"]
    base = selected[["fixture_id", "side", "month", "p_model_bet", "q_bet", "won", "odds"]].copy()
    base = base.rename(columns={"p_model_bet":"p_selected", "q_bet":"q_selected", "won":"y_selected", "odds":"saved_t1_odds"})
    frames = {}
    for horizon, label in (("T-1h", "t1"), ("T-24h", "t24")):
        raw = opportunities.loc[
            opportunities.cohort.eq("oddspapi_common") & opportunities.horizon.eq(horizon)
            & opportunities.source.eq("best_price") & opportunities.audit_primary_eligible
        ].copy()
        result = base.merge(raw, on="fixture_id", how="inner", validate="one_to_one", suffixes=("", f"_{label}"))
        result["odds_h"] = np.where(result.side.eq("A"), result.odds_a, result.odds_b)
        result["q_h"] = np.where(result.side.eq("A"), result.q_consensus_a, 1-result.q_consensus_a)
        result["p_h"] = np.where(result.side.eq("A"), result.p_event, 1-result.p_event)
        result["y_h"] = np.where(result.side.eq("A"), result.y_event, 1-result.y_event)
        result["horizon"] = horizon
        frames[label] = result[["fixture_id", "side", "month", "p_h", "q_h", "y_h", "odds_h", "quote_age_max_hours", "quote_at_a", "quote_at_b", "bookmaker_a", "bookmaker_b", "q_consensus_a", "audit_primary_eligible"]].copy()
    t1 = frames["t1"].rename(columns={c:f"{c}_t1" for c in frames["t1"].columns if c not in keys})
    t24 = frames["t24"].rename(columns={c:f"{c}_t24" for c in frames["t24"].columns if c not in keys})
    paired = t1.merge(t24, on=keys, how="inner", validate="one_to_one", suffixes=("", "_other"))
    paired["delta_odds_t1_minus_t24"] = paired.odds_h_t1 - paired.odds_h_t24
    paired["odds_pct_move_t1_vs_t24"] = paired.odds_h_t1 / paired.odds_h_t24 - 1.0
    paired["delta_q_t1_minus_t24"] = paired.q_h_t1 - paired.q_h_t24
    paired["delta_p_t1_minus_t24"] = paired.p_h_t1 - paired.p_h_t24
    paired["price_direction"] = np.select(
        [paired.delta_odds_t1_minus_t24.lt(-1e-12), paired.delta_odds_t1_minus_t24.gt(1e-12)],
        ["shortened_by_T-1h", "lengthened_by_T-1h"], default="unchanged"
    )
    for suffix in ("t1", "t24"):
        paired[f"return_error_{suffix}"] = (1-tax)*paired[f"odds_h_{suffix}"]*(paired[f"p_h_{suffix}"]-paired[f"y_h_{suffix}"])
    paired["delta_return_error_t1_minus_t24"] = paired.return_error_t1-paired.return_error_t24
    detail = {
        "fixed_T1_selected_ids_n": int(len(selected)), "same_selected_side_and_id_with_T24_available_n": int(len(paired)),
        "price_direction_counts": paired.price_direction.value_counts(dropna=False).to_dict(),
        "mean_t1_odds": _mean(paired.odds_h_t1), "mean_t24_odds": _mean(paired.odds_h_t24),
        "mean_odds_delta_t1_minus_t24": _mean(paired.delta_odds_t1_minus_t24),
        "mean_odds_pct_move_t1_vs_t24": _mean(paired.odds_pct_move_t1_vs_t24),
        "mean_q_delta_t1_minus_t24": _mean(paired.delta_q_t1_minus_t24),
        "mean_p_delta_t1_minus_t24": _mean(paired.delta_p_t1_minus_t24),
        "mean_return_error_T1h": _mean(paired.return_error_t1),
        "mean_return_error_T24h": _mean(paired.return_error_t24),
        "mean_return_error_delta_T1h_minus_T24h": _mean(paired.delta_return_error_t1_minus_t24),
        "not_same_quote_moment_warning": "T-24h and T-1h quotes are separate historical snapshots; all endpoint quote fields remain post-origin diagnostics only.",
    }
    return paired, detail


def _raw_identity_audit(frame: pd.DataFrame, protocol: dict, canonical: pd.DataFrame, output: Path) -> tuple[dict, pd.DataFrame]:
    fixture_rows = json.loads(protocol["paths"]["raw_fixtures"].read_text())
    history_rows = json.loads(protocol["paths"]["raw_history"].read_text())
    fixtures = {str(row.get("fixtureId")): row for row in fixture_rows if row.get("fixtureId") is not None}
    history = {str(row.get("match_id")): row for row in history_rows if row.get("match_id") is not None}
    canonical_index = canonical.set_index(canonical.golgg_match_id.astype(str), drop=False)
    records = []
    for row in frame.drop_duplicates("fixture_id").itertuples(index=False):
        fixture = fixtures.get(str(row.fixture_id))
        record = history.get(str(row.golgg_match_id))
        canonical_row = canonical_index.loc[str(row.golgg_match_id)] if str(row.golgg_match_id) in canonical_index.index else None
        if isinstance(canonical_row, pd.DataFrame):
            canonical_row = canonical_row.iloc[0]
        games = (record.get("games") or []) if record else []
        names_by_id: dict[str, set[str]] = {}
        for game in games:
            for id_key, name_key in (("t1_id", "t1_name"), ("t2_id", "t2_name")):
                team_id = game.get(id_key)
                team_name = game.get(name_key)
                if team_id is not None and team_name:
                    names_by_id.setdefault(str(team_id), set()).add(str(team_name))
        fixture_names = [fixture.get("participant1Name"), fixture.get("participant2Name")] if fixture else [None, None]
        raw_names = [record.get("sname_t1"), record.get("sname_t2")] if record else [None, None]
        if not all(_normal_name(name) for name in raw_names):
            raw_names = [record.get("t1_name"), record.get("t2_name")] if record else [None, None]
        name_pair_exact = None
        if all(_normal_name(name) for name in fixture_names + raw_names):
            name_pair_exact = set(map(_normal_name, fixture_names)) == set(map(_normal_name, raw_names))

        canonical_a = str(canonical_row.team1_id) if canonical_row is not None else None
        canonical_b = str(canonical_row.team2_id) if canonical_row is not None else None
        root_a = str(record.get("t1_id")) if record and record.get("t1_id") is not None else None
        root_b = str(record.get("t2_id")) if record and record.get("t2_id") is not None else None
        root_direct = root_a == canonical_a and root_b == canonical_b
        root_reverse = root_a == canonical_b and root_b == canonical_a
        if root_a is not None and root_b is not None:
            ids_match = bool(root_direct or root_reverse)
            id_evidence = "series_level_team_ids"
        elif canonical_a in names_by_id and canonical_b in names_by_id:
            ids_match = True
            id_evidence = "game_level_team_ids_only"
        else:
            ids_match = None
            id_evidence = "missing"

        canonical_a_names = names_by_id.get(canonical_a, set())
        canonical_b_names = names_by_id.get(canonical_b, set())
        canonical_a_label = next(iter(sorted(canonical_a_names)), None)
        canonical_b_label = next(iter(sorted(canonical_b_names)), None)
        if root_direct:
            canonical_a_label = raw_names[0] or canonical_a_label
            canonical_b_label = raw_names[1] or canonical_b_label
        elif root_reverse:
            canonical_a_label = raw_names[1] or canonical_a_label
            canonical_b_label = raw_names[0] or canonical_b_label

        history_y = None
        root_t1_win = record.get("t1_win") if record else None
        if root_t1_win is not None and root_direct:
            history_y = float(bool(root_t1_win))
        elif root_t1_win is not None and root_reverse:
            history_y = float(not bool(root_t1_win))
        elif record and record.get("won"):
            winner = _normal_name(record.get("won"))
            a_aliases = {_normal_name(name) for name in canonical_a_names if _normal_name(name)}
            b_aliases = {_normal_name(name) for name in canonical_b_names if _normal_name(name)}
            if winner and winner in a_aliases and winner not in b_aliases:
                history_y = 1.0
            elif winner and winner in b_aliases and winner not in a_aliases:
                history_y = 0.0
        canonical_y = _num(canonical_row.get("y"), None) if canonical_row is not None else None
        outcome_match = history_y == canonical_y if history_y is not None and canonical_y is not None else None
        history_best_of = _num(record.get("best_of"), None) if record else None
        canonical_best_of = _num(canonical_row.get("best_of"), None) if canonical_row is not None else None
        best_of_match = history_best_of == canonical_best_of if history_best_of is not None and canonical_best_of is not None else None
        score_one = _num(record.get("t1_score"), None) if record else None
        score_two = _num(record.get("t2_score"), None) if record else None
        internal_result_match = (
            bool(record.get("t1_win")) == (score_one > score_two)
            if record and record.get("t1_win") is not None and score_one is not None and score_two is not None
            else None
        )
        fixture_fields = list(fixture.keys()) if fixture else []
        result_keys = [key for key in fixture_fields if re.search(r"score|winner|result|outcome|team.?[12].?win", key, re.I)]
        fixture_status = fixture.get("statusName") if fixture else None
        result_evidence = history_y is not None or bool(record and record.get("score"))
        status_conflict = fixture_status == "Pre-Game" and result_evidence
        records.append({
            "fixture_id": str(row.fixture_id), "golgg_match_id": str(row.golgg_match_id),
            "fixture_present": fixture is not None, "history_present": record is not None,
            "participant1_id": fixture.get("participant1Id") if fixture else None,
            "participant2_id": fixture.get("participant2Id") if fixture else None,
            "participant1_name": fixture_names[0], "participant2_name": fixture_names[1],
            "history_t1_name": raw_names[0], "history_t2_name": raw_names[1],
            "canonical_team1_label_from_history": canonical_a_label,
            "canonical_team2_label_from_history": canonical_b_label,
            "participant_name_pair_matches_history": name_pair_exact,
            "participant_name_nonexact_is_not_identity_proof": name_pair_exact is False,
            "canonical_ids_match_history": ids_match, "canonical_id_evidence": id_evidence,
            "canonical_best_of": canonical_best_of, "history_best_of": history_best_of,
            "best_of_matches": best_of_match, "canonical_y": canonical_y,
            "history_y_aligned_to_canonical_a": history_y, "result_matches": outcome_match,
            "raw_history_result_matches_score": internal_result_match,
            "fixture_status": fixture_status, "fixture_status_conflicts_with_result_record": status_conflict,
            "fixture_has_odds": fixture.get("hasOdds") if fixture else None,
            "fixture_has_score_or_winner_field": bool(result_keys),
            "fixture_result_field_names": ",".join(result_keys),
            "raw_history_score": record.get("score") if record else None,
        })
    table = pd.DataFrame(records)
    bool_cols = (
        "participant_name_pair_matches_history", "canonical_ids_match_history",
        "best_of_matches", "result_matches", "raw_history_result_matches_score",
        "fixture_status_conflicts_with_result_record",
    )
    for column in bool_cols:
        table[column] = table[column].astype("boolean")
    result_key_counts = pd.Series(
        [key for row in fixture_rows for key in row.keys() if re.search(r"score|winner|result|outcome|team.?[12].?win", key, re.I)]
    ).value_counts().to_dict()
    identity = {
        "audit_rows": int(len(table)), "fixture_found_n": int(table.fixture_present.sum()),
        "raw_history_found_n": int(table.history_present.sum()),
        "participant_name_pair_exact_normalized_n": int(table.participant_name_pair_matches_history.sum()),
        "participant_name_pair_nonexact_n": int(table.participant_name_pair_matches_history.eq(False).sum()),
        "participant_name_pair_unavailable_n": int(table.participant_name_pair_matches_history.isna().sum()),
        "participant_name_note": "Nonexact names are literal normalized string differences; they may be aliases/sponsor names and are not counted as identity conflicts without a contradictory ID binding.",
        "canonical_raw_history_id_match_n": int(table.canonical_ids_match_history.sum()),
        "canonical_raw_history_id_conflict_n": int(table.canonical_ids_match_history.eq(False).sum()),
        "canonical_raw_history_id_unavailable_n": int(table.canonical_ids_match_history.isna().sum()),
        "canonical_id_evidence_counts": table.canonical_id_evidence.value_counts(dropna=False).to_dict(),
        "best_of_match_n": int(table.best_of_matches.sum()),
        "best_of_conflict_n": int(table.best_of_matches.eq(False).sum()),
        "result_match_n": int(table.result_matches.sum()),
        "result_conflict_n": int(table.result_matches.eq(False).sum()),
        "result_unavailable_n": int(table.result_matches.isna().sum()),
        "raw_history_result_matches_its_series_score_n": int(table.raw_history_result_matches_score.sum()),
        "raw_history_result_score_check_unavailable_n": int(table.raw_history_result_matches_score.isna().sum()),
        "fixture_status_counts": table.fixture_status.value_counts(dropna=False).to_dict(),
        "fixture_pre_game_status_with_series_result_n": int(table.fixture_status_conflicts_with_result_record.sum()),
        "fixture_score_or_winner_field_names_across_raw_source": result_key_counts,
        "raw_fixture_result_evidence_absent_note": "Fixture records contain participant labels/status but no score/winner/result keys; result comparisons use raw series-level IDs, BO, score, winner and canonical metadata from separately hashed sources.",
    }
    table.to_csv(output / "identity_audit.csv", index=False)
    return identity, table


def run(tables: dict[str, pd.DataFrame], protocol: dict, output: Path) -> dict:
    opportunities = tables["opportunities"].copy()
    bets = tables["bets"].copy()
    canonical = tables["canonical"].copy()
    tax = float(protocol["fixed_contract"]["tax_rate"])
    seed = int(protocol["fixed_contract"]["seed"])
    if tax != TAX or seed != SEED:
        # The frozen protocol is authoritative; retain support without silently substituting.
        pass
    main = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.source.eq("best_price")
        & opportunities.horizon.eq("T-1h") & _truth(opportunities.audit_primary_eligible)
    ].copy()
    supplied_main = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.source.eq("best_price")
        & opportunities.horizon.eq("T-1h") & _truth(opportunities.available)
    ].copy()
    if main.fixture_id.duplicated().any() or supplied_main.fixture_id.duplicated().any():
        raise ValueError("Primary or as-supplied best-price opportunity fixture keys are not unique")
    selected_raw = _selection_rows(bets, "exclude_q25_conflicts", "raw_ev_gt_5", "best_price")
    selected_current = _selection_rows(bets, "exclude_q25_conflicts", "current_application_filters", "best_price")
    selected_supplied = _selection_rows(bets, "as_supplied_diagnostic", "raw_ev_gt_5", "best_price")
    if len(main) != 1852 or len(selected_raw) != 498:
        raise ValueError(f"Frozen primary market cohort changed: opportunities={len(main)}, selected={len(selected_raw)}")
    if selected_raw.fixture_id.duplicated().any() or selected_supplied.fixture_id.duplicated().any():
        raise ValueError("Saved market selections repeat fixture IDs")
    sides = _side_frame(main, selected_raw, tax)
    selected_sides = sides.loc[sides.selected].copy()
    if len(selected_sides) != len(selected_raw):
        raise ValueError("Could not map every saved selection to one primary side opportunity")

    # Check exact saved quote/model/outcome alignment before interpreting any return decomposition.
    saved_check = selected_raw.merge(
        selected_sides[["fixture_id", "side", "odds", "p", "q", "y"]],
        on=["fixture_id", "side"], how="left", validate="one_to_one", suffixes=("_saved", "_opportunity")
    )
    price_match = np.isclose(_finite(saved_check.odds_saved), _finite(saved_check.odds_opportunity), atol=1e-12, rtol=0, equal_nan=False)
    p_match = np.isclose(_finite(saved_check.p_model_bet), _finite(saved_check.p), atol=1e-12, rtol=0, equal_nan=False)
    q_match = np.isclose(_finite(saved_check.q_bet), _finite(saved_check.q), atol=1e-12, rtol=0, equal_nan=False)
    y_match = np.isclose(_truth(saved_check.won).astype(float), _finite(saved_check.y_opportunity), atol=0, rtol=0, equal_nan=False)
    if not (price_match.all() and p_match.all() and q_match.all() and y_match.all()):
        raise ValueError("Saved best-price selected row does not align with frozen opportunity quote/probabilities/outcome")

    source_counts = _source_count_table(bets)
    source_counts.to_csv(output / "source_selection_counts.csv", index=False)
    primary_summary = _summary(selected_sides)
    current_sides = _side_frame(main, selected_current, tax).loc[lambda x: x.selected]
    ev_band = selected_raw.loc[_finite(selected_raw.ev_bet).gt(0.05) & _finite(selected_raw.ev_bet).le(0.10)].copy()
    ev_band_sides = _side_frame(main, ev_band, tax).loc[lambda x: x.selected]
    supplied_sides = _side_frame(supplied_main, selected_supplied, tax).loc[lambda x: x.selected]
    q25_conflict_sides = supplied_sides.loc[_truth(supplied_sides.conflict_q25)]
    if len(supplied_sides) != len(selected_supplied):
        raise ValueError("Could not map every as-supplied selection, including frozen q25 conflicts")
    selected_controls = {
        "raw_ev_gt_5": primary_summary,
        "current_application_filters": _summary(current_sides),
        "saved_ev_5_to_10_percent_band": _summary(ev_band_sides),
        "ev_band_definition": "Saved ledger ev_bet > 0.05 and <= 0.10, among frozen best_price raw_ev_gt_5 selected rows; no reselection.",
        "q25_sensitivity": {
            "excluded_quality_selected_n": int(len(selected_raw)), "as_supplied_quality_selected_n": int(len(supplied_sides)),
            "as_supplied_q25_conflict_n": int(q25_conflict_sides.shape[0]),
            "as_supplied_selected_nonconflict_n": int(len(supplied_sides)-len(q25_conflict_sides)),
            "selection_key_overlap_n": int(len(set(selected_raw.fixture_id.astype(str)) & set(selected_supplied.fixture_id.astype(str)))),
            "excluded_metrics": _summary(selected_sides), "as_supplied_metrics": _summary(supplied_sides),
            "q25_conflict_only_metrics": _summary(q25_conflict_sides),
            "quote_freshness_true_n_selected": int(_truth(selected_sides.quote_freshness_ok).sum()),
            "quote_age_over_24h_n_selected": int(_finite(selected_sides.quote_age_max_hours).gt(24).sum()),
            "note": "Saved quality inventories compared as frozen; q25 conflict selections included only in the supplied sensitivity, with no price flips.",
        },
    }

    # Complete side-candidate universe: selected flags come only from saved ledger rows.
    side_summary = {
        "all_available_side_candidates": _summary(sides),
        "saved_selected_side_candidates": _summary(sides.loc[sides.selected]),
        "unselected_side_candidates": _summary(sides.loc[sides.unselected]),
        "selection_unit_note": "Each available pair contributes A and B side candidates; exactly the ledger-selected side is marked selected. Unselected candidates are controls, not newly proposed bets; the two sides share one match outcome.",
        "selected_winners": _summary(sides.loc[sides.selected_winner]),
        "selected_losers": _summary(sides.loc[sides.selected_loser]),
    }
    probability_scores = {
        "all_primary_side_candidates": _probability_scores(sides, float(protocol["fixed_contract"]["probability_epsilon"])),
        "saved_selected_sides": _probability_scores(sides.loc[sides.selected], float(protocol["fixed_contract"]["probability_epsilon"])),
        "unselected_side_candidates": _probability_scores(sides.loc[sides.unselected], float(protocol["fixed_contract"]["probability_epsilon"])),
        "current_application_filter_sides": _probability_scores(current_sides, float(protocol["fixed_contract"]["probability_epsilon"])),
        "as_supplied_including_q25_conflicts": _probability_scores(supplied_sides, float(protocol["fixed_contract"]["probability_epsilon"])),
    }
    monthly_blocks = sides.month.fillna("unknown").astype(str).to_numpy()
    monthly = _block_component_evidence(sides, sides.selected, pd.Series(monthly_blocks), "all T-1h primary months, including empty selected subgroups")
    edition_blocks = sides.edition_proxy.fillna("unknown").astype(str).to_numpy()
    edition = _block_component_evidence(sides, sides.selected, pd.Series(edition_blocks), "tournament-name-by-year proxy; not certified tournament editions")
    selected_unselected_effects = {}
    for component in ("sports_vs_market", "market_vs_result", "return_error"):
        selected_unselected_effects[component] = _paired_difference_ci(
            sides[component].to_numpy(), sides.selected.to_numpy(), sides.unselected.to_numpy(), monthly_blocks
        )
    monthly["selected_minus_unselected_paired_effects"] = selected_unselected_effects

    # Price/age/dispersion strata use the same selected identity list and are descriptive, not a new policy.
    dispersion = _book_dispersion(opportunities)
    sides = sides.merge(dispersion, on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    odds_bins = _cut_table(sides, "odds", [1, 1.5, 2, 3, 5, np.inf], ["[1,1.5)", "[1.5,2)", "[2,3)", "[3,5)", "[5,+inf)"], selected_only=True)
    age_bins = _cut_table(sides.loc[sides.selected], "quote_age_max_hours", [0, 1, 6, 24, 72, np.inf], ["[0,1h)", "[1,6h)", "[6,24h)", "[24,72h)", "[72h,+inf)"], selected_only=False)
    dispersion_bins = _cut_table(sides.loc[sides.selected], "dispersion_sd", [0, .01, .02, .05, np.inf], ["[0,.01)", "[.01,.02)", "[.02,.05)", "[.05,+inf)"], selected_only=False)
    strata = {"selected_odds_bins": odds_bins, "selected_quote_age_bins": age_bins, "selected_book_dispersion_sd_bins": dispersion_bins}
    pd.DataFrame([{"stratum":"odds", **row} for row in odds_bins] + [{"stratum":"quote_age_hours", **row} for row in age_bins] + [{"stratum":"book_dispersion_sd", **row} for row in dispersion_bins]).to_csv(output / "decomposition_bins.csv", index=False)

    # Book source and label provenance, chosen book versus a leave-one-book-out quote consensus.
    all_primary_available = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.horizon.eq("T-1h")
        & opportunities.source.eq("best_price") & opportunities.available
    ].copy()
    source_integrity, chosen_book, individual = _market_source_integrity(all_primary_available, selected_raw, opportunities)
    chosen_book.to_csv(output / "chosen_book_leave_one_out.csv", index=False)
    fixed_book_rows, fixed_book_summary = _fixed_selection_bookmaker_comparison(selected_raw, opportunities, tax)
    fixed_book_rows.to_csv(output / "fixed_selection_all_bookmakers.csv", index=False)

    # Fixed, matched quote comparisons; the consensus fair price is a scenario, not a market offer.
    price_rows, fixed_prices = _fixed_price_comparisons(selected_raw, opportunities, tax)
    price_rows.to_csv(output / "fixed_price_same_selection.csv", index=False)
    horizon_rows, horizon_summary = _paired_horizon(selected_raw, opportunities, tax)
    horizon_rows.to_csv(output / "same_selection_T24_vs_T1.csv", index=False)
    horizon_opportunity_n = {}
    horizon_available_n = {}
    horizon_raw_selected_n = {}
    horizon_current_selected_n = {}
    for _horizon in ("T-1h", "T-24h"):
        _op_h = opportunities.loc[
            opportunities.cohort.eq("oddspapi_common") & opportunities.source.eq("best_price")
            & opportunities.horizon.eq(_horizon)
        ]
        horizon_opportunity_n[_horizon] = int(_truth(_op_h.audit_primary_eligible).sum())
        horizon_available_n[_horizon] = int(_truth(_op_h.available).sum())
        _bet_h = bets.loc[
            bets.cohort.eq("oddspapi_common") & bets.horizon.eq(_horizon)
            & bets.source.eq("best_price") & bets.quality.eq("exclude_q25_conflicts")
        ]
        horizon_raw_selected_n[_horizon] = int(_truth(_bet_h.loc[_bet_h.policy.eq("raw_ev_gt_5"), "placed"]).sum())
        horizon_current_selected_n[_horizon] = int(_truth(_bet_h.loc[_bet_h.policy.eq("current_application_filters"), "placed"]).sum())
    horizon_coverage = {
        "eligible_best_price_opportunity_n_by_horizon": horizon_opportunity_n,
        "available_best_price_opportunity_n_by_horizon": horizon_available_n,
        "saved_raw_ev_gt_5_best_price_n_by_horizon": horizon_raw_selected_n,
        "saved_current_filter_best_price_n_by_horizon": horizon_current_selected_n,
        "fixed_T1_selected_rows_with_T24_price_n": int(len(horizon_rows)),
    }

    # Paired T24 sensitivity CI: keep the complete primary monthly opportunity universe and frozen T1 IDs.
    t24_by_id = horizon_rows.set_index(["fixture_id", "side"])
    sides["t24_delta_odds"] = 0.0
    sides["t24_delta_return_error"] = 0.0
    sides["t24_match"] = False
    t1_selected = sides.loc[sides.selected, ["fixture_id", "side"]]
    horizon_index = t1_selected.set_index(["fixture_id", "side"]).index
    if len(horizon_rows):
        matched_idx = pd.MultiIndex.from_frame(sides[["fixture_id", "side"]]).isin(t24_by_id.index)
        sides.loc[matched_idx, "t24_match"] = True
        matched_positions = sides.index[matched_idx]
        delta_lookup = horizon_rows.set_index(["fixture_id", "side"])
        join_idx = pd.MultiIndex.from_frame(sides.loc[matched_idx, ["fixture_id", "side"]])
        sides.loc[matched_idx, "t24_delta_odds"] = delta_lookup.loc[join_idx, "delta_odds_t1_minus_t24"].to_numpy()
        sides.loc[matched_idx, "t24_delta_return_error"] = delta_lookup.loc[join_idx, "delta_return_error_t1_minus_t24"].to_numpy()
    t24_mask = sides.selected & sides.t24_match
    horizon_ci = {
        "same_selected_quote_age_or_price_diagnostic_n": int(t24_mask.sum()),
        "odds_delta_T1h_minus_T24h": _ratio_ci(sides.t24_delta_odds.to_numpy(), t24_mask.to_numpy(), monthly_blocks),
        "return_error_delta_T1h_minus_T24h": _ratio_ci(sides.t24_delta_return_error.to_numpy(), t24_mask.to_numpy(), monthly_blocks),
    }

    # Raw fixture participant/result label audit against the separately hashed canonical series and raw history.
    identity, identity_rows = _raw_identity_audit(main, protocol, canonical, output)
    selected_identity = selected_raw.merge(
        identity_rows[[
            "fixture_id", "participant1_name", "participant2_name", "canonical_team1_label_from_history",
            "canonical_team2_label_from_history", "participant_name_pair_matches_history",
            "canonical_ids_match_history", "canonical_id_evidence", "canonical_best_of", "history_best_of",
            "best_of_matches", "canonical_y", "history_y_aligned_to_canonical_a", "result_matches",
            "raw_history_result_matches_score", "fixture_status", "fixture_status_conflicts_with_result_record",
            "fixture_has_score_or_winner_field",
        ]],
        on="fixture_id", how="left", validate="many_to_one"
    )
    case_identity_cols = [
        "fixture_id", "participant1_name", "participant2_name", "canonical_team1_label_from_history",
        "canonical_team2_label_from_history", "participant_name_pair_matches_history",
        "canonical_ids_match_history", "canonical_best_of", "history_best_of", "best_of_matches",
        "canonical_y", "history_y_aligned_to_canonical_a", "result_matches", "fixture_status",
        "fixture_status_conflicts_with_result_record",
    ]
    top_cases = selected_sides.assign(abs_contribution=lambda f: f.return_error.abs()).sort_values("abs_contribution", ascending=False).head(25)
    top_cases = top_cases.merge(identity_rows[case_identity_cols], on="fixture_id", how="left", validate="many_to_one")
    top_cases["won"] = top_cases.y.astype(bool)
    case_fields = [
        "fixture_id", "golgg_match_id", "date", "tournament_name", "best_of", "side", "p", "q", "y", "odds",
        "bookmaker_a", "bookmaker_b", "quote_age_max_hours", "q_consensus_members", "q_consensus_books",
        "participant1_name", "participant2_name", "canonical_team1_label_from_history", "canonical_team2_label_from_history",
        "return_error", "sports_vs_market", "market_vs_result", "won", "abs_contribution",
    ]
    top_cases[case_fields].to_csv(output / "largest_selected_return_error_cases.csv", index=False)
    top_case_json = [
        {
            key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value.isoformat() if isinstance(value, pd.Timestamp) else value)
            for key, value in row.items()
        }
        for row in top_cases[case_fields].head(10).to_dict("records")
    ]
    positive_cases = selected_sides.loc[selected_sides.return_error.gt(0)].sort_values("return_error", ascending=False).head(10)
    positive_cases["abs_contribution"] = positive_cases.return_error.abs()
    positive_cases = positive_cases.merge(identity_rows[case_identity_cols], on="fixture_id", how="left", validate="many_to_one")
    positive_cases["won"] = positive_cases.y.astype(bool)
    positive_cases.to_csv(output / "largest_positive_return_error_cases.csv", index=False)
    positive_case_fields = case_fields
    positive_case_json = [
        {
            key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value.isoformat() if isinstance(value, pd.Timestamp) else value)
            for key, value in row.items()
        }
        for row in positive_cases[positive_case_fields].to_dict("records")
    ]
    label_fields = [
        "fixture_id", "participant1_name", "participant2_name", "canonical_team1_label_from_history",
        "canonical_team2_label_from_history", "participant_name_pair_matches_history", "canonical_ids_match_history",
        "canonical_id_evidence", "canonical_best_of", "history_best_of", "best_of_matches", "canonical_y",
        "history_y_aligned_to_canonical_a", "result_matches", "raw_history_result_matches_score",
        "fixture_status", "fixture_status_conflicts_with_result_record", "fixture_has_score_or_winner_field",
    ]
    selected_identity[label_fields].drop_duplicates("fixture_id").to_csv(output / "selected_identity_examples.csv", index=False)

    # Future endpoint values are explicitly diagnostic, never substituted into selected prices/features.
    endpoint_status = {}
    for column in ("endpoint_status_a", "endpoint_status_b", "endpoint_label"):
        if column in all_primary_available:
            endpoint_status[column] = all_primary_available[column].value_counts(dropna=False).to_dict()
    endpoint_timestamps = pd.concat([
        pd.to_datetime(all_primary_available.endpoint_quote_at_a, utc=True, errors="coerce"),
        pd.to_datetime(all_primary_available.endpoint_quote_at_b, utc=True, errors="coerce"),
    ], ignore_index=True)
    endpoint_status["endpoint_timestamp_present_n_of_2x_available_quotes"] = int(endpoint_timestamps.notna().sum())
    endpoint_status["usage"] = "Diagnostic endpoint/status fields only; excluded from price, selection, model, and decomposition calculations."

    # Explicit complete-universe row-level identity proof.
    identity_residual_max = float(sides.loc[sides.selected, "identity_residual"].abs().max())
    all_identity_residual_max = float(sides.identity_residual.abs().max())
    monthly_counts = main.groupby("month", dropna=False).size().to_dict()
    results = {
        "summary": {
            "track": "market", "run_id": protocol["run_id"],
            "primary_opportunity_rows": int(len(main)), "primary_unique_fixtures": int(main.fixture_id.nunique()),
            "primary_source": "best_price", "primary_horizon": "T-1h",
            "frozen_selected_raw_ev_gt_5_n": int(len(selected_raw)),
            "frozen_selected_current_application_filters_n": int(len(selected_current)),
            "monthly_universe": monthly_counts,
            "decomposition_identity": {
                "formula": "(1-tax)*O*(p-Y) = (1-tax)*O*(p-q) + (1-tax)*O*(q-Y)",
                "tax_rate": tax, "selected_row_max_abs_residual": identity_residual_max,
                "all_side_candidate_max_abs_residual": all_identity_residual_max,
                "selected_rows_checked": int(len(selected_sides)),
            },
            "saved_alignment": {"saved_quote_match_n": int(price_match.sum()), "saved_model_probability_match_n": int(p_match.sum()), "saved_market_probability_match_n": int(q_match.sum()), "saved_outcome_match_n": int(y_match.sum())},
        },
        "empirical_evidence": {
            "selected_filters_and_q25": selected_controls,
            "source_selection_counts_csv": "source_selection_counts.csv",
            "per_source_selected_N": source_counts.loc[(source_counts.quality.eq("exclude_q25_conflicts")) & (source_counts.policy.eq("raw_ev_gt_5")), ["source", "placed_n", "winners", "losers"]].to_dict("records"),
            "selected_vs_unselected_side_opportunities": side_summary,
            "probability_scores_epsilon_1e_12": probability_scores,
            "central_decomposition_monthly_block_ci": monthly,
            "central_decomposition_tournament_year_proxy_ci": edition,
            "selected_odds_age_dispersion_bins": strata,
            "matched_price_variants": fixed_prices,
            "fixed_selected_sides_across_all_individual_books": fixed_book_summary,
            "chosen_book_vs_leave_one_book_out": source_integrity,
            "T1_and_T24_cohort_coverage": horizon_coverage,
            "T24_vs_T1_same_saved_ids_and_sides": horizon_summary,
            "T24_vs_T1_paired_monthly_ci": horizon_ci,
            "raw_fixture_vs_canonical_identity": identity,
            "future_endpoint_diagnostics": endpoint_status,
            "largest_absolute_contribution_cases_top10": top_case_json,
            "largest_positive_return_error_cases_top10": positive_case_json,
        },
        "confirmed_mechanisms": [],
        "hypotheses_rejected_or_weakened": [],
        "open_hypotheses": [],
        "artifact_paths": {
            "results": "results.json",
            "source_counts": "source_selection_counts.csv",
            "decomposition_bins": "decomposition_bins.csv",
            "fixed_same_selection_prices": "fixed_price_same_selection.csv",
            "fixed_selected_sides_all_books": "fixed_selection_all_bookmakers.csv",
            "T24_T1_matched": "same_selection_T24_vs_T1.csv",
            "chosen_book_LOO": "chosen_book_leave_one_out.csv",
            "identity_audit": "identity_audit.csv",
            "selected_identity_examples": "selected_identity_examples.csv",
            "largest_contribution_cases": "largest_selected_return_error_cases.csv",
            "largest_positive_return_error_cases": "largest_positive_return_error_cases.csv",
        },
        "source_hashes_and_limitations": {
            "prepared_input_hashes": json.loads((Path(protocol["input_dir"]) / "prepared_receipt.json").read_text())["sha256"],
            "raw_fixture_sha256": protocol["sources"]["raw_fixtures"]["sha256"],
            "raw_history_sha256": protocol["sources"]["raw_history"]["sha256"],
            "market_source_hashes_in_primary_rows": {
                "source_sha256_a_unique": int(all_primary_available.source_sha256_a.nunique(dropna=True)),
                "source_sha256_b_unique": int(all_primary_available.source_sha256_b.nunique(dropna=True)),
                "quote_contract_ok_n": int(_truth(all_primary_available.quote_contract_ok).sum()),
                "bookmaker_binding_verified_n": int(_truth(all_primary_available.bookmaker_binding_verified).sum()),
                "participant_binding_status_a": all_primary_available.proof_a_participant_binding_status.value_counts(dropna=False).to_dict(),
                "participant_binding_status_b": all_primary_available.proof_b_participant_binding_status.value_counts(dropna=False).to_dict(),
            },
            "limitations": [
                "All origin rows are frozen historical scenarios; no forecast issuance or quote execution is established.",
                "Bookmaker name/side binding remains explicitly unverified in the archived provenance; best_price may combine different bookmakers across sides.",
                "Consensus q is a no-vig market reference, not assumed to be a true probability.",
                "Raw fixtures expose participant labels/status but no score/winner evidence in their fixture keys; result comparison is against separately hashed raw series history.",
                "Tournament-name-by-year groups are proxy blocks, not certified tournament editions; all subgroup contrasts are exploratory.",
                "Side-candidate opportunity controls include both canonical sides and are not independent match observations or recommendations.",
                "T-24h and endpoint values are separate historical/future diagnostics; no future quote endpoint was used to select, reprice, or explain a frozen decision.",
            ],
        },
    }
    # Keep key findings concrete without converting observational associations into causes.
    sports_mean = primary_summary["mean_sports_vs_market"]
    market_mean = primary_summary["mean_market_vs_result"]
    if sports_mean is not None and market_mean is not None:
        results["confirmed_mechanisms"] = [
            "At fixed saved selections and quotes, selected-return error decomposes exactly into repaired-model minus no-vig-consensus and no-vig-consensus minus realized-outcome components; both components and their paired monthly intervals are reported.",
            "The primary selections use saved best_price prices; matched Pinnacle comparisons hold selected fixture, side, p, q, and outcome fixed wherever a qualifying Pinnacle quote exists.",
        ]
        results["hypotheses_rejected_or_weakened"] = [
            "A causal bookmaker-binding explanation is not verified: all available quote rows retain unverified provider-outcome convention status, and bookmaker_binding_verified is false; source names and prices are diagnostic labels only.",
            "Treating the consensus as truth is not supported; q is only a no-vig reference, so the two decomposition components are descriptive rather than causal attribution.",
        ]
        results["open_hypotheses"] = [
            "Whether stale quote age or cross-book dispersion independently explains selected errors remains associational; report bins with winner and loser controls, with no prespecified causal identification.",
            "Whether bookmaker-specific source labels bind the actual odds participants cannot be settled from these prepared records; quote paths/hashes are present, but participant-binding certification is absent.",
        ]
    return results
