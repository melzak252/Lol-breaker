"""Financial simulation from exact stored C0 forecasts and frozen EXP-039 history."""

from __future__ import annotations

import json
from collections import defaultdict
from heapq import heappop, heappush
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from betting_app.api.deps import get_db, query_df
from betting_app.api.schemas import FinancialAnalysisResponse, FinancialBucket, FinancialLedgerEntry
from betting_app.core.ev import expected_value, fair_market_probabilities
from betting_app.core.models.registry import BAYESIAN_SHRUNK_HYBRID, C0_NATIVE
from betting_app.services.canonical_match_service import align_snapshot_odds
from betting_app.ml.calibration.conformal_contract import conformal_bounds_for_side
from betting_app.services.market_service import kelly_fraction, none_or_float


router = APIRouter(prefix="/financial", tags=["financial"])

TAX_RATE = 0.12
BACKTEST_FEATURES_VERSION = "exp060-db-backfill-v1"

# Frozen retrospective identity, deliberately independent of the active registry.
ARCHIVED_EXP039_MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
ARCHIVED_EXP039_MODEL_VERSION = "exp-039"

HYBRID_MODEL_NAME = BAYESIAN_SHRUNK_HYBRID.name
DEFAULT_HYBRID_VERSION = BAYESIAN_SHRUNK_HYBRID.version

HORIZONS: list[tuple[str, str, float, float | None]] = [
    ("0-2", "0–2 h", 0, 2),
    ("2-6", "2–6 h", 2, 6),
    ("6-12", "6–12 h", 6, 12),
    ("12-24", "12–24 h", 12, 24),
    ("24-48", "24–48 h", 24, 48),
    ("48+", "48 h+", 48, None),
]


def _pick_snapshot(rows: list[dict[str, Any]], odds_mode: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if odds_mode == "open":
        return rows[0]
    if odds_mode == "mid":
        return rows[(len(rows) - 1) // 2]
    return rows[-1]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _has_stored_diagnostics(value: Any) -> bool:
    """Report source-record presence, not safety or issuance certification."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    return isinstance(value, dict) and bool(value)


def _horizon(hours_before: float) -> tuple[str, str]:
    for key, label, low, high in HORIZONS:
        if hours_before >= low and (high is None or hours_before < high):
            return key, label
    return "unknown", "Brak czasu"


def _bucket(key: str, label: str, rows: list[dict[str, Any]]) -> FinancialBucket:
    staked = sum(float(row["stake"]) for row in rows)
    profit = sum(float(row["profit"]) for row in rows)
    clv = [float(row["clv_odds_pct"]) for row in rows if row["clv_odds_pct"] is not None]
    return FinancialBucket(
        key=key,
        label=label,
        bets=len(rows),
        matches=len({int(row["canonical_match_id"]) for row in rows}),
        staked=staked,
        profit=profit,
        roi=profit / staked if staked else None,
        hit_rate=sum(bool(row["won"]) for row in rows) / len(rows) if rows else None,
        avg_ev=sum(float(row["ev"]) for row in rows) / len(rows) if rows else None,
        avg_clv_odds_pct=sum(clv) / len(clv) if clv else None,
        median_clv_odds_pct=median(clv) if clv else None,
    )


@router.get("/analysis", response_model=FinancialAnalysisResponse)
def financial_analysis(
    days_back: int = 90,
    odds_mode: str = "mid",
    model_name: str = HYBRID_MODEL_NAME,
    model_version: str = DEFAULT_HYBRID_VERSION,
    min_ev: float = 0.0,
    staking_mode: str = "kelly",
    initial_bankroll: float = 1000.0,
    fixed_stake: float = 10.0,
    kelly_fraction_multiplier: float = 0.25,
    max_stake_pct: float = 0.05,
    data_scope: str = "historical",
    use_conformal_gating: bool = True,
    max_conformal_width: float = 0.08,
    db=Depends(get_db),
):
    """Simulate historical quote selection and event-time settlement.

    Current scopes consume exact stored C0 forecasts; hybrid forecasts already
    contain their market shrinkage and are never blended again here.
    ``retrospective`` is the isolated frozen EXP-039 backfill. No scope certifies
    recorded bet issuance, execution, calibration or profitability.
    """
    odds_mode = odds_mode.lower().strip()
    if odds_mode not in {"open", "mid", "close"}:
        raise HTTPException(400, "odds_mode must be one of: open, mid, close")
    if staking_mode not in {"fixed", "kelly"}:
        raise HTTPException(400, "staking_mode must be fixed or kelly")
    if data_scope not in {"live", "historical", "retrospective"}:
        raise HTTPException(
            400,
            "data_scope must be live, historical or retrospective",
        )
    if not (
        0 <= min_ev < 2
        and 0 < max_conformal_width <= 1
        and initial_bankroll > 0
        and fixed_stake > 0
        and 0 < kelly_fraction_multiplier <= 1
        and 0 < max_stake_pct <= 1
    ):
        raise HTTPException(400, "Invalid financial simulation parameters")

    use_hybrid = model_name == HYBRID_MODEL_NAME
    if data_scope == "retrospective":
        if (
            model_name != ARCHIVED_EXP039_MODEL_NAME
            or model_version != ARCHIVED_EXP039_MODEL_VERSION
        ):
            raise HTTPException(
                400,
                "retrospective scope requires the explicit frozen EXP-039 identity",
            )
        query_model_name = ARCHIVED_EXP039_MODEL_NAME
        query_model_version = ARCHIVED_EXP039_MODEL_VERSION
        features_version = BACKTEST_FEATURES_VERSION
        effective_model_version = ARCHIVED_EXP039_MODEL_VERSION
    else:
        selected_model = BAYESIAN_SHRUNK_HYBRID if use_hybrid else C0_NATIVE
        if (
            model_name != selected_model.name
            or model_version != selected_model.version
        ):
            raise HTTPException(
                400,
                "Unsupported model/version; use the exact active C0 or C0 hybrid contract",
            )
        query_model_name = selected_model.name
        query_model_version = selected_model.version
        features_version = selected_model.feature_version
        effective_model_version = selected_model.version

    min_start_at = datetime.now(UTC) - timedelta(days=min(days_back, 730))

    prediction_rows = query_df(db, """
        SELECT cm.id AS canonical_match_id, cm.team_a_name, cm.team_b_name, cm.league,
               cm.start_time_normalized, cm.result_recorded_at, cm.winner_side,
               p.prob_a, p.prob_b, p.predicted_at, p.data_cutoff_at,
               p.diagnostics_json, p.id AS prediction_id
        FROM canonical_matches cm
        JOIN canonical_predictions p ON p.canonical_match_id = cm.id
        WHERE cm.status = 'finished'
          AND cm.winner_side IN ('team_a', 'team_b')
          AND p.model_name = :model_name AND p.model_version = :model_version
          AND p.features_version = :features_version
        ORDER BY cm.start_time_normalized, cm.id, p.predicted_at, p.id
    """, {
        "model_name": query_model_name,
        "model_version": query_model_version,
        "features_version": features_version,
    })
    latest_predictions: dict[int, dict[str, Any]] = {}
    for row in prediction_rows:
        predicted_at = _parse_time(row.get("predicted_at"))
        start_at = _parse_time(row.get("start_time_normalized"))
        if (
            predicted_at is None
            or start_at is None
            or start_at < min_start_at
            or predicted_at >= start_at
        ):
            continue
        match_id = int(row["canonical_match_id"])
        previous = latest_predictions.get(match_id)
        if previous is None or predicted_at > _parse_time(previous["predicted_at"]):
            latest_predictions[match_id] = row
    matches = sorted(
        latest_predictions.values(),
        key=lambda row: (str(row["start_time_normalized"]), int(row["canonical_match_id"])),
    )
    historical_diagnostics_status = (
        "available"
        if matches and all(
            _has_stored_diagnostics(row.get("diagnostics_json")) for row in matches
        )
        else "unknown"
    )
    uncertainty_basis = "missing_bounds" if use_conformal_gating else "mean"
    uncertainty_reason = (
        "conformal_bounds_unavailable"
        if use_conformal_gating
        else "uncertainty_not_requested"
    )
    temporal_exclusions: dict[str, int] = {
        "prediction_rows_not_eligible": len(prediction_rows) - len(matches),
        "missing_result_available_at": 0,
        "result_not_after_start": 0,
        "missing_data_cutoff_at": 0,
        "data_cutoff_after_prediction": 0,
        "no_quote_after_prediction": 0,
        "no_quote_before_start": 0,
        "conformal_bounds_unavailable": 0,
        "conformal_bounds_too_wide": 0,
        "no_positive_ev": 0,
        "insufficient_available_balance": 0,
    }
    temporally_eligible_matches: list[dict[str, Any]] = []
    for match in matches:
        start_at = _parse_time(match.get("start_time_normalized"))
        result_available_at = _parse_time(match.get("result_recorded_at"))
        if result_available_at is None:
            temporal_exclusions["missing_result_available_at"] += 1
            continue
        if start_at is None or result_available_at <= start_at:
            temporal_exclusions["result_not_after_start"] += 1
            continue
        if data_scope == "live":
            data_cutoff_at = _parse_time(match.get("data_cutoff_at"))
            predicted_at = _parse_time(match.get("predicted_at"))
            if data_cutoff_at is None:
                temporal_exclusions["missing_data_cutoff_at"] += 1
                continue
            if predicted_at is None or data_cutoff_at > predicted_at:
                temporal_exclusions["data_cutoff_after_prediction"] += 1
                continue
        temporally_eligible_matches.append(match)
    matches = temporally_eligible_matches
    if not matches:
        return FinancialAnalysisResponse(
            methodology=(
                "Historical quote simulation with no eligible forecasts; "
                "not recorded bet issuance or execution."
            ),
            evaluation_kind="historical_quote_simulation",
            result_status="no_eligible_bets",
            uncertainty_basis=uncertainty_basis,
            uncertainty_reason=uncertainty_reason,
            historical_diagnostics_status=historical_diagnostics_status,
            data_scope=data_scope, days_back=days_back,
            odds_mode=odds_mode, model_name=model_name, model_version=effective_model_version,
            staking_mode=staking_mode, min_ev=min_ev, initial_bankroll=initial_bankroll,
            final_bankroll=initial_bankroll, total_bets=0, total_staked=0, total_profit=0,
            temporal_exclusions=temporal_exclusions,
            total_matches=0,
        )

    ids = [int(row["canonical_match_id"]) for row in matches]
    odds_rows: list[dict[str, Any]] = []
    for offset in range(0, len(ids), 500):
        chunk = ids[offset:offset + 500]
        params = {f"id_{i}": value for i, value in enumerate(chunk)}
        placeholders = ",".join(f":id_{i}" for i in range(len(chunk)))
        odds_rows.extend(query_df(db, f"""
            SELECT os.canonical_match_id, os.bookmaker_id, b.name AS bookmaker,
                   os.raw_team_a, os.raw_team_b, os.odds_a, os.odds_b, os.scraped_at
            FROM odds_snapshots os JOIN bookmakers b ON b.id = os.bookmaker_id
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            WHERE os.canonical_match_id IN ({placeholders})
              AND os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a IS NOT NULL AND os.odds_b IS NOT NULL AND os.scraped_at IS NOT NULL
              AND REPLACE(CAST(os.scraped_at AS TEXT), 'T', ' ') < REPLACE(CAST(cm.start_time_normalized AS TEXT), 'T', ' ')
        """, params))

    odds_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in odds_rows:
        odds_by_key[(int(row["canonical_match_id"]), int(row["bookmaker_id"]))].append(row)
    match_by_id = {int(row["canonical_match_id"]): row for row in matches}
    candidates: list[dict[str, Any]] = []
    quote_eligible_match_ids: set[int] = set()
    missing_conformal_match_ids: set[int] = set()
    wide_conformal_match_ids: set[int] = set()
    measured_conformal_match_ids: set[int] = set()
    for (match_id, _bookmaker_id), snapshots in odds_by_key.items():
        match = match_by_id[match_id]
        # Live evaluation requires the prediction's recorded input cutoff and
        # a quote observed after that prediction; other scopes remain marked
        # as non-executable research views.
        available = snapshots
        predicted_at = _parse_time(match.get("predicted_at"))
        data_cutoff_at = _parse_time(match.get("data_cutoff_at"))
        start_at = _parse_time(match.get("start_time_normalized"))
        result_available_at = _parse_time(match.get("result_recorded_at"))
        if (
            start_at is None
            or result_available_at is None
            or result_available_at <= start_at
        ):
            continue
        if data_scope == "live":
            if (
                predicted_at is None
                or data_cutoff_at is None
                or data_cutoff_at > predicted_at
            ):
                continue
            available = [
                row
                for row in snapshots
                if (
                    (quote_at := _parse_time(row.get("scraped_at"))) is not None
                    and predicted_at <= quote_at < start_at
                )
            ]
        else:
            available = [
                row
                for row in snapshots
                if (
                    (quote_at := _parse_time(row.get("scraped_at"))) is not None
                    and quote_at < start_at
                )
            ]
        if available:
            quote_eligible_match_ids.add(match_id)
        entry = _pick_snapshot(available, odds_mode)
        close = _pick_snapshot(available, "close")
        if not entry:
            continue
        aligned = align_snapshot_odds(
            match["team_a_name"] or "", match["team_b_name"] or "",
            entry["raw_team_a"] or "", entry["raw_team_b"] or "", entry["odds_a"], entry["odds_b"],
            league=match.get("league"), source_system="bookmaker", match_date=match.get("start_time_normalized"),
        )
        close_aligned = align_snapshot_odds(
            match["team_a_name"] or "", match["team_b_name"] or "",
            close["raw_team_a"] or "", close["raw_team_b"] or "", close["odds_a"], close["odds_b"],
            league=match.get("league"), source_system="bookmaker", match_date=match.get("start_time_normalized"),
        ) if close else None
        if not aligned:
            continue
        odds_a, odds_b = (none_or_float(aligned[0]), none_or_float(aligned[1]))
        if odds_a is None or odds_b is None or odds_a <= 1 or odds_b <= 1:
            continue
        base_a = none_or_float(match["prob_a"])
        if base_a is None:
            continue
        market_a, market_b = fair_market_probabilities(odds_a, odds_b)
        # A stored hybrid already contains its market shrinkage. Use the
        # selected forecast directly instead of blending it with the quote again.
        model_a = base_a
        model_b = 1 - model_a
        ev_a = expected_value(model_a, odds_a, TAX_RATE)
        ev_b = expected_value(model_b, odds_b, TAX_RATE)

        if use_conformal_gating:
            # A calibrated interval belongs to the direct model output only.
            # Blending it with the contemporaneous market destroys the stated
            # coverage guarantee, so hybrid forecasts cannot pass this gate.
            bounds_a = (
                None
                if use_hybrid
                else conformal_bounds_for_side(match.get("diagnostics_json"), "a")
            )
            bounds_b = (
                None
                if use_hybrid
                else conformal_bounds_for_side(match.get("diagnostics_json"), "b")
            )
            if bounds_a is None or bounds_b is None:
                missing_conformal_match_ids.add(match_id)
                continue
            p_low_a, p_up_a = bounds_a
            p_low_b, p_up_b = bounds_b
            measured_conformal_match_ids.add(match_id)
            if max(p_up_a - p_low_a, p_up_b - p_low_b) > max_conformal_width:
                wide_conformal_match_ids.add(match_id)
                continue
            conf_ev_a = expected_value(p_low_a, odds_a, TAX_RATE)
            conf_ev_b = expected_value(p_low_b, odds_b, TAX_RATE)
            if max(conf_ev_a, conf_ev_b) <= min_ev:
                continue
            side = "a" if conf_ev_a >= conf_ev_b else "b"
            target_prob = p_low_a if side == "a" else p_low_b
            ev = conf_ev_a if side == "a" else conf_ev_b
            row_uncertainty_basis = "measured_bounds"
            row_uncertainty_reason = "persisted_venn_abers"
        else:
            if max(ev_a, ev_b) <= min_ev:
                continue
            side = "a" if ev_a >= ev_b else "b"
            target_prob = model_a if side == "a" else model_b
            ev = ev_a if side == "a" else ev_b
            row_uncertainty_basis = "mean"
            row_uncertainty_reason = "uncertainty_not_requested"

        entry_odds, model_prob, market_prob = (odds_a, model_a, market_a) if side == "a" else (odds_b, model_b, market_b)
        close_odds = None
        if close_aligned:
            close_odds = none_or_float(close_aligned[0] if side == "a" else close_aligned[1])
        entry_at = _parse_time(entry["scraped_at"])
        hours_before = (start_at - entry_at).total_seconds() / 3600 if entry_at else None
        key, label = _horizon(hours_before) if hours_before is not None else ("unknown", "Brak czasu")
        candidates.append({
            "canonical_match_id": match_id, "start_time": match["start_time_normalized"],
            "result_available_at": result_available_at, "league": match.get("league") or "Nieznana liga",
            "team_a_name": match.get("team_a_name"), "team_b_name": match.get("team_b_name"), "bookmaker": entry["bookmaker"],
            "side": side, "entry_odds": entry_odds, "close_odds": close_odds, "hours_before": hours_before,
            "horizon": key, "horizon_label": label, "model_prob": model_prob, "target_prob": target_prob, "market_prob": market_prob, "ev": ev,
            "uncertainty_basis": row_uncertainty_basis,
            "uncertainty_reason": row_uncertainty_reason,
            "won": match["winner_side"] == ("team_a" if side == "a" else "team_b"), "entry_scraped_at": entry.get("scraped_at"),
            "close_scraped_at": close.get("scraped_at") if close else None,
            "clv_odds_pct": (entry_odds / close_odds - 1) if close_odds and close_odds > 0 else None,
        })
    eligible_match_ids = {int(match["canonical_match_id"]) for match in matches}
    no_quote_key = (
        "no_quote_after_prediction"
        if data_scope == "live"
        else "no_quote_before_start"
    )
    temporal_exclusions[no_quote_key] = len(
        eligible_match_ids - quote_eligible_match_ids
    )
    temporal_exclusions["conformal_bounds_unavailable"] = len(
        missing_conformal_match_ids
    )
    temporal_exclusions["conformal_bounds_too_wide"] = len(
        wide_conformal_match_ids
    )
    candidate_match_ids = {
        int(candidate["canonical_match_id"]) for candidate in candidates
    }
    temporal_exclusions["no_positive_ev"] = len(
        quote_eligible_match_ids - candidate_match_ids
        - missing_conformal_match_ids - wide_conformal_match_ids
    )
    if (
        use_conformal_gating
        and measured_conformal_match_ids
        and not missing_conformal_match_ids
    ):
        uncertainty_basis = "measured_bounds"
        uncertainty_reason = "persisted_venn_abers"

    # Select one globally highest-EV bookmaker/side per match, then process
    # quote placement and result availability as separate ledger events.
    candidates_by_match: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_match[int(candidate["canonical_match_id"])].append(candidate)
    candidates = [
        max(
            rows,
            key=lambda row: (
                float(row["ev"]),
                str(row["entry_scraped_at"]),
                str(row["bookmaker"]),
                str(row["side"]),
            ),
        )
        for rows in candidates_by_match.values()
    ]
    candidates.sort(
        key=lambda row: (
            str(row["entry_scraped_at"]),
            int(row["canonical_match_id"]),
        )
    )
    bankroll = initial_bankroll
    reserved = 0.0
    peak = bankroll
    max_drawdown = 0.0
    max_open_stake = 0.0
    max_open_bets = 0
    curve: list[dict[str, float | str | int | None]] = [
        {"index": 0, "bankroll": bankroll, "profit": 0.0, "start_time": None}
    ]
    open_bets: list[tuple[datetime, int, dict[str, Any]]] = []
    event_index = 0

    def settle_next() -> None:
        nonlocal bankroll, reserved, peak, max_drawdown
        _settled_at, _sequence, row = heappop(open_bets)
        stake = float(row["stake"])
        reserved -= stake
        profit = (
            stake * (float(row["entry_odds"]) * (1 - TAX_RATE) - 1)
            if row["won"]
            else -stake
        )
        bankroll += profit
        peak = max(peak, bankroll)
        max_drawdown = max(
            max_drawdown,
            (peak - bankroll) / peak if peak else 0.0,
        )
        row.update({"profit": profit, "bankroll_after": bankroll})
        curve.append(
            {
                "index": len(curve),
                "bankroll": bankroll,
                "profit": profit,
                "start_time": row["start_time"],
                "canonical_match_id": row["canonical_match_id"],
            }
        )

    settled_candidates: list[dict[str, Any]] = []
    for row in candidates:
        placed_at = _parse_time(row["entry_scraped_at"])
        if placed_at is None:
            continue
        while open_bets and open_bets[0][0] <= placed_at:
            settle_next()
        available = max(0.0, bankroll - reserved)
        if staking_mode == "fixed":
            stake = min(fixed_stake, available)
        else:
            target_k_prob = float(row.get("target_prob", row["model_prob"]))
            full_kelly = kelly_fraction(
                target_k_prob,
                float(row["entry_odds"]),
                TAX_RATE,
            )
            stake = min(
                available * max_stake_pct,
                available * full_kelly * kelly_fraction_multiplier,
            )
        if stake <= 0:
            temporal_exclusions["insufficient_available_balance"] += 1
            continue
        reserved += stake
        row["stake"] = stake
        event_index += 1
        heappush(open_bets, (row["result_available_at"], event_index, row))
        settled_candidates.append(row)
        max_open_stake = max(max_open_stake, reserved)
        max_open_bets = max(max_open_bets, len(open_bets))
    while open_bets:
        settle_next()
    candidates = settled_candidates

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {"horizon": defaultdict(list), "bookmaker": defaultdict(list), "league": defaultdict(list)}
    labels: dict[str, str] = {}
    for row in candidates:
        groups["horizon"][str(row["horizon"])].append(row); labels[str(row["horizon"])] = str(row["horizon_label"])
        groups["bookmaker"][str(row["bookmaker"])].append(row)
        groups["league"][str(row["league"])].append(row)
    horizon_buckets = [_bucket(key, label, groups["horizon"].get(key, [])) for key, label, *_ in HORIZONS]
    horizon_buckets += [_bucket("unknown", "Brak czasu", groups["horizon"]["unknown"])] if groups["horizon"].get("unknown") else []
    bookmaker_buckets = sorted((_bucket(key, key, rows) for key, rows in groups["bookmaker"].items()), key=lambda item: item.roi or -999, reverse=True)
    league_buckets = sorted((_bucket(key, key, rows) for key, rows in groups["league"].items()), key=lambda item: item.bets, reverse=True)
    clv = [float(row["clv_odds_pct"]) for row in candidates if row["clv_odds_pct"] is not None]
    ledger = [FinancialLedgerEntry(**row) for row in candidates]
    total_staked = sum(float(row["stake"]) for row in candidates)
    total_profit = bankroll - initial_bankroll
    return FinancialAnalysisResponse(
        methodology=(
            "Historical quote simulation of exact stored forecasts; not recorded "
            "bet issuance or execution. "
            + (
                "Live scope enforces prediction cutoff, quote timing, reserved "
                "capital and post-start result availability."
                if data_scope == "live"
                else "Research-only historical quote selection, not executable live ROI."
            )
        ),
        evaluation_kind="historical_quote_simulation",
        result_status="computed" if candidates else "no_eligible_bets",
        uncertainty_basis=uncertainty_basis,
        uncertainty_reason=uncertainty_reason,
        historical_diagnostics_status=historical_diagnostics_status,
        data_scope=data_scope,
        days_back=days_back,
        odds_mode=odds_mode,
        model_name=model_name,
        model_version=effective_model_version,
        staking_mode=staking_mode,
        min_ev=min_ev,
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        total_bets=len(candidates),
        total_staked=total_staked,
        total_profit=total_profit,
        total_matches=len({int(row["canonical_match_id"]) for row in candidates}),
        roi=total_profit / total_staked if total_staked else None,
        hit_rate=sum(bool(row["won"]) for row in candidates) / len(candidates) if candidates else None,
        max_drawdown_pct=max_drawdown if candidates else None,
        avg_clv_odds_pct=sum(clv) / len(clv) if clv else None,
        positive_clv_rate=sum(value > 0 for value in clv) / len(clv) if clv else None,
        max_open_stake=max_open_stake,
        temporal_exclusions=temporal_exclusions,
        max_open_bets=max_open_bets,
        horizon_buckets=horizon_buckets,
        bookmaker_buckets=bookmaker_buckets,
        league_buckets=league_buckets,
        bankroll_curve=curve,
        ledger=ledger,
    )
