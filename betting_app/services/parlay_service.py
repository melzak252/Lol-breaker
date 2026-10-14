"""Tax-amortized parlay recommendation engine (IDEA-019).

In the Polish regulated betting market (12% turnover tax on stakes):
  Single bet payout = stake * 0.88 * odds_1
  Double parlay payout = stake * 0.88 * (odds_1 * odds_2)

Compounding gross odds before deducting tax pays the tax friction once
instead of twice, reducing the required gross edge from +13.64% down to +6.60%.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import text
from sqlalchemy.orm import Session
from betting_app.core.models import get_active_hybrid
from betting_app.services.prematch_quote_service import (
    load_prematch_quotes, parse_aware_utc, latest_predictions,
    prediction_time_is_actionable, stored_market_context_is_current,
    hybrid_prediction_contract_is_current,
)

from betting_app.api.schemas import (
    ParlayLeg,
    ParlayRecommendation,
    ParlayRecommendationsResponse,
)

ACTIVE_HYBRID = get_active_hybrid()
ACTIVE_HYBRID_NAME = ACTIVE_HYBRID.hybrid_model_name
ACTIVE_HYBRID_VERSION = ACTIVE_HYBRID.hybrid_model_version
POLISH_TAX_RATE = 0.12
TAX_MULTIPLIER = 1.0 - POLISH_TAX_RATE  # 0.88


def calculate_parlay_metrics(
    leg1: ParlayLeg,
    leg2: ParlayLeg,
    tax_rate: float = POLISH_TAX_RATE,
    bankroll: float = 100.0,
) -> dict[str, Any]:
    """Calculate compound economics under an assumed independent-leg probability."""
    tax_mult = 1.0 - tax_rate
    combined_odds = round(leg1.odds * leg2.odds, 3)
    effective_odds = round(combined_odds * tax_mult, 3)
    joint_prob = round(leg1.model_prob * leg2.model_prob, 4)

    # Break-even joint probability after tax: P_be * (O_comb * 0.88) = 1
    breakeven_prob = round(1.0 / (combined_odds * tax_mult), 4) if combined_odds > 0 else 1.0

    # Net EV of 2-leg parlay: P_joint * (O_comb * 0.88) - 1
    ev_parlay = round(joint_prob * combined_odds * tax_mult - 1.0, 4)

    # Counterfactual EV if placed as two equal single bets with tax paid twice
    avg_single_ev = round((leg1.single_ev + leg2.single_ev) / 2.0, 4)
    tax_amortization_gain = round(ev_parlay - avg_single_ev, 4)

    # Fractional Kelly sizing:
    # b = effective net payout per unit staked = effective_odds - 1
    # f* = (P * (b + 1) - 1) / b = EV / b
    b = effective_odds - 1.0
    if b > 0 and ev_parlay > 0:
        full_kelly = ev_parlay / b
        quarter_kelly = round(max(0.0, min(full_kelly * 0.25, 0.10)), 4)
        suggested_stake = round(max(5.0, min(round(bankroll * quarter_kelly, 1), 50.0)), 1)
    else:
        quarter_kelly = 0.0
        suggested_stake = 0.0

    # Confidence badge
    if joint_prob >= 0.55 and ev_parlay >= 0.10:
        confidence_badge = "Wysokie Bezpieczeństwo"
    elif joint_prob >= 0.45 and ev_parlay >= 0.05:
        confidence_badge = "Zbalansowany Dubel"
    else:
        confidence_badge = "Umiarkowane Ryzyko"

    # Polish summary
    summary_pl = (
        f"Kupon AKO amortyzuje 12% podatku: wymagana przewaga na mecz spada z 13.6% do 6.6%. "
        f"Łączny kurs wynosi {combined_odds:.2f} (efektywny po podatku: {effective_odds:.2f}), "
        f"a przy założeniu niezależności prawdopodobieństwo zwycięstwa obu faworytów to {joint_prob * 100:.1f}%. "
        f"Oczekiwana wartość kuponu (EV po podatku) wynosi {ev_parlay * 100:+.1f}%, "
        f"co daje zysk {tax_amortization_gain * 100:+.1f} p.p. z amortyzacji podatku względem dwóch osobnych singli."
    )

    return {
        "combined_odds": combined_odds,
        "effective_odds": effective_odds,
        "joint_prob": joint_prob,
        "breakeven_prob": breakeven_prob,
        "ev": ev_parlay,
        "tax_amortization_gain": tax_amortization_gain,
        "suggested_stake": suggested_stake,
        "quarter_kelly": quarter_kelly,
        "confidence_badge": confidence_badge,
        "summary_pl": summary_pl,
    }


def find_parlay_recommendations(
    db: Session,
    bookmaker: str | None = None,
    max_odds_per_leg: float = 2.20,
    min_prob_per_leg: float = 0.50,
    min_parlay_ev: float = 0.0,
    bankroll: float = 100.0,
    limit: int = 5,
) -> ParlayRecommendationsResponse:
    """Scan upcoming matches and find optimal same-bookmaker 2-leg favorite parlays."""
    now = datetime.now(timezone.utc)
    predictions = [dict(row) for row in db.execute(text("""
        SELECT cp.*, cm.team_a_name, cm.team_b_name, cm.league, cm.start_time_normalized
        FROM canonical_predictions cp JOIN canonical_matches cm ON cm.id=cp.canonical_match_id
        WHERE cp.model_name=:name AND cp.model_version=:version AND cp.prediction_status='active'
        """), {"name": ACTIVE_HYBRID_NAME, "version": ACTIVE_HYBRID_VERSION}).mappings()]
    contexts = load_prematch_quotes(sorted({p["canonical_match_id"] for p in predictions}), decision_at=now, session=db)
    rows = []
    for prediction in latest_predictions(predictions):
        current = contexts[prediction["canonical_match_id"]]
        if not prediction_time_is_actionable(prediction, decision_at=now, start_at=current.start_at,
                                            model_name=ACTIVE_HYBRID_NAME, model_version=ACTIVE_HYBRID_VERSION)[0]:
            continue
        try:
            diag = prediction.get("diagnostics_json")
            diag = json.loads(diag) if isinstance(diag, str) else diag
            if not isinstance(diag, dict) or not isinstance(diag.get("market_context"), dict):
                continue
            saved = diag["market_context"]
            at = parse_aware_utc(saved.get("decision_at"))
            if at is None:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if not hybrid_prediction_contract_is_current(prediction, session=db)[0]:
            continue
        evaluation = load_prematch_quotes([current.canonical_match_id], decision_at=at, session=db)[current.canonical_match_id]
        if not stored_market_context_is_current(saved, evaluation_context=evaluation, current_context=current)[0]:
            continue
        for quote in current.quotes:
            rows.append((current.canonical_match_id, prediction["team_a_name"], prediction["team_b_name"],
                         prediction["league"], current.start_at.isoformat(), quote.bookmaker,
                         quote.odds_a, quote.odds_b, prediction["prob_a"], prediction["prob_b"]))

    # Extract eligible favorite single legs
    candidates_by_bookmaker: dict[str, list[ParlayLeg]] = {}

    for r in rows:
        match_id = r[0]
        team_a = r[1] or "Team A"
        team_b = r[2] or "Team B"
        league = r[3] or "Unknown League"
        start_time = r[4]
        bk = r[5]
        odds_a = float(r[6])
        odds_b = float(r[7])
        p_a = float(r[8])
        p_b = float(r[9])

        if bookmaker and bk.lower() != bookmaker.lower():
            continue

        match_name = f"{team_a} vs {team_b}"

        # Evaluate Side A
        single_ev_a = round(p_a * odds_a * TAX_MULTIPLIER - 1.0, 4)
        if 1.05 <= odds_a <= max_odds_per_leg and p_a >= min_prob_per_leg:
            leg_a = ParlayLeg(
                canonical_match_id=match_id,
                match_name=match_name,
                league=league,
                start_time=start_time,
                side="a",
                team_name=team_a,
                opponent_name=team_b,
                odds=odds_a,
                model_prob=p_a,
                single_ev=single_ev_a,
                is_favorite=True,
            )
            candidates_by_bookmaker.setdefault(bk, []).append(leg_a)

        # Evaluate Side B
        single_ev_b = round(p_b * odds_b * TAX_MULTIPLIER - 1.0, 4)
        if 1.05 <= odds_b <= max_odds_per_leg and p_b >= min_prob_per_leg:
            leg_b = ParlayLeg(
                canonical_match_id=match_id,
                match_name=match_name,
                league=league,
                start_time=start_time,
                side="b",
                team_name=team_b,
                opponent_name=team_a,
                odds=odds_b,
                model_prob=p_b,
                single_ev=single_ev_b,
                is_favorite=True,
            )
            candidates_by_bookmaker.setdefault(bk, []).append(leg_b)

    # Form 2-leg combinations per bookmaker
    parlay_recommendations: list[ParlayRecommendation] = []
    seen_match_pairs: set[tuple[int, int, str]] = set()

    for bk, legs in candidates_by_bookmaker.items():
        n_legs = len(legs)
        for i in range(n_legs):
            for j in range(i + 1, n_legs):
                leg1 = legs[i]
                leg2 = legs[j]

                # Structural exclusions do not establish empirical independence.
                # 1. Must be different matches
                if leg1.canonical_match_id == leg2.canonical_match_id:
                    continue

                # 2. Must not involve the same team twice
                teams_1 = {leg1.team_name.lower(), leg1.opponent_name.lower()}
                teams_2 = {leg2.team_name.lower(), leg2.opponent_name.lower()}
                if teams_1.intersection(teams_2):
                    continue

                # Avoid duplicate pairings across same bookmaker
                pair_key = (min(leg1.canonical_match_id, leg2.canonical_match_id),
                            max(leg1.canonical_match_id, leg2.canonical_match_id),
                            bk)
                if pair_key in seen_match_pairs:
                    continue
                seen_match_pairs.add(pair_key)

                metrics = calculate_parlay_metrics(leg1, leg2, tax_rate=POLISH_TAX_RATE, bankroll=bankroll)

                if metrics["ev"] < min_parlay_ev:
                    continue

                parlay_id = f"parlay_{bk}_{leg1.canonical_match_id}_{leg2.canonical_match_id}"
                rec = ParlayRecommendation(
                    id=parlay_id,
                    bookmaker=bk,
                    legs=[leg1, leg2],
                    combined_odds=metrics["combined_odds"],
                    effective_odds=metrics["effective_odds"],
                    joint_prob=metrics["joint_prob"],
                    breakeven_prob=metrics["breakeven_prob"],
                    ev=metrics["ev"],
                    tax_amortization_gain=metrics["tax_amortization_gain"],
                    suggested_stake=metrics["suggested_stake"],
                    quarter_kelly=metrics["quarter_kelly"],
                    confidence_badge=metrics["confidence_badge"],
                    summary_pl=metrics["summary_pl"],
                )
                parlay_recommendations.append(rec)

    # Sort parlays by risk-adjusted return: EV * joint_prob
    parlay_recommendations.sort(key=lambda p: (p.ev * p.joint_prob, p.ev), reverse=True)

    top_parlay = parlay_recommendations[0] if parlay_recommendations else None
    trimmed_parlays = parlay_recommendations[:limit]

    explanation = (
        "Zgodnie z polskim prawem od każdego zakładu bukmacherskiego pobierany jest 12% podatek obrotowy. "
        "W zakładach pojedynczych (singlach) na faworytów podatek tworzy barierę wymagającą aż +13.6% czystej "
        "przewagi modelu (edge) do osiągnięcia progu rentowności. W kuponach akumulowanych (duble / AKO) podatek "
        "odliczany jest jednokrotnie od stawki całego kuponu, co redukuje wymaganą przewagę na mecz o ponad połowę "
        "(do ok. +6.6%). Rekomendator parlay dobiera wyłącznie sprawdzonych faworytów oferowanych u tego samego bukmachera. "
        "Iloczyn prawdopodobieństw zakłada niezależność wyników; brak wspólnych drużyn nie potwierdza "
        "empirycznej niezależności meczów."
    )

    return ParlayRecommendationsResponse(
        count=len(parlay_recommendations),
        top_parlay=top_parlay,
        parlays=trimmed_parlays,
        tax_rate=POLISH_TAX_RATE,
        explanation=explanation,
    )
