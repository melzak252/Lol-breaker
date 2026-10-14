"""Run the native-C0 upcoming prediction pipeline.

Pure C0 forecasts are persisted first; optional market shrinkage consumes that
saved C0 forecast once. EXP-039 remains a separately callable frozen thesis model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from collections import Counter

from betting_app.core.db import init_db, query_df
from betting_app.services.thesis_inference_service import (
    THESIS_HYBRID_ALPHA,
    THESIS_HYBRID_MODEL_NAME,
    THESIS_HYBRID_TEMPERATURE,
    generate_thesis_hybrid_predictions,
    predict_upcoming_with_thesis_model,
)
from betting_app.core.models import (
    get_active_hybrid,
    get_active_model,
    get_model,
    list_registered_models,
    set_active_model,
)
from betting_app.services.upcoming_inference_service import (
    build_all_upcoming_features,
    generate_hybrid_predictions,
    generate_model_ev_signals,
    predict_all_upcoming,
)
from betting_app.services.bet_qualification_service import DEFAULT_BASE_MIN_EV

DEFAULT_FEATURE_VERSION = get_active_model().feature_version
DEFAULT_RATINGS_VERSION = get_active_model().ratings_version
DEFAULT_W20_VERSION = get_active_model().w20_version
DEFAULT_MODEL_NAME = get_active_model().name
DEFAULT_MODEL_VERSION = get_active_model().version
DEFAULT_HYBRID_MODEL_NAME = get_active_hybrid().hybrid_model_name
DEFAULT_HYBRID_ALPHA = get_active_hybrid().alpha
DEFAULT_HYBRID_TEMPERATURE = get_active_hybrid().temperature

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--ratings-version", default=DEFAULT_RATINGS_VERSION)
    parser.add_argument("--w20-version", default=DEFAULT_W20_VERSION)
    parser.add_argument(
        "--model",
        default=None,
        help="Select an exact registered model version. Defaults to active native C0.",
    )
    parser.add_argument(
        "--operational-hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate separate saved-C0 market shrinkage predictions (default: True).",
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="Run the frozen Sym-Cal LR-ElasticNet-W20-Binomial model.",
    )
    parser.add_argument(
        "--thesis-hybrid",
        action="store_true",
        help="Generate thesis+market hybrid predictions.",
    )
    parser.add_argument(
        "--thesis-hybrid-alpha", type=float, default=THESIS_HYBRID_ALPHA
    )
    parser.add_argument(
        "--thesis-hybrid-temperature", type=float, default=THESIS_HYBRID_TEMPERATURE
    )
    parser.add_argument("--tax-rate", type=float, default=0.12)
    parser.add_argument("--min-ev", type=float, default=DEFAULT_BASE_MIN_EV)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--include-past", action="store_true")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Also invoke inference for rows marked partial (usually fails closed on exact C0 input gaps).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--signals-limit", type=int, default=15)
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Scan and dispatch Value Bet alerts via Discord/Telegram if new EV signals exist.",
    )
    args = parser.parse_args()

    if args.model:
        spec = get_model(args.model)
        if spec is None:
            valid_names = [m["name"] for m in list_registered_models()]
            parser.error(f"Unknown model '{args.model}'. Registered models: {valid_names}")
        set_active_model(spec)
        print(f"Selected model: {spec.name} ({spec.version})")
    init_db()
    features = build_all_upcoming_features(
        feature_version=args.feature_version,
        ratings_version=args.ratings_version,
        w20_version=args.w20_version,
        include_past=args.include_past,
        limit=args.limit,
    )
    feature_counts = Counter(row["status"] for row in features)
    print(f"Features: {len(features)} | {dict(feature_counts)}")

    operational_preds = predict_all_upcoming(
        feature_version=args.feature_version,
        ratings_version=args.ratings_version,
        include_partial=args.include_partial,
    )
    print(f"Operational predictions: {len(operational_preds)}")

    active_model = get_active_model()
    active_hybrid = get_active_hybrid()
    ev_model_name = active_model.name
    ev_model_version = active_model.version
    if args.operational_hybrid:
        if active_model.family == "bayesian_market_hybrid":
            print(
                f"Active model '{active_model.name}' is already an operational Bayesian Shrunk hybrid "
                f"({active_model.version}) with integrated market odds. Preserving active model version."
            )
        else:
            operational_hybrid_version = active_hybrid.hybrid_model_version
            operational_hybrid_preds = generate_hybrid_predictions(
                base_model_name=active_model.name,
                base_model_version=active_model.version,
                alpha=active_hybrid.alpha,
                temperature=active_hybrid.temperature,
                hybrid_model_name=active_hybrid.hybrid_model_name,
                hybrid_model_version=operational_hybrid_version,
                blending_mode=active_hybrid.blending_mode,
            )
            print(
                "Operational hybrid predictions: "
                f"{len(operational_hybrid_preds)} | "
                f"alpha={active_hybrid.alpha:.2f} T={active_hybrid.temperature:.2f}"
            )
            ev_model_name = active_hybrid.hybrid_model_name
            ev_model_version = operational_hybrid_version
    if args.thesis:
        thesis_preds = predict_upcoming_with_thesis_model(
            ratings_version=args.ratings_version,
            w20_version=args.w20_version,
            include_past=args.include_past,
            limit=args.limit,
        )
        print(f"Thesis predictions: {len(thesis_preds)}")

    if args.thesis_hybrid:
        thesis_hybrid_version = (
            f"a{args.thesis_hybrid_alpha:.2f}"
            f"-t{args.thesis_hybrid_temperature:.2f}"
        )
        thesis_hybrid_preds = generate_thesis_hybrid_predictions(
            alpha=args.thesis_hybrid_alpha,
            temperature=args.thesis_hybrid_temperature,
            hybrid_model_version=thesis_hybrid_version,
        )
        print(
            f"Thesis hybrid predictions: {len(thesis_hybrid_preds)} | "
            f"alpha={args.thesis_hybrid_alpha:.2f} "
            f"T={args.thesis_hybrid_temperature:.2f}"
        )
        ev_model_name = THESIS_HYBRID_MODEL_NAME
        ev_model_version = thesis_hybrid_version

    signals, stats = generate_model_ev_signals(
        model_name=ev_model_name,
        model_version=ev_model_version,
        tax_rate=args.tax_rate,
        min_ev=args.min_ev,
        bankroll=args.bankroll,
        check_confirmed_lineups=True,
        return_stats=True,
    )
    print(
        f"\nEV signals: {len(signals)} | tax={args.tax_rate:.2%} | "
        f"min_ev={args.min_ev:.2%}"
    )
    print("\n--- Diagnostic Qualification Breakdown ---")
    print(f"Total fixtures evaluated: {stats['total_fixtures_evaluated']}")
    print(f"Total predictions written/active: {len(operational_preds)}")
    print(f"Candidate bet sides evaluated: {stats['candidate_sides_evaluated']}")
    print(f"Value bets qualified: {stats['value_bets_qualified']}")
    print(f"Total bets quarantined: {stats['total_quarantined']}")
    if stats.get('quarantined_breakdown'):
        print("Quarantined reasons:")
        for reason, count in stats['quarantined_breakdown'].items():
            print(f"  - {reason}: {count}")
    if stats.get('disqualified_breakdown'):
        print("Disqualified reasons (non-quarantine):")
        for reason, count in stats['disqualified_breakdown'].items():
            print(f"  - {reason}: {count}")
    print("------------------------------------------\n")
    for row in signals[: args.signals_limit]:
        side = "A" if row["side"] == "a" else "B"
        print(
            f"#{row['canonical_match_id']} {row['match']} | side={side} | "
            f"{row['bookmaker']} odds={row['odds']:.2f} "
            f"p={row['model_prob']:.3f} EV={row['ev']:.2%} "
            f"stake={row['stake_suggestion']:.2f}"
        )


    if args.notify:
        from betting_app.core.db import get_session
        from betting_app.services.alert_service import scan_and_dispatch_ev_alerts
        print("\nScanning and dispatching Value Bet alerts...")
        with get_session() as session:
            alert_res = scan_and_dispatch_ev_alerts(session)
            print(f"Alerts summary: {alert_res.get('message')}")
    print_readiness_counts()


def print_readiness_counts() -> None:
    counts = query_df(
        """
        SELECT 'canonical_matches' AS table_name, COUNT(*) AS rows FROM canonical_matches
        UNION ALL SELECT 'upcoming_match_features', COUNT(*) FROM upcoming_match_features
        UNION ALL SELECT 'canonical_predictions', COUNT(*) FROM canonical_predictions
        UNION ALL SELECT 'model_ev_signals', COUNT(*) FROM model_ev_signals
        """
    )
    print("\nDB counts:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
