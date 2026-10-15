#!/usr/bin/env python3
"""CLI utility for rapid pre-match predictions using native Causal-C0 and Hybrid-50.

Usage:
    .venv/bin/python scripts/predict_matchup.py C9 KC
    .venv/bin/python scripts/predict_matchup.py "Cloud9" "Karmine Corp" --bo 5
    .venv/bin/python scripts/predict_matchup.py T1 GenG --bo 3 --odds1 2.40 --odds2 1.55
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure ensemble research root points to consolidated benchmark if not set
if "ENSEMBLE_RESEARCH_ROOT" not in os.environ:
    os.environ["ENSEMBLE_RESEARCH_ROOT"] = str(PROJECT_ROOT / "data/05_model_input/canonical_benchmark")

from scipy.special import expit, logit

from betting_app.services.c0_inference import close_c0_inference_worker, predict_c0

COMMON_ALIASES: dict[str, str] = {
    "c9": "cloud9",
    "kc": "karmine corp",
    "t1": "t1",
    "g2": "g2 esports",
    "fnc": "fnatic",
    "gen": "gen.g",
    "geng": "gen.g",
    "hle": "hanwha life esports",
    "dk": "dplus kia",
    "wbg": "weibo gaming",
    "blg": "bilibili gaming",
    "tes": "top esports",
    "fly": "flyquest",
    "tl": "team liquid",
    "mroi": "movistar koi",
    "koi": "movistar koi",
    "gx": "giantx",
    "bds": "team bds",
    "mad": "mad lions",
    "sk": "sk gaming",
    "vitality": "team vitality",
    "vit": "team vitality",
    "100": "100 thieves",
    "100t": "100 thieves",
    "dignitas": "dignitas",
    "dig": "dignitas",
    "nrg": "nrg",
    "sr": "shopify rebellion",
    "lyon": "lyon gaming",
}


def load_raw_matches() -> list[dict]:
    history_path = PROJECT_ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing canonical raw history at {history_path}")
    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_team(query: str, raw_matches: list[dict]) -> dict:
    q = query.lower().strip()
    target = COMMON_ALIASES.get(q, q)

    # 1. Exact match pass
    exact_matches = [
        m for m in raw_matches
        if (m.get("t1_name") or "").lower().strip() == target
        or (m.get("t2_name") or "").lower().strip() == target
    ]

    matches = exact_matches
    if not matches:
        # 2. Substring match, filtering academy/challengers unless explicitly queried
        sub_matches = []
        for m in raw_matches:
            t1 = (m.get("t1_name") or "").lower()
            t2 = (m.get("t2_name") or "").lower()
            if target in t1 or target in t2:
                is_sub = any(k in t1 or k in t2 for k in ("academy", "challengers", "blue", "amateur"))
                if not is_sub or any(k in target for k in ("academy", "challengers", "blue", "amateur")):
                    sub_matches.append(m)
        matches = sub_matches if sub_matches else [
            m for m in raw_matches
            if target in (m.get("t1_name") or "").lower()
            or target in (m.get("t2_name") or "").lower()
        ]

    if not matches:
        raise ValueError(f"Could not resolve any match history for team query '{query}'")

    matches.sort(key=lambda x: x.get("date", ""))
    latest = matches[-1]

    t1_low = (latest.get("t1_name") or "").lower()
    t2_low = (latest.get("t2_name") or "").lower()
    if t1_low == target:
        side = "t1"
    elif t2_low == target:
        side = "t2"
    elif target in t1_low:
        side = "t1"
    else:
        side = "t2"

    team_name = latest.get(f"{side}_name")
    team_id = str(latest.get(f"{side}_id"))
    last_game = latest["games"][-1]
    players_dict = last_game.get(f"{side}_players", {})

    player_names = {role: p.get("player_name") for role, p in players_dict.items()}
    player_ids = [str(p.get("player_id")) for p in players_dict.values()]

    return {
        "name": team_name,
        "id": team_id,
        "last_played_date": latest.get("date"),
        "tournament": latest.get("tournament_name"),
        "players": player_names,
        "player_ids": player_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast pre-match prediction for any League of Legends matchup using Causal-C0 & Hybrid-50.",
    )
    parser.add_argument("team1", help="Team 1 name or alias (e.g. 'C9', 'Cloud9', 'T1')")
    parser.add_argument("team2", help="Team 2 name or alias (e.g. 'KC', 'Karmine Corp', 'GenG')")
    parser.add_argument("--bo", type=int, choices=[1, 3, 5], default=3, help="Best-of series format (default: 3)")
    parser.add_argument("--odds1", type=float, default=None, help="Decimal odds for Team 1 (e.g. 2.10)")
    parser.add_argument("--odds2", type=float, default=None, help="Decimal odds for Team 2 (e.g. 1.70)")
    parser.add_argument("--all-bo", action="store_true", help="Print win probabilities for BO1, BO3, and BO5")

    args = parser.parse_args()

    print("=" * 78)
    print("LoL-Breaker Matchup Predictor (Causal-C0 Native Serving Engine)")
    print("=" * 78)

    raw_matches = load_raw_matches()
    t1_info = resolve_team(args.team1, raw_matches)
    t2_info = resolve_team(args.team2, raw_matches)

    print(f"\nResolved Teams:")
    print(f"  Team A: {t1_info['name']} (ID: {t1_info['id']}, Last Match: {t1_info['last_played_date']})")
    print(f"          Roster: {', '.join(f'{r}: {n}' for r, n in t1_info['players'].items())}")
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    decision_iso = now.isoformat()
    start_iso = (now + timedelta(hours=2)).isoformat()
    formats_to_evaluate = [1, 3, 5] if args.all_bo else [args.bo]

    predictions: dict[int, Any] = {}
    for bo in formats_to_evaluate:
        req = {
            "c0_request": {
                "team1_id": t1_info["id"],
                "team2_id": t2_info["id"],
                "roster_a": t1_info["player_ids"],
                "roster_b": t2_info["player_ids"],
                "best_of": bo,
                "decision_at": decision_iso,
                "start_at": start_iso,
                "competition_context": "Worlds",
                "mode": "full",
            },
            "canonical": {"id": None},
        }
        pred = predict_c0(req)
        predictions[bo] = pred

    primary_bo = args.bo
    primary_pred = predictions[primary_bo]
    p_c0_a = primary_pred.prob_a
    p_c0_b = primary_pred.prob_b

    print(f"\n--- Model Forecast (BO{primary_bo} Series) ---")
    print(f"  {t1_info['name']:<20} Win Prob: {p_c0_a*100:6.2f}%  (Fair Decimal Odds: {1.0/p_c0_a:5.2f})")
    print(f"  {t2_info['name']:<20} Win Prob: {p_c0_b*100:6.2f}%  (Fair Decimal Odds: {1.0/p_c0_b:5.2f})")

    favorite = t1_info['name'] if p_c0_a >= p_c0_b else t2_info['name']
    fav_prob = max(p_c0_a, p_c0_b)
    print(f"\nModel Favorite: {favorite} ({fav_prob*100:.2f}%) | Predicted Winner: {favorite}")

    if args.all_bo:
        print("\n--- Cross-Format Projection Comparison ---")
        print("| Format | Team A Win Prob | Team B Win Prob | Margin (Edge) |")
        print("|---|---:|---:|---:|")
        for bo in [1, 3, 5]:
            pa = predictions[bo].prob_a
            pb = predictions[bo].prob_b
            edge_str = f"{favorite} +{abs(pa - pb)*100:.2f} pp"
            print(f"| Best-of-{bo} | {t1_info['name']}: {pa*100:.2f}% | {t2_info['name']}: {pb*100:.2f}% | {edge_str} |")

    # If bookmaker odds are provided, compute Shin de-vig, Hybrid-50, and 12% Polish tax EV
    if args.odds1 is not None and args.odds2 is not None:
        o1, o2 = args.odds1, args.odds2
        inv1, inv2 = 1.0 / o1, 1.0 / o2
        overround = (inv1 + inv2) - 1.0
        q1 = max(1e-6, min(1.0 - 1e-6, inv1 - overround / 2.0))
        q2 = 1.0 - q1

        # Hybrid-50 logit shrinkage
        l_c0 = logit(max(1e-6, min(1.0 - 1e-6, p_c0_a)))
        l_mkt = logit(q1)
        p_hybrid_a = float(expit(0.50 * l_c0 + 0.50 * l_mkt))
        p_hybrid_b = 1.0 - p_hybrid_a

        # Polish tax multiplier c = 0.88
        tax_mult = 0.88
        ev_c0_a = (p_c0_a * tax_mult * o1) - 1.0
        ev_c0_b = (p_c0_b * tax_mult * o2) - 1.0
        ev_hyb_a = (p_hybrid_a * tax_mult * o1) - 1.0
        ev_hyb_b = (p_hybrid_b * tax_mult * o2) - 1.0

        print("\n--- Market & Betting Evaluation (12% Polish Turnover Tax) ---")
        print(f"Offered Odds:     {t1_info['name']} @ {o1:.2f}  |  {t2_info['name']} @ {o2:.2f} (Margin: {overround*100:.2f}%)")
        print(f"De-vigged Market: {t1_info['name']} {q1*100:5.2f}%  |  {t2_info['name']} {q2*100:5.2f}%")
        print(f"Hybrid-50 Shrink: {t1_info['name']} {p_hybrid_a*100:5.2f}%  |  {t2_info['name']} {p_hybrid_b*100:5.2f}%")
        print(f"Pure C0 EV (Net): {t1_info['name']} {ev_c0_a*100:+5.2f}%  |  {t2_info['name']} {ev_c0_b*100:+5.2f}%")
        print(f"Hybrid-50 EV:     {t1_info['name']} {ev_hyb_a*100:+5.2f}%  |  {t2_info['name']} {ev_hyb_b*100:+5.2f}%")

        # Operational decision gates
        divergence = abs(l_c0 - l_mkt)
        div_gate_ok = divergence <= 0.85
        best_ev = max(ev_hyb_a, ev_hyb_b)
        best_side = t1_info['name'] if ev_hyb_a >= ev_hyb_b else t2_info['name']
        best_odds = o1 if ev_hyb_a >= ev_hyb_b else o2
        best_p_hyb = p_hybrid_a if ev_hyb_a >= ev_hyb_b else p_hybrid_b

        # Wasserstein Robust Gate (eps = 0.025)
        dro_ev = ((best_p_hyb - 0.025) * tax_mult * best_odds) - 1.0

        print(f"\nSafety Gates Check:")
        print(f"  Market-Divergence-Gate (|logit diff| <= 0.85): {divergence:.4f} -> {'PASS' if div_gate_ok else 'VETO'}")
        print(f"  Wasserstein-Robust-Gate (eps = 0.025):        {dro_ev*100:+.2f}% -> {'PASS' if dro_ev > 0 else 'VETO'}")

        if div_gate_ok and dro_ev > 0 and 0.05 <= best_ev <= 0.25 and best_odds <= 4.0:
            print(f"\n>>> ADOPTED BET RECOMMENDATION: {best_side} @ {best_odds:.2f} (Net EV: {best_ev*100:+.2f}%, 1/4 Kelly, Max 150 PLN) <<<")
        else:
            print(f"\n>>> NO BET: Opportunity does not satisfy strict production risk gates. <<<")

    print("\n" + "=" * 78)
    close_c0_inference_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
