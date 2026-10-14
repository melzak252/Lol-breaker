from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect, init_db
from betting_app.scripts.rebuild_ratings import (
    MatchForRatings,
    RATING_SYSTEM_PARAMS,
    game_score_for_match_team1,
    stable_team_id,
)
from betting_app.scripts.rebuild_w20_features import (
    average_history,
    load_all_games_grouped,
    load_all_player_stats_grouped,
    update_team_history,
)
from betting_app.services.thesis_inference_service import (
    ALL_FEATURES,
    BINOMIAL_FEATURES,
    EPSILON,
    OPTUNA_BASE_FEATURES,
    RANK_PROB_FEATURES,
    ROLLING_FULL_FEATURES,
    _logit,
    _series_probability,
    _swap_feature_vector,
    _symmetrize,
)
from src.ratings.manager import RatingManager

PIPELINE_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def safe_metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1 - EPSILON)
    y = np.asarray(y, dtype=int)
    out = {
        "n": int(len(y)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "mean_prob": float(np.mean(p)),
        "target_rate": float(np.mean(y)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = None
    return out




def load_backtest_matches(limit: int | None = None, after_date: str | None = None) -> list[MatchForRatings]:
    """Load DB GOL.GG matches for historical prediction.

    Unlike rebuild_ratings.load_matches(), this uses the first golgg_games row as
    the source of side/team identity and loads first-game rosters by side=t1/t2.
    Recent direct GOL.GG imports often have empty golgg_matches.team*_id or alias
    team names, while golgg_games/golgg_game_players contain the correct side IDs.
    """
    query = """
        SELECT match_id, date
        FROM golgg_matches
        WHERE COALESCE(draw, 0) = 0
          AND date IS NOT NULL
    """
    params: list[Any] = []
    if after_date:
        query += " AND date >= ?"
        params.append(after_date)
    query += " ORDER BY date ASC, CAST(match_id AS INTEGER) ASC"
    if limit:
        query += f" LIMIT {int(limit)}"

    matches: list[MatchForRatings] = []
    with connect() as connection:
        match_rows = connection.execute(query, params).fetchall()
        for row in match_rows:
            match_id = str(row["match_id"])
            games = [dict(g) for g in connection.execute(
                """
                SELECT game_id, team1_id, team2_id, team1_name, team2_name,
                       team1_win, team2_win, draw
                FROM golgg_games
                WHERE match_id = ?
                ORDER BY CAST(game_id AS INTEGER) ASC
                """,
                (match_id,),
            ).fetchall()]
            if not games:
                continue
            first = games[0]
            team1_id = stable_team_id(first.get("team1_id"), first.get("team1_name"))
            team2_id = stable_team_id(first.get("team2_id"), first.get("team2_name"))
            players1 = load_first_game_players_by_side(connection, match_id, "t1")
            players2 = load_first_game_players_by_side(connection, match_id, "t2")
            if not players1 or not players2:
                continue
            matches.append(MatchForRatings(
                match_id=match_id,
                match_date=datetime.fromisoformat(str(row["date"])).date(),
                team1_id=team1_id,
                team2_id=team2_id,
                team1_name=str(first.get("team1_name") or team1_id),
                team2_name=str(first.get("team2_name") or team2_id),
                games=games,
                players1=players1,
                players2=players2,
            ))
    return matches


def load_first_game_players_by_side(connection, match_id: str, side: str) -> list[str]:
    first_game = connection.execute(
        """
        SELECT game_id
        FROM golgg_game_players
        WHERE match_id = ? AND side = ?
        ORDER BY CAST(game_id AS INTEGER) ASC
        LIMIT 1
        """,
        (match_id, side),
    ).fetchone()
    if not first_game:
        return []
    rows = connection.execute(
        """
        SELECT player_id, player_name
        FROM golgg_game_players
        WHERE match_id = ? AND game_id = ? AND side = ?
        ORDER BY role, player_name
        """,
        (match_id, first_game["game_id"], side),
    ).fetchall()
    return [str(r["player_id"] or r["player_name"]) for r in rows]

def load_best_of_map() -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute("SELECT match_id, best_of, games_played FROM golgg_matches").fetchall()
    out: dict[str, int] = {}
    for r in rows:
        bo = r["best_of"] if r["best_of"] is not None else r["games_played"]
        try:
            bo_i = int(bo or 1)
        except Exception:
            bo_i = 1
        if bo_i not in (1, 3, 5):
            # Fall back to the next odd BO that can represent observed games.
            bo_i = 1 if bo_i <= 1 else (3 if bo_i <= 3 else 5)
        out[str(r["match_id"])] = bo_i
    return out


def build_exp039_predictions(min_date: str | None, limit: int | None) -> pd.DataFrame:
    pipeline = joblib.load(PIPELINE_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    best_of_map = load_best_of_map()
    matches = load_backtest_matches(limit=limit, after_date=min_date)
    print(f"Loaded eligible DB GOL.GG matches: {len(matches)}")

    games_by_match = load_all_games_grouped()
    player_stats_by_game_side = load_all_player_stats_grouped()

    manager = RatingManager(RATING_SYSTEM_PARAMS)
    team_history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=20))
    team_match_ids: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))
    rows: list[dict[str, Any]] = []

    for match in tqdm(matches, desc="EXP-039 DB feature/prediction pass"):
        if not match.players1 or not match.players2 or not match.games:
            continue
        manager.update_before_match(match.team1_id, match.team2_id, match.players1, match.players2, match.match_date)
        ratings = manager.predict_match(match.team1_id, match.team2_id, match.players1, match.players2)
        t1_hist = average_history(team_history[match.team1_id])
        t2_hist = average_history(team_history[match.team2_id])

        features: dict[str, float] = {}
        for f in OPTUNA_BASE_FEATURES:
            features[f] = float(ratings.get(f, 0.5))
        for stat in ["win_rate", "kills", "deaths", "gd15", "dpm", "vspm", "towers", "nashors", "gold", "duration"]:
            features[f"t1_rolling_{stat}"] = float(t1_hist[stat])
            features[f"t2_rolling_{stat}"] = float(t2_hist[stat])
        best_of = best_of_map.get(match.match_id, 1)
        rating_probs_array = np.array([features[f] for f in RANK_PROB_FEATURES], dtype=float)
        series_probs = _series_probability(rating_probs_array, best_of)
        for i, feature in enumerate(BINOMIAL_FEATURES):
            features[feature] = float(series_probs[i])

        vec = np.array([[features.get(f, 0.0) for f in ALL_FEATURES]], dtype=float)
        original_prob = float(np.clip(pipeline.predict_proba(vec)[0, 1], EPSILON, 1 - EPSILON))
        swapped_prob = float(np.clip(pipeline.predict_proba(_swap_feature_vector(vec))[0, 1], EPSILON, 1 - EPSILON))
        sym_prob = float(np.clip(_symmetrize(original_prob, swapped_prob), EPSILON, 1 - EPSILON))
        cal_prob = float(np.clip(calibrator.predict_proba(_logit(np.array([sym_prob])))[0, 1], EPSILON, 1 - EPSILON))

        scores = [game_score_for_match_team1(match, g) for g in match.games]
        y_team1 = int(sum(scores) > len(scores) / 2)
        rows.append({
            "golgg_match_id": match.match_id,
            "date": match.match_date.isoformat(),
            "team1_id": match.team1_id,
            "team2_id": match.team2_id,
            "team1_name": match.team1_name,
            "team2_name": match.team2_name,
            "best_of": best_of,
            "n_games": len(scores),
            "y_team1": y_team1,
            "exp039_original_prob_team1": original_prob,
            "exp039_swapped_prob_team1": swapped_prob,
            "exp039_symmetric_prob_team1": sym_prob,
            "exp039_calibrated_prob_team1": cal_prob,
        })

        for g in match.games:
            score_1 = game_score_for_match_team1(match, g)
            manager.update_after_game(match.team1_id, match.team2_id, match.players1, match.players2, score_1, 1 - score_1)
        manager.update_after_match(match.team1_id, match.team2_id, match.players1, match.players2, scores)

        for g in games_by_match.get(str(match.match_id), []):
            update_team_history(team_history, team_match_ids, match.team1_id, match.match_id, g, player_stats_by_game_side)
            update_team_history(team_history, team_match_ids, match.team2_id, match.match_id, g, player_stats_by_game_side)

    return pd.DataFrame(rows)


def load_canonical_market_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    with connect() as conn:
        mapped = conn.execute(
            """
            SELECT cm.id AS canonical_match_id,
                   COALESCE(gm.golgg_match_id, cm.result_source_match_id) AS golgg_match_id,
                   cm.team_a_name, cm.team_b_name, cm.winner_side,
                   cm.start_time_normalized, cm.status
            FROM canonical_matches cm
            LEFT JOIN golgg_match_mappings gm ON gm.canonical_match_id = cm.id
            WHERE COALESCE(gm.golgg_match_id, cm.result_source_match_id) IS NOT NULL
              AND cm.winner_side IN ('team_a','team_b')
            """
        ).fetchall()
        odds = conn.execute(
            """
            SELECT canonical_match_id, bookmaker_id, odds_a, odds_b, scraped_at
            FROM odds_snapshots
            WHERE canonical_match_id IS NOT NULL
              AND market_type = 'match_winner'
              AND COALESCE(is_live, 0) = 0
              AND odds_a IS NOT NULL AND odds_b IS NOT NULL
              AND odds_a > 1.0 AND odds_b > 1.0
            ORDER BY canonical_match_id, bookmaker_id, scraped_at
            """
        ).fetchall()
    return pd.DataFrame([dict(r) for r in mapped]), pd.DataFrame([dict(r) for r in odds])


def aggregate_market(mapped: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if mapped.empty or odds.empty:
        return pd.DataFrame()
    mapped = mapped.copy()
    mapped["canonical_match_id"] = mapped["canonical_match_id"].astype(int)
    mapped["start_dt"] = mapped["start_time_normalized"].map(parse_dt)
    start_by_id = dict(zip(mapped["canonical_match_id"], mapped["start_dt"]))

    odds = odds.copy()
    odds["canonical_match_id"] = odds["canonical_match_id"].astype(int)
    odds["scraped_dt"] = odds["scraped_at"].map(parse_dt)
    odds = odds[odds["scraped_dt"].notna()].copy()
    # Use pre-match odds only when start time is known. Unknown start times are retained.
    odds["start_dt"] = odds["canonical_match_id"].map(start_by_id)
    odds = odds[(odds["start_dt"].isna()) | (odds["scraped_dt"] <= odds["start_dt"])].copy()
    if odds.empty:
        return pd.DataFrame()
    odds["imp_a_raw"] = 1.0 / odds["odds_a"].astype(float)
    odds["imp_b_raw"] = 1.0 / odds["odds_b"].astype(float)
    odds["p_a_novig"] = odds["imp_a_raw"] / (odds["imp_a_raw"] + odds["imp_b_raw"])
    odds["margin"] = odds["imp_a_raw"] + odds["imp_b_raw"] - 1.0

    rows: list[dict[str, Any]] = []
    for (cid, bid), g in odds.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker_id"], sort=False):
        n = len(g)
        if n == 0:
            continue
        open_row = g.iloc[0]
        mid_row = g.iloc[(n - 1) // 2]
        close_row = g.iloc[-1]
        for label, row in [("open", open_row), ("mid", mid_row), ("close", close_row)]:
            rows.append({
                "canonical_match_id": int(cid),
                "bookmaker_id": int(bid) if pd.notna(bid) else None,
                "snapshot_type": label,
                "scraped_at": row["scraped_dt"].isoformat(),
                "p_a_raw": float(row["imp_a_raw"]),
                "p_a_novig": float(row["p_a_novig"]),
                "margin": float(row["margin"]),
                "odds_a": float(row["odds_a"]),
                "odds_b": float(row["odds_b"]),
                "n_snapshots_bookmaker": int(n),
            })
    selected = pd.DataFrame(rows)
    wide_rows: list[dict[str, Any]] = []
    for (cid, typ), g in selected.groupby(["canonical_match_id", "snapshot_type"]):
        wide_rows.append({
            "canonical_match_id": int(cid),
            f"market_{typ}_p_a_novig": float(g["p_a_novig"].mean()),
            f"market_{typ}_p_a_raw": float(g["p_a_raw"].mean()),
            f"market_{typ}_avg_margin": float(g["margin"].mean()),
            f"market_{typ}_books": int(g["bookmaker_id"].nunique()),
        })
    wide = pd.DataFrame(wide_rows)
    if wide.empty:
        return wide
    out = None
    for typ in ["open", "mid", "close"]:
        part = wide[wide[f"market_{typ}_p_a_novig"].notna()] if f"market_{typ}_p_a_novig" in wide else pd.DataFrame()
        if part.empty:
            continue
        cols = ["canonical_match_id"] + [c for c in part.columns if c.startswith(f"market_{typ}_")]
        part = part[cols].drop_duplicates("canonical_match_id")
        out = part if out is None else out.merge(part, on="canonical_match_id", how="outer")
    if out is None:
        return pd.DataFrame()
    return mapped.drop(columns=["start_dt"]).merge(out, on="canonical_match_id", how="inner")


def evaluate_all(preds: pd.DataFrame, market: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": "Sym-Cal LR-ElasticNet-W20-Binomial/exp-039 final artifacts",
        "prediction_note": "Point-in-time DB features are recomputed chronologically, then final serialized EXP-039 pipeline/calibrator is applied. This is not an OOF retraining backtest.",
        "market_note": "Opening/mid/close use non-live odds_snapshots before canonical start time; per-bookmaker no-vig probabilities are averaged. Mid is the middle available pre-match snapshot per bookmaker.",
    }
    report["exp039_all_golgg"] = {
        "n_predictions": int(len(preds)),
        "date_min": str(preds["date"].min()) if not preds.empty else None,
        "date_max": str(preds["date"].max()) if not preds.empty else None,
        "metrics_team1": safe_metric(preds["y_team1"].values, preds["exp039_calibrated_prob_team1"].values) if not preds.empty else None,
    }
    if not preds.empty:
        pred_dates = pd.to_datetime(preds["date"])
        segments: dict[str, Any] = {}
        for label, cutoff in [
            ("since_2020", "2020-01-01"),
            ("since_2021", "2021-01-01"),
            ("since_2024", "2024-01-01"),
            ("since_2026", "2026-01-01"),
        ]:
            sub = preds[pred_dates >= pd.Timestamp(cutoff)]
            if len(sub):
                segments[label] = {
                    "date_min": str(sub["date"].min()),
                    "date_max": str(sub["date"].max()),
                    "metrics_team1": safe_metric(sub["y_team1"].values, sub["exp039_calibrated_prob_team1"].values),
                }
        report["exp039_segments"] = segments
    joined = market.merge(preds, on="golgg_match_id", how="inner")
    if joined.empty:
        report["market_common"] = {"n": 0}
        return report, joined
    joined["y_team_a"] = (joined["winner_side"] == "team_a").astype(int)
    # Infer whether canonical team A corresponds to GOL.GG team1 by comparing known result orientation.
    joined["canonical_a_is_golgg_team1"] = joined["y_team_a"] == joined["y_team1"]
    joined["exp039_prob_team_a"] = np.where(
        joined["canonical_a_is_golgg_team1"],
        joined["exp039_calibrated_prob_team1"],
        1.0 - joined["exp039_calibrated_prob_team1"],
    )
    report["market_common"] = {
        "n_joined_model_market": int(len(joined)),
        "date_min": str(joined["date"].min()),
        "date_max": str(joined["date"].max()),
        "orientation_inferred_from_results": True,
        "exp039_on_market_common": safe_metric(joined["y_team_a"].values, joined["exp039_prob_team_a"].values),
    }
    for typ in ["open", "mid", "close"]:
        for kind in ["novig", "raw"]:
            col = f"market_{typ}_p_a_{kind}"
            if col not in joined:
                continue
            sub = joined.dropna(subset=[col, "y_team_a"])
            if sub.empty:
                continue
            report["market_common"][f"market_{typ}_{kind}"] = {
                **safe_metric(sub["y_team_a"].values, sub[col].values),
                "avg_books": float(sub.get(f"market_{typ}_books", pd.Series(dtype=float)).mean()) if f"market_{typ}_books" in sub else None,
                "avg_margin": float(sub.get(f"market_{typ}_avg_margin", pd.Series(dtype=float)).mean()) if f"market_{typ}_avg_margin" in sub else None,
            }
    # strict common all three no-vig + exp039
    needed = ["market_open_p_a_novig", "market_mid_p_a_novig", "market_close_p_a_novig", "exp039_prob_team_a"]
    strict = joined.dropna(subset=needed + ["y_team_a"])
    report["strict_common_open_mid_close"] = {"n": int(len(strict))}
    if len(strict):
        for col in needed:
            report["strict_common_open_mid_close"][col] = safe_metric(strict["y_team_a"].values, strict[col].values)
    return report, joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest EXP-039 final artifacts on DB GOL.GG matches and market open/mid/close odds.")
    parser.add_argument("--min-date", default=None, help="Optional YYYY-MM-DD lower bound. Default loads all DB history as rating/W20 warm-up.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/exp039_db_market_backtest")
    args = parser.parse_args()

    init_db()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = build_exp039_predictions(args.min_date, args.limit)
    preds.to_csv(out_dir / "exp039_db_predictions.csv", index=False)
    mapped, odds = load_canonical_market_rows()
    market = aggregate_market(mapped, odds)
    market.to_csv(out_dir / "market_open_mid_close.csv", index=False)
    report, joined = evaluate_all(preds, market)
    joined.to_csv(out_dir / "exp039_market_common.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
