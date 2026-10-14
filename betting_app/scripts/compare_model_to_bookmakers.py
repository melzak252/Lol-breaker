"""Compare stored model predictions against bookmaker odds collected by the app.

The script evaluates finished canonical matches that have:

* a stored prediction in ``canonical_predictions`` for the selected model,
* at least one collected two-way bookmaker odds snapshot,
* a known ``winner_side`` in ``canonical_matches``.

For every bookmaker it takes the latest non-live odds snapshot per match and
converts decimal odds to no-vig market probabilities. It then reports standard
probabilistic metrics for:

* the selected model on the bookmaker/common match set,
* each individual bookmaker,
* an average market consensus across bookmakers.

Usage examples::

    python -m betting_app.scripts.compare_model_to_bookmakers
    python -m betting_app.scripts.compare_model_to_bookmakers --min-matches 20
    python -m betting_app.scripts.compare_model_to_bookmakers --model-name Hybrid-Thesis-Market
    python -m betting_app.scripts.compare_model_to_bookmakers --require-mapped
    python -m betting_app.scripts.compare_model_to_bookmakers --require-mapped --allowed-mapping-sources alias,builtin
    python -m betting_app.scripts.compare_model_to_bookmakers --csv-output reports/model_vs_books.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db, query_df
from betting_app.core.ev import fair_market_probabilities
from betting_app.core.models.registry import C0_NATIVE
from betting_app.services.canonical_match_service import align_snapshot_odds
from betting_app.services.mapping_service import suggest_mapping


@dataclass(frozen=True)
class MetricRow:
    rank: int
    source: str
    kind: str
    n: int
    logloss: float
    auc: float | None
    brier: float
    accuracy: float
    ece: float
    avg_prob_a: float
    avg_abs_edge_vs_model: float | None = None
    model_logloss_on_same_matches: float | None = None
    logloss_delta_vs_model: float | None = None


def _winner_to_binary(winner_side: Any) -> int | None:
    """Return 1 if canonical team A won, 0 if team B won, otherwise None."""

    if winner_side is None:
        return None
    value = str(winner_side).strip().lower()
    if value in {"a", "team_a", "teama", "side_a", "blue"}:
        return 1
    if value in {"b", "team_b", "teamb", "side_b", "red"}:
        return 0
    return None


def _clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(y_prob.astype(float), 1e-15, 1.0 - 1e-15)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Simple equal-width ECE for binary probabilities."""

    if len(y_true) == 0:
        return float("nan")
    y_prob = _clip_probs(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_prob = _clip_probs(y_prob)
    out: dict[str, float | None] = {
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(np.mean((y_prob >= 0.5) == y_true)),
        "ece": float(_expected_calibration_error(y_true, y_prob)),
        "avg_prob_a": float(np.mean(y_prob)),
    }
    # AUC is undefined when only one class is present.
    out["auc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else None
    return out


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{value:.{digits}f}"


def load_evaluation_rows(
    *,
    model_name: str,
    model_version: str | None,
    prediction_status: str,
    include_live: bool,
) -> pd.DataFrame:
    """Load selected model predictions joined with collected bookmaker odds."""

    params: dict[str, Any] = {
        "model_name": model_name,
        "prediction_status": prediction_status,
    }
    version_filter = ""
    if model_version:
        version_filter = "AND cp.model_version = :model_version"
        params["model_version"] = model_version
    live_filter = "" if include_live else "AND COALESCE(os.is_live, 0) = 0"

    # Pull candidate rows and choose latest prediction/odds in pandas. This is
    # intentionally portable across PostgreSQL and SQLite fallback.
    return query_df(
        f"""
        SELECT
            cm.id AS canonical_match_id,
            cm.team_a_name,
            cm.team_b_name,
            cm.start_time_normalized,
            cm.league,
            cm.winner_side,
            cp.id AS prediction_id,
            cp.model_name,
            cp.model_version,
            cp.predicted_at,
            cp.prob_a AS model_prob_a,
            b.name AS bookmaker,
            os.id AS odds_snapshot_id,
            os.raw_team_a,
            os.raw_team_b,
            os.odds_a,
            os.odds_b,
            os.is_live,
            os.scraped_at
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        JOIN odds_snapshots os ON os.canonical_match_id = cm.id
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE cm.winner_side IS NOT NULL
          AND cp.model_name = :model_name
          {version_filter}
          AND cp.prediction_status = :prediction_status
          AND cp.prob_a IS NOT NULL
          AND cp.prob_a > 0 AND cp.prob_a < 1
          AND os.odds_a IS NOT NULL AND os.odds_b IS NOT NULL
          AND os.odds_a > 1 AND os.odds_b > 1
          {live_filter}
        """,
        params,
    )


def prepare_market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Align odds, remove margin, and keep one prediction/bookmaker row per match."""

    if raw.empty:
        return raw

    df = raw.copy()
    df["y_true"] = df["winner_side"].map(_winner_to_binary)
    df = df[df["y_true"].notna()].copy()
    if df.empty:
        return df
    df["y_true"] = df["y_true"].astype(int)

    # Latest prediction per match for the selected model/version.
    df["predicted_at_sort"] = pd.to_datetime(df["predicted_at"], errors="coerce", utc=True)
    latest_pred = (
        df[["canonical_match_id", "prediction_id", "predicted_at_sort"]]
        .drop_duplicates()
        .sort_values(["canonical_match_id", "predicted_at_sort", "prediction_id"])
        .groupby("canonical_match_id", as_index=False)
        .tail(1)[["canonical_match_id", "prediction_id"]]
    )
    df = df.merge(latest_pred, on=["canonical_match_id", "prediction_id"], how="inner")

    # Latest odds per match/bookmaker.
    df["scraped_at_sort"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df = (
        df.sort_values(["canonical_match_id", "bookmaker", "scraped_at_sort", "odds_snapshot_id"])
        .groupby(["canonical_match_id", "bookmaker"], as_index=False)
        .tail(1)
        .copy()
    )

    aligned_odds_a: list[float] = []
    aligned_odds_b: list[float] = []
    market_prob_a: list[float] = []
    market_prob_b: list[float] = []

    for row in df.to_dict("records"):
        aligned = align_snapshot_odds(
            str(row["team_a_name"]),
            str(row["team_b_name"]),
            str(row["raw_team_a"]),
            str(row["raw_team_b"]),
            row["odds_a"],
            row["odds_b"],
            league=row.get("league"), source_system="bookmaker",
            match_date=row.get("start_time_normalized"),
        )
        if aligned is None:
            aligned_odds_a.append(float("nan"))
            aligned_odds_b.append(float("nan"))
            market_prob_a.append(float("nan"))
            market_prob_b.append(float("nan"))
            continue
        oa, ob = aligned
        try:
            pa, pb = fair_market_probabilities(float(oa), float(ob))
        except ValueError:
            oa = ob = pa = pb = float("nan")
        aligned_odds_a.append(float(oa))
        aligned_odds_b.append(float(ob))
        market_prob_a.append(float(pa))
        market_prob_b.append(float(pb))

    df["aligned_odds_a"] = aligned_odds_a
    df["aligned_odds_b"] = aligned_odds_b
    df["market_prob_a"] = market_prob_a
    df["market_prob_b"] = market_prob_b
    df = df[df["market_prob_a"].notna()].copy()
    df["model_prob_a"] = df["model_prob_a"].astype(float)
    return df


def filter_confident_mappings(df: pd.DataFrame, allowed_sources: set[str] | None = None) -> pd.DataFrame:
    """Keep only rows where both canonical teams resolve to trusted GOL.GG teams.

    The comparison against bookmakers can be restricted to "pewne połączenia":
    matches where both sides have a live mapping from the current mapping service.
    By default all non-empty, non-blocked mappings are accepted. Passing
    ``allowed_sources`` can restrict this further, e.g. to ``{"alias", "builtin"}``.
    """

    if df.empty:
        return df

    cache: dict[int, tuple[str | None, str | None, str | None, str | None]] = {}

    def mapping_for(row: pd.Series) -> tuple[str | None, str | None, str | None, str | None]:
        match_id = int(row["canonical_match_id"])
        if match_id not in cache:
            team_a_golgg, _conf_a, source_a = suggest_mapping(str(row["team_a_name"]))
            team_b_golgg, _conf_b, source_b = suggest_mapping(str(row["team_b_name"]))
            cache[match_id] = (team_a_golgg, source_a, team_b_golgg, source_b)
        return cache[match_id]

    keep: list[bool] = []
    team_a_golgg_values: list[str | None] = []
    team_b_golgg_values: list[str | None] = []
    source_a_values: list[str | None] = []
    source_b_values: list[str | None] = []

    for _, row in df.iterrows():
        team_a_golgg, source_a, team_b_golgg, source_b = mapping_for(row)
        team_a_golgg_values.append(team_a_golgg)
        team_b_golgg_values.append(team_b_golgg)
        source_a_values.append(source_a)
        source_b_values.append(source_b)
        is_mapped = bool(team_a_golgg and team_b_golgg)
        is_allowed = True
        if allowed_sources is not None:
            is_allowed = (source_a in allowed_sources) and (source_b in allowed_sources)
        keep.append(is_mapped and is_allowed)

    out = df.copy()
    out["team_a_golgg_name"] = team_a_golgg_values
    out["team_b_golgg_name"] = team_b_golgg_values
    out["team_a_mapping_source"] = source_a_values
    out["team_b_mapping_source"] = source_b_values
    return out[pd.Series(keep, index=out.index)].copy()


def build_results(df: pd.DataFrame, min_matches: int) -> tuple[list[MetricRow], pd.DataFrame]:
    """Return ranked metric rows and per-match market consensus."""

    if df.empty:
        return [], pd.DataFrame()

    rows: list[MetricRow] = []

    # Consensus: average no-vig probability from all latest bookmaker rows for each match.
    consensus = (
        df.groupby("canonical_match_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            model_prob_a=("model_prob_a", "first"),
            market_prob_a=("market_prob_a", "mean"),
            bookmaker_count=("bookmaker", "nunique"),
            team_a_name=("team_a_name", "first"),
            team_b_name=("team_b_name", "first"),
            league=("league", "first"),
            start_time_normalized=("start_time_normalized", "first"),
        )
        .copy()
    )

    if len(consensus) >= min_matches:
        y = consensus["y_true"].to_numpy(dtype=int)
        model_p = consensus["model_prob_a"].to_numpy(dtype=float)
        market_p = consensus["market_prob_a"].to_numpy(dtype=float)
        model_m = _metrics(y, model_p)
        market_m = _metrics(y, market_p)
        rows.append(
            MetricRow(
                rank=0,
                source="MODEL",
                kind="model_on_consensus_matches",
                n=len(consensus),
                logloss=float(model_m["logloss"]),
                auc=model_m["auc"],
                brier=float(model_m["brier"]),
                accuracy=float(model_m["accuracy"]),
                ece=float(model_m["ece"]),
                avg_prob_a=float(model_m["avg_prob_a"]),
            )
        )
        rows.append(
            MetricRow(
                rank=0,
                source="MARKET_CONSENSUS_AVG",
                kind="market",
                n=len(consensus),
                logloss=float(market_m["logloss"]),
                auc=market_m["auc"],
                brier=float(market_m["brier"]),
                accuracy=float(market_m["accuracy"]),
                ece=float(market_m["ece"]),
                avg_prob_a=float(market_m["avg_prob_a"]),
                avg_abs_edge_vs_model=float(np.mean(np.abs(model_p - market_p))),
                model_logloss_on_same_matches=float(model_m["logloss"]),
                logloss_delta_vs_model=float(market_m["logloss"] - model_m["logloss"]),
            )
        )

    for bookmaker, book_df in df.groupby("bookmaker"):
        if len(book_df) < min_matches:
            continue
        y = book_df["y_true"].to_numpy(dtype=int)
        book_p = book_df["market_prob_a"].to_numpy(dtype=float)
        model_p = book_df["model_prob_a"].to_numpy(dtype=float)
        book_m = _metrics(y, book_p)
        model_m = _metrics(y, model_p)
        rows.append(
            MetricRow(
                rank=0,
                source=f"BOOKMAKER:{bookmaker}",
                kind="market",
                n=len(book_df),
                logloss=float(book_m["logloss"]),
                auc=book_m["auc"],
                brier=float(book_m["brier"]),
                accuracy=float(book_m["accuracy"]),
                ece=float(book_m["ece"]),
                avg_prob_a=float(book_m["avg_prob_a"]),
                avg_abs_edge_vs_model=float(np.mean(np.abs(model_p - book_p))),
                model_logloss_on_same_matches=float(model_m["logloss"]),
                logloss_delta_vs_model=float(book_m["logloss"] - model_m["logloss"]),
            )
        )

    rows.sort(key=lambda r: r.logloss)
    ranked = [MetricRow(rank=i + 1, **{k: v for k, v in r.__dict__.items() if k != "rank"}) for i, r in enumerate(rows)]
    return ranked, consensus


def print_report(
    rows: list[MetricRow],
    consensus: pd.DataFrame,
    model_name: str,
    model_version: str | None,
    *,
    require_mapped: bool = False,
    allowed_mapping_sources: set[str] | None = None,
) -> None:
    print("=" * 120)
    print("PORÓWNANIE MODELU Z KURSAMI BUKMACHERÓW ZEBRANYMI PRZEZ APLIKACJĘ")
    print("=" * 120)
    print(f"Model: {model_name}" + (f" ({model_version})" if model_version else ""))
    if require_mapped:
        sources = ",".join(sorted(allowed_mapping_sources)) if allowed_mapping_sources else "any non-blocked mapping"
        print(f"Filtr GOL.GG: tylko pewne połączenia obu drużyn; źródła: {sources}")
    print(f"Mecze w konsensusie rynku: {len(consensus)}")
    if not rows:
        print("Brak wystarczających danych do porównania.")
        return
    print()
    print(
        f"{'#':>2} {'Źródło':<28} {'Typ':<28} {'N':>5} "
        f"{'LogLoss':>8} {'AUC':>7} {'Brier':>8} {'Acc':>7} {'ECE':>7} "
        f"{'AvgP(A)':>8} {'|edge|':>8} {'ModelLL':>8} {'ΔLL vs model':>12}"
    )
    print("-" * 120)
    for r in rows:
        print(
            f"{r.rank:>2} {r.source:<28} {r.kind:<28} {r.n:>5} "
            f"{_format_float(r.logloss):>8} {_format_float(r.auc):>7} {_format_float(r.brier):>8} "
            f"{r.accuracy:>6.1%} {_format_float(r.ece):>7} {_format_float(r.avg_prob_a):>8} "
            f"{_format_float(r.avg_abs_edge_vs_model):>8} {_format_float(r.model_logloss_on_same_matches):>8} "
            f"{_format_float(r.logloss_delta_vs_model):>12}"
        )
    print()
    print("Interpretacja:")
    print("- LogLoss niżej = lepiej; to główna metryka probabilistyczna.")
    print("- ΔLL vs model > 0 oznacza, że dane źródło ma gorszy LogLoss niż model na tych samych meczach.")
    print("- Market probabilities są no-vig: 1/kurs po usunięciu marży bukmachera.")
    print("- Dla bookmakerów model jest dodatkowo liczony na dokładnie tym samym zbiorze meczów co dany bookmaker.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a stored model with collected bookmaker odds on finished matches.")
    parser.add_argument("--model-name", default=C0_NATIVE.name, help="canonical_predictions.model_name")
    parser.add_argument("--model-version", default=C0_NATIVE.version, help="canonical_predictions.model_version; empty = any")
    parser.add_argument("--prediction-status", default="active", help="Prediction status to evaluate")
    parser.add_argument("--min-matches", type=int, default=5, help="Minimum matches required for a bookmaker/source")
    parser.add_argument("--include-live", action="store_true", help="Include live odds snapshots; default uses only pre-match/non-live odds")
    parser.add_argument(
        "--require-mapped",
        action="store_true",
        help="Evaluate only matches where both canonical teams currently map to GOL.GG teams",
    )
    parser.add_argument(
        "--allowed-mapping-sources",
        default="alias,builtin",
        help="Comma-separated trusted mapping sources used with --require-mapped; empty = any non-blocked mapping",
    )
    parser.add_argument("--csv-output", help="Optional path to save the ranked metric table as CSV")
    args = parser.parse_args()

    model_version = args.model_version or None
    init_db()
    raw = load_evaluation_rows(
        model_name=args.model_name,
        model_version=model_version,
        prediction_status=args.prediction_status,
        include_live=args.include_live,
    )
    market = prepare_market_frame(raw)
    allowed_sources = None
    if args.require_mapped and args.allowed_mapping_sources.strip():
        allowed_sources = {s.strip() for s in args.allowed_mapping_sources.split(",") if s.strip()}
    if args.require_mapped:
        before_matches = market["canonical_match_id"].nunique() if not market.empty else 0
        market = filter_confident_mappings(market, allowed_sources)
        after_matches = market["canonical_match_id"].nunique() if not market.empty else 0
        print(f"Filtered by GOL.GG mapping coverage: {after_matches}/{before_matches} consensus-candidate matches kept")
    rows, consensus = build_results(market, min_matches=args.min_matches)
    print_report(
        rows,
        consensus,
        args.model_name,
        model_version,
        require_mapped=args.require_mapped,
        allowed_mapping_sources=allowed_sources,
    )

    if args.csv_output and rows:
        out = Path(args.csv_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(out, index=False)
        print(f"\nZapisano CSV: {out}")


if __name__ == "__main__":
    main()
