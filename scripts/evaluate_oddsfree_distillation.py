#!/usr/bin/env python3
"""EXP-089 evaluation-only diagnostics for already-selected A-series probabilities.

No fitting of predictive models, selection, database access, or tournament replay.
Diagnostic calibration regressions never alter the supplied probabilities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_siamese_architectures import monthly_bootstrap, probability_metrics
from scripts.build_siamese_research_dataset import sha256
from src.analysis.probability_metrics import binary_log_loss_vector, calculate_ece

SELECTED = ("p__outcome", "p__market_selected", "p__hybrid_selected")
REQUIRED = (
    "golgg_match_id", "date", "y_true", "best_of", "competition_tier", "tournament",
    "roster_min_prior_series", "odds_a", "odds_b", "market", "player_consensus",
    "team_consensus", *SELECTED,
)
METRIC_KEYS = (
    "log_loss", "brier", "auc", "accuracy", "ece10", "mce10",
    "calibration_intercept", "calibration_slope", "brier_reliability_binned",
    "brier_resolution_binned", "brier_uncertainty", "brier_binning_residual",
    "mean_p", "observed_rate", "calibration_gap",
)
CONFIDENCE_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
ODDS_EDGES = (1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 10.0, None)


def _basic(y: np.ndarray, p: np.ndarray) -> dict:
    if not len(y):
        return {
            "n": 0, "status": "empty", "mean_p": None, "observed_rate": None,
            "calibration_gap": None, "log_loss": None, "brier": None,
        }
    return {
        "n": int(len(y)), "status": "single_class" if len(np.unique(y)) == 1 else "ok",
        "mean_p": float(p.mean()), "observed_rate": float(y.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
        "log_loss": float(binary_log_loss_vector(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _metrics_without_calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """Preserve the benchmark's non-regression metrics if its regression fails.

Use its exact floor(p*10) bin convention; shared calculate_ece deliberately has
its own right-closed convention. No replacement calibration fit is attempted.
    """
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = resolution = max_error = 0.0
    for index in np.unique(bins):
        mask = bins == index
        gap = float(p[mask].mean() - y[mask].mean())
        reliability += float(mask.mean()) * gap**2
        resolution += float(mask.mean()) * float(y[mask].mean() - y.mean()) ** 2
        max_error = max(max_error, abs(gap))
    brier = float(np.mean((p - y) ** 2))
    uncertainty = float(y.mean() * (1 - y.mean()))
    return {
        **_basic(y, p),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y))),
        "ece10": float(calculate_ece(y, p, 10)), "mce10": max_error,
        "calibration_intercept": None, "calibration_slope": None,
        "brier_reliability_binned": reliability, "brier_resolution_binned": resolution,
        "brier_uncertainty": uncertainty,
        "brier_binning_residual": brier - (reliability - resolution + uncertainty),
    }


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if not len(y):
        return {
            **dict.fromkeys(METRIC_KEYS), **_basic(y, p),
            "calibration_status": "unavailable_empty", "calibration_error": None,
            "boundary_probabilities_clipped": 0,
        }
    # The benchmark requires open-interval probabilities. Only exact endpoints
    # are moved by machine epsilon; shared LL uses its established 1e-15 clip.
    bounded = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    calibration_error = None
    if np.ptp(bounded) == 0 and len(np.unique(y)) == 2:
        result = _metrics_without_calibration(y, bounded)
        calibration_status = "unavailable_constant_prediction"
    else:
        try:
            result = probability_metrics(y, bounded)
            calibration_status = "ok" if len(np.unique(y)) == 2 else "unavailable_single_class"
        except ValueError as error:
            if not str(error).startswith("diagnostic calibration fit failed:"):
                raise
            result = _metrics_without_calibration(y, bounded)
            calibration_status = "optimizer_failed"
            calibration_error = str(error)
    if any(
        result.get(key) is not None and not np.isfinite(result[key])
        for key in ("calibration_slope", "calibration_intercept")
    ):
        result["calibration_slope"] = result["calibration_intercept"] = None
        calibration_status = "nonfinite_optimizer_result"
    return {
        **result, **_basic(y, p),
        "calibration_status": calibration_status, "calibration_error": calibration_error,
        "boundary_probabilities_clipped": int(np.count_nonzero(p != bounded)),
    }


def _paired(y: np.ndarray, p: np.ndarray, reference: np.ndarray, dates: pd.Series) -> dict:
    months = int(dates.dt.to_period("M").nunique())
    result = {"n": int(len(y)), "months": months, "status": "ok"}
    for name, delta in (
        ("log_loss", binary_log_loss_vector(y, p) - binary_log_loss_vector(y, reference)),
        ("brier", (p - y) ** 2 - (reference - y) ** 2),
    ):
        if months >= 2:
            result[name] = monthly_bootstrap(delta, dates)
        else:
            result["status"] = "unavailable_fewer_than_two_months"
            result[name] = {
                "mean": float(delta.mean()) if len(delta) else None,
                "ci95_low": None, "ci95_high": None, "p_nonnegative": None,
                "months": months, "resamples": 0,
            }
    return result


def _intervals(values: np.ndarray, edges: tuple):
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = values >= lower
        if upper is not None:
            mask &= values <= upper if upper == edges[-1] else values < upper
        yield lower, upper, mask


def _common_slices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    market = frame.market.to_numpy(float)
    roster = frame.roster_min_prior_series.to_numpy(float)
    disagreement = abs(frame.player_consensus.to_numpy(float) - frame.team_consensus.to_numpy(float))
    confidence = np.maximum(frame.p__outcome.to_numpy(float), 1 - frame.p__outcome.to_numpy(float))
    slices = {
        "all": np.ones(n, dtype=bool),
        "no_market": ~np.isfinite(market), "with_market": np.isfinite(market),
        "roster_prior_lt10": np.isfinite(roster) & (roster < 10),
        "roster_prior_ge10": np.isfinite(roster) & (roster >= 10),
        "roster_prior_missing": ~np.isfinite(roster),
        "rating_disagreement_le008": np.isfinite(disagreement) & (disagreement <= 0.08),
        "rating_disagreement_008_to_015": (disagreement > 0.08) & (disagreement <= 0.15),
        "rating_disagreement_gt015": np.isfinite(disagreement) & (disagreement > 0.15),
        "rating_disagreement_missing": ~np.isfinite(disagreement),
    }
    for year in sorted(frame.date.dt.year.unique()):
        slices[f"year_{year}"] = (frame.date.dt.year == year).to_numpy()
    best_of = frame.best_of.to_numpy(float)
    for value in (1, 3, 5):
        slices[f"bo{value}"] = best_of == value
    slices["best_of_other_or_missing"] = ~np.isin(best_of, (1, 3, 5))
    tiers = frame.competition_tier.fillna("missing").astype(str)
    for tier in sorted(set(tiers) | {"international", "major", "minor", "academy", "unknown", "missing"}):
        slices[f"tier_{tier}"] = (tiers == tier).to_numpy()
    for lower, upper, mask in _intervals(confidence, CONFIDENCE_EDGES):
        slices[f"outcome_confidence_{lower:.2f}_{upper:.2f}"] = mask
    return slices


def _reliability(y: np.ndarray, p: np.ndarray, favored: bool) -> list[dict]:
    edges = CONFIDENCE_EDGES if favored else tuple(np.linspace(0, 1, 11))
    return [
        {"lower": float(lower), "upper": float(upper), **_basic(y[mask], p[mask])}
        for lower, upper, mask in _intervals(p, edges)
    ]


def _data_readiness() -> dict:
    """Read local schema/manifest metadata only; never import operational services."""
    base = Path("data/artifacts/model-redesign-v1")
    manifest_path = base / "source-v2/manifest.json"
    manifest = {}
    if (ROOT / manifest_path).is_file():
        with (ROOT / manifest_path).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    exports = {}
    for relative in (
        base / "source-v2/matches.csv", base / "source-v2/games.csv",
        base / "source-v2/players.csv", base / "replay/snapshots.csv",
    ):
        path = ROOT / relative
        header = None
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
        exports[str(relative)] = {"present": path.is_file(), "columns": header}
    source_paths = [
        "scripts/evaluate_oddsfree_distillation.py",
        "scripts/export_model_research_data.py", "src/models/replay_series.py",
        "betting_app/models/golgg.py", "src/utils/golgg_schema.py",
        "betting_app/services/current_roster_service.py",
        "betting_app/services/tournament_service.py",
        "betting_app/services/liquipedia_bracket_service.py",
        "betting_app/api/routers/tournaments.py", "src/models/tournament_simulator.py",
    ]
    contracts = [
        {
            "contract": "historical_source_availability", "status": "missing",
            "finding": "Retrospective export explicitly has source_available_at=null. Extraction/snapshot timestamps are not historical publication timestamps; raw ORM has event date/source_link/raw_json, not an availability contract.",
            "evidence": [f"{manifest_path}:source_available_at", "scripts/export_model_research_data.py:23-31,51-56", "betting_app/models/golgg.py:27-90"],
        },
        {
            "contract": "pre_tournament_rosters", "status": "missing",
            "finding": "players.csv stores game appearances without announcement/availability times. Raw JSON compatibility can recover realized game lineups. Current roster service has source_match_date/updated_at and overwrites current roles; it is not an immutable tournament-start roster archive.",
            "evidence": [f"{base}/source-v2/players.csv:header", "src/utils/golgg_schema.py:209-240", "betting_app/services/current_roster_service.py:100-178"],
        },
        {
            "contract": "strict_prior_date_series_state", "status": "present_with_limitations",
            "finding": "EXP-083 derives previous-date unambiguous rosters and freezes priors for each calendar date. feature_history_max_date is emitted. This is retrospective per-series replay, not tournament-start state; no within-day timezone/order is invented.",
            "evidence": ["src/models/replay_series.py:1-20,88-105,517-548,617-695", f"{base}/replay/snapshots.csv:header"],
        },
        {
            "contract": "frozen_tournament_start_state_and_counterfactual_features", "status": "missing",
            "finding": "Provided rows contain realized fixtures and per-series predictions, not tournament-start all-pair feature/state snapshots. App TournamentSimulator loads latest team ratings by id, without historical cutoff; its map-rating conversion must not consume these series probabilities.",
            "evidence": ["betting_app/services/tournament_service.py:540-580", "betting_app/api/routers/tournaments.py:111-129,160-179", "src/models/tournament_simulator.py:27-56"],
        },
        {
            "contract": "full_graph_representation", "status": "present_for_curated_current_events_only",
            "finding": "BracketMatchNode supports winner/loser links and destination slots. Three curated 2026 builders exist, but source exports have no graph/seeding/conditional-draw contract covering the historical evaluation cohort.",
            "evidence": ["betting_app/services/tournament_service.py:18-43,531-535", "scripts/export_model_research_data.py:23-31"],
        },
        {
            "contract": "immutable_pre_start_bracket_and_format", "status": "missing",
            "finding": "Current bracket cache stores synced_at and realized teams/scores/winners in one overwritten event file, not versioned pre-start graph snapshots. Current-state sync and manual overrides do not reconstruct historical counterfactual paths.",
            "evidence": ["betting_app/services/liquipedia_bracket_service.py:112-134,434-468,527-547", "betting_app/api/routers/tournaments.py:160-179"],
        },
    ]
    return {
        "status": "exploratory_only", "audit_scope": "Local export headers, manifest, raw ORM/JSON accessor schema and simulator source only; no database or external requests.",
        "tournament_replay": {
            "status": "blocked", "executed": False,
            "reason": "No verified pre-start roster/source availability, frozen tournament-start all-pair state, or historical full graph/format snapshot. Do not substitute realized lineups, realized bracket paths, or series-to-map reinterpretation.",
        },
        "manifest": {
            "path": str(manifest_path), "present": (ROOT / manifest_path).is_file(),
            "sha256": sha256(ROOT / manifest_path) if (ROOT / manifest_path).is_file() else None,
            "source_available_at": manifest.get("source_available_at"),
            "extracted_at": manifest.get("extracted_at"), "snapshot_at": manifest.get("snapshot_at"),
            "availability_caveat": manifest.get("availability_caveat"),
            "tables": manifest.get("tables", {}),
        },
        "exports": exports, "contracts": contracts,
        "source_evidence": {
            path: {"present": (ROOT / path).is_file(), "sha256": sha256(ROOT / path) if (ROOT / path).is_file() else None}
            for path in source_paths
        },
    }


def _calibration_plot(reliability: list[dict], selected: list[str], path: Path, n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, orientation in zip(axes, ("team_a", "model_favored")):
        for column in selected:
            rows = [
                row for row in reliability
                if row["model"] == column and row["orientation"] == orientation and row["n"] >= 30
            ]
            if rows:
                axis.plot(
                    [row["mean_p"] for row in rows], [row["observed_rate"] for row in rows],
                    marker="o", label=column.removeprefix("p__"),
                )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(
            title=f"{'A-series' if orientation == 'team_a' else 'Model-favored side'} reliability",
            xlabel="Mean predicted series-win probability", ylabel="Observed series-win fraction",
            xlim=(0, 1), ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.2, "No bins with N >= 30", ha="center")
    fig.suptitle(f"EXP-089 exploratory calibration (N={n}; displayed bins N >= 30)")
    fig.text(0.5, 0.01, "One side per match in each panel; panels are not independent samples. No test-set calibration applied.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    with path.open("xb") as handle:
        fig.savefig(handle, format="svg")
    plt.close(fig)


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Write fixed-cohort metrics; return compact, finite JSON-compatible summary.

Every p__ column must cover every input row. NaNs are allowed only in diagnostic
metadata such as market/odds/ratings. Existing artifacts are never overwritten.
    """
    missing = sorted(set(REQUIRED) - set(predictions.columns))
    if missing:
        raise ValueError(f"evaluation missing columns: {missing}")
    if not predictions.columns.is_unique:
        raise ValueError("evaluation column names must be unique")
    frame = predictions.reset_index(drop=True).copy()
    if frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("evaluation requires one row per unique nonmissing match ID")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any():
        raise ValueError("evaluation dates must be complete")
    if frame.date.dt.tz is not None:
        # Group using the supplied timezone's calendar months, not an invented UTC date.
        frame["date"] = frame.date.dt.tz_localize(None)
    y = frame.y_true.to_numpy(float)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("evaluation outcomes must be finite binary labels")
    columns = [column for column in frame if column.startswith("p__")]
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{column} must supply bounded finite probabilities on the full cohort")
    for column in ("market", "player_consensus", "team_consensus"):
        values = frame[column].to_numpy(float)
        if np.isinf(values).any() or np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
            raise ValueError(f"{column} must be missing or a bounded probability")
    for column in ("best_of", "roster_min_prior_series", "odds_a", "odds_b"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    selected = [column for column in (*SELECTED, "p__frozen_ridge") if column in columns]
    references = [column for column in ("p__outcome", "p__frozen_ridge") if column in columns]
    output_dir = Path(output_dir)
    paths = {name: output_dir / filename for name, filename in {
        "evaluation": "evaluation.json", "metrics": "metrics.csv",
        "paired_comparisons": "paired_comparisons.csv", "reliability": "reliability.csv",
        "high_confidence_diagnostics": "high_confidence_diagnostics.csv",
        "data_readiness": "data_readiness.json", "calibration": "calibration.svg",
    }.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate, paired = {}, {}
    metric_rows, paired_rows, reliability_rows, high_confidence_rows = [], [], [], []
    slices = _common_slices(frame)
    all_probabilities = {column: frame[column].to_numpy(float) for column in columns}
    for column, p in all_probabilities.items():
        aggregate[column] = _metrics(y, p)
        metric_rows.append({"model": column, "slice": "all", "orientation": "team_a", **aggregate[column]})
        paired[column] = {}
        for reference in references:
            if column == reference:
                continue
            comparison = _paired(y, p, all_probabilities[reference], frame.date)
            paired[column][reference] = comparison
            for score in ("log_loss", "brier"):
                paired_rows.append({
                    "model": column, "reference": reference, "score": score,
                    "n": comparison["n"], "status": comparison["status"], **comparison[score],
                })
        if column not in selected:
            continue
        for name, mask in slices.items():
            if name != "all":
                metric_rows.append({"model": column, "slice": name, "orientation": "team_a", **_metrics(y[mask], p[mask])})
        favored_a = p >= 0.5
        favored_p, favored_y = np.where(favored_a, p, 1 - p), np.where(favored_a, y, 1 - y)
        for orientation, aligned_p, aligned_y in (("team_a", p, y), ("model_favored", favored_p, favored_y)):
            reliability_rows.extend({"model": column, "orientation": orientation, **row} for row in _reliability(aligned_y, aligned_p, orientation == "model_favored"))
        odds_a, odds_b = frame.odds_a.to_numpy(float), frame.odds_b.to_numpy(float)
        valid_odds = np.isfinite(odds_a) & np.isfinite(odds_b) & (odds_a > 1) & (odds_b > 1)
        for orientation, side_a in (
            ("market_underdog", odds_a >= odds_b),
            ("market_favorite", odds_a <= odds_b),
            ("model_favored", favored_a),
        ):
            aligned_odds = np.where(side_a, odds_a, odds_b)
            aligned_p, aligned_y = np.where(side_a, p, 1 - p), np.where(side_a, y, 1 - y)
            for lower, upper, mask in _intervals(aligned_odds, ODDS_EDGES):
                mask &= valid_odds
                name = f"odds_{lower:.2f}_{upper:.2f}" if upper is not None else f"odds_ge{lower:.2f}"
                metric_rows.append({
                    "model": column, "slice": name, "orientation": orientation,
                    "selected_side_a_n": int(np.count_nonzero(mask & side_a)),
                    "selected_side_b_n": int(np.count_nonzero(mask & ~side_a)),
                    **_metrics(aligned_y[mask], aligned_p[mask]),
                })
        metric_rows.append({"model": column, "slice": "odds_missing_or_invalid", "orientation": "team_a", **_metrics(y[~valid_odds], p[~valid_odds])})
        # Outcome conditioning below is an explicit retrospective error audit only.
        for threshold in (0.8, 0.9, 0.95):
            for selector in (column, "p__outcome"):
                selector_p = all_probabilities[selector]
                high = np.maximum(selector_p, 1 - selector_p) >= threshold
                wrong = high & ((selector_p > 0.5) != y)
                for subgroup, mask in (("all_high_confidence", high), ("wrong_high_confidence", wrong)):
                    high_confidence_rows.append({
                        "model": column, "selector": selector, "threshold": threshold,
                        "subgroup": subgroup, "cohort_n": len(y),
                        "high_confidence_n": int(high.sum()), "wrong_n": int(wrong.sum()),
                        "wrong_rate_within_high": float(wrong.sum() / high.sum()) if high.any() else None,
                        **_basic(y[mask], p[mask]),
                    })
                if selector == "p__outcome":
                    break
    readiness = _data_readiness()
    blockers = [
        "Exploratory retrospective evaluation on previously inspected holdouts; no production promotion or superiority claim.",
        "Canonical79 source/roster availability and stored EXP081 training provenance are not fully verified.",
        "Closing-market auxiliary targets are retrospective supervision, not executable price/edge evidence.",
        readiness["tournament_replay"]["reason"],
    ]
    summary = {
        "n": len(frame), "prediction_columns": columns, "selected_columns": selected,
        "aggregate": aggregate, "paired": paired,
        "paths": {name: str(path) for name, path in paths.items()},
        "promotion_blockers": blockers, "data_readiness": readiness,
    }
    report = {
        **summary,
        "contract": {
            "probability_unit": "P(team A wins the complete SERIES), never map probability",
            "cohort": "Same unique match IDs for all supplied p__ columns; no row exclusion for missing odds",
            "selection": "No predictive calibration/weight/hyperparameter selection in this evaluator; supplied selected models are frozen",
            "bootstrap": "Paired candidate-minus-reference monthly-block LL/Brier; 5000 resamples, seed 82; negative favors candidate; no multiplicity adjustment",
            "log_loss_clipping": "Shared binary_log_loss_vector default epsilon=1e-15; Brier uses original probabilities",
            "confidence_bins": "Common slices use max(p__outcome,1-p__outcome); own-model reliability uses its favored side; all thresholds fixed",
            "rating_disagreement": "Absolute difference between player_consensus and team_consensus, each an arithmetic mean of six MAP-level rating probabilities. Descriptive slicing only: neither is scored against series outcomes or converted into a series predictor.",
            "odds_slices": "Each orientation chooses exactly one side per match, flips both p and y for B, and uses that side's decimal price. Equal-price/model-probability ties choose A. Different orientations overlap and are NOT independent samples. Bins are [lower,upper), final open-ended; 3.50–5.00 is explicit.",
            "high_confidence": "Odds-free fixed thresholds .80/.90/.95. Counts and proper scores include every high-confidence row and descriptive outcome-conditioned wrong subset; this is not a training-loss or model-selection rule.",
            "calibration_fit": "Diagnostic only; empty/single-class/constant and failed optimizer fits return null coefficients with status. No fallback slope is invented.",
            "plot": "Only reliability bins with N>=30 displayed; all bins including empty ones are retained in CSV/JSON",
        },
        "slices": metric_rows, "reliability": reliability_rows,
        "high_confidence_diagnostics": high_confidence_rows,
    }
    # Serialize before writing so no NaN/Infinity can escape into a partial report.
    encoded_report = json.dumps(report, indent=2, allow_nan=False) + "\n"
    encoded_readiness = json.dumps(readiness, indent=2, allow_nan=False) + "\n"
    for name, content in (("evaluation", encoded_report), ("data_readiness", encoded_readiness)):
        with paths[name].open("x", encoding="utf-8") as handle:
            handle.write(content)
    for name, rows in (
        ("metrics", metric_rows), ("paired_comparisons", paired_rows),
        ("reliability", reliability_rows), ("high_confidence_diagnostics", high_confidence_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[name], index=False, mode="x")
    _calibration_plot(reliability_rows, selected, paths["calibration"], len(frame))
    return summary
