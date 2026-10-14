#!/usr/bin/env python3
"""Benchmark causal graph-relation predictions against player Glicko-2.

Offline retrospective research only. The graph receives no market, odds, or
future result input. It uses the same strict prior-completed-day replay and
previous-observed-roster proxy as the full historical rating benchmark.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_siamese_architectures import monthly_bootstrap, probability_metrics
from scripts.build_siamese_research_dataset import sha256
from scripts.corrected_history_evaluation import frozen_hashes
from scripts.export_prospective_features import iter_matches
from scripts.graph_relations_history import build_graph_history, history_integrity_checks
from scripts.prospective_sports_features import _prepare_history
from scripts.train_and_tune_siamese_series import fit_platt_scaling, write_json_exclusive
from src.analysis.probability_metrics import binary_log_loss_vector

SOURCE = ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json"
REPLAY = ROOT / "data/artifacts/full-historical-predictions-20260909/replay"
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
EPS = np.finfo(float).eps

VERSION = "graph-relations-player-glicko2-priorday-annual-v1"

def dump(path: Path, value: dict) -> None:
    write_json_exclusive(path, value)


def annual_graph_masks(frame: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    """Partition one annual walk-forward fold without using later labels.

    The final half of the second preceding year is STOP-only for regularization
    selection. The immediately preceding calendar year is CALIBRATION-only.
    """
    if not isinstance(year, int) or isinstance(year, bool):
        raise TypeError("year must be an integer")
    if not {"date", "result_day"} <= set(frame):
        raise ValueError("frame requires date and result_day")
    dates = pd.to_datetime(frame.date, utc=True, errors="raise")
    result_days = pd.to_datetime(frame.result_day, utc=True, errors="raise")
    origin = pd.Timestamp(f"{year}-01-01", tz="UTC")
    calibration_start = pd.Timestamp(f"{year - 1}-01-01", tz="UTC")
    stop_start = pd.Timestamp(f"{year - 2}-07-01", tz="UTC")
    train = (dates < stop_start).to_numpy(bool)
    stop = ((dates >= stop_start) & (dates < calibration_start)).to_numpy(bool)
    calibration = ((dates >= calibration_start) & (dates < origin)).to_numpy(bool)
    test = ((dates >= origin) & (dates < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))).to_numpy(bool)
    masks = {
        "train": train,
        "stop": stop,
        "select": np.zeros(len(frame), dtype=bool),
        "calibration": calibration,
        "test": test,
    }
    assigned = train.astype(np.int8) + stop.astype(np.int8) + calibration.astype(np.int8) + test.astype(np.int8)
    if (assigned > 1).any():
        raise ValueError("annual graph partitions overlap")
    if train.any() and not (result_days[train] < stop_start).all():
        raise ValueError("training labels are unavailable at the STOP boundary")
    if stop.any() and not (result_days[stop] < calibration_start).all():
        raise ValueError("STOP labels are unavailable at the calibration boundary")
    if calibration.any() and not (result_days[calibration] < origin).all():
        raise ValueError("calibration labels are unavailable at the test boundary")
    return masks


def role_players(game: dict, side: str) -> list[str]:
    roster = game.get(f"{side}_players")
    if not isinstance(roster, dict) or set(roster) != set(ROLES):
        raise ValueError(f"{side} roster has no complete canonical role mapping")
    players = []
    for role in ROLES:
        player = roster[role]
        if not isinstance(player, dict):
            raise ValueError(f"{side} {role} player is not an object")
        identifier = player.get("player_id", player.get("id"))
        if isinstance(identifier, bool) or identifier is None or str(identifier).strip() == "":
            raise ValueError(f"{side} {role} player has no exact ID")
        players.append(str(identifier))
    if len(set(players)) != 5:
        raise ValueError(f"{side} roster has duplicate player IDs")
    return players


def snapshot_roster(value: object, column: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{column} must be a JSON roster string")
    roster = json.loads(value)
    if not isinstance(roster, list) or len(roster) != 5:
        raise ValueError(f"{column} must contain five player IDs")
    result = [str(player) for player in roster]
    if any(not player or player.lower() in {"none", "nan", "null"} for player in result):
        raise ValueError(f"{column} contains an unusable player ID")
    if len(set(result)) != 5:
        raise ValueError(f"{column} contains duplicate player IDs")
    return result


def load_inputs() -> tuple[pd.DataFrame, list[dict], dict]:
    audit = json.loads((REPLAY / "audit.json").read_text())
    if audit.get("status") != "complete" or audit.get("temporal_violations") != 0:
        raise ValueError("full historical replay is not a complete temporally valid source")
    snapshots = REPLAY / "snapshots.csv"
    inventory_path = REPLAY / "source_inventory.csv"
    if sha256(snapshots) != audit["outputs"]["snapshots.csv"]:
        raise ValueError("full replay snapshot hash mismatch")
    if sha256(inventory_path) != audit["outputs"]["source_inventory.csv"]:
        raise ValueError("full replay inventory hash mismatch")
    identity = {name: str for name in ("golgg_match_id", "team1_id", "team2_id")}
    frame = pd.read_csv(snapshots, dtype=identity).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    required = {"golgg_match_id", "date", "result_day", "team1_id", "team2_id", "y_true", "player_gl", "roster1_ids", "roster2_ids"}
    if not required <= set(frame):
        raise ValueError("full replay lacks graph/Glicko-2 benchmark fields")
    if frame.golgg_match_id.duplicated().any():
        raise ValueError("full replay has duplicate prediction identities")
    if not np.isfinite(frame.player_gl).all() or not ((frame.player_gl > 0) & (frame.player_gl < 1)).all():
        raise ValueError("player Glicko-2 probabilities must be finite and strictly bounded")
    if not frame.y_true.isin((0, 1)).all():
        raise ValueError("full replay has non-binary graph targets")

    raw_roles: dict[str, list[tuple[str, list[str], list[str]]]] = {}
    source_dates: dict[str, str] = {}

    def records():
        for match in iter_matches(SOURCE):
            identifier = str(match["match_id"])
            if identifier in raw_roles:
                raise ValueError("duplicate source series identity")
            source_dates[identifier] = str(match["date"])
            raw_roles[identifier] = [
                (str(game["t1_id"]), role_players(game, "t1"), role_players(game, "t2"))
                for game in match["games"]
            ]
            yield match

    series, source_audit = _prepare_history(records(), date.max)
    if source_audit["raw_series"] != audit["raw_series"] or len(series) != audit["used_series"]:
        raise ValueError("normalized graph history does not match the full replay inventory")
    inventory = pd.read_csv(inventory_path, dtype={"golgg_match_id": str})
    if set(inventory.golgg_match_id) != set(source_dates):
        raise ValueError("source inventory identities differ from graph source")
    for row in inventory.itertuples(index=False):
        if source_dates[row.golgg_match_id] != row.date:
            raise ValueError("source inventory date differs from graph source")

    released = 0
    latest_roles: dict[str, list[str]] = {}
    events: list[dict] = []
    target_players: list[list[str]] = []
    targets_by_day = frame.groupby("date", sort=True)
    ordered = sorted(series, key=lambda item: (item.games[-1].day, item.order))
    for target_day, rows in targets_by_day:
        day = date.fromisoformat(str(target_day))
        while released < len(ordered) and ordered[released].games[-1].day < day:
            item = ordered[released]
            role_games = raw_roles.get(item.identifier)
            if role_games is None or len(role_games) != len(item.games):
                raise ValueError("source role roster inventory differs from normalized series")
            for index, (game, (raw_t1, raw_p1, raw_p2)) in enumerate(zip(item.games, role_games, strict=True)):
                p1, p2 = (raw_p1, raw_p2) if raw_t1 == item.team1 else (raw_p2, raw_p1)
                if sorted(p1) != game.players1 or sorted(p2) != game.players2:
                    raise ValueError("role roster differs from normalized map roster")
                latest_roles[item.team1] = p1
                latest_roles[item.team2] = p2
                events.append(
                    {
                        "date": item.games[-1].day.isoformat(),
                        "game_id": f"{item.identifier}:{index + 1}",
                        "players": p1 + p2,
                        "y": int(game.score),
                    }
                )
            released += 1
        for row in rows.itertuples(index=False):
            if row.team1_id not in latest_roles or row.team2_id not in latest_roles:
                raise ValueError("graph target has no strictly prior observed roster")
            p1, p2 = latest_roles[row.team1_id], latest_roles[row.team2_id]
            if sorted(p1) != snapshot_roster(row.roster1_ids, "roster1_ids") or sorted(p2) != snapshot_roster(row.roster2_ids, "roster2_ids"):
                raise ValueError("graph role roster does not match the full replay roster proxy")
            if set(p1).intersection(p2):
                raise ValueError("graph target opponents share a player")
            target_players.append(p1 + p2)
    while released < len(ordered):
        item = ordered[released]
        role_games = raw_roles[item.identifier]
        for index, (game, (raw_t1, raw_p1, raw_p2)) in enumerate(zip(item.games, role_games, strict=True)):
            p1, p2 = (raw_p1, raw_p2) if raw_t1 == item.team1 else (raw_p2, raw_p1)
            events.append({"date": item.games[-1].day.isoformat(), "game_id": f"{item.identifier}:{index + 1}", "players": p1 + p2, "y": int(game.score)})
        released += 1
    if len(target_players) != len(frame) or len(events) != audit["history_maps_consumed"]:
        raise ValueError("graph feature targets or transitions were silently narrowed")
    frame["players"] = target_players
    return frame, events, audit


def fold_summary(frame: pd.DataFrame, year: int, masks: dict[str, np.ndarray]) -> dict:
    result = {"year": year, "partitions": {}}
    for name in ("train", "stop", "calibration", "test"):
        part = frame.loc[masks[name]]
        if not len(part) or len(np.unique(part.y_true)) != 2:
            raise ValueError(f"{year}: {name} has insufficient binary labels")
        result["partitions"][name] = {
            "n": len(part),
            "date_min": part.date.min(),
            "date_max": part.date.max(),
            "result_day_max": part.result_day.max(),
        }
    return result


def paired(result: pd.DataFrame, candidate: str, reference: str) -> dict:
    y = result.y_true.to_numpy(float)
    candidate_p = result[candidate].to_numpy(float)
    reference_p = result[reference].to_numpy(float)
    return {
        "log_loss": monthly_bootstrap(binary_log_loss_vector(y, candidate_p) - binary_log_loss_vector(y, reference_p), result.date),
        "brier": monthly_bootstrap((candidate_p - y) ** 2 - (reference_p - y) ** 2, result.date),
    }


def fit_graph_relations(
    direct: np.ndarray,
    baseline_logits: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict, dict]:
    """Fit a symmetric graph-relation logistic model without later labels."""
    candidates = (0.01, 0.1, 1.0)
    train, stop = masks["train"], masks["stop"]
    selection_scaler = StandardScaler(with_mean=False).fit(direct[train])
    selection_x = np.column_stack(
        (baseline_logits, selection_scaler.transform(direct))
    )
    losses = {}
    for penalty in candidates:
        model = LogisticRegression(
            C=penalty, fit_intercept=False, max_iter=10_000, tol=1e-8
        ).fit(selection_x[train], y[train])
        if model.n_iter_[0] >= model.max_iter:
            raise RuntimeError("graph relation logistic model did not converge")
        losses[str(penalty)] = float(
            log_loss(y[stop], expit(model.decision_function(selection_x[stop])))
        )
    selected = min(candidates, key=lambda penalty: (losses[str(penalty)], penalty))
    refit = train | stop
    scaler = StandardScaler(with_mean=False).fit(direct[refit])
    x = np.column_stack((baseline_logits, scaler.transform(direct)))
    model = LogisticRegression(
        C=selected, fit_intercept=False, max_iter=10_000, tol=1e-8
    ).fit(x[refit], y[refit])
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError("refit graph relation logistic model did not converge")
    return model.decision_function(x), {
        "candidate_C": list(candidates),
        "stop_log_loss_by_C": losses,
        "selected_C": selected,
        "scaler_fit_rows": int(refit.sum()),
        "model_fit_rows": int(refit.sum()),
        "solver_iterations": model.n_iter_.tolist(),
    }, {
        "scaler": scaler,
        "model": model,
        "feature_order": ["player_glicko2_logit", "graph_direct"],
    }


def evaluate(result: pd.DataFrame) -> dict:
    models = (
        "player_glicko2",
        "player_glicko2_calibrated",
        "graph_relations_logistic",
    )
    return {
        "models": {
            name: {
                "overall": probability_metrics(result.y_true, result[name]),
                "year": {
                    str(year): probability_metrics(part.y_true, part[name])
                    for year, part in result.groupby(result.date.str[:4])
                },
            }
            for name in models
        },
        "comparisons": {
            "graph_relations_logistic_vs_player_glicko2": paired(
                result, "graph_relations_logistic", "player_glicko2"
            ),
            "graph_relations_logistic_vs_calibrated_player_glicko2": paired(
                result, "graph_relations_logistic", "player_glicko2_calibrated"
            ),
        },
        "limitations": [
            "Retrospective historical replay; source availability is prior completed calendar day, not certified publication time.",
            "The roster input is the same previous-observed-roster proxy as the full rating benchmark, not a historical lineup announcement.",
            "The candidate is a zero-intercept logistic model over player Glicko-2 and causal graph relation differences, not a global temporal GNN.",
            "No market or odds field is read, fitted, or used for eligibility.",
        ],
        "promotion": False,
    }


def run(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen_before = frozen_hashes()
    sources = (
        Path(__file__),
        ROOT / "scripts/graph_relations_history.py",
        ROOT / "scripts/prospective_sports_features.py",
    )
    source_hashes = {}
    for source in sources:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(source.relative_to(ROOT))] = sha256(destination)
    dump(output / "design.json", {
        "version": VERSION,
        "candidate": "zero-intercept graph-relation logistic model with raw player Glicko-2 logit",
        "reference": "raw player Glicko-2; calibration-equivalent Glicko-2 reported separately",
        "folds": list(range(2020, 2027)),
        "protocol": "annual expanding walk-forward; STOP Jul-Dec two years earlier selects C; CALIBRATION prior calendar year",
        "regularization_candidates": [0.01, 0.1, 1.0],
        "history_availability": "all maps released on final source-series day; graph target and relations are strictly earlier calendar days",
        "market_inputs": False,
        "bootstrap_resamples": 5000,
        "source_hashes": source_hashes,
        "runtime": {"python": platform.python_version()},
        "promotion": False,
    })
    guards = history_integrity_checks()
    dump(output / "history_guards.json", guards)
    frame, events, audit = load_inputs()
    dump(output / "input_audit.json", audit)
    print(json.dumps({"stage": "history", "targets": len(frame), "events": len(events)}), flush=True)
    graph = build_graph_history(frame, events)
    del events
    graph["coverage"].to_csv(output / "history_coverage.csv", index=False)
    np.savez_compressed(output / "graph_features.npz", **{key: graph[key] for key in ("nodes", "edges", "direct", "players", "dates")})
    dump(output / "feature_names.json", {key: graph[key] for key in ("node_names", "edge_names", "direct_names")})

    y = frame.y_true.to_numpy(float)
    baseline_logits = logit(np.clip(frame.player_gl.to_numpy(float), EPS, 1 - EPS))
    models = output / "models"
    models.mkdir(exist_ok=False)
    parts, folds = [], []
    for year in range(2020, 2027):
        fold_started = time.monotonic()
        masks = annual_graph_masks(frame, year)
        fold = fold_summary(frame, year, masks)
        fold_dir = models / str(year)
        fold_dir.mkdir(exist_ok=False)
        print(json.dumps({"stage": "fold", "year": year, "partitions": fold["partitions"]}), flush=True)
        graph_logits, model_evidence, state = fit_graph_relations(
            graph["direct"], baseline_logits, y, masks
        )
        joblib.dump(state, fold_dir / "graph_relations_logistic.joblib")
        fold["model_fit"] = model_evidence
        graph_slope = fit_platt_scaling(graph_logits[masks["calibration"]], y[masks["calibration"]])
        glicko_slope = fit_platt_scaling(baseline_logits[masks["calibration"]], y[masks["calibration"]])
        test = masks["test"]
        graph_p = np.clip(expit(graph_slope * graph_logits[test]), EPS, 1 - EPS)
        glicko_p = np.clip(expit(baseline_logits[test]), EPS, 1 - EPS)
        calibrated_glicko_p = np.clip(expit(glicko_slope * baseline_logits[test]), EPS, 1 - EPS)
        part = frame.loc[test, ["golgg_match_id", "date", "result_day", "team1_id", "team2_id", "best_of", "y_true", "player_gl", "feature_history_max_at"]].copy()
        part["fold_year"] = year
        part["player_glicko2"] = glicko_p
        part["player_glicko2_calibrated"] = calibrated_glicko_p
        part["graph_relations_logistic"] = graph_p
        part["graph_raw_logit"] = graph_logits[test]
        part["graph_calibration_slope"] = graph_slope
        part["glicko2_calibration_slope"] = glicko_slope
        if np.max(np.abs(graph_p + expit(-graph_slope * graph_logits[test]) - 1)) > 1e-6:
            raise ValueError("calibrated graph probability symmetry failed")
        part.to_csv(fold_dir / "predictions.csv", index=False, mode="x")
        fold["calibration"] = {"graph_positive_platt_slope": float(graph_slope), "glicko2_positive_platt_slope": float(glicko_slope)}
        fold["metrics"] = {
            "player_glicko2": probability_metrics(part.y_true, part.player_glicko2),
            "player_glicko2_calibrated": probability_metrics(part.y_true, part.player_glicko2_calibrated),
            "graph_relations_logistic": probability_metrics(part.y_true, part.graph_relations_logistic),
        }
        fold["runtime_seconds"] = time.monotonic() - fold_started
        dump(fold_dir / "completed.json", {"status": "complete", **fold})
        parts.append(part)
        folds.append(fold)
        print(json.dumps({"stage": "fold_complete", "year": year, "n": len(part), "glicko2_log_loss": fold["metrics"]["player_glicko2"]["log_loss"], "graph_log_loss": fold["metrics"]["graph_relations_logistic"]["log_loss"]}), flush=True)

    result = pd.concat(parts, ignore_index=True).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    if result.golgg_match_id.duplicated().any():
        raise ValueError("walk-forward graph scoring has duplicate target identities")
    result.to_csv(output / "predictions.csv", index=False, mode="x")
    summary = evaluate(result)
    summary["elapsed_seconds"] = time.monotonic() - started
    dump(output / "summary.json", summary)
    if frozen_hashes() != frozen_before:
        raise ValueError("frozen model artifact changed during graph benchmark")
    dump(output / "verification.json", {
        "source_snapshot_hashes": source_hashes,
        "source_sha256": sha256(SOURCE),
        "replay_audit_sha256": sha256(REPLAY / "audit.json"),
        "history_guards": guards,
        "target_rows": len(result),
        "target_years": sorted(result.date.str[:4].unique().tolist()),
        "frozen_hashes": frozen_before,
        "market_inputs": False,
        "promotion": False,
    })
    print(json.dumps({"stage": "complete", "output": str(output), "glicko2_log_loss": summary["models"]["player_glicko2"]["overall"]["log_loss"], "graph_log_loss": summary["models"]["graph_relations_logistic"]["overall"]["log_loss"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
