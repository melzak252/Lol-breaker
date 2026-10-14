#!/usr/bin/env python3
"""Retrospective graph diagnostics on the pinned corrected legacy replay; no promotion.

The ongoing fresh collection is deliberately NOT an input. This runner preserves
its inherited rejected transitions and observed-first-map-roster limitation.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_model_redesign import temporal_blocks
from scripts.benchmark_siamese_architectures import (
    monthly_bootstrap,
    probability_metrics,
)
from scripts.build_siamese_research_dataset import sha256
from scripts.corrected_history_evaluation import frozen_hashes, validate_series
from scripts.export_prospective_features import iter_matches
from scripts.graph_relations_history import (
    build_graph_history,
    history_integrity_checks,
)
from scripts.graph_relations_neural import fit_graph_variant
from scripts.prospective_sports_features import _series_team_ids
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
REPLAY = ROOT / "data/artifacts/corrected039081-20260908/replay"
SOURCE = ROOT / "data/artifacts/golgg-header-repair-20260908-validated/matches.json"
PRIOR = ROOT / "data/artifacts/corrected-ratings-logloss-20260908-full"
REFERENCE = (
    ROOT / "data/artifacts/corrected039081-20260908/evaluation-complete/predictions.csv"
)
MODEL_NAMES = (
    "baseline",
    "linear_relations",
    "pair_effects",
    "lineup_spectral",
    "hypergraph_smooth",
    "deepsets",
    "player_gnn",
)
SEEDS = (42, 84)
EPS = np.finfo(float).eps


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def role_players(game, side):
    roster = game[side + "_players"]
    if set(roster) != set(ROLES):
        raise ValueError(f"Unrecognized role keys for game {game['game_id']}")
    result = [
        str(roster[role].get("player_id", roster[role].get("id"))) for role in ROLES
    ]
    if any(player in ("", "None") for player in result):
        raise ValueError("Missing exact player identity")
    return result


def load_inputs():
    audit = json.loads((REPLAY / "audit.json").read_text())
    if audit["collection"]["status"] != "legacy_corrected_partial":
        raise ValueError(
            "This diagnostic is pinned to an explicitly partial legacy replay"
        )
    if sha256(SOURCE) != audit["source"]["sha256"]:
        raise ValueError("Raw source hash differs from the replay contract")
    if sha256(REPLAY / "snapshots.csv") != audit["outputs"]["snapshots.csv"]:
        raise ValueError("Replay snapshot hash mismatch")
    identity = {key: str for key in ("golgg_match_id", "team1_id", "team2_id")}
    frame = (
        pd.read_csv(REPLAY / "snapshots.csv", dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    if frame.golgg_match_id.duplicated().any():
        raise ValueError("Duplicate forecast identity")
    rejected_path = REPLAY / "rejected_series.json"
    if sha256(rejected_path) != audit["outputs"]["rejected_series.json"]:
        raise ValueError("Replay rejection inventory hash mismatch")
    rejected = json.loads(rejected_path.read_text())["rows"]
    blocked = {str(row["match_id"]) for row in rejected}
    targets = frame.set_index("golgg_match_id")
    rosters, events, seen_series, seen_maps = {}, [], set(), set()
    accepted = 0
    for match in iter_matches(SOURCE):
        identity_key = str(match["match_id"])
        if identity_key in seen_series:
            raise ValueError("Duplicate source series")
        seen_series.add(identity_key)
        if identity_key in blocked:
            continue
        a, b = _series_team_ids(match)
        wins = sum(
            int(game["t1_win"]) if str(game["t1_id"]) == a else int(game["t2_win"])
            for game in match["games"]
        )
        losses = len(match["games"]) - wins
        # Same explicit in-memory source adaptation used to create this replay.
        match.update(
            tid_1=a,
            tid_2=b,
            t1_id=a,
            t2_id=b,
            score_1=wins,
            t1_score=wins,
            score_2=losses,
            t2_score=losses,
            t1_win=wins > losses,
            t2_win=losses > wins,
        )
        validate_series(match)
        first = match["games"][0]
        pa, pb = role_players(first, "t1"), role_players(first, "t2")
        if str(first["t1_id"]) != a:
            pa, pb = pb, pa
        if identity_key in targets.index:
            row = targets.loc[identity_key]
            same = row.team1_id == a and row.team2_id == b
            reverse = row.team1_id == b and row.team2_id == a
            if not (same or reverse) or row.date != match["date"]:
                raise ValueError("Source/forecast identity or date mismatch")
            label = int(wins > losses) if same else int(losses > wins)
            if label != row.y_true or int(row.best_of) != int(match["best_of"]):
                raise ValueError("Source/forecast target or format mismatch")
            rosters[identity_key] = pa + pb if same else pb + pa
        for game in match["games"]:
            gid = str(game["game_id"])
            if gid in seen_maps:
                raise ValueError("Duplicate accepted map identity")
            seen_maps.add(gid)
            players = role_players(game, "t1") + role_players(game, "t2")
            if len(set(players)) != 10:
                raise ValueError("Duplicate map participant")
            events.append(
                {
                    "date": match["date"],
                    "game_id": gid,
                    "players": players,
                    "y": int(game["t1_win"]),
                }
            )
        accepted += 1
    if (
        accepted != audit["state_transition_series"]
        or len(events) != audit["state_transition_maps"]
    ):
        raise ValueError("Graph transition inventory does not match baseline replay")
    if set(rosters) != set(frame.golgg_match_id):
        raise ValueError("Graph would silently narrow the baseline scoring cohort")
    frame["players"] = frame.golgg_match_id.map(rosters)
    reference = pd.read_csv(REFERENCE, dtype=identity)
    modern = frame.loc[frame.date >= "2024-01-01"]
    reference = reference.set_index("golgg_match_id").loc[modern.golgg_match_id]
    if len(reference) != 9482:
        raise ValueError("Pinned common cohort changed")
    for field in ("date", "team1_id", "team2_id", "best_of", "y_true"):
        if not np.array_equal(modern[field].to_numpy(), reference[field].to_numpy()):
            raise ValueError(f"Reference cohort differs: {field}")
    return frame, events, audit


def fit_offset(features, y, masks, offset, *, penalty=10.0, laplacian=None):
    """Convex logistic residual; no intercept, fixed penalty, training rows only."""
    train_x, train_y, train_z = (
        features[masks["train"]],
        y[masks["train"]],
        offset[masks["train"]],
    )
    width = features.shape[1]

    def objective(weights):
        z = np.asarray(train_x @ weights).ravel() + train_z
        loss = np.logaddexp(0, z).sum() - train_y @ z
        gradient = np.asarray(train_x.T @ (expit(z) - train_y)).ravel()
        loss += 0.5 * penalty * (weights @ weights)
        gradient += penalty * weights
        if laplacian is not None:
            smooth = np.asarray(laplacian @ weights).ravel()
            loss += 10.0 * (weights @ smooth)
            gradient += 20.0 * smooth
        return float(loss), gradient

    fit = minimize(
        objective,
        np.zeros(width),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-6},
    )
    if not fit.success:
        raise RuntimeError(f"Residual logistic fit failed: {fit.message}")
    logits = offset + np.asarray(features @ fit.x).ravel()
    return (
        logits,
        {"weights": fit.x},
        {
            "iterations": int(fit.nit),
            "penalty": penalty,
            "gradient_max": float(np.abs(fit.jac).max()),
        },
    )


def pair_features(players, train):
    def tokens(row):
        return [
            [tuple(sorted(pair)) for pair in combinations(row[start : start + 5], 2)]
            for start in (0, 5)
        ]

    vocab = {}
    for row in players[train]:
        for side in tokens(row):
            for token in side:
                if token not in vocab:
                    vocab[token] = len(vocab)
    rows, cols, values = [], [], []
    unknown = np.zeros((len(players), 2), dtype=int)
    for i, row in enumerate(players):
        for side_index, side in enumerate(tokens(row)):
            for token in side:
                if token in vocab:
                    rows.append(i)
                    cols.append(vocab[token])
                    values.append(1.0 if side_index == 0 else -1.0)
                else:
                    unknown[i, side_index] += 1
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(players), len(vocab)))
    return matrix, vocab, unknown


def lineup_design(players, train):
    """Train-fitted lineup incidence; unseen queries project by player overlap.

    No outcome weights or future lineups enter the fitted graph. A query's own
    roster is the same explicit retrospective first-map information as baseline.
    """
    lineups = [[tuple(sorted(row[:5])), tuple(sorted(row[5:]))] for row in players]
    vocabulary = {}
    for i in np.flatnonzero(train):
        for lineup in lineups[i]:
            if lineup not in vocabulary:
                vocabulary[lineup] = len(vocabulary)
    player_vocab = {}
    rows, cols = [], []
    for lineup, index in vocabulary.items():
        for player in lineup:
            if player not in player_vocab:
                player_vocab[player] = len(player_vocab)
            rows.append(index)
            cols.append(player_vocab[player])
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(vocabulary), len(player_vocab))
    )
    overlap = (incidence @ incidence.T).tocsr()
    overlap.setdiag(0)
    overlap.eliminate_zeros()
    degree = np.asarray(overlap.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=inverse, where=degree > 0)
    normalized = sparse.diags(inverse) @ overlap @ sparse.diags(inverse)
    laplacian = sparse.diags((degree > 0).astype(float)) - normalized
    query_rows, query_cols, query_values = [], [], []
    unseen = np.zeros((len(players), 2), bool)
    unconnected = np.zeros_like(unseen)
    # Cache immutable roster projections; query membership never changes fitted graph.
    projections = {}
    for i, pair in enumerate(lineups):
        for side, lineup in enumerate(pair):
            if lineup not in projections:
                if lineup in vocabulary:
                    projections[lineup] = (
                        np.array([vocabulary[lineup]]),
                        np.array([1.0]),
                    )
                else:
                    known = [
                        player_vocab[player]
                        for player in lineup
                        if player in player_vocab
                    ]
                    weights = np.asarray(incidence[:, known].sum(axis=1)).ravel()
                    indices = np.flatnonzero(weights)
                    projections[lineup] = (
                        indices,
                        weights[indices] / weights.sum()
                        if len(indices)
                        else np.array([]),
                    )
            indices, weights = projections[lineup]
            unseen[i, side] = lineup not in vocabulary
            unconnected[i, side] = len(indices) == 0
            query_rows.extend([i] * len(indices))
            query_cols.extend(indices)
            query_values.extend(weights * (1 if side == 0 else -1))
    query = sparse.csr_matrix(
        (query_values, (query_rows, query_cols)), shape=(len(players), len(vocabulary))
    )
    return query, normalized, laplacian, vocabulary, unseen, unconnected


def calibrate(logits, y, masks):
    slope = fit_platt_scaling(logits[masks["calibration"]], y[masks["calibration"]])
    probability = np.clip(expit(slope * logits), EPS, 1 - EPS)
    reverse = np.clip(expit(-slope * logits), EPS, 1 - EPS)
    error = float(np.max(np.abs(probability + reverse - 1)))
    if error > 1e-6:
        raise ValueError("Calibrated side symmetry failure")
    return probability, {"slope": float(slope), "symmetry_max_error": error}


def paired(result, candidate, reference="baseline"):
    y = result.y_true.to_numpy(float)
    a, b = result[candidate].to_numpy(), result[reference].to_numpy()
    return {
        "log_loss": monthly_bootstrap(
            binary_log_loss_vector(y, a) - binary_log_loss_vector(y, b), result.date
        ),
        "brier": monthly_bootstrap((a - y) ** 2 - (b - y) ** 2, result.date),
    }


def evaluate(result):
    summary = {"models": {}, "comparisons": {}, "risk_slices": {}, "market": {}}
    for name in MODEL_NAMES:
        summary["models"][name] = {
            "overall": probability_metrics(result.y_true, result[name]),
            "year": {
                str(year): probability_metrics(part.y_true, part[name])
                for year, part in result.groupby(result.date.str[:4])
            },
        }
        if name != "baseline":
            summary["comparisons"][name + "_vs_baseline"] = paired(result, name)
    for reference in (
        "linear_relations",
        "deepsets",
        "pair_effects",
        "lineup_spectral",
    ):
        summary["comparisons"]["player_gnn_vs_" + reference] = paired(
            result, "player_gnn", reference
        )
    # Familywise adjustment over declared primary challenger-vs-baseline contrasts.
    names = [name for name in MODEL_NAMES if name != "baseline"]
    ordered = sorted(
        names,
        key=lambda name: summary["comparisons"][name + "_vs_baseline"]["log_loss"][
            "p_nonnegative"
        ],
    )
    previous = 0.0
    for rank, name in enumerate(ordered):
        test = summary["comparisons"][name + "_vs_baseline"]["log_loss"]
        previous = max(previous, min(1.0, (len(names) - rank) * test["p_nonnegative"]))
        test["holm_adjusted_p"] = previous
    confidence = np.maximum(result.baseline.to_numpy(), 1 - result.baseline.to_numpy())
    disagreement = np.abs(result.player_gl.to_numpy() - result.team_gl.to_numpy())
    masks = {
        "confidence/heavy_favorite": confidence >= 0.75,
        "confidence/moderate_favorite": (confidence >= 0.60) & (confidence < 0.75),
        "confidence/close": confidence <= 0.55,
        "roster/rookie_or_sub": result.graph_min_prior_maps.to_numpy() < 10,
        "roster/experienced": result.graph_min_prior_maps.to_numpy() >= 10,
        "roster/unseen_lineup_at_fit": result.unseen_lineup.to_numpy(bool),
        "disagreement/glicko_player_team_agreement": disagreement <= 0.08,
        "disagreement/glicko_player_team_severe": disagreement > 0.15,
    }
    for key in result.best_of.unique():
        masks["format/BO" + str(int(key))] = result.best_of.to_numpy() == key
    for tier in result.competition_tier.unique():
        masks["tier/" + str(tier)] = result.competition_tier.to_numpy() == tier
    for name, mask in masks.items():
        part = result.loc[mask]
        summary["risk_slices"][name] = {
            model: probability_metrics(part.y_true, part[model])
            for model in MODEL_NAMES
        }
    if {"market", "odds_a", "odds_b"} <= set(result):
        market = result.loc[result.market.notna()].copy()
        summary["market"]["timing"] = (
            "Inherited closing diagnostic; availability/quote timestamps uncertified, not executable betting."
        )
        summary["market"]["n"] = len(market)
        if len(market):
            from scipy.stats import pearsonr, spearmanr

            for name in MODEL_NAMES:
                p = market[name].to_numpy()
                m = market.market.to_numpy()
                y = market.y_true.to_numpy()
                summary["market"][name] = {
                    "delta_logloss": float(
                        np.mean(
                            binary_log_loss_vector(y, p) - binary_log_loss_vector(y, m)
                        )
                    ),
                    "pearson": float(pearsonr(p, m).statistic),
                    "spearman": float(spearmanr(p, m).statistic),
                    "mad": float(np.mean(abs(p - m))),
                    "hybrid_curve_diagnostic_not_tuned_policy": {
                        str(alpha): probability_metrics(y, alpha * p + (1 - alpha) * m)[
                            "log_loss"
                        ]
                        for alpha in np.linspace(0, 1, 11)
                    },
                }
            # Orient BOTH sides; these dependent rows are diagnostics, not bootstrap units.
            oa = np.concatenate([market.odds_a, market.odds_b])
            target = np.concatenate([market.y_true, 1 - market.y_true])
            for low, high in ((2.5, 3.5), (3.5, 5.0), (5.0, float("inf"))):
                mask = (oa >= low) & (oa < high)
                summary["risk_slices"][f"odds/underdog_{low}_{high}"] = {
                    name: probability_metrics(
                        target[mask],
                        np.concatenate([market[name], 1 - market[name]])[mask],
                    )
                    for name in MODEL_NAMES
                }
    summary["limitations"] = [
        "Incomplete corrected legacy transition history, not the fresh collection.",
        "Observed first-map roster, no historical announcement availability timestamps.",
        "9482 modern series, below the 10000-series promotion minimum.",
        "Reused historical holdout; no prospective or confirmatory superiority claim.",
        "Roster experience uses prior maps, not certified roster stability/announcement data.",
        "Disagreement slice uses the named Glicko player/team component, not six-family consensus.",
        "Lineup spectral representation is train-fitted overlap SVD, not exact LinNet/node2vec.",
        "Hypergraph model is Laplacian-regularized lineup residual, not exact NBA HAPM or a hypergraph neural network.",
        "Player GNN is a typed current-match 10-player graph with causal historical edge attributes, not a global temporal GNN.",
    ]
    summary["promotion"] = False
    return summary


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen_before = frozen_hashes()
    sources = [
        Path(__file__),
        ROOT / "scripts/graph_relations_history.py",
        ROOT / "scripts/graph_relations_neural.py",
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
        ROOT / "src/analysis/probability_metrics.py",
        ROOT / "scripts/corrected_history_evaluation.py",
        ROOT / "scripts/prospective_sports_features.py",
    ]
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(relative)] = sha256(dest)
    dump(
        out / "design.json",
        {
            "version": "graph-relations-retrospective-v1",
            "models": MODEL_NAMES,
            "seeds": SEEDS,
            "max_epochs": args.max_epochs,
            "ridge_C": 0.1,
            "pair_penalty": 20,
            "lineup_embedding_dimension": 8,
            "hypergraph_l2": 20,
            "hypergraph_laplacian": 20,
            "train_start": "2020-01-01",
            "folds": [2024, 2025, 2026],
            "protocol": "semester-calibration",
            "history_tau_days": 180,
            "history_shrinkage": 20,
            "bootstrap_resamples": 5000,
            "no_test_selection": True,
            "neural_seeds_averaged_as_raw_logits": True,
            "primary_metric": "paired_logloss",
            "promotion": False,
            "source_hashes": source_hashes,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
        },
    )
    guards = history_integrity_checks()
    dump(out / "history_guards.json", guards)
    frame, events, audit = load_inputs()
    dump(out / "input_audit.json", audit)
    print(
        json.dumps({"stage": "history", "targets": len(frame), "events": len(events)}),
        flush=True,
    )
    graph = build_graph_history(frame, events)
    del events
    graph["coverage"].to_csv(out / "history_coverage.csv", index=False)
    np.savez_compressed(
        out / "graph_features.npz",
        **{key: graph[key] for key in ("nodes", "edges", "direct", "players", "dates")},
    )
    dump(
        out / "feature_names.json",
        {key: graph[key] for key in ("node_names", "edge_names", "direct_names")},
    )
    x, names = build_training_features(frame)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite baseline features")
    y = frame.y_true.to_numpy(float)
    players = np.asarray(graph["players"])
    reference = pd.read_csv(REFERENCE, dtype={"golgg_match_id": str}).set_index(
        "golgg_match_id"
    )
    parts = []
    for year in (2024, 2025, 2026):
        masks = temporal_blocks(frame.date, year, protocol="semester-calibration")
        fold = {
            "partitions": {
                key: {
                    "n": int(mask.sum()),
                    "min": str(frame.loc[mask, "date"].min()),
                    "max": str(frame.loc[mask, "date"].max()),
                }
                for key, mask in masks.items()
            },
            "models": {},
        }
        print(
            json.dumps(
                {"stage": "fold", "year": year, "partitions": fold["partitions"]}
            ),
            flush=True,
        )
        scaler = StandardScaler(with_mean=False).fit(x[masks["train"]])
        ridge = LogisticRegression(C=0.1, fit_intercept=False, max_iter=10000, tol=1e-8)
        ridge.fit(scaler.transform(x[masks["train"]]), y[masks["train"]])
        if ridge.n_iter_[0] >= ridge.max_iter:
            raise RuntimeError("Baseline ridge did not converge")
        offset = ridge.decision_function(scaler.transform(x))
        raw = {"baseline": offset}
        states = {"baseline": {"scaler": scaler, "model": ridge, "features": names}}
        relation_scaler = StandardScaler(with_mean=False).fit(
            graph["direct"][masks["train"]]
        )
        relation_x = relation_scaler.transform(graph["direct"]).astype(np.float64)
        (
            raw["linear_relations"],
            states["linear_relations"],
            fold["models"]["linear_relations"],
        ) = fit_offset(relation_x, y, masks, offset)
        states["linear_relations"]["scaler"] = relation_scaler
        pair_x, pair_vocab, unknown_pairs = pair_features(players, masks["train"])
        raw["pair_effects"], states["pair_effects"], fold["models"]["pair_effects"] = (
            fit_offset(pair_x, y, masks, offset, penalty=20)
        )
        states["pair_effects"]["vocabulary"] = pair_vocab
        query, adjacency, laplacian, lineup_vocab, unseen, unconnected = lineup_design(
            players, masks["train"]
        )
        svd = TruncatedSVD(n_components=8, random_state=42)
        embedding = svd.fit_transform(adjacency)
        spectral = np.asarray(query @ embedding)
        spectral_scaler = StandardScaler(with_mean=False).fit(spectral[masks["train"]])
        spectral_x = spectral_scaler.transform(spectral)
        (
            raw["lineup_spectral"],
            states["lineup_spectral"],
            fold["models"]["lineup_spectral"],
        ) = fit_offset(spectral_x, y, masks, offset)
        states["lineup_spectral"].update(
            embedding=embedding, scaler=spectral_scaler, vocabulary=lineup_vocab
        )
        (
            raw["hypergraph_smooth"],
            states["hypergraph_smooth"],
            fold["models"]["hypergraph_smooth"],
        ) = fit_offset(query, y, masks, offset, penalty=20, laplacian=laplacian)
        states["hypergraph_smooth"]["vocabulary"] = lineup_vocab
        fold["lineup_coverage"] = {
            "train_lineups": len(lineup_vocab),
            "train_pairs": len(pair_vocab),
            "test_unseen_lineup_sides": int(unseen[masks["test"]].sum()),
            "test_unconnected_lineup_sides": int(unconnected[masks["test"]].sum()),
            "test_unseen_pair_fraction": float(
                unknown_pairs[masks["test"]].sum() / (20 * masks["test"].sum())
            ),
        }
        for kind in ("deepsets", "player_gnn"):
            seed_logits = []
            fold["models"][kind] = {"seeds": {}}
            for seed in SEEDS:
                print(
                    json.dumps(
                        {"stage": "neural", "year": year, "kind": kind, "seed": seed}
                    ),
                    flush=True,
                )
                logits, metadata, state = fit_graph_variant(
                    kind,
                    graph["nodes"],
                    graph["edges"],
                    offset,
                    y,
                    masks,
                    seed=seed,
                    max_epochs=args.max_epochs,
                )
                seed_logits.append(logits)
                fold["models"][kind]["seeds"][str(seed)] = metadata
                torch.save(state, out / f"{year}-{kind}-{seed}.pt")
            raw[kind] = np.mean(seed_logits, axis=0)
        test = masks["test"]
        columns = [
            "golgg_match_id",
            "date",
            "team1_id",
            "team2_id",
            "best_of",
            "y_true",
            "competition_tier",
            "player_gl",
            "team_gl",
        ]
        part = frame.loc[test, columns].copy()
        part["fold"] = year
        part["unseen_lineup"] = unseen[test].any(axis=1)
        part["graph_min_prior_maps"] = (
            graph["coverage"].loc[test, "min_prior_maps"].to_numpy()
        )
        for name, logits in raw.items():
            p, calibration = calibrate(logits, y, masks)
            fold["models"].setdefault(name, {})["calibration"] = calibration
            part[name] = p[test]
            part[name + "__raw_logit"] = logits[test]
            np.savez_compressed(
                out / f"{year}-{name}-logits.npz",
                logits=logits,
                ids=frame.golgg_match_id.to_numpy(str),
                dates=frame.date.to_numpy(str),
            )
        for name, state in states.items():
            joblib.dump(state, out / f"{year}-{name}.joblib")
        for column in (
            "market",
            "odds_a",
            "odds_b",
            "exp039-corrected-history-research-v1",
            "exp081-corrected-history-research-v1",
        ):
            part[column] = reference.loc[part.golgg_match_id, column].to_numpy()
        dump(out / f"{year}-fold.json", fold)
        parts.append(part)
        pd.concat(parts).to_csv(out / "predictions-in-progress.csv", index=False)
        print(json.dumps({"stage": "fold_complete", "year": year}), flush=True)
    result = (
        pd.concat(parts).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    )
    if len(result) != 9482 or result.golgg_match_id.duplicated().any():
        raise ValueError("Incomplete or duplicate evaluation cohort")
    result.to_csv(out / "predictions.csv", index=False)
    summary = evaluate(result)
    for name in (
        "exp039-corrected-history-research-v1",
        "exp081-corrected-history-research-v1",
    ):
        summary["reference_replica_metrics_" + name] = probability_metrics(
            result.y_true, result[name]
        )
        summary["comparisons"]["player_gnn_vs_" + name] = paired(
            result, "player_gnn", name
        )
    summary["elapsed_seconds"] = time.monotonic() - started
    dump(out / "summary.json", summary)
    if frozen_hashes() != frozen_before or sha256(SOURCE) != audit["source"]["sha256"]:
        raise ValueError("Read-only source/frozen artifact invariant broken")
    verification = {
        "source_hash_unchanged": True,
        "frozen_hashes": frozen_before,
        "source_code_hashes": source_hashes,
        "history_guards": guards,
        "test_rows": len(result),
        "promotion": False,
        "sha256": {path.name: sha256(path) for path in out.iterdir() if path.is_file()},
    }
    dump(out / "verification.json", verification)
    print(
        json.dumps(
            {
                "stage": "complete",
                "output": str(out),
                "metrics": {
                    name: summary["models"][name]["overall"] for name in MODEL_NAMES
                },
            },
            allow_nan=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    run(args)


if __name__ == "__main__":
    main()
