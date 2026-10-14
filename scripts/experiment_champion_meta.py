#!/usr/bin/env python3
"""EXP-086 train-only empirical champion-role PPMI embeddings, not target draft.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python
scripts/experiment_champion_meta.py --source-dir data/artifacts/exp086-champion-meta/source-20260908
--output-dir data/artifacts/exp086-champion-meta/run-20260908

Locked design: 90-day team/player usage, 30-day meta, 20-observation win
shrinkage, 8-dimensional teammate cooccurrence PPMI/SVD, ridge C=.1.
Train-only fitting; no parameter selection using the retrospective test cohort.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import (
    monthly_bootstrap,
    probability_metrics,
)
from scripts.build_siamese_research_dataset import sha256
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
SEED = 86
EPS = np.finfo(float).eps


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_history(path):
    frame = pd.read_csv(
        path,
        dtype={
            k: "string"
            for k in ("game_id", "match_id", "team_id", "player_id", "champion_id")
        },
    )
    audit = {
        "rows": len(frame),
        "missing_fields": {k: int(v) for k, v in frame.isna().sum().items()},
        "roles": {
            str(k): int(v) for k, v in frame.role.value_counts(dropna=False).items()
        },
        "distinct_champions": int(frame.champion_id.nunique()),
        "distinct_games": int(frame.game_id.nunique()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "date_mismatch_rows": int((frame.date != frame.game_date).sum()),
        "date_min": frame.date.min(),
        "date_max": frame.date.max(),
        "local_source_v2": "players.csv lacks champion; games.csv lacks patch; no other local raw champion dataset found",
        "current_patch_known_pre_match": False,
    }
    frame = frame.drop_duplicates().copy()
    if frame.duplicated(["game_id", "team_id", "role"]).any():
        raise ValueError("Ambiguous game/team/role identities")
    if audit["date_mismatch_rows"] or not frame.role.isin(ROLES).all():
        raise ValueError("Unresolved sports date/role discrepancy")
    frame["token"] = frame.role + ":" + frame.champion_id.fillna("UNKNOWN")
    frame["day"] = (
        pd.to_datetime(frame.date).values.astype("datetime64[D]").astype(np.int64)
    )
    frame["observed_game_patch"] = frame.observed_game_patch.fillna("")
    audit["year_coverage"] = {
        str(year): {
            "rows": len(group),
            "champion_missing": int(group.champion_id.isna().sum()),
            "patch_missing": int((group.observed_game_patch == "").sum()),
        }
        for year, group in frame.groupby(frame.date.str[:4])
    }
    return frame, audit


def historical_features(frame, history):
    """Freeze the entire date before updating outcomes, lineup, patch or usage.

    Token columns are lossless storage addresses only. Each year's learned
    vocabulary is independently restricted to train; future-only addresses never
    enter its PPMI matrix, SVD or embedding predictions.
    """
    tokens = sorted(history.token.unique())
    index = {token: i for i, token in enumerate(tokens)}
    history = history.copy()
    history["token_index"] = history.token.map(index)
    by_day = {int(day): group for day, group in history.groupby("day", sort=True)}
    target_days = (
        pd.to_datetime(frame.date).values.astype("datetime64[D]").astype(np.int64)
    )
    targets = defaultdict(list)
    for i, day in enumerate(target_days):
        targets[int(day)].append(i)
    teams = defaultdict(Counter)
    players = defaultdict(Counter)
    roster = defaultdict(dict)
    team_patch = {}
    meta = Counter()
    meta_wins = Counter()
    queue90, queue30 = deque(), deque()
    latest_patch, latest_patch_day = "", None
    latest_history_day = None
    names = [
        f"{pool}_{name}"
        for pool in ("team", "lineup")
        for name in (
            "log_count",
            "log_richness",
            "entropy",
            "top_share",
            "meta_overlap",
            "shrunk_meta_win",
            "missing",
        )
    ]
    names += [
        "lineup_role_coverage",
        "lineup_log_age",
        "latest_observed_patch_match",
        "observed_patch_log_age",
        "observed_patch_missing",
    ]
    values = np.zeros((len(frame), 2, len(names)))
    rows_t, cols_t, data_t, rows_p, cols_p, data_p = [], [], [], [], [], []
    coverage = []

    def describe(pool, meta_prob, rates):
        count = sum(pool.values())
        probs = (
            {key: val / count for key, val in pool.items() if val > 0} if count else {}
        )
        arr = np.array(list(probs.values()))
        return [
            np.log1p(count),
            np.log1p(len(probs)),
            float(-(arr * np.log(arr)).sum()),
            float(arr.max()) if len(arr) else 0.0,
            sum(min(val, meta_prob.get(key, 0.0)) for key, val in probs.items()),
            sum(val * rates.get(key, 0.5) for key, val in probs.items())
            if probs
            else 0.5,
            float(not probs),
        ], probs

    match_ids = frame.golgg_match_id.to_numpy(str)
    match_dates = frame.date.to_numpy(str)
    match_teams = frame[["team1_id", "team2_id"]].to_numpy(str)
    for day in sorted(set(by_day) | set(targets)):
        while queue90 and queue90[0][0] < day - 90:
            _, team, player, token, role = queue90.popleft()
            teams[team][token] -= 1
            players[(player, role)][token] -= 1
        while queue30 and queue30[0][0] < day - 30:
            _, token, won = queue30.popleft()
            meta[token] -= 1
            meta_wins[token] -= won
        if day in targets:
            if latest_history_day is not None and latest_history_day >= day:
                raise ValueError("Whole-day freeze violated")
            mass = sum(meta.values())
            meta_prob = (
                {key: val / mass for key, val in meta.items() if val > 0}
                if mass
                else {}
            )
            rates = {
                key: (meta_wins[key] + 10.0) / (val + 20.0)
                for key, val in meta.items()
                if val > 0
            }
            for i in targets[day]:
                row_cov = {
                    "golgg_match_id": match_ids[i],
                    "date": match_dates[i],
                    "latest_meta_age_days": day - latest_patch_day
                    if latest_patch_day is not None
                    else None,
                    "latest_meta_missing": not bool(latest_patch),
                    "latest_observed_meta_proxy": latest_patch,
                    "max_source_day": str(np.datetime64(latest_history_day, "D"))
                    if latest_history_day is not None
                    else None,
                }
                for side, team in enumerate(match_teams[i]):
                    lineup_pool = Counter()
                    ages = []
                    for role, (player, seen) in roster[team].items():
                        pool = players[(player, role)]
                        n = sum(pool.values())
                        if n:
                            for token, count in pool.items():
                                if count > 0:
                                    lineup_pool[token] += count / n
                        ages.append(day - seen)
                    desc_t, prob_t = describe(teams[team], meta_prob, rates)
                    desc_p, prob_p = describe(lineup_pool, meta_prob, rates)
                    observed_patch, patch_day = team_patch.get(team, ("", None))
                    values[i, side] = (
                        desc_t
                        + desc_p
                        + [
                            len(ages) / 5.0,
                            np.log1p(np.mean(ages)) if ages else 0.0,
                            float(
                                bool(observed_patch) and observed_patch == latest_patch
                            ),
                            np.log1p(day - patch_day) if patch_day else 0.0,
                            float(not observed_patch),
                        ]
                    )
                    for token, probability in prob_t.items():
                        rows_t.append(i * 2 + side)
                        cols_t.append(token)
                        data_t.append(probability)
                    for token, probability in prob_p.items():
                        rows_p.append(i * 2 + side)
                        cols_p.append(token)
                        data_p.append(probability)
                    row_cov[f"side{side + 1}_team_pool_missing"] = bool(desc_t[-1])
                    row_cov[f"side{side + 1}_lineup_pool_missing"] = bool(desc_p[-1])
                    row_cov[f"side{side + 1}_lineup_roles"] = len(ages)
                    row_cov[f"side{side + 1}_lineup_mean_age_days"] = (
                        float(np.mean(ages)) if ages else None
                    )
                coverage.append(row_cov)
        if day in by_day:
            daily = by_day[day]
            for row in daily.itertuples(index=False):
                token, team, player, role = (
                    int(row.token_index),
                    str(row.team_id),
                    str(row.player_id),
                    row.role,
                )
                teams[team][token] += 1
                players[(player, role)][token] += 1
                meta[token] += 1
                meta_wins[token] += float(row.game_win)
                queue90.append((day, team, player, token, role))
                queue30.append((day, token, float(row.game_win)))
            # No chronology is fabricated within a date: majority player by role,
            # ties lexicographic. This is a prior-day proxy, never announced lineup.
            counts = (
                daily.groupby(["team_id", "role", "player_id"])
                .size()
                .reset_index(name="n")
            )
            for (team, role), group in counts.groupby(["team_id", "role"]):
                player = (
                    group.sort_values(["n", "player_id"], ascending=[False, True])
                    .iloc[0]
                    .player_id
                )
                roster[str(team)][role] = (str(player), day)
            for team, group in daily.groupby("team_id"):
                patches = group.loc[
                    group.observed_game_patch != "", "observed_game_patch"
                ]
                if len(patches):
                    team_patch[str(team)] = (
                        patches.value_counts().sort_index().idxmax(),
                        day,
                    )
            patches = daily.loc[daily.observed_game_patch != "", "observed_game_patch"]
            if len(patches):
                latest_patch, latest_patch_day = (
                    patches.value_counts().sort_index().idxmax(),
                    day,
                )
            latest_history_day = day
    shape = (len(frame) * 2, len(tokens))
    return (
        values[:, 0] - values[:, 1],
        names,
        sparse.csr_matrix((data_t, (rows_t, cols_t)), shape=shape),
        sparse.csr_matrix((data_p, (rows_p, cols_p)), shape=shape),
        tokens,
        pd.DataFrame(coverage),
    )


def fit_embedding(history, tokens, year):
    train = history.loc[
        (history.date >= "2020-01-01")
        & (history.date < f"{year - 1}-01-01")
        & history.champion_id.notna()
    ]
    vocab = sorted(train.token.unique())
    index = {token: i for i, token in enumerate(vocab)}
    cooccurrence = np.zeros((len(vocab), len(vocab)))
    for _, group in train.groupby(["game_id", "team_id"], sort=False):
        ids = np.array([index[token] for token in group.token])
        cooccurrence[np.ix_(ids, ids)] += 1.0
    np.fill_diagonal(cooccurrence, 0.0)
    rows, cols = cooccurrence.sum(axis=1), cooccurrence.sum(axis=0)
    denominator = rows[:, None] * cols[None, :]
    ratio = np.divide(
        cooccurrence * cooccurrence.sum(),
        denominator,
        out=np.zeros_like(cooccurrence),
        where=denominator > 0,
    )
    ppmi = np.log(np.maximum(ratio, 1.0))
    svd = TruncatedSVD(n_components=8, n_iter=7, random_state=SEED)
    embedding = svd.fit_transform(ppmi)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = np.divide(
        embedding, norms, out=np.zeros_like(embedding), where=norms > 0
    )
    storage = np.zeros((len(tokens), embedding.shape[1]))
    known = np.zeros(len(tokens), dtype=bool)
    for i, token in enumerate(tokens):
        if token in index:
            storage[i] = embedding[index[token]]
            known[i] = True
    artifact = {
        "method": "same-team same-game role:champion PPMI -> rank8 SVD -> L2 normalize rows",
        "vocabulary": vocab,
        "embedding": embedding,
        "svd": svd,
        "ppmi": ppmi,
        "unknown": "zero embedding, including missing champion; coverage reported, no future-vocab fit",
        "train_min": train.date.min(),
        "train_max": train.date.max(),
        "train_player_game_rows": len(train),
        "champion_names": train[["token", "champion_name"]]
        .drop_duplicates()
        .to_dict("records"),
        "seed": SEED,
        "not_intrinsic_mechanics": True,
    }
    return storage, known, artifact


def fit_ridge(x, y, masks):
    scaler = StandardScaler(with_mean=False).fit(x[masks["train"]])
    model = LogisticRegression(
        C=0.1,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-8,
        random_state=SEED,
    )
    model.fit(scaler.transform(x[masks["train"]]), y[masks["train"]])
    if int(model.n_iter_[0]) >= model.max_iter:
        raise RuntimeError("Ridge did not converge")
    logits = model.decision_function(scaler.transform(x))
    slope = fit_platt_scaling(logits[masks["calibration"]], y[masks["calibration"]])
    raw = np.clip(expit(logits), EPS, 1 - EPS)
    calibrated = np.clip(expit(slope * logits), EPS, 1 - EPS)
    swapped = np.clip(
        expit(slope * model.decision_function(scaler.transform(-x[masks["test"]]))),
        EPS,
        1 - EPS,
    )
    symmetry = float(np.max(abs(calibrated[masks["test"]] + swapped - 1)))
    if symmetry > 1e-6:
        raise ValueError("Paired side symmetry failed")
    return (
        calibrated,
        raw,
        {"scaler": scaler, "model": model, "calibration_slope": slope},
        {
            "calibration_slope": slope,
            "symmetry_max_error": symmetry,
            "iterations": int(model.n_iter_[0]),
            "stop_raw_logloss": float(
                binary_log_loss_vector(y[masks["stop"]], raw[masks["stop"]]).mean()
            ),
            "select_raw_logloss": float(
                binary_log_loss_vector(y[masks["select"]], raw[masks["select"]]).mean()
            ),
        },
    )


def compare(result, names):
    y = result.y_true.to_numpy(float)
    baseline = result.p__baseline79.to_numpy()
    metrics = {}
    for name in names:
        p = result["p__" + name].to_numpy()
        metrics[name] = {
            "aggregate": probability_metrics(y, p),
            "peryear": {},
            "paired_vs_baseline79": {
                "log_loss": monthly_bootstrap(
                    binary_log_loss_vector(y, p) - binary_log_loss_vector(y, baseline),
                    result.date,
                ),
                "brier": monthly_bootstrap(
                    (p - y) ** 2 - (baseline - y) ** 2, result.date
                ),
            },
        }
        for year, group in result.groupby(result.date.str[:4]):
            metrics[name]["peryear"][str(year)] = probability_metrics(
                group.y_true, group["p__" + name]
            )
    pdirect, pemb = (
        result.p__champion_direct.to_numpy(),
        result.p__champion_embedding.to_numpy(),
    )
    return metrics, {
        "log_loss": monthly_bootstrap(
            binary_log_loss_vector(y, pemb) - binary_log_loss_vector(y, pdirect),
            result.date,
        ),
        "brier": monthly_bootstrap((pemb - y) ** 2 - (pdirect - y) ** 2, result.date),
    }


def run(args):
    started = time.monotonic()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source_files = (
        Path(__file__),
        ROOT / "scripts/export_champion_research_data.py",
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
        ROOT / "scripts/build_siamese_research_dataset.py",
        ROOT / "src/analysis/probability_metrics.py",
        ROOT / "src/analysis/metrics.py",
        ROOT / "src/models/symmetric_series.py",
    )
    source_hashes = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        snapshot = output / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        source_hashes[str(relative)] = sha256(snapshot)
    dump(
        output / "source_snapshot_manifest.json",
        {"captured_at_run_start": True, "sha256": source_hashes},
    )
    dump(
        output / "design.json",
        {
            "seed": SEED,
            "variants": ["baseline79", "champion_direct", "champion_embedding"],
            "ridge_C": 0.1,
            "embedding_dim": 8,
            "usage_days": 90,
            "meta_days": 30,
            "performance_pseudocount": 20,
            "protocol": "semester-calibration",
            "selection": "none: locked C and embedding settings; stop/select diagnostics only",
            "historical_proxy_only": True,
            "current_draft_or_patch_inputs": False,
        },
    )
    source = args.source_dir / "champion_history.csv"
    source_manifest = json.loads((args.source_dir / "manifest.json").read_text())
    if sha256(source) != source_manifest["sha256"]:
        raise ValueError("Champion extraction hash mismatch")
    history, source_audit = audit_history(source)
    dump(output / "source_audit.json", source_audit)
    identity = {k: "string" for k in ("golgg_match_id", "team1_id", "team2_id")}
    legacy_path = ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv"
    replay_path = ROOT / "data/artifacts/model-redesign-v1/replay/snapshots.csv"
    baseline_path = (
        ROOT
        / "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/predictions.csv"
    )
    legacy = (
        pd.read_csv(legacy_path, dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    replay = (
        pd.read_csv(replay_path, dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    frame, join_audit = align_legacy_context(legacy, replay)
    baseline_columns = [
        "golgg_match_id",
        "date",
        "y_true",
        "best_of",
        "competition_tier",
        "team1_id",
        "team2_id",
        "fixed_ridge",
        "fixed_ridge__raw",
    ]
    baseline = (
        pd.read_csv(baseline_path, dtype=identity, usecols=baseline_columns)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    x79, canonical_names = build_training_features(frame)
    direct, direct_names, team_usage, player_usage, tokens, coverage = (
        historical_features(frame, history)
    )
    coverage.to_csv(output / "historical_coverage.csv", index=False)
    np.savez_compressed(
        output / "direct_features.npz",
        features=direct,
        ids=frame.golgg_match_id.to_numpy(str),
        names=np.array(direct_names),
    )
    sparse.save_npz(output / "team_usage_storage.npz", team_usage)
    sparse.save_npz(output / "lineup_usage_storage.npz", player_usage)
    dump(output / "usage_storage_tokens.json", tokens)
    y = frame.y_true.to_numpy(float)
    folds, parts = [], []
    for year in (2024, 2025, 2026):
        masks = temporal_blocks(frame.date, year, protocol="semester-calibration")
        embedding, known, encoder = fit_embedding(history, tokens, year)
        encoder_path = output / f"{year}-encoder.joblib"
        joblib.dump(encoder, encoder_path)
        embedded_pools = [
            (usage @ embedding).reshape(len(frame), 2, -1)
            for usage in (team_usage, player_usage)
        ]
        embedding_features = np.concatenate(
            [pool[:, 0] - pool[:, 1] for pool in embedded_pools], axis=1
        )
        unknown = np.asarray((team_usage[:, ~known]).sum(axis=1)).reshape(len(frame), 2)
        unknown_player = np.asarray((player_usage[:, ~known]).sum(axis=1)).reshape(
            len(frame), 2
        )
        test = masks["test"]
        part = frame.loc[
            test,
            [
                "golgg_match_id",
                "date",
                "y_true",
                "best_of",
                "competition_tier",
                "team1_id",
                "team2_id",
            ],
        ].copy()
        reference = (
            baseline.set_index("golgg_match_id").loc[part.golgg_match_id].reset_index()
        )
        for col in (
            "golgg_match_id",
            "date",
            "y_true",
            "best_of",
            "team1_id",
            "team2_id",
        ):
            if not np.array_equal(part[col].to_numpy(), reference[col].to_numpy()):
                raise ValueError(f"Frozen baseline identity mismatch: {col}")
        part["p__baseline79"] = reference.fixed_ridge.to_numpy()
        part["raw__baseline79"] = reference.fixed_ridge__raw.to_numpy()
        fold = {
            "year": year,
            "splits": {
                name: {
                    "n": int(mask.sum()),
                    "date_min": frame.loc[mask, "date"].min(),
                    "date_max": frame.loc[mask, "date"].max(),
                }
                for name, mask in masks.items()
            },
            "models": {},
            "encoder_sha256": sha256(encoder_path),
            "encoder_vocabulary_size": int(known.sum()),
            "mean_test_team_unknown_mass": float(unknown[test].mean()),
            "mean_test_lineup_unknown_mass": float(unknown_player[test].mean()),
            "test_rows_any_unknown_team_mass": int(
                (unknown[test].max(axis=1) > 0).sum()
            ),
        }
        for name, x, feature_names in (
            ("baseline79", x79, canonical_names),
            (
                "champion_direct",
                np.column_stack((x79, direct)),
                canonical_names + direct_names,
            ),
            (
                "champion_embedding",
                np.column_stack((x79, direct, embedding_features)),
                canonical_names
                + direct_names
                + [
                    f"{pool}_empirical_embedding_{i}"
                    for pool in ("team", "lineup")
                    for i in range(8)
                ],
            ),
        ):
            p, raw, artifact, metadata = fit_ridge(x, y, masks)
            artifact.update(
                feature_names=feature_names,
                odd_features_only=True,
                encoder_path=encoder_path.name
                if name == "champion_embedding"
                else None,
            )
            model_path = output / f"{year}-{name}.joblib"
            joblib.dump(artifact, model_path)
            loaded = joblib.load(model_path)
            restored = np.clip(
                expit(
                    loaded["calibration_slope"]
                    * loaded["model"].decision_function(
                        loaded["scaler"].transform(x[test][:20])
                    )
                ),
                EPS,
                1 - EPS,
            )
            np.testing.assert_array_equal(restored, p[test][:20])
            if name == "baseline79":
                gap = float(np.max(abs(p[test] - part.p__baseline79)))
                # CSV round-trips and BLAS memory layout need numerical, not bitwise agreement.
                if gap > 1e-7:
                    raise ValueError(f"Baseline reproduction differs: {gap}")
                metadata["frozen_reproduction_max_error"] = gap
            else:
                part["p__" + name] = p[test]
                part["raw__" + name] = raw[test]
            fold["models"][name] = {
                **metadata,
                "sha256": sha256(model_path),
                "feature_count": x.shape[1],
            }
            print(
                f"{year} {name} LL={binary_log_loss_vector(y[test], p[test]).mean():.6f}",
                flush=True,
            )
        parts.append(part)
        folds.append(fold)
        dump(output / f"{year}-fold.json", fold)
    result = (
        pd.concat(parts).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    )
    if result.golgg_match_id.duplicated().any() or not np.array_equal(
        result.golgg_match_id, baseline.golgg_match_id
    ):
        raise ValueError("Common test cohort changed")
    result.to_csv(output / "predictions.csv", index=False)
    metrics, incremental = compare(
        result, ["baseline79", "champion_direct", "champion_embedding"]
    )
    coverage_test = coverage.set_index("golgg_match_id").loc[result.golgg_match_id]
    coverage_summary = {
        "test_rows": len(result),
        "eligibility_loss": len(baseline) - len(result),
        "any_team_pool_missing": int(
            coverage_test[["side1_team_pool_missing", "side2_team_pool_missing"]]
            .any(axis=1)
            .sum()
        ),
        "any_lineup_pool_missing": int(
            coverage_test[["side1_lineup_pool_missing", "side2_lineup_pool_missing"]]
            .any(axis=1)
            .sum()
        ),
        "any_incomplete_prior_lineup": int(
            (coverage_test[["side1_lineup_roles", "side2_lineup_roles"]] < 5)
            .any(axis=1)
            .sum()
        ),
        "meta_patch_proxy_missing": int(coverage_test.latest_meta_missing.sum()),
        "latest_meta_age_days_quantiles": coverage_test.latest_meta_age_days.quantile(
            [0, 0.5, 0.95, 1]
        ).to_dict(),
    }
    summary = {
        "experiment": "EXP-086",
        "status": "completed",
        "variants": ["baseline79", "champion_direct", "champion_embedding"],
        "baseline_source": str(baseline_path.relative_to(ROOT))
        + " fixed_ridge; independently reproduced each year",
        "seed": SEED,
        "folds": folds,
        "join_audit": join_audit,
        "coverage": coverage_summary,
        "source_audit": source_audit,
        "metrics": metrics,
        "embedding_minus_direct": incremental,
        "bootstrap": {"resamples": 5000, "unit": "calendar month", "seed": 82},
        "limitations": [
            "Legacy rating provenance and stored EXP081 cutoff unverified; retrospective/exploratory only",
            "No historical ingestion availability; no known target draft, lineup or patch",
            "Prior-day majority lineup proxy; rolling latest-observed global meta mixes competitions and patches",
            "Embeddings are empirical teammate associations, not intrinsic champion mechanics or supervised contrastive learning",
            "2024-2026 cohort already inspected; no superiority, promotion or live-profit claim",
        ],
        "runtime_seconds": time.monotonic() - started,
        "environment": {
            "python": platform.python_version(),
            **{
                k: version(k)
                for k in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")
            },
        },
        "inputs_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                legacy_path,
                replay_path,
                baseline_path,
                source,
                args.source_dir / "manifest.json",
            )
        },
        "code_sha256": source_hashes,
        "artifacts_sha256": {
            path.name: sha256(path) for path in output.iterdir() if path.is_file()
        },
    }
    dump(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "coverage": coverage_summary,
                "metrics": {name: val["aggregate"] for name, val in metrics.items()},
                "embedding_minus_direct": incremental,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run name")
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir / "failure.json").exists():
            dump(
                args.output_dir / "failure.json",
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "code_sha256": sha256(Path(__file__)),
                },
            )
        raise


if __name__ == "__main__":
    main()
