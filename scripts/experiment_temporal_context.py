#!/usr/bin/env python3
"""EXP-084, retrospective prior-date dynamics experiment; never production.

CPU environment: uv venv /tmp/exp084-cpu --python .venv/bin/python --system-site-packages
uv pip install --python /tmp/exp084-cpu/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/exp084-cpu/bin/python numpy==2.4.4 pandas==3.0.3 scipy==1.18.1 scikit-learn==1.8.0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/exp084-cpu/bin/python scripts/experiment_temporal_context.py --run semester-seeds42-84
Fixed design: W20 maps, 2-layer 16-channel temporal convolution, 8-dimensional
embedding, 8 epochs InfoNCE or 12 supervised epochs, seeds 42/84; no holdout search.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import (
    monthly_bootstrap,
    probability_metrics,
)
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector

SOURCE = ROOT / "data/artifacts/model-redesign-v1"
INPUTS = {
    "legacy": ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv",
    "replay": SOURCE / "replay/snapshots.csv",
    "baseline": SOURCE / "benchmark-semester/legacy79/predictions.csv",
    "games": SOURCE / "source-v2/games.csv",
    "matches": SOURCE / "source-v2/matches.csv",
}
VALUE_NAMES = [
    "result",
    "kills",
    "towers",
    "dragons",
    "nashors",
    "gold_per_minute",
    "duration_minutes",
    "opponent_prior_win20",
    "opponent_prior_count20",
    "log_age_days",
    "log_gap_days",
]
WINDOW = 20
SEEDS = (42, 84)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_csv(path):
    return pd.read_csv(
        path,
        dtype={
            k: str
            for k in ("golgg_match_id", "match_id", "game_id", "team1_id", "team2_id")
        },
        low_memory=False,
    )


def audit_games(games, matches):
    if games.game_id.duplicated().any() or matches.match_id.duplicated().any():
        raise ValueError("nonunique source identities")
    joined = games.merge(
        matches[["match_id", "date", "team1_id", "team2_id"]],
        on="match_id",
        how="left",
        suffixes=("", "_match"),
        validate="many_to_one",
    )
    same = (joined.team1_id == joined.team1_id_match) & (
        joined.team2_id == joined.team2_id_match
    )
    reverse = (joined.team1_id == joined.team2_id_match) & (
        joined.team2_id == joined.team1_id_match
    )
    valid = (same | reverse) & (joined.date == joined.date_match)
    valid &= joined.team1_win.isin([0, 1]) & (joined.team1_win + joined.team2_win == 1)
    rejected = joined.loc[
        ~valid,
        [
            "game_id",
            "match_id",
            "date",
            "team1_id",
            "team2_id",
            "team1_id_match",
            "team2_id_match",
        ],
    ].copy()
    return games.loc[valid.to_numpy()].sort_values(
        ["date", "match_id", "game_id"]
    ), rejected


def build_histories(frame, games):
    """Freeze ALL target rows of a date before consuming ANY game on that date.

    Within-date game order uses stable source identifiers, not inferred clock time.
    Opponent context attached to a historical event is itself frozen before its day.
    """
    n, d = len(frame), len(VALUE_NAMES)
    values = np.zeros((n, 2, WINDOW, d), np.float32)
    missing = np.ones_like(values, dtype=bool)
    padding = np.ones((n, 2, WINDOW), dtype=bool)
    event_ids = np.full((n, 2, WINDOW), -1, np.int64)
    event_dates = np.full((n, 2, WINDOW), -1, np.int32)
    histories = defaultdict(lambda: deque(maxlen=WINDOW))
    game_groups = {day: group for day, group in games.groupby("date", sort=False)}
    targets = {day: group for day, group in frame.groupby("date", sort=False)}
    for day in sorted(set(game_groups) | set(targets)):
        ordinal = pd.Timestamp(day).toordinal()
        if day in targets:
            for row in targets[day].itertuples():
                for side, team in enumerate((row.team1_id, row.team2_id)):
                    history = histories[team]
                    for slot, (date, gid, v, m) in enumerate(
                        history, WINDOW - len(history)
                    ):
                        if date >= ordinal:
                            raise ValueError("target-day history leakage")
                        values[row.Index, side, slot] = v
                        values[row.Index, side, slot, -2] = np.log1p(ordinal - date)
                        missing[row.Index, side, slot] = m
                        padding[row.Index, side, slot] = False
                        event_ids[row.Index, side, slot] = gid
                        event_dates[row.Index, side, slot] = date
        if day not in game_groups:
            continue
        pending = []
        for row in game_groups[day].itertuples(index=False):
            for side, (team, opponent) in enumerate(
                ((row.team1_id, row.team2_id), (row.team2_id, row.team1_id)), 1
            ):
                stats = json.loads(getattr(row, f"team{side}_stats_json"))
                previous = histories[team]
                opponent_history = histories[opponent]
                duration = float(row.game_duration) / 60
                raw = [getattr(row, f"team{side}_win")]
                raw.extend(
                    stats.get(k) for k in ("kills", "towers", "dragons", "nashors")
                )
                raw.extend(
                    [
                        stats.get("gold") / duration
                        if stats.get("gold") is not None and duration > 0
                        else None,
                        duration if duration > 0 else None,
                        np.mean([h[2][0] for h in opponent_history])
                        if opponent_history
                        else None,
                        len(opponent_history),
                        0.0,
                        np.log1p(ordinal - previous[-1][0]) if previous else None,
                    ]
                )
                mask = np.array([v is None or not np.isfinite(v) for v in raw])
                val = np.array([0.0 if m else v for v, m in zip(raw, mask)], np.float32)
                pending.append((team, (ordinal, int(row.game_id), val, mask)))
        for team, event in pending:
            histories[team].append(event)
    return values, missing, padding, event_ids, event_dates


def history_guard():
    """Executable future/same-day mutation and padding guard on actual builder."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "team1_id": ["a", "a"],
            "team2_id": ["b", "b"],
        }
    )
    rows = []
    for gid, day in enumerate(("2024-01-01", "2024-01-02", "2024-01-03"), 1):
        rows.append(
            {
                "game_id": str(gid),
                "match_id": str(gid),
                "date": day,
                "team1_id": "a",
                "team2_id": "b",
                "team1_win": 1,
                "team2_win": 0,
                "game_duration": 1800,
                "team1_stats_json": '{"kills": 10}',
                "team2_stats_json": '{"kills": 5}',
            }
        )
    source = pd.DataFrame(rows)
    original = build_histories(frame, source)
    mutated = source.copy()
    mutated.loc[mutated.date >= "2024-01-02", "team1_win"] = 0
    mutated.loc[mutated.date >= "2024-01-02", "team1_stats_json"] = '{"kills": 999999}'
    changed = build_histories(frame, mutated)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(original, changed))
    assert not np.array_equal(original[0][1], changed[0][1])
    assert original[2][0, 0].sum() == 19 and original[1][0, 0, -1, 2]
    return {
        "future_and_same_day_mutation": "passed",
        "historical_mutation_sensitivity": "passed",
        "padding_and_missing_masks": "passed",
    }


def fit_history_scaler(values, missing, padding, train):
    observed = (~missing[train]) & (~padding[train, ..., None])
    count = observed.sum(axis=(0, 1, 2))
    mean = (values[train] * observed).sum(axis=(0, 1, 2)) / np.maximum(count, 1)
    variance = (((values[train] - mean) ** 2) * observed).sum(
        axis=(0, 1, 2)
    ) / np.maximum(count, 1)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1
    standard = np.where(missing, 0, (values - mean) / scale)
    standard = np.clip(standard, -10, 10).astype(np.float32)
    standard[padding] = 0
    z = np.concatenate(
        [standard, missing.astype(np.float32), (~padding)[..., None]], axis=-1
    )
    return z, mean, scale


def simple_features(z):
    valid = z[..., -1:]
    vals = z[..., :-1]
    count = np.maximum(valid.sum(axis=2), 1)
    avg = (vals * valid).sum(axis=2) / count
    weights = np.exp(np.arange(-WINDOW + 1, 1) / 5)[None, None, :, None] * valid
    ewma = (vals * weights).sum(axis=2) / np.maximum(weights.sum(axis=2), 1e-6)
    short = (vals[:, :, -5:] * valid[:, :, -5:]).sum(axis=2) / np.maximum(
        valid[:, :, -5:].sum(axis=2), 1
    )
    team = np.concatenate([avg, ewma, short - avg, count / WINDOW], axis=-1)
    return team[:, 0] - team[:, 1]


class TemporalEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 16, 3, padding=2)
        self.conv2 = nn.Conv1d(16, 16, 3, padding=2)
        self.project = nn.Linear(32, 8)

    def forward(self, window):
        valid = window[..., -1:]
        h = F.gelu(self.conv1(window.transpose(1, 2))[..., :WINDOW])
        h = F.gelu(self.conv2(h)[..., :WINDOW]).transpose(1, 2) * valid
        pooled = h.sum(1) / valid.sum(1).clamp_min(1)
        return self.project(torch.cat([pooled, h[:, -1]], dim=1))


def augment(window):
    view = window.clone()
    # Crop old prefix and independently mask observed event contents; padding stays explicit.
    crop = torch.randint(0, 6, (len(view), 1))
    keep = torch.arange(WINDOW)[None, :] >= crop
    mask = (torch.rand(len(view), WINDOW) < 0.15) & keep
    view[~keep] = 0
    view[mask] = 0
    return view


def contrastive_loss(encoder, window):
    a, b = (
        F.normalize(encoder(augment(window)), dim=1),
        F.normalize(encoder(augment(window)), dim=1),
    )
    scores = a @ b.T / 0.2
    labels = torch.arange(len(a))
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def odd_embedding(embedding, base):
    difference = embedding[:, 0] - embedding[:, 1]
    # Only even contexts multiply differences; exchanging sides exactly negates output.
    even = np.mean(abs(base[:, :12]), axis=1, keepdims=True)
    return np.column_stack([difference, difference * even])


def encode_all(encoder, windows):
    encoder.eval()
    with torch.no_grad():
        return (
            torch.cat(
                [
                    encoder(t)
                    for t in windows.reshape(-1, WINDOW, windows.shape[-1]).split(1024)
                ]
            )
            .numpy()
            .reshape(-1, 2, 8)
        )


def train_encoder(kind, seed, windows, base, y, masks, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = TemporalEncoder(windows.shape[-1])
    if kind == "random":
        return encoder, {"epochs": 0, "best_stop_loss": None}
    train_indices, stop_indices = (
        np.flatnonzero(masks["train"]),
        np.flatnonzero(masks["stop"]),
    )
    if kind == "contrastive":
        pool = windows[train_indices].reshape(-1, WINDOW, windows.shape[-1])
        # Deduplicate identical train windows so exact repeated windows are not negatives.
        valid = pool[..., -1].sum(1).numpy() >= 2
        pool = pool[valid]
        flat = pool.numpy().reshape(len(pool), -1)
        _, unique = np.unique(flat, axis=0, return_index=True)
        pool = pool[np.sort(unique)]
        held = windows[stop_indices].reshape(-1, WINDOW, windows.shape[-1])
        held = held[held[..., -1].sum(1) >= 2]
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=0.001, weight_decay=0.001
        )
        epochs = 8
        head = None
    else:
        head = nn.Linear(base.shape[1] + 16, 1, bias=False)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=0.001,
            weight_decay=0.01,
        )
        epochs = 12
        xb = torch.tensor(base, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)

        def supervised(indices):
            emb = encoder(
                windows[indices].reshape(-1, WINDOW, windows.shape[-1])
            ).reshape(-1, 2, 8)
            d = emb[:, 0] - emb[:, 1]
            even = abs(xb[indices, :12]).mean(1, keepdim=True)
            features = torch.cat([xb[indices], d, d * even], dim=1)
            return F.binary_cross_entropy_with_logits(
                head(features).ravel(), labels[indices]
            )

    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        encoder.train()
        order = rng.permutation(len(pool) if kind == "contrastive" else train_indices)
        losses = []
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            if len(indices) < 2:
                continue
            optimizer.zero_grad()
            loss = (
                contrastive_loss(encoder, pool[indices])
                if kind == "contrastive"
                else supervised(indices)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        encoder.eval()
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(seed + 10000)
            if kind == "contrastive":
                stop_loss = np.mean(
                    [
                        float(contrastive_loss(encoder, batch))
                        for batch in held.split(256)
                        if len(batch) > 1
                    ]
                )
            else:
                stop_loss = np.average(
                    [
                        float(supervised(i))
                        for i in np.array_split(
                            stop_indices, max(1, len(stop_indices) // 512)
                        )
                    ]
                )
        log(
            {
                "kind": kind,
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "stop_loss": float(stop_loss),
            }
        )
        if stop_loss < best:
            best, best_state, best_epoch = (
                float(stop_loss),
                copy.deepcopy(encoder.state_dict()),
                epoch + 1,
            )
    encoder.load_state_dict(best_state)
    return encoder, {
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "best_stop_loss": best,
        "pretrain_unique_windows": len(pool) if kind == "contrastive" else None,
        "supervised_head_discarded": kind == "supervised",
    }


def fit_downstream(x, y, masks, out, name, year):
    scale = x[masks["train"]].std(axis=0)
    scale[scale < 1e-8] = 1
    scaled = x / scale
    model = LogisticRegression(
        C=0.1, fit_intercept=False, max_iter=2000, solver="lbfgs"
    ).fit(scaled[masks["train"]], y[masks["train"]])
    if model.n_iter_.max() >= 2000:
        raise ValueError("downstream failed to converge")
    logits = model.decision_function(scaled)
    slope = fit_platt_scaling(logits[masks["calibration"]], y[masks["calibration"]])
    test_logits = logits[masks["test"]]
    p = np.clip(expit(slope * test_logits), 1e-12, 1 - 1e-12)
    reversed_p = expit(slope * model.decision_function(-scaled[masks["test"]]))
    error = float(np.max(abs(p + reversed_p - 1)))
    if error > 1e-6:
        raise ValueError("paired probability symmetry failed")
    np.savez_compressed(
        out / f"{year}-{name}-downstream.npz",
        scale=scale,
        coef=model.coef_,
        slope=slope,
    )
    return (
        p,
        np.clip(expit(test_logits), 1e-12, 1 - 1e-12),
        {
            "C": 0.1,
            "intercept": False,
            "n_features": x.shape[1],
            "slope": slope,
            "symmetry_max_error": error,
            "select_log_loss_raw": float(
                binary_log_loss_vector(
                    y[masks["select"]], expit(logits[masks["select"]])
                ).mean()
            ),
        },
    )


def run_replacement(args):
    """Fixed post-null-result ablation: REMOVE all legacy W20-derived inputs."""
    parent = args.replacement_from.resolve()
    out = ROOT / "data/artifacts/exp084-temporal-context" / args.run
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(1)

    def log(value):
        value["elapsed_seconds"] = round(time.time() - start, 2)
        with (out / "run.jsonl").open("a") as handle:
            handle.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    try:
        old = json.loads((parent / "summary.json").read_text())
        source_paths = [
            Path(__file__),
            ROOT / "scripts/benchmark_model_redesign.py",
            ROOT / "scripts/benchmark_siamese_architectures.py",
            ROOT / "scripts/train_and_tune_siamese_series.py",
            ROOT / "scripts/build_siamese_research_dataset.py",
            ROOT / "src/models/symmetric_series.py",
            ROOT / "src/analysis/probability_metrics.py",
            ROOT / "src/analysis/metrics.py",
        ]
        code_hashes = {}
        for source in source_paths:
            relative = str(source.relative_to(ROOT))
            code_hashes[relative] = sha(source)
            (out / ("source__" + relative.replace("/", "__"))).write_bytes(
                source.read_bytes()
            )
        frame, audit = align_legacy_context(
            load_csv(INPUTS["legacy"]), load_csv(INPUTS["replay"])
        )
        frame = frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
        history_rows = load_csv(parent / "history_rows.csv")
        if not frame[history_rows.columns].equals(history_rows):
            raise ValueError("replacement history row/side/date alignment differs")
        for key in ("legacy", "replay", "baseline"):
            if sha(INPUTS[key]) != old["hashes"][key]:
                raise ValueError("replacement input differs from parent run")
        predictions = load_csv(parent / "predictions.csv")
        targets = frame.loc[
            frame.date >= "2024-01-01", ["golgg_match_id", "date", "y_true", "best_of"]
        ].reset_index(drop=True)
        if not targets.equals(predictions[targets.columns]) or len(targets) != 8860:
            raise ValueError("replacement target contract changed")
        base, names = build_training_features(frame)
        keep = np.array([not name.startswith("w20_") for name in names])
        removed = [name for name, retained in zip(names, keep) if not retained]
        if len(removed) != 10 or int(keep.sum()) != 69:
            raise ValueError(
                "canonical W20 schema changed; audit replacement projection"
            )
        # Prove no hidden W20 interaction survives the name-based projection.
        mutated = frame.copy()
        rolling_columns = [
            c for c in frame if c.startswith(("t1_rolling_", "t2_rolling_"))
        ]
        mutated[rolling_columns] = 123456.0
        changed, changed_names = build_training_features(mutated)
        if changed_names != names or not np.array_equal(
            base[:, keep], changed[:, keep]
        ):
            raise ValueError("legacy W20 mutation affects retained replacement inputs")
        if np.array_equal(base[:, ~keep], changed[:, ~keep]):
            raise ValueError("W20 mutation guard did not exercise removed features")
        ratings = base[:, keep]
        y = frame.y_true.to_numpy(float)
        histories = np.load(parent / "histories.npz")
        original_manifest = json.loads((parent / "manifest.json").read_text())
        if sha(parent / "histories.npz") != original_manifest["histories.npz"]:
            raise ValueError("parent histories mutated")
        all_predictions, models, splits = defaultdict(list), {}, {}
        dependency_hashes = {
            name: sha(parent / name)
            for name in (
                "histories.npz",
                "history_rows.csv",
                "predictions.csv",
                "summary.json",
            )
        }
        guards = {
            "legacy_w20_mutation_invariance": "passed",
            "removed_w20_mutation_sensitivity": "passed",
            "parent_history_manifest": "passed",
            "parent_temporal_guards": old["guards"],
        }
        log({"stage": "replacement_ready", "removed_features": removed, "retained": 69})
        for year in (2024, 2025, 2026):
            masks = temporal_blocks(frame.date, year, protocol="semester-calibration")
            splits[str(year)] = {
                key: {
                    "n": int(mask.sum()),
                    "min_date": frame.loc[mask, "date"].min(),
                    "max_date": frame.loc[mask, "date"].max(),
                }
                for key, mask in masks.items()
            }
            scaler_name = f"{year}-history-scaler.npz"
            scaler = np.load(parent / scaler_name)
            values, missing, padding = (
                histories["values"],
                histories["missing"],
                histories["padding"],
            )
            z = np.where(missing, 0, (values - scaler["mean"]) / scaler["scale"])
            z = np.clip(z, -10, 10).astype(np.float32)
            z[padding] = 0
            z = np.concatenate(
                [z, missing.astype(np.float32), (~padding)[..., None]], axis=-1
            )
            (out / scaler_name).write_bytes((parent / scaler_name).read_bytes())
            dependency_hashes[scaler_name] = sha(parent / scaler_name)
            rating_scale = ratings[masks["train"]].std(axis=0)
            rating_scale[rating_scale < 1e-8] = 1
            normalized_ratings = ratings / rating_scale
            np.savez_compressed(
                out / f"{year}-replacement-rating-scaler.npz",
                scale=rating_scale,
                keep=keep,
            )
            simple = simple_features(z)
            if not np.allclose(simple_features(z[:, ::-1]), -simple, atol=1e-6):
                raise ValueError("replacement recency is not antisymmetric")
            designs = {
                "temporal_replace_rating69": ratings,
                "temporal_replace_recency": np.column_stack([ratings, simple]),
            }
            windows = torch.tensor(z)
            for kind in ("random", "contrastive"):
                for seed in SEEDS:
                    name = f"temporal_replace_{kind}_s{seed}"
                    filename = f"{year}-temporal_{kind}_s{seed}-encoder.pt"
                    if sha(parent / filename) != original_manifest[filename]:
                        raise ValueError("parent encoder mutated")
                    encoder = TemporalEncoder(z.shape[-1])
                    encoder.load_state_dict(
                        torch.load(parent / filename, weights_only=True)
                    )
                    embedding = encode_all(encoder, windows)
                    extra = odd_embedding(embedding, normalized_ratings)
                    if not np.allclose(
                        extra,
                        -odd_embedding(embedding[:, ::-1], -normalized_ratings),
                        atol=1e-6,
                    ):
                        raise ValueError("replacement embedding violates side symmetry")
                    designs[name] = np.column_stack([ratings, extra])
                    (out / filename).write_bytes((parent / filename).read_bytes())
                    dependency_hashes[filename] = sha(parent / filename)
            for name, x in designs.items():
                p, raw, metadata = fit_downstream(x, y, masks, out, name, year)
                all_predictions[name].append(p)
                all_predictions["raw__" + name].append(raw)
                models[f"{year}-{name}"] = metadata
                log(
                    {
                        "stage": "replacement_fit",
                        "year": year,
                        "variant": name,
                        "slope": metadata["slope"],
                    }
                )
        for name, arrays in all_predictions.items():
            predictions[name if name.startswith("raw__") else "p__" + name] = (
                np.concatenate(arrays)
            )
        for kind in ("random", "contrastive"):
            for prefix in ("p__", "raw__"):
                predictions[f"{prefix}temporal_replace_{kind}_ensemble"] = predictions[
                    [f"{prefix}temporal_replace_{kind}_s{seed}" for seed in SEEDS]
                ].mean(axis=1)
        columns = [c for c in predictions if c.startswith(("p__", "raw__"))]
        if (
            predictions.golgg_match_id.duplicated().any()
            or not ((predictions[columns] > 0) & (predictions[columns] < 1)).all().all()
        ):
            raise ValueError("invalid replacement output identity/probability")
        predictions.to_csv(out / "predictions.csv", index=False)
        variant_names = [c for c in predictions if c.startswith("p__")]
        new_names = [c for c in variant_names if c.startswith("p__temporal_replace_")]
        metrics, paired = {}, {}
        for label, cohort in [("all", predictions)] + [
            (str(year), predictions[predictions.date.str[:4] == str(year)])
            for year in (2024, 2025, 2026)
        ]:
            metrics[label] = {
                name: probability_metrics(cohort.y_true, cohort[name])
                for name in variant_names
            }
            paired[label] = {
                reference: {
                    name: {
                        "log_loss": monthly_bootstrap(
                            binary_log_loss_vector(cohort.y_true, cohort[name])
                            - binary_log_loss_vector(cohort.y_true, cohort[reference]),
                            cohort.date,
                        ),
                        "brier": monthly_bootstrap(
                            (cohort[name] - cohort.y_true) ** 2
                            - (cohort[reference] - cohort.y_true) ** 2,
                            cohort.date,
                        ),
                    }
                    for name in new_names
                    if name != reference
                }
                for reference in ("p__baseline79", "p__temporal_replace_rating69")
            }
        result = {
            "experiment": "EXP-084-W20-replacement",
            "status": "completed",
            "interpretation": "Fixed exploratory follow-up declared AFTER original augmentation null result; no holdout tuning or promotion.",
            "parent_run": str(parent.relative_to(ROOT)),
            "parent_dependency_hashes": dependency_hashes,
            "variants": variant_names,
            "new_variants": new_names,
            "features": {
                "retained_rating_rest_format": [n for n, k in zip(names, keep) if k],
                "removed_all_legacy_w20": removed,
                "projection_guard": "All raw t1/t2 rolling fields replaced with constant; retained69 bit-identical",
            },
            "representation_reuse": "Random encoders untrained; InfoNCE encoders trained only on historical sports windows. No supervised W20-assisted encoder reused. Original stopping/scalers retained.",
            "training": {
                "seeds": SEEDS,
                "encoder_retraining": False,
                "downstream_C": 0.1,
                "downstream_tol": 1e-4,
                "max_iter": 2000,
                "fit_intercept": False,
                "hyperselection": "None",
                "calibration": "Separate positive zero-intercept slope per model/year, calibration block only",
                "ensembles": "Mean of two separately calibrated probabilities",
            },
            "split_protocol": "semester-calibration",
            "splits": splits,
            "coverage": {"test_rows": 8860, "eligibility_loss": 0},
            "join_audit": audit,
            "guards": guards,
            "models": models,
            "metrics": metrics,
            "paired_vs_references": paired,
            "hashes": {k: sha(v) for k, v in INPUTS.items()},
            "code_hashes": code_hashes,
            "provenance_blockers": old["provenance_blockers"],
            "environment": old["environment"],
            "elapsed_seconds": time.time() - start,
        }
        save_json(out / "summary.json", result)
        log(
            {
                "stage": "completed",
                "aggregate_log_loss": {
                    name: metrics["all"][name]["log_loss"] for name in new_names
                },
            }
        )
        save_json(
            out / "manifest.json",
            {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        )
    except Exception as exc:
        save_json(out / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--replacement-from",
        type=Path,
        help="Run separately labeled W20-removal follow-up using this frozen original run",
    )
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("run must be a single directory name")
    if args.replacement_from:
        run_replacement(args)
        return
    out = ROOT / "data/artifacts/exp084-temporal-context" / args.run
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(1)

    def log(value):
        value["elapsed_seconds"] = round(time.time() - start, 2)
        with (out / "run.jsonl").open("a") as f:
            f.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    try:
        guards = history_guard()
        frame, join_audit = align_legacy_context(
            load_csv(INPUTS["legacy"]), load_csv(INPUTS["replay"])
        )
        frame = frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
        baseline = (
            load_csv(INPUTS["baseline"])
            .sort_values(["date", "golgg_match_id"])
            .reset_index(drop=True)
        )
        expected = frame.loc[
            frame.date >= "2024-01-01", ["golgg_match_id", "date", "y_true", "best_of"]
        ].reset_index(drop=True)
        if not expected.equals(baseline[expected.columns]) or len(expected) != 8860:
            raise ValueError("common 8860-row target contract changed")
        if (
            not frame.set_index("golgg_match_id")
            .loc[baseline.golgg_match_id, ["team1_id", "team2_id"]]
            .reset_index(drop=True)
            .equals(baseline[["team1_id", "team2_id"]])
        ):
            raise ValueError("baseline side identity differs")
        games, rejected = audit_games(
            load_csv(INPUTS["games"]), load_csv(INPUTS["matches"])
        )
        rejected.to_csv(out / "rejected_history_games.csv", index=False)
        values, missing, padding, event_ids, event_dates = build_histories(frame, games)
        ordinal = pd.to_datetime(frame.date).map(pd.Timestamp.toordinal).to_numpy()
        if np.any((event_dates >= ordinal[:, None, None]) & ~padding):
            raise ValueError("history source date is not strictly prior")
        np.savez_compressed(
            out / "histories.npz",
            values=values,
            missing=missing,
            padding=padding,
            game_ids=event_ids,
            date_ordinals=event_dates,
        )
        frame[["golgg_match_id", "date", "team1_id", "team2_id"]].to_csv(
            out / "history_rows.csv", index=False
        )
        # Real-source future perturbation guard, including target-date outcomes/stats.
        cutoff = "2024-01-14"
        sentinel = frame.loc[frame.date == cutoff].reset_index(drop=True)
        original = build_histories(sentinel, games)
        altered = games.copy()
        altered.loc[altered.date >= cutoff, "team1_win"] = (
            1 - altered.loc[altered.date >= cutoff, "team1_win"]
        )
        altered.loc[altered.date >= cutoff, "team1_stats_json"] = '{"kills": 999999}'
        future = build_histories(sentinel, altered)
        if not all(np.array_equal(a, b) for a, b in zip(original, future)):
            raise ValueError("future source mutation changed sentinel history")
        guards["real_source_future_mutation"] = {
            "status": "passed",
            "cutoff": cutoff,
            "sentinel_rows": len(sentinel),
        }
        base, base_names = build_training_features(frame)
        y = frame.y_true.to_numpy(float)
        predictions = baseline[
            ["golgg_match_id", "date", "y_true", "best_of", "competition_tier"]
        ].copy()
        predictions["p__baseline79"] = baseline.fixed_ridge
        predictions["raw__baseline79"] = baseline.fixed_ridge__raw
        all_predictions, models, splits = defaultdict(list), {}, {}
        log(
            {
                "stage": "histories_ready",
                "rows": len(frame),
                "rejected_games": len(rejected),
                "guards": guards,
            }
        )
        for year in (2024, 2025, 2026):
            masks = temporal_blocks(frame.date, year, protocol="semester-calibration")
            splits[str(year)] = {
                k: {
                    "n": int(v.sum()),
                    "min_date": frame.loc[v, "date"].min(),
                    "max_date": frame.loc[v, "date"].max(),
                }
                for k, v in masks.items()
            }
            z, mean, hscale = fit_history_scaler(
                values, missing, padding, masks["train"]
            )
            np.savez_compressed(
                out / f"{year}-history-scaler.npz", mean=mean, scale=hscale
            )
            base_scale = base[masks["train"]].std(axis=0)
            base_scale[base_scale < 1e-8] = 1
            normalized_base = base / base_scale
            np.savez_compressed(
                out / f"{year}-supervised-base-scaler.npz", scale=base_scale
            )
            simple = simple_features(z)
            if not np.allclose(simple_features(z[:, ::-1]), -simple, atol=1e-6):
                raise ValueError("simple features violate side symmetry")
            designs = {
                "temporal_canonical79": base,
                "temporal_recency": np.column_stack([base, simple]),
            }
            windows = torch.tensor(z)
            for kind in ("random", "supervised", "contrastive"):
                for seed in SEEDS:
                    name = f"temporal_{kind}_s{seed}"
                    encoder, meta = train_encoder(
                        kind,
                        seed,
                        windows,
                        normalized_base,
                        y,
                        masks,
                        lambda v: log({"year": year, **v}),
                    )
                    torch.save(encoder.state_dict(), out / f"{year}-{name}-encoder.pt")
                    embedding = encode_all(encoder, windows)
                    extra = odd_embedding(embedding, normalized_base)
                    reverse_extra = odd_embedding(embedding[:, ::-1], -normalized_base)
                    if not np.allclose(extra, -reverse_extra, atol=1e-6):
                        raise ValueError("embedding interactions violate side symmetry")
                    designs[name] = np.column_stack([base, extra])
                    models[f"{year}-{name}-encoder"] = meta
            for name, x in designs.items():
                p, raw, meta = fit_downstream(x, y, masks, out, name, year)
                all_predictions[name].append(p)
                all_predictions["raw__" + name].append(raw)
                models[f"{year}-{name}"] = meta
                log(
                    {
                        "year": year,
                        "variant": name,
                        "stage": "prediction_saved",
                        "slope": meta["slope"],
                    }
                )
        for name, arrays in all_predictions.items():
            predictions[name if name.startswith("raw__") else "p__" + name] = (
                np.concatenate(arrays)
            )
        for kind in ("random", "supervised", "contrastive"):
            for prefix in ("p__", "raw__"):
                predictions[f"{prefix}temporal_{kind}_ensemble"] = predictions[
                    [f"{prefix}temporal_{kind}_s{s}" for s in SEEDS]
                ].mean(axis=1)
        if predictions.golgg_match_id.duplicated().any():
            raise ValueError("duplicate output identity")
        probability_columns = [c for c in predictions if c.startswith(("p__", "raw__"))]
        if (
            not (
                (predictions[probability_columns] > 0)
                & (predictions[probability_columns] < 1)
            )
            .all()
            .all()
        ):
            raise ValueError("invalid output probabilities")
        predictions.to_csv(out / "predictions.csv", index=False)
        names = [c for c in predictions if c.startswith("p__")]
        metrics, paired = {}, {}
        for label, cohort in [("all", predictions)] + [
            (str(year), predictions[predictions.date.str[:4] == str(year)])
            for year in (2024, 2025, 2026)
        ]:
            metrics[label] = {
                name: probability_metrics(cohort.y_true, cohort[name]) for name in names
            }
            paired[label] = {
                name: {
                    "log_loss": monthly_bootstrap(
                        binary_log_loss_vector(cohort.y_true, cohort[name])
                        - binary_log_loss_vector(cohort.y_true, cohort.p__baseline79),
                        cohort.date,
                    ),
                    "brier": monthly_bootstrap(
                        (cohort[name] - cohort.y_true) ** 2
                        - (cohort.p__baseline79 - cohort.y_true) ** 2,
                        cohort.date,
                    ),
                }
                for name in names
                if name != "p__baseline79"
            }
        # Attribution comparisons are paired on the same test rows, not winner selection.
        attribution = {}
        for left, right in [
            ("temporal_recency", "temporal_canonical79"),
            ("temporal_contrastive_ensemble", "temporal_random_ensemble"),
            ("temporal_supervised_ensemble", "temporal_random_ensemble"),
        ]:
            attribution[f"{left}_minus_{right}"] = monthly_bootstrap(
                binary_log_loss_vector(predictions.y_true, predictions["p__" + left])
                - binary_log_loss_vector(
                    predictions.y_true, predictions["p__" + right]
                ),
                predictions.date,
            )
        summary = {
            "experiment": "EXP-084",
            "status": "completed",
            "variants": names,
            "interpretation": "Retrospective exploratory; no production promotion or live-profit claim.",
            "provenance_blockers": [
                "Legacy rating source-time provenance UNVERIFIED",
                "EXP081 cutoff UNVERIFIED",
                "source-v2 export is retrospective, not point-in-time archival availability",
                "same-day historical maps use stable identifier order, not verified within-day timestamps",
            ],
            "baseline_source": str(INPUTS["baseline"].relative_to(ROOT))
            + ":fixed_ridge and fixed_ridge__raw",
            "join_audit": join_audit,
            "coverage": {
                "test_rows": len(predictions),
                "eligibility_loss": 0,
                "rejected_history_games": len(rejected),
                "empty_team_histories": int(padding.all(axis=2).sum()),
                "partial_team_histories": int(
                    (padding.any(axis=2) & ~padding.all(axis=2)).sum()
                ),
                "missing_rate_nonpadding_by_feature": dict(
                    zip(
                        VALUE_NAMES,
                        (missing & ~padding[..., None])
                        .sum(axis=(0, 1, 2))
                        .astype(float)
                        / max(int((~padding).sum()), 1),
                    )
                ),
            },
            "guards": guards,
            "split_protocol": "semester-calibration",
            "splits": splits,
            "features": {
                "canonical": base_names,
                "history_values": VALUE_NAMES,
                "window": WINDOW,
                "history_masks": "per-value missing + per-event padding",
                "simple": "mean, EWMA decay5, last5-minus20 trend, coverage difference",
                "encoder": "Conv1d channels23->16->16 kernel3 causal left context + mean/last pooling ->8",
                "interactions": "embedding side difference and difference * mean(abs(canonical first12 standardized))",
            },
            "training": {
                "seeds": SEEDS,
                "batch_size": 256,
                "contrastive_epochs": 8,
                "supervised_epochs": 12,
                "optimizer": "AdamW",
                "encoder_lr": 0.001,
                "contrastive_weight_decay": 0.001,
                "supervised_weight_decay": 0.01,
                "downstream_tol": 1e-4,
                "downstream_max_iter": 2000,
                "contrastive": "symmetric cross-view InfoNCE temperature0.2; same historical window positive; random old-prefix crop0..5 and event mask15%; exact train-window deduplication",
                "hyperselection": "None; fixed bounded architectures/C=0.1; select block raw LL diagnostic only",
                "early_stopping": "minimum designated stop-block objective across fixed epochs",
                "calibration": "independent positive zero-intercept Platt slope on calibration block per seed",
                "ensemble": "arithmetic mean of separately calibrated seed probabilities; no additional fitted calibration",
            },
            "models": models,
            "metrics": metrics,
            "paired_vs_baseline79": paired,
            "attribution_log_loss": attribution,
            "hashes": {name: sha(path) for name, path in INPUTS.items()},
            "code_hashes": {
                str(path.relative_to(ROOT)): sha(path)
                for path in [
                    Path(__file__),
                    ROOT / "scripts/benchmark_model_redesign.py",
                    ROOT / "scripts/benchmark_siamese_architectures.py",
                    ROOT / "scripts/train_and_tune_siamese_series.py",
                    ROOT / "src/models/symmetric_series.py",
                    ROOT / "src/analysis/probability_metrics.py",
                    ROOT / "src/analysis/metrics.py",
                ]
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
            },
            "elapsed_seconds": time.time() - start,
        }
        save_json(out / "summary.json", summary)
        log(
            {
                "stage": "completed",
                "aggregate_log_loss": {
                    k: v["log_loss"] for k, v in metrics["all"].items()
                },
            }
        )
        save_json(
            out / "manifest.json",
            {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        )
    except Exception as exc:
        save_json(
            out / "failure.json",
            {
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.time() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()
