"""EXP-055: leakage-safe transformer aggregation of player-game embeddings.

This script tests whether a small sequence model over recent player-game
embeddings improves over the simple mean/std aggregation used in EXP-050/054.
For a match at date T, each side receives only embeddings from games strictly
before T. Histories are updated after all matches with the same date are emitted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db
from betting_app.ml.training.player_embedding_match_dataset import _target_from_match, _team_key, encode_player_game_embeddings
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.player_game_encoder import require_torch
from betting_app.ml.training.strength_dataset import load_golgg_match_results


@dataclass(frozen=True)
class SequenceConfig:
    seq_len: int = 40
    min_prior_events: int = 20
    max_examples: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-artifact", required=True)
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--limit-player-rows", type=int, default=None)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--min-prior-events", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)) if len(y) else None,
    }


def _load_or_encode_embeddings(args: argparse.Namespace) -> pd.DataFrame:
    if args.embedding_cache and args.embedding_cache.exists():
        return pd.read_parquet(args.embedding_cache)
    player_dataset = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_player_rows)
    )
    embeddings = encode_player_game_embeddings(
        player_dataset,
        encoder_artifact=Path(args.encoder_artifact),
        device=args.device,
        batch_size=args.embedding_batch_size,
    )
    if args.embedding_cache:
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        embeddings.to_parquet(args.embedding_cache, index=False)
    return embeddings


def build_sequence_examples(matches: pd.DataFrame, player_embeddings: pd.DataFrame, cfg: SequenceConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], utc=True, errors="coerce")
    matches["match_id"] = matches["match_id"].astype(str)
    matches = matches.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    emb = player_embeddings.copy()
    emb["match_id"] = emb["match_id"].astype(str)
    emb_cols = [c for c in emb.columns if c.startswith("embedding_")]
    if not emb_cols:
        raise RuntimeError("No embedding_* columns found")
    by_match = {mid: g for mid, g in emb.groupby("match_id", sort=False)}
    histories: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=max(cfg.seq_len, 1)))

    x1: list[np.ndarray] = []
    x2: list[np.ndarray] = []
    y: list[int] = []
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    def team_seq(key: str) -> tuple[np.ndarray | None, int]:
        values = list(histories.get(key, ()))
        count = len(values)
        if count < cfg.min_prior_events:
            return None, count
        arr = np.asarray(values[-cfg.seq_len :], dtype=np.float32)
        if len(arr) < cfg.seq_len:
            pad = np.zeros((cfg.seq_len - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr, count

    for _, date_group in matches.groupby("date", sort=True):
        pending: list[str] = []
        for rec in date_group.to_dict(orient="records"):
            mid = str(rec.get("match_id"))
            pending.append(mid)
            target = _target_from_match(rec)
            if target is None:
                skipped["no_target"] += 1
                continue
            k1 = _team_key(rec.get("team1_id"), rec.get("team1_name"))
            k2 = _team_key(rec.get("team2_id"), rec.get("team2_name"))
            if not k1 or not k2 or k1 == k2:
                skipped["bad_team_key"] += 1
                continue
            s1, c1 = team_seq(k1)
            s2, c2 = team_seq(k2)
            if s1 is None or s2 is None:
                skipped["min_prior_events"] += 1
                continue
            x1.append(s1)
            x2.append(s2)
            y.append(int(target))
            rows.append(
                {
                    "match_id": mid,
                    "date": rec.get("date"),
                    "team1_name": rec.get("team1_name"),
                    "team2_name": rec.get("team2_name"),
                    "target": int(target),
                    "team1_event_count": int(c1),
                    "team2_event_count": int(c2),
                }
            )

        for mid in pending:
            current = by_match.get(mid)
            if current is None:
                continue
            for team_id, team_rows in current.groupby("team_id", sort=False):
                key = _team_key(team_id, team_rows["team_name"].iloc[0] if "team_name" in team_rows else None)
                if not key:
                    continue
                for values in team_rows[emb_cols].to_numpy(dtype=np.float32):
                    histories[key].append(values)
        if cfg.max_examples and len(y) >= cfg.max_examples:
            break

    if not y:
        raise RuntimeError(f"No sequence examples built; skipped={dict(skipped)}")
    x = np.stack([np.stack(x1), np.stack(x2)], axis=1).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(rows).reset_index(drop=True)
    metadata = {
        "config": asdict(cfg),
        "raw_matches": int(len(matches)),
        "player_embedding_rows": int(len(player_embeddings)),
        "embedding_dim": int(len(emb_cols)),
        "rows": int(len(meta)),
        "skipped": {k: int(v) for k, v in skipped.items()},
        "date_min": pd.to_datetime(meta["date"], utc=True, errors="coerce").min().isoformat(),
        "date_max": pd.to_datetime(meta["date"], utc=True, errors="coerce").max().isoformat(),
    }
    return x, y_arr, meta, metadata


def _standardize_by_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = x[train_idx].reshape(-1, x.shape[-1])
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32), mean.squeeze(), std.squeeze()


def _folds(n: int, initial: int, test: int, step: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    start = int(initial)
    while start < n:
        end = min(start + int(test), n)
        if end <= start:
            break
        out.append((np.arange(0, start), np.arange(start, end)))
        start += int(step)
    return out


def train_oof_transformer(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch = require_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    class TeamTransformer(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.proj = nn.Linear(in_dim, args.d_model)
            self.pos = nn.Parameter(torch.zeros(1, args.seq_len, args.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dim_feedforward=args.d_model * 3,
                dropout=args.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=args.layers)
            self.head = nn.Sequential(
                nn.LayerNorm(args.d_model * 3 + 2),
                nn.Linear(args.d_model * 3 + 2, args.d_model),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.d_model, 1),
            )

        def encode_team(self, seq: Any) -> Any:
            h = self.proj(seq) + self.pos[:, : seq.shape[1], :]
            h = self.encoder(h)
            return h.mean(dim=1)

        def forward(self, xb: Any) -> Any:
            a = self.encode_team(xb[:, 0])
            b = self.encode_team(xb[:, 1])
            counts = (xb.abs().sum(dim=-1) > 0).float().sum(dim=-1) / max(float(args.seq_len), 1.0)
            feats = torch.cat([a, b, a - b, counts], dim=1)
            return self.head(feats).squeeze(1)

    folds = _folds(len(y), args.initial_train_size, args.test_size, args.step_size)
    if not folds:
        raise RuntimeError(f"No folds for n={len(y)} initial_train_size={args.initial_train_size}")
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_payload: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        x_std, _, _ = _standardize_by_train(x, train_idx)
        model = TeamTransformer(x.shape[-1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        train_ds = TensorDataset(torch.from_numpy(x_std[train_idx]), torch.from_numpy(y[train_idx].astype(np.float32)))
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        epoch_losses: list[float] = []
        for _ in range(args.epochs):
            model.train()
            losses: list[float] = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            epoch_losses.append(float(np.mean(losses)) if losses else math.nan)
        model.eval()
        probs: list[np.ndarray] = []
        test_ds = TensorDataset(torch.from_numpy(x_std[test_idx]))
        for (xb,) in DataLoader(test_ds, batch_size=args.batch_size, shuffle=False):
            with torch.no_grad():
                p = torch.sigmoid(model(xb.to(device))).detach().cpu().numpy()
            probs.append(p.astype(np.float32))
        pred = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
        oof[test_idx] = pred
        fold_payload.append(
            {
                "fold": fold_idx,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "date_min": pd.to_datetime(meta.loc[test_idx, "date"], utc=True).min().isoformat(),
                "date_max": pd.to_datetime(meta.loc[test_idx, "date"], utc=True).max().isoformat(),
                "epoch_losses": epoch_losses,
                "metrics": _metrics(y[test_idx], pred),
            }
        )
    out = meta.copy()
    out["oof_prob_raw"] = oof
    # Reuse existing market-comparison helper from EXP-054. This first EXP-055
    # probe intentionally reports the raw neural probability; calibration can
    # be added if the sequence model shows enough raw signal.
    out["oof_prob_calibrated"] = oof
    mask = ~np.isnan(oof)
    stats = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "fold_count": int(len(folds)),
        "folds": fold_payload,
        "oof_metrics": _metrics(y[mask], oof[mask]),
    }
    return out[mask].reset_index(drop=True), stats


def main() -> None:
    args = parse_args()
    init_db()
    embeddings = _load_or_encode_embeddings(args)
    matches = load_golgg_match_results(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_matches)
    x, y, meta, seq_meta = build_sequence_examples(
        matches,
        embeddings,
        SequenceConfig(seq_len=args.seq_len, min_prior_events=args.min_prior_events, max_examples=args.max_examples),
    )
    oof, train_meta = train_oof_transformer(x, y, meta, args)
    payload: dict[str, Any] = {
        "experiment_id": "EXP-055",
        "description": "Small transformer over recent leakage-safe PlayerGameEncoder histories.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sequence_metadata": seq_meta,
        "training": train_meta,
    }
    try:
        from betting_app.scripts.evaluate_embedding_live_db import _market_compare

        payload["market_comparison"] = _market_compare(oof, "transformer_embedding")
    except Exception as exc:  # noqa: BLE001 - comparison is useful but optional.
        payload["market_comparison_error"] = repr(exc)

    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
