"""Development-only chronological experiment for the shared A1 macro MLP.

Run with an explicit new --output-dir. The fixed protocol fits on 2024,
selects an epoch on released 2025 labels, and reports 2026 TEST predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.a1.engine import A1PredictorEngine, AntiSymmetricMacroMLP

DEFAULT_SEED = 20260926
DEFAULT_EPOCHS = 24
TRAIN_YEAR, VALIDATION_YEAR, TEST_YEAR = 2024, 2025, 2026
TRAIN_ORIGIN = pd.Timestamp("2025-01-01")
TEST_ORIGIN = pd.Timestamp("2026-01-01")
BANK_RELATIVE = Path("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank")
PAIRED_RELATIVE = Path("rating-foundation-20260915/data/08_reporting/paired.parquet")
META_COLUMNS = [
    "golgg_match_id", "team1_id", "team2_id", "date", "result_day",
    "best_of", "y", "feature_source_max_day",
]
PAIRED_COLUMNS = META_COLUMNS + ["p_full"]
IDENTITY_COLUMNS = [
    "team1_id", "team2_id", "date", "result_day", "best_of", "y",
    "feature_source_max_day",
]


def resolve_research_root(
    explicit: str | Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve research data as CLI > ENSEMBLE_RESEARCH_ROOT > local pointer."""
    environ = os.environ if environ is None else environ
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_root = environ.get("ENSEMBLE_RESEARCH_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    root_file = project_root / "data/research_root.txt"
    pointer = root_file.read_text().strip() if root_file.is_file() else ""
    if pointer:
        return Path(pointer).expanduser().resolve()
    raise FileNotFoundError(
        "Research root required via --research-root, ENSEMBLE_RESEARCH_ROOT, "
        "or data/research_root.txt"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(research_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """Read only needed release metadata, W20, and canonical A0 predictions."""
    bank_dir = research_root / BANK_RELATIVE
    metadata_path = bank_dir / "metadata.parquet"
    arrays_path = bank_dir / "arrays.npz"
    paired_path = research_root / PAIRED_RELATIVE
    for path in (metadata_path, arrays_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = pd.read_parquet(metadata_path, columns=META_COLUMNS)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        w20 = arrays["w20"]
    paired = pd.read_parquet(paired_path, columns=PAIRED_COLUMNS)
    hashes = {
        "feature_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "feature_arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path)},
        "canonical_baseline_predictions": {"path": str(paired_path), "sha256": sha256_file(paired_path)},
    }
    return metadata, paired, w20, hashes


def _normalize_source(frame: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    result = frame.loc[:, columns].copy()
    if result["golgg_match_id"].isna().any():
        raise ValueError(f"{source} has a missing match ID")
    result["golgg_match_id"] = result["golgg_match_id"].astype(str)
    if result["golgg_match_id"].duplicated().any():
        raise ValueError(f"{source} has duplicate match IDs")
    for column in ("team1_id", "team2_id"):
        if result[column].isna().any():
            raise ValueError(f"{source} has a missing {column}")
        result[column] = result[column].astype(str)
    for column in ("date", "result_day", "feature_source_max_day"):
        try:
            values = pd.to_datetime(result[column], errors="raise", utc=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}.{column} contains an invalid date") from exc
        if column != "feature_source_max_day" and values.isna().any():
            raise ValueError(f"{source}.{column} contains a missing date")
        result[column] = values.dt.normalize().dt.tz_localize(None)
    try:
        target = pd.to_numeric(result["y"], errors="raise")
        result["best_of"] = pd.to_numeric(result["best_of"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has a non-numeric target or best_of") from exc
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"{source}.y must contain only binary targets")
    if result["best_of"].isna().any():
        raise ValueError(f"{source}.best_of contains a missing value")
    result["y"] = target.astype(np.float32)
    return result


def prepare_cohort(metadata: pd.DataFrame, paired: pd.DataFrame, w20: np.ndarray) -> dict:
    """Align sources, reject identity/release inconsistencies, and partition rows."""
    meta = _normalize_source(metadata, META_COLUMNS, "feature metadata")
    baseline = _normalize_source(paired, PAIRED_COLUMNS, "canonical baseline")
    if baseline.empty:
        raise ValueError("canonical baseline cohort is empty")
    probabilities = pd.to_numeric(baseline["p_full"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities <= 0) | (probabilities >= 1)).any():
        raise ValueError("canonical p_full values must be finite probabilities strictly inside (0, 1)")
    baseline["p_full"] = probabilities

    w20 = np.asarray(w20)
    if w20.ndim != 3 or w20.shape[1:] != (2, 10) or len(w20) != len(meta):
        raise ValueError("W20 must have shape (metadata_rows, 2, 10) in metadata row order")
    meta["_bank_index"] = np.arange(len(meta), dtype=np.int64)
    metadata_join = meta.rename(columns={key: f"{key}_metadata" for key in IDENTITY_COLUMNS})
    joined = baseline.merge(
        metadata_join, on="golgg_match_id", how="left", validate="one_to_one",
        indicator="_metadata_match", sort=False,
    )
    has_metadata = joined["_metadata_match"].eq("both")
    missing_metadata = int((~has_metadata).sum())
    matched = joined.loc[has_metadata].copy()
    if matched.empty:
        raise ValueError("canonical baseline has no rows aligned to feature metadata")

    for column in IDENTITY_COLUMNS:
        left, right = matched[column], matched[f"{column}_metadata"]
        if column in {"y", "best_of"}:
            same = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0, atol=0)
        else:
            same = (left.eq(right) | (left.isna() & right.isna())).to_numpy(dtype=bool)
        if not np.all(same):
            bad_id = matched.loc[~same, "golgg_match_id"].iloc[0]
            raise ValueError(f"feature metadata/canonical baseline {column} mismatch for {bad_id}")
    if (matched["result_day"] < matched["date"]).any():
        raise ValueError("canonical target release precedes match date")
    if (matched["feature_source_max_day"] >= matched["date"]).any():
        raise ValueError("feature history must be released strictly before the target date")
    if (matched["feature_source_max_day_metadata"] >= matched["date_metadata"]).any():
        raise ValueError("metadata feature history must be released strictly before its target date")

    aligned_w20 = w20[matched["_bank_index"].to_numpy(dtype=np.int64)]
    finite_features = np.isfinite(aligned_w20).all(axis=(1, 2))
    unknown_cutoff = matched["feature_source_max_day"].isna().to_numpy()
    empty_history = (aligned_w20 == 0).all(axis=(1, 2))
    unverifiable = unknown_cutoff & ~empty_history
    invalid_feature_count = int((~finite_features).sum())
    included = finite_features & ~unverifiable
    eligible = matched.loc[included].copy()
    features = A1PredictorEngine.extract_macro_diff(
        aligned_w20[included, 0, :], aligned_w20[included, 1, :]
    )
    years = eligible["date"].dt.year.to_numpy()
    eligible["partition"] = np.select(
        [years == TRAIN_YEAR, years == VALIDATION_YEAR, years == TEST_YEAR],
        ["train_2024", "validation_2025", "test_2026"],
        default="outside_fixed_partitions",
    )
    train_late = (eligible["partition"] == "train_2024") & (eligible["result_day"] >= TRAIN_ORIGIN)
    validation_late = (
        (eligible["partition"] == "validation_2025") & (eligible["result_day"] >= TEST_ORIGIN)
    )
    exclusions = {
        "missing_feature_metadata": missing_metadata,
        "nonfinite_macro_features": invalid_feature_count,
        "unverifiable_feature_cutoff": int(unverifiable.sum()),
        "outside_2024_2026": int((eligible["partition"] == "outside_fixed_partitions").sum()),
        "train_label_not_released_before_2025_origin": int(train_late.sum()),
        "validation_label_not_released_before_2026_origin": int(validation_late.sum()),
    }
    eligible = eligible.loc[~(train_late | validation_late)].copy()
    features = features[~(train_late | validation_late)]
    eligible = eligible.reset_index(drop=True)

    canonical_year_counts = {
        str(year): int((baseline["date"].dt.year == year).sum())
        for year in (TRAIN_YEAR, VALIDATION_YEAR, TEST_YEAR)
    }
    partitions = eligible["partition"].to_numpy()
    counts = {
        "canonical_baseline_rows": int(len(baseline)),
        "feature_metadata_rows": int(len(meta)),
        "matched_canonical_rows": int(len(matched)),
        "canonical_baseline_rows_by_year": canonical_year_counts,
        "included_train_2024": int((partitions == "train_2024").sum()),
        "included_validation_2025": int((partitions == "validation_2025").sum()),
        "included_test_2026": int((partitions == "test_2026").sum()),
    }
    return {
        "frame": eligible,
        "features": features,
        "counts": counts,
        "exclusions": exclusions,
        "canonical_year_counts": canonical_year_counts,
    }


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if target.size == 0 or target.shape != probability.shape:
        raise ValueError("log-loss requires non-empty aligned target and probability arrays")
    if not np.isfinite(target).all() or not np.isfinite(probability).all():
        raise ValueError("log-loss inputs must be finite")
    if not np.isin(target, [0, 1]).all() or ((probability <= 0) | (probability >= 1)).any():
        raise ValueError("log-loss requires binary targets and probabilities inside (0, 1)")
    return float(-np.mean(target * np.log(probability) + (1.0 - target) * np.log1p(-probability)))


def _baseline_logits(probability: np.ndarray) -> torch.Tensor:
    probability = np.asarray(probability, dtype=np.float64)
    return torch.from_numpy((np.log(probability) - np.log1p(-probability)).astype(np.float32))


def fit_best_macro_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_baseline: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_baseline: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
) -> tuple[AntiSymmetricMacroMLP, dict]:
    """Fit only on TRAIN and use only validation labels to select an epoch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if len(train_features) == 0 or len(validation_features) == 0:
        raise ValueError("TRAIN and validation partitions must both be non-empty")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = AntiSymmetricMacroMLP(in_dim=7, hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    x_train = torch.as_tensor(np.asarray(train_features, dtype=np.float32))
    y_train = torch.tensor(train_targets, dtype=torch.float32)
    z_train = _baseline_logits(train_baseline)
    x_validation = torch.as_tensor(np.asarray(validation_features, dtype=np.float32))
    y_validation = np.asarray(validation_targets, dtype=np.float32)
    z_validation = _baseline_logits(validation_baseline)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(z_train + model(x_train).squeeze(-1), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(z_validation + model(x_validation).squeeze(-1)).cpu().numpy()
        validation_loss = _log_loss(y_validation, probability)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "validation_log_loss": best_loss,
        "epochs_run": epochs,
        "seed": seed,
    }


def predict_probabilities(
    model: AntiSymmetricMacroMLP,
    features: np.ndarray,
    baseline_probability: np.ndarray,
) -> np.ndarray:
    """Apply a trained checkpoint without consulting labels or refitting parameters."""
    model.eval()
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    with torch.no_grad():
        probability = torch.sigmoid(_baseline_logits(baseline_probability) + model(x).squeeze(-1)).cpu().numpy()
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def load_trained_model(checkpoint_path: Path) -> AntiSymmetricMacroMLP:
    """Load a saved state_dict for inference without repeating training."""
    model = AntiSymmetricMacroMLP(in_dim=7, hidden_dim=16)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _day(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _partition(cohort: dict, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    mask = cohort["frame"]["partition"].to_numpy() == name
    return cohort["frame"].loc[mask].reset_index(drop=True), cohort["features"][mask]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_experiment(
    metadata: pd.DataFrame,
    paired: pd.DataFrame,
    w20: np.ndarray,
    output_dir: Path,
    input_hashes: dict,
    *,
    research_root: Path | None = None,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
) -> dict:
    """Train/select, score the untouched 2026 cohort, and publish a new run."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run destination: {output_dir}")
    protected_dirs = {
        (PROJECT_ROOT / "data/06_models/a0_team_macro").resolve(),
        (PROJECT_ROOT / "data/07_model_output/a0_team_macro").resolve(),
    }
    if output_dir in protected_dirs:
        raise ValueError("macro experiment output must use a new per-run destination")

    cohort = prepare_cohort(metadata, paired, w20)
    train, train_features = _partition(cohort, "train_2024")
    validation, validation_features = _partition(cohort, "validation_2025")
    test, test_features = _partition(cohort, "test_2026")
    if train.empty or validation.empty or test.empty:
        raise ValueError("fixed 2024 TRAIN, 2025 validation, and 2026 TEST must all be non-empty")
    model, selection = fit_best_macro_model(
        train_features, train["y"].to_numpy(dtype=np.float32), train["p_full"].to_numpy(dtype=float),
        validation_features, validation["y"].to_numpy(dtype=np.float32), validation["p_full"].to_numpy(dtype=float),
        seed=seed, epochs=epochs,
    )
    # TEST targets are first read below, after the selected state is fixed.
    test_probability = predict_probabilities(model, test_features, test["p_full"].to_numpy(dtype=float))
    baseline_test_loss = _log_loss(test["y"].to_numpy(dtype=float), test["p_full"].to_numpy(dtype=float))
    macro_test_loss = _log_loss(test["y"].to_numpy(dtype=float), test_probability)
    train_end, train_date_end = _day(train["result_day"].max()), _day(train["date"].max())
    selection_end = _day(validation["result_day"].max())
    selection_date_end = _day(validation["date"].max())

    predictions = test[[
        "golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of",
        "y", "feature_source_max_day", "p_full",
    ]].copy()
    predictions["cohort"] = "TEST_2026"
    predictions["train_end"] = train_end
    predictions["model_selection_end"] = selection_end
    predictions["calibration_end"] = None
    predictions["p_a0_macro"] = test_probability
    test_baseline_count = cohort["canonical_year_counts"][str(TEST_YEAR)]
    test_prediction_count = len(predictions)
    coverage_fraction = test_prediction_count / test_baseline_count if test_baseline_count else 0.0
    metrics = {
        "schema_version": 1,
        "status": "DEVELOPMENT_ONLY",
        "fit_partition": "train_2024",
        "model_selection_partition": "validation_2025",
        "test_partition": "test_2026",
        "cutoffs": {
            "train_end": train_end,
            "model_selection_end": selection_end,
            "calibration_end": None,
        },
        "inputs": input_hashes,
        "test_2026_partial_canonical_coverage": {
            "rows_scored": test_prediction_count,
            "canonical_baseline_rows": test_baseline_count,
            "coverage_fraction": coverage_fraction,
            "baseline_log_loss": baseline_test_loss,
            "macro_log_loss": macro_test_loss,
            "macro_minus_baseline_log_loss": macro_test_loss - baseline_test_loss,
        },
        "validation_2025_model_selection": {
            "rows": int(len(validation)),
            "best_epoch": int(selection["best_epoch"]),
            "selected_log_loss": float(selection["validation_log_loss"]),
        },
    }
    checkpoint = {
        "schema_version": 1,
        "status": "DEVELOPMENT_ONLY",
        "model_state_dict": model.state_dict(),
        "fit_partition": "train_2024",
        "model_selection_partition": "validation_2025",
        "test_partition": "test_2026",
        "train_end": train_end,
        "model_selection_end": selection_end,
        "calibration_end": None,
        "scaler": None,
        "best_epoch": int(selection["best_epoch"]),
        "seed": int(seed),
        "inputs": input_hashes,
    }
    manifest = {
        "schema_version": 1,
        "experiment": "team_macro_mlp_fixed_chronological_holdout",
        "status": "DEVELOPMENT_ONLY",
        "claim_scope": "partial canonical feature intersection; not full 11,550-row canonical coverage",
        "research_root": str(research_root.resolve()) if research_root is not None else None,
        "seed": int(seed),
        "epochs_run": int(selection["epochs_run"]),
        "best_epoch": int(selection["best_epoch"]),
        "architecture": {
            "class": "src.models.a1.engine.AntiSymmetricMacroMLP",
            "input_features": 7,
            "hidden_features": 16,
            "feature_extractor": "src.models.a1.engine.A1PredictorEngine.extract_macro_diff",
            "scaler": None,
        },
        "partitions": {
            "train": {"match_year": TRAIN_YEAR, "match_date_before": "2025-01-01"},
            "validation_model_selection": {"match_year": VALIDATION_YEAR, "match_date_before": "2026-01-01"},
            "test": {"match_year": TEST_YEAR},
        },
        "label_release_policy": {
            "train_labels_released_before_validation_origin": "2025-01-01",
            "validation_labels_released_before_test_origin": "2026-01-01",
            "comparison": "strictly_before",
            "test_labels_used_for_fit_or_selection": False,
        },
        "cutoffs": {
            "train_end": train_end,
            "train_date_end": train_date_end,
            "model_selection_end": selection_end,
            "model_selection_date_end": selection_date_end,
            "calibration_end": None,
            "calibration_performed": False,
        },
        "cohort_counts": {**cohort["counts"], "published_oos_predictions": test_prediction_count},
        "canonical_coverage": {
            "scope": "2026 rows in the supplied canonical baseline; feature-bank intersection only",
            "test_year": TEST_YEAR,
            "canonical_baseline_rows": test_baseline_count,
            "published_test_rows": test_prediction_count,
            "fraction": coverage_fraction,
            "full_11550_claim": False,
        },
        "exclusions": cohort["exclusions"],
        "model_selection": {
            "partition": "validation_2025",
            "metric": "binary_log_loss",
            "best_epoch": int(selection["best_epoch"]),
            "validation_log_loss": float(selection["validation_log_loss"]),
        },
        "calibration": {"performed": False, "method": None, "end": None},
        "inputs": input_hashes,
        "outputs": {
            "checkpoint": "macro_mlp_state.pt",
            "predictions": "predictions.parquet",
            "metrics": "metrics.json",
        },
        "prediction_schema": list(predictions.columns),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    torch.save(checkpoint, output_dir / "macro_mlp_state.pt")
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, help="Research root; overrides environment and local pointer.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New per-run DEVELOPMENT_ONLY directory; must not exist.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    research_root = resolve_research_root(args.research_root)
    if args.output_dir.expanduser().exists():
        raise FileExistsError(f"refusing to overwrite existing run destination: {args.output_dir}")
    metadata, paired, w20, input_hashes = load_inputs(research_root)
    manifest = run_experiment(
        metadata, paired, w20, args.output_dir, input_hashes,
        research_root=research_root, seed=args.seed, epochs=args.epochs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
