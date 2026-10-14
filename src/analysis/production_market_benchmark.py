"""Pure, identity-aligned scoring for the frozen production market snapshot."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import binary_log_loss_vector
from src.analysis.thesis_metrics import _metrics as _shared_metrics
from betting_app.core.ev import best_ev_side
from src.analysis.bootstrap import block_ratio_interval


HORIZONS = ("OPEN", "24h", "48h", "CLOSE")
DRAWS = 5000
SEED = 20260926
PROBABILITY_EPSILON = 1e-15


def _strict_bool(value: Any) -> bool | None:
    """Parse an explicit boolean representation without truthiness coercion."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def _aware_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")


def _aware_series(values: pd.Series) -> pd.Series:
    return pd.Series([_aware_timestamp(value) for value in values], index=values.index, dtype="datetime64[ns, UTC]")


def _key_series(values: pd.Series) -> pd.Series:
    def normalize(value: Any) -> Any:
        if value is None or pd.isna(value):
            return pd.NA
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    result = values.map(normalize).astype("string")
    return result.mask(result.eq(""))


def _assert_unique_nonnull(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise ValueError(f"{label} is missing required key {column!r}")
    keys = _key_series(frame[column])
    if keys.isna().any() or keys.duplicated().any():
        raise ValueError(f"{label} contains missing or duplicate {column} values")


def _valid_probability(value: Any) -> bool:
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(probability) and 0.0 <= probability <= 1.0)


def select_market_quotes(quotes: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Select one valid quote per bookmaker at the common horizon cutoff.

    The input is normally the validated quote ledger. Validation is repeated here
    for directly supplied frames so malformed dates, live flags and markets can
    never leak into a selected horizon.
    """
    horizon = str(horizon).strip()
    horizon_key = horizon.upper()
    normalized_horizons = {"OPEN": "OPEN", "24H": "24h", "48H": "48h", "CLOSE": "CLOSE"}
    if horizon_key not in normalized_horizons:
        raise ValueError(f"Unknown market horizon: {horizon}")
    horizon = normalized_horizons[horizon_key]
    required = {"canonical_match_id", "bookmaker", "scraped_at", "start_dt", "p_market_quote"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Quotes are missing required columns: {sorted(missing)}")
    q = quotes.copy()
    if q.empty:
        if "cutoff" not in q:
            q["cutoff"] = pd.Series(dtype="datetime64[ns, UTC]")
        return q

    q["scraped_at"] = _aware_series(q["scraped_at"])
    q["start_dt"] = _aware_series(q["start_dt"])
    prices = pd.to_numeric(q["p_market_quote"], errors="coerce")
    valid = q["canonical_match_id"].notna() & q["bookmaker"].notna()
    valid &= q["scraped_at"].notna() & q["start_dt"].notna()
    valid &= q["scraped_at"].lt(q["start_dt"])
    valid &= np.isfinite(prices) & prices.ge(0.0) & prices.le(1.0)
    if {"odds_a", "odds_b"}.issubset(q.columns):
        odds_a = pd.to_numeric(q["odds_a"], errors="coerce")
        odds_b = pd.to_numeric(q["odds_b"], errors="coerce")
        valid &= np.isfinite(odds_a) & np.isfinite(odds_b) & odds_a.gt(1.0) & odds_b.gt(1.0)
    if "is_live" in q:
        valid &= q["is_live"].map(_strict_bool).eq(False)
    if "market_type" in q:
        market = q["market_type"].astype("string").str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
        valid &= market.isin({"match_winner", "series_winner"})
    q = q.loc[valid].copy()
    q["p_market_quote"] = prices.loc[q.index].astype(float)
    if q.empty:
        q["cutoff"] = pd.Series(dtype="datetime64[ns, UTC]")
        return q

    if horizon == "OPEN":
        q["cutoff"] = q.groupby("canonical_match_id", sort=False)["scraped_at"].transform("min")
        q = q.loc[q["scraped_at"].eq(q["cutoff"])].copy()
    elif horizon in {"24h", "48h"}:
        hours = int(horizon[:-1])
        q["cutoff"] = q["start_dt"] - pd.Timedelta(hours=hours)
        lower_bound = q["cutoff"] - pd.Timedelta(hours=24)
        q = q.loc[q["scraped_at"].le(q["cutoff"]) & q["scraped_at"].ge(lower_bound)].copy()
    else:
        q["cutoff"] = q["start_dt"]
        lower_bound = q["start_dt"] - pd.Timedelta(hours=2)
        q = q.loc[q["scraped_at"].ge(lower_bound)].copy()

    if q.empty:
        return q
    q["_stable_quote_id"] = q["id"].astype("string") if "id" in q else q.index.astype(str)
    q = q.sort_values(["scraped_at", "_stable_quote_id"], kind="stable")
    q = q.drop_duplicates(["canonical_match_id", "bookmaker"], keep="last")
    return q.drop(columns="_stable_quote_id")


def market_consensus(selected_quotes: pd.DataFrame) -> pd.DataFrame:
    """Average selected probabilities with exactly one vote per bookmaker."""
    columns = ["canonical_match_id", "p_market", "books", "quote_earliest", "quote_latest", "cutoff"]
    if selected_quotes.empty:
        return pd.DataFrame(columns=columns)
    required = {"canonical_match_id", "bookmaker", "p_market_quote", "cutoff"}
    missing = required.difference(selected_quotes.columns)
    if missing:
        raise ValueError(f"Selected quotes are missing required columns: {sorted(missing)}")
    q = selected_quotes.copy()
    q["p_market_quote"] = pd.to_numeric(q["p_market_quote"], errors="coerce")
    q = q.loc[np.isfinite(q["p_market_quote"]) & q["p_market_quote"].between(0.0, 1.0)]
    q = q.loc[q["canonical_match_id"].notna() & q["bookmaker"].notna()].copy()
    if q.empty:
        return pd.DataFrame(columns=columns)
    order = ["canonical_match_id", "bookmaker"]
    if "scraped_at" in q:
        q["scraped_at"] = _aware_series(q["scraped_at"])
        order.append("scraped_at")
    if "id" in q:
        q["_stable_quote_id"] = q["id"].astype("string")
        order.append("_stable_quote_id")
    q = q.sort_values(order, kind="stable").drop_duplicates(["canonical_match_id", "bookmaker"], keep="last")
    if "scraped_at" not in q:
        q["scraped_at"] = _aware_series(q["cutoff"])
    grouped = q.groupby("canonical_match_id", as_index=False, sort=False).agg(
        p_market=("p_market_quote", "mean"),
        books=("bookmaker", "nunique"),
        quote_earliest=("scraped_at", "min"),
        quote_latest=("scraped_at", "max"),
        cutoff=("cutoff", "first"),
    )
    return grouped[columns]


def paired_cluster_interval(
    delta: Any,
    blocks: Any,
    draws: int = DRAWS,
    seed: int = SEED,
    confidence_level: float | None = None,
) -> dict[str, Any]:
    """Bootstrap match-weighted and equal-block means using paired blocks."""
    differences = np.asarray(delta, dtype=float).reshape(-1)
    labels = np.asarray(blocks, dtype=object).reshape(-1)
    if len(differences) != len(labels):
        raise ValueError("paired differences and block labels must align")
    if draws < 1:
        raise ValueError("draws must be positive")
    level = 0.95 if confidence_level is None else float(confidence_level)
    if not 0 < level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    keep = np.isfinite(differences) & pd.notna(labels)
    differences = differences[keep]
    labels = labels[keep].astype(str)
    if not len(differences):
        return dict(n=0, blocks=0, delta=None, ci_lower=None, ci_upper=None,
                    equal_block_delta=None, equal_block_ci_lower=None,
                    equal_block_ci_upper=None, draws=int(draws), seed=int(seed))
    _, block_index = np.unique(labels, return_inverse=True)
    block_counts = np.bincount(block_index).astype(float)
    block_sums = np.bincount(block_index, weights=differences)
    block_means = block_sums / block_counts
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(block_counts), size=(draws, len(block_counts)))
    weighted_samples = block_sums[sampled].sum(axis=1) / block_counts[sampled].sum(axis=1)
    equal_samples = block_means[sampled].mean(axis=1)
    if len(block_counts) > 1:
        tail = (1 - level) / 2
        lower, upper = np.quantile(weighted_samples, [tail, 1 - tail])
        equal_lower, equal_upper = np.quantile(equal_samples, [tail, 1 - tail])
    else:
        lower = upper = equal_lower = equal_upper = np.nan
    return dict(
        n=int(len(differences)), blocks=int(len(block_counts)),
        delta=float(differences.mean()), ci_lower=float(lower), ci_upper=float(upper),
        equal_block_delta=float(block_means.mean()),
        equal_block_ci_lower=float(equal_lower), equal_block_ci_upper=float(equal_upper),
        draws=int(draws), seed=int(seed),
        **({"confidence_level": level} if confidence_level is not None else {}),
    )


def validate_fixture_cohort(
    matches: pd.DataFrame,
    identities: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Return every fixture row with its first identity/coverage/outcome failure."""
    _assert_unique_nonnull(matches, "canonical_match_id", "matches")
    _assert_unique_nonnull(identities, "canonical_match_id", "match identities")
    _assert_unique_nonnull(forecasts, "golgg_match_id", "forecasts")
    required_forecast = {"team1_id", "team2_id", "date", "best_of", "y", "p_full"}
    missing = required_forecast.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts are missing required columns: {sorted(missing)}")
    if "canonical_match_id" not in matches or "canonical_match_id" not in identities:
        raise ValueError("Fixture and identity tables require canonical_match_id")
    if "golgg_match_id" not in identities:
        raise ValueError("Match identities require golgg_match_id")

    fixtures = matches.copy().reset_index(drop=True)
    fixtures["_canonical_join"] = _key_series(fixtures["canonical_match_id"])
    fixtures["canonical_match_id"] = fixtures["_canonical_join"]
    identity_rows = identities.copy()
    identity_rows["_canonical_join"] = _key_series(identity_rows["canonical_match_id"])
    identity_rows = identity_rows.drop(columns=["canonical_match_id"])
    audit = fixtures.merge(identity_rows, on="_canonical_join", how="left", validate="one_to_one", suffixes=("", "_identity"))
    audit["golgg_match_id"] = _key_series(audit["golgg_match_id"])

    prediction_rows = forecasts.copy()
    prediction_rows["_forecast_join"] = _key_series(prediction_rows["golgg_match_id"])
    prediction_rows = prediction_rows.rename(columns={
        "best_of": "prediction_best_of",
    })
    prediction_rows = prediction_rows.drop(columns=["golgg_match_id"])
    audit["_forecast_join"] = audit["golgg_match_id"]
    audit = audit.merge(prediction_rows, on="_forecast_join", how="left", validate="many_to_one", suffixes=("", "_forecast"))
    audit["p_a0"] = pd.to_numeric(audit["p_full"], errors="coerce")
    audit["start_dt"] = _aware_series(audit["start_time_normalized"] if "start_time_normalized" in audit else pd.Series(pd.NaT, index=audit.index))
    audit["exclusion_reason"] = ""

    def exclude(mask: pd.Series | np.ndarray, reason: str) -> None:
        selected = pd.Series(mask, index=audit.index).fillna(False).astype(bool)
        first_failure = audit["exclusion_reason"].eq("")
        audit.loc[selected & first_failure, "exclusion_reason"] = reason

    status = audit.get("identity_status", pd.Series(index=audit.index, dtype=object)).astype("string")
    orientation = audit.get("canonical_a_is_team1", pd.Series(index=audit.index, dtype=object)).map(_strict_bool)
    audit["canonical_a_is_team1"] = orientation
    exclude(status.ne("verified") | audit["golgg_match_id"].isna() | orientation.isna(), "identity_unresolved")
    exclude(audit["p_full"].isna(), "outside_frozen_prediction_coverage")
    team_identity_matches = audit.get("prediction_team_ids_match", pd.Series(False, index=audit.index)).map(_strict_bool).eq(True)
    for identity_column, forecast_column in (("team1_id", "team1_id_forecast"), ("team2_id", "team2_id_forecast")):
        if identity_column in audit and forecast_column in audit:
            identity_ids = _key_series(audit[identity_column])
            forecast_ids = _key_series(audit[forecast_column])
            team_identity_matches &= identity_ids.eq(forecast_ids)
    exclude(~team_identity_matches, "prediction_team_identity_conflict")
    exclude(~audit.get("prediction_date_matches_source", pd.Series(False, index=audit.index)).map(_strict_bool).eq(True), "prediction_date_source_conflict")
    exclude(~audit.get("prediction_best_of_matches_source", pd.Series(False, index=audit.index)).map(_strict_bool).eq(True), "prediction_best_of_source_conflict")
    exclude(audit["start_dt"].isna(), "missing_or_naive_start_time")
    fixture_day = audit["start_dt"].dt.strftime("%Y-%m-%d")
    prediction_day = audit.get("date", pd.Series(index=audit.index, dtype=object)).astype("string").str.slice(0, 10)
    exclude(prediction_day.isna() | fixture_day.ne(prediction_day), "calendar_date_mismatch")
    fixture_best_of = pd.to_numeric(audit.get("best_of", pd.Series(np.nan, index=audit.index)), errors="coerce")
    prediction_best_of = pd.to_numeric(audit["prediction_best_of"], errors="coerce")
    exclude(fixture_best_of.isna() | prediction_best_of.isna() | fixture_best_of.ne(prediction_best_of), "best_of_mismatch")
    duplicates = audit["golgg_match_id"].notna() & audit["golgg_match_id"].duplicated(keep=False)
    exclude(duplicates, "duplicate_golgg_fixture")
    winner = audit.get("winner_side", pd.Series(index=audit.index, dtype=object)).astype("string").str.strip().str.lower()
    exclude(winner.isna() | winner.eq(""), "missing_winner")
    exclude(winner.notna() & ~winner.isin({"team_a", "team_b"}), "invalid_winner_side")
    exclude(~audit["p_a0"].map(_valid_probability), "invalid_model_probability")
    labels = pd.to_numeric(audit.get("y", pd.Series(np.nan, index=audit.index)), errors="coerce")
    exclude(~labels.isin([0, 1]), "invalid_prediction_outcome")

    expected_y = winner.eq("team_a").eq(orientation).where(
        winner.isin(["team_a", "team_b"]) & orientation.notna()
    ).astype("Float64")
    conflict = labels.notna() & labels.isin([0, 1]) & labels.ne(expected_y)
    exclude(conflict, "outcome_conflict_after_identity_alignment")
    return audit.drop(columns=["_canonical_join", "_forecast_join"], errors="ignore")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read required JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _verify_raw_receipt(raw_dir: Path) -> None:
    receipt_path = raw_dir / "receipt.json"
    receipt = _read_json(receipt_path)
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Raw receipt.json has no file hash ledger")
    for name, record in files.items():
        digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(digest, str):
            raise ValueError(f"Raw receipt has no sha256 for {name}")
        filename = str(name)
        candidates = [raw_dir / filename] if Path(filename).suffix else [raw_dir / f"{filename}.parquet", raw_dir / filename]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise ValueError(f"Raw receipt file is missing: {name}")
        if _sha256(source) != digest:
            raise ValueError(f"Raw export hash mismatch: {source.name}")
    snapshot_name = receipt.get("snapshot_file")
    snapshot_digest = receipt.get("snapshot_sha256")
    snapshot_path = raw_dir / str(snapshot_name) if snapshot_name else None
    if not isinstance(snapshot_digest, str) or snapshot_path is None or not snapshot_path.is_file():
        raise ValueError("Raw receipt has no verifiable source snapshot")
    if _sha256(snapshot_path) != snapshot_digest:
        raise ValueError("Raw source snapshot hash mismatch")
    protocol_digest = receipt.get("protocol_sha256")
    if not isinstance(protocol_digest, str) or _sha256(raw_dir / "protocol.json") != protocol_digest:
        raise ValueError("Frozen raw protocol hash mismatch")


def _verify_inputs(
    raw_dir: Path,
    identity_dir: Path,
    forecast_path: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if out_dir.exists():
        raise ValueError(f"Refusing to use an existing output directory: {out_dir}")
    protocol_path = raw_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    _verify_raw_receipt(raw_dir)
    primary = protocol.get("primary_model", {})
    expected_forecast_hash = primary.get("sha256")
    if not isinstance(expected_forecast_hash, str) or not forecast_path.is_file():
        raise ValueError("Frozen protocol or primary forecast hash is missing")
    if _sha256(forecast_path) != expected_forecast_hash:
        raise ValueError("Primary annual forecast hash does not match the frozen protocol")
    if primary.get("column") != "p_full":
        raise ValueError("Frozen protocol does not pin annual p_full as the primary forecast")

    receipt = _read_json(identity_dir / "identity_receipt.json")
    outputs = receipt.get("output_sha256", {})
    for basename in ("match_identity.parquet", "quote_alignment.parquet"):
        artifact = identity_dir / basename
        if not artifact.is_file() or outputs.get(basename) != _sha256(artifact):
            raise ValueError(f"Identity artifact hash mismatch: {basename}")

    sources = receipt.get("source_sha256", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Identity receipt has no input hash ledger")
    verified_sources: set[Path] = set()
    for source_name, expected in sources.items():
        if not isinstance(expected, str):
            raise ValueError(f"Identity receipt has an invalid hash for {source_name}")
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = source_path.resolve()
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"Identity input hash mismatch: {source_name}")
        verified_sources.add(source_path.resolve())
    required_sources = [
        raw_dir / "matches.parquet", raw_dir / "mappings.parquet",
        raw_dir / "odds.parquet", raw_dir / "aliases.parquet", forecast_path,
    ]
    missing_sources = [str(path.resolve()) for path in required_sources if path.resolve() not in verified_sources]
    if missing_sources:
        raise ValueError(f"Identity receipt does not attest required inputs: {missing_sources}")
    return protocol, receipt


def _validate_monthly(monthly_path: Path, annual: pd.DataFrame) -> pd.DataFrame:
    complete = _read_json(monthly_path.parent / "COMPLETE.json")
    if complete.get("prediction_sha256") != _sha256(monthly_path):
        raise ValueError("Frozen monthly prediction hash does not match COMPLETE.json")
    monthly = pd.read_parquet(monthly_path)
    required = {"golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y", "p_a0", "fold_start"}
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"Monthly sensitivity is missing required columns: {sorted(missing)}")
    _assert_unique_nonnull(monthly, "golgg_match_id", "monthly forecasts")
    _assert_unique_nonnull(annual, "golgg_match_id", "annual forecasts")
    annual_by_id = annual.set_index(_key_series(annual["golgg_match_id"]))
    monthly_ids = _key_series(monthly["golgg_match_id"])
    if not monthly_ids.isin(annual_by_id.index).all():
        raise ValueError("Monthly forecast IDs are not an exact subset of annual forecast IDs")
    annual_indexed = annual_by_id.loc[monthly_ids.to_numpy()]
    for column in ("team1_id", "team2_id"):
        left = _key_series(monthly[column])
        right = _key_series(annual_indexed[column].reset_index(drop=True))
        if left.isna().any() or right.isna().any() or not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise ValueError(f"Monthly and annual forecasts disagree on {column} for a shared ID")
    monthly_date = _key_series(monthly["date"]).str.slice(0, 10).reset_index(drop=True)
    annual_date = _key_series(annual_indexed["date"]).str.slice(0, 10).reset_index(drop=True)
    if monthly_date.isna().any() or annual_date.isna().any() or not monthly_date.equals(annual_date):
        raise ValueError("Monthly and annual forecasts disagree on date for a shared ID")
    for column in ("best_of", "y"):
        left = pd.to_numeric(monthly[column], errors="coerce").reset_index(drop=True)
        right = pd.to_numeric(annual_indexed[column], errors="coerce").reset_index(drop=True)
        if left.isna().any() or right.isna().any() or not left.equals(right):
            raise ValueError(f"Monthly and annual forecasts disagree on {column} for a shared ID")
    probability = pd.to_numeric(monthly["p_a0"], errors="coerce")
    if not probability.map(_valid_probability).all():
        raise ValueError("Monthly sensitivity contains invalid probabilities")
    monthly = monthly.copy()
    monthly["golgg_match_id"] = _key_series(monthly["golgg_match_id"])
    monthly["p_a0_monthly"] = probability.astype(float)
    return monthly


def _metric_row(frame: pd.DataFrame, probability_column: str, **metadata: Any) -> dict[str, Any]:
    labels = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    probabilities = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[keep].astype(int)
    probabilities = probabilities[keep]
    if not len(labels):
        return dict(**metadata, model=probability_column, n=0, log_loss=None, brier=None,
                    roc_auc=None, accuracy=None, ece=None, mce=None,
                    calibration_slope=None, calibration_intercept=None,
                    ll_ge_2_5=0, ll_ge_3=0, ll_ge_4=0)
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Scoring rows contain invalid labels or probabilities")
    details, _ = _shared_metrics(labels, probabilities)
    losses = binary_log_loss_vector(labels, probabilities, epsilon=PROBABILITY_EPSILON)
    return dict(
        **metadata, model=probability_column, n=int(len(labels)),
        log_loss=float(losses.mean()), brier=float(np.mean((probabilities - labels) ** 2)),
        roc_auc=float(details["auc"]) if np.isfinite(details["auc"]) else None,
        accuracy=float(np.mean((probabilities >= 0.5) == labels)),
        ece=float(details["ece"]), mce=float(details["mce"]),
        calibration_slope=float(details["calibration_slope"]) if np.isfinite(details["calibration_slope"]) else None,
        calibration_intercept=float(details["calibration_intercept"]) if np.isfinite(details["calibration_intercept"]) else None,
        ll_ge_2_5=int((losses >= 2.5).sum()), ll_ge_3=int((losses >= 3.0).sum()),
        ll_ge_4=int((losses >= 4.0).sum()),
    )


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat() if not pd.isna(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _append_scores(
    frame: pd.DataFrame,
    columns: list[str],
    metrics: list[dict[str, Any]],
    **metadata: Any,
) -> None:
    for column in columns:
        if column in frame:
            metrics.append(_metric_row(frame, column, **metadata))


def _blocks_for(frame: pd.DataFrame, unit: str) -> pd.Series:
    if unit == "calendar_month":
        return pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m").fillna("unknown")
    return frame.get("tournament_name", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)


def _contrast_rows(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    horizon: str,
    cohort: str,
) -> list[dict[str, Any]]:
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(float)
    candidate_p = pd.to_numeric(frame[candidate], errors="coerce").to_numpy(float)
    reference_p = pd.to_numeric(frame[reference], errors="coerce").to_numpy(float)
    candidate_ll = binary_log_loss_vector(y, candidate_p, epsilon=PROBABILITY_EPSILON)
    reference_ll = binary_log_loss_vector(y, reference_p, epsilon=PROBABILITY_EPSILON)
    deltas = {
        "log_loss": candidate_ll - reference_ll,
        "brier": (candidate_p - y) ** 2 - (reference_p - y) ** 2,
    }
    rows = []
    for measure, delta in deltas.items():
        for unit in ("calendar_month", "tournament_name_proxy"):
            rows.append(dict(
                cohort=cohort, horizon=horizon, candidate=candidate, reference=reference,
                metric=measure, unit=unit,
                **paired_cluster_interval(delta, _blocks_for(frame, unit)),
            ))
    return rows


def _profile_rows(frame: pd.DataFrame, horizon: str, cohort: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    working = frame.copy()
    working["calendar_month"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m").fillna("unknown")
    working["best_of_profile"] = working.get("best_of", pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
    working["competition_family_profile"] = working.get("competition_family", pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
    working["competition_tier_profile"] = working.get("competition_tier", pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
    model_probability = pd.to_numeric(working["p_a0"], errors="coerce")
    confidence = np.maximum(model_probability, 1 - model_probability)
    working["model_confidence"] = pd.cut(confidence, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001],
                                         labels=["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
                                         include_lowest=True).astype("string").fillna("unknown")
    disagreement = (model_probability - pd.to_numeric(working["p_market"], errors="coerce")).abs()
    working["model_market_disagreement"] = pd.cut(disagreement, [0, .05, .10, .20, .35, 1.0000001],
                                                   labels=["0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.35", "0.35+"],
                                                   include_lowest=True).astype("string").fillna("unknown")
    dimensions = {
        "calendar_month": "calendar_month", "best_of": "best_of_profile",
        "competition_family": "competition_family_profile", "competition_tier": "competition_tier_profile",
        "model_confidence": "model_confidence", "model_market_disagreement": "model_market_disagreement",
    }
    result: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        for value, part in working.groupby(column, dropna=False, sort=True):
            for model in ("p_a0", "p_market"):
                result.append(_metric_row(part, model, cohort=cohort, horizon=horizon,
                                          dimension=dimension, value=str(value)))
    return result


def _make_report(summary: dict[str, Any], metrics: pd.DataFrame, timing: pd.DataFrame) -> str:
    lines = [
        "# Production market benchmark",
        "",
        "Retrospective identity-aligned probability scoring of the frozen local snapshot.",
        "This is not prospective publication certification, a promotion decision, or an executable ROI estimate.",
        "",
        "## Coverage",
        "",
        f"- Finished fixture rows: {summary['finished_fixtures']}",
        f"- Identity/outcome-eligible fixtures: {summary['eligible_fixtures']}",
        f"- Validated quote rows: {summary['validated_quote_rows']}",
        "- Fixture exclusions: `" + json.dumps(summary["fixture_exclusions"], sort_keys=True, ensure_ascii=False) + "`",
        "- Quote exclusions: `" + json.dumps(summary["quote_exclusions"], sort_keys=True, ensure_ascii=False) + "`",
        "",
        "## Annual A0 and market by horizon",
        "",
        "| Horizon | Model | N | Log loss | Brier | AUC | Accuracy | ECE | LL ≥ 2.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = metrics.loc[metrics.get("cohort", pd.Series(index=metrics.index, dtype=object)).eq("all_horizon_quote_cohort")]
    for _, row in selected.iterrows():
        def fmt(value: Any) -> str:
            return "—" if pd.isna(value) else f"{float(value):.4f}"
        lines.append(f"| {row.get('horizon', '')} | {row.get('model', '')} | {int(row.get('n', 0))} | {fmt(row.get('log_loss'))} | {fmt(row.get('brier'))} | {fmt(row.get('roc_auc'))} | {fmt(row.get('accuracy'))} | {fmt(row.get('ece'))} | {int(row.get('ll_ge_2_5', 0))} |")
    lines.extend(["", "## Timing scope", ""])
    if timing.empty:
        lines.append("No horizon had selected quotes.")
    else:
        lines.extend([
            "Feature-day compatibility is a strict prior-calendar-day scenario; it does not certify an actual historical publication.",
            "",
            "| Horizon | N | Feature day at/after cutoff day | Hypothetical origin after cutoff | Compatible N |",
            "|---|---:|---:|---:|---:|",
        ])
        for _, row in timing.iterrows():
            lines.append(f"| {row['horizon']} | {row['n']} | {row['feature_day_at_or_after_cutoff_day']} | {row['model_origin_after_cutoff']} | {row['time_compatible_n']} |")
    lines.extend([
        "", "## Interpretation", "",
        "Scores use the protocol-pinned annual `p_full`; the optional frozen monthly stream, when present, is reported only as a separately named sensitivity.",
        "Intervals are descriptive paired calendar-month and tournament-name cluster resamples; no classical p-value or adoption claim is made.",
        "No training, scraping, production access, financial simulation, ROI, CLV, or Kelly analysis was performed.", "",
    ])
    return "\n".join(lines)


def run_production_market_benchmark(
    raw_dir: str | Path,
    identity_dir: str | Path,
    forecast_path: str | Path,
    out_dir: str | Path,
    *,
    monthly_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score verified local inputs and write the frozen protocol's report artifacts."""
    raw_dir, identity_dir, forecast_path, out_dir = map(Path, (raw_dir, identity_dir, forecast_path, out_dir))
    monthly_path = Path(monthly_path) if monthly_path is not None else None
    protocol, identity_receipt = _verify_inputs(raw_dir, identity_dir, forecast_path, out_dir)

    matches = pd.read_parquet(raw_dir / "matches.parquet")
    odds = pd.read_parquet(raw_dir / "odds.parquet")
    identities = pd.read_parquet(identity_dir / "match_identity.parquet")
    quote_alignment = pd.read_parquet(identity_dir / "quote_alignment.parquet")
    forecasts = pd.read_parquet(forecast_path)
    _assert_unique_nonnull(odds, "id", "odds")
    _assert_unique_nonnull(quote_alignment, "id", "quote alignment")
    if "golgg_match_id" not in forecasts:
        raise ValueError("Annual forecasts require golgg_match_id")
    audit = validate_fixture_cohort(matches, identities, forecasts)
    eligible = audit.loc[audit["exclusion_reason"].eq("")].copy()
    if eligible.empty:
        raise ValueError("No identity- and outcome-eligible fixtures are available for scoring")
    if monthly_path is not None:
        monthly = _validate_monthly(monthly_path, forecasts)
    else:
        monthly = None

    # Retain every source quote in coverage, including quotes on excluded fixtures.
    q = odds.copy()
    q["_quote_join"] = _key_series(q["id"])
    alignment = quote_alignment.copy()
    alignment["_quote_join"] = _key_series(alignment["id"])
    alignment = alignment.drop(columns=["id"])
    q = q.merge(alignment, on="_quote_join", how="left", validate="one_to_one", suffixes=("", "_identity"))
    q["canonical_match_id"] = _key_series(q["canonical_match_id"])
    if "canonical_match_id_identity" in q:
        q["canonical_match_id_identity"] = _key_series(q["canonical_match_id_identity"])
    match_info = audit[["canonical_match_id", "start_dt", "canonical_a_is_team1", "exclusion_reason"]].copy()
    match_info["canonical_match_id"] = _key_series(match_info["canonical_match_id"])
    match_info = match_info.rename(columns={
        "canonical_a_is_team1": "fixture_canonical_a_is_team1",
        "exclusion_reason": "fixture_exclusion_reason",
    })
    q = q.merge(match_info, on="canonical_match_id", how="left", validate="many_to_one")
    q["scraped_at"] = _aware_series(q.get("scraped_at", pd.Series(pd.NaT, index=q.index)))
    q["odds_a"] = pd.to_numeric(q.get("odds_a", pd.Series(np.nan, index=q.index)), errors="coerce")
    q["odds_b"] = pd.to_numeric(q.get("odds_b", pd.Series(np.nan, index=q.index)), errors="coerce")
    q["exclusion_reason"] = ""

    quote_status = q.get("quote_identity_status", pd.Series(index=q.index, dtype=object)).astype("string")
    q.loc[quote_status.ne("verified"), "exclusion_reason"] = "quote_identity_unresolved"
    aligned_match = q.get("canonical_match_id_identity", pd.Series(index=q.index, dtype=object))
    aligned_match = _key_series(aligned_match)
    odds_match = _key_series(q["canonical_match_id"])
    q.loc[q["exclusion_reason"].eq("") & aligned_match.notna() & odds_match.ne(aligned_match), "exclusion_reason"] = "quote_match_identity_conflict"
    raw_book = _key_series(q.get("bookmaker", pd.Series(index=q.index, dtype=object)))
    q.loc[q["exclusion_reason"].eq("") & raw_book.isna(), "exclusion_reason"] = "missing_bookmaker"
    if "bookmaker_identity" in q:
        aligned_book = _key_series(q["bookmaker_identity"])
        q.loc[q["exclusion_reason"].eq("") & raw_book.notna() & aligned_book.notna() & raw_book.ne(aligned_book), "exclusion_reason"] = "quote_bookmaker_identity_conflict"
    market_type = q.get("market_type", pd.Series(index=q.index, dtype=object)).astype("string").str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
    bad_market = ~market_type.isin({"match_winner", "series_winner"})
    q.loc[q["exclusion_reason"].eq("") & bad_market, "exclusion_reason"] = "non_match_winner_market"
    live = q.get("is_live", pd.Series(index=q.index, dtype=object)).map(_strict_bool)
    q.loc[q["exclusion_reason"].eq("") & ~live.eq(False), "exclusion_reason"] = "live_or_unknown_live_flag"
    valid_prices = np.isfinite(q["odds_a"]) & np.isfinite(q["odds_b"]) & q["odds_a"].gt(1.0) & q["odds_b"].gt(1.0)
    q.loc[q["exclusion_reason"].eq("") & ~valid_prices, "exclusion_reason"] = "invalid_price"
    q.loc[q["exclusion_reason"].eq("") & q["start_dt"].isna(), "exclusion_reason"] = "missing_or_naive_start_time"
    q.loc[q["exclusion_reason"].eq("") & q["scraped_at"].isna(), "exclusion_reason"] = "missing_or_naive_quote_time"
    q.loc[q["exclusion_reason"].eq("") & q["scraped_at"].ge(q["start_dt"]), "exclusion_reason"] = "not_prestart"
    raw_a_canonical = q.get("raw_a_is_canonical_a", pd.Series(index=q.index, dtype=object)).map(_strict_bool)
    canonical_a_team1 = q["fixture_canonical_a_is_team1"].map(_strict_bool)
    derived_raw_a_team1 = raw_a_canonical.eq(canonical_a_team1).where(
        raw_a_canonical.notna() & canonical_a_team1.notna()
    )
    if "raw_a_is_team1" in q:
        raw_a_team1 = q["raw_a_is_team1"].map(_strict_bool)
        orientation_conflict = raw_a_team1.notna() & derived_raw_a_team1.notna() & raw_a_team1.ne(derived_raw_a_team1)
        q.loc[q["exclusion_reason"].eq("") & orientation_conflict, "exclusion_reason"] = "quote_orientation_conflict"
    else:
        raw_a_team1 = derived_raw_a_team1
    q["raw_a_is_team1"] = raw_a_team1
    q.loc[q["exclusion_reason"].eq("") & q["raw_a_is_team1"].isna(), "exclusion_reason"] = "quote_orientation_unresolved"
    q.loc[q["exclusion_reason"].eq("") & q["fixture_exclusion_reason"].isna(), "exclusion_reason"] = "fixture_not_in_snapshot"
    q.loc[q["exclusion_reason"].eq("") & q["fixture_exclusion_reason"].ne(""), "exclusion_reason"] = "fixture_not_eligible"
    safe_odds_a = q["odds_a"].where(valid_prices)
    safe_odds_b = q["odds_b"].where(valid_prices)
    implied_a = (1.0 / safe_odds_a) / ((1.0 / safe_odds_a) + (1.0 / safe_odds_b))
    q["p_market_quote"] = np.where(q["raw_a_is_team1"].eq(True), implied_a, 1.0 - implied_a)
    quote_coverage = q.drop(columns=["_quote_join"], errors="ignore")
    validated_quotes = quote_coverage.loc[quote_coverage["exclusion_reason"].eq("")].copy()

    # All artifacts are prepared in memory before the output path is created.
    selected_by_horizon: dict[str, pd.DataFrame] = {}
    paired_by_horizon: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    book_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    monthly_common = monthly.set_index("golgg_match_id") if monthly is not None else None
    _append_scores(eligible, ["p_a0"], metric_rows, cohort="eligible_model_cohort", horizon="ALL")
    if monthly is not None:
        monthly_eligible = eligible.merge(
            monthly[["golgg_match_id", "p_a0_monthly", "fold_start"]],
            on="golgg_match_id", how="inner", validate="one_to_one",
        )
        _append_scores(monthly_eligible, ["p_a0_monthly"], metric_rows,
                       cohort="monthly_sensitivity_model_only", horizon="ALL")
    compatible_by_horizon: dict[str, pd.DataFrame] = {}

    for horizon in HORIZONS:
        selected = select_market_quotes(validated_quotes, horizon)
        selected_by_horizon[horizon] = selected
        consensus = market_consensus(selected)
        consensus["canonical_match_id"] = _key_series(consensus["canonical_match_id"])
        paired = eligible.merge(consensus, on="canonical_match_id", how="inner", validate="one_to_one")
        if monthly_common is not None:
            if paired.empty:
                paired["p_a0_monthly"] = pd.Series(dtype=float, index=paired.index)
                paired["fold_start"] = pd.Series(dtype=object, index=paired.index)
            else:
                ids = _key_series(paired["golgg_match_id"])
                monthly_index = monthly_common.reindex(ids.to_numpy())
                paired["p_a0_monthly"] = pd.to_numeric(monthly_index["p_a0_monthly"].to_numpy(), errors="coerce")
                paired["fold_start"] = monthly_index["fold_start"].to_numpy()
        paired_by_horizon[horizon] = paired
        _append_scores(paired, ["p_a0", "p_market"], metric_rows,
                       cohort="all_horizon_quote_cohort", horizon=horizon)

        if not paired.empty:
            contrast_rows.extend(_contrast_rows(paired, "p_a0", "p_market", horizon, "all_horizon_quote_cohort"))
            profile_rows.extend(_profile_rows(paired, horizon, "all_horizon_quote_cohort"))
            if monthly_common is not None:
                monthly_paired = paired.loc[paired["p_a0_monthly"].notna()]
                if not monthly_paired.empty:
                    _append_scores(monthly_paired, ["p_a0_monthly", "p_market", "p_a0"], metric_rows,
                                   cohort="monthly_sensitivity_identical_rows", horizon=horizon)
                    contrast_rows.extend(_contrast_rows(monthly_paired, "p_a0_monthly", "p_market", horizon, "monthly_sensitivity"))
                    contrast_rows.extend(_contrast_rows(monthly_paired, "p_a0_monthly", "p_a0", horizon, "monthly_sensitivity"))
            for bookmaker, book_quotes in selected.groupby("bookmaker", sort=True):
                one_book = eligible.merge(book_quotes, on="canonical_match_id", how="inner", validate="one_to_one")
                if one_book.empty:
                    continue
                one_book["p_book"] = pd.to_numeric(one_book["p_market_quote"], errors="coerce")
                for model_column in ("p_a0", "p_book"):
                    book_rows.append(_metric_row(one_book, model_column, horizon=horizon,
                                                 bookmaker=str(bookmaker), cohort="bookmaker_identical_rows"))
                contrast_rows.extend(_contrast_rows(one_book, "p_a0", "p_book", horizon, f"bookmaker:{bookmaker}"))

            cutoff = pd.to_datetime(paired["cutoff"], utc=True, errors="coerce")
            cutoff_day = cutoff.dt.floor("D")
            feature_timestamp = pd.to_datetime(
                paired.get("feature_source_max_day", pd.Series(pd.NaT, index=paired.index)),
                utc=True, errors="coerce",
            )
            feature_day = feature_timestamp.dt.floor("D")
            annual_origin = pd.to_datetime(
                paired["date"].astype(str).str.slice(0, 4) + "-01-01", utc=True, errors="coerce",
            )
            annual_compatible = feature_day.notna() & cutoff_day.notna() & feature_day.lt(cutoff_day)
            paired["annual_model_origin_hypothetical"] = annual_origin
            paired["feature_source_max_day_at_or_after_cutoff_day"] = feature_day.notna() & feature_day.ge(cutoff_day)
            paired["feature_source_max_timestamp_after_cutoff"] = feature_timestamp.gt(cutoff)
            paired["annual_model_origin_after_cutoff"] = annual_origin.gt(cutoff)
            annual_compatible &= annual_origin.notna() & cutoff.notna() & annual_origin.le(cutoff)
            paired["time_compatible_strict_prior_day"] = annual_compatible
            compatible = paired.loc[annual_compatible].copy()
            compatible_by_horizon[horizon] = compatible
            if not compatible.empty:
                _append_scores(compatible, ["p_a0", "p_market"], metric_rows,
                               cohort="strict_prior_day_time_compatible", horizon=horizon)
                contrast_rows.extend(_contrast_rows(compatible, "p_a0", "p_market", horizon,
                                                    "strict_prior_day_time_compatible"))
            monthly_compatible_n = 0
            monthly_origin_after = 0
            if monthly_common is not None:
                monthly_origin = pd.to_datetime(paired["fold_start"], utc=True, errors="coerce")
                monthly_compatible_mask = feature_day.notna() & cutoff_day.notna() & feature_day.lt(cutoff_day)
                monthly_compatible_mask &= monthly_origin.notna() & cutoff.notna() & monthly_origin.le(cutoff)
                paired["monthly_model_origin"] = monthly_origin
                monthly_compatible = paired.loc[monthly_compatible_mask & paired["p_a0_monthly"].notna()].copy()
                paired["monthly_model_origin_after_cutoff"] = monthly_origin.gt(cutoff) & paired["p_a0_monthly"].notna()
                paired["monthly_time_compatible_strict_prior_day"] = monthly_compatible_mask & paired["p_a0_monthly"].notna()
                monthly_compatible_n = int(len(monthly_compatible))
                monthly_origin_after = int((monthly_origin.gt(cutoff) & paired["p_a0_monthly"].notna()).sum())
                if not monthly_compatible.empty:
                    _append_scores(monthly_compatible, ["p_a0_monthly", "p_market", "p_a0"], metric_rows,
                                   cohort="monthly_strict_prior_day_time_compatible", horizon=horizon)
                    contrast_rows.extend(_contrast_rows(monthly_compatible, "p_a0_monthly", "p_market", horizon,
                                                        "monthly_strict_prior_day_time_compatible"))
            timing_rows.append(dict(
                horizon=horizon, n=int(len(paired)),
                feature_day_at_or_after_cutoff_day=int((feature_day.notna() & cutoff_day.notna() & feature_day.ge(cutoff_day)).sum()),
                feature_day_after_cutoff_day=int((feature_timestamp > cutoff).sum()),
                model_origin_after_cutoff=int((annual_origin > cutoff).sum()),
                time_compatible_n=int(len(compatible)), monthly_model_origin_after_cutoff=monthly_origin_after,
                monthly_time_compatible_n=monthly_compatible_n,
                median_books=float(paired["books"].median()),
                median_quote_hours_before_start=float(((paired["start_dt"] - paired["quote_latest"]).dt.total_seconds() / 3600).median()),
            ))
        else:
            compatible_by_horizon[horizon] = paired.copy()
            paired["annual_model_origin_hypothetical"] = pd.Series(dtype="datetime64[ns, UTC]", index=paired.index)
            paired["feature_source_max_day_at_or_after_cutoff_day"] = pd.Series(dtype=bool, index=paired.index)
            paired["feature_source_max_timestamp_after_cutoff"] = pd.Series(dtype=bool, index=paired.index)
            paired["annual_model_origin_after_cutoff"] = pd.Series(dtype=bool, index=paired.index)
            paired["time_compatible_strict_prior_day"] = pd.Series(dtype=bool, index=paired.index)
            if monthly_common is not None:
                paired["monthly_model_origin_after_cutoff"] = pd.Series(dtype=bool, index=paired.index)
                paired["monthly_model_origin"] = pd.Series(dtype="datetime64[ns, UTC]", index=paired.index)
                paired["monthly_time_compatible_strict_prior_day"] = pd.Series(dtype=bool, index=paired.index)
            timing_rows.append(dict(horizon=horizon, n=0, feature_day_at_or_after_cutoff_day=0,
                                   feature_day_after_cutoff_day=0, model_origin_after_cutoff=0,
                                   time_compatible_n=0, monthly_model_origin_after_cutoff=0,
                                   monthly_time_compatible_n=0, median_books=None,
                                   median_quote_hours_before_start=None))

    # Score the intersection across all horizons, independently for all and time-compatible rows.
    common_ids = set.intersection(*[
        set(frame["canonical_match_id"].astype(str)) for frame in paired_by_horizon.values()
    ])
    compatible_ids = set.intersection(*[
        set(frame["canonical_match_id"].astype(str)) for frame in compatible_by_horizon.values()
    ])
    common_cohort_rows: list[dict[str, Any]] = []
    for horizon, frame in paired_by_horizon.items():
        common = frame.loc[frame["canonical_match_id"].astype(str).isin(common_ids)]
        _append_scores(common, ["p_a0", "p_market"], metric_rows,
                       cohort="all_horizon_common_cohort", horizon=horizon)
        for model in ("p_a0", "p_market"):
            common_cohort_rows.append(_metric_row(common, model, cohort="all_horizon_common_cohort", horizon=horizon))
        if not common.empty:
            contrast_rows.extend(_contrast_rows(common, "p_a0", "p_market", horizon,
                                                "all_horizon_common_cohort"))
        if monthly_common is not None:
            monthly_common_rows = common.loc[common["p_a0_monthly"].notna()]
            if not monthly_common_rows.empty:
                _append_scores(monthly_common_rows, ["p_a0_monthly", "p_market", "p_a0"], metric_rows,
                               cohort="monthly_sensitivity_all_horizon_common", horizon=horizon)
                for model in ("p_a0_monthly", "p_market", "p_a0"):
                    common_cohort_rows.append(_metric_row(monthly_common_rows, model,
                                                           cohort="monthly_sensitivity_all_horizon_common",
                                                           horizon=horizon))
                contrast_rows.extend(_contrast_rows(monthly_common_rows, "p_a0_monthly", "p_market",
                                                    horizon, "monthly_sensitivity_all_horizon_common"))
                contrast_rows.extend(_contrast_rows(monthly_common_rows, "p_a0_monthly", "p_a0",
                                                    horizon, "monthly_sensitivity_all_horizon_common"))
        time_common = compatible_by_horizon[horizon]
        time_common = time_common.loc[time_common["canonical_match_id"].astype(str).isin(compatible_ids)]
        _append_scores(time_common, ["p_a0", "p_market"], metric_rows,
                       cohort="all_horizon_common_time_compatible", horizon=horizon)
        if not time_common.empty:
            contrast_rows.extend(_contrast_rows(time_common, "p_a0", "p_market", horizon,
                                                "all_horizon_common_time_compatible"))
        for model in ("p_a0", "p_market"):
            common_cohort_rows.append(_metric_row(time_common, model,
                                                   cohort="all_horizon_common_time_compatible",
                                                   horizon=horizon))
    fixture_exclusions = audit["exclusion_reason"].value_counts(dropna=False).to_dict()
    quote_exclusions = quote_coverage["exclusion_reason"].value_counts(dropna=False).to_dict()
    summary = dict(
        status="RETROSPECTIVE_IDENTITY_ALIGNED_MARKET_BENCHMARK_NOT_PROMOTION_EVIDENCE",
        finished_fixtures=int(len(matches)), eligible_fixtures=int(len(eligible)),
        fixture_exclusions=fixture_exclusions, total_quote_rows=int(len(quote_coverage)),
        validated_quote_rows=int(len(validated_quotes)), quote_exclusions=quote_exclusions,
        horizon_rows={horizon: int(len(frame)) for horizon, frame in paired_by_horizon.items()},
        all_horizon_common_fixtures=int(len(common_ids)),
        all_horizon_common_time_compatible_fixtures=int(len(compatible_ids)),
        monthly_sensitivity_present=bool(monthly is not None),
        forecast_policy="protocol-pinned annual p_full is primary; January-1 model origin is hypothetical only",
        time_compatible_policy="feature source day strictly before cutoff calendar day and model origin no later than cutoff; not actual publication certification",
        uncertainty=dict(unit="paired cluster bootstrap", draws=DRAWS, seed=SEED,
                         weighting=["match-weighted", "equal-block"], blocks=["calendar_month", "tournament_name_proxy"],
                         interpretation="descriptive interval; not a classical p-value"),
        financial_analysis="NOT_RUN", adoption="NOT_ASSESSED",
        inputs={
            "raw_receipt_sha256": _sha256(raw_dir / "receipt.json"),
            "identity_receipt_sha256": _sha256(identity_dir / "identity_receipt.json"),
            "forecast_sha256": _sha256(forecast_path),
            "monthly_forecast_sha256": _sha256(monthly_path) if monthly_path is not None else None,
        },
    )
    metrics = pd.DataFrame(metric_rows)
    contrasts = pd.DataFrame(contrast_rows)
    timing = pd.DataFrame(timing_rows)
    profiles = pd.DataFrame(profile_rows)
    bookmaker_metrics = pd.DataFrame(book_rows)
    common_metrics = pd.DataFrame(common_cohort_rows)

    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(raw_dir / "protocol.json", out_dir / "protocol.json")
    audit.drop(columns=["start_dt"], errors="ignore").to_parquet(out_dir / "fixture_coverage.parquet", index=False)
    quote_coverage.to_parquet(out_dir / "quote_coverage.parquet", index=False)
    validated_quotes.to_parquet(out_dir / "validated_quotes.parquet", index=False)
    for horizon in HORIZONS:
        selected_by_horizon[horizon].to_parquet(out_dir / f"{horizon}_selected_quotes.parquet", index=False)
        paired_by_horizon[horizon].to_parquet(out_dir / f"{horizon}_paired.parquet", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)
    timing.to_csv(out_dir / "timing.csv", index=False)
    profiles.to_csv(out_dir / "profiles.csv", index=False)
    bookmaker_metrics.to_csv(out_dir / "bookmaker_metrics.csv", index=False)
    common_metrics.to_csv(out_dir / "common_cohort_metrics.csv", index=False)
    summary = _safe(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(_make_report(summary, metrics, timing), encoding="utf-8")
    return summary


_A0_REQUIRED_COLUMNS = {
    "golgg_match_id", "horizon", "date", "p_a0", "p_market", "y", "best_odds_team1",
    "best_odds_team2", "best_bookmaker_team1", "best_bookmaker_team2",
    "best_quote_id_team1", "best_quote_id_team2", "best_quote_time_team1",
    "best_quote_time_team2", "cutoff", "start_dt",
}


def _validated_a0_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = _A0_REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"A0 audit panel is missing required columns: {sorted(missing)}")
    frame = panel.copy().reset_index(drop=True)
    if frame.empty:
        return frame
    if frame["horizon"].isna().any() or not frame["horizon"].astype(str).isin(HORIZONS).all():
        raise ValueError("A0 audit panel contains an unknown horizon")
    match_ids = _key_series(frame["golgg_match_id"])
    if match_ids.isna().any() or pd.DataFrame({"horizon": frame["horizon"], "match": match_ids}).duplicated().any():
        raise ValueError("A0 audit panel contains missing or duplicate (horizon, golgg_match_id) keys")
    frame["golgg_match_id"] = match_ids

    probability = pd.to_numeric(frame["p_a0"], errors="coerce")
    market_probability = pd.to_numeric(frame["p_market"], errors="coerce")
    outcome = pd.to_numeric(frame["y"], errors="coerce")
    odds_a = pd.to_numeric(frame["best_odds_team1"], errors="coerce")
    odds_b = pd.to_numeric(frame["best_odds_team2"], errors="coerce")
    if not (np.isfinite(probability) & probability.between(0.0, 1.0)).all():
        raise ValueError("A0 audit panel contains invalid probabilities")
    if not (np.isfinite(market_probability) & market_probability.between(0.0, 1.0)).all():
        raise ValueError("A0 audit panel contains invalid market probabilities")
    if not (np.isfinite(outcome) & outcome.isin([0, 1])).all():
        raise ValueError("A0 audit panel contains invalid outcomes")
    if not (np.isfinite(odds_a) & odds_a.gt(1.0)).all() or not (np.isfinite(odds_b) & odds_b.gt(1.0)).all():
        raise ValueError("A0 audit panel contains invalid odds")
    frame["p_a0"] = probability.astype(float)
    frame["p_market"] = market_probability.astype(float)
    frame["y"] = outcome.astype(int)
    frame["best_odds_team1"] = odds_a.astype(float)
    frame["best_odds_team2"] = odds_b.astype(float)

    cutoff = _aware_series(frame["cutoff"])
    start = _aware_series(frame["start_dt"])
    time_a = _aware_series(frame["best_quote_time_team1"])
    time_b = _aware_series(frame["best_quote_time_team2"])
    valid_times = cutoff.notna() & start.notna() & time_a.notna() & time_b.notna()
    valid_times &= time_a.le(cutoff) & time_b.le(cutoff) & time_a.lt(start) & time_b.lt(start)
    if not valid_times.all():
        raise ValueError("A0 source quote timestamp is invalid, after cutoff, or not strictly before start")
    frame["cutoff"], frame["start_dt"] = cutoff, start
    frame["best_quote_time_team1"], frame["best_quote_time_team2"] = time_a, time_b
    if frame["best_bookmaker_team1"].isna().any() or frame["best_bookmaker_team2"].isna().any():
        raise ValueError("A0 source quote bookmaker is missing")
    if frame["best_quote_id_team1"].isna().any() or frame["best_quote_id_team2"].isna().any():
        raise ValueError("A0 source quote identifier is missing")
    return frame


def select_a0_ev_bets(
    panel: pd.DataFrame,
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    flat_stake: float = 100.0,
) -> pd.DataFrame:
    """Select one strict-positive-threshold A0 bet per production fixture."""
    if not np.isfinite(tax_rate) or not 0.0 <= tax_rate < 1.0:
        raise ValueError("tax_rate must be finite and in [0, 1)")
    if not np.isfinite(min_ev):
        raise ValueError("min_ev must be finite")
    if not np.isfinite(flat_stake) or flat_stake <= 0.0:
        raise ValueError("flat_stake must be finite and positive")
    frame = _validated_a0_panel(panel)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        selected = best_ev_side(
            float(row["p_a0"]), float(row["best_odds_team1"]), float(row["best_odds_team2"]),
            float(tax_rate), float(min_ev),
        )
        if selected is None:
            continue
        side = str(selected["side"]).upper()
        suffix = "team1" if side == "A" else "team2"
        probability = float(selected["model_prob"])
        odds = float(selected["odds"])
        expected_ev = float(selected["ev"])
        won = bool((int(row["y"]) == 1) if side == "A" else (int(row["y"]) == 0))
        unit_return = odds * (1.0 - float(tax_rate)) - 1.0 if won else -1.0
        record = dict(row)
        record.update(
            side=side, probability=probability, odds=odds,
            market_probability=float(row["p_market"] if side == "A" else 1.0 - row["p_market"]),
            expected_ev=expected_ev,
            unit_return=unit_return, won=won, bookmaker=row[f"best_bookmaker_{suffix}"],
            quote_id=row[f"best_quote_id_{suffix}"], quote_time=row[f"best_quote_time_{suffix}"],
            stake=float(flat_stake), pnl=float(flat_stake) * unit_return,
        )
        records.append(record)
    return pd.DataFrame(records, columns=[*frame.columns, "side", "probability", "odds",
                                          "market_probability", "expected_ev", "unit_return",
                                          "won", "bookmaker", "quote_id", "quote_time", "stake", "pnl"])


def _a0_tolerance(gap: float | None, interval: dict[str, Any], tolerance: float) -> str:
    if gap is None or interval.get("ci_lower") is None or interval.get("ci_upper") is None:
        return "NOT_ESTIMABLE"
    lower, upper = float(interval["ci_lower"]), float(interval["ci_upper"])
    if lower >= -tolerance and upper <= tolerance:
        return "WITHIN_TOLERANCE"
    if upper < -tolerance or lower > tolerance:
        return "OUTSIDE_TOLERANCE"
    return "INCONCLUSIVE"



def evaluate_a0_calibration(
    panel: pd.DataFrame,
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    flat_stake: float = 100.0,
    draws: int = 5000,
    seed: int = 20260927,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Evaluate separate all/common/strict-prior-day flat-stake portfolios."""
    if isinstance(draws, bool) or int(draws) != draws or draws <= 0:
        raise ValueError("draws must be a positive integer")
    frame = _validated_a0_panel(panel)
    bets = select_a0_ev_bets(frame, tax_rate=tax_rate, min_ev=min_ev, flat_stake=flat_stake)
    # Define the common sample from the observed horizon intersection; never pool
    # distinct horizon observations into a single portfolio.
    ids_by_horizon = {
        horizon: set(frame.loc[frame["horizon"].eq(horizon), "golgg_match_id"])
        for horizon in HORIZONS
    }
    common_ids = set.intersection(*(ids_by_horizon[h] for h in HORIZONS)) if all(ids_by_horizon.values()) else set()
    if "is_common" in frame:
        marked = frame["is_common"].map(_strict_bool)
        if marked.isna().any() or ((marked.to_numpy(dtype=bool)) != frame["golgg_match_id"].isin(common_ids).to_numpy()).any():
            raise ValueError("A0 common-sample marker disagrees with the four-horizon intersection")
    frame["_month_block"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m")
    if frame["_month_block"].isna().any():
        raise ValueError("A0 panel contains an invalid calendar date")
    strict_column = "time_compatible_strict_prior_day"
    if strict_column not in frame:
        raise ValueError(f"A0 audit panel is missing required cohort flag {strict_column!r}")
    strict = frame[strict_column].map(_strict_bool)
    if strict.isna().any():
        raise ValueError("A0 strict-prior-day cohort flag is invalid")
    frame["_strict_prior_day"] = strict.astype(bool)
    summaries: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    ledger_parts: list[pd.DataFrame] = []
    for cohort in ("all", "common", "strict_prior_day"):
        for horizon in HORIZONS:
            cohort_mask = frame["horizon"].eq(horizon)
            if cohort == "common":
                cohort_mask &= frame["golgg_match_id"].isin(common_ids)
            elif cohort == "strict_prior_day":
                cohort_mask &= frame["_strict_prior_day"]
            sample = frame.loc[cohort_mask]
            selected = bets.loc[bets["horizon"].eq(horizon) & bets["golgg_match_id"].isin(sample["golgg_match_id"])].copy()
            n = len(sample)
            n_bets = len(selected)
            stake_total = float(selected["stake"].sum()) if n_bets else 0.0
            pnl = float(selected["pnl"].sum()) if n_bets else 0.0
            total_wins = int(selected["won"].sum()) if n_bets else 0
            mean_p = float(selected["probability"].mean()) if n_bets else None
            win_rate = float(total_wins / n_bets) if n_bets else None
            mean_ev = float(selected["expected_ev"].mean()) if n_bets else None
            yield_value = pnl / stake_total if stake_total else None
            yield_gap = yield_value - float(selected["expected_ev"].mul(selected["stake"]).sum() / stake_total) if stake_total else None
            win_gap = float(selected["won"].astype(float).sub(selected["probability"]).mean()) if n_bets else None
            months = int(sample["_month_block"].nunique())
            # Keep all rows in the bootstrap block universe; unselected rows have
            # zero contribution and exposure.
            yield_num = np.zeros(n, dtype=float)
            yield_den = np.zeros(n, dtype=float)
            win_num = np.zeros(n, dtype=float)
            win_den = np.zeros(n, dtype=float)
            if n_bets:
                positions = pd.Index(sample["golgg_match_id"]).get_indexer(selected["golgg_match_id"])
                yield_num[positions] = selected["stake"].to_numpy(float) * (
                    selected["unit_return"].to_numpy(float) - selected["expected_ev"].to_numpy(float)
                )
                yield_den[positions] = selected["stake"].to_numpy(float)
                win_num[positions] = selected["won"].to_numpy(float) - selected["probability"].to_numpy(float)
                win_den[positions] = 1.0
            block_values = sample["_month_block"].reset_index(drop=True)
            yield_interval = block_ratio_interval(yield_num, yield_den, block_values, draws=int(draws), seed=seed)
            win_interval = block_ratio_interval(win_num, win_den, block_values, draws=int(draws), seed=seed)
            summaries.append(dict(
                cohort=cohort, horizon=horizon, n_matches=int(n), n_months=months,
                bets=int(n_bets), wins=total_wins, mean_probability=mean_p,
                win_rate=win_rate, mean_ev=mean_ev, **{"yield": yield_value},
                yield_gap=yield_gap, win_rate_gap=win_gap, pnl=pnl, total_staked=stake_total,
                yield_gap_interval=yield_interval, win_rate_gap_interval=win_interval,
                yield_tolerance_decision=_a0_tolerance(yield_gap, yield_interval, 0.05),
                win_rate_tolerance_decision=_a0_tolerance(win_gap, win_interval, 0.03),
            ))
            selected["cohort"] = cohort
            ledger_parts.append(selected)
            probability_bins = np.minimum((selected["probability"] * 10).astype(int), 9)
            for bin_id in range(10):
                part = selected.loc[probability_bins.eq(bin_id)]
                profile_rows.append(_a0_profile(part, cohort, horizon, "probability_decile",
                                                 f"{bin_id / 10:.1f}-{(bin_id + 1) / 10:.1f}"))
            ev_labels = ["<0.10", "0.10-0.20", "0.20-0.30", ">=0.30"]
            ev_masks = [
                selected["expected_ev"].lt(0.10) if not selected.empty else None,
                (selected["expected_ev"].ge(0.10) & selected["expected_ev"].lt(0.20)) if not selected.empty else None,
                (selected["expected_ev"].ge(0.20) & selected["expected_ev"].lt(0.30)) if not selected.empty else None,
                selected["expected_ev"].ge(0.30) if not selected.empty else None,
            ]
            for band, mask in zip(ev_labels, ev_masks):
                part = selected.loc[mask] if mask is not None else selected
                profile_rows.append(_a0_profile(part, cohort, horizon, "ev_band", band))
            near_40 = selected.loc[selected["probability"].ge(0.35) & selected["probability"].lt(0.45)] if not selected.empty else selected
            profile_rows.append(_a0_profile(near_40, cohort, horizon, "near_40_probability", "[0.35,0.45)"))
    return summaries, pd.concat(ledger_parts, ignore_index=True), pd.DataFrame(profile_rows, columns=[
        "cohort", "horizon", "profile_type", "band", "count", "actual_probability",
        "expected_probability", "yield", "mean_ev",
    ])


def _a0_profile(part: pd.DataFrame, cohort: str, horizon: str, profile_type: str, band: str) -> dict[str, Any]:
    n = len(part)
    return dict(
        cohort=cohort, horizon=horizon, profile_type=profile_type, band=band, count=int(n),
        actual_probability=float(part["won"].mean()) if n else None,
        expected_probability=float(part["probability"].mean()) if n else None,
        **{"yield": float(part["pnl"].sum() / part["stake"].sum()) if n else None},
        mean_ev=float(part["expected_ev"].mean()) if n else None,
    )
