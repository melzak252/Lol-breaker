"""Execute the frozen, chronological all-approaches EV comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import calculate_ece

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json"
)
FROZEN_PROTOCOL_SHA256 = "7bca9fc39bad77d3ecb84f4cd607a076861630dd14cb16467abb30f32db2b0cd"
EPSILON = 1e-12
UNCERTAINTY_POLICIES = (
    "P0", "P1", "P2_0.5", "P2_0.75", "P2_1.0", "P2_1.25",
    "P3_1.0", "P3_1.5", "P3_2.0", "P4", "P5",
)
TAX_RATES = (0.12, 0.0)
EV_THRESHOLDS = (0.03, 0.05)
QUALIFICATIONS = ("raw", "application")
STAKING = ("flat", "quarter_kelly")
ASOF_COHORT = "production_asof_previous_roster"
ASOF_FORECASTERS = ("native_annual_a0", "seed_only", "block_bootstrap")

STANDARD_PROBABILITY_SCORE_KINDS = frozenset({
    "probability", "win_probability", "calibrated_probability",
})
BOUNDED_RAW_SCORE_KINDS = frozenset({
    "tilted_probability_score", "focal_score", "smoothed_probability_score",
})
PROBABILITY_SCORE_KINDS = STANDARD_PROBABILITY_SCORE_KINDS | BOUNDED_RAW_SCORE_KINDS
EVALUATION_FORECAST_COLUMNS = (
    "cohort", "horizon", "row_id", "method", "p", "raw_score", "score_kind",
    "status", "missing_reason", "variant", "model_year", "sigma_z", "p_std",
    "p_q10", "p_q90", "allow_a", "allow_b", "p_low_a", "p_low_b",
    "action_side", "action_fraction",
)


def _is_probability_score_kind(score_kind: Any) -> bool:
    return str(score_kind) in PROBABILITY_SCORE_KINDS


def _probability_interpretation(score_kind: Any) -> str:
    return (
        "BOUNDED_RAW_SCORE_NOT_POSTERIOR"
        if str(score_kind) in BOUNDED_RAW_SCORE_KINDS
        else "COHERENT_WIN_PROBABILITY"
        if _is_probability_score_kind(score_kind)
        else "NOT_A_WIN_PROBABILITY"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat() if pd.notna(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


class _CsvSpool:
    """Export heterogeneous ledger chunks without retaining their DataFrames."""

    def __init__(self, destination: Path):
        self.destination = destination
        self.directory = destination.with_name(f".{destination.name}.parts")
        self.directory.mkdir()
        self.columns: dict[str, None] = {}
        self.parts: list[Path] = []

    def append(self, frame: pd.DataFrame) -> None:
        self.columns.update(dict.fromkeys(frame.columns))
        part = self.directory / f"{len(self.parts):06d}.csv"
        frame.to_csv(part, index=False, float_format="%.17g")
        self.parts.append(part)

    def finish(self) -> None:
        csv.field_size_limit(sys.maxsize)
        with self.destination.open("x", encoding="utf-8", newline="") as target:
            if self.columns:
                writer = csv.DictWriter(target, fieldnames=list(self.columns), lineterminator="\n")
                writer.writeheader()
                for part in self.parts:
                    with part.open(encoding="utf-8", newline="") as source:
                        writer.writerows(csv.DictReader(source))
            else:
                target.write("\n")
        for part in self.parts:
            part.unlink()
        self.directory.rmdir()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported prediction table format: {path}")


def _resolve_research_root(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    marker = PROJECT_ROOT / "data/research_root.txt"
    if candidate is None and marker.is_file():
        candidate = marker.read_text().strip()
    if not candidate:
        raise ValueError("Supply --research-root, ENSEMBLE_RESEARCH_ROOT or data/research_root.txt")
    root = Path(candidate).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Research root is not a directory: {root}")
    return root


def _verify_protocol(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {path}")
    actual = _sha256(path)
    if actual != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            "Frozen all-approaches protocol SHA-256 changed; refusing to execute: "
            f"expected {FROZEN_PROTOCOL_SHA256}, got {actual}"
        )
    protocol = json.loads(path.read_text())
    if protocol.get("run_id") != "ev_all_approaches_20260927_220902":
        raise ValueError("Protocol run_id does not identify the frozen all-approaches experiment")
    if int(protocol.get("statistics", {}).get("bootstrap_draws", 0)) < 5000:
        raise ValueError("Frozen paired inference requires at least 5,000 monthly draws")
    return protocol


def _identity_columns(frame: pd.DataFrame | pd.Series) -> list[str]:
    columns = frame.columns if isinstance(frame, pd.DataFrame) else frame.index
    return [
        key for key in (
            "cohort", "horizon", "row_id", "golgg_match_id", "date",
            "decision_at", "scenario_decision_at", "scenario_prediction_at",
            "start_at", "scenario_start_at",
            "label_available_at", "scenario_label_available_at", "prediction_at",
            "fit_origin", "scenario_fit_origin", "model_year",
            "p_s_annual", "p_s_before_asof", "asof_native_status",
            "asof_native_missing_reason", "best_of", "quote_id_a", "quote_id_b",
        ) if key in columns
    ]


def _opportunity_groups(
    opportunities: pd.DataFrame,
) -> Iterable[tuple[tuple[Any, Any], pd.DataFrame]]:
    for (cohort, horizon), group in opportunities.groupby(
        ["cohort", "horizon"], sort=True, dropna=False
    ):
        if group["row_id"].duplicated().any():
            raise ValueError(f"Duplicate opportunity row_id in {cohort}/{horizon}")
        yield (cohort, horizon), group.sort_values(
            ["date", "row_id"], kind="mergesort"
        ).copy()

def _canonical_sports_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Give canonical OOF rows a stable replay identity without inventing actual times."""
    result = frame.copy()
    if not {"golgg_match_id", "date", "y", "p_s"}.issubset(result.columns):
        raise ValueError("Canonical sports inputs require fixture identity, date, label, and p_s")
    if "row_id" not in result:
        result["row_id"] = "canonical_sports:" + result["golgg_match_id"].astype(str)
    result["cohort"] = "canonical_sports"
    result["horizon"] = "annual"
    # The benchmark's legacy q is native confidence, not a within-book market quote.
    result["q"] = np.nan
    if "decision_at" not in result:
        result["decision_at"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    if "label_available_at" not in result:
        result["label_available_at"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    if "scenario_decision_at" not in result and "scenario_prediction_at" in result:
        result["scenario_decision_at"] = result["scenario_prediction_at"]
    if "scenario_label_available_at" not in result:
        result["scenario_label_available_at"] = pd.NaT
    if result["row_id"].duplicated().any():
        raise ValueError("Canonical sports rows have duplicate replay identities")
    return result



def _identity_atom(column: str, value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if column in {"date"}:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(parsed) else parsed.normalize()
    if column.endswith("_at") or column.endswith("_origin"):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(parsed) else parsed
    if column in {"best_of", "model_year"}:
        number = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(number) else int(number)
    if column == "y":
        number = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(number) else int(number)
    return str(value).strip()

def _reason_or(value: Any, fallback: str) -> str:
    if value is None or value is pd.NA or value is pd.NaT or pd.isna(value):
        return fallback
    reason = str(value).strip()
    return reason or fallback



def _assert_prediction_identity(
    target: pd.Series,
    prediction: pd.Series,
    *,
    label: str,
    required: tuple[str, ...],
    comparable: tuple[str, ...],
) -> None:
    missing = [column for column in required if column not in prediction.index or pd.isna(prediction[column])]
    if missing:
        raise ValueError(f"{label} is missing identity values: {missing}")
    for column in comparable:
        if column not in target.index or column not in prediction.index:
            continue
        left = _identity_atom(column, target[column])
        right = _identity_atom(column, prediction[column])
        if left is not None and right is not None and left != right:
            raise ValueError(f"{label} {column} identity conflict")


def _asof_cohort_inputs(
    opportunities: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clone production quotes into the reconstructed-feature cohort."""
    required = {
        "row_id", "golgg_match_id", "cohort", "horizon", "date", "best_of",
        "forecaster", "scenario_decision_at",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"As-of predictions are missing identity columns: {missing}")
    production = opportunities.loc[
        opportunities["cohort"].astype(str).eq("production")
    ].copy()
    if production.empty:
        raise ValueError("As-of predictions require production opportunities")
    if production.duplicated(["cohort", "horizon", "row_id"]).any():
        raise ValueError("Production opportunities have duplicate as-of source identities")
    source = predictions.copy()
    source["cohort"] = source["cohort"].astype("string").str.strip()
    if not source["cohort"].eq(ASOF_COHORT).all():
        raise ValueError(f"As-of predictions must use cohort {ASOF_COHORT!r}")
    source["forecaster"] = source["forecaster"].astype("string").str.strip()
    unknown_forecasters = sorted(set(source["forecaster"].dropna()) - set(ASOF_FORECASTERS))
    if unknown_forecasters:
        raise ValueError(f"As-of predictions contain unknown forecasters: {unknown_forecasters}")
    source_key = ["cohort", "horizon", "row_id", "forecaster"]
    if source[source_key].isna().any().any() or source.duplicated(source_key).any():
        raise ValueError("As-of predictions have missing or duplicate forecast identities")
    expected_ids = set(zip(
        production["horizon"].astype(str), production["row_id"].astype(str),
    ))
    supplied_ids = set(zip(
        source["horizon"].astype(str), source["row_id"].astype(str),
    ))
    unexpected = sorted(supplied_ids - expected_ids)
    if unexpected:
        raise ValueError(f"As-of predictions reference non-production opportunities: {unexpected[:5]}")
    indexed = source.set_index(source_key, drop=False)
    records: list[dict[str, Any]] = []
    unavailable_native = 0
    identity_fields = (
        "golgg_match_id", "date", "best_of", "y", "decision_at",
        "scenario_decision_at", "start_at", "scenario_start_at",
        "label_available_at", "scenario_label_available_at",
    )
    required_prediction_identity = (
        "row_id", "cohort", "horizon", "golgg_match_id", "date",
        "best_of", "scenario_decision_at",
    )
    for _, source_row in production.iterrows():
        cohort_row = source_row.copy()
        cohort_row["cohort"] = ASOF_COHORT
        source_key = (ASOF_COHORT, str(cohort_row["horizon"]), str(cohort_row["row_id"]))
        forecasts = {
            forecaster: indexed.loc[(*source_key, forecaster)]
            for forecaster in ASOF_FORECASTERS
            if (*source_key, forecaster) in indexed.index
        }
        if any(isinstance(forecast, pd.DataFrame) for forecast in forecasts.values()):
            raise ValueError("As-of forecast identity is not unique")
        prediction = forecasts.get("native_annual_a0")
        metadata_prediction = prediction if prediction is not None else next(iter(forecasts.values()), None)
        cohort_row["p_s_annual"] = source_row.get("p_s_annual", source_row.get("p_s"))
        cohort_row["p_s_before_asof"] = source_row.get("p_s")
        cohort_row["p_s"] = np.nan
        cohort_row["asof_native_status"] = "unavailable"
        cohort_row["asof_native_missing_reason"] = "asof_native_forecast_row_missing"
        if metadata_prediction is not None:
            _assert_prediction_identity(
                cohort_row, metadata_prediction, label="As-of forecast",
                required=required_prediction_identity, comparable=identity_fields,
            )
            cohort_row["scenario_decision_at"] = metadata_prediction.get("scenario_decision_at")
            cohort_row["scenario_prediction_at"] = metadata_prediction.get("scenario_decision_at")
            cohort_row["model_year"] = metadata_prediction.get("model_year")
            for column in (
                "scenario_fit_origin", "scenario_timestamp_assumption",
                "decision_time_source", "roster_status", "roster_a", "roster_b",
                "roster_source_a", "roster_source_b", "feature_source_max_day",
                "feature_available", "cutoff_day",
            ):
                if column in metadata_prediction.index:
                    target_column = (
                        column if column in {"scenario_fit_origin", "scenario_timestamp_assumption"}
                        else f"asof_{column}"
                    )
                    cohort_row[target_column] = metadata_prediction.get(column)
        if prediction is not None:
            status = str(prediction.get("status", "available")).strip().lower()
            raw_value = prediction.get("p", prediction.get("p_full"))
            value = pd.to_numeric(raw_value, errors="coerce")
            available = (
                status == "available" and pd.notna(value)
                and np.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            )
            cohort_row["asof_native_status"] = "available" if available else "unavailable"
            cohort_row["asof_native_missing_reason"] = (
                None if available else _reason_or(
                    prediction.get("missing_reason"),
                    "missing_or_invalid_asof_native_probability",
                )
            )
            if available:
                cohort_row["p_s"] = float(value)
        if pd.isna(cohort_row["p_s"]):
            unavailable_native += 1
        records.append(cohort_row.to_dict())
    cohort = pd.DataFrame(records)
    receipt = {
        "status": "ATTACHED",
        "cohort": ASOF_COHORT,
        "source_rows": int(len(source)),
        "source_forecast_identities": int(len(source)),
        "identity_matches": int(source_key and sum(
            (ASOF_COHORT, str(row.horizon), str(row.row_id), str(row.forecaster)) in indexed.index
            for row in source.itertuples(index=False)
        )),
        "retained_rows": int(len(cohort)),
        "retained_unavailable_rows": int(unavailable_native),
        "row_id_preserved_from_production": True,
        "annual_p_s_retained_for_audit_only": True,
    }
    return cohort, source, receipt



def _probability_record(
    row: pd.Series, method: str, probability: Any, status: str = "available",
    reason: str | None = None, score_kind: str = "probability", **extra: Any,
) -> dict[str, Any]:
    record = {key: row.get(key) for key in _identity_columns(row)}
    for column in (
        "scenario_timestamp_assumption", "source_chronology_status",
        "source_chronology_audit_path", "source_chronology_audit_sha256",
        "source_chronology_audit_status", "source_chronology_audit_late_ids",
        "source_receipt_sha256", "source_prediction_sha256",
        "source_prediction_status", "is_oof", "recipe_status", "prediction_role",
    ):
        if column in row.index:
            record[column if column == "scenario_timestamp_assumption" or column.startswith("source_") else f"source_{column}"] = row.get(column)
    source_scenario_fit_origin = extra.get(
        "source_scenario_fit_origin",
        extra.get(
            "external_scenario_fit_origin",
            extra.get("asof_scenario_fit_origin", row.get("scenario_fit_origin")),
        ),
    )
    fit_kind = (
        "online_feedback" if method.startswith("risk:")
        else "monthly_head" if "head_fit_origin" in extra
        else "stateless" if method in {"market_raw", "fixed_logit_50_50"}
        else "frozen_annual"
    )
    if fit_kind == "monthly_head":
        scenario_fit_origin = extra.get("head_fit_origin", pd.NaT)
    elif fit_kind == "online_feedback":
        scenario_fit_origin = extra.get("effective_decision_at", row.get("scenario_decision_at", row.get("decision_at", pd.NaT)))
    elif fit_kind == "stateless":
        scenario_fit_origin = pd.NaT
    else:
        scenario_fit_origin = source_scenario_fit_origin
    record["fit_kind"] = fit_kind
    record["fit_origin"] = pd.NaT
    record["scenario_fit_origin"] = scenario_fit_origin
    record["source_scenario_fit_origin"] = source_scenario_fit_origin
    record["heldout_y"] = row.get("y")
    record["heldout_label_available_at"] = row.get("label_available_at")
    record["heldout_scenario_label_available_at"] = row.get("scenario_label_available_at")
    for column in row.index:
        if str(column).startswith(("close_", "scenario_close_")):
            record[f"heldout_{column}"] = row.get(column)
    try:
        p = float(probability)
    except (TypeError, ValueError, OverflowError):
        p = np.nan
    is_probability = _is_probability_score_kind(score_kind)
    if not is_probability:
        p = np.nan
    elif not np.isfinite(p) or p < 0.0 or p > 1.0:
        p = np.nan
        if str(status).lower() == "available":
            status = "unavailable"
            reason = reason or "missing_or_invalid_probability"
    record.update({
        "method": method,
        "p": p,
        "raw_score": np.nan,
        "score_kind": score_kind,
        "probability_interpretation": _probability_interpretation(score_kind),
        "probability_certification": _probability_interpretation(score_kind),
        "status": str(status).lower(),
        "missing_reason": reason,
    })
    record.update(extra)
    record["fit_kind"] = fit_kind
    record["fit_origin"] = pd.NaT
    record["scenario_fit_origin"] = scenario_fit_origin
    record["source_scenario_fit_origin"] = source_scenario_fit_origin
    return record


def _fixed_fifty(p_s: Any, q: Any) -> float:
    try:
        a, b = float(p_s), float(q)
    except (TypeError, ValueError):
        return np.nan
    if not (np.isfinite(a) and np.isfinite(b) and 0.0 < a < 1.0 and 0.0 < b < 1.0):
        return np.nan
    return float(1.0 / (1.0 + math.exp(-0.5 * (math.log(a / (1.0 - a)) + math.log(b / (1.0 - b))))))


def _score_metrics(frame: pd.DataFrame, probability: str = "p") -> dict[str, Any]:
    y = pd.to_numeric(frame.get("y", pd.Series(index=frame.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(frame.get(probability, pd.Series(index=frame.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isin(y, (0.0, 1.0)) & np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    n_scored = int(valid.sum())
    result: dict[str, Any] = {
        "n_opportunities": int(len(frame)),
        "n_scored": n_scored,
        "n_unavailable": int(len(frame) - n_scored),
        "coverage": float(n_scored / len(frame)) if len(frame) else None,
        "boundary_clips": 0,
        "log_loss": None,
        "brier": None,
        "auc": None,
        "accuracy": None,
        "ece10": None,
        "tail_log_loss_ge_2_5": None,
    }
    if not n_scored:
        return result
    observed, values = y[valid], p[valid]
    clipped = np.clip(values, EPSILON, 1.0 - EPSILON)
    result["boundary_clips"] = int(np.count_nonzero(values != clipped))
    ll = -(observed * np.log(clipped) + (1.0 - observed) * np.log1p(-clipped))
    result["log_loss"] = float(ll.mean())
    result["brier"] = float(np.mean((values - observed) ** 2))
    result["accuracy"] = float(np.mean((values >= 0.5) == observed))
    if len(np.unique(observed)) == 2:
        ranks = rankdata(values, method="average")
        positive = observed == 1.0
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        result["auc"] = float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    result["ece10"] = float(calculate_ece(observed, clipped, n_bins=10))
    result["tail_log_loss_ge_2_5"] = int(np.count_nonzero(ll >= 2.5))
    return result


def _ranking_metrics(frame: pd.DataFrame, score: str = "raw_score") -> dict[str, Any]:
    y = pd.to_numeric(frame.get("y", pd.Series(index=frame.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame.get(score, pd.Series(index=frame.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isin(y, (0.0, 1.0)) & np.isfinite(values)
    observed, ranked = y[valid], values[valid]
    auc = None
    status = "UNAVAILABLE_SINGLE_CLASS" if len(np.unique(observed)) < 2 else "AVAILABLE"
    if status == "AVAILABLE":
        ranks = rankdata(ranked, method="average")
        positive = observed == 1.0
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        auc = float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return {
        "rank_n_scored": int(valid.sum()),
        "rank_coverage": float(valid.sum() / len(frame)) if len(frame) else None,
        "rank_auc": auc,
        "rank_discrimination_status": status if valid.any() else "UNAVAILABLE_NO_RANK_SCORES",
    }




def _score_pair_bootstrap(
    universe: pd.DataFrame,
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Paired monthly LL/Brier contrasts on common observed forecast coverage."""
    key = ["row_id"]
    paired = universe[["row_id", "date", "y"]].merge(
        method_a[key + ["p"]].rename(columns={"p": "p_a"}), on=key,
        how="left", validate="one_to_one",
    ).merge(
        method_b[key + ["p"]].rename(columns={"p": "p_b"}), on=key,
        how="left", validate="one_to_one",
    )
    paired["date"] = pd.to_datetime(paired["date"], utc=True, errors="coerce")
    paired["month"] = paired["date"].dt.strftime("%Y-%m")
    y = pd.to_numeric(paired["y"], errors="coerce").to_numpy(dtype=float)
    pa = pd.to_numeric(paired["p_a"], errors="coerce").to_numpy(dtype=float)
    pb = pd.to_numeric(paired["p_b"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isin(y, (0.0, 1.0)) & np.isfinite(pa) & np.isfinite(pb) & (pa >= 0) & (pa <= 1) & (pb >= 0) & (pb <= 1)
    months = sorted(paired["month"].dropna().unique().tolist())
    if not valid.any():
        return {
            "status": "UNAVAILABLE_NO_COMMON_PREDICTIONS",
            "n_paired": 0, "months": months, "draws": draws,
        }
    yy, pa, pb = y[valid], pa[valid], pb[valid]
    a, b = np.clip(pa, EPSILON, 1.0 - EPSILON), np.clip(pb, EPSILON, 1.0 - EPSILON)
    ll_delta = np.zeros(len(paired), dtype=float)
    brier_delta = np.zeros(len(paired), dtype=float)
    ll_delta[valid] = (
        -(yy * np.log(a) + (1.0 - yy) * np.log1p(-a))
        + (yy * np.log(b) + (1.0 - yy) * np.log1p(-b))
    )
    brier_delta[valid] = (pa - yy) ** 2 - (pb - yy) ** 2
    monthly = pd.DataFrame({
        "month": paired["month"], "count": valid.astype(int),
        "log_loss": ll_delta, "brier": brier_delta,
    }).groupby("month", sort=True).sum()
    counts = monthly["count"].to_numpy(dtype=float)
    active = counts > 0
    result = {
        "n_paired": int(valid.sum()),
        "n_unpaired": int((~valid).sum()),
        "months": months,
        "missing_months": monthly.index[~active].tolist(),
        "draws": draws, "seed": seed,
    }
    for metric in ("log_loss", "brier"):
        sums = monthly[metric].to_numpy(dtype=float)
        weighted = block_ratio_interval(sums, counts, monthly.index, draws, seed)
        monthly_means = np.divide(sums, counts, out=np.zeros_like(sums), where=active)
        equal_month = block_ratio_interval(
            monthly_means, active.astype(float), monthly.index, draws, seed,
        )
        result[f"delta_{metric}_a_minus_b"] = weighted["estimate"]
        result[f"equal_month_delta_{metric}_a_minus_b"] = equal_month["estimate"]
        result[f"{metric}_ci"] = [weighted["ci_lower"], weighted["ci_upper"]]
        result[f"equal_month_{metric}_ci"] = [equal_month["ci_lower"], equal_month["ci_upper"]]
        result[f"{metric}_bootstrap"] = weighted
        result[f"equal_month_{metric}_bootstrap"] = equal_month
    result["status"] = result["log_loss_bootstrap"]["status"]
    return result


def _match_uncertainty(
    opportunities: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> list[dict[str, Any]]:
    if uncertainty.empty:
        return []
    role_column = (
        "prediction_role" if "prediction_role" in uncertainty
        else "partition" if "partition" in uncertainty else None
    )
    if role_column is None:
        raise ValueError("Uncertainty rows must declare prediction_role or partition (TEST/CAL)")
    roles = uncertainty[role_column].astype(str).str.upper()
    if "partition" in uncertainty:
        test_mask = uncertainty["partition"].astype(str).str.upper().eq("TEST")
    else:
        test_mask = roles.isin({"TEST", "OOF_PREDICTION"})
    test = uncertainty.loc[test_mask].copy()
    if test.empty:
        return []
    identity = ["golgg_match_id", "date", "best_of"]
    if not set(identity).issubset(opportunities.columns) or not set(identity).issubset(test.columns):
        raise ValueError("Uncertainty attachment requires match ID, date, and best_of identity")
    left = opportunities.copy()
    right = test.copy()
    for table in (left, right):
        table["golgg_match_id"] = table["golgg_match_id"].astype(str)
        table["date"] = pd.to_datetime(table["date"], utc=True, errors="coerce")
        table["best_of"] = pd.to_numeric(table["best_of"], errors="coerce").astype("Int64")
    variants = sorted(right["variant"].dropna().astype(str).unique()) if "variant" in right else []
    if not variants:
        raise ValueError("TEST uncertainty rows must identify their variant")
    records: list[dict[str, Any]] = []
    for variant in variants:
        part = right.loc[right["variant"].astype(str).eq(variant)].copy()
        keys = identity
        if (
            "row_id" in part and "row_id" in left
            and part["row_id"].astype(str).isin(left["row_id"].astype(str)).all()
        ):
            keys = [*identity, "row_id"]
        if part.duplicated(keys).any():
            raise ValueError(f"Duplicate TEST uncertainty identity for variant {variant}")
        source_columns = (
            "p", "sigma_z", "p_std", "p_q10", "p_q90", "model_year",
            "variant", "is_oof", "recipe_status", "prediction_role",
            "fit_origin", "scenario_fit_origin", "prediction_at", "source_prediction_status",
        )
        right_names = {
            column: f"uncertainty_{column}"
            for column in source_columns if column in part and column not in keys
        }
        right_columns = keys + [
            column for column in source_columns if column in part and column not in keys
        ]
        if "y" in part:
            right_columns.append("y")
        right_values = part[right_columns].rename(
            columns={**right_names, "y": "y_uncertainty"}
        )
        matched = left.merge(
            right_values, on=keys, how="left", validate="many_to_one", indicator=True,
        )
        if "y_uncertainty" in matched:
            both = matched["y_uncertainty"].notna()
            if bool((matched.loc[both, "y"] != matched.loc[both, "y_uncertainty"]).any()):
                raise ValueError(f"Uncertainty y conflicts with opportunity outcomes for {variant}")
        for _, row in matched.iterrows():
            p_value = row.get("uncertainty_p")
            valid = (
                pd.notna(p_value) and np.isfinite(float(p_value))
                and 0.0 < float(p_value) < 1.0
            )
            status = "available" if valid else "unavailable"
            reason = None if valid else (
                "uncertainty_test_row_not_found"
                if row["_merge"] != "both" else "missing_or_invalid_uncertainty_probability"
            )
            source_fit_origin = row.get("uncertainty_scenario_fit_origin")
            if pd.isna(source_fit_origin):
                source_fit_origin = row.get("scenario_fit_origin")
            if pd.isna(source_fit_origin):
                model_year = row.get("uncertainty_model_year")
                source_fit_origin = (
                    pd.Timestamp(year=int(model_year), month=1, day=1, tz="UTC")
                    if pd.notna(model_year) else pd.NaT
                )
            records.append(_probability_record(
                row, f"uncertainty:{variant}", p_value, status, reason,
                variant=variant,
                sigma_z=row.get("uncertainty_sigma_z"), p_std=row.get("uncertainty_p_std"),
                p_q10=row.get("uncertainty_p_q10"), p_q90=row.get("uncertainty_p_q90"),
                model_year=row.get("uncertainty_model_year"), prediction_role="TEST",
                uncertainty_is_oof=row.get("uncertainty_is_oof"),
                uncertainty_recipe_status=row.get("uncertainty_recipe_status"),
                uncertainty_fit_origin=row.get("uncertainty_fit_origin"),
                uncertainty_prediction_at=row.get("uncertainty_prediction_at"),
                uncertainty_source_prediction_status=row.get("uncertainty_source_prediction_status"),
                availability_source="frozen_uncertainty_TEST",
                source_scenario_fit_origin=source_fit_origin,
            ))
    return records


def _source_groups(source: pd.DataFrame, source_name: str) -> list[tuple[tuple[Any, ...], pd.DataFrame, list[str]]]:
    if source_name == "asof":
        group_columns = ["forecaster"]
    else:
        group_columns = [column for column in ("objective", "arm") if column in source]
        if not group_columns and "method" in source:
            group_columns = ["method"]
        if not group_columns:
            raise ValueError(f"{source_name} predictions must identify their target objective")
    grouped = source.groupby(group_columns, dropna=False, sort=True)
    output = []
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        output.append((key_tuple, part, group_columns))
    return output


def _native_target_source_receipt(path: Path, source: pd.DataFrame) -> dict[str, Any]:
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.is_file():
        return {"status": "NOT_SUPPLIED", "path": None}
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "VERIFIED_COMPLETE_NATIVE_TARGET_INPUT":
        raise ValueError("Native-target source receipt is not verified complete")
    if receipt.get("sha256") != _sha256(path):
        raise ValueError("Native-target source receipt hash does not match its prediction file")
    if int(receipt.get("rows", -1)) != len(source):
        raise ValueError("Native-target source receipt row count does not match its predictions")
    if "objective" not in source:
        raise ValueError("Native-target candidate file must preserve objective identities")
    actual_objectives = {
        str(objective): int(count)
        for objective, count in source["objective"].astype(str).value_counts().items()
    }
    expected_objectives = {
        str(objective): int(count)
        for objective, count in receipt.get("rows_per_objective", {}).items()
    }
    if actual_objectives != expected_objectives:
        raise ValueError("Native-target source receipt objective counts do not match")
    input_hashes = receipt.get("inputs_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("Native-target source receipt omits its pinned inputs")
    for name, expected_hash in input_hashes.items():
        if Path(name).name != name:
            raise ValueError("Native-target source receipt contains a nonlocal input path")
        input_path = path.parent / name
        if not input_path.is_file() or _sha256(input_path) != expected_hash:
            raise ValueError(f"Native-target source receipt input does not match: {name}")
    return {
        "status": "VERIFIED",
        "path": str(receipt_path),
        "receipt": receipt,
    }


def _external_prediction_rows(
    opportunities: pd.DataFrame,
    path: Path | None,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"status": "NOT_SUPPLIED", "path": None, "attached": 0}
    source = _read_table(path).copy()
    source_receipt = (
        _native_target_source_receipt(path, source)
        if source_name == "native_target" else None
    )
    required = {"golgg_match_id", "date", "best_of"}
    if not required.issubset(source.columns):
        raise ValueError(f"{source_name} must include full fixture identity columns {sorted(required)}")
    if source_name == "asof":
        raise ValueError("As-of predictions must be attached through the reconstructed opportunity cohort")
    if source["golgg_match_id"].isna().any() or source["date"].isna().any():
        raise ValueError(f"{source_name} fixture identity contains missing values")
    if source["best_of"].isna().any():
        raise ValueError(f"{source_name} best_of identity contains missing values")
    groups = _source_groups(source, source_name)
    left_by_fixture: dict[tuple[Any, ...], list[pd.Series]] = {}
    for _, target in opportunities.iterrows():
        key = tuple(
            _identity_atom(column, target[column])
            for column in ("golgg_match_id", "date", "best_of")
        )
        left_by_fixture.setdefault(key, []).append(target)
    methods: set[str] = set()
    records: list[dict[str, Any]] = []
    matched_count = 0
    available_count = 0
    for group_key, part, group_columns in groups:
        lookup: dict[tuple[Any, ...], pd.Series] = {}
        for _, prediction in part.iterrows():
            key = tuple(
                _identity_atom(column, prediction[column])
                for column in ("golgg_match_id", "date", "best_of")
            )
            if key in lookup:
                raise ValueError(f"{source_name} contains duplicate objective/fixture identities")
            lookup[key] = prediction
            if key not in left_by_fixture:
                raise ValueError(f"{source_name} contains a fixture outside its scoring universe")
        values = dict(zip(group_columns, group_key))
        objective = values.get("objective")
        arm = values.get("arm")
        label = objective if objective is not None and not pd.isna(objective) else arm
        if label is None or pd.isna(label):
            label = values.get("method")
        if label is None or pd.isna(label):
            label = ":".join(str(value) for value in group_key)
        method = f"{source_name}:{label}"
        if method in methods:
            method += ":" + ":".join(str(value) for value in group_key)
        methods.add(method)
        for fixture_key, targets in left_by_fixture.items():
            prediction = lookup.get(fixture_key)
            if prediction is not None:
                for target in targets:
                    _assert_prediction_identity(
                        target, prediction, label=source_name,
                        required=("golgg_match_id", "date", "best_of"),
                        comparable=("golgg_match_id", "date", "best_of", "y"),
                    )
                    matched_count += 1
            for target in targets:
                raw_score_kind = (
                    prediction.get("score_kind", "probability")
                    if prediction is not None else "probability"
                )
                score_kind = (
                    str(raw_score_kind).strip()
                    if pd.notna(raw_score_kind) and str(raw_score_kind).strip()
                    else "probability"
                )
                probability_columns = ("p", "p_full", "probability")
                value_column = (
                    next((column for column in probability_columns if column in source), None)
                    if _is_probability_score_kind(score_kind)
                    else "raw_score" if "raw_score" in source
                    else next((column for column in ("score", "p", "p_full", "probability") if column in source), None)
                )
                if value_column is None:
                    raise ValueError(f"{source_name} has no prediction or raw-score column")
                value = prediction.get(value_column) if prediction is not None else np.nan
                number = pd.to_numeric(value, errors="coerce")
                raw_status = (
                    prediction.get("status")
                    if prediction is not None and "status" in prediction.index else "available"
                )
                source_status = (
                    str(raw_status).strip().lower()
                    if pd.notna(raw_status) and str(raw_status).strip()
                    else "unavailable"
                )
                available = (
                    source_status == "available" and pd.notna(number)
                    and np.isfinite(float(number))
                )
                if _is_probability_score_kind(score_kind):
                    available = bool(available and 0.0 <= float(number) <= 1.0)
                reason_fallback = (
                    "external_prediction_not_found" if prediction is None
                    else f"external_source_status_{source_status}"
                    if source_status != "available"
                    else "missing_or_invalid_external_prediction"
                )
                reason = None if available else _reason_or(
                    prediction.get("missing_reason") if prediction is not None else None,
                    reason_fallback,
                )
                extra: dict[str, Any] = {
                    "availability_source": source_name,
                    "source_objective": objective,
                    "source_arm": arm,
                }
                if objective is not None and not pd.isna(objective):
                    extra["objective"] = objective
                if arm is not None and not pd.isna(arm):
                    extra["arm"] = arm
                if prediction is not None:
                    for column in (
                        "sigma_z", "p_std", "p_q10", "p_q90", "model_year",
                        "status", "missing_reason",
                    ):
                        if column in prediction.index:
                            extra[f"external_{column}" if column in {"status", "missing_reason"} else column] = prediction.get(column)
                    for column in prediction.index:
                        if column not in {
                            *required, "row_id", "cohort", "horizon", "decision_at",
                            "scenario_decision_at", "start_at", "scenario_start_at",
                            "label_available_at", "scenario_label_available_at",
                            "y", "p", "p_full", "probability", "raw_score", "score",
                            "score_kind", "status", "missing_reason", *group_columns,
                            "sigma_z", "p_std", "p_q10", "p_q90", "model_year",
                        }:
                            extra[f"external_{column}"] = prediction.get(column)
                record = _probability_record(
                    target, method,
                    number if available and _is_probability_score_kind(score_kind) else np.nan,
                    "available" if available else "unavailable",
                    reason,
                    score_kind=score_kind,
                    raw_score=(
                        float(number) if available and not _is_probability_score_kind(score_kind)
                        else np.nan
                    ),
                    **extra,
                )
                records.append(record)
                available_count += int(available)
    return records, {
        "status": "ATTACHED",
        "path": str(path),
        "source_rows": int(len(source)),
        "identity_matches": int(matched_count),
        "available_predictions": int(available_count),
        "rows": len(records),
        "method_ids": sorted(methods),
        "sha256": _sha256(path),
        "source_receipt": source_receipt,
    }


def _asof_prediction_records(
    opportunities: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_keys = ["cohort", "horizon", "row_id", "forecaster"]
    indexed = predictions.set_index(index_keys, drop=False)
    identity_fields = (
        "golgg_match_id", "date", "best_of", "y", "decision_at",
        "scenario_decision_at", "start_at", "scenario_start_at",
        "label_available_at", "scenario_label_available_at",
    )
    records: list[dict[str, Any]] = []
    matched_count = 0
    available_count = 0
    for _, target in opportunities.iterrows():
        for forecaster in ASOF_FORECASTERS:
            key = (ASOF_COHORT, str(target["horizon"]), str(target["row_id"]), forecaster)
            prediction = indexed.loc[key] if key in indexed.index else None
            if isinstance(prediction, pd.DataFrame):
                raise ValueError("As-of forecast identity is not unique")
            if prediction is not None:
                _assert_prediction_identity(
                    target, prediction, label="As-of forecast",
                    required=(
                        "row_id", "cohort", "horizon", "golgg_match_id",
                        "date", "best_of", "scenario_decision_at",
                    ),
                    comparable=identity_fields,
                )
                matched_count += 1
            score_kind = str(
                prediction.get("score_kind", "probability")
                if prediction is not None else "probability"
            )
            value = (
                prediction.get("p", prediction.get("p_full"))
                if prediction is not None else np.nan
            )
            number = pd.to_numeric(value, errors="coerce")
            source_status = (
                str(prediction.get("status", "available")).lower()
                if prediction is not None else "unavailable"
            )
            available = (
                source_status == "available" and pd.notna(number)
                and np.isfinite(float(number)) and 0.0 <= float(number) <= 1.0
            )
            method = (
                "native_annual_a0" if forecaster == "native_annual_a0"
                else f"uncertainty:{forecaster}"
            )
            reason_fallback = (
                "asof_forecast_row_missing" if prediction is None
                else "missing_or_invalid_asof_probability"
            )
            reason = None if available else _reason_or(
                prediction.get("missing_reason") if prediction is not None else None,
                reason_fallback,
            )
            extras: dict[str, Any] = {
                "availability_source": "asof_reconstructed_prediction",
                "asof_forecaster": forecaster,
                "source_scenario_fit_origin": prediction.get("scenario_fit_origin") if prediction is not None else target.get("scenario_fit_origin"),
                "model_year": prediction.get("model_year") if prediction is not None else None,
                "asof_status": source_status,
                "asof_missing_reason": reason,
                "asof_fit_origin": prediction.get("fit_origin") if prediction is not None else pd.NaT,
                "asof_scenario_fit_origin": prediction.get("scenario_fit_origin") if prediction is not None else target.get("scenario_fit_origin"),
            }
            for column in (
                "sigma_z", "p_std", "p_q10", "p_q90", "fit_origin_source",
                "decision_time_source", "roster_status", "roster_source_a",
                "roster_source_b", "feature_source_max_day", "feature_available",
            ):
                if prediction is not None and column in prediction.index:
                    extras[column if column in {"sigma_z", "p_std", "p_q10", "p_q90"} else f"asof_{column}"] = prediction.get(column)
            record = _probability_record(
                target, method, float(number) if available else np.nan,
                "available" if available else "unavailable",
                reason, score_kind,
                variant=forecaster if forecaster != "native_annual_a0" else None,
                **extras,
            )
            records.append(record)
            available_count += int(available)
    return records, {
        "status": "ATTACHED",
        "cohort": ASOF_COHORT,
        "source_rows": int(len(predictions)),
        "identity_matches": int(matched_count),
        "available_predictions": int(available_count),
        "rows": int(len(records)),
        "method_ids": ["native_annual_a0", "uncertainty:seed_only", "uncertainty:block_bootstrap"],
    }


def _scope_views(group: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    views = [("all", group)]
    if "is_common" in group:
        views.append(("common", group.loc[group["is_common"].fillna(False).astype(bool)]))
    if "strict_prior_day" in group:
        views.append((
            "strict_prior_day",
            group.loc[group["strict_prior_day"].fillna(False).astype(bool)],
        ))
    elif "time_compatible_strict_prior_day" in group:
        views.append((
            "strict_prior_day",
            group.loc[group["time_compatible_strict_prior_day"].fillna(False).astype(bool)],
        ))
    return views


def _forecast_view(
    opportunities: pd.DataFrame,
    forecasts: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    source = forecasts.loc[forecasts["method"].eq(method)].copy()
    keys = ["cohort", "horizon", "row_id"]
    if source.duplicated(keys).any():
        raise ValueError(f"Forecast {method} has duplicate opportunity identity")
    values = [column for column in source.columns if column not in keys]
    merged = opportunities.merge(source[keys + values], on=keys, how="left", validate="one_to_one", suffixes=("", "__forecast"))
    for column in ("p", "raw_score", "score_kind", "status", "missing_reason", "variant", "model_year", "sigma_z", "p_std", "p_q10", "p_q90", "allow_a", "allow_b", "p_low_a", "p_low_b", "action_side", "action_fraction"):
        forecast_column = f"{column}__forecast"
        if forecast_column in merged:
            merged[column] = merged[forecast_column]
            merged.drop(columns=forecast_column, inplace=True)
    return merged


def _effective_decision_mask(frame: pd.DataFrame) -> pd.Series:
    actual = (
        frame["decision_at"].notna()
        if "decision_at" in frame else pd.Series(False, index=frame.index)
    )
    scenario = (
        frame["scenario_decision_at"].notna()
        if "scenario_decision_at" in frame else pd.Series(False, index=frame.index)
    )
    assumption = (
        frame["scenario_timestamp_assumption"].notna()
        & frame["scenario_timestamp_assumption"].astype(str).str.strip().ne("")
        if "scenario_timestamp_assumption" in frame
        else pd.Series(False, index=frame.index)
    )
    return actual | (scenario & assumption)


def _actual_release_values(frame: pd.DataFrame) -> pd.Series:
    release = (
        frame["label_available_at"].copy()
        if "label_available_at" in frame
        else pd.Series(pd.NaT, index=frame.index)
    )
    if "settle_at" in frame:
        source = frame.get(
            "settlement_time_source", pd.Series("actual", index=frame.index)
        ).astype("string")
        release = release.where(
            release.notna(), frame["settle_at"].where(source.eq("actual").fillna(False)),
        )
    return release



def _action_bets(frame: pd.DataFrame, tax_rate: float) -> pd.DataFrame:
    rows = []
    tax = float(tax_rate)
    for row in frame.itertuples(index=False):
        side = getattr(row, "action_side", None)
        if side not in {"A", "B"}:
            continue
        decision_at = getattr(row, "decision_at", pd.NaT)
        scenario_decision_at = getattr(row, "scenario_decision_at", pd.NaT)
        assumption = getattr(row, "scenario_timestamp_assumption", None)
        has_scenario_assumption = (
            assumption is not None and not pd.isna(assumption) and str(assumption).strip()
        )
        if pd.isna(decision_at) and not (
            not pd.isna(scenario_decision_at) and has_scenario_assumption
        ):
            continue
        letter = side.lower()
        raw_odds = getattr(row, f"odds_{letter}")
        if pd.isna(raw_odds) or not np.isfinite(float(raw_odds)) or float(raw_odds) <= 1.0:
            continue
        odds = float(raw_odds)
        y = getattr(row, "y")
        won = None if pd.isna(y) else bool(y == (1 if side == "A" else 0))
        identity = getattr(row, "row_id")
        label_available_at = getattr(row, "label_available_at", pd.NaT)
        scenario_label_available_at = getattr(row, "scenario_label_available_at", pd.NaT)
        actual_settle_at = label_available_at
        if pd.isna(actual_settle_at):
            actual_settle_at = getattr(row, "settle_at", pd.NaT)
        settlement_time_source = (
            "actual" if not pd.isna(actual_settle_at)
            else "scenario" if not pd.isna(scenario_label_available_at) and has_scenario_assumption
            else None
        )
        rows.append({
            "bet_id": f"{identity}:{side}",
            "row_id": identity,
            "cohort": getattr(row, "cohort", None),
            "horizon": getattr(row, "horizon", None),
            "method": getattr(row, "method", None),
            "selection_type": "decision_only_action",
            "side": side,
            "action_side": side,
            "action_fraction": float(getattr(row, "action_fraction")),
            "odds": odds,
            "bookmaker": getattr(row, f"bookmaker_{letter}"),
            "quote_id": getattr(row, f"quote_id_{letter}"),
            "quote_at": getattr(row, f"quote_at_{letter}", pd.NaT),
            "decision_at": decision_at,
            "scenario_decision_at": scenario_decision_at,
            "scenario_timestamp_assumption": assumption,
            "start_at": getattr(row, "start_at", pd.NaT),
            "label_available_at": label_available_at,
            "settle_at": actual_settle_at,
            "settled_at": actual_settle_at,
            "scenario_label_available_at": scenario_label_available_at,
            "settlement_time_source": settlement_time_source,
            "won": won,
            "tax_rate": tax,
            "p": np.nan,
            "p_for_ev": np.nan,
            "ev": np.nan,
            "unit_return": (
                None if won is None else odds * (1.0 - tax) - 1.0 if won else -1.0
            ),
            "reference_p_s": getattr(row, "p_s", np.nan),
            "raw_score": getattr(row, "raw_score", np.nan),
            "league": getattr(row, "league", None),
            "best_of": getattr(row, "best_of", None),
            "close_odds": getattr(row, f"close_odds_{letter}", getattr(row, "close_odds", np.nan)),
            "close_available_at": getattr(row, "close_available_at", pd.NaT),
            "close_same_book": getattr(row, f"close_same_book_{letter}", getattr(row, "close_same_book", False)),
            "close_bookmaker": getattr(row, f"close_bookmaker_{letter}", getattr(row, "close_bookmaker", None)),
        })
    columns = [
        "bet_id", "row_id", "cohort", "horizon", "method", "selection_type",
        "side", "action_side", "action_fraction", "odds", "bookmaker", "quote_id",
        "quote_at", "decision_at", "scenario_decision_at",
        "scenario_timestamp_assumption", "start_at", "label_available_at",
        "settle_at", "settled_at", "scenario_label_available_at",
        "settlement_time_source", "won", "tax_rate",
        "p", "p_for_ev", "ev", "unit_return", "reference_p_s", "raw_score",
        "league", "best_of", "close_odds", "close_available_at", "close_same_book",
        "close_bookmaker",
    ]
    return pd.DataFrame(rows, columns=columns)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result
    raise TypeError(f"Expected a serializable mapping summary, got {type(value).__name__}")


def _chronological_head_matrix(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    lambdas: tuple[float, ...],
    min_train: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from src.analysis.ev_approach_data import effective_times
    from src.analysis.ev_approach_evaluation import (
        _empty_prediction, chronological_head_predictions,
    )

    predictions: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    for (cohort, horizon), group in _opportunity_groups(frame):
        effective_decision = effective_times(group, "decision_at")
        known = effective_decision.notna()
        if (~known).any():
            unavailable = group.loc[~known]
            for method in methods:
                predictions.append(pd.DataFrame([
                    _empty_prediction(row, method, pd.NaT, "unknown_effective_decision_time")
                    for _, row in unavailable.iterrows()
                ]))
                receipts.append({
                    "cohort": cohort, "horizon": horizon, "method": method,
                    "status": "UNAVAILABLE_NO_EFFECTIVE_DECISION_TIME",
                    "missing_reason": "unknown_effective_decision_time",
                    "unavailable_row_ids": unavailable["row_id"].astype(str).tolist(),
                    "train_ids": [], "calibration_ids": [],
                })
        if known.any():
            group_predictions, group_receipts = chronological_head_predictions(
                group.loc[known].copy(),
                methods=methods,
                lambdas=lambdas,
                min_train=min_train,
            )
            if not group_predictions.empty:
                predictions.append(group_predictions)
            receipts.extend(group_receipts)
    matrix = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    return matrix, receipts



def _p5_threshold_receipts(sigma_cal: pd.DataFrame) -> list[dict[str, Any]]:
    if sigma_cal.empty or not {"variant", "model_year", "sigma_z"}.issubset(sigma_cal.columns):
        return []
    if "partition" in sigma_cal:
        partition = sigma_cal["partition"].astype(str).str.upper()
        if not partition.eq("CAL").all():
            raise ValueError("P5 thresholds may use CAL rows only")
    if "prediction_role" in sigma_cal:
        roles = sigma_cal["prediction_role"].dropna().astype(str).str.upper().unique()
        if len(roles) > 1 or (len(roles) == 1 and roles[0] not in {"SIGMA_REFERENCE", "CAL_REFERENCE"}):
            raise ValueError("P5 threshold inputs have a non-CAL prediction role")
    release_column = (
        "label_available_at" if "label_available_at" in sigma_cal else None
    )
    release = (
        pd.to_datetime(sigma_cal[release_column], utc=True, errors="coerce")
        if release_column is not None
        else pd.Series(pd.NaT, index=sigma_cal.index, dtype="datetime64[ns, UTC]")
    )
    if "scenario_label_available_at" in sigma_cal:
        scenario_release = pd.to_datetime(
            sigma_cal["scenario_label_available_at"], utc=True, errors="coerce",
        )
        if "scenario_timestamp_assumption" in sigma_cal:
            assumption = (
                sigma_cal["scenario_timestamp_assumption"].notna()
                & sigma_cal["scenario_timestamp_assumption"].astype(str).str.strip().ne("")
            )
        else:
            assumption = pd.Series(False, index=sigma_cal.index)
        release = release.where(release.notna(), scenario_release.where(assumption))
    thresholds: list[dict[str, Any]] = []
    for (variant, year), part in sigma_cal.groupby(
        ["variant", "model_year"], dropna=False, sort=True,
    ):
        try:
            model_year = int(float(year))
        except (TypeError, ValueError, OverflowError):
            continue
        part_release = release.loc[part.index]
        sigma_values = pd.to_numeric(part["sigma_z"], errors="coerce")
        finite = sigma_values.notna() & np.isfinite(sigma_values)
        origin = pd.Timestamp(year=model_year, month=1, day=1, tz="UTC")
        eligible = finite & part_release.notna() & part_release.lt(origin)
        cal_sigma = sigma_values.loc[eligible]
        if cal_sigma.empty:
            continue
        max_release = part_release.loc[eligible].max()
        role_values = (
            part.loc[eligible, "prediction_role"].dropna().astype(str).str.upper().unique().tolist()
            if "prediction_role" in part else []
        )
        identity_column = "row_id" if "row_id" in part else "golgg_match_id"
        if identity_column not in part:
            raise ValueError("P5 CAL references require source row identities")
        identities = part.loc[eligible, identity_column].astype("string").str.strip()
        if identities.isna().any() or identities.eq("").any() or identities.duplicated().any():
            raise ValueError("P5 CAL source identities must be nonmissing and unique")
        thresholds.append({
            "cal_reference_id": f"{variant}:{model_year}",
            "variant": variant,
            "model_year": model_year,
            "partition": "CAL",
            "sigma_z_q75": float(cal_sigma.quantile(0.75)),
            "cal_rows": int(len(cal_sigma)),
            "cal_row_ids": identities.tolist(),
            "prediction_role": role_values[0] if len(role_values) == 1 else "CAL_REFERENCE",
            "source": "frozen_CAL_sigma_z_only",
            "is_whole_recipe_oof": False,
            "max_calibration_label_release_at": max_release.isoformat(),
            "status": "AVAILABLE_CAL_REFERENCE",
        })
    return thresholds


def _asof_source_receipt(path: Path) -> dict[str, Any]:
    run_id = path.parent.name
    candidates = [
        path.parent / "asof_replay_receipt.json",
        PROJECT_ROOT / "data/08_reporting/ev_asof_replay" / run_id / "asof_replay_receipt.json",
    ]
    receipt_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if receipt_path is None:
        return {"status": "NOT_SUPPLIED", "path": None, "predictions_path": str(path)}
    producer_receipt = json.loads(receipt_path.read_text())
    output_path = producer_receipt.get("output_paths", {}).get("predictions")
    if output_path is None or Path(output_path).expanduser().resolve() != path.resolve():
        raise ValueError("As-of producer receipt does not identify the supplied prediction file")
    expected = producer_receipt.get("output_hashes", {}).get("stage07_asof_predictions.csv")
    actual = _sha256(path)
    if expected is None or actual != expected:
        raise ValueError("As-of producer receipt prediction hash does not match the supplied file")
    if producer_receipt.get("cohort") != ASOF_COHORT:
        raise ValueError("As-of producer receipt cohort does not match its predictions")
    return {
        "status": "VERIFIED",
        "path": str(receipt_path),
        "predictions_path": str(path),
        "predictions_sha256": actual,
        "producer_receipt": producer_receipt,
    }

def _evaluate_forecasts(
    opportunities: pd.DataFrame,
    canonical_sports: pd.DataFrame,
    forecast_table: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    protocol: dict[str, Any],
    stage07: Path,
    stage08: Path,
) -> dict[str, Any]:
    """Score frozen forecasts and replay every declared decision scenario."""
    from src.analysis.ev_approach_evaluation import paired_monthly_comparison
    from src.analysis.ev_approach_policies import select_policy_bets, simulate_portfolio

    scoring_universe = pd.concat([opportunities, canonical_sports], ignore_index=True, sort=False)
    probability_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for (cohort, horizon), group in _opportunity_groups(scoring_universe):
        for scope, scope_frame in _scope_views(group):
            for method in sorted(forecast_table.loc[
                forecast_table["cohort"].eq(cohort) & forecast_table["horizon"].eq(horizon), "method"
            ].unique()):
                view = _forecast_view(scope_frame, forecast_table, method)
                status = view.get("status", pd.Series("available", index=view.index)).astype(str).str.lower()
                view.loc[status.ne("available"), "p"] = np.nan
                row = {"cohort": cohort, "horizon": horizon, "scope": scope, "method": method}
                row.update(_score_metrics(view))
                row.update(_ranking_metrics(view))
                score_kinds = view.get("score_kind", pd.Series("probability", index=view.index)).dropna().unique().tolist()
                row["score_kind"] = score_kinds[0] if score_kinds else "unknown"
                row["probability_interpretation"] = _probability_interpretation(row["score_kind"])
                row["probability_certification"] = row["probability_interpretation"]
                if not _is_probability_score_kind(row["score_kind"]):
                    row.update({key: None for key in ("log_loss", "brier", "auc", "accuracy", "ece10", "tail_log_loss_ge_2_5")})
                    row["status"] = "DECISION_ONLY_NOT_A_PROBABILITY"
                elif row["score_kind"] in BOUNDED_RAW_SCORE_KINDS:
                    row["status"] = "AVAILABLE_BOUNDED_RAW_SCORE_NOT_POSTERIOR" if row["n_scored"] else "UNAVAILABLE"
                else:
                    row["status"] = "AVAILABLE" if row["n_scored"] else "UNAVAILABLE"
                probability_rows.append(row)
                for profile in ("best_of", "league"):
                    if profile in view:
                        for profile_value, subset in view.groupby(profile, dropna=False, sort=True):
                            profile_record = {
                                "cohort": cohort, "horizon": horizon, "scope": scope,
                                "method": method, "profile": profile, "profile_value": profile_value,
                            }
                            profile_record.update(_ranking_metrics(subset))
                            profile_record.update(_score_metrics(subset))
                            profile_rows.append(profile_record)
                for column, label in (("best_of", "best_of"), ("league", "league"), ("odds_a", "odds_a"), ("odds_b", "odds_b")):
                    if column in view:
                        if column.startswith("odds_"):
                            band = pd.cut(pd.to_numeric(view[column], errors="coerce"), [1, 1.8, 2.5, 3.5, np.inf], labels=["1.0-1.8", "1.8-2.5", "2.5-3.5", ">3.5"], right=False)
                            bins = view.assign(__bin=band).groupby("__bin", dropna=False, sort=True)
                        else:
                            bins = view.groupby(column, dropna=False, sort=True)
                        for bin_value, subset in bins:
                            coverage_rows.append({
                                "cohort": cohort, "horizon": horizon, "scope": scope,
                                "method": method, "dimension": label, "value": bin_value,
                                "rows": int(len(subset)),
                                "available": int(subset.get("status", pd.Series("available", index=subset.index)).astype(str).str.lower().eq("available").sum()),
                                "unavailable": int(subset.get("status", pd.Series("available", index=subset.index)).astype(str).str.lower().ne("available").sum()),
                            })
    probability_table = pd.DataFrame(probability_rows)
    profile_table = pd.DataFrame(profile_rows)
    coverage_table = pd.DataFrame(coverage_rows)
    probability_table.to_csv(stage08 / "probability_scores.csv", index=False, float_format="%.17g")
    profile_table.to_csv(stage08 / "profile_breakdown.csv", index=False, float_format="%.17g")
    coverage_table.to_csv(stage08 / "coverage.csv", index=False, float_format="%.17g")

    selection_rows: list[dict[str, Any]] = []
    fixed_selection_rows: list[dict[str, Any]] = []
    selected_ledger_rows = _CsvSpool(stage07 / "selected_bets.csv")
    portfolio_rows = _CsvSpool(stage07 / "portfolio_ledger.csv")
    summary_rows: list[dict[str, Any]] = []
    flat_portfolio_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
    p5_cal = threshold_frame.copy()
    bootstrap_draws = int(protocol["statistics"]["bootstrap_draws"])
    seed = int(protocol["seed"])

    for (cohort, horizon), group in _opportunity_groups(opportunities):
        methods = forecast_table.loc[
            forecast_table["cohort"].eq(cohort) & forecast_table["horizon"].eq(horizon), "method"
        ].drop_duplicates().tolist()
        for scope, scope_frame in _scope_views(group):
            method_frames: dict[str, pd.DataFrame] = {
                method: _forecast_view(scope_frame, forecast_table, method)
                for method in methods
            }
            for method, method_frame in method_frames.items():
                score_kind_values = method_frame.get("score_kind", pd.Series("probability", index=method_frame.index)).dropna().unique().tolist()
                score_kind = score_kind_values[0] if score_kind_values else "probability"
                is_probability = score_kind in STANDARD_PROBABILITY_SCORE_KINDS
                is_uncertainty = method.startswith("uncertainty:")
                is_action = method in {"direct_profit", "kelly_utility"}
                if is_action:
                    for tax in TAX_RATES:
                        action_bets = _action_bets(method_frame, tax)
                        action_mask = method_frame.get(
                            "action_side", pd.Series(index=method_frame.index, dtype=object)
                        ).isin(["A", "B"])
                        action_decisions = int(action_mask.sum())
                        action_time_ready = _effective_decision_mask(method_frame)
                        missing_action_times = int((action_mask & ~action_time_ready).sum())
                        action_key = {
                            "cohort": cohort, "horizon": horizon, "scope": scope,
                            "method": method, "policy": "DIRECT_ACTION",
                            "score_kind": score_kind, "tax_rate": tax,
                            "min_ev": "N/A", "qualification": "N/A",
                            "selection_type": "decision_only_action",
                            "selection_status": "DECISION_ONLY_ACTIONS",
                            "probability_scores": "N/A_NOT_A_PROBABILITY",
                        }
                        selection_rows.append({
                            **action_key,
                            "staking": "provided_fraction_and_flat100",
                            "opportunities": int(len(scope_frame)),
                            "action_decisions": action_decisions,
                            "unavailable_decision_times": missing_action_times,
                            "selected_count": int(len(action_bets)),
                            "unavailable_quotes": int(action_decisions - len(action_bets)),
                            "status": (
                                "AVAILABLE" if not action_bets.empty
                                else "UNAVAILABLE_NO_EFFECTIVE_DECISION_TIME"
                                if action_decisions and missing_action_times == action_decisions
                                else "NO_ACTIONS"
                            ),
                            "missing_reason": None if not action_decisions or len(action_bets) == action_decisions else (
                                "missing_effective_decision_time" if missing_action_times == action_decisions
                                else "missing_or_invalid_action_quote_or_time"
                            ),
                        })
                        if not action_bets.empty:
                            selected_ledger_rows.append(action_bets.assign(**action_key))
                        for strategy in ("flat", "provided_fraction"):
                            sim_bets = action_bets
                            if sim_bets.empty:
                                sim_bets = pd.DataFrame(columns=[
                                    "bet_id", "decision_at", "odds",
                                    "action_side", "action_fraction",
                                ])
                            ledger, financial = simulate_portfolio(
                                sim_bets,
                                strategy=strategy,
                                initial_bankroll=float(protocol["decision_evaluation"]["initial_bankroll"]),
                                flat_stake=float(protocol["decision_evaluation"]["flat_stake"]),
                                tax_rate=tax,
                                kelly_fraction=float(protocol["decision_evaluation"]["kelly_fraction"]),
                                max_fraction=float(protocol["decision_evaluation"]["kelly_max_free_capital_fraction"]),
                            )
                            if not ledger.empty:
                                ledger["settled_at"] = _actual_release_values(ledger)
                            key = {
                                **action_key,
                                "staking": "flat100" if strategy == "flat" else "provided_fraction",
                                "sizing_method": "flat" if strategy == "flat" else "fitted_quarter_kelly_fraction",
                            }
                            financial.update(key)
                            summary_rows.append(financial)
                            if not ledger.empty:
                                portfolio_rows.append(ledger.assign(**key))
                    continue
                if not is_probability:
                    is_bounded_raw = score_kind in BOUNDED_RAW_SCORE_KINDS
                    for tax in TAX_RATES:
                        selection_rows.append({
                            "cohort": cohort, "horizon": horizon, "scope": scope,
                            "method": method,
                            "policy": "N/A_NONPOSTERIOR_SCORE" if is_bounded_raw else "N/A_DECISION_SCORE",
                            "score_kind": score_kind,
                            "probability_interpretation": _probability_interpretation(score_kind),
                            "tax_rate": tax, "min_ev": "N/A",
                            "qualification": "N/A", "staking": "N/A",
                            "status": "NOT_A_POSTERIOR_PROBABILITY" if is_bounded_raw else "DECISION_ONLY_NOT_PROBABILITY",
                            "missing_reason": (
                                "bounded_raw_score_is_not_a_posterior_probability"
                                if is_bounded_raw else "raw_score_does_not_define_win_probability_or_kelly_input"
                            ),
                            "selected_count": None,
                        })
                    continue
                status_values = method_frame.get(
                    "status", pd.Series("available", index=method_frame.index)
                ).astype(str).str.lower()
                probability_values = pd.to_numeric(method_frame["p"], errors="coerce")
                probability_mask = (
                    status_values.eq("available") & probability_values.between(0.0, 1.0)
                )
                decision_time_ready = _effective_decision_mask(method_frame)
                unavailable_decision_times = int((probability_mask & ~decision_time_ready).sum())
                prediction_ready = method_frame.loc[
                    probability_mask & decision_time_ready
                ].copy()
                odds_a = pd.to_numeric(prediction_ready["odds_a"], errors="coerce")
                odds_b = pd.to_numeric(prediction_ready["odds_b"], errors="coerce")
                executable_quotes = (
                    np.isfinite(odds_a.to_numpy(dtype=float, na_value=np.nan))
                    & np.isfinite(odds_b.to_numpy(dtype=float, na_value=np.nan))
                    & odds_a.gt(1.0).to_numpy()
                    & odds_b.gt(1.0).to_numpy()
                )
                ready = prediction_ready.loc[executable_quotes].copy()
                policies = UNCERTAINTY_POLICIES if is_uncertainty else ("P0",)
                variant = method.split(":", 1)[1] if is_uncertainty else None
                cal_thresholds = None
                if is_uncertainty and not p5_cal.empty and "variant" in p5_cal:
                    variant_cal = p5_cal.loc[p5_cal["variant"].astype(str).eq(variant)]
                    if not variant_cal.empty:
                        cal_thresholds = variant_cal

                def year_key(value: Any) -> str | None:
                    if pd.isna(value):
                        return None
                    try:
                        return str(int(float(value)))
                    except (TypeError, ValueError):
                        return str(value)

                available_cal_years = set()
                if cal_thresholds is not None and {"model_year", "sigma_z_q75"}.issubset(cal_thresholds.columns):
                    cal_sigma_q75 = pd.to_numeric(cal_thresholds["sigma_z_q75"], errors="coerce")
                    valid_cal = cal_sigma_q75.notna() & np.isfinite(cal_sigma_q75)
                    available_cal_years = {
                        year_key(value)
                        for value in cal_thresholds.loc[valid_cal, "model_year"].dropna().unique()
                    }
                p5_reference_lookup = (
                    {
                        (str(item.variant), year_key(item.model_year)): item.cal_reference_id
                        for item in cal_thresholds.itertuples(index=False)
                    }
                    if cal_thresholds is not None and "cal_reference_id" in cal_thresholds
                    else {}
                )
                if "model_year" in ready and ready["model_year"].notna().any():
                    required_cal_years = {
                        year_key(value) for value in ready["model_year"].dropna().unique()
                    }
                    if ready["model_year"].isna().any():
                        required_cal_years.add(None)
                else:
                    required_cal_years = {None} if not ready.empty else set()
                missing_cal_years = sorted(
                    (
                        year for year in required_cal_years
                        if year is None or year not in available_cal_years
                    ),
                    key=lambda value: "" if value is None else value,
                )
                for policy in policies:
                    for tax in TAX_RATES:
                        for min_ev in EV_THRESHOLDS:
                            for qualification in QUALIFICATIONS:
                                key = (str(cohort), str(horizon), scope, method, policy, tax, min_ev, qualification)
                                if probability_mask.sum() == 0:
                                    selected = pd.DataFrame()
                                    selection_status = "UNAVAILABLE_NO_PREDICTIONS"
                                elif prediction_ready.empty:
                                    selected = pd.DataFrame()
                                    selection_status = "UNAVAILABLE_NO_EFFECTIVE_DECISION_TIME"
                                elif ready.empty:
                                    selected = pd.DataFrame()
                                    selection_status = "UNAVAILABLE_NO_EXECUTABLE_QUOTES"
                                elif policy == "P5" and missing_cal_years and len(missing_cal_years) == len(required_cal_years):
                                    selected = pd.DataFrame()
                                    selection_status = "UNAVAILABLE_MISSING_CAL_THRESHOLD"
                                else:
                                    selected = select_policy_bets(
                                        ready,
                                        probability_col="p",
                                        policy=policy,
                                        tax_rate=tax,
                                        min_ev=min_ev,
                                        qualification=qualification,
                                        cal_thresholds=cal_thresholds,
                                    )
                                    partial_reasons = []
                                    if policy == "P5" and missing_cal_years:
                                        partial_reasons.append("MISSING_CAL_THRESHOLD")
                                    if unavailable_decision_times:
                                        partial_reasons.append("MISSING_EFFECTIVE_DECISION_TIME")
                                    selection_status = (
                                        "PARTIAL_" + "_AND_".join(partial_reasons)
                                        if partial_reasons else "AVAILABLE"
                                    )
                                p5_cal_reference_ids: list[str] = []
                                if policy == "P5" and not selected.empty:
                                    selected_variants = selected["p5_cal_variant"]
                                    selected_years = selected["p5_cal_model_year"]
                                    references = [
                                        p5_reference_lookup.get((str(variant_value), year_key(year_value)))
                                        for variant_value, year_value in zip(selected_variants, selected_years)
                                    ]
                                    if any(reference is None for reference in references):
                                        raise ValueError("Selected P5 bet has no causal CAL reference ID")
                                    selected["p5_cal_reference_id"] = references
                                    p5_cal_reference_ids = sorted(set(references))
                                if "p5_threshold_source_ids" in selected.columns:
                                    selected = selected.drop(columns="p5_threshold_source_ids")
                                select_key = {
                                    "cohort": cohort, "horizon": horizon, "scope": scope,
                                    "method": method, "policy": policy,
                                    "score_kind": score_kind, "tax_rate": tax,
                                    "probability_interpretation": _probability_interpretation(score_kind),
                                    "min_ev": min_ev, "qualification": qualification,
                                    "selection_type": "probability_policy",
                                }
                                selected_count = int(len(selected))
                                selection_rows.append({
                                    **select_key,
                                    "selection_status": selection_status,
                                    "opportunities": int(len(scope_frame)),
                                    "available_predictions": int(len(prediction_ready)),
                                    "unavailable_predictions": int(len(scope_frame) - len(prediction_ready)),
                                    "unavailable_decision_times": unavailable_decision_times,
                                    "missing_cal_threshold_years": missing_cal_years if policy == "P5" else [],
                                    "p5_cal_reference_ids": p5_cal_reference_ids,
                                    "selected_count": selected_count,
                                    "selected_fraction": float(selected_count / len(scope_frame)) if len(scope_frame) else None,
                                    "status": "AVAILABLE" if selection_status == "AVAILABLE" else selection_status,
                                })
                                if not selected.empty:
                                    tagged_selected = selected.assign(**select_key)
                                    tagged_selected["settled_at"] = _actual_release_values(tagged_selected)
                                    selected_ledger_rows.append(tagged_selected)
                                for strategy in STAKING:
                                    sim_bets = selected
                                    if sim_bets.empty:
                                        sim_bets = pd.DataFrame(columns=["bet_id", "decision_at", "odds"])
                                    ledger, financial = simulate_portfolio(
                                        sim_bets,
                                        strategy=strategy,
                                        initial_bankroll=float(protocol["decision_evaluation"]["initial_bankroll"]),
                                        flat_stake=float(protocol["decision_evaluation"]["flat_stake"]),
                                        tax_rate=tax,
                                        kelly_fraction=float(protocol["decision_evaluation"]["kelly_fraction"]),
                                        max_fraction=float(protocol["decision_evaluation"]["kelly_max_free_capital_fraction"]),
                                    )
                                    if not ledger.empty:
                                        ledger["settled_at"] = _actual_release_values(ledger)
                                    staking_name = "flat100" if strategy == "flat" else "quarter_kelly"
                                    financial.update({
                                        **select_key,
                                        "staking": staking_name,
                                        "sizing_method": "flat" if strategy == "flat" else "quarter_kelly",
                                        "selection_status": selection_status,
                                        "probability_coverage": float(len(prediction_ready) / len(scope_frame)) if len(scope_frame) else None,
                                    })
                                    summary_rows.append(financial)
                                    if not ledger.empty:
                                        portfolio_rows.append(ledger.assign(
                                            **select_key, staking=staking_name, sizing_method=staking_name,
                                        ))
                                    if (
                                        strategy == "flat" and qualification == "raw"
                                        and tax == 0.12 and min_ev == 0.05
                                        and (is_uncertainty or method in {"joint_logloss", "market_cal", "market_shape"})
                                    ):
                                        if not ledger.empty:
                                            placed = ledger.get("placed", pd.Series(False, index=ledger.index)).fillna(False)
                                            settled = ledger.get("matured", pd.Series(False, index=ledger.index)).fillna(False)
                                            comparison_columns = [
                                                column for column in (
                                                    "row_id", "date", "stake", "pnl", "won",
                                                    "p", "p_for_ev", "unit_return", "odds", "tax_rate",
                                                ) if column in ledger
                                            ]
                                            flat_portfolio_cache[key] = ledger.loc[
                                                placed & settled, comparison_columns
                                            ].copy()
                                        else:
                                            flat_portfolio_cache[key] = ledger

            # The frozen fixed-selection audit uses native annual A0 decisions once;
            # every forecast changes only its EV/probability residuals, not cashflow.
            native = method_frames.get("native_annual_a0")
            if native is not None and not native.empty:
                native_time_ready = _effective_decision_mask(native)
                native_probability_mask = (
                    native.get("status", pd.Series("available", index=native.index)).astype(str).str.lower().eq("available")
                    & pd.to_numeric(native["p"], errors="coerce").between(0, 1)
                )
                fixed_unavailable_decision_times = int(
                    (native_probability_mask & ~native_time_ready).sum()
                )
                native_candidates = native.loc[
                    native_probability_mask & native_time_ready
                ].copy()
                native_odds_a = pd.to_numeric(native_candidates["odds_a"], errors="coerce")
                native_odds_b = pd.to_numeric(native_candidates["odds_b"], errors="coerce")
                native_quote_mask = (
                    np.isfinite(native_odds_a.to_numpy(dtype=float, na_value=np.nan))
                    & np.isfinite(native_odds_b.to_numpy(dtype=float, na_value=np.nan))
                    & native_odds_a.gt(1.0).to_numpy()
                    & native_odds_b.gt(1.0).to_numpy()
                )
                native_ready = native_candidates.loc[native_quote_mask]
                fixed_bets = select_policy_bets(
                    native_ready, policy="P0", tax_rate=0.12, min_ev=0.05, qualification="raw"
                ) if not native_ready.empty else pd.DataFrame()
                if not fixed_bets.empty:
                    realized_returns = pd.to_numeric(fixed_bets["unit_return"], errors="coerce")
                    realized_pnl = realized_returns * float(protocol["decision_evaluation"]["flat_stake"])
                    observed = pd.to_numeric(fixed_bets["won"], errors="coerce")
                    for method, method_frame in method_frames.items():
                        kinds = method_frame.get("score_kind", pd.Series("probability", index=method_frame.index)).dropna().unique().tolist()
                        if kinds and not _is_probability_score_kind(kinds[0]):
                            continue
                        indexed = method_frame.set_index("row_id")
                        selected = fixed_bets.copy()
                        selected["p_selected"] = [
                            float(indexed.loc[row_id, "p"] if side == "A" else 1.0 - indexed.loc[row_id, "p"])
                            if row_id in indexed.index and pd.notna(indexed.loc[row_id, "p"]) else np.nan
                            for row_id, side in zip(selected.row_id, selected.side)
                        ]
                        selected["p_error"] = selected["p_selected"] - observed
                        selected["predicted_ev"] = selected["p_selected"] * selected["odds"] * (1.0 - selected["tax_rate"]) - 1.0
                        selected["realized_return"] = realized_returns
                        selected["ev_error"] = selected["realized_return"] - selected["predicted_ev"]
                        fixed_selection_rows.append({
                            "cohort": cohort, "horizon": horizon, "scope": scope,
                            "reference_method": "native_annual_a0", "method": method,
                            "reference_bets": int(len(selected)),
                            "matured_reference_bets": int(realized_returns.notna().sum()),
                            "unsettled_reference_bets": int(realized_returns.isna().sum()),
                            "unavailable_decision_times": fixed_unavailable_decision_times,
                            "available_probability_rows": int(selected.p_selected.notna().sum()),
                            "p_error": float(selected.p_error.mean()) if selected.p_error.notna().any() else None,
                            "ev_error": float(selected.ev_error.mean()) if selected.ev_error.notna().any() else None,
                            "realized_pnl": (
                                float(realized_pnl.sum()) if realized_pnl.notna().any() else None
                            ),
                            "fixed_selection_pnl_invariant": True,
                            "same_sides_prices_and_stakes": True,
                        })

    selection_table = pd.DataFrame(selection_rows)
    fixed_table = pd.DataFrame(fixed_selection_rows)
    selection_table.to_csv(stage08 / "selection_metrics.csv", index=False, float_format="%.17g")
    fixed_table.to_csv(stage08 / "fixed_selection.csv", index=False, float_format="%.17g")
    selected_ledger_rows.finish()
    portfolio_rows.finish()

    paired_rows: list[dict[str, Any]] = []
    for (cohort, horizon), group in _opportunity_groups(opportunities):
        for scope, scope_frame in _scope_views(group):
            primary_view = forecast_table.loc[
                forecast_table["cohort"].eq(cohort) & forecast_table["horizon"].eq(horizon)
            ]
            for method_a, method_b in (("joint_logloss", "market_cal"), ("joint_logloss", "market_shape")):
                frame_a = primary_view.loc[primary_view.method.eq(method_a)]
                frame_b = primary_view.loc[primary_view.method.eq(method_b)]
                score_comparison = _score_pair_bootstrap(
                    scope_frame, frame_a, frame_b, bootstrap_draws, seed,
                )
                paired_rows.append({
                    "cohort": cohort, "horizon": horizon, "scope": scope,
                    "comparison_type": "probability_score",
                    "method_a": method_a, "method_b": method_b,
                    "tax_rate": None, "min_ev": None, "qualification": None,
                    **score_comparison,
                })
            for method_a, method_b in (
                ("joint_logloss", "market_cal"), ("joint_logloss", "market_shape"),
            ):
                key_a = (str(cohort), str(horizon), scope, method_a, "P0", 0.12, 0.05, "raw")
                key_b = (str(cohort), str(horizon), scope, method_b, "P0", 0.12, 0.05, "raw")
                if scope_frame.empty:
                    finance_comparison = {
                        "status": "UNAVAILABLE_EMPTY_SCOPE",
                        "months": [], "paired_months": [],
                        "bootstrap": {
                            "draws": bootstrap_draws, "empty_replicates": None,
                            "ci_lower": None, "ci_upper": None,
                            "status": "UNAVAILABLE_EMPTY_SCOPE",
                        },
                    }
                elif key_a not in flat_portfolio_cache or key_b not in flat_portfolio_cache:
                    finance_comparison = {
                        "status": "UNAVAILABLE_MISSING_SELECTION",
                        "months": [], "paired_months": [],
                        "bootstrap": {
                            "draws": bootstrap_draws, "empty_replicates": None,
                            "ci_lower": None, "ci_upper": None,
                            "status": "UNAVAILABLE_MISSING_SELECTION",
                        },
                    }
                else:
                    finance_comparison = paired_monthly_comparison(
                        scope_frame,
                        flat_portfolio_cache[key_a],
                        flat_portfolio_cache[key_b],
                        draws=bootstrap_draws,
                        seed=seed,
                    )
                paired_rows.append({
                    "cohort": cohort, "horizon": horizon, "scope": scope,
                    "comparison_type": "paired_method_finance",
                    "method_a": method_a, "method_b": method_b,
                    "policy_a": "P0", "policy_b": "P0",
                    "tax_rate": 0.12, "min_ev": 0.05, "qualification": "raw",
                    "staking": "flat100",
                    **finance_comparison,
                })
            for variant in ("seed_only", "block_bootstrap"):
                p0_key = (str(cohort), str(horizon), scope, f"uncertainty:{variant}", "P0", 0.12, 0.05, "raw")
                if p0_key not in flat_portfolio_cache:
                    continue
                base_bets = flat_portfolio_cache[p0_key]
                for policy in UNCERTAINTY_POLICIES:
                    policy_key = (str(cohort), str(horizon), scope, f"uncertainty:{variant}", policy, 0.12, 0.05, "raw")
                    if policy_key not in flat_portfolio_cache:
                        continue
                    compare_bets = flat_portfolio_cache[policy_key]
                    if scope_frame.empty:
                        comparison = {
                            "status": "UNAVAILABLE_EMPTY_SCOPE",
                            "months": [], "paired_months": [],
                            "bootstrap": {
                                "draws": bootstrap_draws, "empty_replicates": None,
                                "ci_lower": None, "ci_upper": None,
                                "status": "UNAVAILABLE_EMPTY_SCOPE",
                            },
                        }
                    else:
                        comparison = paired_monthly_comparison(
                            scope_frame, base_bets, compare_bets,
                            draws=bootstrap_draws, seed=seed,
                        )
                    paired_rows.append({
                        "cohort": cohort, "horizon": horizon, "scope": scope,
                        "comparison_type": "uncertainty_policy_finance",
                        "method_a": f"uncertainty:{variant}:P0",
                        "method_b": f"uncertainty:{variant}:{policy}",
                        "tax_rate": 0.12, "min_ev": 0.05, "qualification": "raw",
                        **comparison,
                    })
    for (cohort, horizon), group in _opportunity_groups(canonical_sports):
        primary_view = forecast_table.loc[
            forecast_table["cohort"].eq(cohort) & forecast_table["horizon"].eq(horizon)
        ]
        for method_a, method_b in (
            ("joint_logloss", "market_cal"), ("joint_logloss", "market_shape"),
        ):
            score_comparison = _score_pair_bootstrap(
                group, primary_view.loc[primary_view.method.eq(method_a)],
                primary_view.loc[primary_view.method.eq(method_b)],
                bootstrap_draws, seed,
            )
            paired_rows.append({
                "cohort": cohort, "horizon": horizon, "scope": "all",
                "comparison_type": "probability_score",
                "method_a": method_a, "method_b": method_b,
                "tax_rate": None, "min_ev": None, "qualification": None,
                **score_comparison,
            })
    paired_table = pd.DataFrame(paired_rows)
    paired_table.to_csv(stage08 / "paired_intervals.csv", index=False, float_format="%.17g")

    return {
        "probability_score_rows": int(len(probability_table)),
        "selection_rows": int(len(selection_table)),
        "fixed_selection_rows": int(len(fixed_table)),
        "portfolio_scenarios": int(len(summary_rows)),
        "paired_comparisons": int(len(paired_rows)),
        "results": summary_rows,
    }


def run_approach_comparison(
    protocol_path: str | Path,
    output_dir: str | Path,
    research_root: str | Path | None = None,
    asof_predictions: str | Path | None = None,
    native_predictions: str | Path | None = None,
) -> Path:
    """Prepare data, replay the frozen matrix, and write stage 06-08 artifacts."""
    from src.analysis.ev_approach_data import prepare_datasets, validate_opportunities
    from src.analysis.ev_approach_evaluation import chronological_head_predictions
    from src.analysis.ev_approach_policies import select_policy_bets
    from src.analysis.ev_approach_risk import METHODS as RISK_METHODS, chronological_risk_predictions
    from src.models.ev_approach_models import APPROACHES, fit_head

    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = _verify_protocol(protocol_file)
    root = _resolve_research_root(research_root)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"--output-dir must not already exist: {destination}")
    destination.mkdir(parents=True)
    stage06 = destination / "stage06_models"
    stage07 = destination / "stage07_predictions_ledgers"
    stage08 = destination / "stage08_report"
    stage06.mkdir()
    stage07.mkdir()
    stage08.mkdir()

    inputs = protocol.get("inputs", {})
    panel_path = PROJECT_ROOT / inputs["production_panel"] if inputs.get("production_panel") else None
    quote_path = PROJECT_ROOT / inputs["production_quotes"] if inputs.get("production_quotes") else None
    uncertainty_dir = PROJECT_ROOT / inputs["uncertainty_predictions"] if inputs.get("uncertainty_predictions") else None
    datasets = prepare_datasets(
        research_root=root,
        panel_path=panel_path,
        quote_path=quote_path,
        uncertainty_dir=uncertainty_dir,
    )
    opportunities = validate_opportunities(datasets["opportunities"])
    required_opportunity = {
        "row_id", "golgg_match_id", "cohort", "horizon", "date", "decision_at",
        "label_available_at", "y", "p_s", "q", "odds_a", "odds_b", "best_of",
        "league", "bookmaker_a", "bookmaker_b", "quote_id_a", "quote_id_b",
    }
    missing = required_opportunity - set(opportunities.columns)
    if missing:
        raise ValueError(f"Prepared opportunities are missing runner columns: {sorted(missing)}")
    canonical_sports = _canonical_sports_frame(datasets["canonical"])

    asof_records: list[dict[str, Any]] = []
    asof_artifact_receipt: dict[str, Any] = {
        "status": "NOT_SUPPLIED", "path": None, "attached": 0,
    }
    asof_producer_receipt: dict[str, Any] = {"status": "NOT_SUPPLIED"}
    if asof_predictions is not None:
        asof_path = Path(asof_predictions).expanduser().resolve()
        asof_source = _read_table(asof_path)
        asof_cohort, asof_source, asof_artifact_receipt = _asof_cohort_inputs(
            opportunities, asof_source,
        )
        asof_cohort = validate_opportunities(asof_cohort)
        asof_records, asof_forecast_receipt = _asof_prediction_records(
            asof_cohort, asof_source,
        )
        asof_producer_receipt = _asof_source_receipt(asof_path)
        asof_artifact_receipt.update(asof_forecast_receipt)
        asof_artifact_receipt.update({
            "path": str(asof_path),
            "sha256": _sha256(asof_path),
            "producer_receipt_status": asof_producer_receipt["status"],
        })
        opportunities = pd.concat(
            [opportunities, asof_cohort], ignore_index=True, sort=False,
        )
    _write_json(stage07 / "asof_source_receipt.json", {
        "attachment": asof_artifact_receipt,
        "producer_receipt": asof_producer_receipt,
    })

    stage07.joinpath("prepared_opportunities.csv").write_text(
        opportunities.to_csv(index=False, float_format="%.17g")
    )
    stage07.joinpath("canonical_sports_inputs.csv").write_text(
        canonical_sports.to_csv(index=False, float_format="%.17g")
    )

    min_train = int(protocol["timing"]["head_min_train"])
    lambdas = tuple(float(value) for value in protocol["timing"]["lambda_grid"])
    heads, fit_receipts = _chronological_head_matrix(
        opportunities, methods=list(APPROACHES), lambdas=lambdas, min_train=min_train,
    )
    canonical_heads, canonical_fit_receipts = _chronological_head_matrix(
        canonical_sports, methods=list(APPROACHES), lambdas=lambdas, min_train=min_train,
    )
    fit_receipts.extend(canonical_fit_receipts)
    _write_json(stage06 / "fit_receipts.json", {
        "lambda_grid": list(lambdas), "min_train": min_train,
        "methods": list(APPROACHES), "receipts": fit_receipts,
    })
    predictions: list[dict[str, Any]] = []
    head_rows = int(len(heads))
    canonical_head_rows = int(len(canonical_heads))
    del fit_receipts, canonical_fit_receipts
    opportunity_index = opportunities.set_index(
        ["cohort", "horizon", "row_id"], drop=False,
    )
    for _, row in opportunities.iterrows():
        p_s = row.get("p_s")
        q = row.get("q")
        feature_violation = str(row.get("feature_time_violation", "")).strip().lower() in {"true", "1", "1.0"}
        native_reason = (
            "feature_time_violation"
            if str(row.get("cohort")) == "production" and feature_violation else None
        )
        base_rows = [
            ("market_raw", q, None),
            ("fixed_logit_50_50", _fixed_fifty(p_s, q), None),
        ]
        if str(row.get("cohort")) != ASOF_COHORT:
            base_rows.insert(0, ("native_annual_a0", np.nan if native_reason else p_s, native_reason))
        for method, value, reason in base_rows:
            source_provenance = {}
            if method == "native_annual_a0":
                source_provenance = {
                    "source_is_oof": row.get("is_oof"),
                    "source_recipe_status": row.get("recipe_status"),
                    "source_prediction_status": row.get("source_prediction_status"),
                    "source_prediction_role": row.get("prediction_role"),
                    "source_fit_origin": row.get("fit_origin"),
                    "source_prediction_at": row.get("prediction_at"),
                }
            predictions.append(_probability_record(
                row, method, value,
                "unavailable" if reason else "available",
                reason, availability_source="frozen_opportunity_input",
                **source_provenance,
            ))
    if not heads.empty:
        for _, row in heads.iterrows():
            identity = opportunity_index.loc[
                (row["cohort"], row["horizon"], row["row_id"])
            ]
            record = _probability_record(
                identity, str(row["method"]), row.get("p"),
                str(row.get("status", "unavailable")).lower(), row.get("missing_reason"),
                str(row.get("score_kind", "probability")),
                raw_score=row.get("raw_score"),
                action_side=row.get("action_side"), action_fraction=row.get("action_fraction"),
                head_fit_origin=row.get("fit_origin"),
            )
            predictions.append(record)
    for _, row in canonical_sports.iterrows():
        for method, value, score_kind, status, reason in (
            ("native_annual_a0", row.get("p_s"), "probability", "available", None),
            ("market_raw", np.nan, "probability", "unavailable", "CANONICAL_SPORTS_HAS_NO_WITHIN_BOOK_Q"),
            ("fixed_logit_50_50", np.nan, "probability", "unavailable", "CANONICAL_SPORTS_HAS_NO_WITHIN_BOOK_Q"),
        ):
            predictions.append(_probability_record(
                row, method, value, status, reason, score_kind,
                availability_source="canonical_sports_input",
                source_prediction_at=row.get("prediction_at"),
                source_fit_origin=row.get("fit_origin"),
            ))
    canonical_index = canonical_sports.set_index("row_id", drop=False)
    if not canonical_heads.empty:
        for _, row in canonical_heads.iterrows():
            identity = canonical_index.loc[row["row_id"]]
            predictions.append(_probability_record(
                identity, str(row["method"]), row.get("p"),
                str(row.get("status", "unavailable")).lower(), row.get("missing_reason"),
                str(row.get("score_kind", "probability")),
                raw_score=row.get("raw_score"), head_fit_origin=row.get("fit_origin"),
                action_side=row.get("action_side"), action_fraction=row.get("action_fraction"),
            ))

    risk_receipts: list[dict[str, Any]] = []
    for (cohort, horizon), group in opportunities.groupby(
        ["cohort", "horizon"], sort=True, dropna=False,
    ):
        source_index = group.set_index("row_id", drop=False)
        for method in RISK_METHODS:
            risk_frame, receipt = chronological_risk_predictions(
                group, method, alpha=float(protocol["risk_methods"]["alpha"]),
            )
            receipt.update({"cohort": cohort, "horizon": horizon})
            risk_receipts.append(receipt)
            if risk_frame["row_id"].duplicated().any():
                raise ValueError(f"Risk method {method} returned duplicate row IDs")
            for _, risk_row in risk_frame.iterrows():
                source_row = source_index.loc[risk_row["row_id"]]
                record = _probability_record(
                    source_row, f"risk:{method}", risk_row.get("p"),
                    str(risk_row.get("status", "unavailable")).lower(),
                    risk_row.get("missing_reason"), "probability",
                    risk_method=method,
                    effective_decision_at=risk_row.get("effective_decision_at"),
                    feedback_count=risk_row.get("feedback_count"),
                    max_feedback_release_at=risk_row.get("max_feedback_release_at"),
                    feedback_ids_sha256=risk_row.get("feedback_ids_sha256"),
                    allow_a=bool(risk_row.get("allow_a", False)),
                    allow_b=bool(risk_row.get("allow_b", False)),
                    p_low_a=risk_row.get("p_low_a"), p_low_b=risk_row.get("p_low_b"),
                    risk_diagnostics={
                        key: risk_row.get(key) for key in risk_row.index
                        if key not in {"row_id", "method", "p", "status", "missing_reason"}
                    },
                )
                predictions.append(record)
    for (cohort, horizon), group in canonical_sports.groupby(
        ["cohort", "horizon"], sort=True, dropna=False,
    ):
        source_index = group.set_index("row_id", drop=False)
        for method in RISK_METHODS:
            risk_frame, receipt = chronological_risk_predictions(
                group, method, alpha=float(protocol["risk_methods"]["alpha"]),
            )
            receipt.update({"cohort": cohort, "horizon": horizon})
            risk_receipts.append(receipt)
            for _, risk_row in risk_frame.iterrows():
                source_row = source_index.loc[risk_row["row_id"]]
                predictions.append(_probability_record(
                    source_row, f"risk:{method}", risk_row.get("p"),
                    str(risk_row.get("status", "unavailable")).lower(),
                    risk_row.get("missing_reason"), "probability",
                    risk_method=method, allow_a=bool(risk_row.get("allow_a", False)),
                    effective_decision_at=risk_row.get("effective_decision_at"),
                    feedback_count=risk_row.get("feedback_count"),
                    max_feedback_release_at=risk_row.get("max_feedback_release_at"),
                    feedback_ids_sha256=risk_row.get("feedback_ids_sha256"),
                    allow_b=bool(risk_row.get("allow_b", False)),
                    p_low_a=risk_row.get("p_low_a"), p_low_b=risk_row.get("p_low_b"),
                    risk_diagnostics={
                        key: risk_row.get(key) for key in risk_row.index
                        if key not in {"row_id", "method", "p", "status", "missing_reason"}
                    },
                ))

    uncertainty = datasets.get("uncertainty", pd.DataFrame())
    annual_uncertainty_opportunities = opportunities.loc[
        ~opportunities["cohort"].astype(str).isin({"production", ASOF_COHORT})
    ]
    uncertainty_records = _match_uncertainty(annual_uncertainty_opportunities, uncertainty)
    uncertainty_records.extend(_match_uncertainty(canonical_sports, uncertainty))
    predictions.extend(uncertainty_records)
    predictions.extend(asof_records)

    external_receipts = {
        "asof": {
            **asof_artifact_receipt,
            "producer_receipt_path": asof_producer_receipt.get("path"),
        },
    }
    archival_opportunities = opportunities.loc[
        opportunities["cohort"].astype(str).eq("archival_market")
    ]
    native_targets = pd.concat(
        [canonical_sports, archival_opportunities], ignore_index=True, sort=False,
    )
    native_records, native_receipt = _external_prediction_rows(
        native_targets,
        Path(native_predictions).expanduser().resolve() if native_predictions else None,
        "native_target",
    )
    external_receipts["native_target"] = native_receipt
    predictions.extend(native_records)
    _write_json(stage07 / "external_prediction_receipts.json", external_receipts)

    forecast_table = pd.DataFrame(predictions)
    del predictions, uncertainty_records, native_records, asof_records
    del heads, canonical_heads
    if forecast_table.empty:
        raise ValueError("No forecasts were prepared from frozen inputs")
    forecast_identity = ["cohort", "horizon", "row_id", "method"]
    if forecast_table.duplicated(forecast_identity).any():
        duplicates = forecast_table.loc[forecast_table.duplicated(forecast_identity, keep=False), forecast_identity]
        raise ValueError(f"Duplicate forecast identities: {duplicates.head().to_dict('records')}")
    forecast_table.to_csv(stage07 / "predictions.csv", index=False, float_format="%.17g")
    _write_json(stage08 / "risk_receipts.json", risk_receipts)
    forecast_table = forecast_table.loc[:, [
        column for column in EVALUATION_FORECAST_COLUMNS if column in forecast_table
    ]].copy()

    sigma_cal = datasets.get("sigma_cal", pd.DataFrame())
    thresholds = _p5_threshold_receipts(sigma_cal)
    threshold_columns = [
        "cal_reference_id", "variant", "model_year", "partition",
        "sigma_z_q75", "cal_rows", "cal_row_ids", "prediction_role",
        "source", "is_whole_recipe_oof", "max_calibration_label_release_at",
        "status",
    ]
    threshold_frame = pd.DataFrame(thresholds, columns=threshold_columns)
    threshold_frame.to_csv(stage08 / "thresholds.csv", index=False, float_format="%.17g")
    evaluation = _evaluate_forecasts(
        opportunities, canonical_sports, forecast_table, threshold_frame,
        protocol, stage07, stage08,
    )


    _write_json(stage08 / "summary.json", {
        "run_id": protocol["run_id"],
        "status": "RETROSPECTIVE_DEVELOPMENT_EXPERIMENT",
        "evidence_limit": protocol.get("evidence_limit"),
        "decision": "No confirmatory winner, financial certification, or deployment claim is made.",
        "opportunity_rows": int(len(opportunities)),
        "cohort_horizon_groups": int(opportunities.groupby(["cohort", "horizon"], dropna=False).ngroups),
        "head_rows": head_rows,
        "canonical_sports_rows": int(len(canonical_sports)),
        "canonical_sports_head_rows": canonical_head_rows,
        "scoring_rows": int(len(opportunities) + len(canonical_sports)),
        **evaluation,
        "external_predictions": external_receipts,
        "missingness": "Unavailable forecasts remain explicit and are never scored as zero loss.",
        "primary_frozen_contrasts": ["joint_logloss - market_cal", "joint_logloss - market_shape"],
    })
    receipt = datasets["receipt"]
    input_hashes = receipt["input_hashes"]
    code_files = [
        Path(__file__).resolve(),
        Path(chronological_head_predictions.__code__.co_filename).resolve(),
        Path(select_policy_bets.__code__.co_filename).resolve(),
        Path(chronological_risk_predictions.__code__.co_filename).resolve(),
        Path(prepare_datasets.__code__.co_filename).resolve(),
        Path(fit_head.__code__.co_filename).resolve(),
        protocol_file,
    ]
    code_hashes = {str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path): _sha256(path) for path in code_files if path.is_file()}
    outputs = [path for path in destination.rglob("*") if path.is_file()]
    manifest = {
        "run_id": protocol["run_id"], "protocol_path": str(protocol_file),
        "protocol_sha256": _sha256(protocol_file), "research_root": str(root),
        "input_receipt": receipt, "input_hashes": input_hashes,
        "code_sha256": code_hashes,
        "external_predictions": external_receipts,
        "artifacts": {
            str(path.relative_to(destination)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        },
        "matrix": {
            "heads": list(APPROACHES), "risk_methods": list(RISK_METHODS),
            "uncertainty_variants": ["seed_only", "block_bootstrap"],
            "uncertainty_policies": list(UNCERTAINTY_POLICIES),
            "tax_rates": list(TAX_RATES), "min_ev": list(EV_THRESHOLDS),
            "qualifications": list(QUALIFICATIONS), "staking": list(STAKING),
            "paired_draws": int(protocol["statistics"]["bootstrap_draws"]),
        },
        "limitations": [
            "Historical development evidence; every year is development-seen and this is not an untouched confirmatory holdout.",
            "Historical scenario timestamps remain assumptions, not observed creation or publication times.",
            "The frozen sigma CAL threshold references are uncertainty CAL only, not whole-recipe OOF calibration evidence.",
            "Risk-method guarantees are not proven; diagnostics are limited to this finite auditor scope.",
            "Historical observations are non-exchangeable; monthly block resampling does not support prospective or causal claims.",
            "Observed CLV is same-book only and is not imputed from another bookmaker.",
            "Shuffled monthly bootstrap blocks are not valid chronological Kelly bankroll paths.",
        ],
    }
    _write_json(stage08 / "manifest.json", manifest)
    return destination


def approach_comparison_main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["c0-ev-loss-study"]:
        from src.analysis.c0_ev_loss_study import main
        main(arguments[1:])
        return
    if arguments[:1] == ["c0-input-aligned-study"]:
        from src.analysis.c0_input_aligned_study import main
        main(arguments[1:])
        return
    if arguments[:1] == ["c0-historical-development"]:
        from src.analysis.c0_betting_development import main
        main(arguments[1:])
        return
    parser = argparse.ArgumentParser(
        prog="run_unified_betting_evaluation.py approach-comparison",
        description="Run the frozen all-approaches historical betting evaluation.",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--asof-predictions", type=Path)
    parser.add_argument("--native-predictions", type=Path)
    args = parser.parse_args(argv)
    result = run_approach_comparison(
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        research_root=args.research_root,
        asof_predictions=args.asof_predictions,
        native_predictions=args.native_predictions,
    )
    print(f"Wrote frozen all-approaches evaluation to {result}")
