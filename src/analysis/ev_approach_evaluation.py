"""Chronological low-capacity head replay and paired opportunity-month evaluation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _utc_series(values: pd.Series) -> pd.Series:
    """Normalize known and missing timestamps without inventing scenario times."""
    return pd.to_datetime(values, utc=True, errors="coerce")


def _effective(frame: pd.DataFrame, column: str) -> pd.Series:
    from src.analysis.ev_approach_data import effective_times

    return effective_times(frame, column)


def _time_basis(frame: pd.DataFrame, actual: str, scenario: str) -> dict[str, int]:
    if frame.empty:
        return {"actual": 0, "scenario": 0, "unavailable": 0}
    actual_known = (
        frame[actual].notna() if actual in frame
        else pd.Series(False, index=frame.index)
    )
    scenario_known = (
        frame[scenario].notna() if scenario in frame
        else pd.Series(False, index=frame.index)
    )
    scenario_used = ~actual_known & scenario_known
    return {
        "actual": int(actual_known.sum()),
        "scenario": int(scenario_used.sum()),
        "unavailable": int((~actual_known & ~scenario_known).sum()),
    }


def _status_id_map(frame: pd.DataFrame, status_column: str) -> dict[str, list[str]]:
    if status_column not in frame:
        return {}
    result = {}
    for status, rows in frame.groupby(status_column, dropna=False, sort=True):
        name = "UNSPECIFIED" if pd.isna(status) else str(status)
        result[name] = rows["row_id"].astype(str).tolist()
    return result


def _month_starts(values: pd.Series) -> pd.Series:
    return _utc_series(values).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")


def _close_target_mask(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    if "close_available_at" not in frame or "close_q" not in frame:
        return pd.Series(False, index=frame.index)
    released = _effective(frame, "close_available_at")
    target = pd.to_numeric(frame["close_q"], errors="coerce")
    available = target.notna() & np.isfinite(target) & target.gt(0.0) & target.lt(1.0)
    return released.notna() & released.lt(origin) & available




def _complete_input_rows(frame: pd.DataFrame, columns: set[str] | frozenset[str]) -> tuple[pd.Series, list[str]]:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        return pd.Series(False, index=frame.index), missing
    valid = pd.Series(True, index=frame.index)
    for column in sorted(columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        current = values.notna() & np.isfinite(values)
        if column in {"p_s", "q"}:
            current &= values.ge(0.0) & values.le(1.0)
        elif column == "close_q":
            current &= values.gt(0.0) & values.lt(1.0)
        elif column.startswith("odds_"):
            current &= values.gt(1.0)
        valid &= current
    return valid, []


def _native_oof_rows_before(
    frame: pd.DataFrame, origin: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    from src.analysis.ev_approach_data import calibration_rows_before, effective_times

    selected = calibration_rows_before(frame, origin, require_oof=True)
    keep = frame.index.isin(selected.index)
    excluded = frame.loc[~keep]
    if excluded.empty:
        return selected, {}

    reasons = pd.Series("", index=excluded.index, dtype="object")
    complete = excluded["recipe_status"].astype("string").eq("COMPLETE-RECIPE")
    reasons.loc[~complete] = "INCOMPLETE_RECIPE"
    oof = excluded["is_oof"].map(
        lambda value: (
            bool(value) if isinstance(value, (bool, np.bool_))
            else value == 1 if isinstance(value, (int, np.integer, float, np.floating))
            else str(value).strip().lower() in {"true", "1", "1.0"}
        )
    )
    reasons.loc[complete & ~oof] = "NOT_OOF"

    scenario = excluded.get(
        "information_status", pd.Series("", index=excluded.index)
    ).astype("string").eq("historical_scenario")
    assumption_column = (
        "scenario_timestamp_assumption"
        if "scenario_timestamp_assumption" in excluded else "timestamp_assumption"
    )
    assumption = (
        excluded[assumption_column].astype("string").str.strip().fillna("").ne("")
    )
    remaining = reasons.eq("")
    unattested = scenario & ~assumption
    reasons.loc[remaining & unattested] = "UNATTESTED_SCENARIO_TIME"
    remaining &= ~unattested
    origin_utc = pd.Timestamp(origin)
    if origin_utc.tzinfo is None:
        origin_utc = origin_utc.tz_localize("UTC")
    else:
        origin_utc = origin_utc.tz_convert("UTC")
    for column, reason in (
        ("prediction_at", "PREDICTION_TIME_NOT_STRICTLY_BEFORE_ORIGIN"),
        ("fit_origin", "FIT_ORIGIN_NOT_STRICTLY_BEFORE_ORIGIN"),
        ("label_available_at", "LABEL_NOT_STRICTLY_RELEASED_BEFORE_ORIGIN"),
    ):
        values = effective_times(excluded, column)
        invalid = values.isna() | values.ge(origin_utc)
        reasons.loc[remaining & invalid] = reason
        remaining &= ~invalid
    reasons.loc[remaining] = "SOURCE_PROVENANCE_INELIGIBLE"
    return selected, {
        reason: excluded.loc[reasons.eq(reason), "row_id"].astype(str).tolist()
        for reason in sorted(reasons.unique()) if reason
    }


def _reason_counts(reason_ids: dict[str, list[str]]) -> dict[str, int]:
    return {reason: len(ids) for reason, ids in sorted(reason_ids.items())}


def _predictable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    hidden = {
        "y", "label_available_at", "scenario_label_available_at", "close_q",
        "close_odds_a", "close_odds_b", "close_available_at",
        "scenario_close_q", "scenario_close_odds_a", "scenario_close_odds_b",
        "scenario_close_available_at", "close_at", "scenario_close_at",
    }
    return frame.drop(columns=[name for name in hidden if name in frame], errors="ignore")


def _empty_prediction(
    row: pd.Series,
    method: str,
    origin: pd.Timestamp,
    reason: str,
    score_kind: str | None = None,
) -> dict[str, Any]:
    if score_kind is None:
        score_kind = {
            "ranking": "ranking_score",
            "direct_profit": "decision_score",
            "kelly_utility": "decision_score",
            "class_weighted": "tilted_probability_score",
            "focal": "focal_score",
            "label_smoothing": "smoothed_probability_score",
        }.get(method, "probability")
    return {
        "cohort": row["cohort"], "horizon": row["horizon"],
        "row_id": row["row_id"], "method": method, "fit_origin": origin,
        "p": np.nan, "raw_score": np.nan, "score_kind": score_kind,
        "status": "UNAVAILABLE", "missing_reason": reason,
    }


def chronological_head_predictions(
    frame: pd.DataFrame,
    methods: Sequence[str] | None = None,
    lambdas: Sequence[float] = (0.01, 0.1, 1.0),
    min_train: int = 100,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Replay candidate heads independently by cohort and horizon at monthly origins."""
    from src.models.ev_approach_models import APPROACHES, APPROACH_INPUTS, fit_head

    if min_train < 1:
        raise ValueError("min_train must be positive")
    if not lambdas or any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("lambdas must be non-empty, finite, positive values")
    selected_methods = list(APPROACHES if methods is None else methods)
    unknown = sorted(set(selected_methods) - set(APPROACHES))
    if unknown:
        raise ValueError(f"Unknown approach methods: {unknown}")
    required = {"row_id", "decision_at", "label_available_at", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Replay input is missing required columns: {sorted(missing)}")
    frame = frame.copy().reset_index(drop=True)
    if "cohort" not in frame:
        frame["cohort"] = "all"
    if "horizon" not in frame:
        frame["horizon"] = "all"
    if frame["row_id"].isna().any():
        raise ValueError("Replay row_id values must not be missing")
    if frame.duplicated(["cohort", "horizon", "row_id"]).any():
        raise ValueError("Replay row_id values must be unique within each cohort/horizon")

    decision_at = _effective(frame, "decision_at")
    label_at = _effective(frame, "label_available_at")
    if decision_at.isna().any():
        raise ValueError("Replay rows require actual or authorized scenario decision times")
    base = frame.copy()
    base["__decision_at"] = decision_at
    base["__label_at"] = label_at
    base["__y"] = pd.to_numeric(frame["y"], errors="coerce")
    base["__origin"] = _month_starts(decision_at)

    def fitting_frame(part: pd.DataFrame, method: str) -> pd.DataFrame:
        result = part.drop(columns=["__decision_at", "__label_at", "__y", "__origin"])
        if method not in {"close_target", "clv_target"}:
            result = result.drop(columns=[
                name for name in (
                    "close_q", "close_odds_a", "close_odds_b", "close_available_at",
                    "scenario_close_q", "scenario_close_odds_a", "scenario_close_odds_b",
                    "scenario_close_available_at",
                ) if name in result
            ])
        return result

    predictions: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for (cohort, horizon), group in base.groupby(
        ["cohort", "horizon"], sort=True, dropna=False
    ):
        group = group.sort_values(["__decision_at", "row_id"], kind="mergesort")
        native_eligibility: dict[pd.Timestamp, tuple[pd.Index, pd.Series]] = {}
        fitted_folds: dict[tuple[str, float, pd.Timestamp], Any] = {}

        def native_rows_before(part: pd.DataFrame, cutoff: pd.Timestamp):
            if cutoff not in native_eligibility:
                eligible, reason_ids = _native_oof_rows_before(group, cutoff)
                reason_by_id = {
                    row_id: reason for reason, ids in reason_ids.items() for row_id in ids
                }
                native_eligibility[cutoff] = (
                    eligible.index, group["row_id"].astype(str).map(reason_by_id),
                )
            eligible_index, reasons = native_eligibility[cutoff]
            keep = part.index.isin(eligible_index)
            excluded = part.loc[~keep]
            exclusions = {
                str(reason): rows["row_id"].astype(str).tolist()
                for reason, rows in excluded.groupby(reasons.loc[excluded.index], sort=True)
            }
            return part.loc[keep], exclusions

        for origin, current in group.groupby("__origin", sort=True, dropna=False):
            origin = pd.Timestamp(origin)
            current_index = current.index
            for method in selected_methods:
                outer_mask = (
                    group["__decision_at"].lt(origin)
                    & group["__label_at"].notna()
                    & group["__label_at"].lt(origin)
                    & group["__y"].isin([0, 1])
                )
                eligible_train = group.loc[outer_mask].copy()
                fit_inputs = set(APPROACH_INPUTS[method]).difference({"y"})
                prediction_inputs = fit_inputs.difference({"close_q", "odds_a", "odds_b"})
                if method in {"direct_profit", "kelly_utility"}:
                    prediction_inputs.update({"odds_a", "odds_b"})
                fit_input_mask, missing_fit_columns = _complete_input_rows(
                    eligible_train, fit_inputs,
                )
                input_excluded_train = eligible_train.loc[~fit_input_mask].copy()
                outer_train = eligible_train.loc[fit_input_mask].copy()
                native_oof_required = "p_s" in fit_inputs
                native_oof_exclusion_ids: dict[str, list[str]] = {}
                if native_oof_required:
                    outer_train, native_oof_exclusion_ids = native_rows_before(
                        outer_train, origin,
                    )
                close_ineligible_train_ids: list[str] = []
                if method in {"close_target", "clv_target"}:
                    close_input_mask = _close_target_mask(outer_train, origin)
                    close_ineligible_train_ids = outer_train.loc[
                        ~close_input_mask, "row_id",
                    ].astype(str).tolist()
                    outer_train = outer_train.loc[close_input_mask].copy()
                common_receipt: dict[str, Any] = {
                    "cohort": cohort,
                    "horizon": horizon,
                    "fit_origin": origin,
                    "method": method,
                    "required_fit_inputs": sorted(fit_inputs),
                    "required_prediction_inputs": sorted(prediction_inputs),
                    "labelled_train_candidates": int(len(eligible_train)),
                    "input_excluded_train_ids": input_excluded_train["row_id"].astype(str).tolist(),
                    "missing_fit_input_columns": missing_fit_columns,
                    "native_oof_required": native_oof_required,
                    "native_oof_excluded_train_ids": sorted(
                        row_id for ids in native_oof_exclusion_ids.values() for row_id in ids
                    ),
                    "native_oof_excluded_train_count": sum(
                        len(ids) for ids in native_oof_exclusion_ids.values()
                    ),
                    "native_oof_exclusion_reason_ids": native_oof_exclusion_ids,
                    "native_oof_exclusion_reason_counts": _reason_counts(native_oof_exclusion_ids),
                    "close_target_ineligible_train_ids": close_ineligible_train_ids,
                    "outer_train_candidate_ids": outer_train["row_id"].astype(str).tolist(),
                    "train_ids": [],
                    "calibration_ids": [],
                    "inner_validation_ids": [],
                    "calibration_role": "NONE",
                    "inner_validation_role": "EXPANDING_HYPERPARAMETER_SELECTION; MAY_BE_INCLUDED_IN_FINAL_REFIT",
                    "target_kind": (
                        "closing_probability" if method == "close_target"
                        else "future_price_clv" if method == "clv_target"
                        else "series_winner"
                    ),
                    "max_outer_train_candidate_label_available_at": (
                        outer_train["__label_at"].max() if not outer_train.empty else pd.NaT
                    ),
                    "max_train_label_available_at": pd.NaT,
                    "max_train_close_available_at": pd.NaT,
                    "max_calibration_label_available_at": pd.NaT,
                    "max_inner_validation_label_available_at": pd.NaT,
                    "warmup_required": int(min_train),
                    "warmup_covered": len(outer_train) >= min_train,
                    "selected_lambda": None,
                    "validation_losses": {},
                    "status": "UNAVAILABLE",
                    "missing_reason": None,
                    "source_recipe_provenance": (
                        "NO_ELIGIBLE_COMPLETE-RECIPE-OOF" if native_oof_required
                        else "NOT_REQUIRED_MARKET_ONLY"
                    ),
                }
                time_observations = pd.concat([
                    outer_train.get("scenario_timestamp_assumption", pd.Series(dtype=object)),
                    current.get("scenario_timestamp_assumption", pd.Series(dtype=object)),
                ]).dropna().astype(str).unique().tolist()
                common_receipt.update({
                    "fit_origin_semantics": "MONTH_START_OF_EFFECTIVE_DECISION_TIME",
                    "fit_origin_time_basis": _time_basis(
                        current, "decision_at", "scenario_decision_at",
                    ),
                    "training_decision_time_basis": _time_basis(
                        outer_train, "decision_at", "scenario_decision_at",
                    ),
                    "training_label_time_basis": _time_basis(
                        outer_train, "label_available_at", "scenario_label_available_at",
                    ),
                    "max_train_actual_label_available_at": pd.NaT,
                    "max_train_scenario_label_available_at": pd.NaT,
                    "scenario_timestamp_assumptions": time_observations,
                    "source_chronology_status_train": _status_id_map(
                        outer_train, "source_chronology_status",
                    ),
                    "source_chronology_status_prediction": _status_id_map(
                        current, "source_chronology_status",
                    ),
                })
                if native_oof_required:
                    common_receipt["source_recipe_provenance"] = (
                        "COMPLETE-RECIPE-OOF" if not outer_train.empty
                        else "NO_ELIGIBLE_COMPLETE-RECIPE-OOF"
                    )
                    common_receipt["source_oof_ids"] = outer_train["row_id"].astype(str).tolist()

                prediction_input_mask, missing_prediction_columns = _complete_input_rows(
                    current, prediction_inputs,
                )
                method_predictions = {}
                for index, row in current.iterrows():
                    if prediction_input_mask.loc[index]:
                        method_predictions[int(index)] = _empty_prediction(
                            row, method, origin, "NOT_FIT",
                        )
                    else:
                        reason = (
                            "MISSING_REQUIRED_PREDICTION_COLUMNS:"
                            + ",".join(missing_prediction_columns)
                            if missing_prediction_columns else
                            "MISSING_REQUIRED_PREDICTION_INPUT_VALUES"
                        )
                        method_predictions[int(index)] = _empty_prediction(
                            row, method, origin, reason,
                        )
                common_receipt["required_prediction_inputs"] = sorted(prediction_inputs)
                common_receipt["current_prediction_input_rows"] = int(prediction_input_mask.sum())
                common_receipt["current_prediction_input_missing_ids"] = current.loc[
                    ~prediction_input_mask, "row_id",
                ].astype(str).tolist()
                common_receipt["missing_prediction_columns"] = missing_prediction_columns
                if len(outer_train) < min_train:
                    if method in {"close_target", "clv_target"} and outer_train.empty:
                        common_receipt["missing_reason"] = "MISSING_CLOSE"
                    elif missing_fit_columns:
                        common_receipt["missing_reason"] = "MISSING_REQUIRED_INPUT_COLUMNS"
                    elif native_oof_exclusion_ids:
                        common_receipt["missing_reason"] = "SOURCE_OOF_COVERAGE_WARMUP"
                    elif len(eligible_train) >= min_train and len(input_excluded_train) > 0:
                        common_receipt["missing_reason"] = "INPUT_COVERAGE_WARMUP"
                    else:
                        common_receipt["missing_reason"] = "WARMUP"
                else:
                    inner_folds: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
                    inner_train_input_excluded: set[str] = set()
                    inner_validation_input_excluded: set[str] = set()
                    inner_train_oof_exclusions: dict[str, set[str]] = {}
                    inner_validation_oof_exclusions: dict[str, set[str]] = {}
                    for inner_origin, validation_month in group.groupby("__origin", sort=True):
                        inner_origin = pd.Timestamp(inner_origin)
                        if inner_origin >= origin:
                            continue
                        validation = validation_month.loc[
                            validation_month["__label_at"].notna()
                            & validation_month["__label_at"].lt(origin)
                            & validation_month["__y"].isin([0, 1])
                        ].copy()
                        inner_mask = (
                            group["__decision_at"].lt(inner_origin)
                            & group["__label_at"].notna()
                            & group["__label_at"].lt(inner_origin)
                            & group["__y"].isin([0, 1])
                        )
                        inner_train_candidates = group.loc[inner_mask].copy()
                        train_mask, _ = _complete_input_rows(inner_train_candidates, fit_inputs)
                        inner_train_input_excluded.update(
                            inner_train_candidates.loc[~train_mask, "row_id"].astype(str)
                        )
                        inner_train = inner_train_candidates.loc[train_mask].copy()
                        validation_mask, _ = _complete_input_rows(validation, fit_inputs)
                        inner_validation_input_excluded.update(
                            validation.loc[~validation_mask, "row_id"].astype(str)
                        )
                        validation = validation.loc[validation_mask].copy()
                        if native_oof_required:
                            inner_train, train_oof_reason_ids = native_rows_before(
                                inner_train, inner_origin,
                            )
                            validation, validation_oof_reason_ids = native_rows_before(
                                validation, origin,
                            )
                            for reason, ids in train_oof_reason_ids.items():
                                inner_train_oof_exclusions.setdefault(reason, set()).update(ids)
                            for reason, ids in validation_oof_reason_ids.items():
                                inner_validation_oof_exclusions.setdefault(reason, set()).update(ids)
                        if method in {"close_target", "clv_target"}:
                            validation = validation.loc[_close_target_mask(validation, origin)]
                            inner_train = inner_train.loc[_close_target_mask(inner_train, inner_origin)]
                        if len(inner_train) >= min_train and not validation.empty:
                            inner_folds.append((inner_origin, inner_train, validation))
                    inner_train_oof_reason_ids = {
                        reason: sorted(ids)
                        for reason, ids in sorted(inner_train_oof_exclusions.items())
                    }
                    inner_validation_oof_reason_ids = {
                        reason: sorted(ids)
                        for reason, ids in sorted(inner_validation_oof_exclusions.items())
                    }
                    common_receipt["inner_train_input_excluded_ids"] = sorted(inner_train_input_excluded)
                    common_receipt["inner_validation_input_excluded_ids"] = sorted(inner_validation_input_excluded)
                    common_receipt["inner_train_native_oof_excluded_ids"] = sorted(
                        row_id for ids in inner_train_oof_exclusions.values() for row_id in ids
                    )
                    common_receipt["inner_validation_native_oof_excluded_ids"] = sorted(
                        row_id for ids in inner_validation_oof_exclusions.values() for row_id in ids
                    )
                    common_receipt["inner_train_native_oof_exclusion_reason_ids"] = inner_train_oof_reason_ids
                    common_receipt["inner_validation_native_oof_exclusion_reason_ids"] = inner_validation_oof_reason_ids
                    common_receipt["inner_train_native_oof_exclusion_reason_counts"] = _reason_counts(
                        inner_train_oof_reason_ids
                    )
                    common_receipt["inner_validation_native_oof_exclusion_reason_counts"] = _reason_counts(
                        inner_validation_oof_reason_ids
                    )


                    if not inner_folds:
                        common_receipt["missing_reason"] = (
                            "INNER_SOURCE_OOF_COVERAGE"
                            if inner_train_oof_exclusions or inner_validation_oof_exclusions
                            else "INNER_INPUT_COVERAGE"
                            if inner_train_input_excluded or inner_validation_input_excluded
                            else "INNER_VALIDATION_WARMUP"
                        )
                    else:
                        losses: dict[float, list[float]] = {
                            float(value): [] for value in lambdas
                        }
                        validation_ids: list[str] = []
                        for inner_origin, inner_train, validation in inner_folds:
                            validation_ids.extend(validation["row_id"].astype(str).tolist())
                            for regularization in lambdas:
                                cache_key = (method, float(regularization), inner_origin)
                                if cache_key not in fitted_folds:
                                    fitted_folds[cache_key] = fit_head(
                                        method,
                                        fitting_frame(inner_train, method),
                                        regularization=float(regularization),
                                        tax_rate=0.12,
                                    )
                                fitted_inner = fitted_folds[cache_key]
                                score = float(fitted_inner.validation_loss(
                                    fitting_frame(validation, method)
                                ))
                                if np.isfinite(score):
                                    losses[float(regularization)].append(score)
                        mean_losses = {
                            value: float(np.mean(scores))
                            for value, scores in losses.items() if scores
                        }
                        common_receipt["validation_losses"] = {
                            str(value): score for value, score in mean_losses.items()
                        }
                        unique_validation_ids = list(dict.fromkeys(validation_ids))
                        common_receipt["inner_validation_ids"] = unique_validation_ids
                        if unique_validation_ids:
                            max_validation_release = group.loc[
                                group["row_id"].astype(str).isin(unique_validation_ids),
                                "__label_at",
                            ].max()
                            common_receipt["max_inner_validation_label_available_at"] = max_validation_release

                        if mean_losses:
                            # Higher regularization wins an exact loss tie.
                            chosen = min(mean_losses, key=lambda value: (mean_losses[value], -value))
                            common_receipt["selected_lambda"] = float(chosen)
                            common_receipt["status"] = "AVAILABLE"
                            common_receipt["missing_reason"] = None
                            cache_key = (method, float(chosen), origin)
                            if cache_key not in fitted_folds:
                                fitted_folds[cache_key] = fit_head(
                                    method,
                                    fitting_frame(outer_train, method),
                                    regularization=float(chosen),
                                    tax_rate=0.12,
                                )
                            fitted = fitted_folds[cache_key]
                            metadata = dict(fitted.metadata)
                            partition = metadata["fit_partition"]
                            fit_ids = partition["fit_ids"]
                            cal_ids = partition["calibration_ids"]
                            candidate_ids = set(common_receipt["outer_train_candidate_ids"])
                            if (
                                len(set(fit_ids)) != len(fit_ids)
                                or len(set(cal_ids)) != len(cal_ids)
                                or set(fit_ids) & set(cal_ids)
                                or not (set(fit_ids) | set(cal_ids)).issubset(candidate_ids)
                            ):
                                raise ValueError(f"{method} emitted invalid executed fitting partitions")
                            row_ids = outer_train["row_id"].astype(str)
                            actual_train = outer_train.loc[row_ids.isin(fit_ids)]
                            actual_cal = outer_train.loc[row_ids.isin(cal_ids)]
                            common_receipt.update({
                                "train_ids": fit_ids,
                                "calibration_ids": cal_ids,
                                "calibration_role": (
                                    "DISJOINT_POSTERIOR_PROBABILITY_RECOVERY" if cal_ids else "NONE"
                                ),
                                "max_train_label_available_at": actual_train["__label_at"].max(),
                                "max_calibration_label_available_at": actual_cal["__label_at"].max(),
                                "max_train_actual_label_available_at": (
                                    _utc_series(actual_train["label_available_at"]).max()
                                ),
                                "max_train_scenario_label_available_at": (
                                    _utc_series(actual_train["scenario_label_available_at"]).max()
                                    if "scenario_label_available_at" in actual_train else pd.NaT
                                ),
                            })
                            if method in {"close_target", "clv_target"}:
                                common_receipt["max_train_close_available_at"] = _effective(
                                    actual_train, "close_available_at",
                                ).max()
                            score_kind = str(metadata.get("score_kind", "probability"))
                            current_with_inputs = current.loc[prediction_input_mask].copy()
                            if current_with_inputs.empty:
                                common_receipt["status"] = "AVAILABLE_FIT_NO_CURRENT_INPUTS"
                                common_receipt["missing_reason"] = "MISSING_REQUIRED_PREDICTION_INPUTS"
                            else:
                                prediction_frame = _predictable_frame(
                                    current_with_inputs.drop(
                                        columns=["__decision_at", "__label_at", "__y", "__origin"]
                                    )
                                )
                                raw_scores = np.asarray(
                                    fitted.predict(prediction_frame), dtype=float
                                ).reshape(-1)
                                if len(raw_scores) != len(current_with_inputs):
                                    raise ValueError(
                                        f"{method} returned {len(raw_scores)} scores for "
                                        f"{len(current_with_inputs)} eligible rows"
                                    )
                                if not np.isfinite(raw_scores).all():
                                    raise ValueError(f"{method} produced non-finite held-out scores")
                                is_probability = score_kind in {
                                    "probability", "win_probability", "calibrated_probability",
                                    "tilted_probability_score", "focal_score", "smoothed_probability_score",
                                }
                                if is_probability and (
                                    (raw_scores < 0).any() or (raw_scores > 1).any()
                                ):
                                    raise ValueError(f"{method} produced invalid bounded prediction scores")
                                utility_actions = None
                                if method in {"direct_profit", "kelly_utility"}:
                                    utility_actions = fitted.decision_actions(prediction_frame)
                                for position, (index, row) in enumerate(current_with_inputs.iterrows()):
                                    item = {
                                        "cohort": cohort,
                                        "horizon": horizon,
                                        "row_id": row["row_id"],
                                        "method": method,
                                        "fit_origin": origin,
                                        "p": float(raw_scores[position]) if is_probability else np.nan,
                                        "raw_score": np.nan if is_probability else float(raw_scores[position]),
                                        "score_kind": score_kind,
                                        "status": "AVAILABLE",
                                        "missing_reason": None,
                                    }
                                    if utility_actions is not None:
                                        if isinstance(utility_actions, pd.DataFrame):
                                            action = utility_actions.iloc[position]
                                            item["action_side"] = action["action_side"]
                                            item["action_fraction"] = float(action["action_fraction"])
                                        else:
                                            item["action_side"] = utility_actions["action_side"][position]
                                            item["action_fraction"] = float(
                                                utility_actions["action_fraction"][position]
                                            )
                                    method_predictions[int(index)] = item
                                common_receipt["status"] = "AVAILABLE"
                                common_receipt["missing_reason"] = None
                            common_receipt["score_kind"] = score_kind
                            common_receipt["model_metadata"] = metadata

                for index in current_index:
                    prediction = method_predictions[int(index)]
                    if prediction["missing_reason"] == "NOT_FIT":
                        prediction["missing_reason"] = common_receipt["missing_reason"] or "NO_FINITE_INNER_VALIDATION_SCORE"
                    predictions.append(prediction)
                receipts.append(common_receipt)
    output = pd.DataFrame(predictions)
    if not output.empty:
        output = output.sort_values(
            ["cohort", "horizon", "method", "fit_origin", "row_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    return output, receipts


def _money(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) or not np.isfinite(number) else float(number)


def _prepare_bets(opportunities: pd.DataFrame, bets: pd.DataFrame, label: str) -> pd.DataFrame:
    if bets is None:
        bets = pd.DataFrame()
    if bets.empty:
        return pd.DataFrame(columns=["row_id", "month", "stake", "pnl", "won", "p", "ev_error", "unit_return"])
    if "row_id" not in bets:
        raise ValueError(f"{label} bets require row_id")
    if bets["row_id"].duplicated().any():
        raise ValueError(f"{label} must select at most one bet per opportunity row_id")
    op_dates = opportunities.set_index("row_id")["__comparison_date"]
    result = bets.copy()
    if "date" in result:
        bet_dates = _utc_series(result["date"])
        mapped = result["row_id"].map(op_dates)
        if mapped.notna().all() and not bet_dates.eq(mapped).all():
            raise ValueError(f"{label} bet dates disagree with the opportunity universe")
        result["__comparison_date"] = bet_dates.where(bet_dates.notna(), mapped)
    else:
        result["__comparison_date"] = result["row_id"].map(op_dates)
    if result["__comparison_date"].isna().any():
        raise ValueError(f"{label} has bets outside the full opportunity universe")
    if not result["row_id"].isin(opportunities["row_id"]).all():
        raise ValueError(f"{label} has bets outside the full opportunity universe")
    result["month"] = _utc_series(result["__comparison_date"]).dt.strftime("%Y-%m")
    result["stake"] = result["stake"].map(_money) if "stake" in result else 0.0
    won_known = (
        result["won"].notna()
        if "won" in result else pd.Series(False, index=result.index)
    )
    result["won"] = result["won"].fillna(False).astype(bool) if "won" in result else False
    result["p"] = pd.to_numeric(result["p"], errors="coerce") if "p" in result else np.nan
    if "pnl" in result:
        result["pnl"] = result["pnl"].map(_money)
    else:
        if "odds" not in result or not won_known.all():
            raise ValueError(f"{label} bets require pnl or odds and won")
        tax = pd.to_numeric(result.get("tax_rate", 0.0), errors="coerce").fillna(0.0) if isinstance(result.get("tax_rate", 0.0), pd.Series) else 0.0
        result["pnl"] = result["stake"] * (result["won"].astype(float) * pd.to_numeric(result["odds"], errors="coerce") * (1 - tax) - 1)
    if "unit_return" in result:
        result["unit_return"] = pd.to_numeric(result["unit_return"], errors="coerce")
    else:
        result["unit_return"] = pd.Series(np.nan, index=result.index)
    pnl_unit_return = np.where(
        result["stake"].ne(0), result["pnl"] / result["stake"], np.nan
    )
    result["unit_return"] = result["unit_return"].where(
        result["unit_return"].notna(), pnl_unit_return
    )
    if won_known.any() and {"odds", "tax_rate"}.issubset(result.columns):
        odds = pd.to_numeric(result["odds"], errors="coerce")
        tax = pd.to_numeric(result["tax_rate"], errors="coerce")
        observed_from_outcome = pd.Series(np.nan, index=result.index)
        observed_from_outcome.loc[won_known] = np.where(
            result.loc[won_known, "won"],
            odds.loc[won_known] * (1.0 - tax.loc[won_known]) - 1.0,
            -1.0,
        )
        result["unit_return"] = result["unit_return"].where(
            result["unit_return"].notna(), observed_from_outcome
        )
    probability = (
        pd.to_numeric(result["p_for_ev"], errors="coerce")
        if "p_for_ev" in result else pd.Series(np.nan, index=result.index)
    )
    probability = probability.where(probability.notna(), result["p"])
    odds = (
        pd.to_numeric(result["odds"], errors="coerce")
        if "odds" in result else pd.Series(np.nan, index=result.index)
    )
    tax = (
        pd.to_numeric(result["tax_rate"], errors="coerce")
        if "tax_rate" in result else pd.Series(np.nan, index=result.index)
    )
    valid_ev_inputs = (
        probability.between(0.0, 1.0)
        & odds.gt(1.0)
        & np.isfinite(odds)
        & tax.between(0.0, 1.0)
        & np.isfinite(result["unit_return"])
    )
    expected_net_ev = probability * odds * (1.0 - tax) - 1.0
    result["ev_error"] = (result["unit_return"] - expected_net_ev).where(valid_ev_inputs)
    return result


def _policy_aggregates(bets: pd.DataFrame, months: list[str], opportunity_counts: dict[str, int]) -> dict[str, Any]:
    month_rows: dict[str, dict[str, float | int | None]] = {}
    for month in months:
        current = bets.loc[bets["month"].eq(month)] if not bets.empty else bets
        stake = float(current["stake"].sum()) if not current.empty else 0.0
        pnl = float(current["pnl"].sum()) if not current.empty else 0.0
        count = int(len(current))
        errors = current["ev_error"].dropna() if not current.empty else pd.Series(dtype=float)
        month_rows[month] = {
            "stake": stake, "pnl": pnl, "count": count,
            "opportunities": int(opportunity_counts[month]),
            "roi": pnl / stake if stake > 0 else None,
            "pnl_per_opportunity": pnl / opportunity_counts[month] if opportunity_counts[month] else None,
            "ev_error": float(errors.mean()) if not errors.empty else None,
        }
    stake = float(bets["stake"].sum()) if not bets.empty else 0.0
    pnl = float(bets["pnl"].sum()) if not bets.empty else 0.0
    opportunity_total = int(sum(opportunity_counts.values()))
    errors = bets["ev_error"].dropna() if not bets.empty else pd.Series(dtype=float)
    weighted_month_pnl = [
        month_rows[month]["pnl_per_opportunity"] for month in months
        if month_rows[month]["pnl_per_opportunity"] is not None
    ]
    defined_month_rois = [month_rows[month]["roi"] for month in months if month_rows[month]["roi"] is not None]
    return {
        "bets": int(len(bets)), "stake": stake, "pnl": pnl,
        "roi": pnl / stake if stake > 0 else None,
        "pnl_per_opportunity": pnl / opportunity_total if opportunity_total else None,
        "ev_error": float(errors.mean()) if not errors.empty else None,
        "monthly": month_rows,
        "mean_weighted": {
            "pnl_per_opportunity": pnl / opportunity_total if opportunity_total else None,
            "roi": pnl / stake if stake > 0 else None,
        },
        "equal_month": {
            "pnl_per_opportunity": float(np.mean(weighted_month_pnl)) if weighted_month_pnl else None,
            "roi": float(np.mean(defined_month_rois)) if defined_month_rois else None,
        },
    }


def paired_monthly_comparison(
    opportunities: pd.DataFrame,
    bets_a: pd.DataFrame,
    bets_b: pd.DataFrame,
    draws: int = 5000,
    seed: int = 20260928,
) -> dict[str, Any]:
    """Compare two fixed policies on the entire shared preselection month universe.

    Bootstrap resamples complete calendar months, preserving zero-bet months.
    Conditional ROI intervals are undefined if any draw has zero stake; these draws
    are counted, never dropped or imputed. Realized PnL uses only the selected
    price, stake and result, never the probability column.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    if not {"row_id", "date"}.issubset(opportunities.columns):
        raise ValueError("opportunities require row_id and date for full-universe month pairing")
    if opportunities["row_id"].isna().any() or opportunities["row_id"].duplicated().any():
        raise ValueError("opportunity row_id must be unique and non-missing")
    full = opportunities[["row_id", "date"]].copy()
    full["__comparison_date"] = _utc_series(full["date"])
    if full["__comparison_date"].isna().any():
        raise ValueError("the full opportunity universe requires known/scenario dates")
    full["month"] = full["__comparison_date"].dt.strftime("%Y-%m")
    months = sorted(full["month"].unique().tolist())
    if not months:
        raise ValueError("the full opportunity universe has no calendar months")
    counts = full.groupby("month", sort=True).size().astype(int).to_dict()
    a = _prepare_bets(full, bets_a, "policy_a")
    b = _prepare_bets(full, bets_b, "policy_b")
    summary_a = _policy_aggregates(a, months, counts)
    summary_b = _policy_aggregates(b, months, counts)

    monthly_arrays: dict[str, dict[str, np.ndarray]] = {}
    for label, bets in (("a", a), ("b", b)):
        month_map = {month: np.zeros(4, dtype=float) for month in months}
        for month, current in bets.groupby("month", sort=False):
            month_map[month] = np.array([
                float(current["pnl"].sum()), float(current["stake"].sum()),
                float(len(current)), float(current["ev_error"].mean()) if current["ev_error"].notna().any() else np.nan,
            ])
        monthly_arrays[label] = {
            key: np.vstack([month_map[month] for month in months])
            for key in ("values",)
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(months), size=(draws, len(months)))
    opportunity_array = np.array([counts[month] for month in months], dtype=np.int64)
    values_a = monthly_arrays["a"]["values"][:, :2]
    values_b = monthly_arrays["b"]["values"][:, :2]
    totals_a = values_a[indices].sum(axis=1)
    totals_b = values_b[indices].sum(axis=1)
    opportunities_sampled = opportunity_array[indices].sum(axis=1)
    delta_pnl_per_opportunity = (
        totals_b[:, 0] - totals_a[:, 0]
    ) / opportunities_sampled
    delta_equal_month_pnl_per_opportunity = (
        (values_b[:, 0] / opportunity_array)[indices].mean(axis=1)
        - (values_a[:, 0] / opportunity_array)[indices].mean(axis=1)
    )
    roi_defined = (totals_a[:, 1] > 0) & (totals_b[:, 1] > 0)
    empty_replicates = int(np.count_nonzero(~roi_defined))
    delta_roi = 100.0 * (
        totals_b[roi_defined, 0] / totals_b[roi_defined, 1]
        - totals_a[roi_defined, 0] / totals_a[roi_defined, 1]
    )

    full_a = np.array([summary_a["pnl"], summary_a["stake"], summary_a["bets"]], dtype=float)
    full_b = np.array([summary_b["pnl"], summary_b["stake"], summary_b["bets"]], dtype=float)
    observed_delta_roi = (
        100.0 * (full_b[0] / full_b[1] - full_a[0] / full_a[1])
        if full_a[1] > 0 and full_b[1] > 0 else None
    )
    if empty_replicates:
        ci_lower = ci_upper = None
        bootstrap_status = "UNDEFINED_REPLICATES"
    else:
        ci_lower, ci_upper = map(float, np.quantile(delta_roi, [0.025, 0.975]))
        bootstrap_status = "ESTIMATED"
    pnl_ci = list(map(float, np.quantile(delta_pnl_per_opportunity, [0.025, 0.975])))
    equal_month_ci = list(map(float, np.quantile(delta_equal_month_pnl_per_opportunity, [0.025, 0.975])))
    return {
        "months": months,
        "paired_months": months,
        "policy_a": summary_a,
        "policy_b": summary_b,
        "mean_weighted": {
            "delta_pnl_per_opportunity": (
                summary_b["pnl_per_opportunity"] - summary_a["pnl_per_opportunity"]
                if summary_a["pnl_per_opportunity"] is not None and summary_b["pnl_per_opportunity"] is not None else None
            ),
            "ci_lower": pnl_ci[0], "ci_upper": pnl_ci[1],
        },
        "equal_month": {
            "delta_pnl_per_opportunity": (
                summary_b["equal_month"]["pnl_per_opportunity"] - summary_a["equal_month"]["pnl_per_opportunity"]
                if summary_a["equal_month"]["pnl_per_opportunity"] is not None and summary_b["equal_month"]["pnl_per_opportunity"] is not None else None
            ),
            "ci_lower": equal_month_ci[0], "ci_upper": equal_month_ci[1],
        },
        "bootstrap": {
            "draws": int(draws), "empty_replicates": int(empty_replicates),
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "delta_roi_pct": observed_delta_roi,
            "status": bootstrap_status,
            "pnl_per_opportunity_ci": pnl_ci,
            "equal_month_pnl_per_opportunity_ci": equal_month_ci,
            "undefined_roi_fraction": float(empty_replicates / draws),
        },
    }


def evaluate_c0_market_heads(panel, heads, *, draws=5000, seed=20260929):
    """Selection-independent cash denominator and shared full-universe resamples."""
    from scipy.special import xlogy
    from sklearn.metrics import roc_auc_score
    months = sorted(set(panel.attrs["monthly_universe"])) if "monthly_universe" in panel.attrs else sorted(
        m for m in set(panel["month"].astype(str)) if len(m) == 7 and m[4] == "-")
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(months), (draws, len(months))) if months else np.empty((draws, 0), int)
    records, summary, monthly = [], {}, {}
    common = panel["native_admitted"].fillna(False) & panel["p_C0"].notna() & panel["q"].notna() & panel["y"].notna()
    for column in ("p_C0","q","y"):
        numeric = pd.to_numeric(panel[column], errors="coerce")
        common &= (numeric.notna() & np.isfinite(numeric) & numeric.between(0,1)
                   & ~panel[column].map(lambda v:isinstance(v,(bool,np.bool_))))
    common &= panel["y"].isin([0,1])
    for recipe, head in heads.items():
        rows = []
        for _, row in panel.iterrows():
            output = {"row_id": row["row_id"], "recipe": recipe, "month": row["month"],
                      "financial_eligible": bool(row["financial_eligible"]), "profit": 0.,
                      "selected": False, "absence": None, "probability": None}
            try:
                if not row.get("primary_decision", True):
                    raise ValueError("non_primary_event")
                if recipe in {"S", "S-cal", "Fixed50", "H"} and not row["native_admitted"]:
                    raise ValueError("native_not_admitted")
                details = head.predict_details(pd.DataFrame([row]))
                p = float(details["probability"][0])
                output.update(probability=p, raw_probability=float(details["raw_probability"][0]),
                              raw_logit=float(details["logit"][0]), clipped=bool(details["clipped"][0]))
                quotes = row.get("quotes", [])
                choices = []
                for side, probability in (("A", p), ("B", 1-p)):
                    available = [quote for quote in quotes if
                        isinstance(quote.get("odds_" + side.lower()), (int, float))
                        and not isinstance(quote.get("odds_" + side.lower()), bool)
                        and np.isfinite(quote["odds_" + side.lower()])
                        and quote["odds_" + side.lower()] > 1]
                    if not available:
                        continue
                    quote = sorted(available, key=lambda q: (-q["odds_" + side.lower()], q["bookmaker"],
                                                             str(q.get("quote_id", q.get("odds_snapshot_id", "")))))[0]
                    odds = quote["odds_" + side.lower()]
                    choices.append((.88 * odds * probability - 1, side, probability, odds, quote))
                if choices and row["financial_eligible"]:
                    ev, side, probability, odds, quote = sorted(choices, key=lambda t: (-t[0], t[1]))[0]
                    if ev > .05:
                        y = row["y"] if side == "A" else 1-row["y"]
                        q = row["q"] if side == "A" else None if pd.isna(row["q"]) else 1-row["q"]
                        profit = .88 * odds * y - 1
                        output.update(selected=True, side=side, odds=odds,
                                      quote_id=quote.get("quote_id",quote.get("odds_snapshot_id")),
                                      bookmaker=quote["bookmaker"], predicted_ev=ev, profit=profit,
                                      overstatement=.88 * odds * (probability-y),
                                      model_market_component=None if pd.isna(q) else .88*odds*(probability-q),
                                      market_noise_component=None if pd.isna(q) else .88*odds*(q-y),
                                      near_band=bool(.05 < ev <= .10))
            except (ValueError, KeyError, TypeError) as error:
                output["absence"] = str(error)
            rows.append(output)
        ledger = pd.DataFrame(rows) if rows else pd.DataFrame(columns=(
            "row_id","recipe","month","financial_eligible","profit","selected","absence","probability"))
        ledger["selected"] = ledger["selected"].astype(bool)
        ledger["financial_eligible"] = ledger["financial_eligible"].astype(bool)
        records.extend(rows)
        pairs = ledger.loc[common.to_numpy() & ledger["probability"].notna().to_numpy()]
        y = panel.loc[pairs.index, "y"].to_numpy(float)
        p = pairs["probability"].to_numpy(float)
        ll = -(xlogy(y, p) + xlogy(1-y, 1-p))
        brier = (p-y)**2
        metric = {"common_count": len(p), "coverage_count": int(ledger["probability"].notna().sum()),
                  "LL": float(np.mean(ll)) if len(p) else None,
                  "Brier": float(np.mean(brier)) if len(p) else None,
                  "AUC": float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
                  "accuracy": float(np.mean((p >= .5) == y)) if len(p) else None}
        metric["absence_counts"] = ledger["absence"].dropna().value_counts().to_dict()
        extra_mask = (panel["q"].isna() & panel["native_admitted"] & panel["y"].isin([0,1])
                      & ledger["probability"].notna())
        extra_p = ledger.loc[extra_mask,"probability"].to_numpy(float)
        extra_y = panel.loc[extra_mask,"y"].to_numpy(float)
        metric["S_only_descriptive_scores"] = {"count":len(extra_p),
            "LL":float(np.mean(-(xlogy(extra_y,extra_p)+xlogy(1-extra_y,1-extra_p)))) if len(extra_p) else None,
            "Brier":float(np.mean((extra_p-extra_y)**2)) if len(extra_p) else None}
        metric["calibration_bins"] = []
        for lower, upper in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
            mask = (p >= lower) & (p < upper if upper < 1 else p <= 1)
            metric["calibration_bins"].append({"lower":float(lower),"upper":float(upper),
                "count":int(mask.sum()),"mean_probability":float(p[mask].mean()) if mask.any() else None,
                "observed_rate":float(y[mask].mean()) if mask.any() else None})
        metric["tails"] = {str(t):{"count":int((p>=t).sum()),
            "observed_rate":float(y[p>=t].mean()) if (p>=t).any() else None} for t in (.8,.9,.95)}
        selected_all = ledger.loc[ledger["selected"]]
        metric["selected_count"] = len(selected_all)
        metric["mean_selected_ev"] = float(selected_all["predicted_ev"].mean()) if len(selected_all) else None
        metric["financial_eligible_count"] = int(panel["financial_eligible"].sum())
        primary_mask = panel["primary_decision"] if "primary_decision" in panel else pd.Series(True,index=panel.index)
        excluded = panel.loc[primary_mask & ~panel["financial_eligible"]]
        metric["settlement_exclusions"] = (excluded["settlement_status"].value_counts().to_dict()
            if "settlement_status" in excluded else {"unverified":len(excluded)})
        metric["pending_exclusion_count"] = metric["settlement_exclusions"].get("pending",0)
        metric["non_primary_event_count"] = int((~primary_mask).sum())
        metric["S_only_count"] = int((panel["native_admitted"] & panel["q"].isna()).sum())
        metric["market_only_count"] = int((~panel["native_admitted"] & panel["q"].notna()).sum())
        metric["edition_sensitivity_basis"] = "authenticated_edition" if "authenticated_edition" in panel else "descriptive_source_label_proxy"
        month_values = []
        for month in months:
            block = ledger.loc[ledger["month"] == month]
            selected = block.loc[block["selected"]]
            pair_indices = pairs.index[pairs["month"] == month]
            positions = pairs.index.get_indexer(pair_indices)
            near = selected.loc[selected.get("near_band", pd.Series(False,index=selected.index)).fillna(False)]
            month_values.append({"eligible": int(block["financial_eligible"].sum()),
                "bets": len(selected), "profit": float(selected["profit"].sum()),
                "error": float(selected["overstatement"].sum()) if len(selected) else 0.,
                "near_bets": len(near), "near_error": float(near["overstatement"].sum()) if len(near) else 0.,
                "pairs": len(positions), "ll_sum": float(ll[positions].sum()),
                "brier_sum": float(brier[positions].sum())})
        monthly[recipe] = month_values
        def interval(numerator, denominator):
            contributing = sum(v[denominator] > 0 for v in month_values)
            if contributing < 2:
                return {"status": "NOT_ESTIMABLE", "interval": None, "undefined_draws": draws}
            ratios = []
            undefined = 0
            for sample in resamples:
                den = sum(month_values[i][denominator] for i in sample)
                if den == 0:
                    undefined += 1
                else:
                    ratios.append(sum(month_values[i][numerator] for i in sample)/den)
            return {"status": "NOT_ESTIMABLE" if undefined else "ESTIMABLE",
                    "interval": None if undefined else np.quantile(ratios,[.025,.975]).tolist(),
                    "undefined_draws": undefined}
        for name, num, den in (("ROI","profit","bets"),("profit_per_opportunity","profit","eligible"),
                               ("overstatement","error","bets"),("near_overstatement","near_error","near_bets")):
            total = sum(v[den] for v in month_values)
            metric[name] = {"point": sum(v[num] for v in month_values)/total if total else None, **interval(num,den)}
            metric[name]["equal_month_point"] = (float(np.mean([v[num]/v[den] for v in month_values]))
                if month_values and all(v[den] for v in month_values) else None)
            equal_values = []
            equal_undefined = 0
            for sample in resamples:
                if any(month_values[i][den] == 0 for i in sample):
                    equal_undefined += 1
                else:
                    equal_values.append(float(np.mean([month_values[i][num]/month_values[i][den] for i in sample])))
            metric[name]["equal_month_interval"] = (np.quantile(equal_values,[.025,.975]).tolist()
                if len(months) >= 2 and equal_undefined == 0 else None)
            metric[name]["equal_month_undefined_draws"] = equal_undefined
        metric["monthly"] = dict(zip(months,month_values))
        summary[recipe] = metric
    contrasts = {}
    for control in ("M-cal","M-shape"):
        values = []
        contributing = sum(a["pairs"] > 0 and b["pairs"] > 0 for a,b in zip(monthly["H"],monthly[control]))
        undefined = 0
        for sample in resamples:
            n = sum(monthly["H"][i]["pairs"] for i in sample)
            if n == 0:
                undefined += 1
            else:
                values.append(sum(monthly["H"][i]["ll_sum"]-monthly[control][i]["ll_sum"] for i in sample)/n)
        contrasts[control] = {"LL_interval": np.quantile(values,[.025,.975]).tolist()
            if contributing >= 2 and not undefined else None, "undefined_draws": undefined,
            "status": "ESTIMABLE" if contributing >= 2 and not undefined else "NOT_ESTIMABLE"}
    increment = []
    undefined = 0
    for sample in resamples:
        den = sum(monthly["H"][i]["eligible"] for i in sample)
        if not den:
            undefined += 1
        else:
            increment.append(sum(monthly["H"][i]["profit"]-monthly["M-cal"][i]["profit"] for i in sample)/den)
    increment_ci = np.quantile(increment,[.025,.975]).tolist() if len(months)>=2 and not undefined else None
    h = summary["H"]
    def bounded(value):
        ci = value["interval"]
        return ci is not None and ci[0] >= -.02 and ci[1] <= .02
    controls_pass = all(contrasts[c]["LL_interval"] is not None and contrasts[c]["LL_interval"][1] < 0
        and h["Brier"] is not None and summary[c]["Brier"] is not None and h["Brier"] <= summary[c]["Brier"]
        and h["coverage_count"] >= summary[c]["coverage_count"] for c in contrasts)
    profit_ci = h["profit_per_opportunity"]["interval"]
    passed = (controls_pass and bounded(h["overstatement"]) and bounded(h["near_overstatement"])
              and profit_ci is not None and profit_ci[0] > 0 and increment_ci is not None
              and increment_ci[0] > 0 and h["ROI"]["interval"] is not None)
    from src.analysis.devig_comparison import devig_power, devig_additive
    sensitivities = []
    for _, row in panel.iterrows():
        valid = [q for q in row.get("quotes", []) if all(
            isinstance(q.get(name), (int,float)) and not isinstance(q.get(name), bool)
            and np.isfinite(q[name]) and q[name] > 1 for name in ("odds_a","odds_b"))]
        entry = {"row_id":row["row_id"],"classification":"DIAGNOSTIC_NOT_SELECTION",
                 "power_q":None,"additive_q":None,"leave_execution_book_out":[]}
        if len({q["bookmaker"] for q in valid}) >= 2:
            try:
                entry["power_q"] = float(np.mean([devig_power(q["odds_a"],q["odds_b"])[0] for q in valid]))
                entry["additive_q"] = float(np.mean([devig_additive(q["odds_a"],q["odds_b"])[0] for q in valid]))
            except (ValueError,RuntimeError) as error:
                entry["margin_method_failure"] = str(error)
        for execution in valid:
            peers = [q for q in valid if q["bookmaker"] != execution["bookmaker"]]
            entry["leave_execution_book_out"].append({"execution_book":execution["bookmaker"],
                "q":float(np.mean([q["odds_b"]/(q["odds_a"]+q["odds_b"]) for q in peers]))
                    if len({q["bookmaker"] for q in peers}) >= 2 else None,
                "reason":None if len({q["bookmaker"] for q in peers}) >= 2 else "two_remaining_books_required"})
        sensitivities.append(entry)
    return {"schema":"c0_assessment_v1","metrics":summary,"contrasts":contrasts,
            "profit_increment_interval":increment_ci,"months":months,"draws":draws,"seed":seed,
            "decision":"DEVELOPMENT_CANDIDATE" if passed else "NO_BETTING_EDGE_CLAIM",
            "rows":records,"sensitivities":sensitivities,
            "estimands":{"proper_score":"match_weighted_common_population",
                "conditional":"bet_weighted","cash":"eligible_opportunity_weighted",
                "equal_month":"secondary_full_universe_no_finite_subset"}}
