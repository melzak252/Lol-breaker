"""Frozen, provenance-preserving inputs for the all-approaches EV experiment.

Historical day-only sources stay day-only: recorded UTC columns are never
filled from date strings. Scenario times are separate columns and carry an
assumption label; callers opt into them only through :func:`effective_times`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_RELATIVE = Path("data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json")
PROTOCOL_SHA256 = "7bca9fc39bad77d3ecb84f4cd607a076861630dd14cb16467abb30f32db2b0cd"
CANONICAL_MANIFEST_RELATIVE = Path("conf/base/research_benchmark.json")
CANONICAL_MANIFEST_SHA256 = "dd1c60bbc85329a21018e8f044dad50fef02e1095964c0a39462635dc925be6e"
ARCHIVE_QUOTES_SHA256 = "84300db56904793b9f4ec66ce267270c39ca139a529c81f4ec1d80045e6dcdb9"
ARCHIVE_RECEIPT_SHA256 = "7bf471f844cfc8c0c1f549ddfba7f5fecf7bed33cfd614597407ba253d220926"
PRODUCTION_PANEL_SHA256 = "b0abe8cd8768b0d0ba4fd4cd0ae74255d299095ae6ec119b1efbdf979352de2d"
PRODUCTION_QUOTES_SHA256 = "b8c4b614093abc57ef8fc25b0f177f4908c417d3c3d1735a40f7351c66cb0558"
UNCERTAINTY_RUN = "a0_uncertainty_20260926_222155"
UNCERTAINTY_COMPLETION_SHA256 = "e140e0ff8d4cab626c12769d10879f7bd265f118ddf51d9240c8e23180641524"
UNCERTAINTY_VARIANTS = ("seed_only", "block_bootstrap")
UNCERTAINTY_YEARS = (2024, 2025, 2026)

OPPORTUNITY_COLUMNS = (
    "row_id", "golgg_match_id", "cohort", "horizon", "date", "decision_at",
    "start_at", "label_available_at", "y", "p_s", "q", "odds_a", "odds_b",
    "bookmaker_a", "bookmaker_b", "quote_id_a", "quote_id_b", "quote_at_a",
    "quote_at_b", "best_of", "league", "information_status", "strict_prior_day",
    "is_common",
)
ACTUAL_TIME_COLUMNS = (
    "prediction_at", "fit_origin", "decision_at", "start_at", "label_available_at",
    "quote_at_a", "quote_at_b", "feature_available_at", "close_available_at",
    "close_quote_at_a", "close_quote_at_b",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read frozen JSON receipt {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Frozen JSON receipt must contain an object: {path}")
    return result


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen {label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _research_artifact_path(value: str | Path, research_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidate = (research_root / path).resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Missing research receipt artifact: {candidate}")
    resolved = path.resolve()
    try:
        resolved.relative_to(research_root.resolve())
        return resolved
    except ValueError:
        pass
    parts = path.parts
    for index, part in enumerate(parts):
        suffix = part.rsplit("-", 1)
        if len(suffix) == 2 and len(suffix[1]) == 8 and suffix[1].isdigit():
            candidate = research_root.joinpath(*parts[index:]).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Receipt artifact cannot be resolved under research_root: {path}")


def _verify_receipt_hashes(
    entries: Any,
    research_root: Path,
    label: str,
    verified_hashes: dict[Path, str],
) -> dict[str, str]:
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{label} receipt must pin at least one artifact")
    verified: dict[str, str] = {}
    for recorded_path, expected_value in entries.items():
        expected = str(expected_value)
        if len(expected) != 64:
            raise ValueError(f"{label} receipt has an invalid SHA-256 for {recorded_path}")
        path = _research_artifact_path(str(recorded_path), research_root)
        actual = verified_hashes.get(path)
        if actual is None:
            actual = _verify_sha256(path, expected, f"{label} artifact")
            verified_hashes[path] = actual
        elif actual != expected:
            raise ValueError(f"{label} receipt SHA-256 conflicts for {path}")
        verified[_display_path(path, research_root)] = actual
    return verified


def _receipt_digest_for_suffix(entries: Any, suffix: str, label: str) -> tuple[str, str]:
    if not isinstance(entries, dict):
        raise ValueError(f"{label} receipt has no hash map")
    matches = [
        (str(path), str(digest))
        for path, digest in entries.items()
        if Path(str(path)).as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} receipt must identify exactly one artifact ending in {suffix}")
    return matches[0]


def _resolve_research_root(research_root: str | Path | None) -> Path:
    if research_root is not None:
        root = Path(research_root).expanduser()
    elif os.environ.get("ENSEMBLE_RESEARCH_ROOT"):
        root = Path(os.environ["ENSEMBLE_RESEARCH_ROOT"]).expanduser()
    else:
        locator = PROJECT_ROOT / "data/research_root.txt"
        if not locator.is_file():
            raise FileNotFoundError(
                "Research root is unresolved; pass research_root, set ENSEMBLE_RESEARCH_ROOT, "
                "or configure data/research_root.txt"
            )
        value = locator.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"Research-root locator is empty: {locator}")
        root = Path(value).expanduser()
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Research root is not a directory: {root}")
    return root


def _project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return (PROJECT_ROOT / default).resolve()
    supplied = Path(path).expanduser()
    return (PROJECT_ROOT / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()

def _display_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _key_series(values: pd.Series) -> pd.Series:
    return values.map(_key).astype("string")


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _utc_timestamp(value: Any, *, label: str, reject_naive: bool = True) -> pd.Timestamp | pd.NaT:
    if _is_missing(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid timestamp in {label}: {value!r}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        if reject_naive:
            raise ValueError(f"Timezone-aware timestamp required in {label}: {value!r}")
        return pd.NaT
    return timestamp.tz_convert("UTC")


def _utc_series(values: pd.Series, *, label: str, reject_naive: bool = True) -> pd.Series:
    return pd.Series(
        [_utc_timestamp(value, label=label, reject_naive=reject_naive) for value in values],
        index=values.index,
        dtype="datetime64[ns, UTC]",
    )


def _scenario_assumption(frame: pd.DataFrame) -> pd.Series:
    if "scenario_timestamp_assumption" in frame:
        value = frame["scenario_timestamp_assumption"]
    elif "timestamp_assumption" in frame:
        value = frame["timestamp_assumption"]
    else:
        value = pd.Series(pd.NA, index=frame.index, dtype="string")
    return value.astype("string").str.strip().replace("", pd.NA)


def effective_times(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return UTC effective times, preferring recorded values over explicit scenarios.

    Scenario fallback is row-local and requires both ``information_status``
    equal to ``historical_scenario`` and a non-empty timestamp-assumption field.
    The returned series never mutates either source column.
    """
    scenario_column = f"scenario_{column}"
    if column not in frame and scenario_column not in frame:
        raise KeyError(f"Neither actual {column!r} nor scenario {scenario_column!r} is present")
    if column in frame:
        actual = _utc_series(frame[column], label=column)
    else:
        actual = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if scenario_column not in frame or "information_status" not in frame:
        return actual
    scenario = _utc_series(frame[scenario_column], label=scenario_column)
    allowed = frame["information_status"].astype("string").eq("historical_scenario")
    allowed &= _scenario_assumption(frame).notna()
    return actual.where(actual.notna(), scenario.where(allowed, pd.NaT))


def _assert_timezone_aware_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in frame:
            continue
        for value in frame[column]:
            _utc_timestamp(value, label=column)


def _probabilities(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        supplied = frame[column].notna()
        invalid = supplied & (~np.isfinite(numeric) | ~numeric.between(0.0, 1.0))
        if invalid.any():
            raise ValueError(f"Invalid probability values in {column}")


def validate_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate side orientation, identity uniqueness, prices and actual timing.

    Rows with unavailable quote/market fields remain in the frame so source
    coverage is not silently reduced. Non-null values must be valid.
    """
    required = {
        "row_id", "golgg_match_id", "cohort", "horizon", "date", "decision_at",
        "start_at", "label_available_at", "y", "p_s", "q", "odds_a", "odds_b",
        "bookmaker_a", "bookmaker_b", "quote_id_a", "quote_id_b", "quote_at_a",
        "quote_at_b", "best_of", "league", "information_status", "strict_prior_day",
        "is_common",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Opportunities are missing required columns: {missing}")
    result = frame.copy()
    if result[["row_id", "golgg_match_id", "cohort", "horizon"]].isna().any().any():
        raise ValueError("Opportunities require non-null row and fixture identities, cohort, and horizon")
    for column in ("row_id", "golgg_match_id", "cohort", "horizon"):
        result[column] = _key_series(result[column])
    if result["row_id"].duplicated().any():
        raise ValueError("Opportunities contain duplicate row_id values")
    fixture_key = ["golgg_match_id", "cohort", "horizon"]
    if result.duplicated(fixture_key).any():
        raise ValueError("Opportunities contain a repeated fixture within cohort and horizon")

    outcomes = pd.to_numeric(result["y"], errors="coerce")
    supplied_outcome = result["y"].notna()
    if (supplied_outcome & ~outcomes.isin([0, 1])).any():
        raise ValueError("Invalid binary outcome y")
    result["y"] = outcomes.astype("Int64")
    _probabilities(result, ("p_s", "q", "close_q", "sigma_z", "p_std", "p_q10", "p_q90"))
    for column in ("odds_a", "odds_b", "close_odds_a", "close_odds_b"):
        if column not in result:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        supplied = result[column].notna()
        if (supplied & (~np.isfinite(numeric) | numeric.le(1.0))).any():
            raise ValueError(f"Invalid decimal odds in {column}")
        result[column] = numeric

    _assert_timezone_aware_columns(result, ACTUAL_TIME_COLUMNS)
    for column in ACTUAL_TIME_COLUMNS:
        if column in result:
            result[column] = _utc_series(result[column], label=column)
    if "decision_at" in result:
        decision = result["decision_at"]
        for observed_column in ("quote_at_a", "quote_at_b", "feature_available_at"):
            if observed_column not in result:
                continue
            observed = result[observed_column]
            invalid = observed.notna() & decision.notna() & observed.gt(decision)
            if invalid.any():
                raise ValueError(f"{observed_column} source was observed after decision_at")

    scenario_columns = [
        column for column in result
        if column.startswith("scenario_") and column != "scenario_timestamp_assumption"
    ]
    scenario_rows = result[scenario_columns].notna().any(axis=1) if scenario_columns else pd.Series(False, index=result.index)
    if (scenario_rows & _scenario_assumption(result).isna()).any():
        raise ValueError("Scenario timestamps require a declared timestamp assumption")
    for column in scenario_columns:
        result[column] = _utc_series(result[column], label=column)
    for column in ("strict_prior_day", "is_common"):
        if result[column].isna().any():
            raise ValueError(f"{column} must be an explicit boolean")
        values = result[column].map(_strict_bool)
        if values.isna().any():
            raise ValueError(f"{column} contains an invalid boolean")
        result[column] = values.astype(bool)
    return result


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_missing(value):
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


def _aware_origin(origin: Any) -> pd.Timestamp:
    value = _utc_timestamp(origin, label="origin")
    if pd.isna(value):
        raise ValueError("A timezone-aware origin is required")
    return value


def calibration_rows_before(
    frame: pd.DataFrame,
    origin: Any,
    require_oof: bool = True,
) -> pd.DataFrame:
    """Select strictly past released rows from complete-recipe OOF predictions.

    Explicit historical scenarios may provide effective prediction, fit and
    label-release times, but those remain separate from the actual timestamps.
    With ``require_oof=False`` only designated ``sigma_reference`` rows may
    additionally contribute; in-sample CAL is never used as ordinary OOF.
    """
    if not isinstance(require_oof, (bool, np.bool_)):
        raise ValueError("require_oof must be boolean")
    if frame.empty:
        return frame.copy()
    origin_utc = _aware_origin(origin)
    required = {"prediction_at", "fit_origin", "label_available_at", "is_oof", "recipe_status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Calibration candidates are missing columns: {missing}")
    prediction_time = effective_times(frame, "prediction_at")
    fit_time = effective_times(frame, "fit_origin")
    label_time = effective_times(frame, "label_available_at")
    complete = frame["recipe_status"].astype("string").eq("COMPLETE-RECIPE")
    is_oof = frame["is_oof"].map(_strict_bool).eq(True)
    selected = complete & is_oof
    if not require_oof:
        role = frame.get("prediction_role", pd.Series(pd.NA, index=frame.index)).astype("string")
        sigma_reference = role.eq("sigma_reference")
        in_sample_cal = frame["recipe_status"].astype("string").isin({"IN-SAMPLE-CAL", "IN_SAMPLE_CAL"})
        selected |= sigma_reference & in_sample_cal
    selected &= prediction_time.notna() & prediction_time.lt(origin_utc)
    selected &= fit_time.notna() & fit_time.lt(origin_utc)
    selected &= label_time.notna() & label_time.lt(origin_utc)
    if "information_status" in frame:
        scenario = frame["information_status"].astype("string").eq("historical_scenario")
        assumption = _scenario_assumption(frame)
        selected &= ~scenario | assumption.notna()
    return frame.loc[selected].copy()


def _date_strings(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Missing or invalid calendar date in {label}")
    return parsed.dt.strftime("%Y-%m-%d").astype("string")


def _verify_canonical_oof_provenance(
    canonical: pd.DataFrame,
    research_root: Path,
    phase_root: Path,
) -> dict[str, Any]:
    verified_hashes: dict[Path, str] = {}
    manifest_path = phase_root / "data/06_models/a0_manifest_causal.json"
    manifest = _read_json(manifest_path)
    if manifest.get("model") != "causal_a0_same_recipe":
        raise ValueError("Canonical annual OOF source is not the corrected same-recipe causal A0")
    manifest_years = pd.to_numeric(pd.Series(manifest.get("years", [])), errors="coerce")
    if (
        manifest_years.isna().any()
        or manifest_years.duplicated().any()
        or set(manifest_years.astype(int)) != set(UNCERTAINTY_YEARS)
    ):
        raise ValueError("Corrected causal A0 manifest does not cover every frozen model year")
    manifest_hashes = _verify_receipt_hashes(
        manifest.get("hashes"), research_root, "corrected causal A0 model manifest", verified_hashes,
    )

    slice_receipt_path = phase_root / "data/08_reporting/availability_slice_receipt.json"
    slice_receipt = _read_json(slice_receipt_path)
    if not str(slice_receipt.get("status", "")).startswith("COMPLETE_FULL_CAUSAL_RETRAIN"):
        raise ValueError("Corrected causal A0 availability receipt is incomplete")
    if "correctedbank" not in str(slice_receipt.get("model_repair", "")).lower():
        raise ValueError("Availability receipt does not attest corrected-bank retraining")
    audit_relative = "data/08_reporting/availability_training_calibration.json"
    audit_path = phase_root / audit_relative
    audit_hash_key = f"{phase_root.name}/{audit_relative}"
    expected_audit_hash = slice_receipt.get("outputs_sha256", {}).get(audit_hash_key)
    if not expected_audit_hash:
        raise ValueError("Availability slice receipt does not pin the historical training audit")
    audit_sha = _verify_sha256(audit_path, str(expected_audit_hash), "historical A0 training-availability audit")
    audit = _read_json(audit_path)
    old_audit_ids = sorted({
        key
        for error in audit.get("errors", [])
        for value in error.get("late_ids", [])
        if (key := _key(value)) is not None
    })
    audit_model_paths = [str(record.get("path", "")) for record in audit.get("records", [])]
    old_roots = ("unified-ratings-20260911", "nonlinear-history-20260912")
    if (
        audit.get("status") != "FAIL"
        or len(old_audit_ids) != 2
        or not audit_model_paths
        or any(not any(root_name in path for root_name in old_roots) for path in audit_model_paths)
    ):
        raise ValueError("Recorded FAIL is not the pinned pre-correction training audit")
    if any("causal_a0" in Path(path).parts for path in audit_model_paths):
        raise ValueError("Historical release-day FAIL unexpectedly targets corrected causal_a0 artifacts")
    audit_scope = str(slice_receipt.get("original_training_audit", ""))
    if "FAIL retained" not in audit_scope:
        raise ValueError("Availability slice receipt does not identify the FAIL as an original audit")
    audit_report_relative = "data/08_reporting/availability_report.md"
    audit_report_path = phase_root / audit_report_relative
    audit_report_hash = slice_receipt.get("outputs_sha256", {}).get(
        f"{phase_root.name}/{audit_report_relative}",
    )
    if not audit_report_hash:
        raise ValueError("Availability slice receipt does not pin the audit scope report")
    audit_report_sha = _verify_sha256(
        audit_report_path, str(audit_report_hash), "historical A0 availability scope report",
    )

    proof_relative = "data/08_reporting/availability_causal_replay_proof.json"
    proof_path = phase_root / proof_relative
    expected_proof_hash = slice_receipt.get("outputs_sha256", {}).get(f"{phase_root.name}/{proof_relative}")
    if not expected_proof_hash:
        raise ValueError("Availability slice receipt does not pin corrected causal replay proof")
    proof_sha = _verify_sha256(proof_path, str(expected_proof_hash), "corrected causal replay proof")
    proof = _read_json(proof_path)
    if proof.get("status") != "PASS":
        raise ValueError("Corrected causal A0 replay proof is not PASS")
    replay_hashes = _verify_receipt_hashes(
        proof.get("full_a0_artifact_hashes"), research_root, "corrected causal A0 replay", verified_hashes,
    )
    if any(manifest_hashes.get(path) != digest for path, digest in replay_hashes.items()):
        raise ValueError("Corrected causal replay proof disagrees with the annual model manifest")

    calibration_code = phase_root / "code/causal_calibrate.py"
    year_receipts: dict[int, dict[str, Any]] = {}
    for model_year in UNCERTAINTY_YEARS:
        receipt_relative = f"data/06_models/causal_a0/{model_year}/calibration.json"
        receipt_path = phase_root / receipt_relative
        expected_receipt_path, expected_receipt_sha = _receipt_digest_for_suffix(
            manifest.get("hashes"), receipt_relative, f"causal A0 {model_year} manifest",
        )
        if _research_artifact_path(expected_receipt_path, research_root) != receipt_path.resolve():
            raise ValueError(f"Corrected causal A0 {model_year} receipt path differs from its manifest")
        receipt_sha = _verify_sha256(receipt_path, expected_receipt_sha, f"causal A0 {model_year} calibration receipt")
        record = _read_json(receipt_path)
        if record.get("status") != "COMPLETE" or int(record.get("year", -1)) != model_year:
            raise ValueError(f"Corrected causal A0 {model_year} recipe receipt is incomplete")
        code_sha = str(record.get("code_sha256", ""))
        if len(code_sha) != 64:
            raise ValueError(f"Corrected causal A0 {model_year} receipt omits calibration code hash")
        source_inputs = record.get("inputs")
        if not isinstance(source_inputs, dict) or not source_inputs:
            raise ValueError(f"Corrected causal A0 {model_year} receipt has no source input hashes")
        actual_code_sha = _verify_sha256(calibration_code, code_sha, "causal A0 calibration source")
        bank_receipts = [
            str(digest)
            for path, digest in source_inputs.items()
            if Path(str(path)).as_posix().endswith("/release_corrected_bank/completed.json")
        ]
        if len(bank_receipts) != 1 or bank_receipts[0] != record.get("bank_receipt_sha256"):
            raise ValueError(f"Corrected causal A0 {model_year} does not pin its corrected feature bank receipt")
        def normalize_membership_ids(values: Any, label: str) -> list[str]:
            if not isinstance(values, list):
                raise ValueError(f"Corrected causal A0 {model_year} {label} is not a membership list")
            normalized = [_key(value) for value in values]
            if any(value is None for value in normalized) or len(normalized) != len(set(normalized)):
                raise ValueError(f"Corrected causal A0 {model_year} {label} has invalid identities")
            return [str(value) for value in normalized]

        input_hashes = _verify_receipt_hashes(
            source_inputs, research_root, f"causal A0 {model_year} source inputs", verified_hashes,
        )
        output_hashes = _verify_receipt_hashes(
            record.get("outputs"), research_root, f"causal A0 {model_year} source outputs", verified_hashes,
        )

        membership: dict[str, list[str]] = {}
        for field in ("train_ids", "annual_cal_ids", "mixture_cal_ids", "test_ids"):
            raw_ids = record.get(field)
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError(f"Corrected causal A0 {model_year} receipt has no {field}")
            ids = [_key(value) for value in raw_ids]
            if any(value is None for value in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"Corrected causal A0 {model_year} receipt has invalid {field}")
            membership[field] = [str(value) for value in ids]
        test_ids = set(membership["test_ids"])
        expected_memberships = {
            "train": membership["train_ids"],
            "calibration": membership["annual_cal_ids"],
            "test": membership["test_ids"],
        }
        input_paths = list(record["inputs"])
        neural_receipts = [
            path for path in input_paths
            if "/data/08_reporting/causal_nn/" in Path(str(path)).as_posix()
            and Path(str(path)).name.startswith(f"{model_year}_")
            and Path(str(path)).suffix == ".json"
        ]
        expert_receipts = [
            path for path in input_paths
            if "/data/06_models/causal_a0/" in Path(str(path)).as_posix()
            and "/other/" in Path(str(path)).as_posix()
            and Path(str(path)).suffix == ".json"
        ]
        if len(neural_receipts) != 3 or len(expert_receipts) != 4:
            raise ValueError(f"Corrected causal A0 {model_year} lacks all annual expert membership receipts")
        for source_path in neural_receipts:
            expert = _read_json(_research_artifact_path(source_path, research_root))
            for field, expected_ids in (
                ("train_ids", expected_memberships["train"]),
                ("cal_ids", expected_memberships["calibration"]),
                ("test_ids", expected_memberships["test"]),
            ):
                if normalize_membership_ids(expert.get(field), field) != expected_ids:
                    raise ValueError(f"Corrected causal A0 {model_year} neural receipt disagrees in {field}")
        for source_path in expert_receipts:
            expert = _read_json(_research_artifact_path(source_path, research_root))
            source_memberships = expert.get("membership_ids", {})
            for field, expected_ids in expected_memberships.items():
                if normalize_membership_ids(source_memberships.get(field), field) != expected_ids:
                    raise ValueError(f"Corrected causal A0 {model_year} expert receipt disagrees in {field}")
        mixture_ids = set(membership["mixture_cal_ids"])
        if not mixture_ids.issubset(set(membership["annual_cal_ids"])):
            raise ValueError(f"Corrected causal A0 {model_year} mixture CAL extends beyond annual CAL")
        if set(membership["train_ids"]).intersection(membership["annual_cal_ids"]):
            raise ValueError(f"Corrected causal A0 {model_year} TRAIN and CAL membership overlaps")
        if test_ids.intersection(membership["train_ids"]) or test_ids.intersection(membership["annual_cal_ids"]):
            raise ValueError(f"Corrected causal A0 {model_year} TEST membership overlaps fitting rows")
        year_rows = canonical.loc[canonical["model_year"].eq(model_year)]
        canonical_ids = set(year_rows["golgg_match_id"].astype(str))
        if test_ids != canonical_ids:
            raise ValueError(f"Corrected causal A0 {model_year} TEST IDs do not exactly cover canonical targets")

        prediction_name = f"causal_a0_{model_year}.parquet"
        output_matches = [
            path for path in record["outputs"]
            if Path(str(path)).name == prediction_name
        ]
        if len(output_matches) != 1:
            raise ValueError(f"Corrected causal A0 {model_year} has no unique annual TEST prediction output")
        prediction_path = _research_artifact_path(output_matches[0], research_root)
        expected_prediction_sha = str(record["outputs"][output_matches[0]])
        prediction = pd.read_parquet(
            prediction_path,
            columns=["golgg_match_id", "date", "y", "best_of", "team1_id", "team2_id", "p_full"],
        )
        prediction["golgg_match_id"] = _key_series(prediction["golgg_match_id"])
        if prediction["golgg_match_id"].isna().any() or prediction["golgg_match_id"].duplicated().any():
            raise ValueError(f"Corrected causal A0 {model_year} output has invalid fixture identities")
        prediction_ids = set(prediction["golgg_match_id"].astype(str))
        if prediction_ids != test_ids:
            raise ValueError(f"Corrected causal A0 {model_year} output differs from its TEST receipt IDs")
        expected = year_rows.set_index("golgg_match_id").loc[prediction["golgg_match_id"]].reset_index(drop=True)
        for column in ("date", "y", "best_of"):
            left = _date_strings(prediction[column], f"causal A0 {model_year} {column}") if column == "date" else prediction[column].astype("string")
            right = expected[column].astype("string")
            if not left.reset_index(drop=True).eq(right.reset_index(drop=True)).all():
                raise ValueError(f"Corrected causal A0 {model_year} output disagrees with canonical {column}")
        for column in ("team1_id", "team2_id"):
            if column in prediction:
                left = _key_series(prediction[column]).reset_index(drop=True)
                right = expected[column].astype("string").reset_index(drop=True)
                if not left.eq(right).all():
                    raise ValueError(f"Corrected causal A0 {model_year} output disagrees with canonical {column}")
        annual_p = pd.to_numeric(prediction["p_full"], errors="coerce").to_numpy(dtype=float)
        canonical_p = expected["p_s"].to_numpy(dtype=float)
        if not np.isfinite(annual_p).all() or not np.allclose(annual_p, canonical_p, rtol=0.0, atol=1e-12):
            raise ValueError(f"Corrected causal A0 {model_year} TEST predictions differ from canonical p_full")
        year_receipts[model_year] = {
            "status": record["status"],
            "recipe_status": "COMPLETE-RECIPE",
            "is_oof": True,
            "test_rows": len(test_ids),
            "test_ids_sha256": hashlib.sha256(
                json.dumps(sorted(test_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "calibration_receipt": {
                "path": _display_path(receipt_path, research_root),
                "sha256": receipt_sha,
            },
            "calibration_code": {
                "path": _display_path(calibration_code, research_root),
                "sha256": actual_code_sha,
            },
            "annual_test_predictions": {
                "path": _display_path(prediction_path, research_root),
                "sha256": expected_prediction_sha,
            },
            "source_input_hashes": input_hashes,
            "source_output_hashes": output_hashes,
            "membership_counts": {field: len(ids) for field, ids in membership.items()},
            "expert_membership_receipts_verified": {
                "neural_receipts": len(neural_receipts),
                "other_expert_receipts": len(expert_receipts),
            },
            "chronology_status": "CORRECTED_CAUSAL_A0_PAST_ONLY_RECEIPT_VERIFIED",
        }
    audit_record = {
        "path": _display_path(audit_path, research_root),
        "sha256": audit_sha,
        "status": audit["status"],
        "scope": "PRE_CORRECTION_AUDIT_NOT_APPLICABLE_TO_CORRECTED_CAUSAL_A0",
        "original_training_audit_note": audit_scope,
        "late_ids": old_audit_ids,
        "slice_receipt_path": _display_path(slice_receipt_path, research_root),
        "slice_receipt_sha256": _sha256(slice_receipt_path),
        "availability_scope_report_path": _display_path(audit_report_path, research_root),
        "availability_scope_report_sha256": audit_report_sha,
        "corrected_causal_replay_proof_path": _display_path(proof_path, research_root),
        "corrected_causal_replay_proof_sha256": proof_sha,
        "corrected_causal_model_manifest_path": _display_path(manifest_path, research_root),
        "corrected_causal_model_manifest_sha256": _sha256(manifest_path),
    }
    return {"years": year_receipts, "pre_correction_audit": audit_record}


def _attach_oof_provenance(
    canonical: pd.DataFrame,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    result = canonical.copy()
    year_records = provenance["years"]
    result["scenario_fit_origin"] = pd.to_datetime(
        result["model_year"].astype(str) + "-01-01", errors="raise",
    ).dt.tz_localize("UTC")
    result["scenario_prediction_at"] = pd.to_datetime(
        result["date"], errors="raise",
    ).dt.normalize().dt.tz_localize("UTC")
    result["is_oof"] = result["model_year"].map(
        {year: record["is_oof"] for year, record in year_records.items()}
    ).astype("boolean")
    result["recipe_status"] = result["model_year"].map(
        {year: record["recipe_status"] for year, record in year_records.items()}
    ).astype("string")
    result["prediction_role"] = "oof_prediction"
    result["source_prediction_status"] = "PINNED_ANNUAL_TEST_OOF_MEMBERSHIP_VERIFIED"
    result["source_chronology_status"] = result["model_year"].map(
        {year: record["chronology_status"] for year, record in year_records.items()}
    ).astype("string")
    audit = provenance["pre_correction_audit"]
    result["source_chronology_audit_path"] = audit["path"]
    result["source_chronology_audit_sha256"] = audit["sha256"]
    result["source_chronology_audit_status"] = "FAIL_PRE_CORRECTION_NOT_APPLICABLE"
    result["source_chronology_audit_late_ids"] = "|".join(audit["late_ids"])
    result["source_receipt_sha256"] = result["model_year"].map(
        {year: record["calibration_receipt"]["sha256"] for year, record in year_records.items()}
    ).astype("string")
    result["source_prediction_sha256"] = result["model_year"].map(
        {year: record["annual_test_predictions"]["sha256"] for year, record in year_records.items()}
    ).astype("string")
    result["scenario_timestamp_assumption"] = (
        "annual TEST membership and source recipe verified from pinned corrected causal A0 receipts; "
        "scenario fit origin is model-year Jan 1 and scenario prediction time is target-day UTC midnight; "
        "actual creation/publication/label timestamps remain unknown, with label release modeled at next-day "
        "UTC midnight after result_day"
    )
    return result


def _normalize_canonical(frame: pd.DataFrame, expected_rows: int | None) -> pd.DataFrame:
    required = {
        "golgg_match_id", "team1_id", "team2_id", "date", "result_day",
        "feature_source_max_day", "best_of", "y", "p_full",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Canonical benchmark is missing columns: {missing}")
    result = frame.copy()
    result["golgg_match_id"] = _key_series(result["golgg_match_id"])
    if result["golgg_match_id"].isna().any() or result["golgg_match_id"].duplicated().any():
        raise ValueError("Canonical benchmark has missing or duplicate golgg_match_id")
    for column in ("team1_id", "team2_id"):
        result[column] = _key_series(result[column])
        if result[column].isna().any():
            raise ValueError(f"Canonical benchmark has missing {column}")
    if result["team1_id"].eq(result["team2_id"]).any():
        raise ValueError("Canonical benchmark contains identical team identities")
    result["date"] = _date_strings(result["date"], "canonical benchmark")
    result["result_day"] = _date_strings(result["result_day"], "canonical result release")
    result["feature_source_max_day"] = _date_strings(result["feature_source_max_day"], "canonical feature history")
    if not result["feature_source_max_day"].lt(result["date"]).all():
        raise ValueError("Canonical feature history must precede each target day")
    if not result["result_day"].ge(result["date"]).all():
        raise ValueError("Canonical result release precedes its target day")
    labels = pd.to_numeric(result["y"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Canonical benchmark has an invalid outcome")
    result["y"] = labels.astype("int8")
    best_of = pd.to_numeric(result["best_of"], errors="coerce")
    if best_of.isna().any() or not best_of.isin([1, 3, 5]).all():
        raise ValueError("Canonical benchmark has an invalid best_of value")
    result["best_of"] = best_of.astype("int8")
    _probabilities(result, ("p_full", "market_open", "market_close"))
    result["p_s"] = pd.to_numeric(result["p_full"], errors="coerce")
    if result["p_s"].isna().any():
        raise ValueError("Canonical benchmark must retain a native p_full for every target row")
    result["prediction_at"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["fit_origin"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["label_available_at"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["scenario_label_available_at"] = _daily_release_scenario(result["result_day"])
    result["is_oof"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["recipe_status"] = "OOF_STATUS_UNATTESTED"
    result["prediction_role"] = "canonical_native_baseline"
    result["source_prediction_status"] = "PINNED_CANONICAL_ANNUAL_BASELINE_OOF_UNATTESTED"
    result["information_status"] = "historical_scenario"
    result["scenario_timestamp_assumption"] = (
        "canonical annual p_full creation, fit origin, and actual label release are unrecorded; "
        "result_day is treated as released at next-day UTC midnight only as an explicit scenario"
    )
    result["model_year"] = pd.to_datetime(result["date"], errors="raise").dt.year.astype(int)
    if expected_rows is not None and len(result) != int(expected_rows):
        raise ValueError(f"Canonical manifest row count mismatch: expected {expected_rows}, got {len(result)}")
    return result.reset_index(drop=True)


def _valid_decimal_odds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric) & numeric.gt(1.0))


def _normalize_timing(values: pd.Series) -> pd.Series:
    lookup = {
        "OPEN": "OPEN", "OPENING": "OPEN", "24H": "24h", "24HR": "24h",
        "48H": "48h", "48HR": "48h", "CLOSE": "CLOSE", "CLOSING": "CLOSE",
    }
    normalized = values.astype("string").str.strip().str.upper()
    return normalized.map(lookup).astype("string")


def _stable_text(values: pd.Series) -> pd.Series:
    return values.map(_key).astype("string").fillna("")


def _collapse_archive_quotes(quotes: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "golgg_match_id", "source_row", "bookmaker", "timing", "odds_a", "odds_b",
        "market", "team1_id", "team2_id",
    }
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise ValueError(f"Corrected archival quotes are missing columns: {missing}")
    work = quotes.copy()
    for column in ("golgg_match_id", "team1_id", "team2_id", "bookmaker", "source_row"):
        work[column] = _key_series(work[column])
    work["timing"] = _normalize_timing(work["timing"])
    work["odds_a"] = _valid_decimal_odds(work["odds_a"])
    work["odds_b"] = _valid_decimal_odds(work["odds_b"])
    work["market"] = work["market"].astype("string").str.strip()
    canonical_ids = set(canonical["golgg_match_id"].astype(str))
    matched = work["golgg_match_id"].isin(canonical_ids)
    identity_by_id = canonical.set_index("golgg_match_id")[["team1_id", "team2_id"]]
    joined = work.loc[matched].join(identity_by_id, on="golgg_match_id", rsuffix="_canonical")
    identity_conflict = (
        joined["team1_id"].ne(joined["team1_id_canonical"])
        | joined["team2_id"].ne(joined["team2_id_canonical"])
    )
    if identity_conflict.any():
        bad = joined.loc[identity_conflict, "golgg_match_id"].drop_duplicates().tolist()[:10]
        raise ValueError(f"Corrected archival quote team identity mismatch: {bad}")

    valid = matched & work["timing"].notna() & work["bookmaker"].notna()
    valid &= work["source_row"].notna() & work["market"].notna()
    valid &= work["odds_a"].notna() & work["odds_b"].notna()
    quotes_invalid = int((matched & ~valid).sum())
    work = work.loc[valid].copy()
    if work.empty:
        coverage = {
            "source_rows": int(len(quotes)), "matched_rows": int(matched.sum()),
            "eligible_rows": 0, "invalid_matched_rows": quotes_invalid,
            "unmatched_rows": int((~matched).sum()), "duplicate_rows_collapsed": 0,
        }
        return work.assign(q_book=pd.Series(dtype=float)), coverage

    # Same source record repeated verbatim is harmless; conflicting duplicate
    # source identities are an integrity error, not an arbitrary tie-break.
    source_key = ["golgg_match_id", "timing", "source_row", "bookmaker"]
    duplicate_identity = work.duplicated(source_key, keep=False)
    if duplicate_identity.any():
        comparisons = ["market", "team1_id", "team2_id", "odds_a", "odds_b"]
        distinct = work.loc[duplicate_identity].groupby(source_key, dropna=False)[comparisons].nunique(dropna=False)
        if distinct.gt(1).any(axis=None):
            raise ValueError("Conflicting corrected archive rows share a source_row identity")
    before_exact = len(work)
    work = work.drop_duplicates(source_key + ["market", "odds_a", "odds_b"], keep="first")
    exact_collapsed = before_exact - len(work)

    book_key = ["golgg_match_id", "timing", "bookmaker"]
    per_book = work.groupby(book_key, dropna=False, sort=False)
    ambiguous = per_book[["odds_a", "odds_b", "market", "team1_id", "team2_id"]].nunique(dropna=False)
    if ambiguous.gt(1).any(axis=None):
        raise ValueError("Corrected archive has conflicting prices or identities for one bookmaker/timing")
    before_book_dedupe = len(work)
    work = work.sort_values([*book_key, "source_row"], kind="stable").drop_duplicates(book_key, keep="first")
    per_book_collapsed = before_book_dedupe - len(work)
    work["q_book"] = work["odds_b"] / (work["odds_a"] + work["odds_b"])
    if not np.isfinite(work["q_book"].to_numpy(dtype=float)).all() or not work["q_book"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("Corrected archive contains invalid within-book no-vig probability")
    coverage = {
        "source_rows": int(len(quotes)), "matched_rows": int(matched.sum()),
        "eligible_rows": int(len(work)), "invalid_matched_rows": quotes_invalid,
        "unmatched_rows": int((~matched).sum()), "duplicate_source_rows_collapsed": int(exact_collapsed),
        "duplicate_book_timing_rows_collapsed": int(per_book_collapsed),
        "rows_by_timing": {str(key): int(value) for key, value in work["timing"].value_counts(dropna=False).items()},
        "books_by_timing": {str(key): int(value) for key, value in work.groupby("timing")["bookmaker"].nunique().items()},
        "one_vote_per_book_verified": True,
    }
    return work.reset_index(drop=True), coverage


def _archive_market_tables(quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if quotes.empty:
        ids = pd.DataFrame(columns=["golgg_match_id", "q", "book_count"])
        offers = pd.DataFrame(columns=["golgg_match_id", "odds_a", "bookmaker_a", "quote_id_a", "odds_b", "bookmaker_b", "quote_id_b"])
        closes = pd.DataFrame(columns=["golgg_match_id", "close_odds_a", "close_odds_b", "close_bookmaker_a", "close_bookmaker_b", "close_quote_id_a", "close_quote_id_b"])
        return ids, offers, closes
    means = quotes.groupby(["golgg_match_id", "timing"], as_index=False, sort=False).agg(
        q=("q_book", "mean"), book_count=("bookmaker", "nunique"),
    )
    opened = quotes.loc[quotes["timing"].eq("OPEN")].copy()
    if opened.empty:
        offers = pd.DataFrame(columns=["golgg_match_id", "odds_a", "bookmaker_a", "quote_id_a", "odds_b", "bookmaker_b", "quote_id_b"])
    else:
        by_a = opened.sort_values(
            ["golgg_match_id", "odds_a", "bookmaker", "source_row"],
            ascending=[True, False, True, True], kind="stable",
        ).drop_duplicates("golgg_match_id", keep="first")[["golgg_match_id", "odds_a", "bookmaker", "source_row"]]
        by_a = by_a.rename(columns={"bookmaker": "bookmaker_a", "source_row": "quote_id_a"})
        by_b = opened.sort_values(
            ["golgg_match_id", "odds_b", "bookmaker", "source_row"],
            ascending=[True, False, True, True], kind="stable",
        ).drop_duplicates("golgg_match_id", keep="first")[["golgg_match_id", "odds_b", "bookmaker", "source_row"]]
        by_b = by_b.rename(columns={"bookmaker": "bookmaker_b", "source_row": "quote_id_b"})
        offers = by_a.merge(by_b, on="golgg_match_id", how="outer", validate="one_to_one")

    close = quotes.loc[quotes["timing"].eq("CLOSE")]
    open_offers = offers
    close_rows: list[dict[str, Any]] = []
    if not close.empty and not open_offers.empty:
        close_lookup = close.set_index(["golgg_match_id", "bookmaker"], drop=False)
        for row in open_offers.itertuples(index=False):
            record: dict[str, Any] = {"golgg_match_id": row.golgg_match_id}
            for side, odds_col, book_col, id_col in (
                ("a", "odds_a", "bookmaker_a", "quote_id_a"),
                ("b", "odds_b", "bookmaker_b", "quote_id_b"),
            ):
                bookmaker = getattr(row, book_col)
                if pd.isna(bookmaker) or (row.golgg_match_id, bookmaker) not in close_lookup.index:
                    record[f"close_odds_{side}"] = np.nan
                    record[f"close_bookmaker_{side}"] = pd.NA
                    record[f"close_quote_id_{side}"] = pd.NA
                    continue
                close_row = close_lookup.loc[(row.golgg_match_id, bookmaker)]
                if isinstance(close_row, pd.DataFrame):
                    raise ValueError("Archive close lookup is not unique per fixture/book")
                record[f"close_odds_{side}"] = float(close_row[odds_col])
                record[f"close_bookmaker_{side}"] = bookmaker
                record[f"close_quote_id_{side}"] = close_row["source_row"]
            close_rows.append(record)
    closes = pd.DataFrame(close_rows)
    if closes.empty:
        closes = pd.DataFrame(columns=["golgg_match_id", "close_odds_a", "close_odds_b", "close_bookmaker_a", "close_bookmaker_b", "close_quote_id_a", "close_quote_id_b"])
    return means, offers, closes


def _daily_release_scenario(day_values: pd.Series) -> pd.Series:
    """Represent daily result release as next-day UTC midnight, never as actual time."""
    dates = pd.to_datetime(day_values.astype("string").str.slice(0, 10), errors="coerce")
    result = dates.dt.normalize() + pd.Timedelta(days=1)
    return pd.Series(result.dt.tz_localize("UTC"), index=day_values.index, dtype="datetime64[ns, UTC]")


def _archival_opportunities(canonical: pd.DataFrame, archive: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    market, offers, closes = _archive_market_tables(archive)
    opened = market.loc[market["timing"].eq("OPEN")].drop(columns="timing").rename(
        columns={"q": "archive_open_q", "book_count": "archive_open_book_count"}
    ) if not market.empty else pd.DataFrame(columns=["golgg_match_id", "archive_open_q", "archive_open_book_count"])
    closed_market = market.loc[market["timing"].eq("CLOSE")].drop(columns="timing").rename(
        columns={"q": "archive_close_q", "book_count": "archive_close_book_count"}
    ) if not market.empty else pd.DataFrame(columns=["golgg_match_id", "archive_close_q", "archive_close_book_count"])
    base = canonical.merge(opened, on="golgg_match_id", how="left", validate="one_to_one")
    base = base.merge(closed_market, on="golgg_match_id", how="left", validate="one_to_one")
    base = base.merge(offers, on="golgg_match_id", how="left", validate="one_to_one")
    base = base.merge(closes, on="golgg_match_id", how="left", validate="one_to_one")
    # The archived paired file is a pinned no-vig market source. Quote-derived
    # one-vote-per-book values take precedence; missing quote rows stay visible.
    market_open = (
        pd.to_numeric(base["market_open"], errors="coerce")
        if "market_open" in base else pd.Series(np.nan, index=base.index, dtype=float)
    )
    market_close = (
        pd.to_numeric(base["market_close"], errors="coerce")
        if "market_close" in base else pd.Series(np.nan, index=base.index, dtype=float)
    )
    base["q"] = base["archive_open_q"].combine_first(market_open)
    base["q_source"] = np.select(
        [base["archive_open_q"].notna(), market_open.notna()],
        ["corrected_quote_one_vote_per_book", "canonical_market_open"],
        default="unavailable",
    )
    base["close_q"] = base["archive_close_q"].combine_first(market_close)
    base["close_q_source"] = np.select(
        [base["archive_close_q"].notna(), market_close.notna()],
        ["corrected_quote_one_vote_per_book", "canonical_market_close"],
        default="unavailable",
    )
    base["archive_open_market_alignment_error"] = (base["archive_open_q"] - market_open).abs()
    base["archive_close_market_alignment_error"] = (base["archive_close_q"] - market_close).abs()

    rows = pd.DataFrame(index=base.index)
    rows["row_id"] = "archival_market:" + base["golgg_match_id"].astype(str) + ":OPEN"
    rows["golgg_match_id"] = base["golgg_match_id"]
    rows["cohort"] = "archival_market"
    rows["horizon"] = "OPEN"
    rows["date"] = base["date"]
    rows["decision_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["scenario_decision_at"] = base["scenario_prediction_at"]
    rows["start_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["label_available_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["scenario_label_available_at"] = base["scenario_label_available_at"]
    rows["y"] = base["y"]
    rows["p_s"] = base["p_s"]
    rows["p_s_annual"] = base["p_s"]
    for column in (
        "prediction_at", "fit_origin", "scenario_prediction_at", "scenario_fit_origin",
        "is_oof", "recipe_status", "prediction_role", "source_prediction_status",
        "source_chronology_status", "source_chronology_audit_path",
        "source_chronology_audit_sha256", "source_chronology_audit_status",
        "source_chronology_audit_late_ids", "source_receipt_sha256",
        "source_prediction_sha256",
    ):
        rows[column] = base[column]
    rows["q"] = base["q"]
    rows["q_source"] = base["q_source"]
    rows["archive_open_market_alignment_error"] = base["archive_open_market_alignment_error"]
    rows["archive_close_market_alignment_error"] = base["archive_close_market_alignment_error"]
    rows["archive_open_book_count"] = base["archive_open_book_count"]
    rows["archive_close_book_count"] = base["archive_close_book_count"]
    for source, target in (("odds_a", "odds_a"), ("odds_b", "odds_b"),
                           ("bookmaker_a", "bookmaker_a"), ("bookmaker_b", "bookmaker_b"),
                           ("quote_id_a", "quote_id_a"), ("quote_id_b", "quote_id_b")):
        rows[target] = base[source] if source in base else np.nan
    rows["quote_at_a"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["quote_at_b"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    for column in ("close_odds_a", "close_odds_b", "close_bookmaker_a", "close_bookmaker_b", "close_quote_id_a", "close_quote_id_b"):
        rows[column] = base[column] if column in base else pd.NA
    rows["close_q"] = base["close_q"]
    rows["close_available_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["scenario_close_available_at"] = base["scenario_label_available_at"].where(rows["close_q"].notna())
    rows["best_of"] = base["best_of"]
    rows["league"] = base["league"] if "league" in base else pd.NA
    rows["information_status"] = "historical_scenario"
    rows["strict_prior_day"] = False
    rows["is_common"] = False
    rows["feature_available_at"] = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    rows["scenario_timestamp_assumption"] = (
        "archived OPEN is assumed available at target-day start only for historical reconstruction; "
        "CLOSE is assumed available at the modeled result-release time; corrected causal A0 annual "
        "TEST membership is receipt-verified; actual quote, fit, prediction and result publication "
        "timestamps remain unknown, so equal-time availability and executable prices are not certified"
    )
    rows["same_book_close_a_available"] = rows["close_odds_a"].notna()
    rows["same_book_close_b_available"] = rows["close_odds_b"].notna()
    result = rows.reset_index(drop=True)
    metrics = {
        "canonical_rows_retained": int(len(result)),
        "open_market_consensus_rows": int(result["q"].notna().sum()),
        "open_quote_consensus_rows": int(result["q_source"].eq("corrected_quote_one_vote_per_book").sum()),
        "canonical_market_open_fallback_rows": int(result["q_source"].eq("canonical_market_open").sum()),
        "open_market_consensus_unavailable_rows": int(result["q_source"].eq("unavailable").sum()),
        "open_executable_side_a_rows": int(result["odds_a"].notna().sum()),
        "open_executable_side_b_rows": int(result["odds_b"].notna().sum()),
        "same_book_close_side_a_rows": int(result["same_book_close_a_available"].sum()),
        "same_book_close_side_b_rows": int(result["same_book_close_b_available"].sum()),
        "max_open_market_alignment_error": float(result["archive_open_market_alignment_error"].max()) if result["archive_open_market_alignment_error"].notna().any() else None,
        "max_close_market_alignment_error": float(result["archive_close_market_alignment_error"].max()) if result["archive_close_market_alignment_error"].notna().any() else None,
        "actual_quote_times_known": 0,
        "actual_result_release_times_known": 0,
        "scenario_result_release_times_known": int(result["scenario_label_available_at"].notna().sum()),
    }
    return result, metrics


def _panel_horizon(value: Any) -> str | None:
    return _normalize_timing(pd.Series([value])).iloc[0]


def _production_receipt_paths(panel_path: Path, quote_path: Path) -> dict[str, Any]:
    receipt_path = PROJECT_ROOT / "data/08_reporting/production_market_calibration_ev5_20260926_210430/input_receipt.json"
    receipt = _read_json(receipt_path)
    if receipt.get("panel_sha256") != PRODUCTION_PANEL_SHA256:
        raise ValueError("Production source receipt panel hash differs from frozen input")
    if receipt.get("aligned_quotes_sha256") != PRODUCTION_QUOTES_SHA256:
        raise ValueError("Production source receipt aligned-quotes hash differs from frozen input")
    actual_panel_hash = _verify_sha256(panel_path, PRODUCTION_PANEL_SHA256, "production audit panel")
    actual_quote_hash = _verify_sha256(quote_path, PRODUCTION_QUOTES_SHA256, "production aligned quotes")
    return {
        "receipt_path": str(receipt_path.relative_to(PROJECT_ROOT)),
        "panel_sha256": actual_panel_hash,
        "aligned_quotes_sha256": actual_quote_hash,
        "source_sha256": receipt.get("source_sha256", {}),
    }


def _production_quote_ledger(
    quotes: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"canonical_match_id", "horizon", "bookmaker", "quote_id_string", "scraped_at", "odds_team1", "odds_team2", "p_market_quote"}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise ValueError(f"Production aligned quotes are missing columns: {missing}")
    work = quotes.rename(columns={"quote_id_string": "quote_id"}).copy()
    work["canonical_match_id"] = _key_series(work["canonical_match_id"])
    work["bookmaker"] = _key_series(work["bookmaker"])
    work["quote_id"] = _key_series(work["quote_id"])
    work["horizon"] = work["horizon"].map(_panel_horizon).astype("string")
    work["scraped_at"] = _utc_series(work["scraped_at"], label="production quote scraped_at")
    work["odds_team1"] = _valid_decimal_odds(work["odds_team1"])
    work["odds_team2"] = _valid_decimal_odds(work["odds_team2"])
    work["p_market_quote"] = pd.to_numeric(work["p_market_quote"], errors="coerce")
    valid = work["canonical_match_id"].notna() & work["bookmaker"].notna() & work["horizon"].notna() & work["quote_id"].notna()
    valid &= work["scraped_at"].notna() & work["odds_team1"].notna() & work["odds_team2"].notna()
    valid &= np.isfinite(work["p_market_quote"]) & work["p_market_quote"].between(0.0, 1.0)
    source_rows = len(work)
    valid_source_rows = int(valid.sum())
    invalid_source_rows = source_rows - valid_source_rows
    work = work.loc[valid].copy()
    derived = work["odds_team2"] / (work["odds_team1"] + work["odds_team2"])
    errors = (derived - work["p_market_quote"]).abs()
    if len(errors) and float(errors.max()) > 1e-10:
        raise ValueError("Production quote p_market_quote disagrees with within-book no-vig odds")
    work["no_vig_alignment_error"] = errors

    panel_keys = panel[["canonical_match_id", "golgg_match_id", "horizon", "cutoff", "start_dt"]].copy()
    panel_keys["canonical_match_id"] = _key_series(panel_keys["canonical_match_id"])
    panel_keys["horizon"] = panel_keys["horizon"].map(_panel_horizon).astype("string")
    panel_key = ["canonical_match_id", "horizon"]
    if panel_keys.duplicated(panel_key).any():
        raise ValueError("Production panel has duplicate canonical_match_id/horizon identities")
    matched = work.merge(panel_keys, on=panel_key, how="inner", validate="many_to_one", suffixes=("_quote", ""))
    unmatched_panel_rows = len(work) - len(matched)
    work = matched
    start = _utc_series(work["start_dt"], label="production match start", reject_naive=False)
    cutoff = _utc_series(work["cutoff"], label="production horizon cutoff", reject_naive=False)
    if "cutoff_quote" in work:
        companion_cutoff = _utc_series(work["cutoff_quote"], label="companion horizon cutoff", reject_naive=False)
        if not companion_cutoff.eq(cutoff).all():
            raise ValueError("Companion quote cutoff disagrees with the production panel")
    before_start = work["scraped_at"].lt(start)
    horizon_valid = before_start.copy()
    open_mask = work["horizon"].eq("OPEN")
    close_mask = work["horizon"].eq("CLOSE")
    timed_horizon = work["horizon"].isin(["24h", "48h"])
    horizon_valid.loc[open_mask] &= work.loc[open_mask, "scraped_at"].le(start.loc[open_mask])
    close_lower = start - pd.Timedelta(hours=2)
    horizon_valid.loc[close_mask] &= work.loc[close_mask, "scraped_at"].ge(close_lower.loc[close_mask])
    horizon_valid.loc[timed_horizon] &= work.loc[timed_horizon, "scraped_at"].le(cutoff.loc[timed_horizon])
    horizon_valid.loc[timed_horizon] &= work.loc[timed_horizon, "scraped_at"].ge(cutoff.loc[timed_horizon] - pd.Timedelta(hours=24))
    wrong_time_rows = int((~horizon_valid).sum())
    work = work.loc[horizon_valid].copy()
    horizon_eligible_rows = len(work)
    if work.empty:
        return work, {
            "source_rows": int(source_rows), "valid_source_rows": valid_source_rows,
            "invalid_source_rows": int(invalid_source_rows),
            "unmatched_panel_rows": int(unmatched_panel_rows),
            "horizon_eligible_rows": 0, "selected_book_horizon_rows": 0,
            "bad_time_rows": wrong_time_rows, "one_vote_per_book_verified": True,
            "max_no_vig_probability_alignment_error": float(errors.max()) if len(errors) else None,
        }

    work["_quote_stable"] = _stable_text(work["quote_id"])
    # Keep one observation per book and horizon: first opening quote, latest
    # cutoff-window quote for 24/48h, and last valid same-book close.
    work = work.sort_values(
        ["canonical_match_id", "horizon", "bookmaker", "scraped_at", "_quote_stable"],
        ascending=[True, True, True, True, True], kind="stable",
    )
    if "OPEN" in set(work["horizon"]):
        earliest = work.loc[work["horizon"].eq("OPEN")].groupby("canonical_match_id")["scraped_at"].transform("min")
        open_rows = work.loc[work["horizon"].eq("OPEN")]
        keep_open = open_rows.loc[open_rows["scraped_at"].eq(earliest)].copy()
    else:
        keep_open = work.iloc[0:0].copy()
    later = work.loc[work["horizon"].ne("OPEN")].sort_values(
        ["canonical_match_id", "horizon", "bookmaker", "scraped_at", "_quote_stable"],
        ascending=[True, True, True, False, True], kind="stable",
    ).drop_duplicates(["canonical_match_id", "horizon", "bookmaker"], keep="first")
    if not keep_open.empty:
        keep_open = keep_open.drop_duplicates(["canonical_match_id", "horizon", "bookmaker"], keep="first")
    selected = pd.concat([keep_open, later], ignore_index=True, sort=False)
    # Identical snapshots are one vote; a different second snapshot was resolved
    # by the declared first/last rule above, never counted as another bookmaker.
    selected = selected.drop(columns="_quote_stable", errors="ignore")
    coverage = {
        "source_rows": int(source_rows),
        "valid_source_rows": int(valid_source_rows),
        "invalid_source_rows": int(invalid_source_rows),
        "unmatched_panel_rows": int(unmatched_panel_rows),
        "horizon_eligible_rows": int(horizon_eligible_rows),
        "valid_quote_rows": int(horizon_eligible_rows),
        "selected_book_horizon_rows": int(len(selected)),
        "bad_time_rows": int(wrong_time_rows),
        "book_horizon_groups": int(selected[["canonical_match_id", "horizon", "bookmaker"]].drop_duplicates().shape[0]),
        "one_vote_per_book_verified": True,
        "max_no_vig_probability_alignment_error": float(errors.max()) if len(errors) else None,
    }
    return selected, coverage


def _production_opportunities(
    panel: pd.DataFrame,
    quotes: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "canonical_match_id", "golgg_match_id", "date", "y", "p_a0", "p_market", "cutoff",
        "start_dt", "league", "best_of", "horizon", "is_common",
        "time_compatible_strict_prior_day", "best_odds_team1", "best_odds_team2",
        "best_bookmaker_team1", "best_bookmaker_team2", "best_quote_id_team1",
        "best_quote_id_team2", "best_quote_time_team1", "best_quote_time_team2",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Production audit panel is missing columns: {missing}")
    source = panel.copy()
    source["canonical_match_id"] = _key_series(source["canonical_match_id"])
    source["golgg_match_id"] = _key_series(source["golgg_match_id"])
    source["horizon"] = source["horizon"].map(_panel_horizon).astype("string")
    source["date"] = _date_strings(source["date"], "production panel")
    key = ["golgg_match_id", "horizon"]
    if source[key].isna().any().any() or source.duplicated(key).any():
        raise ValueError("Production panel has missing or duplicate fixture/horizon identities")
    if source["canonical_match_id"].isna().any():
        raise ValueError("Production panel contains an unresolved canonical_match_id")
    numeric = ["y", "p_a0", "p_market", "best_of", "best_odds_team1", "best_odds_team2"]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source["y"].notna().any() and not source.loc[source["y"].notna(), "y"].isin([0, 1]).all():
        raise ValueError("Production panel has invalid outcomes")
    _probabilities(source, ("p_a0", "p_market"))
    for column in ("best_odds_team1", "best_odds_team2"):
        invalid = source[column].notna() & (~np.isfinite(source[column]) | source[column].le(1.0))
        if invalid.any():
            raise ValueError(f"Production panel contains invalid executable odds in {column}")

    canonical_index = canonical.set_index("golgg_match_id", drop=False)
    unmatched = ~source["golgg_match_id"].isin(canonical_index.index)
    if unmatched.any():
        raise ValueError(f"Production panel fixture absent from canonical targets: {source.loc[unmatched, 'golgg_match_id'].head(10).tolist()}")
    canonical_rows = canonical_index.loc[source["golgg_match_id"]].reset_index(drop=True)
    for column in ("date", "y", "best_of"):
        panel_values = source[column].to_numpy()
        canonical_values = canonical_rows[column].to_numpy()
        if not np.array_equal(panel_values, canonical_values):
            raise ValueError(f"Production panel disagrees with canonical {column} identity")
    panel_p = source["p_a0"].to_numpy(dtype=float)
    canonical_p = canonical_rows["p_s"].to_numpy(dtype=float)
    if not np.isfinite(panel_p).all() or not np.allclose(panel_p, canonical_p, rtol=0.0, atol=1e-12):
        raise ValueError("Production panel p_a0 disagrees with canonical p_full orientation")
    source["p_s_annual"] = canonical_p
    for column in (
        "prediction_at", "fit_origin", "label_available_at",
        "scenario_prediction_at", "scenario_fit_origin", "scenario_label_available_at",
        "is_oof", "recipe_status", "prediction_role", "source_prediction_status",
        "source_chronology_status", "source_chronology_audit_path",
        "source_chronology_audit_sha256", "source_chronology_audit_status",
        "source_chronology_audit_late_ids", "source_receipt_sha256",
        "source_prediction_sha256",
    ):
        source[column] = canonical_rows[column].to_numpy()
    source["result_day"] = canonical_rows["result_day"].to_numpy()
    source["canonical_market_open"] = canonical_rows.get("market_open", pd.Series(np.nan, index=source.index)).to_numpy()
    source["canonical_market_close"] = canonical_rows.get("market_close", pd.Series(np.nan, index=source.index)).to_numpy()

    selected_quotes, quote_coverage = _production_quote_ledger(quotes, source)
    if not selected_quotes.empty:
        selected_quotes["_book_key"] = _key_series(selected_quotes["bookmaker"])
    quote_groups = {
        (str(mid), str(horizon)): part.copy()
        for (mid, horizon), part in selected_quotes.groupby(["canonical_match_id", "horizon"], sort=False)
    } if not selected_quotes.empty else {}
    row_records: list[dict[str, Any]] = []
    mismatch_counts: dict[str, int] = {"panel_offer_missing_from_companion": 0, "panel_consensus_alignment_mismatch": 0}

    for row in source.itertuples(index=False):
        horizon = str(row.horizon)
        cutoff = _utc_timestamp(row.cutoff, label="production cutoff", reject_naive=False)
        start_at = _utc_timestamp(row.start_dt, label="production start", reject_naive=False)
        quote_rows = quote_groups.get((str(row.canonical_match_id), horizon), pd.DataFrame())
        q_books = quote_rows["bookmaker"].nunique() if not quote_rows.empty else 0
        q = float(quote_rows["p_market_quote"].mean()) if q_books and not quote_rows.empty else (
            float(row.p_market) if pd.notna(row.p_market) else np.nan
        )
        q_source = "aligned_quotes_one_vote_per_book" if q_books else "audit_panel_p_market_no_quote_rows"
        panel_q_error = abs(q - float(row.p_market)) if pd.notna(row.p_market) and np.isfinite(q) else np.nan
        if np.isfinite(panel_q_error) and panel_q_error > 1e-8:
            mismatch_counts["panel_consensus_alignment_mismatch"] += 1

        best = {}
        close_rows = quote_groups.get((str(row.canonical_match_id), "CLOSE"), pd.DataFrame())
        for side, odds_col, book_col, id_col, time_col, quote_odds_col in (
            ("a", "best_odds_team1", "best_bookmaker_team1", "best_quote_id_team1", "best_quote_time_team1", "odds_team1"),
            ("b", "best_odds_team2", "best_bookmaker_team2", "best_quote_id_team2", "best_quote_time_team2", "odds_team2"),
        ):
            price = getattr(row, odds_col)
            bookmaker = _key(getattr(row, book_col))
            quote_id = _key(getattr(row, id_col))
            quote_time = _utc_timestamp(getattr(row, time_col), label=time_col, reject_naive=False)
            companion_hit = pd.DataFrame()
            if not quote_rows.empty and bookmaker is not None:
                mask = _key_series(quote_rows["bookmaker"]).eq(bookmaker)
                if pd.notna(price):
                    mask &= pd.to_numeric(quote_rows[quote_odds_col], errors="coerce").sub(float(price)).abs().le(1e-10)
                if quote_id is not None:
                    mask &= _key_series(quote_rows["quote_id"]).eq(quote_id)
                companion_hit = quote_rows.loc[mask]
            if pd.notna(price) and companion_hit.empty:
                mismatch_counts["panel_offer_missing_from_companion"] += 1
            if quote_time is pd.NaT or pd.isna(quote_time):
                if not companion_hit.empty:
                    quote_time = _utc_timestamp(companion_hit.iloc[0]["scraped_at"], label=f"companion {side} quote time", reject_naive=False)
            close_price = np.nan
            close_time = pd.NaT
            close_id = pd.NA
            close_book = pd.NA
            if not close_rows.empty and bookmaker is not None:
                same_book = close_rows.loc[_key_series(close_rows["bookmaker"]).eq(bookmaker)]
                if not same_book.empty:
                    same_book = same_book.sort_values("scraped_at", ascending=False, kind="stable")
                    close_row = same_book.iloc[0]
                    close_price = float(close_row[quote_odds_col])
                    close_time = _utc_timestamp(close_row["scraped_at"], label=f"close quote time {side}")
                    close_id = _key(close_row["quote_id"])
                    close_book = bookmaker
            best[side] = {
                "odds": float(price) if pd.notna(price) else np.nan,
                "bookmaker": bookmaker if bookmaker is not None else pd.NA,
                "quote_id": quote_id if quote_id is not None else pd.NA,
                "quote_at": quote_time,
                "close_odds": close_price,
                "close_bookmaker": close_book,
                "close_quote_id": close_id,
                "close_quote_at": close_time,
            }

        strict_source = _strict_bool(row.time_compatible_strict_prior_day)
        common_source = _strict_bool(row.is_common)
        strict_flag = bool(strict_source) if strict_source is not None else False
        common_flag = bool(common_source) if common_source is not None else False
        feature_violation = not strict_source if strict_source is not None else pd.NA
        common_fields = {
            "golgg_match_id": str(row.golgg_match_id),
            "horizon": horizon,
            "date": str(row.date),
            "decision_at": pd.NaT,
            "scenario_decision_at": cutoff,
            "start_at": start_at,
            "label_available_at": pd.NaT,
            "scenario_label_available_at": row.scenario_label_available_at,
            "y": int(row.y) if pd.notna(row.y) else pd.NA,
            "p_s": float(row.p_a0) if strict_flag and pd.notna(row.p_a0) else np.nan,
            "p_s_annual": float(row.p_s_annual),
            "p_s_unavailable_reason": (
                "ANNUAL_FEATURE_CUTOFF_INCOMPATIBLE_OR_UNKNOWN" if not strict_flag
                else "MISSING_ANNUAL_PREDICTION" if pd.isna(row.p_a0) else None
            ),
            "prediction_at": row.prediction_at,
            "fit_origin": row.fit_origin,
            "scenario_prediction_at": row.scenario_prediction_at,
            "scenario_fit_origin": row.scenario_fit_origin,
            "is_oof": row.is_oof,
            "recipe_status": row.recipe_status,
            "prediction_role": row.prediction_role,
            "source_prediction_status": row.source_prediction_status,
            "source_chronology_status": row.source_chronology_status,
            "source_chronology_audit_path": row.source_chronology_audit_path,
            "source_chronology_audit_sha256": row.source_chronology_audit_sha256,
            "source_chronology_audit_status": row.source_chronology_audit_status,
            "source_chronology_audit_late_ids": row.source_chronology_audit_late_ids,
            "source_receipt_sha256": row.source_receipt_sha256,
            "source_prediction_sha256": row.source_prediction_sha256,
            "p_a0": float(row.p_a0) if pd.notna(row.p_a0) else np.nan,
            "q": q,
            "q_source": q_source,
            "panel_p_market": float(row.p_market) if pd.notna(row.p_market) else np.nan,
            "panel_q_alignment_error": panel_q_error,
            "market_book_count": int(q_books),
            "best_of": int(row.best_of) if pd.notna(row.best_of) else pd.NA,
            "league": row.league,
            "strict_prior_day": strict_flag,
            "strict_prior_day_known": strict_source is not None,
            "is_common": common_flag,
            "is_common_known": common_source is not None,
            "information_status": "historical_scenario",
            "feature_available_at": pd.NaT,
            "feature_time_violation": feature_violation,
            "time_compatible_strict_prior_day": strict_flag,
            "scenario_timestamp_assumption": "decision_at is a simulated horizon cutoff at start_dt minus horizon; recorded quote scrape times are actual; annual TEST membership/recipe source is receipt-verified; actual prediction creation and label publication times remain unknown",
            "canonical_match_id": str(row.canonical_match_id),
        }
        for side in ("a", "b"):
            common_fields[f"odds_{side}"] = best[side]["odds"]
            common_fields[f"bookmaker_{side}"] = best[side]["bookmaker"]
            common_fields[f"quote_id_{side}"] = best[side]["quote_id"]
            common_fields[f"quote_at_{side}"] = best[side]["quote_at"]
            common_fields[f"close_odds_{side}"] = best[side]["close_odds"]
            common_fields[f"close_bookmaker_{side}"] = best[side]["close_bookmaker"]
            common_fields[f"close_quote_id_{side}"] = best[side]["close_quote_id"]
            common_fields[f"close_quote_at_{side}"] = best[side]["close_quote_at"]
            common_fields[f"same_book_close_{side}_available"] = pd.notna(best[side]["close_odds"])
        close_times = [best[side]["close_quote_at"] for side in ("a", "b")]
        common_fields["close_available_at"] = max(close_times) if all(pd.notna(value) for value in close_times) else pd.NaT
        close_market = quote_groups.get((str(row.canonical_match_id), "CLOSE"), pd.DataFrame())
        common_fields["close_q"] = float(close_market["p_market_quote"].mean()) if not close_market.empty else np.nan
        common_fields["archive_market_open"] = row.canonical_market_open
        common_fields["archive_market_close"] = row.canonical_market_close
        common_fields["row_id"] = f"production:{row.golgg_match_id}:{horizon}"
        common_fields["cohort"] = "production"
        row_records.append(common_fields)

    result = pd.DataFrame(row_records)
    if not result.empty:
        result["date"] = _date_strings(result["date"], "production opportunities")
    metrics = {
        "panel_rows": int(len(panel)),
        "opportunity_rows_by_cohort": result.groupby(["cohort", "horizon"], dropna=False).size().rename("rows").reset_index().to_dict("records") if not result.empty else [],
        "horizons": {str(key): int(value) for key, value in panel["horizon"].map(_panel_horizon).value_counts(dropna=False).items()},
        "common_rows": int(panel["is_common"].map(_strict_bool).eq(True).sum()),
        "strict_prior_day_rows": int(panel["time_compatible_strict_prior_day"].map(_strict_bool).eq(True).sum()),
        "same_book_close_side_a_rows": int(result["same_book_close_a_available"].sum()) if not result.empty else 0,
        "same_book_close_side_b_rows": int(result["same_book_close_b_available"].sum()) if not result.empty else 0,
        "panel_offer_missing_from_companion_side_count": mismatch_counts["panel_offer_missing_from_companion"],
        "panel_consensus_alignment_mismatch_rows": mismatch_counts["panel_consensus_alignment_mismatch"],
        "actual_decision_times_known": int(result["decision_at"].notna().sum()) if not result.empty else 0,
        "actual_quote_times_known_a": int(result["quote_at_a"].notna().sum()) if not result.empty else 0,
        "actual_quote_times_known_b": int(result["quote_at_b"].notna().sum()) if not result.empty else 0,
        "scenario_result_release_times_known": int(result["scenario_label_available_at"].notna().sum()) if not result.empty else 0,
        "quotes": quote_coverage,
    }
    return result, metrics


def _normalize_uncertainty_frame(
    frames: list[pd.DataFrame],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not frames:
        raise ValueError("No annual uncertainty prediction files were loaded")
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "golgg_match_id", "date", "result_day", "train_end", "calibration_end",
        "y", "best_of", "model_year", "variant", "partition", "p", "sigma_z",
    }
    missing = sorted(required.difference(all_rows.columns))
    if missing:
        raise ValueError(f"Annual uncertainty predictions are missing columns: {missing}")
    all_rows["golgg_match_id"] = _key_series(all_rows["golgg_match_id"])
    all_rows["date"] = _date_strings(all_rows["date"], "uncertainty predictions")
    all_rows["result_day"] = _date_strings(all_rows["result_day"], "uncertainty result release")
    for column in ("train_end", "calibration_end"):
        values = all_rows[column].astype("string")
        if values.isna().any() or not values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"Annual uncertainty {column} must be a calendar-day release bound")
        pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
        all_rows[column] = values
    all_rows["partition"] = all_rows["partition"].astype("string").str.lower().str.strip()
    all_rows["variant"] = all_rows["variant"].astype("string").str.strip()
    year = pd.to_numeric(all_rows["model_year"], errors="coerce")
    if year.isna().any() or not np.equal(year.to_numpy(dtype=float), np.floor(year.to_numpy(dtype=float))).all():
        raise ValueError("Annual uncertainty predictions contain invalid model_year")
    all_rows["model_year"] = year.astype(int)
    if set(all_rows["variant"].dropna()) != set(UNCERTAINTY_VARIANTS):
        raise ValueError("Annual uncertainty variants differ from the frozen registry")
    if set(all_rows["model_year"].unique()) != set(UNCERTAINTY_YEARS):
        raise ValueError("Annual uncertainty files do not cover all frozen model years")
    if not set(all_rows["partition"].dropna()).issubset({"cal", "test"}):
        raise ValueError("Annual uncertainty partition must be CAL or TEST")
    if all_rows[["golgg_match_id", "partition", "variant", "model_year"]].isna().any().any():
        raise ValueError("Annual uncertainty identity columns may not be null")
    if not pd.to_numeric(all_rows["p"], errors="coerce").notna().all():
        raise ValueError("Annual uncertainty p is missing")
    if not all_rows["result_day"].ge(all_rows["date"]).all():
        raise ValueError("Annual uncertainty result release precedes its target date")
    for variant in UNCERTAINTY_VARIANTS:
        for model_year in UNCERTAINTY_YEARS:
            group = all_rows.loc[
                all_rows["variant"].eq(variant) & all_rows["model_year"].eq(model_year)
            ]
            if group.empty:
                raise ValueError(f"{variant}/{model_year} has no annual uncertainty rows")
            partitions = {
                partition: group.loc[group["partition"].eq(partition)]
                for partition in ("cal", "test")
            }
            for partition, part in partitions.items():
                if part.empty:
                    raise ValueError(
                        f"{variant}/{model_year} has an empty {partition.upper()} partition; "
                        "absence cannot establish chronology"
                    )

            origin_day = f"{model_year}-01-01"
            train_boundary = f"{model_year - 1}-01-01"
            if not (group["train_end"] < train_boundary).all():
                raise ValueError(f"{variant}/{model_year} has a future train_end")
            if not (group["calibration_end"] < origin_day).all():
                raise ValueError(f"{variant}/{model_year} has a future calibration_end")
            if not (group["train_end"] <= group["calibration_end"]).all():
                raise ValueError(f"{variant}/{model_year} train_end follows calibration_end")
            for partition, expected_date_year in (("cal", model_year - 1), ("test", model_year)):
                part = partitions[partition]
                date_year = pd.to_datetime(part["date"], errors="raise").dt.year
                if not date_year.eq(expected_date_year).all():
                    raise ValueError(
                        f"{variant}/{model_year}/{partition} contains dates outside its annual partition"
                    )
                if partition == "cal":
                    if not (part["result_day"] < origin_day).all():
                        raise ValueError(f"{variant}/{model_year} CAL label release reaches TEST origin")
                    if not (part["result_day"] <= part["calibration_end"]).all():
                        raise ValueError(f"{variant}/{model_year} CAL label release follows calibration_end")
    duplicate_key = ["variant", "model_year", "partition", "golgg_match_id"]
    if all_rows.duplicated(duplicate_key).any():
        raise ValueError("Annual uncertainty predictions have duplicate variant/year/partition identities")
    _probabilities(all_rows, ("p", "p_std", "p_q10", "p_q90", *[f"p_member_{i}" for i in range(5)]))
    sigma = pd.to_numeric(all_rows["sigma_z"], errors="coerce")
    supplied_sigma = all_rows["sigma_z"].notna()
    if (supplied_sigma & (~np.isfinite(sigma) | sigma.lt(0.0))).any():
        raise ValueError("Annual uncertainty predictions contain invalid sigma_z")
    all_rows["sigma_z"] = sigma
    member_columns = [f"p_member_{index}" for index in range(5)]
    if set(member_columns).issubset(all_rows.columns):
        member_mean = all_rows[member_columns].mean(axis=1)
        has_all_members = all_rows[member_columns].notna().all(axis=1)
        if (has_all_members & (member_mean - pd.to_numeric(all_rows["p"], errors="coerce")).abs().gt(1e-7)).any():
            raise ValueError("Annual native p does not equal its five-member ensemble mean")

    canonical_index = canonical.set_index("golgg_match_id", drop=False)
    test = all_rows.loc[all_rows["partition"].eq("test")].copy()
    for (variant, model_year), rows in test.groupby(["variant", "model_year"], sort=False):
        expected = canonical.loc[canonical["model_year"].eq(int(model_year))]
        if set(rows["golgg_match_id"]) != set(expected["golgg_match_id"]):
            raise ValueError(f"{variant}/{model_year} TEST IDs do not cover the canonical year")
        aligned = canonical_index.loc[rows["golgg_match_id"]].reset_index(drop=True)
        for column in ("date", "y", "best_of", "team1_id", "team2_id"):
            if column not in rows or column not in aligned:
                continue
            if not rows[column].astype("string").reset_index(drop=True).eq(aligned[column].astype("string")).all():
                raise ValueError(f"{variant}/{model_year} TEST identity mismatch in {column}")
        if "result_day" in rows and "result_day" in aligned:
            left = rows["result_day"].astype("string").reset_index(drop=True)
            right = aligned["result_day"].astype("string").reset_index(drop=True)
            if not left.eq(right).all():
                raise ValueError(f"{variant}/{model_year} TEST identity mismatch in result_day")

    if "result_day" in all_rows:
        scenario_labels = _daily_release_scenario(all_rows["result_day"])
    else:
        scenario_labels = pd.Series(pd.NaT, index=all_rows.index, dtype="datetime64[ns, UTC]")
    dates = pd.to_datetime(all_rows["date"], errors="raise")
    # These are explicitly modeled scenario times, never claims about when the
    # saved prediction was created or when the result was actually published.
    all_rows["prediction_at"] = pd.Series(pd.NaT, index=all_rows.index, dtype="datetime64[ns, UTC]")
    all_rows["fit_origin"] = pd.Series(pd.NaT, index=all_rows.index, dtype="datetime64[ns, UTC]")
    all_rows["label_available_at"] = pd.Series(pd.NaT, index=all_rows.index, dtype="datetime64[ns, UTC]")
    all_rows["scenario_prediction_at"] = dates.dt.normalize().dt.tz_localize("UTC")
    all_rows["scenario_fit_origin"] = pd.to_datetime(all_rows["model_year"].astype(str) + "-01-01", errors="raise").dt.tz_localize("UTC")
    all_rows["scenario_label_available_at"] = scenario_labels
    all_rows["information_status"] = "historical_scenario"
    all_rows["scenario_timestamp_assumption"] = "annual_oof_origin_at_calendar_year_start_and_daily_release_after_result_day; not recorded creation/publication times"
    all_rows["is_oof"] = all_rows["partition"].eq("test")
    all_rows["recipe_status"] = np.where(all_rows["partition"].eq("test"), "COMPLETE-RECIPE", "IN-SAMPLE-CAL")
    all_rows["prediction_role"] = np.where(all_rows["partition"].eq("test"), "oof_prediction", "sigma_reference")
    all_rows["p_native_mean"] = pd.to_numeric(all_rows["p"], errors="coerce")
    uncertainty = all_rows.loc[all_rows["partition"].eq("test")].copy().reset_index(drop=True)
    sigma_cal = all_rows.loc[all_rows["partition"].eq("cal")].copy().reset_index(drop=True)
    coverage = {
        "test_rows": int(len(uncertainty)), "cal_rows": int(len(sigma_cal)),
        "test_by_variant_year": uncertainty.groupby(["variant", "model_year"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "cal_by_variant_year": sigma_cal.groupby(["variant", "model_year"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "test_sigma_z_available": int(uncertainty["sigma_z"].notna().sum()),
        "cal_sigma_z_available": int(sigma_cal["sigma_z"].notna().sum()),
        "cal_role": "in-sample native CAL sigma reference; not complete-recipe OOF calibration",
        "optional_diagnostic_coverage": {
            column: {"available": int(uncertainty[column].notna().sum()), "missing": int(uncertainty[column].isna().sum())}
            if column in uncertainty else {"available": 0, "missing": int(len(uncertainty)), "status": "source_column_missing"}
            for column in ("p_std", "p_q10", "p_q90")
        },
    }
    return uncertainty, sigma_cal, coverage


def prepare_datasets(
    research_root: str | Path | None = None,
    panel_path: str | Path | None = None,
    quote_path: str | Path | None = None,
    uncertainty_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and verify the frozen canonical, quote, production and uncertainty sources.

    Returns ``canonical`` (every pinned target row), ``opportunities`` (separate
    archival and production cohort/horizon rows), ``uncertainty`` (TEST only),
    ``sigma_cal`` (in-sample CAL reference only), and a full provenance receipt.
    """
    root = _resolve_research_root(research_root)
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol_sha = _verify_sha256(protocol_path, PROTOCOL_SHA256, "frozen experiment protocol")
    protocol = _read_json(protocol_path)
    if protocol.get("run_id") != "ev_all_approaches_20260927_220902":
        raise ValueError("Frozen experiment protocol run_id mismatch")

    manifest_path = PROJECT_ROOT / CANONICAL_MANIFEST_RELATIVE
    manifest_sha = _verify_sha256(manifest_path, CANONICAL_MANIFEST_SHA256, "canonical research manifest")
    manifest = _read_json(manifest_path)
    canonical_spec = manifest.get("input", {})
    canonical_relative = Path(str(canonical_spec.get("path", "")))
    canonical_expected_sha = str(canonical_spec.get("sha256", ""))
    frozen_canonical = protocol.get("inputs", {}).get("canonical_sha256")
    if not canonical_relative.parts or canonical_expected_sha != frozen_canonical:
        raise ValueError("Canonical manifest source differs from the frozen experiment protocol")
    canonical_path = root / canonical_relative
    canonical_sha = _verify_sha256(canonical_path, canonical_expected_sha, "canonical paired benchmark")
    canonical_raw = pd.read_parquet(canonical_path)
    canonical = _normalize_canonical(canonical_raw, manifest.get("expected_rows"))
    corrected_bank = Path(str(protocol["inputs"]["corrected_bank_relative"]))
    if len(corrected_bank.parents) < 3:
        raise ValueError("Frozen corrected-bank path cannot resolve the annual A0 phase root")
    phase_root = root / corrected_bank.parents[2]
    canonical_oof_provenance = _verify_canonical_oof_provenance(canonical, root, phase_root)
    canonical = _attach_oof_provenance(canonical, canonical_oof_provenance)

    archive_relative = Path(protocol["inputs"]["corrected_archival_quotes_relative"])
    archive_path = root / archive_relative
    archive_sha = _verify_sha256(archive_path, ARCHIVE_QUOTES_SHA256, "corrected archival quote source")
    archive_receipt_path = root / "market-identity-20260912/data/08_reporting/identity_materialized.json"
    archive_receipt_sha = _verify_sha256(archive_receipt_path, ARCHIVE_RECEIPT_SHA256, "market identity materialization receipt")
    archive_receipt = _read_json(archive_receipt_path)
    receipt_quote_sha = archive_receipt.get("outputs", {}).get("data/03_primary/aligned_book_quotes.parquet")
    if receipt_quote_sha != archive_sha:
        raise ValueError("Market identity receipt does not pin the corrected quote source")
    archive_raw = pd.read_parquet(archive_path)
    archive_quotes, archive_input_coverage = _collapse_archive_quotes(archive_raw, canonical)
    archival_opportunities, archival_coverage = _archival_opportunities(canonical, archive_quotes)

    resolved_panel_path = _project_path(panel_path, Path("data/08_reporting/production_market_calibration_ev5_20260926_210430/audit_panel.parquet"))
    resolved_quote_path = _project_path(quote_path, Path("data/08_reporting/production_market_calibration_ev5_20260926_210430/aligned_quotes.parquet"))
    production_hashes = _production_receipt_paths(resolved_panel_path, resolved_quote_path)
    panel = pd.read_parquet(resolved_panel_path)
    production_quotes = pd.read_parquet(resolved_quote_path)
    production_opportunities, production_coverage = _production_opportunities(panel, production_quotes, canonical)

    uncertainty_default = Path("data/07_model_output/a0_uncertainty") / UNCERTAINTY_RUN
    resolved_uncertainty_dir = _project_path(uncertainty_dir, uncertainty_default)
    completed_path = PROJECT_ROOT / "data/08_reporting/a0_uncertainty" / UNCERTAINTY_RUN / "completed.json"
    completed_sha = _verify_sha256(
        completed_path, UNCERTAINTY_COMPLETION_SHA256, "annual uncertainty completion receipt",
    )
    completed = _read_json(completed_path)
    if completed.get("status") != "COMPLETE" or completed.get("run_id") != UNCERTAINTY_RUN:
        raise ValueError("Frozen annual uncertainty completion receipt is incomplete or has the wrong run_id")
    uncertainty_protocol_path = PROJECT_ROOT / "data/05_model_input/a0_uncertainty" / UNCERTAINTY_RUN / "protocol.json"
    uncertainty_protocol_sha = _verify_sha256(
        uncertainty_protocol_path, completed["protocol_sha256"], "original uncertainty protocol",
    )
    uncertainty_parent_hashes = []
    for parent in (
        {"path": completed["input_manifest_path"], "sha256": completed["input_manifest_sha256"]},
        completed["input_manifest_ancestry"],
    ):
        parent_path = PROJECT_ROOT / parent["path"]
        parent_sha = _verify_sha256(parent_path, parent["sha256"], "uncertainty input manifest")
        parent_manifest = _read_json(parent_path)
        if parent_manifest.get("protocol_sha256") != uncertainty_protocol_sha:
            raise ValueError("Uncertainty input manifest refers to a different protocol")
        uncertainty_parent_hashes.append({"path": str(parent_path.relative_to(PROJECT_ROOT)), "sha256": parent_sha})
    artifact_hashes = completed.get("artifact_sha256", {})
    annual_frames: list[pd.DataFrame] = []
    uncertainty_hashes: dict[str, str] = {}
    for variant in UNCERTAINTY_VARIANTS:
        for model_year in UNCERTAINTY_YEARS:
            filename = f"{variant}_{model_year}.parquet"
            path = resolved_uncertainty_dir / filename
            relative_receipt_path = f"data/07_model_output/a0_uncertainty/{UNCERTAINTY_RUN}/{filename}"
            expected = artifact_hashes.get(relative_receipt_path)
            if not expected:
                raise ValueError(f"Uncertainty completion receipt does not pin {relative_receipt_path}")
            digest = _verify_sha256(path, str(expected), f"annual uncertainty prediction {filename}")
            uncertainty_hashes[relative_receipt_path] = digest
            annual_frame = pd.read_parquet(path)
            if {"variant", "model_year"}.difference(annual_frame.columns):
                raise ValueError(f"{filename} is missing annual variant/model_year provenance")
            labels = annual_frame["variant"].astype("string").str.strip()
            file_years = pd.to_numeric(annual_frame["model_year"], errors="coerce")
            if (
                annual_frame.empty
                or labels.isna().any()
                or not labels.eq(variant).all()
                or file_years.isna().any()
                or not np.equal(file_years.to_numpy(dtype=float), model_year).all()
            ):
                raise ValueError(f"{filename} rows do not match their annual variant/year filename")
            annual_frames.append(annual_frame)
    uncertainty, sigma_cal, uncertainty_coverage = _normalize_uncertainty_frame(annual_frames, canonical)

    archival_rows = validate_opportunities(archival_opportunities)
    production_rows = validate_opportunities(production_opportunities) if not production_opportunities.empty else production_opportunities
    opportunities = pd.concat([archival_rows, production_rows], ignore_index=True, sort=False)
    canonical_market_coverage = {
        "rows": int(len(canonical)),
        "unique_fixtures": int(canonical["golgg_match_id"].nunique()),
        "p_s_available": int(canonical["p_s"].notna().sum()),
        "annual_oof_membership_verified_rows": int(canonical["is_oof"].fillna(False).sum()),
        "market_open_available": int(canonical["market_open"].notna().sum()) if "market_open" in canonical else 0,
        "market_close_available": int(canonical["market_close"].notna().sum()) if "market_close" in canonical else 0,
        "rows_by_year": {str(int(year)): int(count) for year, count in canonical.groupby("model_year").size().items()},
        "oof_rows_by_year": {
            str(int(year)): int(group["is_oof"].fillna(False).sum())
            for year, group in canonical.groupby("model_year", sort=True)
        },
    }
    source_hashes = {
        "canonical_manifest": {"path": str(CANONICAL_MANIFEST_RELATIVE), "sha256": manifest_sha},
        "canonical": {"path": str(canonical_path.relative_to(root)), "sha256": canonical_sha},
        "corrected_archival_quotes": {"path": str(archive_path.relative_to(root)), "sha256": archive_sha},
        "identity_materialization_receipt": {"path": str(archive_receipt_path.relative_to(root)), "sha256": archive_receipt_sha},
        "production_panel": {"path": _display_path(resolved_panel_path, PROJECT_ROOT), "sha256": production_hashes["panel_sha256"]},
        "production_quotes": {"path": _display_path(resolved_quote_path, PROJECT_ROOT), "sha256": production_hashes["aligned_quotes_sha256"]},
        "canonical_annual_oof": canonical_oof_provenance["years"],
        "pre_correction_training_audit": canonical_oof_provenance["pre_correction_audit"],
        "uncertainty_completion_receipt": {"path": str(completed_path.relative_to(PROJECT_ROOT)), "sha256": completed_sha},
        "experiment_protocol": {"path": str(PROTOCOL_RELATIVE), "sha256": protocol_sha},
        "uncertainty_protocol": {"path": str(uncertainty_protocol_path.relative_to(PROJECT_ROOT)), "sha256": uncertainty_protocol_sha},
        "uncertainty_input_manifests": uncertainty_parent_hashes,
        "uncertainty_annual_predictions": uncertainty_hashes,
    }
    receipt = {
        "schema_version": 1,
        "run_id": protocol["run_id"],
        "evidence_limit": protocol.get("evidence_limit"),
        "input_hashes": source_hashes,
        "coverage": {
            "canonical": canonical_market_coverage,
            "archival_source_quotes": archive_input_coverage,
            "archival_opportunities": archival_coverage,
            "production": production_coverage,
            "uncertainty": uncertainty_coverage,
        },
        "missingness": {
            "archival_rows_without_open_executable_side_a": archival_coverage["canonical_rows_retained"] - archival_coverage["open_executable_side_a_rows"],
            "archival_rows_without_open_executable_side_b": archival_coverage["canonical_rows_retained"] - archival_coverage["open_executable_side_b_rows"],
            "archival_rows_without_same_book_close_a": archival_coverage["canonical_rows_retained"] - archival_coverage["same_book_close_side_a_rows"],
            "archival_rows_without_same_book_close_b": archival_coverage["canonical_rows_retained"] - archival_coverage["same_book_close_side_b_rows"],
            "production_quotes_outside_declared_horizon_window": production_coverage.get("quotes", {}).get("bad_time_rows", 0),
            "production_panel_offer_not_found_in_companion": production_coverage.get("panel_offer_missing_from_companion_side_count", 0),
            "production_market_consensus_disagreement_rows": production_coverage.get("panel_consensus_alignment_mismatch_rows", 0),
        },
        "timing_provenance": {
            "archival_quotes": "Date/timing labels only; actual quote_at_a/quote_at_b/decision_at remain NaT. No verified UTC publication time.",
            "production_quotes": "Quote observation timestamps are retained only from timezone-aware panel/companion source fields; no forecast-creation time is inferred.",
            "canonical_sports": "Canonical p_full matches pinned corrected causal A0 annual TEST outputs; membership is independently verified, actual fit/prediction/publication times remain NaT, and model-year/target-day clocks are explicit historical scenarios only.",
            "features": "feature_available_at remains NaT; production strict-prior-day compatibility is preserved as a flag, not promoted to a fabricated timestamp.",
            "uncertainty": "Saved prediction/fit creation timestamps remain NaT; annual OOF and daily label-release times are explicit scenario fields only.",
        },
        "opportunity_rows": int(len(opportunities)),
        "cohort_horizon_rows": opportunities.groupby(["cohort", "horizon"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "production_source_receipt": production_hashes,
    }
    return {
        "canonical": canonical,
        "opportunities": opportunities,
        "uncertainty": uncertainty,
        "sigma_cal": sigma_cal,
        "receipt": receipt,
    }

# The native study is intentionally separate from the sealed annual-A0 reader.
C0_PHASES = {
    "TRAIN": ("2026-11-01T00:00:00Z", "2027-11-01T00:00:00Z", "2027-11-15T00:00:00Z", "2027-12-01T00:00:00Z"),
    "CAL": ("2027-12-01T00:00:00Z", "2028-06-01T00:00:00Z", "2028-06-15T00:00:00Z", "2028-07-01T00:00:00Z"),
    "DEVELOPMENT": ("2028-07-01T00:00:00Z", "2029-07-01T00:00:00Z", "2029-07-15T00:00:00Z", None),
}


def c0_canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def c0_digest(value: Any) -> str:
    return hashlib.sha256(c0_canonical_bytes(value)).hexdigest()


def c0_clock(value: Any) -> pd.Timestamp:
    clock = pd.Timestamp(value)
    if pd.isna(clock) or clock.tzinfo is None:
        raise ValueError("C0 clock must be aware")
    return clock.tz_convert("UTC")


def validate_c0_protocol(protocol: dict[str, Any]) -> None:
    if (protocol.get("schema") != "c0_market_study_v1"
            or protocol.get("protocol_id") != "native-c0-market-study-v1"
            or protocol.get("native_identity") != "Causal-C0:c0-native-2026-w32-e12-v1"
            or protocol.get("mode") != "full"
            or protocol.get("calendar") != {k: list(v) for k, v in C0_PHASES.items()}
            or protocol.get("epsilon") != 1e-6
            or protocol.get("lambdas") != [.01, .1, 1.]):
        raise ValueError("Wrong frozen native-C0 protocol")


def prepare_c0_market_study_rows(input_dir, protocol, *, phase):
    """Verify an immutable phase export before opening its labels.

    Missing evidence is a counted admission failure. Corrupt retained evidence
    is an input error, never an abstention silently omitted from the universe.
    """
    validate_c0_protocol(protocol)
    if phase not in C0_PHASES:
        raise ValueError("Unknown phase")
    root = Path(input_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if (manifest.get("schema") != "c0_market_study_bundle_v1"
            or manifest.get("phase") != phase
            or manifest.get("protocol_id") != protocol["protocol_id"]
            or manifest.get("protocol_sha256") != c0_digest(protocol)):
        raise ValueError("Wrong phase/protocol bundle")
    filenames = ("opportunities.jsonl", "envelopes.jsonl", "captures.jsonl",
                 "bindings.jsonl", "dependencies.jsonl", "authentication.jsonl",
                 "labels.jsonl", "authorization.json")
    for name in filenames:
        entry = manifest.get("files", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing manifest entry {name}")
        _verify_sha256(root / name, entry["sha256"], name)
    for name, entry in manifest.get("files", {}).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not isinstance(entry,dict):
            raise ValueError("Manifest file path/schema invalid")
        _verify_sha256(path,entry["sha256"],name)
    for group in ("retained_bodies", "retained_receipts"):
        for sha, relative in manifest.get(group, {}).items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Retained path escapes bundle")
            _verify_sha256(path, sha, group)
    def rows(name):
        result = []
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL record must be object")
            if line.encode("utf-8") != c0_canonical_bytes(value):
                raise ValueError("Noncanonical phase JSONL")
            result.append(value)
        count = manifest["files"][name].get("row_count")
        if count is not None and count != len(result):
            raise ValueError("Manifest row count mismatch")
        return result
    opportunities = rows("opportunities.jsonl")
    # Close the membership before reading any outcome content.
    if "membership.json" not in manifest["files"]:
        raise ValueError("Missing closed membership seal")
    membership_record = _read_json(root/"membership.json")
    membership = c0_digest(membership_record)
    if (manifest.get("membership_sha256") != membership
            or membership_record.get("phase") != phase
            or membership_record.get("protocol_id") != protocol["protocol_id"]
            or membership_record.get("opportunity_sha256") != manifest["files"]["opportunities.jsonl"]["sha256"]
            or membership_record.get("event_keys") != sorted(e["event_key"] for e in opportunities)):
        raise ValueError("Opportunity membership mismatch")
    envelope_rows, capture_rows = rows("envelopes.jsonl"), rows("captures.jsonl")
    envelopes = {r["evidence_key"]: r for r in envelope_rows}
    captures = {str(r["capture_id"]): r for r in capture_rows}
    if len(envelopes) != len(envelope_rows) or len(captures) != len(capture_rows):
        raise ValueError("Duplicate immutable evidence identity")
    bindings = rows("bindings.jsonl")
    dependencies = rows("dependencies.jsonl")
    authentication = rows("authentication.jsonl")
    authorization = _read_json(root / "authorization.json")
    pinned_authority = protocol.get("activation_authorization_sha256")
    if pinned_authority is not None and c0_digest(authorization) != pinned_authority:
        raise ValueError("Bundle acquisition authority differs from frozen activation")
    import base64
    def retained_record(record, collection):
        capture = captures.get(str(record.get("capture_id")))
        if not capture or record.get("source_sha256") != capture.get("sha256"):
            raise ValueError(f"{collection} retained capture mismatch")
        body = base64.b64decode(capture["content_base64"], validate=True)
        if hashlib.sha256(body).hexdigest() != capture["sha256"]:
            raise ValueError("Retained capture byte hash mismatch")
        decoded = json.loads(body)
        annotations = {"capture_id","source_sha256","source_id","published_at","response_received_at",
                       "first_observed_at","observed_at","recorded_at","label_lock_at"}
        source_records = decoded.get(collection, []) if isinstance(decoded, dict) else []
        if collection == "bindings" and isinstance(decoded, dict):
            source_records = [*decoded.get("bindings", []), *decoded.get("participant_bindings", [])]
        original = next((item for item in source_records if isinstance(item, dict)
            and all(record.get(k) == v for k, v in item.items())
            and set(record)-set(item) <= annotations), None)
        if original is None:
            raise ValueError(f"{collection} record differs from retained source bytes")
        expected_annotations = {**capture, "source_sha256":capture["sha256"],
                                "label_lock_at":C0_PHASES[phase][2]}
        for key in set(record)-set(original):
            actual_value, expected_value = record[key], expected_annotations.get(key)
            if key.endswith("_at") and actual_value is not None and expected_value is not None:
                matches = c0_clock(actual_value) == c0_clock(expected_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(f"{collection} annotation differs from retained capture clock/identity")
        return capture
    for binding in bindings:
        retained_record(binding, "bindings")
    from betting_app.services.c0_shadow_readiness import (
        evaluate_c0_readiness, validate_dependency_receipt, validate_source_adjudication,
        _encoded_from_envelope, _retained_ref, c0_forecast_link_reasons,
    )
    evidence_context = {**protocol,
        "captures":[*protocol.get("captures", []), *captures.values()],
        "authentication_receipts":[*protocol.get("authentication_receipts", []), *authentication]}
    readiness = evaluate_c0_readiness(protocol, now=pd.Timestamp.now(tz="UTC").to_pydatetime(),
                                      authorization=authorization)
    authority_pass = readiness.get("status") == "READY_AUTHORIZED" and not readiness.get("blockers")
    complete_recipe = {}
    if authority_pass:
        compatibility = _retained_ref(authorization["native_compatibility_receipt_ref"],
                                      protocol, trusted_kind="native_compatibility")
        complete_recipe = compatibility.get("complete_recipe", {})
    effective_sources = set()
    effective_capture_ids = set()
    for adjudication in authentication:
        if not validate_source_adjudication(adjudication, protocol=protocol, captures=list(captures.values())):
            effective_sources.add(adjudication.get("source_id"))
            effective_capture_ids.update(str(ref["capture_id"]) for ref in adjudication["capture_refs"])
    start, end, lock, _ = C0_PHASES[phase]
    panel = []
    seen = set()
    for event in opportunities:
        payload = event.get("payload", event)
        fixture = payload.get("canonical_match_id", event.get("canonical_match_id"))
        cutoff_value = event.get("nominal_cutoff_at", payload.get("nominal_cutoff_at",
                      payload.get("origin_at", payload.get("requested_decision_at"))))
        reasons = []
        primary = event.get("primary_decision") is True and event.get("event_type") == "primary_decision"
        if not primary:
            reasons.append("non_primary_event")
        try:
            cutoff = c0_clock(cutoff_value)
        except (ValueError, TypeError):
            cutoff = pd.NaT
            reasons.append("cutoff_unknown")
        if primary and pd.notna(cutoff) and not c0_clock(start) <= cutoff < c0_clock(end):
            raise ValueError("Opportunity outside phase")
        key = fixture
        if primary:
            if key in seen:
                reasons.append("duplicate_primary_decision")
            seen.add(key)
        envelope = envelopes.get(payload.get("evidence_key", event.get("forecast_evidence_key")))
        native = envelope.get("payload", {}) if envelope else {}
        if not authority_pass:
            reasons.append("G0_BLOCKED")
        if not envelope:
            reasons.append("forecast_missing")
        model = native.get("model", {})
        if model.get("c0_serving_contract") != "native-c0-forward-v1":
            reasons.append("native_contract_unverified")
        if model.get("mode", native.get("input", {}).get("mode")) != "full":
            reasons.append("wrong_native_mode")
        if (model.get("name", model.get("model_name")) != "Causal-C0"
                or model.get("version", model.get("model_version")) != "c0-native-2026-w32-e12-v1"):
            reasons.append("wrong_native_identity")
        for wrapper, digest_field in ((event, "payload_sha256"), (envelope or {}, "content_sha256")):
            if wrapper.get(digest_field) and c0_digest(wrapper["payload"]) != wrapper[digest_field]:
                raise ValueError("Immutable payload hash mismatch")
        sources = native.get("sources", [])
        encoded = _encoded_from_envelope(native) if native else None
        matching_dependencies = [d for d in dependencies if encoded is not None
                                 and d.get("snapshot_sha256") == c0_digest(encoded)]
        if len(matching_dependencies) != 1:
            reasons.append("native_history_coverage_unverified")
        else:
            reasons.extend(validate_dependency_receipt(matching_dependencies[0], protocol=protocol,
                                                       encoded_snapshot=encoded, sources=sources))
        if not sources or any(str(s.get("source_id")) not in effective_capture_ids for s in sources):
            reasons.append("source_authentication_missing")
        available_bindings = []
        for binding in bindings:
            capture = captures.get(str(binding.get("capture_id")), {})
            try:
                if (str(binding.get("capture_id")) in effective_capture_ids and pd.notna(cutoff)
                        and max(c0_clock(capture[k]) for k in
                            ("response_received_at", "first_observed_at", "observed_at") if capture.get(k)) <= cutoff):
                    available_bindings.append(binding)
            except (KeyError, ValueError, TypeError):
                continue
        fixture_bindings = [b for b in available_bindings if
            str(b.get("canonical_match_id", b.get("canonical_fixture_id"))) == str(fixture)]
        binding_pass = False
        fixture_identity_pass = False
        for binding in fixture_bindings:
            try:
                if (c0_clock(binding["effective_from"]) <= cutoff < c0_clock(binding["effective_until"])
                        and str(binding.get("provider_event_id")) == str(payload.get("provider_fixture_id"))
                        and binding.get("participant_a_id") and binding.get("participant_b_id")
                        and binding["participant_a_id"] != binding["participant_b_id"]):
                    fixture_identity_pass = True
                    if binding.get("market_type") == "match_winner":
                        binding_pass = True
            except (KeyError, ValueError, TypeError):
                continue
        if not fixture_identity_pass:
            reasons.append("provider_binding_missing")
        target = native.get("input", {}).get("target", {})
        roster_results = payload.get("roster_results", {})
        native_roster = native.get("roster", {})
        roster_pass = len(roster_results) == 2
        for route in roster_results.values():
            announcement = route.get("announcement") or {}
            roster_pass = roster_pass and (
                route.get("status") in {"announced", "prior_known"}
                and len(announcement.get("roster", [])) == 5
                and bool(announcement.get("competitive_unit_id"))
                and announcement.get("competition_context") == target.get("competition_context")
                and not route.get("missing_reason") and not route.get("ambiguity_evidence"))
        if not roster_pass or any(len(native_roster.get(side, target.get(side, []))) != 5
                                  for side in ("roster_a", "roster_b")):
            reasons.append("roster_context_unverified")
        clocks = {**native.get("forecast", {}), **native.get("clocks", {}),
                  "recorded_at": (envelope or {}).get("recorded_at"),
                  "evaluated_at": payload.get("clocks", {}).get("decision_at")}
        try:
            opportunity_clocks = payload["clocks"]
            if (c0_clock(opportunity_clocks["scheduled_start_at"])-cutoff != pd.Timedelta(hours=1)
                    or c0_clock(opportunity_clocks["schedule_available_at"]) > cutoff):
                reasons.append("frozen_primary_origin_unverified")
            candidate = {"protocol_id":protocol["protocol_id"],
                "provider_id":event.get("provider_id",payload.get("provider_id")),
                "fixture_id":event.get("fixture_id",payload.get("provider_fixture_id")),
                "fixture":{"team_ids":payload.get("team_ids",[]),"canonical_fixture_id":fixture,
                    "best_of":payload.get("best_of"),"competition_context":payload.get("competition_context"),
                    "competitive_unit_ids":payload.get("competitive_unit_ids")},
                "origin_at":cutoff.to_pydatetime(),
                "scheduled_start_at":c0_clock(opportunity_clocks["scheduled_start_at"]).to_pydatetime(),
                "known_start_bound":c0_clock(opportunity_clocks["known_start_bound"]).to_pydatetime(),
                "rosters":{str(team):(route.get("announcement") or {}) for team,route in roster_results.items()},
                "source_captures":[captures[str(c)] for c in payload.get("source_capture_ids",[]) if str(c) in captures]}
            candidate["protocol"] = evidence_context
            candidate["authorization"] = authorization
            reasons.extend(c0_forecast_link_reasons(native,candidate,
                recorded_at=c0_clock(clocks["recorded_at"]).to_pydatetime()))
        except (KeyError, ValueError, TypeError):
            reasons.append("native_fixture_origin_link_unverified")
        try:
            chain = [cutoff] + [c0_clock(clocks[k]) for k in
                     ("generated_at", "parent_received_at", "recorded_at", "evaluated_at")]
            if any(a > b for a, b in zip(chain, chain[1:])) or chain[-1] > cutoff + pd.Timedelta(seconds=60):
                reasons.append("forecast_completion_late")
        except (KeyError, TypeError, ValueError):
            reasons.append("forecast_clock_unverified")
        inputs = native.get("input", {})
        for capture_id in inputs.get("source_ids", []):
            capture = captures.get(str(capture_id))
            try:
                if not capture or c0_clock(capture.get("response_received_at")) > cutoff:
                    reasons.append("input_available_after_cutoff")
            except (ValueError, TypeError):
                reasons.append("input_clock_unknown")
        prediction = native.get("forecast", {})
        try:
            probabilities = [prediction["prob_a"], prediction["prob_b"]]
            if (any(isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p)
                    or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-8):
                reasons.append("native_probability_invalid")
        except KeyError:
            reasons.append("native_probability_missing")
        for clock_name in ("fit_cutoff_at", "calibration_cutoff_at"):
            try:
                recipe_clock = complete_recipe.get(clock_name)
                if c0_clock(recipe_clock) >= c0_clock(clocks["generated_at"]):
                    reasons.append("native_recipe_cutoff_invalid")
            except (KeyError, ValueError, TypeError):
                reasons.append("native_recipe_cutoff_unverified")
        quotes = native.get("quotes", payload.get("quote_evidence", []))
        source_states = {}
        clock_unknown = set()
        for quote in quotes:
            book = quote.get("bookmaker_slug", quote.get("bookmaker"))
            if not isinstance(book, str) or not book.strip():
                continue
            book = book.strip().casefold()
            capture = captures.get(str(quote.get("capture_id")), {})
            try:
                observed = (c0_clock(quote["scraped_at"]) if quote.get("scraped_at") else
                    max(c0_clock(capture[k]) for k in ("response_received_at","first_observed_at","observed_at")
                        if capture.get(k)))
                if observed > cutoff:
                    continue
                order = (observed, str(quote.get("quote_id",quote.get("odds_snapshot_id",quote.get("capture_id")))))
                if book not in source_states or order > source_states[book][0]:
                    source_states[book] = (order,quote)
            except (ValueError,KeyError,TypeError):
                clock_unknown.add(book)
        quotes = [value[1] for book,value in sorted(source_states.items()) if book not in clock_unknown]
        normalized_quotes = []
        from src.analysis.quote_provenance import (
            extract_winner_quote, _validate_participant_bindings, WINNER_MARKET_ID,
            OUTCOME_TEAM_1, OUTCOME_TEAM_2)
        for quote in quotes:
            if "odds_a" in quote:
                try:
                    retained_record(quote, "quotes")
                    bookmaker = quote["bookmaker"]
                    source_fixture_id = str(quote["source_fixture_id"])
                    if (quote["market_type"] != "match_winner" or str(quote["market_id"]) != WINNER_MARKET_ID
                            or source_fixture_id != str(payload.get("provider_fixture_id"))):
                        continue
                    fixture_contexts = [b["fixture"] for b in fixture_bindings
                        if isinstance(b.get("fixture"), dict)
                        and str(b["fixture"].get("fixtureId")) == source_fixture_id]
                    if not fixture_contexts:
                        continue
                    outcome_a, outcome_b = str(quote["outcome_id_a"]), str(quote["outcome_id_b"])
                    if {outcome_a, outcome_b} != {OUTCOME_TEAM_1, OUTCOME_TEAM_2}:
                        continue
                    identities = {outcome_a: {"participant_id": str(quote["participant_a_id"]),
                        "participant_name": quote.get("participant_a_name"), "conflict": False},
                        outcome_b: {"participant_id": str(quote["participant_b_id"]),
                        "participant_name": quote.get("participant_b_name"), "conflict": False}}
                    status, _, indexes, _ = _validate_participant_bindings(
                        [b for b in available_bindings
                         if b.get("bookmaker") == bookmaker and str(b.get("fixture_id")) == source_fixture_id],
                        source_fixture_id=source_fixture_id, bookmaker=bookmaker, market_id=WINNER_MARKET_ID,
                        fixture=fixture_contexts[0], cutoff=cutoff.to_pydatetime(),
                        start=c0_clock(opportunity_clocks["known_start_bound"]).to_pydatetime(),
                        identities=(identities[OUTCOME_TEAM_1], identities[OUTCOME_TEAM_2]))
                    if status != "admitted_with_bookmaker_evidence" or indexes != {outcome_a: 1, outcome_b: 2}:
                        continue
                    normalized_quotes.append({**quote, "underlying_bookmaker": bookmaker.strip().casefold()})
                except (ValueError, KeyError, TypeError):
                    pass
                continue
            capture = captures.get(str(quote.get("capture_id")))
            if not capture:
                continue
            try:
                source_payload = json.loads(base64.b64decode(capture["content_base64"], validate=True))
                fixture_contexts = [b["fixture"] for b in fixture_bindings if isinstance(b.get("fixture"),dict)]
                if not fixture_contexts:
                    continue
                parsed = extract_winner_quote(source_payload, quote["bookmaker_slug"],
                    cutoff_at=cutoff.to_pydatetime(), fixture=fixture_contexts[0],
                    participant_bindings=[b for b in available_bindings
                        if b.get("bookmaker") == quote["bookmaker_slug"]
                        and str(b.get("fixture_id")) == str(source_payload.get("fixtureId"))])
                if not parsed or parsed["quote_admission_status"] != "admitted_with_bookmaker_evidence":
                    continue
                indexes = parsed["outcome_fixture_participant_indexes"]
                # Outcome IDs are retained, not inferred from price order.
                orientation = [indexes.get(parsed["outcome_id_1"]), indexes.get(parsed["outcome_id_2"])]
                if orientation not in ([1,2],[2,1]):
                    continue
                a,b = parsed["odds_1"],parsed["odds_2"]
                if orientation == [2,1]:
                    a,b = b,a
                books = source_payload.get("bookmakerOdds") or source_payload.get("bookmakers") or {}
                scopes = [source_payload, source_payload.get("fixture"),
                          books.get(quote["bookmaker_slug"]), parsed["source_market_evidence"],
                          parsed["source_outcome_evidence_1"], parsed["source_outcome_evidence_2"]]
                if any(isinstance(scope, dict) and any(scope.get(key) is True
                       for key in ("isLive", "is_live", "live", "inPlay", "inplay")) for scope in scopes):
                    continue
                prematch_proof = any(isinstance(scope,dict) and any(scope.get(key) is False
                    for key in ("isLive","is_live","live","inPlay","inplay")) for scope in scopes)
                observed = max(c0_clock(capture[k]) for k in
                    ("response_received_at","first_observed_at","observed_at") if capture.get(k))
                normalized_quotes.append({**quote,"bookmaker":quote["bookmaker_slug"],
                    "underlying_bookmaker":quote["bookmaker_slug"].strip().casefold(),
                    "source_sha256":capture["sha256"],"scraped_at":observed.isoformat(),
                    "quote_id":f'{capture["capture_id"]}:{parsed["outcome_id_1"]}:{parsed["outcome_id_2"]}',
                    "odds_a":a,"odds_b":b,"is_live":False if prematch_proof else None})
            except (KeyError,TypeError,ValueError):
                continue
        quotes = normalized_quotes
        latest_quotes = {}
        unknown_clock_books = set()
        for quote in quotes:
            book = quote.get("underlying_bookmaker")
            if not book:
                continue
            try:
                observed = c0_clock(quote["scraped_at"])
                if observed > cutoff:
                    continue
                identity = quote.get("quote_id", quote.get("odds_snapshot_id"))
                if identity is None:
                    raise ValueError("Immutable quote identity missing")
                order = (observed, str(identity))
                if book not in latest_quotes or order > latest_quotes[book][0]:
                    latest_quotes[book] = (order, quote)
            except (KeyError, ValueError, TypeError):
                unknown_clock_books.add(book)
        valid_quotes = []
        books = set()
        for book, (_, quote) in sorted(latest_quotes.items()):
            if book in unknown_clock_books:
                continue
            try:
                capture = captures.get(str(quote.get("capture_id")))
                if (not capture or str(quote.get("capture_id")) not in effective_capture_ids
                        or quote.get("source_sha256") != capture.get("sha256")
                        or c0_clock(capture["response_received_at"]) > cutoff
                        or not binding_pass
                        or isinstance(quote.get("odds_a"), bool) or isinstance(quote.get("odds_b"), bool)):
                    continue
                observed = c0_clock(quote["scraped_at"])
                a, b = float(quote["odds_a"]), float(quote["odds_b"])
                book = quote["underlying_bookmaker"]
                if (quote.get("is_live") is not False or not np.isfinite([a, b]).all()
                        or min(a, b) <= 1 or not 0 <= (cutoff - observed).total_seconds() <= 300):
                    continue
                if book not in books:
                    valid_quotes.append({**quote, "odds_a": a, "odds_b": b})
                    books.add(book)
            except (KeyError, ValueError, TypeError):
                continue
        q = np.mean([v["odds_b"] / (v["odds_a"] + v["odds_b"]) for v in valid_quotes]) if len(books) >= 2 else None
        panel.append({"row_id": str(event.get("event_key", len(panel))), "canonical_match_id": fixture,
                      "decision_at": cutoff, "month": str(cutoff)[:7], "p_C0": prediction.get("prob_a"),
                      "q": q, "quotes": valid_quotes, "admission_reasons": sorted(set(reasons)),
                      "native_admitted": not reasons, "financial_eligible": False, "y": None,
                      "evaluated_at": clocks.get("evaluated_at"), "edition": payload.get("edition"),
                      "settlement_status": "pending", "primary_decision":primary})
    admission = {"schema": "c0_admission_v1", "phase": phase, "membership_sha256": membership,
                 "protocol_sha256": c0_digest(protocol), "opportunity_count": len(panel),
                 "prelabel_admission_sha256": c0_digest([{k: None if v is pd.NaT else
                     str(v) if isinstance(v, pd.Timestamp) else v
                     for k, v in row.items() if k != "y"} for row in panel])}
    labels = rows("labels.jsonl")
    label_by_fixture = {}
    phase_fixtures = {str(r["canonical_match_id"]) for r in panel}
    for label in labels:
        label_fixture = str(label.get("canonical_match_id", label.get("canonical_fixture_id")))
        if (label_fixture not in phase_fixtures
                or label.get("phase", phase) != phase):
            raise ValueError("Out-of-phase label cannot be opened for this membership")
        if label_fixture in label_by_fixture:
            raise ValueError("Duplicate released label")
        capture = retained_record(label, "labels")
        c0_label_available_at(label, capture, phase=phase)
        if label.get("source_id") != capture.get("source_id"):
            raise ValueError("Label source identity mismatch")
        label_by_fixture[label_fixture] = label
    financial_fixtures = set()
    for row in panel:
        label = label_by_fixture.get(str(row["canonical_match_id"]))
        if label:
            row["settlement_status"] = label.get("settlement_status", "unmatured")
            if str(label.get("capture_id")) not in effective_capture_ids:
                row["settlement_status"] = "settlement_unverified"
        if (label and label.get("settlement_status") == "normal" and label.get("winner") in ("A", "B")
                and str(label.get("capture_id")) in effective_capture_ids):
            row["y"] = int(label["winner"] == "A")
            row["settlement_status"] = label["settlement_status"]
            row["financial_eligible"] = (authority_pass and row["primary_decision"]
                                        and row["canonical_match_id"] not in financial_fixtures
                                        and label.get("source_id") in effective_sources)
            if row["primary_decision"]:
                financial_fixtures.add(row["canonical_match_id"])
            try:
                actual_start = c0_clock(label["actual_start_at"])
                cutoff = c0_clock(row["decision_at"])
                market_evaluation = cutoff if row["evaluated_at"] is None else c0_clock(row["evaluated_at"])
                if not cutoff <= market_evaluation <= cutoff + pd.Timedelta(seconds=60) or market_evaluation >= actual_start:
                    raise ValueError("decision_not_before_actual_start")
            except (KeyError, ValueError, TypeError):
                row["native_admitted"] = False
                row["q"], row["quotes"] = None, []
                row["admission_reasons"].append("decision_not_before_verified_actual_start")
    frame = pd.DataFrame(panel, columns=("row_id", "canonical_match_id", "decision_at", "month",
        "p_C0", "q", "quotes", "admission_reasons", "native_admitted", "financial_eligible",
        "y", "evaluated_at", "edition", "settlement_status", "primary_decision"))
    for column in ("native_admitted","financial_eligible","primary_decision"):
        frame[column] = frame[column].astype(bool)
    frame.attrs["c0_bindings"] = {"protocol_sha256": c0_digest(protocol), "membership_sha256": membership,
                                 "code_hashes": protocol["code_hashes"]}
    frame.attrs["monthly_universe"] = [clock.strftime("%Y-%m") for clock in
        pd.date_range(c0_clock(start), c0_clock(end)-pd.Timedelta(microseconds=1), freq="MS")]
    admission["monthly_universe"] = frame.attrs["monthly_universe"]
    admission["admitted_count"] = sum(r["native_admitted"] for r in panel)
    admission["status"] = "PASS" if authority_pass else "G0_BLOCKED"
    return frame, admission


def c0_label_available_at(label, capture, *, phase):
    """Actual retained receipt availability, never an invented publication time."""
    observed = max(c0_clock(capture[key]) for key in
        ("response_received_at","first_observed_at","observed_at","recorded_at") if capture.get(key))
    available = c0_clock(label["available_at"]) if label.get("available_at") else observed
    if available < observed:
        raise ValueError("Label availability precedes retained actual capture")
    if label.get("published_at") is not None and c0_clock(label["published_at"]) > available:
        raise ValueError("Label publication follows release")
    if available >= c0_clock(C0_PHASES[phase][2]):
        raise ValueError("Label released after phase lock")
    return available
