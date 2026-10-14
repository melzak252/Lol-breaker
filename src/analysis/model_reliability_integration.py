"""Triggered native-A0/rating integration under the frozen reliability recipe.

This is exposed historical-development evidence only. It fits annual scalar blend
weights on the exact native annual CAL cohort, reuses saved native calibration,
and never retrains native experts or refits their annual calibrators.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from src.analysis import model_reliability as reliability_study
from src.analysis.a0_data_focus import _summary
from src.analysis.research_benchmark import losses
from src.models import a0_data_focus as native_focus
from src.models import a0_reliability as annual_native
from src.models import a0_uncertainty as uncertainty
from src.models.a0_native_refit import _contract_digest, _sha256, _write_json_exclusive

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "model_reliability_20261001_055321"
FAMILY_RUN_ID = "model_reliability_20261001_055321_family_alignment_v2"
DESIGN_RELATIVE = Path("data/05_model_input/model_reliability") / RUN_ID / "integration_design.json"
DESIGN_SHA256 = "8653a346ba8daf85f0c54fc17ebc1b9f48b3d41cae6c66b51c7b58b7857fd1c7"
EVALUATION_PROTOCOL_SHA256 = "5a040eb07dc0415ffbe078f5309527d1d4e08a4c64aadb0a4e337ed21c3ffcf0"
YEARS = (2024, 2025, 2026)
CRITICAL = ("bo3", "bo5", "major", "source_unseen_player")
INTEGRATION_SOURCES = (
    "scripts/run_model_benchmark.py",
    "src/analysis/model_reliability_integration.py",
    "src/analysis/model_reliability.py",
    "src/analysis/a0_data_focus.py",
    "src/analysis/model_benchmark.py",
    "src/analysis/production_market_benchmark.py",
    "src/analysis/research_benchmark.py",
    "src/models/a0_data_focus.py",
    "src/models/a0_reliability.py",
    "src/models/a0_uncertainty.py",
    "src/models/a0_native_refit.py",
    "src/models/a0_target_experiment.py",
    "src/models/a0_history_reconstruction.py",
    "src/models/a0_target_objectives.py",
    "src/models/competition_tiers.py",
)


def _json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError(f"receipt checksum differs: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_hash(path: Path, expected: str) -> None:
    if not Path(path).is_file() or _sha256(path) != expected:
        raise ValueError(f"input checksum differs: {path}")


def _strict_probability(values: Any, label: str) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if (not np.isfinite(probability).all()
            or np.any((probability <= 0.0) | (probability >= 1.0))):
        raise ValueError(f"{label} must contain finite strict-interior probabilities")
    return probability


def blend_series_probabilities(
    native_probability: Any,
    rating_probability: Any,
    alpha: float,
) -> np.ndarray:
    """Blend series probabilities in logit space, with no clipping or intercept."""
    native_probability = _strict_probability(native_probability, "native probability")
    rating_probability = _strict_probability(rating_probability, "rating probability")
    if native_probability.shape != rating_probability.shape:
        raise ValueError("blend probability shapes differ")
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    alpha = float(alpha)
    if alpha == 0.0:
        return native_probability.copy()
    if alpha == 1.0:
        return rating_probability.copy()
    blended = expit((1.0 - alpha) * logit(native_probability) + alpha * logit(rating_probability))
    return _strict_probability(blended, "blended probability")


def _log_loss(labels: Any, probability: Any) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = _strict_probability(probability, "log-loss probability")
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary labels must match probability shape")
    return float(losses(y, p).mean())


def _fit_alpha_record(native_probability: Any, rating_probability: Any, labels: Any) -> dict[str, Any]:
    native_probability = _strict_probability(native_probability, "native CAL probability")
    rating_probability = _strict_probability(rating_probability, "rating CAL probability")
    y = np.asarray(labels, dtype=np.float64)
    if (native_probability.ndim != 1 or rating_probability.shape != native_probability.shape
            or y.shape != native_probability.shape or len(y) == 0
            or not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all()):
        raise ValueError("alpha fitting requires a nonempty matched binary CAL cohort")
    native_logit = logit(native_probability)
    logit_delta = logit(rating_probability) - native_logit

    def objective(alpha: float) -> float:
        return _log_loss(y, expit(native_logit + alpha * logit_delta))

    endpoint_zero = _log_loss(y, native_probability)
    endpoint_one = _log_loss(y, rating_probability)
    optimized = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not optimized.success or not np.isfinite(optimized.fun) or not 0.0 <= optimized.x <= 1.0:
        raise ValueError("bounded alpha minimization did not converge")

    interior_alpha = float(optimized.x)
    interior_loss = objective(interior_alpha)
    # Endpoints are independent candidates, not merely implicit optimizer limits.
    minimum = min(endpoint_zero, endpoint_one, interior_loss)
    tie_tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(minimum))
    if endpoint_zero <= minimum + tie_tolerance:
        alpha, selected_loss, selected = 0.0, endpoint_zero, "alpha_zero_endpoint"
    elif endpoint_one <= minimum + tie_tolerance:
        alpha, selected_loss, selected = 1.0, endpoint_one, "alpha_one_endpoint"
    else:
        alpha, selected_loss, selected = interior_alpha, interior_loss, "bounded_interior"
    return {
        "alpha": float(alpha),
        "method": "bounded_scalar_binary_log_loss_with_explicit_endpoints",
        "bounds": [0.0, 1.0],
        "intercept": False,
        "cal_rows": int(len(y)),
        "cal_log_loss": float(selected_loss),
        "endpoint_alpha_0_log_loss": float(endpoint_zero),
        "endpoint_alpha_1_log_loss": float(endpoint_one),
        "bounded_optimizer_alpha": interior_alpha,
        "bounded_optimizer_log_loss": float(interior_loss),
        "selected_candidate": selected,
        "optimizer_success": bool(optimized.success),
    }


def fit_alpha(native_probability: Any, rating_probability: Any, labels: Any) -> float:
    """Fit one annual alpha; compare alpha=0 and alpha=1 explicitly."""
    return float(_fit_alpha_record(native_probability, rating_probability, labels)["alpha"])


def _require_identical_membership(expected_ids: Any, actual_ids: Any) -> None:
    """Reject missing, extra, duplicate, or null row identities (order may differ)."""
    def normalize(values: Any) -> list[str]:
        normalized = []
        for value in values:
            if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
                raise ValueError("CAL membership differs; exact complete membership is required")
            token = str(value)
            if token in ("", "None", "nan", "NaT", "<NA>"):
                raise ValueError("CAL membership differs; exact complete membership is required")
            normalized.append(token)
        return normalized

    expected = normalize(expected_ids)
    actual = normalize(actual_ids)
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError("CAL membership differs; exact complete membership is required")


def _family_ids_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _same_identity(left: pd.DataFrame, right: pd.DataFrame, keys: tuple[str, ...], label: str) -> None:
    missing = sorted((set(keys) - set(left.columns)) | (set(keys) - set(right.columns)))
    if missing:
        raise ValueError(f"{label} identity fields are missing: {missing}")
    for key in keys:
        a, b = left[key].tolist(), right[key].tolist()
        if len(a) != len(b):
            raise ValueError(f"{label} row counts differ on {key}")
        for first, second in zip(a, b):
            if isinstance(first, (list, tuple, np.ndarray)) or isinstance(second, (list, tuple, np.ndarray)):
                same = tuple(map(str, first)) == tuple(map(str, second))
            elif key in ("y", "best_of"):
                same = int(first) == int(second)
            elif key in ("date", "result_day"):
                same = pd.Timestamp(first).normalize() == pd.Timestamp(second).normalize()
            else:
                same = str(first) == str(second)
            if not same:
                raise ValueError(f"{label} identity/orientation mismatch: {key}")


def _verify_source_snapshot(snapshot_path: Path, source_hashes: dict[str, str]) -> None:
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"preserved v2 assessment source snapshot missing: {snapshot_path}")
    with zipfile.ZipFile(snapshot_path) as archive:
        names = set(archive.namelist())
        for relative, digest in source_hashes.items():
            if relative not in names:
                raise ValueError(f"preserved assessment source missing from archive: {relative}")
            if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                raise ValueError(f"preserved assessment source differs from protocol: {relative}")


def _verify_design(path: Path) -> dict[str, Any]:
    return _json(path, DESIGN_SHA256)


def _check_assessment_trigger(
    evaluation_protocol_path: Path,
    assessment_completion_path: Path,
    research_root: Path,
    design: dict[str, Any],
) -> dict[str, Any]:
    evaluation_protocol_path = Path(evaluation_protocol_path).resolve(strict=True)
    assessment_completion_path = Path(assessment_completion_path).resolve(strict=True)
    protocol = _json(evaluation_protocol_path)
    if (_sha256(evaluation_protocol_path) != EVALUATION_PROTOCOL_SHA256
            or protocol.get("schema") != "model_reliability_evaluation_v1"
            or protocol.get("run_id") != RUN_ID
            or protocol.get("neural_protocol_sha256") != design["training_protocol_sha256"]
            or protocol.get("scope") != "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE"
            or protocol.get("evaluation", {}).get("resamples") != 5000
            or protocol.get("evaluation", {}).get("seed") != 20261001
            or protocol.get("evaluation", {}).get("probability_epsilon") is not None):
        raise ValueError("the exact frozen v2 assessment protocol is required")
    if protocol.get("canonical_path") != "rating-foundation-20260915/data/08_reporting/paired.parquet":
        raise ValueError("assessment canonical population differs")
    if protocol.get("family_completion_sha256") != "38860e1467c0828daa0c6f499d7c97b2a09c56ff73a6be74c989905897fdd23c":
        raise ValueError("the frozen family completion receipt differs")
    if protocol.get("family_protocol_sha256") != "2941500ab545f23daaf6bf0a9caf9c10960f69b66609a08278f3e3fac599b7d8":
        raise ValueError("the frozen family protocol differs")
    source_snapshot = PROJECT_ROOT / "data/05_model_input/model_reliability" / RUN_ID / "evaluation_sources_v2.zip"
    _verify_source_snapshot(source_snapshot, protocol.get("source_sha256", {}))

    completion = _json(assessment_completion_path)
    protocol_sha256 = _sha256(evaluation_protocol_path)
    if (completion.get("run_id") != RUN_ID
            or completion.get("status") != "DEVELOPMENT_ONLY"
            or completion.get("protocol_sha256") != protocol_sha256
            or completion.get("n") != 11550
            or completion.get("conditional_a0_family_integration") != "REQUIRED_NEXT"):
        raise ValueError("complete matching 11,550-row v2 assessment is required")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("assessment completion must bind its complete artifact inventory")
    for relative, expected in artifacts.items():
        _require_hash(PROJECT_ROOT / relative, expected)

    assessment_dir = assessment_completion_path.parent
    evaluation_path = assessment_dir / "evaluation.json"
    if artifacts.get(str(evaluation_path.relative_to(PROJECT_ROOT))) != _sha256(evaluation_path):
        raise ValueError("assessment evaluation artifact is not bound by completion")
    evaluation = _json(evaluation_path)
    gate = evaluation.get("gates", {}).get("family", {})
    contrast = evaluation.get("primary_contrasts", {}).get("family", {})
    logloss = contrast.get("log_loss_monthly", {})
    brier = contrast.get("brier_monthly", {})
    if (evaluation.get("protocol_sha256") != protocol_sha256
            or evaluation.get("run_id") != RUN_ID
            or evaluation.get("n") != 11550
            or evaluation.get("status") != "DEVELOPMENT_ONLY"
            or not gate.get("primary_improvement")
            or gate.get("primary_profile") != "all"
            or completion.get("gates", {}).get("family") != gate
            or logloss.get("n") != 11550
            or logloss.get("blocks") != 33
            or logloss.get("draws") != 5000
            or logloss.get("seed") != 20261001
            or not reliability_study._improvement(contrast, protocol["evaluation"])
            or brier.get("n") != 11550
            or brier.get("blocks") != 33
            or brier.get("draws") != 5000
            or brier.get("seed") != 20261001):
        raise ValueError("the exact frozen family primary improvement gate did not pass")

    root_canonical = research_root / protocol["canonical_path"]
    _require_hash(root_canonical, protocol["canonical_sha256"])
    canonical = pd.read_parquet(root_canonical)
    joined_path = assessment_dir / "joined.parquet"
    joined = pd.read_parquet(joined_path)
    if (len(canonical) != 11550 or canonical.golgg_match_id.astype(str).duplicated().any()
            or len(joined) != 11550 or joined.golgg_match_id.astype(str).duplicated().any()):
        raise ValueError("complete unique canonical assessment coverage is required")
    canonical = canonical.copy()
    joined = joined.copy()
    canonical["golgg_match_id"] = canonical.golgg_match_id.astype(str)
    joined["golgg_match_id"] = joined.golgg_match_id.astype(str)
    _require_identical_membership(canonical.golgg_match_id.tolist(), joined.golgg_match_id.tolist())
    canonical_by_id = canonical.set_index("golgg_match_id").loc[joined.golgg_match_id]
    _same_identity(joined, canonical_by_id.reset_index(),
                   ("golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y"),
                   "canonical assessment")

    family_protocol = _json(PROJECT_ROOT / protocol["family_protocol"], protocol["family_protocol_sha256"])
    family_receipt_path = PROJECT_ROOT / protocol["family_completion"]
    family_receipt = _json(family_receipt_path, protocol["family_completion_sha256"])
    if (family_protocol.get("schema") != "family_reliability_protocol_v1"
            or family_protocol.get("run_id") != FAMILY_RUN_ID):
        raise ValueError("the frozen family protocol identity differs")
    reliability_study._check_family_receipt(family_receipt, protocol)
    family_paths = family_receipt["output_paths"]
    family = pd.read_parquet(family_paths["predictions.parquet"])
    family["golgg_match_id"] = family.golgg_match_id.astype(str)
    _require_identical_membership(joined.golgg_match_id.tolist(), family.golgg_match_id.tolist())
    aligned_family = family.set_index("golgg_match_id").loc[joined.golgg_match_id].reset_index()
    family_identity = ("golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "roster_a", "roster_b")
    _same_identity(joined, aligned_family, family_identity, "full-population family predictions")
    for column in ("p_player_raw", "p_family_raw", "p_player", "p_family"):
        values = _strict_probability(aligned_family[column].to_numpy(float), f"family {column}")
        if column in joined and not np.array_equal(joined[column].to_numpy(float), values):
            raise ValueError(f"assessment and frozen family prediction differ: {column}")

    # Verify the serialized annual population and that its source transform is causal.
    calibrators_path = Path(family_paths["family_calibrators.json"])
    calibrators = _json(calibrators_path)
    if (calibrators.get("schema") != "family_annual_calibrators_v1"
            or calibrators.get("run_id") != FAMILY_RUN_ID
            or calibrators.get("calibration_applies_to") != "online_algorithm_not_frozen_weights"):
        raise ValueError("frozen annual family calibration provenance differs")
    family_calibration = pd.read_parquet(family_paths["annual_calibration_predictions.parquet"])
    for year in YEARS:
        record = calibrators.get("calibrators", {}).get(str(year), {})
        member = family_calibration.loc[family_calibration.calibration_for_year.eq(year)]
        member_ids = member.golgg_match_id.astype(str).tolist()
        origin = pd.Timestamp(year=year, month=1, day=1)
        event_dates = pd.to_datetime(member.date, errors="raise")
        release_dates = pd.to_datetime(member.result_day, errors="raise")
        if (len(member_ids) != record.get("calibration_rows")
                or _family_ids_hash(sorted(member_ids)) != record.get("calibration_ids_sha256")
                or member_ids == []
                or not member.calibration_origin.eq(origin.strftime("%Y-%m-%d")).all()
                or event_dates.dt.year.ne(year - 1).any()
                or release_dates.ge(origin).any()
                or release_dates.max().date().isoformat() != record.get("calibration_end")
                or not np.isin(member.y.to_numpy(dtype=int), (0, 1)).all()):
            raise ValueError(f"annual family temperature CAL membership or availability differs for {year}")
        for kind in ("player", "family"):
            _strict_probability(member[f"p_{kind}_raw"].to_numpy(float), f"family {kind} temperature CAL {year}")
        if (record.get("origin") != f"{year}-01-01"
                or record.get("calibration_year") != year - 1
                or record.get("player", {}).get("intercept", "missing") is not None
                or record.get("family", {}).get("intercept", "missing") is not None):
            raise ValueError(f"annual family scalar temperature provenance differs for {year}")
        rows = aligned_family.loc[pd.to_datetime(aligned_family.date).dt.year.eq(year)]
        for kind in ("player", "family"):
            temperature = float(record[kind]["temperature"])
            if not np.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(f"invalid {kind} scalar temperature for {year}")
            expected = expit(logit(_strict_probability(rows[f"p_{kind}_raw"].to_numpy(float), f"raw {kind}")) / temperature)
            actual = _strict_probability(rows[f"p_{kind}"].to_numpy(float), f"calibrated {kind}")
            if not np.allclose(expected, actual, rtol=0.0, atol=2e-12):
                raise ValueError(f"annual {kind} probabilities do not match the frozen temperature for {year}")

    # Load the saved neural predictions through the frozen assessment's own complete-row aligner.
    saved_neural = {
        column: joined[column].to_numpy(float).copy()
        for column in joined.columns if column.startswith(("control__", "supervised_18__"))
        and "__" in column
    }
    reliability_study._load_neural(joined, protocol)
    for column, expected in saved_neural.items():
        if not np.array_equal(joined[column].to_numpy(float), expected):
            raise ValueError(f"saved annual neural assessment changed during alignment: {column}")
    if "control__full__full" not in joined:
        raise ValueError("fixed native12 control prediction is missing")
    _strict_probability(joined["control__full__full"].to_numpy(float), "native12 full-context assessment")
    return {
        "protocol": protocol,
        "completion": completion,
        "evaluation": evaluation,
        "joined": joined,
        "canonical": canonical,
        "family_protocol": family_protocol,
        "family_receipt": family_receipt,
        "family": family,
        "family_aligned": aligned_family,
        "family_calibrators": calibrators,
        "family_paths": family_paths,
        "assessment_dir": assessment_dir,
        "evaluation_protocol_path": evaluation_protocol_path,
        "assessment_completion_path": assessment_completion_path,
        "design": design,
        "source_snapshot": source_snapshot,
    }


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(PROJECT_ROOT / relative) for relative in INTEGRATION_SOURCES}


def _receipt_artifact_paths(receipt: dict[str, Any]) -> list[Path]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("frozen completion receipt has no artifact inventory")
    return [PROJECT_ROOT / relative for relative in artifacts]


def _collect_input_hashes(context: dict[str, Any], research_root: Path) -> dict[str, str]:
    paths: set[Path] = {
        PROJECT_ROOT / DESIGN_RELATIVE,
        context["evaluation_protocol_path"],
        context["assessment_completion_path"],
        context["source_snapshot"],
        research_root / context["protocol"]["canonical_path"],
        PROJECT_ROOT / context["protocol"]["neural_protocol"],
        PROJECT_ROOT / context["protocol"]["family_protocol"],
        PROJECT_ROOT / context["protocol"]["family_completion"],
        PROJECT_ROOT / context["protocol"]["family_predictions"],
    }
    recipe_path = Path(context["family_protocol"]["recipe_path"]).resolve()
    if _sha256(recipe_path) != context["family_protocol"]["recipe_sha256"]:
        raise ValueError("frozen family pre-fit recipe changed")
    paths.add(recipe_path)
    for relative, expected in context["family_protocol"].get("inputs_sha256", {}).items():
        if context["family_receipt"]["inputs_sha256"].get(relative) != expected:
            raise ValueError(f"family input receipt hash differs: {relative}")
    paths.update(PROJECT_ROOT / key for key in context["completion"]["artifacts"])
    paths.update(Path(path).resolve() for path in context["family_receipt"]["input_paths"].values())
    paths.update(Path(path).resolve() for path in context["family_receipt"]["output_paths"].values())
    paths.update(PROJECT_ROOT / relative for relative in context["family_protocol"].get("sources_sha256", {}))
    neural_protocol = _json(PROJECT_ROOT / context["protocol"]["neural_protocol"])
    for relative in neural_protocol.get("inputs", {}):
        paths.add(research_root / relative)
    neural_completion_path = PROJECT_ROOT / "data/08_reporting/model_reliability" / RUN_ID / "neural_completed.json"
    paths.add(neural_completion_path)
    neural_completion = _json(neural_completion_path, context["protocol"]["neural_completion_sha256"])
    for origin in neural_completion.get("origins", {}).values():
        receipt_path = PROJECT_ROOT / origin["completed_path"]
        paths.add(receipt_path)
        receipt = _json(receipt_path, origin["completed_sha256"])
        paths.update(_receipt_artifact_paths(receipt))
    reference = neural_protocol["reference_2026"]
    reference_completion_path = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / reference["run_id"] / "completed.json"
    reference_completion = _json(reference_completion_path, reference["completed_sha256"])
    paths.add(reference_completion_path)
    paths.update(_receipt_artifact_paths(reference_completion))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"integration evidence input missing: {missing[0]}")
    return {str(path.resolve()): _sha256(path) for path in sorted(paths, key=str)}


def _validate_frozen_integration(protocol_path: Path, research_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = Path(protocol_path).resolve(strict=True)
    if protocol_path.parent != (PROJECT_ROOT / DESIGN_RELATIVE.parent).resolve():
        raise ValueError("conditional protocol must remain beside the frozen run design")
    protocol = _json(protocol_path)
    if (protocol.get("schema") != "model_reliability_conditional_integration_v1"
            or protocol.get("run_id") != RUN_ID
            or protocol.get("status") != "FROZEN_AFTER_REQUIRED_FAMILY_GATE"
            or protocol.get("design_path") != str(DESIGN_RELATIVE)
            or protocol.get("design_sha256") != DESIGN_SHA256
            or protocol.get("evaluation_protocol_sha256") != EVALUATION_PROTOCOL_SHA256
            or protocol.get("training_protocol_sha256") != "30e917883631701e9460b936c2ac3f2fb686d2433daed4b956bc2014d8fc4837"
            or protocol.get("scope") != "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE"
            or protocol.get("base") != "fixed native12 control from the annual neural protocol; never supervised_18"
            or protocol.get("arms") != ["native12_alone", "native12_plus_player", "native12_plus_family"]
            or protocol.get("assembly") != "sigmoid((1-alpha)*logit(p_native12_full)+alpha*logit(p_rating))"
            or protocol.get("origins") != list(YEARS)):
        raise ValueError("unsupported or unfrozen conditional integration protocol")
    design = _verify_design(PROJECT_ROOT / protocol["design_path"])
    if (protocol.get("alpha") != design["alpha"]
            or protocol.get("native_calibration") != "frozen control beta calibration, expert slopes and constant mixture; CAL reused, explicitly not cross-fitted"
            or protocol.get("rating_calibration") != "frozen annual scalar temperature applied to causal raw CAL forecasts from the exact native CAL membership"
            or protocol.get("fit_population") != "exact native annual CAL membership for each origin; no missing forecasts or silent intersection"
            or protocol.get("output_policy") != "new exclusive output directory; every file hash recorded in completed evidence receipt"
            or protocol.get("research_root") != str(research_root)):
        raise ValueError("conditional integration recipe differs from frozen design")
    expected_evaluation = {
        "primary_population": "all 11550 canonical 2024-2026 rows",
        "primary_contrasts": list(design["primary_contrasts"]),
        "resamples": int(design["resamples"]),
        "seed": int(design["seed"]),
        "probability_epsilon": None,
        "critical_profiles": list(CRITICAL),
        "calibration_scope": "full-context overall and critical profiles; aggregate support, not per-match certainty",
        "tournament_name_year_sensitivity": "tournament name plus calendar year proxy; not verified independent editions",
    }
    if protocol.get("evaluation") != expected_evaluation:
        raise ValueError("conditional integration evaluation contract differs")
    if protocol.get("source_sha256") != _source_hashes():
        raise ValueError("integration source changed after protocol freeze")
    for filename, expected in protocol.get("input_sha256", {}).items():
        _require_hash(Path(filename), expected)
    context = _check_assessment_trigger(
        PROJECT_ROOT / protocol["evaluation_protocol_path"],
        PROJECT_ROOT / protocol["assessment_completion_path"],
        research_root,
        design,
    )
    frozen_trigger = protocol.get("trigger", {})
    if (protocol.get("assessment_artifacts") != context["completion"]["artifacts"]
            or protocol.get("family_completion_sha256") != context["protocol"]["family_completion_sha256"]
            or frozen_trigger.get("family_primary_gate") != context["evaluation"]["gates"]["family"]
            or frozen_trigger.get("family_primary_contrast") != context["evaluation"]["primary_contrasts"]["family"]
            or _collect_input_hashes(context, research_root) != protocol.get("input_sha256")):
        raise ValueError("frozen assessment inputs or family trigger changed")
    if (_sha256(context["evaluation_protocol_path"]) != protocol["evaluation_protocol_sha256"]
            or _sha256(context["assessment_completion_path"]) != protocol["assessment_completion_sha256"]):
        raise ValueError("assessment receipt changed after conditional protocol freeze")
    return protocol, context


def freeze_integration(
    run_id: str,
    evaluation_protocol: str | Path,
    assessment_completion: str | Path,
    *,
    output_name: str = "integration_protocol.json",
    research_root: str | Path | None = None,
) -> dict[str, str]:
    """Freeze conditional integration only after the exact family trigger verifies."""
    if run_id != RUN_ID or Path(run_id).name != run_id:
        raise ValueError("only the frozen 2026-10-01 reliability run can trigger this integration")
    if Path(output_name).name != output_name or output_name in ("", ".", ".."):
        raise ValueError("protocol output name must be a file name")
    root = native_focus.resolve_research_root(research_root)
    design_path = PROJECT_ROOT / DESIGN_RELATIVE
    design = _verify_design(design_path)
    context = _check_assessment_trigger(Path(evaluation_protocol), Path(assessment_completion), root, design)
    destination = design_path.parent / output_name
    protocol = {
        "schema": "model_reliability_conditional_integration_v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_AFTER_REQUIRED_FAMILY_GATE",
        "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE",
        "design_path": str(DESIGN_RELATIVE),
        "design_sha256": DESIGN_SHA256,
        "evaluation_protocol_path": str(context["evaluation_protocol_path"].relative_to(PROJECT_ROOT)),
        "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
        "assessment_completion_path": str(context["assessment_completion_path"].relative_to(PROJECT_ROOT)),
        "assessment_completion_sha256": _sha256(context["assessment_completion_path"]),
        "assessment_artifacts": context["completion"]["artifacts"],
        "family_completion_sha256": context["protocol"]["family_completion_sha256"],
        "training_protocol_sha256": design["training_protocol_sha256"],
        "research_root": str(root),
        "base": "fixed native12 control from the annual neural protocol; never supervised_18",
        "arms": list(design["arms"]),
        "assembly": design["assembly"],
        "alpha": design["alpha"],
        "origins": list(YEARS),
        "native_calibration": "frozen control beta calibration, expert slopes and constant mixture; CAL reused, explicitly not cross-fitted",
        "rating_calibration": "frozen annual scalar temperature applied to causal raw CAL forecasts from the exact native CAL membership",
        "fit_population": "exact native annual CAL membership for each origin; no missing forecasts or silent intersection",
        "evaluation": {
            "primary_population": "all 11550 canonical 2024-2026 rows",
            "primary_contrasts": list(design["primary_contrasts"]),
            "resamples": design["resamples"],
            "seed": design["seed"],
            "probability_epsilon": None,
            "critical_profiles": list(CRITICAL),
            "calibration_scope": "full-context overall and critical profiles; aggregate support, not per-match certainty",
            "tournament_name_year_sensitivity": "tournament name plus calendar year proxy; not verified independent editions",
        },
        "trigger": {
            "family_primary_gate": context["evaluation"]["gates"]["family"],
            "family_primary_contrast": context["evaluation"]["primary_contrasts"]["family"],
        },
        "source_sha256": _source_hashes(),
        "input_sha256": _collect_input_hashes(context, root),
        "output_policy": "new exclusive output directory; every file hash recorded in completed evidence receipt",
        "prospective_confirmation": "NOT_RUN",
        "production_promotion": "NOT_AUTHORIZED",
    }
    _write_json_exclusive(destination, protocol)
    return {"protocol": str(destination), "sha256": _sha256(destination)}


def _load_native_calibration_data(research_root: Path, context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct native CAL probabilities from persisted control logits/checkpoints."""
    neural_protocol = _json(PROJECT_ROOT / context["protocol"]["neural_protocol"],
                            context["protocol"]["neural_protocol_sha256"])
    data = uncertainty._load_training_data(research_root)
    if (annual_native._inputs(data, research_root) != neural_protocol["inputs"]
            or data["bank_receipt_sha256"] != neural_protocol["bank_receipt_sha256"]):
        raise ValueError("native CAL input identities differ from frozen neural protocol")
    frame = data["frame"]
    splits = {year: annual_native.annual_partitions(frame, year) for year in YEARS}
    ids_by_year: dict[int, list[str]] = {}
    for year in YEARS:
        split = splits[year]
        protocol_partition = neural_protocol["partitions"][str(year)]
        for part in ("train", "cal", "validation"):
            ids = frame.iloc[split[part]].golgg_match_id.astype(str).tolist()
            if (len(ids) != protocol_partition["counts"][part]
                    or _contract_digest(ids) != protocol_partition["membership_sha256"][part]):
                raise ValueError(f"native {part} membership changed for origin {year}")
        ids_by_year[year] = frame.iloc[split["cal"]].golgg_match_id.astype(str).tolist()

    # Feature/identity construction above deliberately precedes archive namespace activation.
    backend, backend_hashes = uncertainty._activate_archive_backend(research_root)
    focus_sources = native_focus._source_hashes(backend_hashes)
    annual_sources = annual_native._sources(backend_hashes)
    neural_completed_path = PROJECT_ROOT / "data/08_reporting/model_reliability" / RUN_ID / "neural_completed.json"
    neural_completed = _json(neural_completed_path, context["protocol"]["neural_completion_sha256"])
    cal_rows: list[pd.DataFrame] = []
    for year in YEARS:
        split = splits[year]
        cal_indices = split["cal"]
        validation_indices = split["validation"]
        score_indices = np.concatenate((cal_indices, validation_indices))
        expected_cal_ids = ids_by_year[year]
        if year == 2026:
            reference = neural_protocol["reference_2026"]
            reference_completion_path = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / reference["run_id"] / "completed.json"
            reference_completion = _json(reference_completion_path, reference["completed_sha256"])
            reference_contract = reference_completion["contract"]
            if (reference_completion.get("status") != "TRAINING_COMPLETE_NOT_SCORED"
                    or reference_contract.get("split_ids_sha256") != neural_protocol["partitions"]["2026"]["membership_sha256"]
                    or native_focus._relative_input_hashes(reference_contract["input_sha256"], research_root)
                    != annual_native._inputs(data, research_root)
                    or reference_contract.get("source_sha256") != focus_sources):
                raise ValueError("the frozen 2026 native reference contract differs")
            model_root = PROJECT_ROOT / "data/06_models/a0_data_focus" / reference["run_id"]
            output_root = PROJECT_ROOT / "data/07_model_output/a0_data_focus" / reference["run_id"]
            variant = "control"
            expected_contract = {**reference_contract, "component": "calibration", "variant": variant}
        else:
            origin = neural_completed["origins"][str(year)]
            receipt_path = PROJECT_ROOT / origin["completed_path"]
            receipt = _json(receipt_path, origin["completed_sha256"])
            annual_contract = receipt["contract"]
            if (receipt.get("status") != "TRAINING_COMPLETE_NOT_SCORED"
                    or annual_contract.get("protocol_sha256") != context["protocol"]["neural_protocol_sha256"]
                    or annual_contract.get("run_id") != RUN_ID
                    or annual_contract.get("year") != year
                    or annual_contract.get("split_ids_sha256") != neural_protocol["partitions"][str(year)]["membership_sha256"]
                    or annual_contract.get("source_sha256") != annual_sources):
                raise ValueError(f"frozen annual native control receipt differs for {year}")
            model_root = PROJECT_ROOT / "data/06_models/model_reliability" / RUN_ID / "neural" / str(year)
            output_root = PROJECT_ROOT / "data/07_model_output/model_reliability" / RUN_ID / "neural" / str(year)
            variant = "control"
            expected_contract = {**annual_contract, "component": "calibration", "variant": variant}

        calibration = native_focus._load_checkpoint(
            model_root / variant / "calibration/model.joblib", expected_contract,
        )
        logits_path = output_root / "control_logits.npz"
        shared_logits_path = output_root / "shared_logits.npz"
        with np.load(logits_path, allow_pickle=False) as saved:
            saved_indices = saved["indices"]
            neural_logits = saved["logits"]
        with np.load(shared_logits_path, allow_pickle=False) as saved:
            shared_indices = saved["indices"]
            expert_logits = saved["other_logits"]
        if (not np.array_equal(saved_indices, score_indices)
                or not np.array_equal(shared_indices, score_indices)
                or neural_logits.shape[:2] != (len(score_indices), 3)
                or expert_logits.shape != (len(score_indices), 3, 4)):
            raise ValueError(f"saved native CAL logit membership differs for {year}")
        n_cal = len(cal_indices)
        best_of = frame.best_of.to_numpy(dtype=np.int64)[cal_indices]
        gates = np.asarray(data["raw"]["gates"])[cal_indices]
        native_full, _, _ = native_focus._predict_mixture(
            backend,
            neural_logits[:n_cal],
            expert_logits[:n_cal],
            calibration["slopes"],
            gates,
            best_of,
            calibration,
        )
        native_probability = _strict_probability(native_full[:, 0], f"native CAL full probability {year}")
        cal = frame.iloc[cal_indices].copy().reset_index(drop=True)
        cal["golgg_match_id"] = cal.golgg_match_id.astype(str)
        cal["origin"] = int(year)
        cal["p_native12_full"] = native_probability
        cal["cal_position"] = np.arange(len(cal), dtype=np.int64)
        if cal.golgg_match_id.tolist() != expected_cal_ids:
            raise ValueError(f"native CAL ordering changed for {year}")
        cal_rows.append(cal)
    return data["frame"], pd.concat(cal_rows, ignore_index=True)


def _calibrate_rating_cal(
    native_cal: pd.DataFrame,
    family_calibration: pd.DataFrame,
    family_calibrators: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = (
        "golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y",
        "roster_a", "roster_b", "p_player_raw", "p_family_raw", "calibration_for_year",
        "calibration_origin", "state_receipt_sha256", "roster_receipt_sha256",
    )
    missing = sorted(set(required) - set(family_calibration.columns))
    if missing:
        raise ValueError(f"causal family CAL forecasts are missing fields: {missing}")
    if family_calibration.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("causal family CAL forecast ids are duplicated")
    family_calibration = family_calibration.copy()
    family_calibration["golgg_match_id"] = family_calibration.golgg_match_id.astype(str)
    output: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    identity = ("golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "roster_a", "roster_b")
    for year in YEARS:
        expected = native_cal.loc[native_cal.origin.eq(year)].copy()
        ids = expected.golgg_match_id.astype(str).tolist()
        member = family_calibration.loc[family_calibration.calibration_for_year.eq(year)].copy()
        # Rating temperatures use a full prior year; alpha uses only native CAL.
        # Select that declared subset, then require every requested identity.
        member = member.loc[member.golgg_match_id.isin(ids)]
        member_ids = member.golgg_match_id.astype(str).tolist()
        _require_identical_membership(ids, member_ids)
        by_id = member.set_index("golgg_match_id").loc[ids].reset_index()
        _same_identity(expected, by_id, identity, f"native/family CAL origin {year}")
        origin = pd.Timestamp(year=year, month=1, day=1)
        event_dates = pd.to_datetime(by_id.date, errors="raise")
        release_dates = pd.to_datetime(by_id.result_day, errors="raise")
        if (not by_id.calibration_origin.eq(origin.strftime("%Y-%m-%d")).all()
                or (event_dates >= origin).any() or (release_dates >= origin).any()
                or by_id.state_receipt_sha256.isna().any() or by_id.roster_receipt_sha256.isna().any()):
            raise ValueError(f"family CAL forecasts are not causal and receipt-bound for {year}")
        record = family_calibrators["calibrators"][str(year)]
        if (record["calibration_year"] != year - 1
                or record["origin"] != origin.strftime("%Y-%m-%d")):
            raise ValueError(f"family scalar calibration origin differs for {year}")
        for kind in ("player", "family"):
            raw = _strict_probability(by_id[f"p_{kind}_raw"].to_numpy(float), f"{kind} raw CAL {year}")
            temperature = float(record[kind]["temperature"])
            if not np.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(f"invalid frozen {kind} temperature for {year}")
            calibrated = _strict_probability(expit(logit(raw) / temperature), f"{kind} calibrated CAL {year}")
            expected[f"p_{kind}_raw"] = raw
            expected[f"p_{kind}"] = calibrated
        expected["family_state_receipt_sha256"] = by_id.state_receipt_sha256.to_numpy()
        expected["family_roster_receipt_sha256"] = by_id.roster_receipt_sha256.to_numpy()
        expected["player_temperature"] = float(record["player"]["temperature"])
        expected["family_temperature"] = float(record["family"]["temperature"])
        expected["family_calibration_rows"] = int(record["calibration_rows"])
        expected["family_calibration_ids_sha256"] = record["calibration_ids_sha256"]
        output.append(expected)
        provenance[str(year)] = {
            "origin": record["origin"],
            "prior_calendar_year": int(record["calibration_year"]),
            "prior_calendar_year_calibration_rows": int(record["calibration_rows"]),
            "prior_calendar_year_calibration_ids_sha256": record["calibration_ids_sha256"],
            "calibration_end": record["calibration_end"],
            "player_temperature": float(record["player"]["temperature"]),
            "family_temperature": float(record["family"]["temperature"]),
            "temperature_fits_have_intercept": False,
            "cal_membership_sha256": _contract_digest(ids),
            "cal_membership_rows": len(ids),
            "native_calibration": "frozen native control calibration and mixture; same CAL rows reused, not cross-fitted",
        }
    return pd.concat(output, ignore_index=True), provenance




def _evaluate_artifacts(context: dict[str, Any], native_cal: pd.DataFrame) -> dict[str, Any]:
    joined = context["joined"].copy()
    joined["golgg_match_id"] = joined.golgg_match_id.astype(str)
    cal_family_path = Path(context["family_paths"]["annual_calibration_predictions.parquet"])
    family_cal_raw = pd.read_parquet(cal_family_path)
    calibrated_cal, temperature_provenance = _calibrate_rating_cal(
        native_cal,
        family_cal_raw,
        context["family_calibrators"],
    )
    alphas: dict[str, Any] = {}
    cal_output_rows: list[pd.DataFrame] = []
    for year in YEARS:
        cal = calibrated_cal.loc[calibrated_cal.origin.eq(year)].copy()
        base = _strict_probability(cal.p_native12_full.to_numpy(float), f"native CAL {year}")
        labels = cal.y.to_numpy(dtype=int)
        player = _strict_probability(cal.p_player.to_numpy(float), f"player CAL {year}")
        family = _strict_probability(cal.p_family.to_numpy(float), f"family CAL {year}")
        player_fit = _fit_alpha_record(base, player, labels)
        family_fit = _fit_alpha_record(base, family, labels)
        player_blend = blend_series_probabilities(base, player, player_fit["alpha"])
        family_blend = blend_series_probabilities(base, family, family_fit["alpha"])
        cal["p_native12_plus_player_cal"] = player_blend
        cal["p_native12_plus_family_cal"] = family_blend
        cal_output_rows.append(cal)
        alphas[str(year)] = {
            "native12_alone": {
                "alpha": 0.0,
                "fit_status": "fixed_native_control_no_blend_fit",
                "cal_rows": len(cal),
                "cal_log_loss": _log_loss(labels, base),
            },
            "native12_plus_player": player_fit,
            "native12_plus_family": family_fit,
            "cal_membership_sha256": _contract_digest(cal.golgg_match_id.astype(str).tolist()),
            "cal_membership_rows": len(cal),
            "family_calibration": temperature_provenance[str(year)],
        }

    # Apply each origin's CAL-fitted alpha only to that origin's held-out validation rows.
    year_values = pd.to_datetime(joined.date).dt.year.to_numpy(dtype=int)
    joined["p_native12_full"] = _strict_probability(joined["control__full__full"].to_numpy(float), "fixed native12 control")
    joined["p_player_rating"] = _strict_probability(joined["p_player"].to_numpy(float), "player rating")
    joined["p_family_rating"] = _strict_probability(joined["p_family"].to_numpy(float), "family rating")
    joined["native12_alone"] = joined["p_native12_full"].to_numpy(float)
    joined["native12_plus_player"] = np.empty(len(joined), dtype=np.float64)
    joined["native12_plus_family"] = np.empty(len(joined), dtype=np.float64)
    for year in YEARS:
        mask = year_values == year
        if not mask.any():
            raise ValueError(f"canonical held-out assessment rows are missing for {year}")
        joined.loc[mask, "native12_plus_player"] = blend_series_probabilities(
            joined.loc[mask, "p_native12_full"].to_numpy(float),
            joined.loc[mask, "p_player_rating"].to_numpy(float),
            alphas[str(year)]["native12_plus_player"]["alpha"],
        )
        joined.loc[mask, "native12_plus_family"] = blend_series_probabilities(
            joined.loc[mask, "p_native12_full"].to_numpy(float),
            joined.loc[mask, "p_family_rating"].to_numpy(float),
            alphas[str(year)]["native12_plus_family"]["alpha"],
        )
    for column in ("native12_alone", "native12_plus_player", "native12_plus_family"):
        _strict_probability(joined[column].to_numpy(float), column)

    metric_columns = {
        "native12_alone": "native12_alone",
        "native12_plus_player": "native12_plus_player",
        "native12_plus_family": "native12_plus_family",
        "player_rating": "p_player_rating",
        "family_rating": "p_family_rating",
    }
    metrics = [{"model": model, "context": "full", **_summary(joined, column)}
               for model, column in metric_columns.items()]
    evaluation = context["protocol"]["evaluation"]
    contrasts = {
        "native12_plus_family_minus_native12_plus_player": reliability_study._contrast(
            joined, "native12_plus_family", "native12_plus_player", evaluation,
        ),
        "native12_plus_family_minus_native12_alone": reliability_study._contrast(
            joined, "native12_plus_family", "native12_alone", evaluation,
        ),
        "native12_plus_player_minus_native12_alone": reliability_study._contrast(
            joined, "native12_plus_player", "native12_alone", evaluation,
        ),
    }

    masks = reliability_study._profile_masks(joined, context["protocol"]["family_profiles"])
    calibration: dict[str, Any] = {}
    for model, column in metric_columns.items():
        profile_results = {}
        for profile in ("all", *CRITICAL):
            profile_results[profile] = reliability_study._calibration(
                joined, column, evaluation, masks[profile],
            )
        calibration[model] = {
            "scope": "full context only",
            "profiles": profile_results,
            "interpretation": "aggregate cohort calibration support; not per-match certainty",
        }

    family_base = contrasts["native12_plus_family_minus_native12_alone"]
    family_player = contrasts["native12_plus_family_minus_native12_plus_player"]
    primary = {
        "family_minus_player": {
            "log_loss_ci_upper_below_zero": bool(family_player["log_loss_monthly"]["ci_upper"] is not None
                                                   and family_player["log_loss_monthly"]["ci_upper"] < 0.0),
            "brier_point_delta_at_most_zero": bool(family_player["brier_monthly"]["delta"] <= 0.0),
        },
        "family_minus_base": {
            "log_loss_ci_upper_below_zero": bool(family_base["log_loss_monthly"]["ci_upper"] is not None
                                                   and family_base["log_loss_monthly"]["ci_upper"] < 0.0),
            "brier_point_delta_at_most_zero": bool(family_base["brier_monthly"]["delta"] <= 0.0),
        },
    }
    primary_pass = all(all(parts.values()) for parts in primary.values())
    guard_rows = {}
    for profile in CRITICAL:
        mask = masks[profile]
        selected = joined.loc[mask].copy()
        if selected.empty:
            raise ValueError(f"predeclared critical guard cohort is empty: {profile}")
        contrast = reliability_study._contrast(selected, "native12_plus_family", "native12_alone", evaluation)
        guard = reliability_study._guard(contrast, evaluation)
        guard_rows[profile] = {"status": guard, "contrast": contrast}
    guards_pass = all(item["status"] == "PASS" for item in guard_rows.values())
    gates = {
        "status": "PASS" if primary_pass and guards_pass else "NOT_PASSED",
        "primary_contrasts": primary,
        "critical_noninferiority_guards_vs_native12_alone": guard_rows,
        "primary_pass": bool(primary_pass),
        "critical_guards_pass": bool(guards_pass),
        "success": bool(primary_pass and guards_pass),
        "calibration_is_diagnostic_not_a_per_match_certainty_claim": True,
        "prospective_confirmation": "NOT_RUN",
        "production_promotion": "NOT_AUTHORIZED",
    }
    for column in ("native12_alone", "native12_plus_player", "native12_plus_family"):
        if not np.isfinite(joined[column].to_numpy(float)).all():
            raise ValueError(f"nonfinite full-population prediction in {column}")
    cal_output = pd.concat(cal_output_rows, ignore_index=True)
    return {
        "predictions": joined,
        "cal_probabilities": cal_output,
        "cal_membership": cal_output[
            ["origin", "cal_position", "golgg_match_id", "date", "result_day", "team1_id", "team2_id", "roster_a", "roster_b", "best_of", "y"]
        ].copy(),
        "alphas": alphas,
        "metrics": pd.DataFrame(metrics),
        "contrasts": contrasts,
        "calibration": calibration,
        "gates": gates,
    }


def _write_outputs(output_dir: Path, context: dict[str, Any], protocol_path: Path,
                   artifacts: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    expected_root = (PROJECT_ROOT / "data/08_reporting/model_reliability" / RUN_ID).resolve()
    try:
        relative_output = output_dir.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"integration output must remain under {expected_root}") from exc
    if not relative_output.parts:
        raise ValueError("integration output must be a new child directory")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        artifacts["predictions"].to_parquet(staging / "predictions.parquet", index=False)
        artifacts["cal_probabilities"].to_parquet(staging / "cal_probabilities.parquet", index=False)
        artifacts["cal_membership"].to_parquet(staging / "cal_membership.parquet", index=False)
        artifacts["metrics"].to_csv(staging / "metrics.csv", index=False)
        _write_json_exclusive(staging / "contrasts.json", artifacts["contrasts"])
        _write_json_exclusive(staging / "calibration.json", artifacts["calibration"])
        _write_json_exclusive(staging / "gates.json", artifacts["gates"])
        _write_json_exclusive(staging / "alpha_calibration.json", artifacts["alphas"])
        artifact_hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        frozen_protocol = _json(protocol_path)
        research_root = Path(frozen_protocol["research_root"])
        source_hashes = _source_hashes()
        input_hashes = _collect_input_hashes(context, research_root)
        if source_hashes != frozen_protocol["source_sha256"]:
            raise ValueError("integration source changed while evaluation was running")
        if input_hashes != frozen_protocol["input_sha256"]:
            raise ValueError("integration input changed while evaluation was running")
        receipt = {
            "schema": "model_reliability_conditional_integration_completion_v1",
            "run_id": RUN_ID,
            "status": "DEVELOPMENT_ONLY",
            "integration_protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
            "integration_protocol_sha256": _sha256(protocol_path),
            "assessment_protocol_sha256": _sha256(context["evaluation_protocol_path"]),
            "assessment_completion_sha256": _sha256(context["assessment_completion_path"]),
            "design_sha256": DESIGN_SHA256,
            "n": int(len(artifacts["predictions"])),
            "cal_rows_by_origin": {year: int((artifacts["cal_membership"].origin == year).sum()) for year in YEARS},
            "source_sha256": source_hashes,
            "input_sha256": input_hashes,
            "artifact_sha256": artifact_hashes,
            "gates": artifacts["gates"],
            "blocking_unit": "calendar month; tournament-name plus calendar-year sensitivity is a proxy, not verified independent editions",
            "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_FULL_CONTEXT_ONLY",
            "probability_interpretation": "series probabilities; calibration status is aggregate cohort evidence, not per-match certainty",
            "native_calibration_reused_on_CAL": True,
            "native_CAL_cross_fitted": False,
            "fit_description": "one bounded alpha per annual origin and rating arm; native-only arm is fixed alpha zero",
            "prospective_confirmation": "NOT_RUN",
            "production_promotion": "NOT_AUTHORIZED",
        }
        _write_json_exclusive(staging / "completed.json", receipt)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    completion_path = output_dir / "completed.json"
    receipt["completed_path"] = str(completion_path)
    receipt["completed_sha256"] = _sha256(completion_path)
    return receipt


def evaluate_integration(
    protocol_path: str | Path,
    output_dir: str | Path,
    research_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the frozen annual blending weights and write complete immutable evidence."""
    root = native_focus.resolve_research_root(research_root)
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol, context = _validate_frozen_integration(protocol_path, root)
    native_frame, native_cal = _load_native_calibration_data(root, context)
    if native_frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("native training identities are not unique")
    artifacts = _evaluate_artifacts(context, native_cal)
    receipt = _write_outputs(Path(output_dir), context, protocol_path, artifacts)
    return receipt
