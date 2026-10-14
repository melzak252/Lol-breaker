"""Frozen historical context-reliability persistence and calibration study.

This runner reuses saved series-winner forecasts and fits only reliability
slopes. Its artifacts describe exposed historical development, not live or
prospective-valid forecasts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import zipfile
from itertools import product

import numpy as np
import pandas as pd

from src.models import a0_uncertainty
from src.models.a0_native_refit import _write_json_exclusive
from src.models.a0_context_calibration import ContextSlopeCalibration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REQUIREMENTS_RELATIVE = "conf/base/a0_context_reliability_sources.json"
SOURCE_REQUIREMENTS_PATH = PROJECT_ROOT / SOURCE_REQUIREMENTS_RELATIVE
SOURCE_REQUIREMENTS_BYTES = SOURCE_REQUIREMENTS_PATH.read_bytes()
SOURCE_REQUIREMENTS = json.loads(SOURCE_REQUIREMENTS_BYTES)
SOURCE_REQUIREMENTS_SHA256 = hashlib.sha256(SOURCE_REQUIREMENTS_BYTES).hexdigest()
EXECUTION_CONTRACT_RELATIVE = "conf/base/a0_context_reliability_execution.json"
EXECUTION_CONTRACT_PATH = PROJECT_ROOT / EXECUTION_CONTRACT_RELATIVE
EXECUTION_CONTRACT_BYTES = EXECUTION_CONTRACT_PATH.read_bytes()
EXECUTION_CONTRACT = json.loads(EXECUTION_CONTRACT_BYTES)
EXECUTION_CONTRACT_SHA256 = hashlib.sha256(EXECUTION_CONTRACT_BYTES).hexdigest()
SOURCE_BLOCKED = {
    "schema": "c0_context_reliability_source_activation_summary_v1",
    "activation": SOURCE_REQUIREMENTS["activation"],
    "approved_for_collection": SOURCE_REQUIREMENTS["approved_for_collection"],
    "requirements_artifact": SOURCE_REQUIREMENTS_RELATIVE,
    "requirements_sha256": SOURCE_REQUIREMENTS_SHA256,
}
REQUIRED_RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow")
SOURCE_CLOSURE = (
    "src/models/a0_context_reliability_runner.py",
    "src/models/a0_context_calibration.py",
    "src/analysis/a0_context_reliability.py",
    "src/analysis/production_market_benchmark.py",
    "src/analysis/metrics.py",
    "src/analysis/probability_metrics.py",
    "src/analysis/bootstrap.py",
    "src/analysis/thesis_metrics.py",
    "betting_app/core/ev.py",
    "src/models/a0_uncertainty.py",
    "src/models/a0_correction_experiment.py",
    "src/models/a0_data_focus.py",
    "src/models/a0_reliability.py",
    "src/models/a0_target_experiment.py",
    "src/models/a0_target_objectives.py",
    "src/models/a0_history_reconstruction.py",
    "src/models/competition_tiers.py",
    "src/analysis/ev_approach_data.py",
    "src/analysis/research_benchmark.py",
    SOURCE_REQUIREMENTS_RELATIVE,
    EXECUTION_CONTRACT_RELATIVE,
)
RUN_ID = "context_reliability_20261004_01"
INPUT_CONTRACT = {
    "anchor": "native12__full__full",
    "canonical_manifest": "conf/base/research_benchmark.json",
    "canonical_relative": "rating-foundation-20260915/data/08_reporting/paired.parquet",
    "canonical_sha256": "8041b6c7ffd4a99e4c67d55c3665c22c8cd653c8dffe4f153b10dfd9b264dfaf",
    "joined": "data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment/joined_predictions.parquet",
    "joined_sha256": "1ebfac99fa425a83712f138b37b9a8292e26e546ab7f109b66296cede63c63fb",
    "joined_receipt": "data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment/evaluation_completed.json",
    "native_protocol": "data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json",
    "native_protocol_sha256": "6c1d127a801f6a2075ab44e76646bac2b571b376e6283d4072293578a046b434",
    "release_source": "data/05_model_input/a0_sports_correction/c0_raw_connector_20261002_033658/joined.parquet",
    "release_source_sha256": "eaf18bdd8de8626b5a5318e4c96eee22eed830064a86d618a96a1ade3bba7ab1",
    "release_receipt": "data/05_model_input/a0_sports_correction/c0_raw_connector_20261002_033658/prepared.json",
    "join_keys": ["golgg_match_id", "date", "result_day", "team1_id", "team2_id", "best_of", "y"],
    "rows": 11550,
    "primary_rows": 3103,
}
CONTEXT_NAMES = ("low_statistics", "novel_roster_support", "high_sports_disagreement")
CONDITIONAL_ORIGINS = ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01",
                       "2026-01-01", "2026-04-01", "2026-07-01")
DIAGNOSTIC_ORIGINS = ("2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01")
PENALTIES = (1.0, 10.0, 100.0)
IDENTITY_KEYS = ("golgg_match_id", "date", "result_day", "team1_id", "team2_id", "best_of", "y")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=str).encode()).hexdigest()


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path: Path, value: dict) -> None:
    _write_json_exclusive(path, _safe_json(value))


def _replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_safe_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(path)

def _paths(run_id: str) -> dict[str, Path]:
    return {name: PROJECT_ROOT / f"data/{stage}/a0_context_reliability" / run_id
            for name, stage in (("input", "05_model_input"), ("models", "06_models"),
                                ("output", "07_model_output"), ("report", "08_reporting"))}

def _runtime_versions() -> dict:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": {name: importlib.metadata.version(name) for name in REQUIRED_RUNTIME_PACKAGES},
        "torch": torch_version,
    }


def _verify_live_source_closure(source_hashes: dict[str, str], research_root: Path) -> None:
    expected_keys = {f"project:{relative}" for relative in SOURCE_CLOSURE}
    if set(source_hashes) != expected_keys:
        raise ValueError("frozen source hash closure differs from executed source closure")
    for key, expected in source_hashes.items():
        if key.startswith("research:"):
            path = research_root / key.removeprefix("research:")
        else:
            path = PROJECT_ROOT / key.removeprefix("project:")
        _verify_file(path, expected, f"live executed source {key}")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be one nonempty path component")


def eligible_history_rows(frame: pd.DataFrame, origin: str) -> np.ndarray:
    """Rows whose event, candidate-head, and source-label clocks precede origin."""
    if "date" not in frame or "label_release_day" not in frame:
        raise ValueError("historical membership requires date and authentic label_release_day")
    event = pd.to_datetime(frame["date"], errors="raise")
    release_clocks = [pd.to_datetime(frame["label_release_day"], errors="raise")]
    if "head_label_release_day" in frame:
        release_clocks.append(pd.to_datetime(frame["head_label_release_day"], errors="raise"))
    cutoff = pd.Timestamp(origin)
    if event.isna().any() or any(clock.isna().any() for clock in release_clocks):
        raise ValueError("invalid event/label-release chronology")
    if any((clock < event).any() for clock in release_clocks):
        raise ValueError("label release precedes its event")
    released_before = np.logical_and.reduce([clock < cutoff for clock in release_clocks])
    return ((event < cutoff) & released_before).to_numpy(dtype=bool)


def validate_join_identity(joined: pd.DataFrame, source: pd.DataFrame) -> None:
    """Require exact row-order and fixture identity; no inferred side reversal."""
    if len(joined) != len(source):
        raise ValueError("joined/source coverage differs")
    for column in IDENTITY_KEYS:
        if column not in joined or column not in source:
            raise ValueError(f"identity input lacks {column}")
        left = joined[column]
        right = source[column]
        if column in ("date", "result_day"):
            left = pd.to_datetime(left, errors="raise").dt.strftime("%Y-%m-%d")
            right = pd.to_datetime(right, errors="raise").dt.strftime("%Y-%m-%d")
        else:
            left, right = left.astype(str), right.astype(str)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise ValueError(f"fixture identity mismatch: {column}")
    if joined.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("duplicate fixture identity")
    for clock in ("head_label_release_day", "label_release_day"):
        if clock in source and clock in joined:
            left = pd.to_datetime(joined[clock], errors="raise").dt.strftime("%Y-%m-%d")
            right = pd.to_datetime(source[clock], errors="raise").dt.strftime("%Y-%m-%d")
            if not np.array_equal(left.to_numpy(), right.to_numpy()):
                raise ValueError(f"authentic release clock mismatch: {clock}")


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"frozen {label} checksum mismatch: {path}")


def _verify_receipt_artifacts(receipt_path: Path, receipt: dict) -> None:
    for relative, expected in receipt.get("artifacts", {}).items():
        artifact = Path(relative)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        _verify_file(artifact, expected, "receipt artifact")
    for relative, expected in receipt.get("report_artifacts_sha256", {}).items():
        artifact = Path(relative)
        if not artifact.is_absolute():
            artifact = receipt_path.parent / artifact
        _verify_file(artifact, expected, "receipt report artifact")


def _verify_archived_sources(receipt: dict, archive_paths: list[Path], research_root: Path) -> dict:
    """Verify code hash closure, accepting a changed checkout only with archived proof."""
    expected_sources = receipt.get("contract", {}).get("source_sha256", {})
    archives: list[tuple[Path, zipfile.ZipFile]] = []
    try:
        for archive_path in archive_paths:
            if archive_path.is_file():
                archive = zipfile.ZipFile(archive_path)
                archives.append((archive_path, archive))
        verified = {}
        for key, expected in expected_sources.items():
            if key.startswith("research:"):
                path = research_root / key.removeprefix("research:")
            else:
                path = PROJECT_ROOT / key.removeprefix("project:")
            if path.is_file() and _sha256(path) == expected:
                verified[key] = {"source": "current", "sha256": expected}
                continue
            relative = key.replace(":", "/", 1)
            proved = False
            for archive_path, archive in archives:
                try:
                    actual = hashlib.sha256(archive.read(relative)).hexdigest()
                except KeyError:
                    continue
                if actual == expected:
                    verified[key] = {"source": "archived_snapshot", "archive": str(archive_path),
                                     "archive_sha256": _sha256(archive_path), "sha256": expected}
                    proved = True
                    break
            if not proved:
                raise ValueError(f"source hash lacks current or frozen snapshot proof: {key}")
        return verified
    finally:
        for _, archive in archives:
            archive.close()


def _source_records(research_root: Path) -> tuple[dict, dict]:
    native_protocol_path = PROJECT_ROOT / INPUT_CONTRACT["native_protocol"]
    native_protocol = _json(native_protocol_path)
    _verify_file(native_protocol_path, INPUT_CONTRACT["native_protocol_sha256"], "native protocol")
    native_receipt_path = PROJECT_ROOT / INPUT_CONTRACT["joined_receipt"]
    native_receipt = _json(native_receipt_path)
    _verify_file(PROJECT_ROOT / INPUT_CONTRACT["joined"], INPUT_CONTRACT["joined_sha256"], "native joined input")
    if not isinstance(native_receipt.get("evaluation_sha256"), str):
        raise ValueError("native evaluation receipt has no evaluation digest")
    _verify_receipt_artifacts(native_receipt_path, native_receipt)
    native_snapshot = PROJECT_ROOT / native_protocol["source_snapshot_path"]
    _verify_file(native_snapshot, native_protocol["source_snapshot_sha256"], "native source snapshot")
    _verify_archived_sources({"contract": {"source_sha256": native_protocol.get("source_sha256", {})}},
                             [native_snapshot], research_root)
    for relative, digest in native_protocol.get("inputs", {}).items():
        path = Path(relative)
        if path.is_absolute():
            input_path = path
        elif relative.startswith("research:"):
            input_path = research_root / relative.removeprefix("research:")
        else:
            input_path = research_root / path
        _verify_file(input_path, digest, "native frozen input")
    completion_paths = [PROJECT_ROOT / f"data/08_reporting/model_reliability/model_reliability_20261001_055321/neural/{year}/completed.json"
                        for year in (2024, 2025)]
    completion_paths.append(PROJECT_ROOT / "data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/completed.json")
    archive_candidates = [native_snapshot]
    prepared_path = PROJECT_ROOT / INPUT_CONTRACT["release_receipt"]
    prepared = _json(prepared_path)
    _verify_file(PROJECT_ROOT / INPUT_CONTRACT["release_source"], INPUT_CONTRACT["release_source_sha256"], "release source")
    execution_path = prepared_path.parent / "execution.json"
    if not execution_path.is_file():
        raise FileNotFoundError(f"release preparation execution receipt missing: {execution_path}")
    execution = _json(execution_path)
    _verify_file(prepared_path, execution["prepared_sha256"], "release preparation receipt")
    if execution.get("status") != "SEALED_BEFORE_C0_HEAD_FITS":
        raise ValueError("release preparation was not sealed before C0 head fits")
    connector_protocol_path = prepared_path.parent / "protocol.json"
    _verify_file(connector_protocol_path, execution["protocol_sha256"], "release source protocol")
    connector_protocol = _json(connector_protocol_path)
    from src.models.a0_correction_experiment import _verify_c0_prepared_sources
    prepared_closure = _verify_c0_prepared_sources(prepared, connector_protocol, research_root)
    for relative, digest in prepared_closure["source_archives"].items():
        archive_path = Path(relative)
        if not archive_path.is_absolute():
            archive_path = PROJECT_ROOT / archive_path
        _verify_file(archive_path, digest, "verified historical source archive")
        archive_candidates.append(archive_path)
    for proof in prepared_closure["archived_sources"].values():
        archive_path = Path(proof["archive"])
        if not archive_path.is_absolute():
            archive_path = PROJECT_ROOT / archive_path
        if archive_path not in archive_candidates:
            archive_candidates.append(archive_path)
    closures = {}
    for path in completion_paths:
        receipt = _json(path)
        if receipt.get("status") != "TRAINING_COMPLETE_NOT_SCORED":
            raise ValueError(f"native source is not a completed unscored artifact: {path}")
        _verify_receipt_artifacts(path, receipt)
        closures[str(path.relative_to(PROJECT_ROOT))] = _verify_archived_sources(receipt, archive_candidates, research_root)
    return {"native_protocol": native_protocol, "native_receipt": native_receipt,
            "prepared": prepared, "preparation_execution_sha256": _sha256(execution_path),
            "prepared_source_closure": prepared_closure, "source_closures": closures}




def _source_frame(research_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    native_path = PROJECT_ROOT / INPUT_CONTRACT["joined"]
    source_path = PROJECT_ROOT / INPUT_CONTRACT["release_source"]
    canonical_path = research_root / INPUT_CONTRACT["canonical_relative"]
    _verify_file(canonical_path, INPUT_CONTRACT["canonical_sha256"], "canonical source")
    native = pd.read_parquet(native_path)
    source = pd.read_parquet(source_path)
    canonical = pd.read_parquet(canonical_path)
    validate_join_identity(native, source)
    if len(native) != INPUT_CONTRACT["rows"]:
        raise ValueError("native C0 input row count differs from frozen contract")
    if canonical.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("canonical source has duplicate IDs")
    canonical = canonical.rename(columns={"p_full": "p_a0"})
    return native, source, canonical


def _aligned_bundle(research_root: Path) -> tuple[pd.DataFrame, dict, dict]:
    native, source, canonical = _source_frame(research_root)
    left = native.set_index("golgg_match_id", drop=False)
    right = source.set_index("golgg_match_id", drop=False)
    if set(left.index) != set(right.index):
        raise ValueError("native C0 and prepared source IDs differ")
    right = right.loc[left.index]
    validate_join_identity(left.reset_index(drop=True), right.reset_index(drop=True))
    frame = left.copy().reset_index(drop=True)
    for key in ("history_stat_coverage", "csd_xpd_coverage", "unknown_players", "co_play_min_mean",
                "w20_min_gate", "feature_source_max_day", "tournament_name", "competition_tier"):
        if key not in frame and key in right:
            frame[key] = right[key].to_numpy()
    canon = canonical.set_index("golgg_match_id", drop=False)
    canon_ids = frame.golgg_match_id.astype(str)
    if not set(canon_ids).issubset(set(canon.index.astype(str))):
        raise ValueError("C0 assessment rows do not resolve in canonical input")
    canon = canon.loc[canon_ids]
    validate_join_identity(frame, canon.reset_index(drop=True))
    frame["p_a0"] = pd.to_numeric(canon["p_a0"], errors="raise").to_numpy(dtype=float)
    if "label_release_day" not in frame:
        raise ValueError("authentic native label_release_day is absent")
    if "head_label_release_day" not in right:
        raise ValueError("authentic prepared head_label_release_day is absent")
    frame["label_release_day"] = pd.to_datetime(
        frame.label_release_day, errors="raise").dt.strftime("%Y-%m-%d").to_numpy()
    frame["head_label_release_day"] = pd.to_datetime(
        right.head_label_release_day, errors="raise").dt.strftime("%Y-%m-%d").to_numpy()
    frame, raw = _reconstruct_saved_experts(frame, research_root)
    provenance = {"joined_sha256": _sha256(PROJECT_ROOT / INPUT_CONTRACT["joined"]),
                  "release_sha256": _sha256(PROJECT_ROOT / INPUT_CONTRACT["release_source"]),
                  "canonical_sha256": _sha256(research_root / INPUT_CONTRACT["canonical_relative"])}
    return frame, provenance, raw
def _reconstruct_saved_experts(frame: pd.DataFrame, research_root: Path) -> tuple[pd.DataFrame, dict]:
    """Reconstruct native grouped experts from verified cached logits and calibrations."""
    import joblib
    backend, _ = a0_uncertainty._activate_archive_backend(research_root)
    from src.models import a0_target_experiment as target
    data = a0_uncertainty._load_training_data(research_root)
    feature_frame = data["frame"].reset_index(drop=True)
    if feature_frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("feature metadata has duplicate match identities")
    model_root_base = PROJECT_ROOT / "data/06_models"
    output_root_base = PROJECT_ROOT / "data/07_model_output"
    computed = {}
    completion_records = {}
    for year in (2024, 2025, 2026):
        if year < 2026:
            relative = Path(f"model_reliability/model_reliability_20261001_055321/neural/{year}")
            completion_path = PROJECT_ROOT / f"data/08_reporting/{relative}/completed.json"
            model_root = model_root_base / relative / "control"
            output_root = output_root_base / relative
        else:
            relative = Path("a0_data_focus/a0_history_reconstruction_20261001_010503")
            completion_path = PROJECT_ROOT / f"data/08_reporting/{relative}/completed.json"
            model_root = model_root_base / relative / "control"
            output_root = output_root_base / relative
        receipt = _json(completion_path)
        if receipt.get("status") != "TRAINING_COMPLETE_NOT_SCORED":
            raise ValueError(f"saved native component is not completed unscored source: {completion_path}")
        _verify_receipt_artifacts(completion_path, receipt)
        calibration_path = model_root / "calibration/model.joblib"
        neural_path = output_root / "control_logits.npz"
        shared_path = output_root / "shared_logits.npz"
        for path in (calibration_path, neural_path, shared_path):
            relative_path = str(path.relative_to(PROJECT_ROOT))
            expected = receipt.get("artifacts", {}).get(relative_path)
            if expected is None:
                raise ValueError(f"native completion receipt does not bind component: {relative_path}")
            _verify_file(path, expected, "native cached component")
        calibration = joblib.load(calibration_path)
        with np.load(neural_path, allow_pickle=False) as neural_archive, np.load(shared_path, allow_pickle=False) as shared_archive:
            indices = neural_archive["indices"].astype(int)
            shared_indices = shared_archive["indices"].astype(int)
            if not np.array_equal(indices, shared_indices):
                raise ValueError(f"native neural/shared cached-logit memberships differ for {year}")
            if np.any(indices < 0) or np.any(indices >= len(feature_frame)) or len(np.unique(indices)) != len(indices):
                raise ValueError(f"native cached-logit indices invalid for {year}")
            neural_logits = neural_archive["logits"].astype(float)
            other_logits = shared_archive["other_logits"].astype(float)
        if neural_logits.shape != (len(indices), 3, 3) or other_logits.shape != (len(indices), 3, 4):
            raise ValueError(f"native cached-logit dimensions differ for {year}")
        indexed_frame = feature_frame.iloc[indices].reset_index(drop=True)
        raw = {key: values[indices] for key, values in data["raw"].items()}
        best_of = indexed_frame.best_of.to_numpy(dtype=int)
        full, neural, _ = target._predict_mixture(
            backend, neural_logits, other_logits, calibration["slopes"], raw["gates"], best_of, calibration)
        grouped = backend["tail_reliability"].calibrated_experts(
            np.concatenate((neural_logits, other_logits), axis=2), calibration["slopes"], raw["gates"], 0)
        grouped[:, 0] = neural[:, 0]
        groups = pd.DataFrame({
            "golgg_match_id": indexed_frame.golgg_match_id.astype(str),
            "p_c0_reconstructed": full[:, 0], "p_neural": neural[:, 0],
            "p_rating": grouped[:, 1], "p_mlp": grouped[:, 2],
        })
        annual = groups.loc[pd.to_datetime(indexed_frame.date).dt.year.eq(year).to_numpy()]
        if annual.golgg_match_id.duplicated().any():
            raise ValueError(f"native annual group IDs duplicate for {year}")
        computed.update(annual.set_index("golgg_match_id").to_dict(orient="index"))
        completion_records[str(year)] = {
            "receipt": str(completion_path.relative_to(PROJECT_ROOT)), "receipt_sha256": _sha256(completion_path),
            "logit_membership_sha256": _digest(indexed_frame.golgg_match_id.astype(str).tolist()),
            "component_rows": len(annual),
        }
    ids = frame.golgg_match_id.astype(str).tolist()
    missing = [match_id for match_id in ids if match_id not in computed]
    if missing:
        raise ValueError(f"saved annual native groups do not cover joined fixtures ({len(missing)} missing)")
    for column in ("p_c0_reconstructed", "p_neural", "p_rating", "p_mlp"):
        frame[column] = [computed[match_id][column] for match_id in ids]
    anchor_column = "native12__full__full"
    neural_anchor_column = "native12__neural__full"
    if anchor_column not in frame or neural_anchor_column not in frame:
        raise ValueError("saved native C0/neural anchor probabilities are absent")
    max_error = float(np.max(np.abs(frame.p_c0_reconstructed.to_numpy(float) -
                                    frame[anchor_column].to_numpy(float))))
    max_neural_error = float(np.max(np.abs(frame.p_neural.to_numpy(float) -
                                           frame[neural_anchor_column].to_numpy(float))))
    if max_error > 2e-6 or max_neural_error > 2e-6:
        raise ValueError(f"saved native reconstruction parity failed: full={max_error}, neural={max_neural_error}")
    frame["p_c0"] = pd.to_numeric(frame[anchor_column], errors="raise").to_numpy(dtype=float)
    frame["p_neural"] = pd.to_numeric(frame[neural_anchor_column], errors="raise").to_numpy(dtype=float)
    frame.attrs["native_reconstruction"] = {"max_abs_anchor_error": max_error,
                                              "max_abs_neural_error": max_neural_error,
                                              "annual_receipts": completion_records}
    return frame, data


def _resolve_research_root(research_root: str | Path | None) -> Path:
    if research_root is not None:
        root = Path(research_root).expanduser()
    elif os.environ.get("ENSEMBLE_RESEARCH_ROOT"):
        root = Path(os.environ["ENSEMBLE_RESEARCH_ROOT"]).expanduser()
    else:
        pointer = PROJECT_ROOT / "data/research_root.txt"
        if not pointer.is_file():
            raise FileNotFoundError(f"research root pointer not found: {pointer}")
        root = Path(pointer.read_text().strip()).expanduser()
    root = root.resolve(strict=True)
    required = (
        root / "a0-phase-walkforward-20260914/code/causal_nn.py",
        root / "nonlinear-history-20260912/code/src/models/nonlinear_history.py",
        root / "auxiliary-player-20260912/data/03_primary/targets.npy",
        root / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl",
    )
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("research root is missing pinned A0 inputs: " + ", ".join(absent))
    return root


def load_context_bundle(research_root=None) -> tuple[pd.DataFrame, dict]:
    root = _resolve_research_root(research_root)
    sources = _source_records(root)
    frame, source_provenance, raw = _aligned_bundle(root)
    bank = raw["frame"]
    ids = frame.golgg_match_id.astype(str)
    bank_ids = bank.golgg_match_id.astype(str)
    indexed = bank.assign(_context_id=bank_ids).set_index("_context_id")
    if indexed.index.duplicated().any() or not set(ids).issubset(set(indexed.index)):
        raise ValueError("feature-bank IDs do not cover frozen joined C0")
    aligned_bank = indexed.loc[ids].reset_index(drop=True)
    validate_join_identity(frame, aligned_bank)
    for field in ("history_stat_coverage", "unknown_players", "tournament_name", "competition_tier"):
        if field not in frame and field in aligned_bank:
            frame[field] = aligned_bank[field].to_numpy()
    mask = np.asarray(raw["history"]["mask"])
    if mask.shape[0] != len(bank) or len(bank) != len(raw["history"]["history"]):
        raise ValueError("history support mask is not aligned to feature metadata")
    mask_lookup = pd.Series(np.arange(len(bank)), index=bank_ids)
    positions = mask_lookup.loc[ids].to_numpy(dtype=int)
    frame["history_token_fraction"] = mask[positions].reshape(len(positions), -1).astype(bool).mean(axis=1)
    return frame, {**source_provenance, "source_receipts": sources,
                   "feature_bank_sha256": raw["bank_receipt_sha256"],
                   "native_reconstruction": frame.attrs.get("native_reconstruction", {}),
                   "feature_bank_alignment": "exact golgg_match_id"}




def _prepare_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from src.analysis.a0_context_reliability import extract_context_features
    return extract_context_features(frame)


def _source_snapshot(run_paths: dict[str, Path], protocol_sources: dict[str, str],
                     research_root: Path, source_receipts: dict) -> str:
    archive_path = run_paths["input"] / "source_snapshot.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in protocol_sources:
            if relative.startswith("research:"):
                path = research_root / relative.removeprefix("research:")
            else:
                path = PROJECT_ROOT / relative.removeprefix("project:")
            archive.write(path, relative.replace(":", "/", 1))
        archive.write(SOURCE_REQUIREMENTS_PATH, "source_activation_requirements.json")
        archive.write(EXECUTION_CONTRACT_PATH, "execution_contract.json")
        archive.writestr("source_activation_summary.json",
                         json.dumps(SOURCE_BLOCKED, indent=2, sort_keys=True) + "\n")
        archive.writestr("input_contract.json", json.dumps(INPUT_CONTRACT, indent=2, sort_keys=True) + "\n")
        archive.writestr("source_receipts.json", json.dumps(source_receipts, indent=2, sort_keys=True) + "\n")
    return _sha256(archive_path)


def freeze_run(run_id: str, research_root=None) -> Path:
    _validate_run_id(run_id)
    paths = _paths(run_id)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    root = _resolve_research_root(research_root)
    frame, provenance = load_context_bundle(root)
    features, feature_metadata = _prepare_context(frame)
    ids = features.golgg_match_id.astype(str)
    if len(features) != 11550 or ids.nunique() != 11550:
        raise ValueError("frozen context cohort must contain exactly 11,550 unique fixtures")
    primary = pd.to_datetime(features.date).ge(pd.Timestamp("2026-01-01"))
    if int(primary.sum()) != 3103 or int((~primary).sum()) != 8447:
        raise ValueError("frozen cohort does not have exact 8,447/3,103 warmup/primary membership")
    _verify_file(SOURCE_REQUIREMENTS_PATH, SOURCE_REQUIREMENTS_SHA256,
                 "authoritative source-activation requirements")
    _verify_file(EXECUTION_CONTRACT_PATH, EXECUTION_CONTRACT_SHA256,
                 "reviewed execution contract")
    source_paths = {
        f"project:{relative}": PROJECT_ROOT / relative for relative in SOURCE_CLOSURE
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_sources = {key: _sha256(path) for key, path in source_paths.items()}
    input_hashes = {**provenance}
    protocol = {
        "schema": "c0_context_reliability_v1", "status": "FROZEN_BEFORE_DATA_SCORING",
        "run_id": run_id, "anchor_identity": INPUT_CONTRACT["anchor"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "EXPOSED_HISTORICAL_DEVELOPMENT",
        "scenario_origin": "2024-07-01", "valid_until": "2026-09-09",
        "scope": ("exposed historical development: persistence detection in 2024H2, replication in 2025, "
                  "and assessment across 2026 Jan-Sep; 8,447 C0 warmup rows, 3,103 primary rows, "
                  "with active calibrated origins Jan/Apr/Jul 2026; not current/future-live valid"),
        "inputs": INPUT_CONTRACT, "input_sha256": input_hashes,
        "source_sha256": protocol_sources, "runtime": _runtime_versions(),
        "source_activation_requirements_sha256": SOURCE_REQUIREMENTS_SHA256,
        "execution_contract": {
            "artifact": EXECUTION_CONTRACT_RELATIVE,
            "sha256": EXECUTION_CONTRACT_SHA256,
        },
        "joined_input_sha256": None, "source_snapshot_sha256": None,
        "disagreement_threshold": feature_metadata["high_sports_disagreement_threshold"],
        "disagreement_tie_atol": feature_metadata["disagreement_tie_atol"],
        "cohort": {"rows": len(features), "warmup_rows": int((~primary).sum()),
                   "primary_rows": int(primary.sum()), "ids_sha256": _digest(ids.tolist())},
        "partition_membership": {origin: {"rows": int(mask.sum()), "ids_sha256": _digest(ids[mask].tolist())}
                                  for origin, mask in ((origin, eligible_history_rows(features, origin))
                                                       for origin in (*DIAGNOSTIC_ORIGINS, *CONDITIONAL_ORIGINS))},
        "diagnostic_global_origins": list(DIAGNOSTIC_ORIGINS),
        "conditional_origins": list(CONDITIONAL_ORIGINS),
        "delivered_active_origins": ["2026-01-01", "2026-04-01", "2026-07-01"],
        "penalties": list(PENALTIES), "global_bounds": [0.05, 20.0], "minimum_fit_rows": 500,
        "persistence": {"detection_interval": ["2024-07-01", "2025-01-01"],
                         "replication_interval": ["2025-01-01", "2026-01-01"],
                         "claims": 6, "confidence_level": 0.9916666666666667,
                         "draws": 5000, "seed": 20261004},
        "guard_populations": {
            "bo3": "best_of == 3",
            "bo5": "best_of == 5",
            "major": "frozen competition_tier normalized with astype('string').str.casefold() == 'major'; not reclassified from tournament_name",
            "source_unseen_player": "unknown_players > 0; UNKNOWN is not source-unseen",
            "comparators": ["p_c0", "p_a0"],
        },
        "conditional_fit_budget": {"skip": {"diagnostic_global_fits": 6, "conditional_context_fits": 0,
                                              "global_fits": 0, "total_fits": 6},
                                    "execute": {"diagnostic_global_fits": 6, "conditional_context_fits": 42,
                                                "additional_global_fits": 3, "distinct_global_fits": 9,
                                                "total_fits": 51}},
        "new_base_fits": 0, "source_activation": SOURCE_BLOCKED,
        "token_support": "pinned mask.npy, aligned through feature-bank metadata ID; descriptive fraction only",
        "roster_turnover": "UNASSESSED: no cutoff-causal announcement/turnover evidence",
        "prediction_contract": {"kind": "full_context_series_winner_only", "no_intercept": True,
                                "no_second_best_of_projection": True, "no_new_clipping": True,
                                "valid_until": "2026-09-09", "native_replay_tolerance": 2e-6},
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    staged_input = Path(tempfile.mkdtemp(prefix=f".{run_id}.freeze-", dir=paths["input"].parent))
    staged_report = Path(tempfile.mkdtemp(prefix=f".{run_id}.freeze-", dir=paths["report"].parent))
    staged_snapshot = staged_input / "source_snapshot.zip"
    protocol["source_snapshot_sha256"] = _source_snapshot(
        {"input": staged_input}, protocol_sources, root, provenance.get("source_receipts", {}))
    protocol["joined_input_sha256"] = None
    features.to_parquet(staged_input / "joined_input.parquet", index=False)
    protocol["joined_input_sha256"] = _sha256(staged_input / "joined_input.parquet")
    if _sha256(staged_snapshot) != protocol["source_snapshot_sha256"]:
        raise ValueError("preflight source snapshot hash changed before staging")
    (staged_input / "source_activation_requirements.json").write_bytes(SOURCE_REQUIREMENTS_BYTES)
    (staged_report / "source_activation_requirements.json").write_bytes(SOURCE_REQUIREMENTS_BYTES)
    (staged_input / "execution_contract.json").write_bytes(EXECUTION_CONTRACT_BYTES)
    (staged_report / "execution_contract.json").write_bytes(EXECUTION_CONTRACT_BYTES)
    protocol_bytes = (json.dumps(_safe_json(protocol), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    input_provenance = {
        "protocol_sha256": protocol_sha, "input_sha256": input_hashes,
        "joined_input_sha256": protocol["joined_input_sha256"],
        "rows": len(features), "membership_sha256": protocol["cohort"]["ids_sha256"],
    }
    _save_json(staged_input / "input_provenance.json", input_provenance)
    _save_json(staged_report / "input_provenance.json", input_provenance)
    (staged_input / "protocol.json").write_bytes(protocol_bytes)
    (staged_report / "protocol.json").write_bytes(protocol_bytes)
    paths["models"].mkdir(parents=True, exist_ok=False)
    paths["output"].mkdir(parents=True, exist_ok=False)
    os.replace(staged_report, paths["report"])
    os.replace(staged_input, paths["input"])
    return paths["input"] / "protocol.json"


def _validate_protocol(protocol_path: Path) -> tuple[dict, dict[str, Path]]:
    protocol_path = Path(protocol_path).resolve()
    protocol = _json(protocol_path)
    if protocol.get("schema") != "c0_context_reliability_v1" or protocol.get("status") != "FROZEN_BEFORE_DATA_SCORING":
        raise ValueError("unsupported or unfrozen context reliability protocol")
    if protocol.get("anchor_identity") != INPUT_CONTRACT["anchor"] or protocol.get("inputs") != INPUT_CONTRACT:
        raise ValueError("frozen context protocol differs from reviewed input contract")
    if protocol["run_id"] != protocol_path.parent.name:
        raise ValueError("protocol path does not match run_id")
    paths = _paths(protocol["run_id"])
    if protocol_path != paths["input"] / "protocol.json":
        raise ValueError("protocol is not located in its frozen run input root")
    if (protocol.get("scenario_origin") != "2024-07-01" or protocol.get("valid_until") != "2026-09-09"
            or protocol.get("classification") != "EXPOSED_HISTORICAL_DEVELOPMENT"
            or protocol.get("source_activation") != SOURCE_BLOCKED):
        raise ValueError("frozen historical validity or blocked-source contract differs")
    if protocol.get("runtime") != _runtime_versions():
        raise ValueError("frozen numerical runtime differs")
    if protocol.get("source_activation_requirements_sha256") != SOURCE_REQUIREMENTS_SHA256:
        raise ValueError("authoritative source-activation requirements digest differs")
    if protocol.get("execution_contract") != {
            "artifact": EXECUTION_CONTRACT_RELATIVE, "sha256": EXECUTION_CONTRACT_SHA256}:
        raise ValueError("reviewed execution contract reference differs")
    snapshot = paths["input"] / "source_snapshot.zip"
    _verify_file(snapshot, protocol["source_snapshot_sha256"], "frozen source snapshot")
    with zipfile.ZipFile(snapshot) as archive:
        if json.loads(archive.read("input_contract.json")) != INPUT_CONTRACT:
            raise ValueError("frozen snapshot input contract differs")
        requirements = archive.read("source_activation_requirements.json")
        if (hashlib.sha256(requirements).hexdigest() != SOURCE_REQUIREMENTS_SHA256
                or requirements != SOURCE_REQUIREMENTS_BYTES):
            raise ValueError("frozen authoritative source-activation requirements differ")
        execution_contract = archive.read("execution_contract.json")
        if (hashlib.sha256(execution_contract).hexdigest() != EXECUTION_CONTRACT_SHA256
                or execution_contract != EXECUTION_CONTRACT_BYTES):
            raise ValueError("frozen reviewed execution contract differs")
        summary = json.loads(archive.read("source_activation_summary.json"))
        if summary != SOURCE_BLOCKED:
            raise ValueError("frozen source activation summary differs")
        for key, expected in protocol["source_sha256"].items():
            member = key.replace(":", "/", 1)
            if hashlib.sha256(archive.read(member)).hexdigest() != expected:
                raise ValueError(f"frozen snapshot source differs: {key}")
    _verify_live_source_closure(protocol["source_sha256"], PROJECT_ROOT)
    return protocol, paths


def _global_model(p: np.ndarray, y: np.ndarray) -> ContextSlopeCalibration:
    contexts = np.full((len(p), 3), np.nan, dtype=float)
    model = ContextSlopeCalibration(kind="global", penalty=1.0).fit(p, y, contexts)
    slopes = [float(model.to_dict()["parameters"]["a0"])]
    if not all(0.05 <= value <= 20.0 for value in slopes):
        raise ValueError("global fit violates frozen positive slope bounds")
    return model


def _model(kind: str, penalty: float, p: np.ndarray, y: np.ndarray,
           contexts: np.ndarray) -> ContextSlopeCalibration:
    model = ContextSlopeCalibration(kind=kind, penalty=penalty).fit(p, y, contexts)
    serialized = model.to_dict()
    parameters = serialized["parameters"]
    if not all(math.isfinite(float(value)) for value in parameters.values()):
        raise ValueError("calibration fit produced a nonfinite parameter")
    if kind == "global":
        slopes = [parameters["a0"]]
    else:
        slopes = [parameters["a0"] + sum((parameters[f"a_{name}"] - parameters["a0"]) * flag
                                          for name, flag in zip(CONTEXT_NAMES, bits))
                  for bits in product((0, 1), repeat=3)]
    if any(not 0.05 <= float(slope) <= 20.0 for slope in slopes):
        raise ValueError("calibration fit violates all-context strict slope bounds")
    diagnostics = serialized.get("fit_diagnostics", {})
    if diagnostics.get("optimizer_success") is not True:
        raise ValueError("calibration optimizer failed; no clipping or fallback is permitted")
    return model


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("fit/evaluation probabilities must be finite and strictly interior")
    return float(-np.log(np.where(y == 1, p, 1.0 - p)).sum())


def _artifact_ref(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _contexts(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"x_{name}" for name in CONTEXT_NAMES]].to_numpy(dtype=float)


def _fit_globals(frame: pd.DataFrame, origins: tuple[str, ...], output_dir: Path) -> tuple[pd.DataFrame, dict]:
    records = []
    artifacts = {}
    for origin in origins:
        origin_mask = eligible_history_rows(frame, origin)
        training = frame.loc[origin_mask]
        if len(training) < 500:
            raise ValueError(f"global origin {origin} has fewer than 500 prior eligible rows")
        model = _global_model(training.p_c0.to_numpy(float), training.y.to_numpy(float))
        model_dict = model.to_dict()
        model_path = output_dir / "models" / "global" / f"{origin}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _save_json(model_path, model_dict)
        pred_mask = (pd.to_datetime(frame.date) >= pd.Timestamp(origin)) & (
            pd.to_datetime(frame.date) < pd.Timestamp(origin) + pd.offsets.QuarterBegin(startingMonth=1))
        # The final origin extends through the frozen 2026 assessment end.
        if not pred_mask.any():
            pred_mask = pd.to_datetime(frame.date) >= pd.Timestamp(origin)
        predictions = model.predict_proba(frame.loc[pred_mask].p_c0.to_numpy(float),
                                          np.full((int(pred_mask.sum()), 3), np.nan))
        part = frame.loc[pred_mask, ["golgg_match_id", "date", "label_release_day"]].copy()
        part["origin"] = origin
        part["q_global"] = np.asarray(predictions, dtype=float)
        records.append(part)
        artifacts[origin] = {"fit_rows": len(training), "fit_ids_sha256": _digest(training.golgg_match_id.astype(str).tolist()),
                             "artifact": _artifact_ref(model_path), "sha256": _sha256(model_path),
                             "parameters": model_dict["parameters"]}
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["golgg_match_id", "date", "label_release_day", "origin", "q_global"])
    return result, artifacts


def _persistence(frame: pd.DataFrame, q_global: pd.DataFrame, report_dir: Path,
                 config: dict | None = None) -> dict:
    from src.analysis.a0_context_reliability import evaluate_persistence
    result = evaluate_persistence(frame, q_global, config=config)
    _save_json(report_dir / "persistence.json", result)
    passed = bool(result.get("gate", {}).get("pass", False))
    return {"result": result, "passed": passed}


def _fit_conditional(frame: pd.DataFrame, origins: tuple[str, ...], output_dir: Path,
                     q_global: pd.DataFrame) -> dict:
    del q_global
    result = {"fits": [], "selected": [], "predictions": [], "oof_predictions": []}
    oof_by_arm: dict[tuple[str, float], list[pd.DataFrame]] = {}
    dates = pd.to_datetime(frame.date)
    for origin in origins:
        train = frame.loc[eligible_history_rows(frame, origin)]
        if len(train) < 500:
            raise ValueError(f"conditional origin {origin} has fewer than 500 prior eligible rows")
        current_predictions = frame.loc[(dates >= pd.Timestamp(origin)) &
                                        (dates < pd.Timestamp(origin) + pd.offsets.QuarterBegin(startingMonth=1))]
        if origin == origins[-1]:
            current_predictions = frame.loc[dates >= pd.Timestamp(origin)]
        selected_predictions = current_predictions[list(IDENTITY_KEYS)].copy()
        selected_predictions["origin"] = origin
        for kind in ("pooled", "independent"):
            prior_by_penalty = {}
            for penalty in PENALTIES:
                prior_parts = oof_by_arm.get((kind, penalty), [])
                if prior_parts:
                    prior = pd.concat(prior_parts, ignore_index=True)
                    keep = eligible_history_rows(prior, origin)
                    prior_by_penalty[penalty] = prior.loc[keep]
                else:
                    prior_by_penalty[penalty] = pd.DataFrame(columns=[
                        *IDENTITY_KEYS, "label_release_day", "head_label_release_day",
                        "origin", "y", f"p_{kind}",
                    ])
            common_keys = None
            for prior in prior_by_penalty.values():
                ids = set(prior["golgg_match_id"].astype(str))
                common_keys = ids if common_keys is None else common_keys & ids
            common_keys = common_keys or set()
            candidate_records = []
            candidate_models = {}
            for penalty in PENALTIES:
                model = _model(kind, penalty, train.p_c0.to_numpy(float), train.y.to_numpy(float), _contexts(train))
                prior = prior_by_penalty[penalty]
                prior = prior.loc[prior["golgg_match_id"].astype(str).isin(common_keys)]
                score = None
                if len(prior):
                    score = _ll(prior[f"p_{kind}"].to_numpy(float), prior.y.to_numpy(float)) / len(prior)
                candidate_records.append({
                    "penalty": penalty, "validation_rows": len(prior),
                    "validation_ids_sha256": _digest(sorted(prior["golgg_match_id"].astype(str).tolist())) if len(prior) else None,
                    "unpenalized_match_weighted_ll": score,
                })
                candidate_models[penalty] = model
                model_path = output_dir / "models" / kind / origin / f"lambda_{int(penalty)}.json"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _save_json(model_path, model.to_dict())
                artifact_receipt_path = model_path.with_suffix(".receipt.json")
                _save_json(artifact_receipt_path, {
                    "model_sha256": _sha256(model_path), "scenario_origin": origin,
                    "valid_until": "2026-09-09",
                    "scope": "historical-only; not current/future-live valid",
                })
                fit_record = {
                    "origin": origin, "kind": kind, "penalty": penalty, "train_rows": len(train),
                    "train_ids_sha256": _digest(train.golgg_match_id.astype(str).tolist()),
                    "train_last_label_release_day": str(train.label_release_day.max()),
                    "artifact": _artifact_ref(model_path), "sha256": _sha256(model_path),
                    "artifact_receipt": _artifact_ref(artifact_receipt_path),
                    "artifact_receipt_sha256": _sha256(artifact_receipt_path),
                    "cv_rows": len(prior), "cv_ids_sha256": candidate_records[-1]["validation_ids_sha256"],
                    "cv_unpenalized_match_weighted_ll": score,
                }
                result["fits"].append(fit_record)
                forecast = current_predictions[list(IDENTITY_KEYS)].copy()
                forecast["label_release_day"] = current_predictions.label_release_day.to_numpy()
                forecast["head_label_release_day"] = current_predictions.head_label_release_day.to_numpy()
                forecast["origin"] = origin
                forecast["y"] = current_predictions.y.to_numpy()
                forecast[f"p_{kind}"] = np.asarray(
                    model.predict_proba(current_predictions.p_c0.to_numpy(float), _contexts(current_predictions)),
                    dtype=float)
                oof_by_arm.setdefault((kind, penalty), []).append(forecast)
                result["oof_predictions"].append(forecast.assign(penalty=penalty))
            scored = [item for item in candidate_records if item["unpenalized_match_weighted_ll"] is not None]
            if scored:
                best = min(item["unpenalized_match_weighted_ll"] for item in scored)
                tied = [item for item in scored if item["unpenalized_match_weighted_ll"] <= best + 1e-12]
                selected_penalty = max(item["penalty"] for item in tied)
                selection_reason = "past eligible OOF assessment LL; strongest within 1e-12"
            else:
                selected_penalty = 100.0
                selection_reason = "no eligible closed-past OOF support; strongest penalty"
            selected_model = candidate_models[selected_penalty]
            selected_artifact = output_dir / "models" / kind / origin / f"lambda_{int(selected_penalty)}.json"
            result["selected"].append({
                "origin": origin, "kind": kind, "penalty": selected_penalty,
                "artifact": _artifact_ref(selected_artifact),
                "artifact_sha256": _sha256(selected_artifact), "candidates": candidate_records,
                "reason": selection_reason, "refit_after_selection": False,
            })
            selected_predictions[f"p_{kind}"] = np.asarray(
                selected_model.predict_proba(current_predictions.p_c0.to_numpy(float),
                                             _contexts(current_predictions)), dtype=float)
        result["predictions"].append(selected_predictions)
    return result




def _evaluate_final(frame: pd.DataFrame, candidate_predictions: pd.DataFrame,
                    q_global: pd.DataFrame, report_dir: Path, output_dir: Path) -> dict:
    from src.analysis.a0_context_reliability import evaluate_context_predictions
    primary = pd.to_datetime(frame.date).dt.year.eq(2026)
    result_frame = frame.copy()
    result_frame["p_context"] = result_frame.p_c0.to_numpy(float)
    result_frame["p_independent"] = result_frame.p_c0.to_numpy(float)
    result_frame["p_global"] = result_frame.p_c0.to_numpy(float)
    active_origins = ("2026-01-01", "2026-04-01", "2026-07-01")
    active = candidate_predictions.loc[candidate_predictions.origin.isin(active_origins)]
    if active.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("active conditional forecasts contain duplicate identities")
    active = active.set_index("golgg_match_id")
    primary_ids = result_frame.loc[primary, "golgg_match_id"].astype(str)
    if set(primary_ids) != set(active.index.astype(str)):
        raise ValueError("active context heads do not cover exactly the 3,103 primary IDs")
    aligned_active = active.loc[primary_ids]
    result_frame.loc[primary, "p_context"] = aligned_active["p_pooled"].to_numpy(float)
    result_frame.loc[primary, "p_independent"] = aligned_active["p_independent"].to_numpy(float)
    active_global = q_global.loc[q_global.origin.isin(active_origins)]
    if active_global.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("active global forecasts contain duplicate identities")
    active_global = active_global.assign(_key=active_global.golgg_match_id.astype(str)).set_index("_key")
    if set(primary_ids) != set(active_global.index.astype(str)):
        raise ValueError("active global controls do not cover exactly the 3,103 primary IDs")
    aligned_global = active_global.loc[primary_ids]
    primary_dates = pd.to_datetime(result_frame.loc[primary, "date"], errors="raise")
    expected_origins = np.where(
        primary_dates < pd.Timestamp("2026-04-01"), "2026-01-01",
        np.where(primary_dates < pd.Timestamp("2026-07-01"), "2026-04-01", "2026-07-01"))
    if not np.array_equal(aligned_active.origin.astype(str).to_numpy(), expected_origins):
        raise ValueError("active context heads have wrong-origin/quarter assignments")
    if not np.array_equal(aligned_global.origin.astype(str).to_numpy(), expected_origins):
        raise ValueError("active global controls have wrong-origin/quarter assignments")
    for column in ("date", "result_day", "team1_id", "team2_id", "best_of", "y"):
        expected = result_frame.loc[primary, column].reset_index(drop=True)
        actual = aligned_active[column].reset_index(drop=True)
        if not np.array_equal(expected.to_numpy(), actual.to_numpy()):
            raise ValueError(f"active context candidate fixture identity mismatch: {column}")
    for column in ("label_release_day", "head_label_release_day"):
        expected = result_frame.loc[primary, column].reset_index(drop=True)
        actual = aligned_active[column].reset_index(drop=True)
        left = pd.to_datetime(expected, errors="raise").dt.strftime("%Y-%m-%d")
        right = pd.to_datetime(actual, errors="raise").dt.strftime("%Y-%m-%d")
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise ValueError(f"active context candidate release clock mismatch: {column}")
    for column in ("date", "label_release_day"):
        expected = result_frame.loc[primary, column].reset_index(drop=True)
        actual = aligned_global[column].reset_index(drop=True)
        left = pd.to_datetime(expected, errors="raise").dt.strftime("%Y-%m-%d")
        right = pd.to_datetime(actual, errors="raise").dt.strftime("%Y-%m-%d")
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise ValueError(f"active global control identity/clock mismatch: {column}")
    result_frame.loc[primary, "p_context"] = aligned_active["p_pooled"].to_numpy(float)
    result_frame.loc[primary, "p_independent"] = aligned_active["p_independent"].to_numpy(float)
    result_frame.loc[primary, "p_global"] = aligned_global["q_global"].to_numpy(float)
    result_frame["head_status"] = np.where(primary, "CONDITIONAL_CANDIDATE", "C0_WARMUP")
    metrics = evaluate_context_predictions(result_frame)
    result_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    _save_json(report_dir / "metrics.json", metrics)
    return metrics
def run_protocol(protocol_path: str | Path) -> dict:
    protocol, paths = _validate_protocol(Path(protocol_path))
    protocol_sha = _sha256(protocol_path)
    joined_path = paths["input"] / "joined_input.parquet"
    provenance = _json(paths["input"] / "input_provenance.json")
    if (provenance.get("protocol_sha256") != protocol_sha
            or provenance.get("input_sha256") != protocol["input_sha256"]
            or provenance.get("joined_input_sha256") != protocol["joined_input_sha256"]):
        raise ValueError("frozen input provenance mismatch")
    _verify_file(joined_path, protocol["joined_input_sha256"], "frozen joined input")
    _verify_file(paths["input"] / "source_activation_requirements.json",
                 SOURCE_REQUIREMENTS_SHA256, "input source-activation requirements")
    _verify_file(paths["report"] / "source_activation_requirements.json",
                 SOURCE_REQUIREMENTS_SHA256, "report source-activation requirements")
    _verify_file(paths["input"] / "execution_contract.json",
                 EXECUTION_CONTRACT_SHA256, "input reviewed execution contract")
    _verify_file(paths["report"] / "execution_contract.json",
                 EXECUTION_CONTRACT_SHA256, "report reviewed execution contract")
    _verify_file(paths["report"] / "protocol.json", protocol_sha, "report frozen protocol")
    if _json(paths["report"] / "input_provenance.json") != provenance:
        raise ValueError("report frozen input provenance differs")
    frame = pd.read_parquet(joined_path)
    if len(frame) != 11550 or frame.golgg_match_id.astype(str).nunique() != 11550:
        raise ValueError("frozen joined input row count/IDs differ")
    receipt_path = paths["report"] / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"run already has a receipt; refusing overwrite/refit: {receipt_path}")
    expected_frozen_report_files = {
        "protocol.json", "input_provenance.json", "source_activation_requirements.json",
        "execution_contract.json",
    }
    for output_root in (paths["output"], paths["models"]):
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"run artifacts already exist; refusing overwrite/refit: {output_root}")
    if (not paths["report"].is_dir()
            or {entry.name for entry in paths["report"].iterdir()} != expected_frozen_report_files):
        raise FileExistsError(f"run report contains unexpected artifacts: {paths['report']}")
    receipt = {"schema": "c0_context_reliability_run_v1", "run_id": protocol["run_id"],
               "protocol_sha256": protocol_sha, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "RUNNING", "scope": protocol["scope"], "source_activation": SOURCE_BLOCKED,
               "new_base_fits": 0, "global_fit_artifacts": [], "conditional_fit_artifacts": [],
               "failed_optimizer_attempts": []}
    _save_json(receipt_path, receipt)
    try:
        q_diagnostic, diagnostic_artifacts = _fit_globals(frame, DIAGNOSTIC_ORIGINS, paths["models"])
        q_diagnostic.to_parquet(paths["output"] / "diagnostic_global_oof.parquet", index=False)
        persistence = _persistence(frame, q_diagnostic, paths["report"])
        receipt["persistence"] = persistence["result"]
        receipt["persistence_passed"] = persistence["passed"]
        receipt["diagnostic_global_fits"] = diagnostic_artifacts
        receipt["global_fit_artifacts"] = list(diagnostic_artifacts.values())
        receipt["conditional_fit_count"] = 0
        receipt["conditional_assessment"] = "NOT_RUN"
        if not persistence["passed"]:
            output = frame.copy()
            primary = pd.to_datetime(output.date).dt.year.eq(2026)
            output["p_context"] = np.where(primary, np.nan, output.p_c0)
            output["p_global"] = np.where(primary, np.nan, output.p_c0)
            output["p_independent"] = np.where(primary, np.nan, output.p_c0)
            output["head_status"] = np.where(primary, "SKIPPED_BY_PERSISTENCE_GATE", "C0_WARMUP")
            output.to_parquet(paths["output"] / "predictions.parquet", index=False)
            _save_json(paths["report"] / "metrics.json", {
                "status": "NOT_RUN", "reason": "frozen persistence gate failed",
                "primary_rows": int(primary.sum()), "primary_predictions": 0,
                "source_activation": SOURCE_BLOCKED,
            })
            _save_json(paths["report"] / "fit_membership.json", {
                "conditional_assessment": "NOT_RUN", "reason": "frozen persistence gate failed",
                "diagnostic_global_fit_artifacts": diagnostic_artifacts,
                "conditional_fit_membership": [], "additional_global_fit_membership": [],
            })
            receipt.update({"status": "CONDITIONAL_FITS_SKIPPED_PERSISTENCE_GATE",
                            "conditional_assessment": "NOT_RUN",
                            "primary_output_rows": int(primary.sum()), "primary_head_absence": True,
                            "fit_counts": {"diagnostic_global_fits": len(diagnostic_artifacts),
                                           "conditional_context_fits": 0, "additional_global_fits": 0,
                                           "distinct_global_fits": len(diagnostic_artifacts),
                                           "total_fits": len(diagnostic_artifacts)}})
            _save_json(paths["report"] / "decision.json", {
                "status": receipt["status"], "persistence_passed": False,
                "conditional_assessment": "NOT_RUN", "fit_counts": receipt["fit_counts"],
                "source_activation": SOURCE_BLOCKED,
            })
            _replace_json(receipt_path, receipt)
            return receipt
        extra_origins = ("2026-01-01", "2026-04-01", "2026-07-01")
        q_extra, extra_artifacts = _fit_globals(frame, extra_origins, paths["models"])
        q_all = pd.concat([q_diagnostic, q_extra], ignore_index=True)
        q_all.to_parquet(paths["output"] / "global_oof_predictions.parquet", index=False)
        conditional = _fit_conditional(frame, CONDITIONAL_ORIGINS, paths["models"], q_all)
        all_selected = pd.concat(conditional["predictions"], ignore_index=True)
        all_selected.to_parquet(paths["output"] / "conditional_oof_predictions.parquet", index=False)
        all_cv = pd.concat(conditional["oof_predictions"], ignore_index=True)
        all_cv.to_parquet(paths["output"] / "conditional_penalty_oof_predictions.parquet", index=False)
        _save_json(paths["report"] / "fit_membership.json", {
            "diagnostic_global": diagnostic_artifacts, "additional_global": extra_artifacts,
            "conditional_fits": conditional["fits"], "selected": conditional["selected"],
            "global_predictions_sha256": _digest(q_all.to_dict(orient="records")),
        })
        metrics = _evaluate_final(frame, all_selected, q_all, paths["report"], paths["output"])
        fit_counts = {"diagnostic_global_fits": len(diagnostic_artifacts),
                      "conditional_context_fits": len(conditional["fits"]),
                      "additional_global_fits": len(extra_artifacts),
                      "distinct_global_fits": len(diagnostic_artifacts) + len(extra_artifacts),
                      "total_fits": len(diagnostic_artifacts) + len(conditional["fits"]) + len(extra_artifacts)}
        receipt.update({"status": "COMPLETE_HISTORICAL_DEVELOPMENT_ONLY",
                        "additional_global_fits": extra_artifacts,
                        "conditional_assessment": "EXECUTED",
                        "conditional_fit_count": len(conditional["fits"]),
                        "global_fit_artifacts": [*diagnostic_artifacts.values(), *extra_artifacts.values()],
                        "conditional_fit_artifacts": conditional["fits"], "metrics": metrics,
                        "fit_counts": fit_counts})
        _save_json(paths["report"] / "decision.json", {
            "status": receipt["status"], "persistence_passed": True, "fit_counts": fit_counts,
            "promotion": "NOT_AUTOMATIC; exposed historical development only",
            "source_activation": SOURCE_BLOCKED,
        })
        _replace_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        failed = "optimizer" in str(exc).lower() and "fail" in str(exc).lower()
        if failed:
            receipt["failed_optimizer_attempts"].append({
                "error_type": type(exc).__name__, "error": str(exc),
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        partial_models = sorted(paths["models"].rglob("*.json"))
        receipt.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                        "partial_global_fit_artifacts": receipt.get("global_fit_artifacts", []),
                        "partial_conditional_fit_artifacts": receipt.get("conditional_fit_artifacts", []),
                        "partial_model_artifacts": [
                            {"path": _artifact_ref(path), "sha256": _sha256(path)}
                            for path in partial_models
                        ],
                        "partial_model_artifact_count": sum(
                            not path.name.endswith(".receipt.json") for path in partial_models),
                        "failed_optimizer_attempt": failed})
        _replace_json(receipt_path, receipt)
        raise


def _smoke_frame(seed: int, persistent: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    def add(day: pd.Timestamp, p: float, risk: bool, history: bool) -> None:
        if history:
            y = int(rng.random() < (0.9 if risk else 0.1)) if persistent else int(rng.random() < p)
        else:
            y = int(rng.random() < p)
        value = float(risk)
        match_id = f"toy-{seed}-{len(records):06d}"
        records.append({"golgg_match_id": match_id, "date": day,
                        "result_day": day + pd.Timedelta(days=1),
                        "label_release_day": day + pd.Timedelta(days=1), "y": y, "p_c0": p,
                        "head_label_release_day": day + pd.Timedelta(days=1),
                        "team1_id": f"{match_id}-a", "team2_id": f"{match_id}-b",
                        "x_low_statistics": value, "x_novel_roster_support": value,
                        "x_high_sports_disagreement": value, "best_of": 3,
                        "competition_tier": "other", "tournament_name": f"toy-{day.year}",
                        "history_stat_coverage": 0.5, "unknown_players": 0})

    bins = (0.55, 0.65, 0.75, 0.85, 0.95)
    for day in pd.date_range("2024-01-01", "2024-06-29", freq="D"):
        for rep in range(4):
            add(day, bins[(len(records) + rep) % len(bins)], bool(rep % 2), False)
    for start, stop in (("2024-07-01", "2025-01-01"), ("2025-01-01", "2026-01-01")):
        for month in pd.date_range(start, stop, freq="MS", inclusive="left"):
            for p in bins:
                for risk in (False, True):
                    for rep in range(30):
                        add(month + pd.Timedelta(days=rep % 20), p, risk, True)
    for month in pd.date_range("2026-01-01", "2026-09-01", freq="MS"):
        for p in bins:
            for rep in range(6):
                add(month + pd.Timedelta(days=rep), p, bool(rep % 2), False)
    return pd.DataFrame.from_records(records)


def run_smoke(output_dir: str | Path) -> dict:
    """Exercise both persistence-gated branches on deterministic synthetic data."""
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    branch_receipts = {}
    for scenario, persistent in (("conditional_pass", True), ("conditional_skip", False)):
        branch = root / scenario
        branch.mkdir()
        frame = _smoke_frame(20261004 + int(persistent), persistent)
        frame.to_parquet(branch / "input.parquet", index=False)
        q_global, global_artifacts = _fit_globals(frame, DIAGNOSTIC_ORIGINS, branch)
        q_global.to_parquet(branch / "diagnostic_global_oof.parquet", index=False)
        persistence = _persistence(
            frame, q_global, branch,
            {"draws": 200, "seed": 20261004, "minimum_group_rows_per_year": 100,
             "minimum_group_months_per_year": 6, "minimum_risk_and_reference_rows_per_bin": 20,
             "confidence_level": 0.99},
        )
        conditional = {"fits": [], "selected": [], "predictions": []}
        if persistence["passed"]:
            conditional = _fit_conditional(frame, CONDITIONAL_ORIGINS, branch, q_global)
            selected_record = next(item for item in reversed(conditional["selected"])
                                   if item["origin"] == CONDITIONAL_ORIGINS[-1] and item["kind"] == "pooled")
            model_path = Path(selected_record["artifact"])
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            prediction_input = branch / "prediction_input.parquet"
            active_rows = frame.loc[pd.to_datetime(frame.date) >= pd.Timestamp(CONDITIONAL_ORIGINS[-1])]
            active_rows.to_parquet(prediction_input, index=False)
            prediction_info = predict_artifact(
                model_path, prediction_input, branch / "predictions.parquet")
            candidate = pd.read_parquet(branch / "predictions.parquet")
            candidate["head_status"] = "CONDITIONAL_CANDIDATE"
            candidate.to_parquet(branch / "predictions.parquet", index=False)
        else:
            candidate = frame.copy()
            candidate["p_context"] = np.nan
            candidate["head_status"] = "SKIPPED_BY_PERSISTENCE_GATE"
            candidate.to_parquet(branch / "predictions.parquet", index=False)
            prediction_info = {"status": "SKIPPED_BY_PERSISTENCE_GATE", "rows": len(candidate),
                               "output_sha256": _sha256(branch / "predictions.parquet")}
        result = {
            "status": "SYNTHETIC_SOFTWARE_PROOF_NOT_HISTORICAL_COVERAGE",
            "branch": "EXECUTED" if persistence["passed"] else "SKIPPED",
            "persistence_gate": persistence["result"]["gate"],
            "diagnostic_global_fit_count": len(global_artifacts),
            "diagnostic_global_fit_artifacts": list(global_artifacts.values()),
            "conditional_fit_count": len(conditional["fits"]), "conditional_fit_artifacts": conditional["fits"],
            "conditional_selected": conditional["selected"], "rows": len(frame),
            "prediction_rows": len(candidate), "prediction_sha256": _sha256(branch / "predictions.parquet"),
            "actual_historical_coverage": False, "source_activation": SOURCE_BLOCKED,
        }
        _save_json(branch / "receipt.json", result)
        branch_receipts[scenario] = result
    final = {"status": "SYNTHETIC_SOFTWARE_PROOF_NOT_HISTORICAL_COVERAGE",
             "branches": branch_receipts, "actual_historical_coverage": False,
             "source_activation": SOURCE_BLOCKED}
    _save_json(root / "receipt.json", final)
    return final


def predict_artifact(artifact_path: str | Path, input_path: str | Path, output_path: str | Path) -> dict:
    artifact_path, input_path, output_path = Path(artifact_path), Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    artifact_receipt_path = artifact_path.with_suffix(".receipt.json")
    artifact_receipt = _json(artifact_receipt_path)
    _verify_file(artifact_path, artifact_receipt["model_sha256"], "calibration artifact")
    scenario_origin = pd.Timestamp(artifact_receipt["scenario_origin"])
    valid_until = pd.Timestamp(artifact_receipt["valid_until"])
    if artifact_receipt.get("scope") != "historical-only; not current/future-live valid":
        raise ValueError("calibration artifact lacks its frozen historical-only scope")
    model = ContextSlopeCalibration.from_dict(_json(artifact_path))
    frame = pd.read_parquet(input_path)
    if not {"p_c0", "date", "x_low_statistics", "x_novel_roster_support",
            "x_high_sports_disagreement"}.issubset(frame):
        raise ValueError("prediction input misses frozen context fields")
    dates = pd.to_datetime(frame.date, errors="raise")
    if (dates < scenario_origin).any() or (dates >= valid_until).any():
        raise ValueError("prediction rows fall outside artifact historical validity interval")
    contexts = _contexts(frame)
    predictions = np.asarray(model.predict_proba(frame.p_c0.to_numpy(float), contexts), dtype=float)
    if not np.isfinite(predictions).all() or np.any((predictions <= 0) | (predictions >= 1)):
        raise ValueError("artifact generated nonfinite or noninterior probabilities")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    result["p_context"] = predictions
    result.to_parquet(output_path, index=False)
    return {"status": "PREDICTIONS_WRITTEN", "rows": len(result), "output_sha256": _sha256(output_path),
            "scenario_origin": scenario_origin.strftime("%Y-%m-%d"),
            "valid_until": valid_until.strftime("%Y-%m-%d"),
            "scope": artifact_receipt["scope"], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_activation": SOURCE_BLOCKED}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--research-root")
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--inputs", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = {"protocol": str(freeze_run(args.run_id, args.research_root))}
    elif args.command == "run":
        result = run_protocol(args.protocol)
    elif args.command == "smoke":
        result = run_smoke(args.output_dir)
    else:
        result = predict_artifact(args.artifact, args.inputs, args.output)
    print(json.dumps(_safe_json(result), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
