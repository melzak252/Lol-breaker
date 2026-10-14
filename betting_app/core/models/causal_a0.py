"""Local adapter for the hash-pinned, frozen annual causal A0 predictor."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

from betting_app.core.models.contract import UnifiedPredictionResult
from src.analysis import ev_asof_replay as replay


NATIVE_MANIFEST_SHA256 = "89c3cfa86d7b31cd6175b6a9c2b0e4baf976583fe637d5c7f812524095d3f5b8"
SUPPORTED_YEARS = (2024, 2025, 2026)
MODES = ("full", "no_w20", "no_organization")
_MODE_INDEX = {name: index for index, name in enumerate(MODES)}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# These are the archived modules imported by A0Predictor's inference path.
# Pin them as well as the four native A0 sources before importing either tree.
_DEPENDENCY_SOURCE_SHA256 = {
    "src/models/beta_calibration.py": "9605d8f626dc749b67ac7585e44a85112bad1d6e994e01c6014ec54a23dd76f1",
    "src/models/beta_terminal.py": "c4d53445913e4cb7e33d5fc0883c8777242bcf8e047324454700a51efeac49a0",
    "src/models/roster_optional.py": "9a4efe5be2b7f2b01f26b3070977afeda1f4e56a12f85032684361fdee11f7f3",
    "src/models/tail_reliability.py": "3f1eb5f5d32125341b2a6e607c3ef5ac7bb461c3b4ce35001b5c7c534805376b",
    "src/models/nonlinear_history.py": "f6a500683f197cee5fc5c9f51f87847b894215cd882b364dd5dc168da0b42025",
    "src/models/beta_training.py": "e9d5e8d8a24c05e68bb18c85e1fec26b9e575dfa906efdfa4966c2cebb49c467",
    "src/models/unified_ratings.py": "62c9e94376d0b0a023089edaab4147c2944644a9777775657c80c85d40e51d98",
    "src/models/coherent_series.py": "7ab105ba63cdbe45627e1bc9eb558aacc18c0f38b6872f450991b70cf378367e",
    "src/models/temporal_training.py": "d0407ee651f7c2fb3bf7636273f01678f354565f7806dbf4da65f9cd549ee913",
    "src/models/temporal_architectures.py": "8f67ac1dd634ae7f9d0ac643c0222a7ffc1d431b39599ca05d89d74852d2ccb2",
    "src/models/auxiliary_player.py": "f8a4a24401b318835a35b921db1a93d3a3c4be1b62ebbc6e1c2734d3cca6df0a",
    "src/models/beta_training_torch.py": "56ba596993c9352d283fec39950213427b27b3ab9ea666fbe195a1b48a476f6f",
    "src/models/stopped_series_torch.py": "b6b1ab2780fe8b7664c5b669d14659f8b50fe56509c1b12f80995ba0737dbf98",
}
_ACTUAL_CLOCK_FIELDS = (
    "prediction_at", "fit_origin", "decision_at", "start_at", "label_available_at",
    "quote_at_a", "quote_at_b", "feature_available_at", "close_available_at",
    "close_quote_at_a", "close_quote_at_b",
)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"Invalid SHA-256 in {label}: {expected!r}")
    return replay._verify_hash(path, expected, label)


def _aware_timestamp(value: Any, label: str) -> pd.Timestamp | None:
    if _is_missing(value):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must include a timezone: {value!r}")
    return timestamp.tz_convert("UTC")


def _parse_id_sequence(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} must be a JSON player sequence") from error
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a five-player sequence")
    if any(isinstance(item, (list, tuple, dict, set)) for item in value):
        raise ValueError(f"{label} must contain scalar player IDs")
    normalized = [replay._normalize_id(item, f"{label} player ID") for item in value]
    if len(normalized) != 5 or any(item is None for item in normalized):
        raise ValueError(f"{label} must contain five identified players")
    result = [str(item) for item in normalized]
    if len(set(result)) != 5:
        raise ValueError(f"{label} must contain five unique players")
    return result

def _input_hashes(raw: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in (*raw.items(), *hist.items())
    }


def _validate_arrays(
    arrays: Mapping[str, Any] | None,
    shapes: Mapping[str, tuple[int, ...]],
    label: str,
    *,
    mask: bool = False,
    champions: bool = False,
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError(f"{label} must contain exactly {sorted(shapes)}")
    validated: dict[str, np.ndarray] = {}
    for key, shape in shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != shape:
            raise ValueError(f"{label}.{key} must have unbatched shape {shape}, got {value.shape}")
        if champions and key == "champions":
            if value.dtype.kind not in "iu" or (value < 0).any():
                raise ValueError("hist.champions must contain integer nonnegative codes")
        elif mask and key == "mask":
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all() or ((value != 0) & (value != 1)).any():
                raise ValueError("hist.mask must contain only finite binary values")
        elif value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError(f"{label}.{key} must contain only finite numeric values")
        validated[key] = value
    return validated


def _validate_declared_clocks(target: Mapping[str, Any], cutoff: date) -> None:
    actual: dict[str, pd.Timestamp] = {}
    for field in _ACTUAL_CLOCK_FIELDS:
        value = target.get(field)
        timestamp = _aware_timestamp(value, field)
        if timestamp is not None:
            actual[field] = timestamp

    effective = _aware_timestamp(target.get("effective_decision_at"), "effective_decision_at")
    scenario: dict[str, pd.Timestamp] = {}
    for field, value in target.items():
        if field.startswith("scenario_") and field != "scenario_timestamp_assumption":
            timestamp = _aware_timestamp(value, field)
            if timestamp is not None:
                scenario[field] = timestamp

    decision_source = target.get("decision_time_source")
    if not _is_missing(decision_source) and decision_source not in ("actual", "scenario"):
        raise ValueError(f"Unsupported decision_time_source: {decision_source!r}")
    decision = actual.get("decision_at")
    start = actual.get("start_at")
    label_available = actual.get("label_available_at")
    scenario_decision = scenario.get("scenario_decision_at")
    declared_decision = effective if effective is not None else decision
    if declared_decision is None:
        declared_decision = scenario_decision
    declared_start = start if start is not None else scenario.get("scenario_start_at")
    if decision_source == "actual" and decision is None:
        raise ValueError("decision_time_source='actual' requires an actual decision_at")
    if decision_source == "scenario" and decision is not None:
        raise ValueError("scenario decision source cannot replace a supplied actual decision_at")
    if decision_source == "scenario" and (effective is None or scenario_decision is None):
        raise ValueError("a scenario decision requires effective_decision_at and scenario_decision_at")
    if decision is not None and effective is not None and decision != effective:
        raise ValueError("effective_decision_at differs from the declared actual decision_at")
    if decision_source == "actual" and effective is None:
        raise ValueError("an actual decision_at requires its effective_decision_at")
    if declared_decision is not None and declared_start is not None and declared_decision >= declared_start:
        raise ValueError("declared decision must precede the actual or scenario start")
    if start is not None and label_available is not None and label_available < start:
        raise ValueError("actual label_available_at cannot precede start_at")
    for observed_name in ("quote_at_a", "quote_at_b", "feature_available_at"):
        observed = actual.get(observed_name)
        if observed is not None and declared_decision is not None and observed > declared_decision:
            raise ValueError(f"{observed_name} cannot follow the declared decision cutoff")
    if declared_decision is not None and declared_decision.date() != cutoff:
        raise ValueError("cutoff_day differs from the UTC date of the declared decision")
    if scenario_decision is not None and effective is not None and decision_source == "scenario" and scenario_decision != effective:
        raise ValueError("effective_decision_at differs from the declared scenario_decision_at")




def validate_snapshot(snapshot: replay.AsOfSnapshot, *, year: int) -> int:
    """Validate a snapshot against the original annual A0 recipe year."""
    if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)) or int(year) not in SUPPORTED_YEARS:
        raise ValueError(f"year must be one of {SUPPORTED_YEARS}")
    return _validate_snapshot_contract(snapshot, year=int(year))


def _validate_snapshot_contract(snapshot: replay.AsOfSnapshot, *, year: int) -> int:
    """Validate the shared native arrays/clocks with an explicit cutoff year."""
    if not isinstance(snapshot, replay.AsOfSnapshot):
        raise TypeError("snapshot must be an AsOfSnapshot")
    if not isinstance(snapshot.available, (bool, np.bool_)) or not bool(snapshot.available):
        raise ValueError("snapshot must be feature-available")
    target = snapshot.target
    if not isinstance(target, Mapping):
        raise ValueError("snapshot.target must be a mapping")

    raw = _validate_arrays(snapshot.raw, replay._RAW_SHAPES, "raw")
    hist = _validate_arrays(snapshot.hist, replay._TEMPORAL_SHAPES, "hist", mask=True, champions=True)

    cutoff_value = target.get("cutoff_day")
    try:
        cutoff = replay._as_day(cutoff_value, "snapshot cutoff day")
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot requires a valid cutoff_day") from error
    if cutoff.year != int(year):
        raise ValueError("cutoff year differs from the annual A0 recipe year")
    model_year = target.get("model_year")
    if not _is_missing(model_year):
        if isinstance(model_year, (bool, np.bool_)) or not isinstance(model_year, (int, np.integer)) or int(model_year) != int(year):
            raise ValueError("target model_year differs from the annual A0 recipe year")
    best_of = target.get("best_of")
    if isinstance(best_of, (bool, np.bool_)) or not isinstance(best_of, (int, np.integer)) or int(best_of) not in (1, 3, 5):
        raise ValueError("best_of must be one of 1, 3, or 5")

    rosters: dict[str, list[str]] = {}
    for side in ("a", "b"):
        roster = _parse_id_sequence(target.get(f"roster_{side}"), f"roster_{side}")
        tensor = _parse_id_sequence(target.get(f"tensor_player_order_{side}"), f"tensor_player_order_{side}")
        raw_order = _parse_id_sequence(target.get(f"raw_tensor_player_order_{side}"), f"raw_tensor_player_order_{side}")
        temporal_order = _parse_id_sequence(target.get(f"temporal_tensor_player_order_{side}"), f"temporal_tensor_player_order_{side}")
        if set(roster) != set(tensor):
            raise ValueError(f"tensor player slots do not match roster_{side} membership")
        if raw_order != tensor or temporal_order != tensor:
            raise ValueError(f"raw, temporal, and tensor player slot order differs for side {side.upper()}")
        rosters[side] = roster
    if set(rosters["a"]).intersection(rosters["b"]):
        raise ValueError("team rosters must be disjoint")

    source_day = target.get("feature_source_max_day")
    history_exists = any(np.any(value != 0) for value in (*raw.values(), *hist.values()))
    if _is_missing(source_day):
        if history_exists:
            raise ValueError("feature_source_max_day is required for nonempty feature history")
    else:
        latest = replay._as_day(source_day, "feature source max day")
        if latest >= cutoff:
            raise ValueError("feature_source_max_day must be strictly before cutoff_day")

    _validate_declared_clocks(target, cutoff)
    return int(best_of)

FIXED_2026_CUTOFF_START = "2026-11-01"
FIXED_2026_CUTOFF_END = "2029-07-01"  # exclusive; the E3 development window ends here
FIXED_2026_CONTRACT_VERSION = "frozen-a0-2026-forward-v1"


def validate_fixed_2026_snapshot(snapshot: replay.AsOfSnapshot) -> int:
    """Validate a causal snapshot for the explicitly declared frozen-2026 service window."""
    if not isinstance(snapshot, replay.AsOfSnapshot) or not isinstance(snapshot.target, Mapping):
        raise TypeError("snapshot must be an AsOfSnapshot with a target mapping")
    try:
        cutoff = replay._as_day(snapshot.target.get("cutoff_day"), "snapshot cutoff day")
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot requires a valid cutoff_day") from error
    start = date.fromisoformat(FIXED_2026_CUTOFF_START)
    end = date.fromisoformat(FIXED_2026_CUTOFF_END)
    if cutoff < start or cutoff >= end:
        raise ValueError(
            f"cutoff {cutoff.isoformat()} is outside the fixed 2026 serving contract "
            f"[{FIXED_2026_CUTOFF_START}, {FIXED_2026_CUTOFF_END})"
        )
    # The snapshot's calendar model_year is its cutoff year; the serving
    # artifact remains fixed to 2026 and is identified separately.
    return _validate_snapshot_contract(snapshot, year=cutoff.year)


def _checkpoint_paths(year: int) -> tuple[str, ...]:
    return (
        f"data/06_models/causal_a0/{year}/nn_20260910.joblib",
        f"data/06_models/causal_a0/{year}/nn_20260911.joblib",
        f"data/06_models/causal_a0/{year}/nn_20260912.joblib",
        f"data/06_models/causal_a0/{year}/other/{year}_ratings_raw_20260910.joblib",
        f"data/06_models/causal_a0/{year}/other/{year}_full_mlp_20260910.joblib",
        f"data/06_models/causal_a0/{year}/other/{year}_full_mlp_20260911.joblib",
        f"data/06_models/causal_a0/{year}/other/{year}_full_mlp_20260912.joblib",
        f"data/06_models/causal_a0/{year}/mixture_full.joblib",
        f"data/06_models/causal_a0/{year}/mixture_no_w20.joblib",
        f"data/06_models/causal_a0/{year}/mixture_no_organization.joblib",
    )


class CausalA0Adapter:
    """A lazy, read-only adapter over one hash-verified annual native A0 checkpoint."""

    def __init__(self, year: int, research_root: str | Path | None = None):
        if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)) or int(year) not in SUPPORTED_YEARS:
            raise ValueError(f"year must be one of {SUPPORTED_YEARS}")
        self.year = int(year)
        self.research_root = replay._resolve_research_root(research_root)
        self._identity: dict[str, Any] | None = None
        self._predictor: Any | None = None

    def _verify_identity(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._identity is not None and not refresh:
            return self._identity
        self._identity = None

        archive_root = self.research_root / replay.CAUSAL_ARCHIVE_RELATIVE
        manifest_path = archive_root / "data/06_models/a0_manifest_causal.json"
        _verify_hash(manifest_path, NATIVE_MANIFEST_SHA256, "pinned native A0 manifest")
        manifest = replay._read_json(manifest_path)
        if (
            manifest.get("model") != "causal_a0_same_recipe"
            or manifest.get("policy") != "frozen annual A0; causal histories per real origin; no monthly refit"
            or manifest.get("years") != list(SUPPORTED_YEARS)
        ):
            raise ValueError("native A0 manifest does not describe the pinned frozen annual recipe")

        source_records = manifest.get("predictor_sha256"), manifest.get("causal_predictor_sha256")
        if source_records != (
            replay.NATIVE_SOURCE_SHA256["code/a0_predictor.py"],
            replay.NATIVE_SOURCE_SHA256["code/causal_predictor.py"],
        ):
            raise ValueError("native A0 manifest source identity differs from the pinned adapter")
        source_hashes = replay._verify_native_sources(self.research_root)
        model_source_hashes = {
            Path(path).resolve().relative_to(self.research_root).as_posix(): digest
            for path, digest in source_hashes.items()
        }

        dependency_root = self.research_root / "nonlinear-history-20260912/code"
        dependency_hashes: dict[str, str] = {}
        for relative, expected in _DEPENDENCY_SOURCE_SHA256.items():
            path = dependency_root / relative
            dependency_hashes[f"nonlinear-history-20260912/code/{relative}"] = _verify_hash(
                path, expected, f"native inference dependency {relative}",
            )

        from src.analysis.ev_approach_data import _research_artifact_path

        manifest_hashes = manifest.get("hashes")
        if not isinstance(manifest_hashes, dict):
            raise ValueError("native A0 manifest has no checkpoint hash inventory")
        resolved_manifest_hashes: dict[Path, str] = {}
        for recorded_path, digest in manifest_hashes.items():
            path = _research_artifact_path(str(recorded_path), self.research_root)
            resolved = path.resolve()
            if resolved in resolved_manifest_hashes:
                raise ValueError(f"native manifest repeats a checkpoint path: {resolved}")
            resolved_manifest_hashes[resolved] = str(digest)

        calibration_relative = f"data/06_models/causal_a0/{self.year}/calibration.json"
        calibration_path = archive_root / calibration_relative
        calibration_digest = resolved_manifest_hashes.get(calibration_path.resolve())
        if calibration_digest is None:
            raise ValueError("native A0 manifest omits the selected year's calibration receipt")
        _verify_hash(calibration_path, calibration_digest, "native annual calibration receipt")
        receipt = replay._read_json(calibration_path)
        if receipt.get("status") != "COMPLETE" or receipt.get("year") != self.year:
            raise ValueError("native annual calibration receipt is incomplete or belongs to another year")
        if receipt.get("code_sha256") != replay.NATIVE_SOURCE_SHA256["code/causal_calibrate.py"]:
            raise ValueError("native annual calibration receipt source identity differs")

        required_checkpoint_paths = _checkpoint_paths(self.year)
        expected_artifacts: dict[str, str] = {}
        for relative in required_checkpoint_paths:
            path = archive_root / relative
            digest = resolved_manifest_hashes.get(path.resolve())
            if digest is None:
                raise ValueError(f"native A0 manifest omits checkpoint {relative}")
            _verify_hash(path, digest, f"native A0 checkpoint {relative}")
            expected_artifacts[relative] = digest

        verified_receipt_assets: dict[str, str] = {}
        for section in ("inputs", "outputs"):
            entries = receipt.get(section)
            if not isinstance(entries, dict) or not entries:
                raise ValueError(f"native annual calibration receipt lacks {section}")
            for recorded_path, digest in entries.items():
                path = _research_artifact_path(str(recorded_path), self.research_root).resolve()
                actual = _verify_hash(path, digest, f"native calibration {section} artifact")
                verified_receipt_assets[path.relative_to(self.research_root).as_posix()] = actual
        expected_checkpoint_hashes = {
            (archive_root / relative).resolve(): digest for relative, digest in expected_artifacts.items()
        }
        for path, digest in expected_checkpoint_hashes.items():
            if verified_receipt_assets.get(path.relative_to(self.research_root).as_posix()) != digest:
                raise ValueError(f"calibration receipt disagrees with pinned native checkpoint {path}")

        source_hashes_json = {
            **model_source_hashes,
            **dependency_hashes,
        }
        self._identity = {
            "name": "causal_a0_native_annual",
            "version": f"frozen-causal-a0-v1-{self.year}",
            "year": self.year,
            "manifest_sha256": NATIVE_MANIFEST_SHA256,
            "calibration_sha256": calibration_digest,
            "artifact_root": replay.CAUSAL_ARCHIVE_RELATIVE.as_posix(),
            "adapter_source_sha256": replay._sha256(Path(__file__)),
            "artifact_hashes": dict(sorted(expected_artifacts.items())),
            "source_hashes": dict(sorted(source_hashes_json.items())),
        }
        return self._identity

    @property
    def identity(self) -> dict[str, Any]:
        """Verified JSON-safe model identity; does not unpickle model artifacts."""
        identity = self._verify_identity()
        return {
            **identity,
            "artifact_hashes": dict(identity["artifact_hashes"]),
            "source_hashes": dict(identity["source_hashes"]),
        }

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        identity = self._verify_identity(refresh=True)

        dependency_root = (self.research_root / "nonlinear-history-20260912/code/src/models").resolve()
        models_package = importlib.import_module("src.models")
        package_paths = [str(Path(path).resolve()) for path in models_package.__path__]
        archive_models = str(dependency_root)
        models_package.__path__ = [archive_models, *(path for path in package_paths if path != archive_models)]

        for relative in _DEPENDENCY_SOURCE_SHA256:
            name = Path(relative).stem
            loaded = sys.modules.get(f"src.models.{name}")
            expected = dependency_root / f"{name}.py"
            if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != expected:
                raise ImportError(f"Refusing non-archived native A0 dependency {name}")

        for relative, expected_digest in _DEPENDENCY_SOURCE_SHA256.items():
            name = Path(relative).stem
            module_name = f"src.models.{name}"
            module = importlib.import_module(module_name)
            expected_path = dependency_root / Path(relative).name
            if Path(module.__file__).resolve() != expected_path:
                raise ImportError(f"Refusing non-archived native A0 dependency {module_name}")
            _verify_hash(expected_path, expected_digest, f"imported native A0 dependency {module_name}")

        code_root = (self.research_root / replay.CAUSAL_ARCHIVE_RELATIVE / "code").resolve()
        for name in ("a0_predictor", "causal_predictor"):
            loaded = sys.modules.get(name)
            expected = code_root / f"{name}.py"
            if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != expected:
                raise ImportError(f"Refusing shadowed native A0 module {name}")
        code_path = str(code_root)
        if code_path in sys.path:
            sys.path.remove(code_path)
        sys.path.insert(0, code_path)
        importlib.invalidate_caches()
        base_module = importlib.import_module("a0_predictor")
        replay._verify_imported_module(
            base_module, code_root / "a0_predictor.py",
            replay.NATIVE_SOURCE_SHA256["code/a0_predictor.py"], "a0_predictor",
        )
        native_module = importlib.import_module("causal_predictor")
        replay._verify_imported_module(
            native_module, code_root / "causal_predictor.py",
            replay.NATIVE_SOURCE_SHA256["code/causal_predictor.py"], "causal_predictor",
        )
        if not issubclass(native_module.CausalA0Predictor, base_module.A0Predictor):
            raise TypeError("Archived CausalA0Predictor does not inherit the archived A0 inference contract")

        predictor = native_module.CausalA0Predictor(self.year)
        expected_hashes = {
            str((self.research_root / replay.CAUSAL_ARCHIVE_RELATIVE / relative).resolve()): digest
            for relative, digest in identity["artifact_hashes"].items()
        }
        expected_hashes[str((self.research_root / replay.CAUSAL_ARCHIVE_RELATIVE / "data/06_models/causal_a0" / str(self.year) / "calibration.json").resolve())] = identity["calibration_sha256"]
        # CausalA0Predictor records the annual receipt plus every unpickled checkpoint.
        observed_hashes = {str(Path(path).resolve()): str(digest) for path, digest in predictor.hashes.items()}
        if observed_hashes != expected_hashes:
            raise ValueError("loaded native A0 artifact identity differs from its verified receipt")
        self._predictor = predictor
        return predictor

    def _validate_snapshot(self, snapshot: replay.AsOfSnapshot) -> int:
        return validate_snapshot(snapshot, year=self.year)


    def predict_snapshot(
        self,
        snapshot: replay.AsOfSnapshot,
        *,
        mode: str = "full",
    ) -> UnifiedPredictionResult:
        if not isinstance(mode, str) or mode not in _MODE_INDEX:
            raise ValueError(f"mode must be one of {MODES}")
        best_of = self._validate_snapshot(snapshot)
        target = snapshot.target
        information_status = target.get("information_status", "local_fixture")
        if not isinstance(information_status, str) or not information_status.strip():
            raise ValueError("information_status must be a nonempty declared status")

        raw = _validate_arrays(snapshot.raw, replay._RAW_SHAPES, "raw")
        hist = _validate_arrays(snapshot.hist, replay._TEMPORAL_SHAPES, "hist", mask=True, champions=True)
        before = _input_hashes(raw, hist)
        predictor = self._load_predictor()
        raw_batch = {key: value[None, ...].copy() for key, value in raw.items()}
        hist_batch = {key: value[None, ...].copy() for key, value in hist.items()}
        all_modes = np.asarray(
            predictor.predict(raw_batch, hist_batch, np.asarray([best_of], dtype=np.int64)),
            dtype=float,
        )
        after = _input_hashes(raw, hist)
        if before != after:
            raise ValueError("native A0 inference mutated the input snapshot")
        if (
            all_modes.shape != (1, len(MODES))
            or not np.isfinite(all_modes).all()
            or ((all_modes <= 0.0) | (all_modes >= 1.0)).any()
        ):
            raise ValueError("native A0 returned invalid all-mode series probabilities")

        mode_probabilities = {name: float(all_modes[0, index]) for index, name in enumerate(MODES)}
        probability = mode_probabilities[mode]
        identity = self.identity
        origin = target.get("effective_decision_at")
        if _is_missing(origin):
            origin = target.get("scenario_decision_at")
        if _is_missing(origin):
            origin = None
        elif isinstance(origin, (pd.Timestamp, datetime, date)):
            origin = origin.isoformat()
        else:
            origin = str(origin)
        diagnostics = {
            "actual_decision_at": (
                _aware_timestamp(target.get("decision_at"), "decision_at").isoformat()
                if not _is_missing(target.get("decision_at")) else None
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "information_status": information_status,
            "information_status_basis": "declared_not_authenticated",
            "mode": mode,
            "mode_probabilities": mode_probabilities,
            "input_hashes": before,
            "model_identity": identity,
            "simulated_origin_at": origin,
        }
        return UnifiedPredictionResult(
            canonical_match_id=None,
            prob_a=probability,
            prob_b=1.0 - probability,
            map_prob_a=None,
            map_prob_b=None,
            model_name=identity["name"],
            model_version=identity["version"],
            feature_version="causal-a0-corrected-history-v1",
            best_of=best_of,
            diagnostics=diagnostics,
        )
class FixedArtifactA0Adapter(CausalA0Adapter):
    """Explicit frozen-2026 weights/calibration with causal later-cutoff inputs."""

    def __init__(self, research_root: str | Path | None = None):
        super().__init__(2026, research_root=research_root)

    def _validate_snapshot(self, snapshot: replay.AsOfSnapshot) -> int:
        return validate_fixed_2026_snapshot(snapshot)

    @property
    def identity(self) -> dict[str, Any]:
        native = super().identity
        return {
            **native,
            "name": "causal_a0_frozen_2026_fixed_forward",
            "version": FIXED_2026_CONTRACT_VERSION,
            "artifact_year": 2026,
            "serving_contract": {
                "version": FIXED_2026_CONTRACT_VERSION,
                "valid_cutoff_start_inclusive": FIXED_2026_CUTOFF_START,
                "valid_cutoff_end_exclusive": FIXED_2026_CUTOFF_END,
                "history_policy": "causal release-aware history as of each decision cutoff; no refit",
                "training_calibration_artifact_year": 2026,
                "annual_artifact_identity": native["version"],
                "manifest_sha256": NATIVE_MANIFEST_SHA256,
                "calibration_sha256": native["calibration_sha256"],
                "artifact_hashes": dict(native["artifact_hashes"]),
            },
        }
