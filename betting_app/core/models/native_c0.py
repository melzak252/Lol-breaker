"""Hash-pinned native C0 inference primitives; load this only in its isolated worker."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from datetime import date, datetime, time, timezone
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "Causal-C0"
MODEL_VERSION = "c0-native-2026-w32-e12-v1"
FEATURE_VERSION = "native-c0-history16-v1"
MODEL_FAMILY = "native_c0"
_CANONICAL_RAW_HISTORY_RELATIVE = "data/artifacts/golgg-database-recovery-20260909/matches.json"
_CANONICAL_RAW_HISTORY_SHA256 = "991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca"
_CANONICAL_RAW_HISTORY_KIND = "canonical_frozen_golgg_history"
VARIANT = "control"
_MODEL_ROOT_RELATIVE = "data/06_models/a0_data_focus/a0_data_focus_20260929_192443"
_COMPLETION_RELATIVE = "data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/completed.json"
MODES = ("full", "no_w20", "no_organization")
_MINIMUM_DECISION_DATE = date(2026, 1, 1)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# These pins identify the adopted original C0 control run, never the separate W32/E12
# capacity challenger. Completion and execution bytes bind the training contract and
# receipt inventory; the selected joblib bytes are checked against both static pins
# and the immutable completion inventory before any archive module or pickle is loaded.
_EXECUTION_SHA256 = "02a41a6b99097086cb0d75c67e85e832643a24ed2214d9c4dcaa32d3613ee2b8"
_COMPLETION_SHA256 = "9056ecfdc5e3f3764e61320017c3114aea6e3554d316ca2c8c21b6e1bd735a15"
_ARTIFACT_SHA256 = {
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/calibration/model.joblib": "2303add90fd36b1724e1e381c91706860221484080edf66968e170017710182a",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/calibration/model.joblib.receipt.json": "12a2a4770598c47604782478116a048e3b99640d1fa5e7db2cf55c3cd6857552",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260910/model.joblib": "ffa561bc54982ed71019edca568b06584e792ede4b3f210e5aa1593b7069b1e8",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260910/model.joblib.receipt.json": "982f270a8ee67dcbee48b18ff08785fc687069d8b6f76e97db088428bd3b87ba",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260911/model.joblib": "3cc676e54f0ff298bb1f3becf125e18ed7743d460101f7b5941931da54091112",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260911/model.joblib.receipt.json": "1586fd7c31d0257acbc5c7142a90e7f0bf872143b9d30238ef28f19b49655553",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260912/model.joblib": "f6237f85c137aeb06cdd1e4c07373ffc91b128b930b473957389c0effb5c405e",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control/seed_20260912/model.joblib.receipt.json": "9c6dcb3d72ca49d7420b47760784bb005ec4ab43bdde50d4d9b8afc46b459cd0",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib": "94106b0bc9a675108eaff6b11c9ff22c670b183471c262e56c8aff5cc61c8c78",
    "data/06_models/a0_data_focus/a0_data_focus_20260929_192443/shared/model.joblib.receipt.json": "7d75bb93779a3ad2c2f80a54fa9eabe4d93ea159dec461068af9763bed502244",
}
_SOURCE_SHA256 = {
    "project:src/models/a0_accelerator.py": "4d9cc185fdc4a6392586c59b00f8c977000a06142a0977564a8526f8a2f1d8f6",
    "project:src/models/a0_uncertainty.py": "1e82c30bae6c30d13c1fc32601c438b7a97aef7d5476cd07f387976eb0d42b83",
    "research:a0-phase-walkforward-20260914/code/causal_calibrate.py": "fdb526e652f1a1316810f1d4e30b090d1791b92f859794c628995b5e95528a26",
    "research:a0-phase-walkforward-20260914/code/causal_nn.py": "afabdd9169c9eb8df95b74f3c225fd37c158e989cf3a1ea41801d41ad97a3d3c",
    "research:a0-phase-walkforward-20260914/code/retrain_other_experts.py": "c51eac9d1b07292a48daab9dd638758b3c58bdfa45df18bcc187abdbb40fe0f6",
    "research:nonlinear-history-20260912/code/src/models/auxiliary_player.py": "f8a4a24401b318835a35b921db1a93d3a3c4be1b62ebbc6e1c2734d3cca6df0a",
    "research:nonlinear-history-20260912/code/src/models/beta_calibration.py": "9605d8f626dc749b67ac7585e44a85112bad1d6e994e01c6014ec54a23dd76f1",
    "research:nonlinear-history-20260912/code/src/models/beta_terminal.py": "c4d53445913e4cb7e33d5fc0883c8777242bcf8e047324454700a51efeac49a0",
    "research:nonlinear-history-20260912/code/src/models/beta_training.py": "e9d5e8d8a24c05e68bb18c85e1fec26b9e575dfa906efdfa4966c2cebb49c467",
    "research:nonlinear-history-20260912/code/src/models/beta_training_torch.py": "56ba596993c9352d283fec39950213427b27b3ab9ea666fbe195a1b48a476f6f",
    "research:nonlinear-history-20260912/code/src/models/coherent_series.py": "7ab105ba63cdbe45627e1bc9eb558aacc18c0f38b6872f450991b70cf378367e",
    "research:nonlinear-history-20260912/code/src/models/competition_tiers.py": "5ace7ea27148dee921b09e9d6b47ebcbcc33d83fcc3fe0423280473b115f825c",
    "research:nonlinear-history-20260912/code/src/models/nonlinear_history.py": "f6a500683f197cee5fc5c9f51f87847b894215cd882b364dd5dc168da0b42025",
    "research:nonlinear-history-20260912/code/src/models/roster_optional.py": "9a4efe5be2b7f2b01f26b3070977afeda1f4e56a12f85032684361fdee11f7f3",
    "research:nonlinear-history-20260912/code/src/models/stopped_series_torch.py": "b6b1ab2780fe8b7664c5b669d14659f8b50fe56509c1b12f80995ba0737dbf98",
    "research:nonlinear-history-20260912/code/src/models/tail_reliability.py": "3f1eb5f5d32125341b2a6e607c3ef5ac7bb461c3b4ce35001b5c7c534805376b",
    "research:nonlinear-history-20260912/code/src/models/temporal_architectures.py": "8f67ac1dd634ae7f9d0ac643c0222a7ffc1d431b39599ca05d89d74852d2ccb2",
    "research:nonlinear-history-20260912/code/src/models/temporal_training.py": "d0407ee651f7c2fb3bf7636273f01678f354565f7806dbf4da65f9cd549ee913",
    "research:nonlinear-history-20260912/code/src/models/unified_ratings.py": "62c9e94376d0b0a023089edaab4147c2944644a9777775657c80c85d40e51d98",
    "scripts/train_a0_data_focus.py": "cb5407fb109b1600cb0a9f771e5fb1bae04c51a138d304ca91df8d6160c87784",
    "src/models/a0_data_focus.py": "086443c014cbaf7e46efe4e97a1098e5b400eec6f375f8ff8311d9a6e7aaec71",
    "src/models/a0_native_refit.py": "271f3e31ae6e26995b668fade52275b4dce00e3f8339a9934277c48aa1a72946",
    "src/models/a0_target_experiment.py": "a61006d22292e1845899deaa08ce280545f32e168d859cc9df515414e864d4ac",
    "src/models/a0_target_objectives.py": "6408ceee1c1a794fab46a6804b8482f32c4a6c25c570052b7ca07d307d82a3ca",
    "src/models/competition_tiers.py": "5ace7ea27148dee921b09e9d6b47ebcbcc33d83fcc3fe0423280473b115f825c",
}
_SERVING_SOURCE_SHA256 = {
    "project:src/models/a0_data_focus.py": "e7caf834f0a0f58cf302ccf7411c06ad93af111a8130ac062e15f91d4972b27f",
    "project:src/models/a0_native_refit.py": "271f3e31ae6e26995b668fade52275b4dce00e3f8339a9934277c48aa1a72946",
    "project:src/models/a0_target_experiment.py": "a61006d22292e1845899deaa08ce280545f32e168d859cc9df515414e864d4ac",
    "project:src/models/a0_target_objectives.py": "6408ceee1c1a794fab46a6804b8482f32c4a6c25c570052b7ca07d307d82a3ca",
    "project:src/models/a0_uncertainty.py": "1e82c30bae6c30d13c1fc32601c438b7a97aef7d5476cd07f387976eb0d42b83",
    "project:src/models/a0_accelerator.py": "4d9cc185fdc4a6392586c59b00f8c977000a06142a0977564a8526f8a2f1d8f6",
    "project:src/models/a0_history_reconstruction.py": "147fa35c6848ea9a2a98357509c127ae61a412428818c4cfed5ecb8f821b6b91",
    "project:src/models/competition_tiers.py": "5ace7ea27148dee921b09e9d6b47ebcbcc33d83fcc3fe0423280473b115f825c",
    "project:src/analysis/ev_asof_replay.py": "f01fdc9eef9ff551cc1bc592628f44b0b17dacba84750dfb0328f21b9994e58d",
}
_FEATURE_REPLAY_SHA256 = "f01fdc9eef9ff551cc1bc592628f44b0b17dacba84750dfb0328f21b9994e58d"
_RAW_SHAPES = {
    "players": (2, 5, 21), "core": (2, 5), "team": (2, 11), "w20": (2, 10), "gates": (2, 2),
}
_TEMPORAL_SHAPES = {"history": (2, 5, 16, 33), "mask": (2, 5, 16), "champions": (2, 5, 16)}
_WIRE_DTYPES = {
    "float32": np.dtype("float32"), "float16": np.dtype("float16"),
    "uint8": np.dtype("uint8"), "uint16": np.dtype("uint16"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"Invalid pinned SHA-256 for {label}")
    if not path.is_file():
        raise FileNotFoundError(f"Pinned {label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")
    return actual


def _decode_json(content: bytes, label: str, path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {label}")

    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON {label}: {path}") from exc


def _read_json(path: Path, label: str) -> Any:
    return _decode_json(path.read_bytes(), label, path)


def _read_pinned_json(path: Path, expected: str, label: str) -> tuple[Any, str]:
    if _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"Invalid pinned SHA-256 for {label}")
    try:
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Pinned {label} is missing: {path}") from exc
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")
    return _decode_json(content, label, path), actual


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _resolve_research_root() -> Path:
    configured = next((os.environ[name] for name in (
        "C0_RESEARCH_ROOT", "ENSEMBLE_RESEARCH_ROOT", "RESEARCH_ROOT",
    ) if os.environ.get(name)), None)
    if configured is None:
        pointer = PROJECT_ROOT / "data/research_root.txt"
        if not pointer.is_file():
            raise FileNotFoundError(f"C0 research-root pointer is missing: {pointer}")
        configured = pointer.read_text(encoding="utf-8").strip()
    if not configured:
        raise ValueError("C0 research root must be nonempty")
    root = Path(configured).expanduser().resolve(strict=True)
    required = (
        root / "a0-phase-walkforward-20260914/code/causal_nn.py",
        root / "nonlinear-history-20260912/code/src/models/nonlinear_history.py",
        root / "unified-ratings-20260911/code/scripts/temporal_player_features.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("C0 research root is incomplete: " + ", ".join(missing))
    return root


def _source_path(source_id: str, research_root: Path) -> Path:
    if ":" in source_id:
        namespace, relative = source_id.split(":", 1)
        if namespace == "project":
            root = PROJECT_ROOT
        elif namespace == "research":
            root = research_root
        else:
            raise ValueError(f"Unknown C0 source namespace: {namespace!r}")
    else:
        # The frozen execution contract predates namespaced identities for its local
        # scripts/model sources; preserve its literal keys and resolve them in-project.
        relative = source_id
        root = PROJECT_ROOT
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe C0 source identity: {source_id!r}")
    path = (root / Path(*relative_path.parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"C0 source resolves outside its pinned root: {source_id}")
    return path


def _required_path(config: Mapping[str, Any], env_name: str, config_key: str) -> Path:
    override = os.environ.get(env_name)
    return _project_path(override if override else str(config[config_key]))


def _resolve_raw_history_source(
    config: Mapping[str, Any], override: str | None,
) -> tuple[Path, str, bool]:
    canonical_path = _project_path(_CANONICAL_RAW_HISTORY_RELATIVE)
    if _project_path(str(config["raw_history_path"])) != canonical_path:
        raise ValueError("C0 default raw history must be the pinned canonical frozen dataset")
    if override is not None and not override.strip():
        raise ValueError("C0_RAW_HISTORY_PATH override must be a nonempty path")
    raw_path = _project_path(override) if override else canonical_path
    raw_path = raw_path.resolve(strict=True)
    if not raw_path.is_file():
        raise FileNotFoundError(f"C0 raw-history source is not a regular file: {raw_path}")
    is_canonical = raw_path == canonical_path
    source_kind = _CANONICAL_RAW_HISTORY_KIND if is_canonical else "explicit_raw_history_override"
    return raw_path, source_kind, is_canonical


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return _project_path(config_path)
    configured = os.environ.get("C0_SERVING_CONFIG")
    return _project_path(configured or "conf/base/c0_serving.json")


def load_serving_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    path = _resolve_config_path(config_path)
    content = path.read_bytes()
    value = _decode_json(content, "C0 serving configuration", path)
    config_sha = hashlib.sha256(content).hexdigest()
    if not isinstance(value, dict) or value.get("schema") != "native_c0_serving_v1":
        raise ValueError("Unsupported native C0 serving configuration")
    config_keys = {
        "schema", "model_name", "model_version", "feature_version", "model_family",
        "model_root", "completion_report", "variant", "raw_history_path",
        "raw_history_sha256", "raw_history_source_kind", "release_map_relative",
        "startup_timeout_seconds", "request_timeout_seconds", "max_worker_requests",
        "max_worker_lifetime_seconds", "minimum_decision_date", "feature_replay_sha256",
        "forward_status",
    }
    if set(value) != config_keys:
        raise ValueError("C0 serving configuration fields differ from the pinned schema")
    expected = {
        "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION, "model_family": MODEL_FAMILY,
        "variant": VARIANT, "forward_status": "frozen_static_2026_artifact_not_prospective_certified",
        "raw_history_path": _CANONICAL_RAW_HISTORY_RELATIVE,
        "raw_history_sha256": _CANONICAL_RAW_HISTORY_SHA256,
        "raw_history_source_kind": _CANONICAL_RAW_HISTORY_KIND,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"C0 configuration {key} differs from the pinned native identity")
    for key in ("model_root", "completion_report", "release_map_relative"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"C0 configuration {key} must be a nonempty path")
    for key, lower, upper in (
        ("startup_timeout_seconds", 10, 1800),
        ("request_timeout_seconds", 1, 900),
        ("max_worker_requests", 1, 100000),
        ("max_worker_lifetime_seconds", 60, 86400),
    ):
        item = value.get(key)
        if type(item) is not int or not lower <= item <= upper:
            raise ValueError(f"C0 configuration {key} must be an integer in [{lower}, {upper}]")
    if value.get("minimum_decision_date") != _MINIMUM_DECISION_DATE.isoformat():
        raise ValueError("C0 configuration minimum decision date differs from its frozen input contract")
    if value.get("feature_replay_sha256") != _FEATURE_REPLAY_SHA256:
        raise ValueError("C0 feature replay source differs from its pinned input contract")
    return value, path, config_sha


class NativeC0Adapter:
    """Worker-only adapter over the immutable C0 control trio/shared/CAL assembly."""

    def __init__(self, config_path: str | Path | None = None):
        if os.environ.get("BETTING_APP_C0_WORKER") != "1":
            raise RuntimeError("Native C0 model loading is restricted to its isolated worker process")
        self.config, self.config_path, self.config_sha256 = load_serving_config(config_path)
        self.project_model_root = _required_path(self.config, "C0_MODEL_ROOT", "model_root")
        self.research_root = _resolve_research_root()
        self._identity: dict[str, Any] | None = None
        self._backend: Any = None
        self._shared: Any = None
        self._models: Any = None
        self._calibration: Any = None
        self._prepared: dict[str, Any] | None = None
        self._feature_replay: Any = None
        self._release_map_cache: tuple[Path, tuple[int, int, int, int, int], str, dict[str, str]] | None = None
        self._contract: dict[str, Any] | None = None

    def _preflight(self) -> dict[str, Any]:
        model_root = self.project_model_root
        expected_model_root = _project_path(_MODEL_ROOT_RELATIVE)
        if model_root != expected_model_root:
            raise ValueError("C0 model root must resolve to the pinned frozen control run")
        execution_path = model_root / "execution.json"
        completion_path = _required_path(self.config, "C0_COMPLETION_REPORT", "completion_report")
        expected_completion_path = _project_path(_COMPLETION_RELATIVE)
        if completion_path != expected_completion_path:
            raise ValueError("C0 completion report must resolve to the pinned frozen run receipt")
        execution, execution_sha = _read_pinned_json(
            execution_path, _EXECUTION_SHA256, "frozen C0 execution contract",
        )
        completion, completion_sha = _read_pinned_json(
            completion_path, _COMPLETION_SHA256, "frozen C0 completion receipt",
        )
        contract = execution.get("contract") if isinstance(execution, dict) else None
        protocol = execution.get("protocol") if isinstance(execution, dict) else None
        if (
            not isinstance(contract, dict)
            or contract.get("schema") != "a0_data_focus_execution_v1"
            or contract.get("run_id") != "a0_data_focus_20260929_192443"
            or contract.get("scope") != "DEVELOPMENT_ONLY"
            or not isinstance(protocol, dict)
            or protocol.get("schema") != "a0_data_focus_v1"
            or protocol.get("run_id") != "a0_data_focus_20260929_192443"
            or completion.get("status") != "TRAINING_COMPLETE_NOT_SCORED"
            or completion.get("contract") != contract
        ):
            raise ValueError("C0 completion/execution contract is incomplete or belongs to another run")
        if contract.get("source_sha256") != _SOURCE_SHA256:
            raise ValueError("Frozen C0 source inventory differs from the pinned native contract")
        artifacts = completion.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("C0 completion receipt omits its artifact inventory")
        artifact_hashes: dict[str, str] = {}
        for relative, expected in _ARTIFACT_SHA256.items():
            if artifacts.get(relative) != expected:
                raise ValueError(f"C0 completion receipt does not pin the selected asset {relative}")
            path = _project_path(relative)
            artifact_hashes[relative] = _verify_hash(path, expected, f"C0 artifact {relative}")
        if _project_path(str(self.config["completion_report"])) != completion_path:
            raise ValueError("C0 completion report configuration was changed")
        if _project_path(str(self.config["model_root"])) != model_root:
            raise ValueError("C0 model-root configuration was changed")

        # Training sources are immutable provenance declared by the separately pinned
        # execution receipt. Verify the original archived backends before any imports;
        # current project serving modules use their own hash closure below.
        for source_id, expected in _SOURCE_SHA256.items():
            if source_id.startswith("research:"):
                _verify_hash(
                    _source_path(source_id, self.research_root),
                    expected,
                    f"C0 frozen training backend source {source_id}",
                )
        serving_source_hashes: dict[str, str] = {}
        for source_id, expected in _SERVING_SOURCE_SHA256.items():
            serving_source_hashes[source_id] = _verify_hash(
                _source_path(source_id, self.research_root),
                expected,
                f"C0 serving source {source_id}",
            )
        training_source_hashes = dict(sorted(contract["source_sha256"].items()))
        replay_source_id = "project:src/analysis/ev_asof_replay.py"
        raw_feature_replay_source = {
            "implementation_path": str(_source_path(replay_source_id, self.research_root)),
            "implementation_sha256": serving_source_hashes[replay_source_id],
            "feature_archive_path": str(self.research_root / "unified-ratings-20260911"),
        }
        canonical_raw_history_source = {
            "path": str(_project_path(_CANONICAL_RAW_HISTORY_RELATIVE)),
            "sha256": _CANONICAL_RAW_HISTORY_SHA256,
            "source_kind": _CANONICAL_RAW_HISTORY_KIND,
        }
        self._contract = dict(contract)
        return {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "family": MODEL_FAMILY,
            "feature_version": FEATURE_VERSION,
            "has_uncertainty": False,
            "run_id": "a0_data_focus_20260929_192443",
            "variant": VARIANT,
            "artifact_path": str(model_root),
            "execution_sha256": execution_sha,
            "completion_sha256": completion_sha,
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "training_source_hashes": training_source_hashes,
            "training_source_provenance": "pinned_frozen_execution_contract",
            "serving_source_hashes": dict(sorted(serving_source_hashes.items())),
            "raw_feature_replay_source": raw_feature_replay_source,
            "canonical_raw_history_source": canonical_raw_history_source,
            "serving_config_path": str(self.config_path),
            "serving_config_sha256": self.config_sha256,
            "scope": "DEVELOPMENT_ONLY",
            "forward_status": self.config["forward_status"],
        }

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            self._identity = self._preflight()
        return json.loads(json.dumps(self._identity, allow_nan=False))

    def load(self) -> dict[str, Any]:
        if self._backend is not None:
            return self.identity
        identity = self.identity
        contract = self._contract
        if contract is None:
            raise RuntimeError("C0 artifact contract was not preflight verified")
        from src.models import a0_data_focus

        if Path(a0_data_focus.__file__).resolve() != (PROJECT_ROOT / "src/models/a0_data_focus.py").resolve():
            raise ImportError("C0 assembly did not resolve to the pinned local a0_data_focus module")
        backend, backend_source_hashes = a0_data_focus.uncertainty._activate_archive_backend(self.research_root)
        expected_backend_sources = {
            key: digest for key, digest in _SOURCE_SHA256.items()
            if key.startswith("research:nonlinear-history-20260912/")
            or key.startswith("research:a0-phase-walkforward-20260914/")
            or key in {"project:src/models/a0_uncertainty.py", "project:src/models/a0_accelerator.py"}
        }
        if backend_source_hashes != expected_backend_sources:
            raise ValueError("Loaded C0 native backend modules differ from the pinned inference source inventory")
        shared = a0_data_focus._load_checkpoint(
            self.project_model_root / "shared/model.joblib", {**contract, "component": "shared"},
        )
        models = [
            a0_data_focus._load_checkpoint(
                self.project_model_root / VARIANT / f"seed_{seed}/model.joblib",
                {**contract, "component": "neural", "variant": VARIANT, "seed": seed},
            )
            for seed in a0_data_focus.SEEDS
        ]
        calibration = a0_data_focus._load_checkpoint(
            self.project_model_root / VARIANT / "calibration/model.joblib",
            {**contract, "component": "calibration", "variant": VARIANT},
        )
        self._backend, self._shared, self._models, self._calibration = backend, shared, models, calibration
        self._identity = identity
        return self.identity

    def _load_history(self) -> tuple[Path, str, str, dict[str, Any]]:
        raw_path, source_kind, raw_is_canonical = _resolve_raw_history_source(
            self.config, os.environ.get("C0_RAW_HISTORY_PATH"),
        )
        release_override = os.environ.get("C0_RELEASE_MAP_PATH")
        release_relative = str(self.config["release_map_relative"])
        release_path = _project_path(release_override) if release_override else (self.research_root / release_relative).resolve()
        release_stat = release_path.stat()
        release_signature = (
            release_stat.st_dev, release_stat.st_ino, release_stat.st_size,
            release_stat.st_mtime_ns, release_stat.st_ctime_ns,
        )
        cached_release = self._release_map_cache
        if cached_release is not None and cached_release[0] == release_path and cached_release[1] == release_signature:
            release_sha, release_days = cached_release[2], cached_release[3]
        else:
            release_expected = "e54914ba79c5620a819a48a03bf877c373c6e53bd019219e43ffdd7db47c2620"
            release_days, release_sha = _read_pinned_json(
                release_path, release_expected, "frozen C0 release-day map",
            )
            contract = self._contract
            input_hashes = contract.get("input_sha256") if isinstance(contract, dict) else None
            if not isinstance(input_hashes, dict):
                raise ValueError("Frozen C0 execution contract omits its input inventory")
            release_hashes = {
                digest for source, digest in input_hashes.items()
                if isinstance(source, str) and source.endswith("/release_days.json")
            }
            if release_hashes != {release_sha}:
                raise ValueError("Release-day map hash differs from the frozen C0 input inventory")
            if not isinstance(release_days, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in release_days.items()
            ):
                raise ValueError("Frozen C0 release-day map must be an ID-to-date JSON object")
            release_after = release_path.stat()
            release_after_signature = (
                release_after.st_dev, release_after.st_ino, release_after.st_size,
                release_after.st_mtime_ns, release_after.st_ctime_ns,
            )
            if release_after_signature != release_signature or _sha256(release_path) != release_sha:
                raise ValueError("C0 release-day map changed while it was being loaded")
            self._release_map_cache = (release_path, release_signature, release_sha, release_days)
        fingerprint = hashlib.sha256(
            (str(raw_path) + "\\0" + str(release_path) + "\\0" + release_sha).encode("utf-8")
        ).hexdigest()
        raw_stat = raw_path.stat()
        stat_signature = (raw_stat.st_dev, raw_stat.st_ino, raw_stat.st_size, raw_stat.st_mtime_ns, raw_stat.st_ctime_ns)
        # Stat changes force a digest/replay refresh. A stable open-file signature is
        # checked again after parsing to reject a source replaced during preparation.
        prepared = self._prepared
        if prepared is not None and prepared["fingerprint"] == fingerprint and prepared["stat_signature"] == stat_signature:
            return raw_path, prepared["raw_sha256"], release_sha, prepared
        raw_sha = _sha256(raw_path)
        if raw_is_canonical and raw_sha != _CANONICAL_RAW_HISTORY_SHA256:
            raise ValueError(
                "Canonical C0 raw-history bytes differ from the immutable source pin: "
                f"expected {_CANONICAL_RAW_HISTORY_SHA256}, got {raw_sha}"
            )
        replay_path = (PROJECT_ROOT / "src/analysis/ev_asof_replay.py").resolve()
        _verify_hash(replay_path, _FEATURE_REPLAY_SHA256, "pinned native temporal feature builder")
        from src.analysis import ev_asof_replay as replay

        if Path(replay.__file__).resolve() != replay_path:
            raise ImportError("C0 temporal feature builder resolved outside the pinned project source")
        feature_backend, feature_hashes = replay._activate_feature_backend(self.research_root)
        series, history_metadata, extra, roster_events, history_audit, release_audit, _ = replay._stream_history(
            raw_path, release_days, feature_backend,
        )
        after = raw_path.stat()
        after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if after_signature != stat_signature or _sha256(raw_path) != raw_sha:
            raise ValueError("C0 raw history changed while its as-of state was being prepared")
        series.sort(key=lambda item: (item.games[-1].day, item.order))
        raw_history_max_source_game_day = max(
            (str(event["original_map_day"]) for event in roster_events), default=None,
        )
        raw_history_max_effective_release_day = max(
            (str(event["effective_release_day"]) for event in roster_events), default=None,
        )
        self._feature_replay = replay
        prepared = {
            "fingerprint": fingerprint,
            "stat_signature": stat_signature,
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "release_path": str(release_path),
            "release_sha256": release_sha,
            "release_days": release_days,
            "series": series,
            "history_metadata": history_metadata,
            "extra": extra,
            "roster_events": roster_events,
            "history_audit": history_audit,
            "release_audit": release_audit,
            "feature_backend": feature_backend,
            "source_kind": source_kind,
            "raw_history_max_source_game_day": raw_history_max_source_game_day,
            "raw_history_max_effective_release_day": raw_history_max_effective_release_day,
            "feature_hashes": feature_hashes,
            "state": None,
            "cursor": 0,
            "state_cutoff": None,
        }
        self._prepared = prepared
        return raw_path, raw_sha, release_sha, prepared

    @staticmethod
    def _aware_datetime(value: Any, label: str, *, optional: bool = False) -> datetime | None:
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a timezone-aware ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid {label}: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{label} must include an explicit timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _identifier(value: Any, label: str) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{label} must be an explicit nonempty ID")
        if isinstance(value, (int, np.integer)):
            result = str(int(value))
        elif isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)) or not float(value).is_integer():
                raise ValueError(f"{label} must be a finite integral ID")
            result = str(int(value))
        elif isinstance(value, str):
            result = value.strip()
        else:
            raise ValueError(f"{label} must be a scalar string or numeric ID")
        if not result or len(result) > 200:
            raise ValueError(f"{label} must be nonempty and at most 200 characters")
        return result

    def _validate_request(self, request: Any) -> dict[str, Any]:
        required = {
            "team1_id", "team2_id", "roster_a", "roster_b", "best_of",
            "decision_at", "competition_context", "start_at", "mode",
        }
        if not isinstance(request, Mapping) or set(request) != required:
            raise ValueError(f"c0_request must contain exactly {sorted(required)}")
        team_a = self._identifier(request["team1_id"], "team1_id")
        team_b = self._identifier(request["team2_id"], "team2_id")
        if team_a == team_b:
            raise ValueError("C0 opponents must have distinct team IDs")
        if type(request["best_of"]) is not int or request["best_of"] not in (1, 3, 5):
            raise ValueError("best_of must be exactly 1, 3, or 5")
        mode = request["mode"]
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        decision = self._aware_datetime(request["decision_at"], "decision_at")
        start = self._aware_datetime(request["start_at"], "start_at", optional=True)
        if decision.date() < _MINIMUM_DECISION_DATE:
            raise ValueError("Native C0 refuses pre-2026-01-01 inputs to prevent training leakage")
        if start is not None and start <= decision:
            raise ValueError("start_at must be strictly after decision_at")
        context = request["competition_context"]
        if context is not None and (not isinstance(context, str) or not context.strip() or len(context) > 300):
            raise ValueError("competition_context must be null or a nonempty exact context string")
        rosters: list[list[str]] = []
        for label, value in (("roster_a", request["roster_a"]), ("roster_b", request["roster_b"])):
            if not isinstance(value, (list, tuple)) or len(value) != 5:
                raise ValueError(f"{label} must contain exactly five explicit player IDs")
            roster = [self._identifier(item, f"{label} player") for item in value]
            if len(set(roster)) != 5:
                raise ValueError(f"{label} must contain five unique player IDs")
            rosters.append(sorted(roster))
        if set(rosters[0]) & set(rosters[1]):
            raise ValueError("C0 opposing rosters must not share player IDs")
        return {
            "team1_id": team_a, "team2_id": team_b,
            "roster_a": rosters[0], "roster_b": rosters[1],
            "best_of": request["best_of"], "decision_at": decision,
            "start_at": start, "competition_context": context.strip() if context else None,
            "mode": mode,
        }

    @staticmethod
    def _array_digest(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
        return {
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in sorted(arrays.items())
        }

    @staticmethod
    def _decode_array(value: Any, shape: tuple[int, ...], label: str, dtypes: set[str]) -> np.ndarray:
        if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data_base64"}:
            raise ValueError(f"{label} must use exact dtype/shape/base64 tensor serialization")
        dtype_name = value["dtype"]
        if dtype_name not in dtypes or dtype_name not in _WIRE_DTYPES:
            raise ValueError(f"Unsupported native C0 tensor dtype for {label}: {dtype_name!r}")
        if value["shape"] != list(shape):
            raise ValueError(f"{label} must have shape {shape}")
        encoded = value["data_base64"]
        if not isinstance(encoded, str):
            raise ValueError(f"{label}.data_base64 must be a string")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"{label}.data_base64 is not valid base64") from exc
        dtype = _WIRE_DTYPES[dtype_name]
        expected_size = int(np.prod(shape)) * dtype.itemsize
        if len(payload) != expected_size:
            raise ValueError(f"{label} tensor byte count does not match shape/dtype")
        array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{label} contains non-finite values")
        if label.endswith(".mask") and ((array != 0) & (array != 1)).any():
            raise ValueError("hist.mask must contain only binary values")
        return array

    def _decode_snapshot(self, snapshot: Any, request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not isinstance(snapshot, Mapping) or set(snapshot) != {"target", "raw", "hist"}:
            raise ValueError("c0_snapshot must contain exactly target, raw, and hist")
        target = snapshot["target"]
        if not isinstance(target, Mapping):
            raise ValueError("c0_snapshot.target must be an object")
        target_team_a = self._identifier(target.get("team1_id"), "snapshot team1_id")
        target_team_b = self._identifier(target.get("team2_id"), "snapshot team2_id")
        if (target_team_a, target_team_b) != (request["team1_id"], request["team2_id"]):
            raise ValueError("C0 snapshot team orientation differs from c0_request")
        if target.get("best_of") != request["best_of"] or isinstance(target.get("best_of"), bool):
            raise ValueError("C0 snapshot best_of differs from c0_request")
        actual_decision = self._aware_datetime(target.get("decision_at"), "snapshot decision_at", optional=True)
        effective_value = target.get("effective_decision_at")
        effective_decision = (
            self._aware_datetime(effective_value, "snapshot effective_decision_at")
            if effective_value is not None else actual_decision
        )
        target_decision = actual_decision if actual_decision is not None else effective_decision
        if target_decision != request["decision_at"] or effective_decision != request["decision_at"]:
            raise ValueError("C0 snapshot decision clocks differ from c0_request")
        expected_cutoff = request["decision_at"].date().isoformat()
        if target.get("cutoff_day") != expected_cutoff:
            raise ValueError("C0 snapshot cutoff_day differs from the requested UTC decision date")
        for field_name, expected in (("roster_a", request["roster_a"]), ("roster_b", request["roster_b"])):
            value = target.get(field_name)
            if (
                not isinstance(value, (list, tuple))
                or sorted(self._identifier(item, field_name) for item in value) != expected
            ):
                raise ValueError(f"C0 snapshot {field_name} differs from c0_request")
            side = "a" if field_name == "roster_a" else "b"
            for order_key in (
                f"tensor_player_order_{side}",
                f"raw_tensor_player_order_{side}",
                f"temporal_tensor_player_order_{side}",
            ):
                if order_key in target:
                    order = target[order_key]
                    if not isinstance(order, (list, tuple)) or [
                        self._identifier(item, f"snapshot {order_key} player") for item in order
                    ] != expected:
                        raise ValueError(f"C0 snapshot {order_key} differs from exact sorted player slots")
                else:
                    raise ValueError(f"C0 snapshot omits exact native player-slot order {order_key}")
        if target.get("competition_context", target.get("tournament_name")) != request["competition_context"]:
            raise ValueError("C0 snapshot competition context differs from c0_request")
        target_start = self._aware_datetime(target.get("start_at"), "snapshot start_at", optional=True)
        if target_start != request["start_at"]:
            raise ValueError("C0 snapshot start_at differs from c0_request")
        if "feature_source_max_day" not in target:
            raise ValueError("C0 snapshot must disclose its native feature-history boundary")
        feature_day = target.get("feature_source_max_day")
        if feature_day is not None:
            try:
                parsed_feature_day = date.fromisoformat(str(feature_day))
            except ValueError as exc:
                raise ValueError("C0 snapshot feature_source_max_day must be an ISO calendar date or null") from exc
            if parsed_feature_day >= request["decision_at"].date():
                raise ValueError("C0 snapshot includes feature history on or after the decision-day boundary")
        if target.get("feature_available") is not True:
            raise ValueError("C0 snapshot must explicitly declare available native inputs")
        raw_value, hist_value = snapshot["raw"], snapshot["hist"]
        if not isinstance(raw_value, Mapping) or set(raw_value) != set(_RAW_SHAPES):
            raise ValueError(f"snapshot.raw must contain exactly {sorted(_RAW_SHAPES)}")
        if not isinstance(hist_value, Mapping) or set(hist_value) != set(_TEMPORAL_SHAPES):
            raise ValueError(f"snapshot.hist must contain exactly {sorted(_TEMPORAL_SHAPES)}")
        raw = {
            key: self._decode_array(raw_value[key], shape, f"raw.{key}", {"float32"})
            for key, shape in _RAW_SHAPES.items()
        }
        hist = {
            "history": self._decode_array(hist_value["history"], _TEMPORAL_SHAPES["history"], "hist.history", {"float16"}),
            "mask": self._decode_array(hist_value["mask"], _TEMPORAL_SHAPES["mask"], "hist.mask", {"uint8"}),
            "champions": self._decode_array(hist_value["champions"], _TEMPORAL_SHAPES["champions"], "hist.champions", {"uint16"}),
        }
        return dict(target), raw, hist

    def _advance_history(self, cutoff: date, prepared: dict[str, Any]) -> Any:
        replay = self._feature_replay
        if replay is None:
            raise RuntimeError("C0 feature replay backend is not initialized")
        if prepared["state"] is None or (prepared["state_cutoff"] is not None and cutoff < prepared["state_cutoff"]):
            state = prepared["feature_backend"]["TemporalState"]()
            state.vocabulary = dict(prepared["feature_backend"]["champion_vocabulary"])
            prepared["state"] = state
            prepared["cursor"] = 0
            prepared["state_cutoff"] = None
        state = prepared["state"]
        cursor = prepared["cursor"]
        series = prepared["series"]
        while cursor < len(series) and series[cursor].games[-1].day < cutoff:
            item = series[cursor]
            state.consume(item, prepared["history_metadata"][item.identifier], prepared["extra"][item.identifier])
            cursor += 1
        prepared["cursor"] = cursor
        prepared["state_cutoff"] = cutoff
        last_day = state.base.base.last_day
        if last_day is not None and last_day >= cutoff:
            raise ValueError("C0 temporal state reached the requested decision-day boundary")
        return state

    def _raw_snapshot(self, request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        raw_path, raw_sha, release_sha, prepared = self._load_history()
        if "feature_backend" not in prepared:
            raise RuntimeError("Prepared C0 history has no pinned temporal backend")
        cutoff = request["decision_at"].date()
        state = self._advance_history(cutoff, prepared)
        target = {
            "team1_id": request["team1_id"], "team2_id": request["team2_id"],
            "best_of": request["best_of"], "decision_at": request["decision_at"].isoformat(),
            "effective_decision_at": request["decision_at"].isoformat(),
            "start_at": request["start_at"].isoformat() if request["start_at"] else None,
            "competition_context": request["competition_context"],
            "roster_a": request["roster_a"], "roster_b": request["roster_b"],
            "cutoff_day": cutoff.isoformat(),
            "feature_source_max_day": state.base.base.last_day.isoformat() if state.base.base.last_day else None,
            "feature_available": True,
        }
        raw, history, pair_proof = self._feature_replay._feature_values(
            state, target, (request["roster_a"], request["roster_b"]),
        )
        if pair_proof["cutoff_day"] != cutoff.isoformat():
            raise ValueError("Native C0 feature builder returned a different cutoff")
        if pair_proof["tensor_player_order_a"] != request["roster_a"] or pair_proof["tensor_player_order_b"] != request["roster_b"]:
            raise ValueError("Native C0 feature builder changed exact sorted player slot order")
        for side in ("a", "b"):
            for prefix in ("tensor", "raw_tensor", "temporal_tensor"):
                key = f"{prefix}_player_order_{side}"
                if pair_proof.get(key) != request[f"roster_{side}"]:
                    raise ValueError(f"Native C0 feature builder changed {key}")
                target[key] = pair_proof[key]
        arrays_raw = {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}
        arrays_hist = {
            "history": np.asarray(history["history"], dtype=np.float16),
            "mask": np.asarray(history["mask"], dtype=np.uint8),
            "champions": np.asarray(history["champions"], dtype=np.uint16),
        }
        if any(arrays_raw[key].shape != shape for key, shape in _RAW_SHAPES.items()):
            raise ValueError("Native C0 feature builder returned an invalid raw tensor shape")
        if any(arrays_hist[key].shape != shape for key, shape in _TEMPORAL_SHAPES.items()):
            raise ValueError("Native C0 feature builder returned an invalid temporal tensor shape")
        if any(not np.isfinite(value).all() for value in (*arrays_raw.values(), *arrays_hist.values())):
            raise ValueError("Native C0 feature builder returned non-finite tensor values")
        source_provenance = {
            "status": prepared["source_kind"],
            "source_kind": prepared["source_kind"],
            "dataset_freshness": {
                "raw_history_max_source_game_day": prepared["raw_history_max_source_game_day"],
                "raw_history_max_effective_release_day": prepared["raw_history_max_effective_release_day"],
                "feature_history_release_cutoff_day": target["feature_source_max_day"],
                "query_decision_day_exclusive_cutoff": target["cutoff_day"],
            },
            "raw_history_path": str(raw_path),
            "raw_history_sha256": raw_sha,
            "release_map_path": prepared["release_path"],
            "release_map_sha256": release_sha,
            "release_day_policy": prepared["release_audit"]["effective_release_precedence"],
            "source_game_day_fallback_event_count": prepared["release_audit"]["source_game_day_fallback_event_count"],
            "history_series_total": len(prepared["series"]),
            "history_series_consumed": prepared["cursor"],
            "feature_source_max_day": target["feature_source_max_day"],
            "tensor_player_order_a": pair_proof["tensor_player_order_a"],
            "tensor_player_order_b": pair_proof["tensor_player_order_b"],
            "raw_feature_replay_source": {
                **self.identity["raw_feature_replay_source"],
                "verified_feature_archive_sources": dict(sorted(prepared["feature_hashes"].items())),
                "verified_model_backend_sources": dict(sorted(
                    prepared["feature_backend"]["model_backend_hashes"].items()
                )),
            },
        }
        return target, arrays_raw, arrays_hist, source_provenance

    def _predict_arrays(
        self,
        request: Mapping[str, Any],
        target: Mapping[str, Any],
        raw: Mapping[str, np.ndarray],
        hist: Mapping[str, np.ndarray],
        provenance: Mapping[str, Any],
        canonical_match_id: int | None,
    ) -> dict[str, Any]:
        if self._backend is None:
            self.load()
        if not raw or not hist:
            raise ValueError("Native C0 requires real snapshot arrays; empty tensors are not a fallback")
        before = self._array_digest({**raw, **hist})
        raw_batch = {key: value[None, ...].copy() for key, value in raw.items()}
        hist_batch = {key: value[None, ...].copy() for key, value in hist.items()}
        from src.models import a0_data_focus

        predictions = a0_data_focus.predict_models(
            self._backend, self._shared, self._models, self._calibration,
            raw_batch, hist_batch, np.asarray([request["best_of"]], dtype=np.int64),
        )
        if self._array_digest({**raw, **hist}) != before:
            raise ValueError("C0 native inference mutated its input snapshot")
        full = np.asarray(predictions.get("full"), dtype=np.float64)
        if full.shape != (1, len(MODES)) or not np.isfinite(full).all() or ((full <= 0.) | (full >= 1.)).any():
            raise ValueError("Genuine C0 returned invalid all-mode series probabilities")
        mode_probabilities = {name: float(full[0, index]) for index, name in enumerate(MODES)}
        probability = mode_probabilities[request["mode"]]
        cutoff = datetime.combine(request["decision_at"].date(), time.min, tzinfo=timezone.utc).isoformat()
        diagnostics = {
            "c0_request": {
                key: (
                    value.isoformat() if isinstance(value, datetime)
                    else list(value) if isinstance(value, list)
                    else value
                )
                for key, value in request.items()
            },
            "mode": request["mode"],
            "mode_probabilities": mode_probabilities,
            "requested_decision_at": request["decision_at"].isoformat(),
            "data_cutoff_at": cutoff,
            "feature_source_max_day": target.get("feature_source_max_day"),
            "feature_provenance": dict(provenance),
            "input_hashes": before,
            "model_identity": self.identity,
            "information_status": provenance.get("status", "provided_native_snapshot"),
            "information_status_basis": "declared_input_provenance_not_authenticated",
            "probability_target": "series_win_already_projected",
            "uncertainty_available": False,
        }
        return {
            "canonical_match_id": canonical_match_id,
            "prob_a": probability,
            "prob_b": 1.0 - probability,
            "map_prob_a": None,
            "map_prob_b": None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "best_of": request["best_of"],
            "diagnostics": diagnostics,
        }

    def predict(
        self,
        request_value: Any,
        *,
        snapshot_value: Any = None,
        canonical_match_id: int | None = None,
    ) -> dict[str, Any]:
        request, target, raw, hist, provenance = self._prepare_prediction(request_value, snapshot_value)
        return self._predict_arrays(request, target, raw, hist, provenance, canonical_match_id)

    def _prepare_prediction(self, request_value: Any, snapshot_value: Any):
        self.load()
        request = self._validate_request(request_value)
        if snapshot_value is None:
            target, raw, hist, provenance = self._raw_snapshot(request)
        else:
            target, raw, hist = self._decode_snapshot(snapshot_value, request)
            provenance = {
                "status": "provided_native_snapshot",
                "feature_source_max_day": target.get("feature_source_max_day"),
                "snapshot_source": "caller_supplied_native_arrays",
                "raw_history_sha256": None,
                "release_map_sha256": None,
                "raw_feature_replay_source": None,
            }
        return request, target, raw, hist, provenance

    def predict_evidence(
        self, request: Any, *, snapshot: Any = None,
        canonical_match_id: int | None = None,
    ) -> dict[str, Any]:
        request, target, raw, hist, provenance = self._prepare_prediction(request, snapshot)
        prediction = self._predict_arrays(request, target, raw, hist, provenance, canonical_match_id)
        encoded = encode_snapshot({"target": target, "raw": raw, "hist": hist})
        return {
            "prediction": prediction,
            "encoded_snapshot": encoded,
            "native_history_provenance": dict(provenance),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def predict_identity(self) -> dict[str, Any]:
        return {"model_identity": self.load()}


def encode_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Encode NumPy native snapshot tensors with strict dtype and byte preservation."""
    if not isinstance(snapshot, Mapping) or set(snapshot) != {"target", "raw", "hist"}:
        raise ValueError("c0_snapshot must contain exactly target, raw, and hist")
    target = snapshot["target"]
    if not isinstance(target, Mapping):
        raise ValueError("c0_snapshot.target must be an object")
    target_fields = (
        "team1_id", "team2_id", "best_of", "decision_at", "effective_decision_at",
        "start_at", "competition_context", "tournament_name", "roster_a", "roster_b",
        "cutoff_day", "feature_source_max_day", "feature_available",
        "tensor_player_order_a", "tensor_player_order_b",
        "raw_tensor_player_order_a", "raw_tensor_player_order_b",
        "temporal_tensor_player_order_a", "temporal_tensor_player_order_b",
    )
    output: dict[str, Any] = {
        "target": {key: target[key] for key in target_fields if key in target},
        "raw": {},
        "hist": {},
    }
    for family, shapes, permitted in (
        ("raw", _RAW_SHAPES, {"float32"}),
        ("hist", _TEMPORAL_SHAPES, {"history": "float16", "mask": "uint8", "champions": "uint16"}),
    ):
        values = snapshot[family]
        if not isinstance(values, Mapping) or set(values) != set(shapes):
            raise ValueError(f"c0_snapshot.{family} must contain exactly {sorted(shapes)}")
        for key, shape in shapes.items():
            array = np.asarray(values[key])
            expected_dtype = permitted if isinstance(permitted, set) else {permitted[key]}
            if array.shape != shape or array.dtype.name not in expected_dtype:
                raise ValueError(f"c0_snapshot.{family}.{key} requires shape {shape} and dtype {sorted(expected_dtype)}")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"c0_snapshot.{family}.{key} contains non-finite values")
            output[family][key] = {
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "data_base64": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
            }
    return output


def snapshot_request(snapshot: Mapping[str, Any], mode: str = "full") -> dict[str, Any]:
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("target"), Mapping):
        raise ValueError("c0_snapshot.target must be an object")
    target = snapshot["target"]
    decision_at = target.get("decision_at", target.get("effective_decision_at"))
    if not isinstance(decision_at, str):
        raise ValueError("c0_snapshot target must include an aware decision_at")
    return {
        "team1_id": target.get("team1_id"),
        "team2_id": target.get("team2_id"),
        "roster_a": target.get("roster_a"),
        "roster_b": target.get("roster_b"),
        "best_of": target.get("best_of"),
        "decision_at": decision_at,
        "competition_context": target.get("competition_context", target.get("tournament_name")),
        "start_at": target.get("start_at"),
        "mode": mode,
    }
