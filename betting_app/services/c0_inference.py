"""Isolated-process client for genuine native C0 series probabilities."""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Mapping

import numpy as np

from betting_app.core.models.contract import UnifiedPredictionResult
from betting_app.core.models.native_c0 import (
    FEATURE_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    PROJECT_ROOT,
    encode_snapshot,
    load_serving_config,
    snapshot_request,
    NativeC0Adapter,
)


class C0InferenceError(RuntimeError):
    """Worker startup, protocol, or model errors that are not a built-in input error."""


class C0WorkerTimeout(C0InferenceError):
    """A bounded C0 worker startup or prediction deadline was exceeded."""


@dataclass(frozen=True)
class C0EvidenceResult:
    prediction: UnifiedPredictionResult
    encoded_snapshot: Mapping[str, Any]
    native_history_provenance: Mapping[str, Any]
    generated_at: datetime
    parent_received_at: datetime


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key in C0 worker response: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant in C0 worker response: {value}")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("C0 request contains a non-finite number")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("C0 request object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"C0 request contains unsupported JSON value {type(value).__name__}")


def _known_exception(kind: str, message: str) -> BaseException:
    exceptions: dict[str, type[Exception]] = {
        "ValueError": ValueError,
        "TypeError": TypeError,
        "FileNotFoundError": FileNotFoundError,
        "PermissionError": PermissionError,
        "ImportError": ImportError,
        "RuntimeError": RuntimeError,
        "OSError": OSError,
    }
    exception = exceptions.get(kind)
    if exception is None:
        return C0InferenceError(f"Native C0 worker {kind}: {message}")
    return exception(f"Native C0 worker {kind}: {message}")


def _validate_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise C0InferenceError("C0 worker omitted its runtime model identity")
    required = {
        "name", "version", "family", "feature_version", "has_uncertainty", "run_id", "variant",
        "artifact_path", "execution_sha256", "completion_sha256", "artifact_hashes",
        "training_source_hashes", "training_source_provenance", "serving_source_hashes",
        "raw_feature_replay_source", "canonical_raw_history_source",
        "serving_config_path", "serving_config_sha256", "scope", "forward_status",
    }
    if set(value) != required or (
        value["name"] != MODEL_NAME or value["version"] != MODEL_VERSION
        or value["family"] != "native_c0" or value["feature_version"] != FEATURE_VERSION
        or value["has_uncertainty"] is not False
        or value["run_id"] != "a0_data_focus_20260929_192443" or value["variant"] != "control"
        or value["scope"] != "DEVELOPMENT_ONLY"
        or value["forward_status"] != "frozen_static_2026_artifact_not_prospective_certified"
        or value["training_source_provenance"] != "pinned_frozen_execution_contract"
        or not isinstance(value["artifact_hashes"], Mapping)
        or not isinstance(value["training_source_hashes"], Mapping)
        or not isinstance(value["serving_source_hashes"], Mapping)
        or not isinstance(value["raw_feature_replay_source"], Mapping)
        or not isinstance(value["canonical_raw_history_source"], Mapping)
    ):
        raise C0InferenceError("C0 worker returned a runtime identity outside the pinned C0 contract")
    replay_source = value["raw_feature_replay_source"]
    if set(replay_source) != {"implementation_path", "implementation_sha256", "feature_archive_path"}:
        raise C0InferenceError("C0 worker omitted the raw feature replay source identity")
    if replay_source["implementation_sha256"] != value["serving_source_hashes"].get(
        "project:src/analysis/ev_asof_replay.py"
    ):
        raise C0InferenceError("C0 raw feature replay source differs from the verified serving-source pin")
    raw_source = value["canonical_raw_history_source"]
    if set(raw_source) != {"path", "sha256", "source_kind"} or raw_source["source_kind"] != "canonical_frozen_golgg_history":
        raise C0InferenceError("C0 worker omitted the canonical raw-history source identity")
    return json.loads(json.dumps(value, allow_nan=False))


class C0InferenceClient:
    """A bounded, durable local process client; no native archive enters this process."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        startup_timeout: float | None = None,
        request_timeout: float | None = None,
    ):
        config, path, _ = load_serving_config(config_path)
        self.config_path = path
        self.startup_timeout = float(config["startup_timeout_seconds"] if startup_timeout is None else startup_timeout)
        self.request_timeout = float(config["request_timeout_seconds"] if request_timeout is None else request_timeout)
        if not math.isfinite(self.startup_timeout) or not 1.0 <= self.startup_timeout <= 1800.0:
            raise ValueError("C0 startup timeout must be finite and in [1, 1800] seconds")
        if not math.isfinite(self.request_timeout) or not 1.0 <= self.request_timeout <= 900.0:
            raise ValueError("C0 request timeout must be finite and in [1, 900] seconds")
        self.max_worker_requests = int(config["max_worker_requests"])
        self.max_worker_lifetime = int(config["max_worker_lifetime_seconds"])
        self._process: subprocess.Popen[bytes] | None = None
        self._buffer = bytearray()
        self._lock = threading.RLock()
        self._request_id = 0
        self._worker_requests = 0
        self._started_at = 0.0
        self._identity: dict[str, Any] | None = None
        self._closed = False

    @property
    def identity(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_worker()
            assert self._identity is not None
            return json.loads(json.dumps(self._identity, allow_nan=False))

    def _readline(self, *, deadline: float) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise C0InferenceError("C0 worker stdout is unavailable")
        fd = process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise C0WorkerTimeout("C0 worker response deadline exceeded")
                newline = self._buffer.find(b"\n")
                if newline >= 0:
                    line = bytes(self._buffer[:newline])
                    del self._buffer[:newline + 1]
                    return line
                if len(self._buffer) > 16 * 1024 * 1024:
                    raise C0InferenceError("C0 worker response exceeded the protocol frame bound")
                if not selector.select(remaining):
                    raise C0WorkerTimeout("C0 worker response deadline exceeded")
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise C0InferenceError("C0 worker exited before completing its response")
                self._buffer.extend(chunk)

    @staticmethod
    def _decode_line(line: bytes) -> dict[str, Any]:
        try:
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_json_pairs, parse_constant=_reject_constant)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise C0InferenceError("C0 worker returned invalid strict JSON") from exc
        if not isinstance(value, dict):
            raise C0InferenceError("C0 worker response frame is not a JSON object")
        return value

    def _ensure_worker(self) -> None:
        if self._closed:
            raise C0InferenceError("C0 inference client is closed")
        process = self._process
        expired = process is not None and (
            self._worker_requests >= self.max_worker_requests
            or time.monotonic() - self._started_at >= self.max_worker_lifetime
        )
        if process is not None and process.poll() is None and not expired:
            return
        if process is not None:
            self._stop_worker(graceful=not expired)
        env = os.environ.copy()
        env["BETTING_APP_C0_WORKER"] = "1"
        env["C0_SERVING_CONFIG"] = str(self.config_path)
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) if not python_path else str(PROJECT_ROOT) + os.pathsep + python_path
        command = [sys.executable, "-m", "betting_app.services.c0_worker", "--config", str(self.config_path)]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
            self._buffer.clear()
            self._worker_requests = 0
            self._identity = None
            ready = self._decode_line(self._readline(deadline=time.monotonic() + self.startup_timeout))
            if ready.get("type") == "startup_error":
                error = ready.get("error", {})
                raise _known_exception(str(error.get("type", "RuntimeError")), str(error.get("message", "startup failed")))
            if ready.get("type") != "ready" or ready.get("ok") is not True:
                raise C0InferenceError("C0 worker returned an invalid startup handshake")
            self._identity = _validate_identity(ready.get("model_identity"))
            self._started_at = time.monotonic()
        except BaseException:
            self._stop_worker(graceful=False)
            raise

    def _write_frame(self, frame: dict[str, Any], *, deadline: float) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise C0InferenceError("C0 worker stdin is unavailable")
        payload = json.dumps(_json_value(frame), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("C0 request exceeds the worker protocol frame bound")
        fd = process.stdin.fileno()
        os.set_blocking(fd, False)
        with memoryview(payload) as view, selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            offset = 0
            while offset < len(view):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise C0WorkerTimeout("C0 worker request write deadline exceeded")
                if not selector.select(remaining):
                    raise C0WorkerTimeout("C0 worker request write deadline exceeded")
                try:
                    written = os.write(fd, view[offset:])
                except BlockingIOError:
                    continue
                if written == 0:
                    raise C0InferenceError("C0 worker stdin closed before completing its request")
                offset += written

    def _exchange(self, frame: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self._ensure_worker()
        deadline = time.monotonic() + timeout
        self._request_id += 1
        request_id = self._request_id
        framed = {**frame, "id": request_id}
        try:
            self._write_frame(framed, deadline=deadline)
            line = self._readline(deadline=deadline)
            received_at = datetime.now(timezone.utc)
            reply = self._decode_line(line)
        except (BrokenPipeError, OSError, C0InferenceError):
            self._stop_worker(graceful=False)
            raise
        if reply.get("id") != request_id or type(reply.get("ok")) is not bool:
            self._stop_worker(graceful=False)
            raise C0InferenceError("C0 worker response identity or status is invalid")
        if reply["ok"] is not True:
            error = reply.get("error")
            if not isinstance(error, Mapping) or set(error) != {"type", "message"}:
                self._stop_worker(graceful=False)
                raise C0InferenceError("C0 worker error response is malformed")
            raise _known_exception(str(error["type"]), str(error["message"]))
        result = reply.get("result")
        if not isinstance(result, dict):
            self._stop_worker(graceful=False)
            raise C0InferenceError("C0 worker response omitted its result object")
        self._last_received_at = received_at
        if frame.get("op") in {"predict", "predict_evidence"}:
            self._worker_requests += 1
        return result

    def predict_request(
        self,
        request: Mapping[str, Any],
        *,
        snapshot: Mapping[str, Any] | None = None,
        canonical_match_id: int | None = None,
    ) -> UnifiedPredictionResult:
        if canonical_match_id is not None and (type(canonical_match_id) is not int or canonical_match_id < 0):
            raise ValueError("canonical_match_id must be a nonnegative integer or null")
        if not isinstance(request, Mapping):
            raise TypeError("c0_request must be a mapping")
        with self._lock:
            frame: dict[str, Any] = {
                "op": "predict",
                "c0_request": _json_value(request),
                "canonical_match_id": canonical_match_id,
            }
            if snapshot is not None:
                frame["c0_snapshot"] = encode_snapshot(snapshot)
            result = self._exchange(frame, timeout=self.request_timeout)
        return _prediction_result(result)

    def predict_evidence(
        self, request: Mapping[str, Any], *,
        snapshot: Mapping[str, Any] | None = None,
        canonical_match_id: int | None = None,
    ) -> C0EvidenceResult:
        if canonical_match_id is not None and (type(canonical_match_id) is not int or canonical_match_id < 0):
            raise ValueError("canonical_match_id must be a nonnegative integer or null")
        if not isinstance(request, Mapping):
            raise TypeError("c0_request must be a mapping")
        with self._lock:
            frame = {
                "op": "predict_evidence", "c0_request": _json_value(request),
                "canonical_match_id": canonical_match_id,
            }
            if snapshot is not None:
                frame["c0_snapshot"] = encode_snapshot(snapshot)
            result = self._exchange(frame, timeout=self.request_timeout)
            received_at = self._last_received_at
        if set(result) != {"prediction", "encoded_snapshot", "native_history_provenance", "generated_at"}:
            raise C0InferenceError("Native C0 evidence wire schema changed")
        prediction = _prediction_result(result["prediction"])
        if result["prediction"]["canonical_match_id"] != canonical_match_id:
            raise C0InferenceError("Native C0 evidence fixture identity changed")
        try:
            generated_at = datetime.fromisoformat(result["generated_at"])
            if generated_at.tzinfo is None or generated_at.utcoffset() is None:
                raise ValueError("naive completion")
            generated_at = generated_at.astimezone(timezone.utc)
            if generated_at > received_at:
                raise ValueError("completion follows receipt")
            validator = object.__new__(NativeC0Adapter)
            validated_request = validator._validate_request(frame["c0_request"])
            diagnostics = result["prediction"]["diagnostics"]
            if (
                diagnostics.get("mode") != validated_request["mode"]
                or diagnostics.get("requested_decision_at") != validated_request["decision_at"].isoformat()
                or result["prediction"]["best_of"] != validated_request["best_of"]
            ):
                raise ValueError("evidence prediction differs from requested mode, clock or format")
            _, raw, hist = validator._decode_snapshot(result["encoded_snapshot"], validated_request)
            hashes = validator._array_digest({**raw, **hist})
            if hashes != result["prediction"]["diagnostics"].get("input_hashes"):
                raise ValueError("snapshot hashes differ from predicted arrays")
            provenance = result["native_history_provenance"]
            if not isinstance(provenance, Mapping) or provenance != result["prediction"]["diagnostics"].get("feature_provenance"):
                raise ValueError("native history provenance differs from prediction")
        except (TypeError, ValueError, KeyError) as exc:
            raise C0InferenceError(f"Invalid native C0 evidence: {exc}") from exc
        return C0EvidenceResult(
            prediction, result["encoded_snapshot"], provenance, generated_at, received_at,
        )

    def predict_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        mode: str = "full",
        canonical_match_id: int | None = None,
    ) -> UnifiedPredictionResult:
        if not isinstance(snapshot, Mapping):
            raise TypeError("c0_snapshot must be a mapping")
        return self.predict_request(
            snapshot_request(snapshot, mode=mode), snapshot=snapshot,
            canonical_match_id=canonical_match_id,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop_worker(graceful=True)

    def _stop_worker(self, *, graceful: bool) -> None:
        process = self._process
        if process is None:
            return
        self._identity = None
        if graceful and process.poll() is None:
            try:
                deadline = time.monotonic() + 1.0
                self._request_id += 1
                self._write_frame({"op": "shutdown", "id": self._request_id}, deadline=deadline)
                reply = self._decode_line(self._readline(deadline=deadline))
                if reply.get("id") == self._request_id and reply.get("ok") is True:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except (BrokenPipeError, OSError, C0InferenceError, subprocess.TimeoutExpired):
                pass
        self._process = None
        self._buffer.clear()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2.0)
        for handle in (process.stdin, process.stdout):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass


def _prediction_result(value: Mapping[str, Any]) -> UnifiedPredictionResult:
    required = {
        "canonical_match_id", "prob_a", "prob_b", "map_prob_a", "map_prob_b",
        "model_name", "model_version", "feature_version", "best_of", "diagnostics",
    }
    if set(value) != required:
        raise C0InferenceError("Native C0 worker returned an unexpected prediction schema")
    if value["map_prob_a"] is not None or value["map_prob_b"] is not None:
        raise C0InferenceError("Native C0 is already a series probability and must not return map outputs")
    try:
        prob_a, prob_b = float(value["prob_a"]), float(value["prob_b"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise C0InferenceError("Native C0 worker returned nonnumeric probabilities") from exc
    if not math.isfinite(prob_a) or not math.isfinite(prob_b) or not 0.0 < prob_a < 1.0 or not 0.0 < prob_b < 1.0:
        raise C0InferenceError("Native C0 worker returned invalid finite interior probabilities")
    if not math.isclose(prob_a + prob_b, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise C0InferenceError("Native C0 worker side probabilities do not sum to one")
    if value["model_name"] != MODEL_NAME or value["model_version"] != MODEL_VERSION or value["feature_version"] != FEATURE_VERSION:
        raise C0InferenceError("Native C0 worker prediction identity changed")
    if type(value["best_of"]) is not int or value["best_of"] not in (1, 3, 5):
        raise C0InferenceError("Native C0 worker returned an invalid best_of")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise C0InferenceError("Native C0 worker omitted prediction diagnostics")
    _validate_identity(diagnostics.get("model_identity"))
    if diagnostics.get("uncertainty_available") is not False:
        raise C0InferenceError("Native C0 worker must not report invented uncertainty")
    if diagnostics.get("probability_target") != "series_win_already_projected":
        raise C0InferenceError("Native C0 worker prediction target is not a direct series probability")
    canonical_match_id = value["canonical_match_id"]
    if canonical_match_id is not None and (type(canonical_match_id) is not int or canonical_match_id < 0):
        raise C0InferenceError("Native C0 worker returned an invalid canonical match ID")
    return UnifiedPredictionResult(
        canonical_match_id=canonical_match_id,
        prob_a=prob_a,
        prob_b=prob_b,
        map_prob_a=None,
        map_prob_b=None,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        best_of=value["best_of"],
        diagnostics=diagnostics,
    )


_DEFAULT_CLIENT: C0InferenceClient | None = None
_DEFAULT_CLIENT_LOCK = threading.Lock()


def _default_client() -> C0InferenceClient:
    global _DEFAULT_CLIENT
    configured = os.environ.get("C0_SERVING_CONFIG")
    expected_path = Path(configured).expanduser() if configured else PROJECT_ROOT / "conf/base/c0_serving.json"
    if not expected_path.is_absolute():
        expected_path = PROJECT_ROOT / expected_path
    expected_path = expected_path.resolve()
    with _DEFAULT_CLIENT_LOCK:
        if _DEFAULT_CLIENT is None or _DEFAULT_CLIENT.config_path != expected_path:
            if _DEFAULT_CLIENT is not None:
                _DEFAULT_CLIENT.close()
            _DEFAULT_CLIENT = C0InferenceClient(expected_path)
        return _DEFAULT_CLIENT


def predict_c0(features: dict[str, Any]) -> UnifiedPredictionResult:
    """Predict one C0 series from explicit request metadata or a native tensor snapshot."""
    if not isinstance(features, dict):
        raise TypeError("native C0 features must be a dictionary")
    request = features.get("c0_request")
    snapshot = features.get("c0_snapshot")
    if request is None:
        if snapshot is None:
            raise ValueError("native C0 requires c0_request or an explicit c0_snapshot")
        request = snapshot_request(snapshot)
    if not isinstance(request, Mapping):
        raise TypeError("features.c0_request must be a mapping")
    canonical = features.get("canonical")
    canonical_id = canonical.get("id") if isinstance(canonical, Mapping) else None
    if isinstance(canonical_id, np.generic):
        canonical_id = canonical_id.item()
    if canonical_id is not None and (type(canonical_id) is not int or canonical_id < 0):
        raise ValueError("features.canonical.id must be a nonnegative integer or null")
    return _default_client().predict_request(request, snapshot=snapshot, canonical_match_id=canonical_id)


def predict_c0_snapshot(
    snapshot: Mapping[str, Any],
    *,
    mode: str = "full",
    canonical_match_id: int | None = None,
) -> UnifiedPredictionResult:
    """Public direct-snapshot parity entry point; arrays retain exact native dtype/bytes."""
    return _default_client().predict_snapshot(
        snapshot, mode=mode, canonical_match_id=canonical_match_id,
    )


def get_c0_artifact_identity() -> dict[str, Any]:
    """Return the verified runtime artifact identity from the isolated model process."""
    return _default_client().identity


def close_c0_inference_worker() -> None:
    """Explicitly stop the process-global local worker; safe to call repeatedly."""
    global _DEFAULT_CLIENT
    with _DEFAULT_CLIENT_LOCK:
        if _DEFAULT_CLIENT is not None:
            _DEFAULT_CLIENT.close()
            _DEFAULT_CLIENT = None


def _cleanup_default_client() -> None:
    try:
        close_c0_inference_worker()
    except Exception:
        pass


atexit.register(_cleanup_default_client)
