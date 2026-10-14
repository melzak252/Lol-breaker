"""Private JSON-lines process for native C0; model archives never enter the API worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

MAX_FRAME_BYTES = 16 * 1024 * 1024


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _emit(stream: Any, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    stream.write(encoded)
    stream.flush()


def _response_error(request_id: Any, error: BaseException) -> dict[str, Any]:
    return {
        "id": request_id,
        "ok": False,
        "error": {"type": type(error).__name__, "message": str(error)[:4000]},
    }


def _parse_request(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_FRAME_BYTES:
        raise ValueError(f"C0 request exceeds {MAX_FRAME_BYTES} bytes")
    try:
        value = json.loads(line.decode("utf-8"), object_pairs_hook=_json_pairs, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("C0 request is not valid strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("C0 request frame must be a JSON object")
    if type(value.get("id")) is not int or value["id"] < 1:
        raise ValueError("C0 request id must be a positive integer")
    if value.get("op") not in {"predict", "predict_evidence", "identity", "shutdown"}:
        raise ValueError("Unsupported C0 worker operation")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args(argv)

    # Keep stdout an exclusive protocol channel even if an imported native module prints.
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    os.environ["BETTING_APP_C0_WORKER"] = "1"
    from betting_app.core.models.native_c0 import NativeC0Adapter, load_serving_config

    try:
        config, _, _ = load_serving_config(arguments.config)
        adapter = NativeC0Adapter(arguments.config)
        identity = adapter.load()
    except BaseException as error:
        _emit(protocol_output, {"type": "startup_error", "ok": False, "error": {
            "type": type(error).__name__, "message": str(error)[:4000],
        }})
        return 2

    _emit(protocol_output, {"type": "ready", "ok": True, "model_identity": identity})
    started = time.monotonic()
    requests = 0
    while True:
        line = sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1)
        if len(line) > MAX_FRAME_BYTES:
            _emit(protocol_output, _response_error(None, ValueError(f"C0 request exceeds {MAX_FRAME_BYTES} bytes")))
            return 1
        if not line:
            return 0
        request_id: Any = None
        try:
            request = _parse_request(line)
            request_id = request["id"]
            operation = request["op"]
            if operation == "shutdown":
                if set(request) != {"id", "op"}:
                    raise ValueError("shutdown frame contains unexpected fields")
                _emit(protocol_output, {"id": request_id, "ok": True, "result": {"closed": True}})
                return 0
            if time.monotonic() - started >= config["max_worker_lifetime_seconds"]:
                raise RuntimeError("C0 worker lifetime bound reached; restart required")
            if requests >= config["max_worker_requests"]:
                raise RuntimeError("C0 worker request bound reached; restart required")
            if operation == "identity":
                if set(request) != {"id", "op"}:
                    raise ValueError("identity frame contains unexpected fields")
                result = {"model_identity": identity}
            else:
                expected = {"id", "op", "c0_request", "c0_snapshot", "canonical_match_id"}
                if set(request) - expected or not {"id", "op", "c0_request", "canonical_match_id"}.issubset(request):
                    raise ValueError("predict frame has missing or unexpected fields")
                if operation == "predict_evidence":
                    result = adapter.predict_evidence(
                        request["c0_request"], snapshot=request.get("c0_snapshot"),
                        canonical_match_id=request["canonical_match_id"],
                    )
                else:
                    result = adapter.predict(
                        request["c0_request"], snapshot_value=request.get("c0_snapshot"),
                        canonical_match_id=request["canonical_match_id"],
                    )
            _emit(protocol_output, {"id": request_id, "ok": True, "result": result})
            requests += 1
        except BaseException as error:
            _emit(protocol_output, _response_error(request_id, error))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
