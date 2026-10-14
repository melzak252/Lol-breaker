"""Native C0 transport deadlines using owned, model-free subprocesses."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import selectors
import subprocess
import sys
import threading
import time

import pytest

from betting_app.services.c0_inference import C0InferenceClient, C0InferenceError, C0WorkerTimeout


_WORKER_PREAMBLE = """
import hashlib
import json
import os
import sys
import time

control_fd = int(sys.argv[1])
event_fd = int(sys.argv[2])
os.write(event_fd, b"R")
"""


def _wait_event(fd: int, expected: bytes) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        assert selector.select(5.0), "Owned transport worker did not reach its synchronization point"
        assert os.read(fd, 1) == expected


@contextmanager
def _transport(worker_body: str):
    control_read, control_write = os.pipe()
    event_read, event_write = os.pipe()
    process = None
    client = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _WORKER_PREAMBLE + worker_body, str(control_read), str(event_write)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=(control_read, event_write),
            start_new_session=True,
            bufsize=0,
        )
        # Install an already-ready real transport without reading configuration or loading a model.
        client = C0InferenceClient.__new__(C0InferenceClient)
        client._process = process
        client._buffer = bytearray()
        client._lock = threading.RLock()
        client._request_id = 0
        client._worker_requests = 0
        client._started_at = time.monotonic()
        client._identity = None
        client._closed = False
        client.max_worker_requests = 100
        client.max_worker_lifetime = 3600
        _wait_event(event_read, b"R")
        yield client, process, control_write, event_read
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.0)
        if client is not None:
            client._stop_worker(graceful=False)
        for fd in (control_read, control_write, event_read, event_write):
            os.close(fd)


@contextmanager
def _cleanup_watchdog(process: subprocess.Popen, timeout: float = 5.0):
    fired = threading.Event()

    def kill_owned_child() -> None:
        fired.set()
        try:
            process.kill()
        except ProcessLookupError:
            pass

    watchdog = threading.Timer(timeout, kill_owned_child)
    watchdog.daemon = True
    watchdog.start()
    try:
        yield fired
    finally:
        watchdog.cancel()
        watchdog.join(timeout=5.0)
        assert not watchdog.is_alive(), "Owned cleanup watchdog failed to stop"


@contextmanager
def _release_after_prefix(event_fd: int, control_fd: int, *, delay: float = 0.0):
    cancelled = threading.Event()
    failures = []

    def release() -> None:
        try:
            _wait_event(event_fd, b"P")
            if not cancelled.wait(delay):
                os.write(control_fd, b"G")
        except BaseException as exc:
            failures.append(exc)

    helper = threading.Thread(target=release, daemon=True)
    helper.start()
    try:
        yield
    finally:
        cancelled.set()
        helper.join(timeout=6.0)
        assert not helper.is_alive(), "Owned synchronization helper failed to stop"
        assert not failures, failures


def _fill_stdin_pipe(process: subprocess.Popen) -> None:
    assert process.stdin is not None
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    written = 0
    try:
        while True:
            written += os.write(fd, b"x" * 4096)
    except BlockingIOError:
        assert written > 0
    finally:
        os.set_blocking(fd, True)


def _assert_reaped(client: C0InferenceClient, process: subprocess.Popen) -> None:
    assert client._process is None
    assert process.poll() is not None
    assert process.stdin is not None and process.stdin.closed
    assert process.stdout is not None and process.stdout.closed


def test_full_pipe_request_write_times_out_and_reaps_worker() -> None:
    with _transport("os.read(control_fd, 1)\n") as (client, process, _, _):
        _fill_stdin_pipe(process)
        with _cleanup_watchdog(process, timeout=3.0) as watchdog_fired:
            started = time.monotonic()
            with pytest.raises(C0WorkerTimeout):
                client._exchange({"op": "predict", "payload": "blocked"}, timeout=0.3)
            assert time.monotonic() - started < 2.0
            assert not watchdog_fired.is_set()
            _assert_reaped(client, process)


def test_partial_request_writes_and_fragmented_response_preserve_exact_payload() -> None:
    worker = """
prefix = os.read(0, 1024)
os.write(event_fd, b"P")
assert os.read(control_fd, 1) == b"G"
request = json.loads(prefix + sys.stdin.buffer.readline())
payload = request["payload"]
reply = {
    "id": request["id"], "ok": True,
    "result": {"characters": len(payload), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()},
}
encoded = json.dumps(reply).encode("utf-8") + b"\\n"
for offset in range(0, len(encoded), 7):
    os.write(1, encoded[offset:offset + 7])
    time.sleep(0.005)
os.read(control_fd, 1)
"""
    payload = "zażółć" * 100_000
    with _transport(worker) as (client, process, control_fd, event_fd):
        with _cleanup_watchdog(process), _release_after_prefix(event_fd, control_fd):
            result = client._exchange({"op": "predict", "payload": payload}, timeout=4.0)
        assert result == {
            "characters": len(payload),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        assert client._worker_requests == 1
        assert process.poll() is None


def test_request_write_and_response_read_share_one_deadline() -> None:
    worker = """
prefix = os.read(0, 1024)
os.write(event_fd, b"P")
assert os.read(control_fd, 1) == b"G"
request = json.loads(prefix + sys.stdin.buffer.readline())
time.sleep(1.0)
os.write(1, json.dumps({"id": request["id"], "ok": True, "result": {}}).encode("utf-8") + b"\\n")
os.read(control_fd, 1)
"""
    with _transport(worker) as (client, process, control_fd, event_fd):
        with _cleanup_watchdog(process), _release_after_prefix(event_fd, control_fd, delay=1.0):
            started = time.monotonic()
            with pytest.raises(C0WorkerTimeout):
                client._exchange({"op": "predict", "payload": "x" * 1_000_000}, timeout=1.5)
            assert time.monotonic() - started < 2.5
            _assert_reaped(client, process)


def test_request_serialization_consumes_the_request_deadline() -> None:
    class SlowRequest(dict):
        def items(self):
            time.sleep(0.6)
            return super().items()

    worker = """
request = json.loads(sys.stdin.buffer.readline())
os.write(1, json.dumps({"id": request["id"], "ok": True, "result": {}}).encode("utf-8") + b"\\n")
os.read(control_fd, 1)
"""
    with _transport(worker) as (client, process, _, _):
        with _cleanup_watchdog(process):
            with pytest.raises(C0WorkerTimeout):
                client._exchange({"op": "predict", "payload": SlowRequest(value="serialized")}, timeout=0.2)
            _assert_reaped(client, process)


def test_graceful_shutdown_is_bounded_when_worker_stdin_is_full() -> None:
    with _transport("os.read(control_fd, 1)\n") as (client, process, _, _):
        _fill_stdin_pipe(process)
        with _cleanup_watchdog(process, timeout=4.0) as watchdog_fired:
            started = time.monotonic()
            client.close()
            assert time.monotonic() - started < 2.5
            assert not watchdog_fired.is_set()
            assert client._closed is True
            _assert_reaped(client, process)


@pytest.mark.parametrize(
    "response",
    [
        b'{"id":1,"ok":true,"result":{},"ok":true}\n',
        b'{"id":1,"ok":true,"result":{"value":NaN}}\n',
        b'{"id":1,"ok":true,"result":{"value":"\xff"}}\n',
    ],
)
def test_fragmented_response_preserves_strict_json_rejection(response: bytes) -> None:
    worker = f"""
sys.stdin.buffer.readline()
response = {response!r}
for offset in range(0, len(response), 5):
    os.write(1, response[offset:offset + 5])
    time.sleep(0.005)
os.read(control_fd, 1)
"""
    with _transport(worker) as (client, process, _, _):
        with _cleanup_watchdog(process):
            with pytest.raises(C0InferenceError, match="invalid strict JSON"):
                client._exchange({"op": "predict"}, timeout=2.0)
            _assert_reaped(client, process)


def test_unterminated_response_preserves_protocol_frame_bound() -> None:
    worker = """
sys.stdin.buffer.readline()
sys.stdout.buffer.write(b"x" * (16 * 1024 * 1024 + 1))
sys.stdout.buffer.flush()
os.read(control_fd, 1)
"""
    with _transport(worker) as (client, process, _, _):
        with _cleanup_watchdog(process):
            with pytest.raises(C0InferenceError, match="protocol frame bound"):
                client._exchange({"op": "predict"}, timeout=3.0)
            _assert_reaped(client, process)
