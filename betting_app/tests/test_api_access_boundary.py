"""Browser-origin and validation-response boundaries on the real API.

Remote authentication belongs to Caddy, not CORS. Proxy authentication and
port isolation require the separate real loopback Caddy smoke. These tests
exercise production middleware without database or task execution. Nonfinite
invalid inputs must remain descriptive strings in strict-JSON 422 responses.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json

import pytest
from fastapi.testclient import TestClient
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from betting_app.api.deps import get_db
from betting_app.api.main import app, request_validation_error_handler


@pytest.fixture
def boundary_client():
    # The API lifespan does not initialize storage or start the scheduler.
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:3000", "http://localhost:3000"],
    ids=["ipv4-loopback", "localhost"],
)
def test_default_local_origin_preflight_allows_wallet_write(boundary_client, origin):
    response = boundary_client.options(
        "/wallets",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "origin" in response.headers["vary"].lower()


@pytest.mark.parametrize(
    "origin", ["https://foreign.invalid", "http://localhost:3000.foreign.invalid", "null"],
    ids=["foreign", "lookalike", "opaque"],
)
@pytest.mark.parametrize(
    "path, method",
    [("/wallets", "POST"), ("/scheduler/trigger/prediction_pipeline", "POST"),
     ("/matches/1/props/analysis", "GET")],
    ids=["wallet-write", "task-trigger", "state-changing-get"],
)
def test_foreign_origin_preflight_is_rejected(boundary_client, origin, path, method):
    # OPTIONS is intercepted by CORSMiddleware: none of these work routes runs.
    response = boundary_client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:3000", "http://localhost:3000"],
    ids=["ipv4-loopback", "localhost"],
)
def test_default_local_origin_response_is_explicit_not_wildcard(boundary_client, origin):
    response = boundary_client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "origin" in response.headers["vary"].lower()


@pytest.mark.parametrize("cookie", [False, True], ids=["without-cookie", "with-cookie"])
def test_foreign_origin_cannot_read_actual_health_response(boundary_client, cookie):
    headers = {"Origin": "https://foreign.invalid"}
    if cookie:
        headers["Cookie"] = "boundary_probe=disposable"
    response = boundary_client.get("/health", headers=headers)

    # Explicit foreign origins are rejected before handlers; absent Origin still
    # relies on the separate proxy authentication and loopback port boundary.
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.fixture
def validation_client(boundary_client, monkeypatch):
    class NoDatabaseExecution:
        def execute(self, *args, **kwargs):
            pytest.fail("A schema-invalid request must not execute SQL")

    monkeypatch.setitem(app.dependency_overrides, get_db, NoDatabaseExecution)
    return boundary_client


@pytest.mark.parametrize(
    "path,kwargs,location,error_type,invalid_input",
    [
        pytest.param(
            "/bets", {"content": '{"stake":', "headers": {"Content-Type": "application/json"}},
            ["body", 9], "json_invalid", {}, id="malformed-json",
        ),
        pytest.param(
            "/bets", {"json": {}},
            ["body", "stake"], "missing", {}, id="missing-field",
        ),
        pytest.param(
            "/bets", {"json": {"bookmaker_account_id": 1, "side": "a", "stake": 0, "odds": 2}},
            ["body", "stake"], "greater_than", 0, id="finite-out-of-range",
        ),
        pytest.param(
            "/wallets", {"params": {"bookmaker_id": "not-int"}},
            ["query", "bookmaker_id"], "int_parsing", "not-int", id="invalid-query-parameter",
        ),
    ],
)
def test_ordinary_validation_errors_keep_detail_contract(
    validation_client, path, kwargs, location, error_type, invalid_input
):
    request_kwargs = dict(kwargs)
    headers = {**request_kwargs.pop("headers", {}), "Origin": "http://localhost:3000"}
    response = validation_client.post(path, headers=headers, **request_kwargs)

    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    error = next(error for error in errors if error["loc"] == location)
    assert error["type"] == error_type
    assert error["input"] == invalid_input
    assert isinstance(error["msg"], str) and error["msg"]
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_validation_error_encoding_sanitizes_nested_inputs_and_context_only():
    context_error = ValueError("invalid nested numbers")
    context_error.values = [float("-inf"), Decimal("NaN")]
    errors = [{
        "type": "value_error",
        "loc": ("body", "stake"),
        "msg": "Value error, invalid nested numbers",
        "input": {"numbers": [float("nan"), float("inf"), Decimal("-Infinity"), Decimal("1.25")]},
        "ctx": {"error": context_error, "limit": Decimal("Infinity"), "finite_limit": Decimal("2")},
    }]
    response = asyncio.run(request_validation_error_handler(
        Request({"type": "http"}), RequestValidationError(errors),
    ))

    assert response.status_code == 422
    detail = json.loads(
        response.body,
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"),
    )["detail"]
    assert detail == [{
        "type": "value_error",
        "loc": ["body", "stake"],
        "msg": "Value error, invalid nested numbers",
        "input": {"numbers": ["NaN", "Infinity", "-Infinity", 1.25]},
        "ctx": {"error": {"values": ["-Infinity", "NaN"]}, "limit": "Infinity", "finite_limit": 2},
    }]


def test_finite_validation_error_encoding_matches_fastapi_default():
    exception = RequestValidationError([{
        "type": "greater_than",
        "loc": ("body", "stake"),
        "msg": "Input should be greater than 0",
        "input": Decimal("-1.25"),
        "ctx": {"gt": Decimal("0"), "error": ValueError("ordinary invalid input")},
    }])
    request = Request({"type": "http"})
    expected = asyncio.run(request_validation_exception_handler(request, exception))
    actual = asyncio.run(request_validation_error_handler(request, exception))

    assert actual.status_code == expected.status_code
    assert actual.headers == expected.headers
    assert actual.body == expected.body


