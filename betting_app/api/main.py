"""LoL-Breaker Betting API — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
import math
import os
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.encoders import decimal_encoder, jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from betting_app.api.routers import matches, predictions, bets
from betting_app.api.routers.embeddings import router as embeddings_router
from betting_app.api.routers.system import router as system_router
from betting_app.api.routers.scheduler import router as scheduler_router
from betting_app.api.routers.timing import router as timing_router
from betting_app.api.routers.bootstrap import router as bootstrap_router
from betting_app.api.routers.financial import router as financial_router
from betting_app.api.routers.rankings import router as rankings_router
from betting_app.api.routers.tournaments import router as tournaments_router
from betting_app.api.routers.players import router as players_router
from betting_app.api.routers.alerts import router as alerts_router
from betting_app.api.routers.props import router as props_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema is managed via Alembic or init_db() in CLI scripts
    yield

def configured_allowed_origins() -> list[str]:
    """Read exact browser origins; CORS is not the authentication boundary."""
    value = os.environ.get(
        "BETTING_API_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [origin.strip() for origin in value.split(",")]
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            not origin
            or "*" in origin
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("BETTING_API_ALLOWED_ORIGINS requires comma-separated exact HTTP(S) origins")
        # Force validation of malformed ports at startup, not on first request.
        parsed.port
    return list(dict.fromkeys(origins))


allowed_origins = configured_allowed_origins()


app = FastAPI(
    title="LoL-Breaker Betting API",
    description="LoL-Breaker betting research and EV intelligence backend.",
    version="0.2.0",
    lifespan=lifespan,
)


def _validation_number_json(value: float | Decimal) -> int | float | str:
    if isinstance(value, Decimal):
        return decimal_encoder(value) if value.is_finite() else str(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """Keep FastAPI's 422 detail contract while describing nonfinite inputs.

    Validators already reject these numbers. Their original input (including
    nested context) must become a descriptive string, not a JSON NaN/Infinity
    token that would make JSONResponse raise instead of returning the error.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(
                exc.errors(),
                custom_encoder={float: _validation_number_json, Decimal: _validation_number_json},
            ),
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reject_disallowed_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin is not None and origin not in allowed_origins:
        return JSONResponse(status_code=400, content={"detail": "Origin is not allowed"})
    return await call_next(request)

app.include_router(system_router)
app.include_router(scheduler_router)
app.include_router(matches.router)
app.include_router(predictions.router)
app.include_router(bets.router)
app.include_router(timing_router)
app.include_router(bootstrap_router)
app.include_router(financial_router)
app.include_router(embeddings_router)
app.include_router(rankings_router)
app.include_router(tournaments_router)
app.include_router(players_router)
app.include_router(props_router)
app.include_router(alerts_router)
