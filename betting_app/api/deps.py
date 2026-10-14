"""FastAPI dependencies — SQLAlchemy session, query helpers."""

from __future__ import annotations

import datetime
from collections.abc import Generator
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from betting_app.core.db import get_session


def get_db() -> Generator[Any, None, None]:
    """Yield an SQLAlchemy session, close on teardown."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def query_df(db: Any, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute raw SQL with named parameters, return list of dicts (datetimes as strings)."""
    result = db.execute(text(sql), params or {})
    columns = list(result.keys())
    rows = result.fetchall()
    out = []
    for row in rows:
        d = dict(row._mapping) if hasattr(row, "_mapping") else dict(zip(columns, row))
        # Convert datetimes to ISO strings, Decimals to float
        for k, v in d.items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                d[k] = v.isoformat()
            elif isinstance(v, Decimal):
                d[k] = float(v)
        out.append(d)
    return out


def query_one(db: Any, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Execute raw SQL, return first row as dict or None."""
    result = db.execute(text(sql), params or {})
    columns = list(result.keys())
    row = result.fetchone()
    if row is None:
        return None
    d = dict(row._mapping) if hasattr(row, "_mapping") else dict(zip(columns, row))
    for k, v in d.items():
        if isinstance(v, (datetime.datetime, datetime.date)):
            d[k] = v.isoformat()
        elif isinstance(v, Decimal):
            d[k] = float(v)
    return d
