"""SQLAlchemy DeclarativeBase and engine/session helpers.

Supports both async (PostgreSQL) and sync (SQLite fallback) backends.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# ── Backend selection ────────────────────────────────────────────────────────

DB_URL = os.getenv("DATABASE_URL") or ""
USE_ASYNC = DB_URL.startswith("postgresql+asyncpg")


def is_timescale() -> bool:
    return True


def is_sqlite() -> bool:
    return False


# ── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── Sync engine (SQLite fallback) ────────────────────────────────────────────

_sync_engine = None
_SyncSession = None


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        from betting_app.core.db import database_url

        url = database_url()
        _sync_engine = create_engine(
            url.replace("+asyncpg", "+psycopg2"),
            pool_pre_ping=True,
        )
    return _sync_engine


def get_sync_session():
    engine = _get_sync_engine()
    return sessionmaker(bind=engine)


# ── Async engine (PostgreSQL) ────────────────────────────────────────────────

_async_engine = None
_AsyncSession = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        from betting_app.core.db import database_url

        url = database_url()
        if "postgresql" in url and not url.startswith("postgresql+asyncpg"):
            url = url.replace("postgresql://", "postgresql+asyncpg://").replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        _async_engine = create_async_engine(url, pool_pre_ping=True)
    return _async_engine


def get_async_session():
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(bind=get_async_engine(), expire_on_commit=False)
