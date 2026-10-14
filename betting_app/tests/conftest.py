"""pytest: isolated PostgreSQL test DB per test."""

from __future__ import annotations

import os

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://betting:betting_local_password@localhost:5432/betting_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, pool, text
from sqlalchemy.orm import Session

from betting_app.core.db import dispose_engine
from betting_app.models.base import Base
import betting_app.models  # ensure all models are registered
from betting_app.api.main import app

_test_engine = None

def _get_test_engine():
    global _test_engine
    if _test_engine is None:
        _test_engine = create_engine(TEST_DATABASE_URL, poolclass=pool.NullPool)
        Base.metadata.create_all(_test_engine)
        with _test_engine.begin() as conn:
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_unique_snapshot
                ON odds_snapshots (canonical_match_id, bookmaker_id, scraped_at)
                WHERE market_type = 'match_winner'
            """))
            conn.execute(text("ALTER TABLE prop_odds_snapshots ADD COLUMN IF NOT EXISTS outcomes_payload TEXT;"))
            conn.execute(text("ALTER TABLE model_ev_signals ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE;"))
            conn.execute(text("ALTER TABLE model_ev_signals ADD COLUMN IF NOT EXISTS market_context_json TEXT;"))
    return _test_engine

@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    engine = _get_test_engine()

    # Registry services also create tables outside Base.metadata. Clear those
    # between tests and repeated suite runs rather than leaking prior evaluations.
    quote = engine.dialect.identifier_preparer.quote
    table_names = [quote(name) for name in inspect(engine).get_table_names()]
    truncate_sql = f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE;"

    with engine.begin() as conn:
        conn.execute(text(truncate_sql))
        conn.execute(
            text(
                "INSERT INTO bookmakers(id, name, base_url) VALUES "
                "(1,'manual',NULL),(2,'sts','https://www.sts.pl/'),(3,'betclic','https://www.betclic.pl/'),"
                "(4,'superbet','https://superbet.pl/'),(5,'efortuna','https://www.efortuna.pl/'),"
                "(6,'fortuna','https://www.efortuna.pl/'),(7,'betfan','https://betfan.pl/'),"
                "(8,'totalbet','https://totalbet.pl/'),(9,'lebull','https://www.lebull.pl/'),"
                "(10,'pinnacle','https://www.pinnacle.com/'),(11,'kalshi','https://kalshi.com/') "
                "ON CONFLICT (id) DO NOTHING"
            )
        )

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    dispose_engine()

    with TestClient(app) as c:
        yield c

    dispose_engine()
