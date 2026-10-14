"""Alembic environment configuration — loads SQLAlchemy models for autogenerate."""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from betting_app.models.base import Base

# Import all models so they are registered on Base.metadata
import betting_app.models.bookmaker  # noqa: F401
import betting_app.models.golgg       # noqa: F401
import betting_app.models.match      # noqa: F401
import betting_app.models.odds       # noqa: F401
import betting_app.models.prediction  # noqa: F401
import betting_app.models.identity  # noqa: F401
import betting_app.models.automation  # noqa: F401


config = context.config

# Override sqlalchemy.url with DATABASE_URL env var if present (Docker/production)
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Alembic needs a synchronous driver; convert async URLs if needed
    database_url = database_url.replace("+asyncpg", "+psycopg2").replace(
        "+asyncpg", "+psycopg2"
    )
    config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
