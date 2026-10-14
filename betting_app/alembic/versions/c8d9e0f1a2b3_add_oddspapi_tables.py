"""Add OddsPapi fixture mappings and request audit log tables.

Revision ID: c8d9e0f1a2b3
Revises: a6b7c8d9e0f1
Create Date: 2026-09-05 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
revision = "c8d9e0f1a2b3"
down_revision = "a6b7c8d9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "oddspapi_fixture_mappings" not in tables:
        op.create_table(
            "oddspapi_fixture_mappings",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("fixture_id", sa.String(length=100), nullable=False),
            sa.Column("canonical_match_id", sa.Integer(), nullable=True),
            sa.Column("sport_id", sa.Integer(), server_default="18", nullable=False),
            sa.Column("league", sa.String(length=100), nullable=True),
            sa.Column("provider_team_1", sa.String(length=200), nullable=False),
            sa.Column("provider_team_2", sa.String(length=200), nullable=False),
            sa.Column("provider_team_1_is_a", sa.Integer(), nullable=True),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("has_odds", sa.Integer(), server_default="1", nullable=False),
            sa.Column(
                "last_synced_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["canonical_match_id"], ["canonical_matches.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("fixture_id", name="uq_oddspapi_fixture_id"),
        )
        op.create_index(
            "ix_oddspapi_fixture_mappings_canonical_match_id",
            "oddspapi_fixture_mappings",
            ["canonical_match_id"],
        )

    if "oddspapi_request_logs" not in tables:
        op.create_table(
            "oddspapi_request_logs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("endpoint", sa.String(length=100), nullable=False),
            sa.Column("fixture_id", sa.String(length=100), nullable=True),
            sa.Column("status_code", sa.Integer(), nullable=False),
            sa.Column("response_time_ms", sa.Integer(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_oddspapi_request_logs_created_at",
            "oddspapi_request_logs",
            ["created_at"],
        )
        op.create_index(
            "ix_oddspapi_request_logs_fixture_id",
            "oddspapi_request_logs",
            ["fixture_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "oddspapi_request_logs" in tables:
        op.drop_table("oddspapi_request_logs")
    if "oddspapi_fixture_mappings" in tables:
        op.drop_table("oddspapi_fixture_mappings")
