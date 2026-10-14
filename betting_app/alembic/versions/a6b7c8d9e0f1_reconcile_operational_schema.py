"""Reconcile operational tables omitted from the historical baseline.

Revision ID: a6b7c8d9e0f1
Revises: f5a6b7c8d9e0
Create Date: 2026-09-04 01:00:00.000000

The live schema acquired these objects before they were represented by a
replayable Alembic chain.  Every operation is conditional so upgrading an
existing installation is a no-op while a fresh PostgreSQL database converges
to the operational contract.
"""

from alembic import op
import sqlalchemy as sa

revision = "a6b7c8d9e0f1"
down_revision = "f5a6b7c8d9e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "golgg_match_mappings" not in tables:
        op.create_table(
            "golgg_match_mappings",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("canonical_match_id", sa.Integer(), nullable=False),
            sa.Column("golgg_match_id", sa.String(50), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
            sa.Column("mapped_by", sa.String(50), nullable=False, server_default="auto"),
            sa.Column("mapped_at", sa.String(50), nullable=True),
            sa.ForeignKeyConstraint(["canonical_match_id"], ["canonical_matches.id"]),
            sa.ForeignKeyConstraint(["golgg_match_id"], ["golgg_matches.match_id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("canonical_match_id", name="uq_golgg_mapping_canonical_match"),
            sa.UniqueConstraint("golgg_match_id", name="uq_golgg_mapping_source_match"),
        )

    canonical_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("canonical_matches")
    }
    missing_columns = (
        ("winner_name", sa.String(200)),
        ("loser_name", sa.String(200)),
        ("winner_normalized", sa.String(200)),
        ("winner_side", sa.String(20)),
        ("result_source", sa.String(50)),
        ("result_source_match_id", sa.String(50)),
        ("result_recorded_at", sa.DateTime(timezone=True)),
    )
    for name, column_type in missing_columns:
        if name not in canonical_columns:
            op.add_column(
                "canonical_matches",
                sa.Column(name, column_type, nullable=True),
            )


def downgrade() -> None:
    """Reconciliation is intentionally irreversible on existing databases."""
