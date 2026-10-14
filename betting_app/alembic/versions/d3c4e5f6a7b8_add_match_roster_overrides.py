"""Add manually confirmed upcoming-match rosters.

Revision ID: d3c4e5f6a7b8
Revises: c37d9a24e5b1
Create Date: 2026-08-05 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "d3c4e5f6a7b8"
down_revision = "c37d9a24e5b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_roster_overrides",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("canonical_match_id", sa.Integer(), sa.ForeignKey("canonical_matches.id"), nullable=False),
        sa.Column("team_side", sa.String(length=1), nullable=False),
        sa.Column("roster_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.String(length=50), nullable=True),
        sa.UniqueConstraint("canonical_match_id", "team_side", name="uq_match_roster_override_side"),
    )
    op.create_index(
        "ix_match_roster_overrides_canonical_match_id",
        "match_roster_overrides",
        ["canonical_match_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_match_roster_overrides_canonical_match_id", table_name="match_roster_overrides")
    op.drop_table("match_roster_overrides")
