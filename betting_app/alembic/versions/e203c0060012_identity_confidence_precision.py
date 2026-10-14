"""Preserve fractional identity confidence in the application float contract.

Integer storage previously rounded valid fractions; recovered precision cannot
restore already rounded historical values. Existing stored values are retained.
"""
from alembic import op
import sqlalchemy as sa

revision = "e203c0060012"
down_revision = "e203c0060011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "canonical_matches", "match_confidence",
        existing_type=sa.Integer(), type_=sa.Float(), existing_nullable=False,
        server_default="1.0", postgresql_using="match_confidence::double precision",
    )


def downgrade() -> None:
    raise RuntimeError("Identity confidence migration is forward-only; fractional values must not be rounded")
