"""add last_seen_at to upcoming_matches

Revision ID: e1abb483d396
Revises: 6ff3a3de9d63
Create Date: 2026-06-05 04:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e1abb483d396'
down_revision = '6ff3a3de9d63'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'upcoming_matches',
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('upcoming_matches', 'last_seen_at')
