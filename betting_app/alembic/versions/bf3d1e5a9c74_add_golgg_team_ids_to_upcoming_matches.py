"""add golgg team ids to upcoming_matches

Revision ID: bf3d1e5a9c74
Revises: a8f2c4d7e912
Create Date: 2026-06-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bf3d1e5a9c74'
down_revision = 'a8f2c4d7e912'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'upcoming_matches',
        sa.Column('team_a_golgg_id', sa.Integer(), sa.ForeignKey('golgg_teams.id'), nullable=True),
    )
    op.add_column(
        'upcoming_matches',
        sa.Column('team_b_golgg_id', sa.Integer(), sa.ForeignKey('golgg_teams.id'), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('upcoming_matches', 'team_b_golgg_id')
    op.drop_column('upcoming_matches', 'team_a_golgg_id')
