"""add current team roster players

Revision ID: e4f5a6b7c8d9
Revises: d3c4e5f6a7b8
Create Date: 2026-08-05 17:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e4f5a6b7c8d9"
down_revision = "d3c4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "team_current_roster_players",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("team_id", sa.String(length=50), nullable=True),
        sa.Column("team_name", sa.String(length=200), nullable=False),
        sa.Column("normalized_team_name", sa.String(length=200), nullable=False),
        sa.Column("player_id", sa.String(length=50), nullable=False),
        sa.Column("player_name", sa.String(length=200), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="auto"),
        sa.Column("source_match_id", sa.String(length=50), nullable=True),
        sa.Column("source_game_id", sa.String(length=50), nullable=True),
        sa.Column("source_match_date", sa.String(length=50), nullable=True),
        sa.Column("updated_at", sa.String(length=50), nullable=True),
        sa.UniqueConstraint("normalized_team_name", "role", name="uq_team_current_roster_role"),
    )
    op.create_index("ix_team_current_roster_players_team_id", "team_current_roster_players", ["team_id"])
    op.create_index("ix_team_current_roster_players_normalized_team_name", "team_current_roster_players", ["normalized_team_name"])
    op.create_index("ix_team_current_roster_players_player_id", "team_current_roster_players", ["player_id"])


def downgrade() -> None:
    op.drop_index("ix_team_current_roster_players_player_id", table_name="team_current_roster_players")
    op.drop_index("ix_team_current_roster_players_normalized_team_name", table_name="team_current_roster_players")
    op.drop_index("ix_team_current_roster_players_team_id", table_name="team_current_roster_players")
    op.drop_table("team_current_roster_players")
