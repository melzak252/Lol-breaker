"""add durable identity and mapping review registry

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-09-04 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "f5a6b7c8d9e0"
down_revision = "e4f5a6b7c8d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "canonical_teams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_name", sa.String(200), nullable=False),
        sa.Column("normalized_name", sa.String(200), nullable=False),
        sa.Column("squad_type", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name"),
    )
    op.create_table(
        "canonical_competitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_name", sa.String(200), nullable=False),
        sa.Column("normalized_name", sa.String(200), nullable=False),
        sa.Column("family", sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name"),
    )
    op.create_table(
        "source_team_identities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_team_id", sa.Integer(), nullable=False),
        sa.Column("source_system", sa.String(50), nullable=False),
        sa.Column("source_team_id", sa.String(100), nullable=False),
        sa.Column("source_name", sa.String(200), nullable=False),
        sa.Column("normalized_source_name", sa.String(200), nullable=False),
        sa.Column("competition_scope", sa.String(200), server_default="", nullable=False),
        sa.Column("valid_from", sa.String(20), server_default="", nullable=False),
        sa.Column("valid_to", sa.String(20), server_default="", nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1", nullable=False),
        sa.Column("review_status", sa.String(50), server_default="approved", nullable=False),
        sa.ForeignKeyConstraint(["canonical_team_id"], ["canonical_teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_system", "source_team_id", "competition_scope", "valid_from", "valid_to", name="uq_source_team_identity_scope"),
    )
    op.create_index("ix_source_team_identity_lookup", "source_team_identities", ["source_system", "normalized_source_name", "competition_scope"], unique=False)
    op.create_table(
        "source_competition_identities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_competition_id", sa.Integer(), nullable=False),
        sa.Column("source_system", sa.String(50), nullable=False),
        sa.Column("source_competition_id", sa.String(200), nullable=False),
        sa.Column("source_name", sa.String(300), nullable=False),
        sa.Column("normalized_source_name", sa.String(300), nullable=False),
        sa.ForeignKeyConstraint(["canonical_competition_id"], ["canonical_competitions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_system", "source_competition_id", name="uq_source_competition_identity"),
    )
    op.create_table(
        "mapping_review_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_match_id", sa.Integer(), nullable=False),
        sa.Column("old_golgg_match_id", sa.String(50), nullable=True),
        sa.Column("new_golgg_match_id", sa.String(50), nullable=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("operator", sa.String(100), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["canonical_match_id"], ["canonical_matches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_mapping_review_match_time", "mapping_review_decisions", ["canonical_match_id", "decided_at"], unique=False)

    op.execute("""
        INSERT INTO canonical_teams (canonical_name, normalized_name)
        SELECT MIN(team_name), normalized_name
        FROM (
            SELECT team_a_name AS team_name, normalized_team_a AS normalized_name FROM canonical_matches
            UNION ALL
            SELECT team_b_name, normalized_team_b FROM canonical_matches
        ) names
        WHERE normalized_name IS NOT NULL AND normalized_name <> ''
        GROUP BY normalized_name
        ON CONFLICT (normalized_name) DO NOTHING
    """)
    op.execute("""
        INSERT INTO canonical_competitions (canonical_name, normalized_name, family)
        SELECT MIN(league), lower(trim(league)), lower(trim(league))
        FROM canonical_matches
        WHERE league IS NOT NULL AND trim(league) <> ''
        GROUP BY lower(trim(league))
        ON CONFLICT (normalized_name) DO NOTHING
    """)
    op.execute("""
        INSERT INTO source_team_identities (
            canonical_team_id, source_system, source_team_id, source_name,
            normalized_source_name, confidence, review_status
        )
        SELECT ct.id, 'golgg', COALESCE(CAST(gt.team_id AS VARCHAR), 'name:' || gt.normalized_name),
               gt.team_name, gt.normalized_name, 1.0, 'backfilled-name-exact'
        FROM golgg_teams gt
        JOIN canonical_teams ct ON ct.normalized_name = gt.normalized_name
        WHERE gt.normalized_name IS NOT NULL AND gt.normalized_name <> ''
        ON CONFLICT DO NOTHING
    """)
    op.execute("""
        INSERT INTO source_competition_identities (
            canonical_competition_id, source_system, source_competition_id,
            source_name, normalized_source_name
        )
        SELECT cc.id, 'bookmaker', cc.normalized_name, cc.canonical_name, cc.normalized_name
        FROM canonical_competitions cc
        ON CONFLICT DO NOTHING
    """)


def downgrade() -> None:
    op.drop_index("ix_mapping_review_match_time", table_name="mapping_review_decisions")
    op.drop_table("mapping_review_decisions")
    op.drop_table("source_competition_identities")
    op.drop_index("ix_source_team_identity_lookup", table_name="source_team_identities")
    op.drop_table("source_team_identities")
    op.drop_table("canonical_competitions")
    op.drop_table("canonical_teams")
