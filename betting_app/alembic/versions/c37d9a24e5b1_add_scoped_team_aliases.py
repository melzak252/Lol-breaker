"""add scoped team aliases

Revision ID: c37d9a24e5b1
Revises: bf3d1e5a9c74
Create Date: 2026-07-23 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c37d9a24e5b1"
down_revision = "bf3d1e5a9c74"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("team_aliases", "source", type_=sa.String(length=255), existing_type=sa.String(length=50))
    op.add_column("team_aliases", sa.Column("raw_name", sa.String(length=200), nullable=True))
    op.add_column("team_aliases", sa.Column("normalized_alias", sa.String(length=200), nullable=True))
    op.add_column("team_aliases", sa.Column("source_system", sa.String(length=50), nullable=True))
    op.add_column("team_aliases", sa.Column("league_pattern", sa.String(length=200), nullable=True))
    op.add_column("team_aliases", sa.Column("tournament_pattern", sa.String(length=200), nullable=True))
    op.add_column("team_aliases", sa.Column("valid_from", sa.String(length=20), nullable=True))
    op.add_column("team_aliases", sa.Column("valid_to", sa.String(length=20), nullable=True))
    op.add_column("team_aliases", sa.Column("confidence", sa.Float(), nullable=True, server_default="1.0"))
    op.add_column("team_aliases", sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("team_aliases", sa.Column("is_blocked", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("team_aliases", sa.Column("review_status", sa.String(length=50), nullable=True, server_default="approved"))
    op.add_column("team_aliases", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column("team_aliases", sa.Column("created_at", sa.String(length=50), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")))
    op.add_column("team_aliases", sa.Column("updated_at", sa.String(length=50), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")))
    op.create_index("idx_team_aliases_lookup", "team_aliases", ["normalized_name", "is_active", "source_system", "league_pattern"], unique=False)

    # Backfill target normal form for existing aliases.  normalize_team_name is
    # intentionally not called from migrations; this SQL only makes old rows
    # queryable until the normalizer refreshes them on next write.
    op.execute("UPDATE team_aliases SET normalized_alias = lower(alias) WHERE normalized_alias IS NULL")

    # Scoped short aliases observed in production.  They are intentionally NOT
    # global Python aliases: they only apply inside matching contexts whose
    # league/tournament contains the configured pattern.
    alias_table = sa.sql.table(
        "team_aliases",
        sa.Column("normalized_name", sa.String),
        sa.Column("alias", sa.String),
        sa.Column("source", sa.String),
        sa.Column("raw_name", sa.String),
        sa.Column("normalized_alias", sa.String),
        sa.Column("source_system", sa.String),
        sa.Column("league_pattern", sa.String),
        sa.Column("confidence", sa.Float),
        sa.Column("is_active", sa.Integer),
        sa.Column("is_blocked", sa.Integer),
        sa.Column("review_status", sa.String),
        sa.Column("notes", sa.Text),
    )
    aliases = [
        ("use", "USE", "Unicorns Of Love Sexy Edition", "unicornsoflovese", "golgg-short", "Prime League", "GOL.GG Prime League abbreviates Unicorns Of Love Sexy Edition as USE"),
        ("khk", "KHK", "Kaufland Hangry Knights", "hangryknights", "golgg-short", "Prime League", "GOL.GG Prime League abbreviation"),
        ("eins", "EINS", "Eintracht Spandau", "eintracht spandau", "golgg-short", "Prime League", "GOL.GG Prime League abbreviation"),
        ("sge", "SGE", "Eintracht Frankfurt", "frankfurt", "golgg-short", "Prime League", "GOL.GG Prime League abbreviation"),
        ("ross", "ROSS", "Rossmann Centaurs", "rossmann centaurs", "golgg-short", "Prime League", "GOL.GG Prime League abbreviation"),
        ("tog", "TOG", "Teamorangegaming", "teamorangegaming", "golgg-short", "Prime League", "GOL.GG Prime League abbreviation"),
        ("blg", "BLG", "Bilibili Gaming", "bilibiligaming", "golgg-short", "MSI", "GOL.GG tournament-table abbreviation"),
        ("hle", "HLE", "Hanwha Life Esports", "hanwhalife", "golgg-short", "MSI", "GOL.GG tournament-table abbreviation"),
        ("tsw", "TSW", "Team Secret Whales", "secretwhales", "golgg-short", "MSI", "GOL.GG tournament-table abbreviation"),
    ]
    op.bulk_insert(
        alias_table,
        [
            {
                "normalized_name": normalized_name,
                "alias": alias,
                "source": f"{source}:golgg:{league_pattern}:*:*:*",
                "raw_name": raw_name,
                "normalized_alias": normalized_alias,
                "source_system": "golgg",
                "league_pattern": league_pattern,
                "confidence": 1.0,
                "is_active": 1,
                "is_blocked": 0,
                "review_status": "approved",
                "notes": notes,
            }
            for normalized_name, raw_name, alias, normalized_alias, source, league_pattern, notes in aliases
        ],
    )


def downgrade() -> None:
    op.drop_index("idx_team_aliases_lookup", table_name="team_aliases")
    op.drop_column("team_aliases", "updated_at")
    op.drop_column("team_aliases", "created_at")
    op.drop_column("team_aliases", "notes")
    op.drop_column("team_aliases", "review_status")
    op.drop_column("team_aliases", "is_blocked")
    op.drop_column("team_aliases", "is_active")
    op.drop_column("team_aliases", "confidence")
    op.drop_column("team_aliases", "valid_to")
    op.drop_column("team_aliases", "valid_from")
    op.drop_column("team_aliases", "tournament_pattern")
    op.drop_column("team_aliases", "league_pattern")
    op.drop_column("team_aliases", "source_system")
    op.drop_column("team_aliases", "normalized_alias")
    op.drop_column("team_aliases", "raw_name")
    op.alter_column("team_aliases", "source", type_=sa.String(length=50), existing_type=sa.String(length=255))
