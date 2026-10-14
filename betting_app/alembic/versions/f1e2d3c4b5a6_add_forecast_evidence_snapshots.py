"""Add append-only local forecast evidence snapshots.

Revision ID: f1e2d3c4b5a6
Revises: c8d9e0f1a2b3
Create Date: 2026-10-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "f1e2d3c4b5a6"
down_revision = "c8d9e0f1a2b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "forecast_evidence_snapshots" not in sa.inspect(bind).get_table_names():
        op.create_table(
            "forecast_evidence_snapshots",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("evidence_key", sa.String(length=200), nullable=False),
            sa.Column("content_sha256", sa.String(length=64), nullable=False),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column(
                "recorded_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("clock_timestamp()"),
                nullable=False,
            ),
            sa.UniqueConstraint("evidence_key", name="uq_forecast_evidence_key"),
        )
    op.execute("""
        CREATE OR REPLACE FUNCTION deny_forecast_evidence_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'forecast evidence snapshots are append-only';
        END;
        $$
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS forecast_evidence_no_mutation
        ON forecast_evidence_snapshots
    """)
    op.execute("""
        CREATE TRIGGER forecast_evidence_no_mutation
        BEFORE UPDATE OR DELETE ON forecast_evidence_snapshots
        FOR EACH ROW EXECUTE FUNCTION deny_forecast_evidence_mutation()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS forecast_evidence_no_mutation ON forecast_evidence_snapshots")
    op.execute("DROP FUNCTION IF EXISTS deny_forecast_evidence_mutation()")
    op.drop_table("forecast_evidence_snapshots")
