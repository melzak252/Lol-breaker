"""Add immutable source captures and opportunity events.

Revision ID: d4e5f6a7b8c9
Revises: f1e2d3c4b5a6
Create Date: 2026-10-02 05:04:05.779765
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "f1e2d3c4b5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "source_captures" not in tables:
        op.create_table(
            "source_captures",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("capture_id", sa.String(length=200), nullable=False),
            sa.Column("source_id", sa.String(length=200), nullable=False),
            sa.Column("evidence_uri", sa.Text(), nullable=False),
            sa.Column("sha256", sa.String(length=64)),
            sa.Column("content_base64", sa.Text()),
            sa.Column("body_uri", sa.Text()),
            sa.Column("body_representation", sa.String(length=40), nullable=False),
            sa.Column("content_encoding", sa.String(length=100)),
            sa.Column("character_encoding", sa.String(length=80)),
            sa.Column("request_started_at", sa.DateTime(timezone=True)),
            sa.Column("response_received_at", sa.DateTime(timezone=True)),
            sa.Column("retrieved_at", sa.DateTime(timezone=True)),
            sa.Column("first_observed_at", sa.DateTime(timezone=True)),
            sa.Column("observed_at", sa.DateTime(timezone=True)),
            sa.Column("published_at", sa.DateTime(timezone=True)),
            sa.Column("source_effective_at", sa.DateTime(timezone=True)),
            sa.Column("source_revision", sa.String(length=200)),
            sa.Column("http_status", sa.Integer()),
            sa.Column("source_class", sa.String(length=50), nullable=False),
            sa.Column("availability_basis", sa.String(length=100), nullable=False),
            sa.Column("attempt_status", sa.String(length=20), nullable=False),
            sa.Column("failure_reason", sa.Text()),
            sa.Column("request_metadata_json", sa.Text(), nullable=False),
            sa.Column("response_metadata_json", sa.Text(), nullable=False),
            sa.Column("source_authentication_status", sa.String(length=30), nullable=False),
            sa.Column(
                "recorded_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("clock_timestamp()"),
                nullable=False,
            ),
            sa.UniqueConstraint("capture_id", name="uq_source_capture_id"),
            sa.CheckConstraint(
                "(sha256 IS NULL AND content_base64 IS NULL AND body_uri IS NULL) OR "
                "(sha256 IS NOT NULL AND content_base64 IS NOT NULL AND body_uri IS NOT NULL)",
                name="ck_source_capture_body_reference",
            ),
            sa.CheckConstraint(
                "(request_started_at IS NULL OR response_received_at IS NULL OR "
                "request_started_at <= response_received_at) AND "
                "(request_started_at IS NULL OR observed_at IS NULL OR "
                "request_started_at <= observed_at) AND "
                "(response_received_at IS NULL OR observed_at IS NULL OR "
                "response_received_at <= observed_at) AND "
                "(first_observed_at IS NULL OR observed_at IS NULL OR "
                "first_observed_at <= observed_at) AND "
                "(retrieved_at IS NULL OR retrieved_at = response_received_at)",
                name="ck_source_capture_clock_order",
            ),
            sa.CheckConstraint(
                "attempt_status IN ('captured', 'failed')",
                name="ck_source_capture_attempt_status",
            ),
            sa.CheckConstraint(
                "source_authentication_status IN ('local_fixture', 'unverified', 'authenticated')",
                name="ck_source_capture_authentication_status",
            ),
        )
    if "opportunity_events" not in tables:
        op.create_table(
            "opportunity_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_key", sa.String(length=200), nullable=False),
            sa.Column("protocol_id", sa.String(length=120), nullable=False),
            sa.Column("provider_id", sa.String(length=120), nullable=False),
            sa.Column("fixture_id", sa.String(length=200), nullable=False),
            sa.Column("origin_at", sa.DateTime(timezone=True)),
            sa.Column("event_type", sa.String(length=40), nullable=False),
            sa.Column("primary_decision", sa.Boolean(), server_default=sa.text("false"), nullable=False),
            sa.Column("parent_event_key", sa.String(length=200)),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column("payload_sha256", sa.String(length=64), nullable=False),
            sa.Column("forecast_evidence_key", sa.String(length=200)),
            sa.Column(
                "recorded_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("clock_timestamp()"),
                nullable=False,
            ),
            sa.UniqueConstraint("event_key", name="uq_opportunity_event_key"),
            sa.ForeignKeyConstraint(
                ["parent_event_key"], ["opportunity_events.event_key"],
                name="fk_opportunity_parent_event",
            ),
            sa.ForeignKeyConstraint(
                ["forecast_evidence_key"], ["forecast_evidence_snapshots.evidence_key"],
                name="fk_opportunity_forecast_evidence",
            ),
            sa.CheckConstraint(
                "NOT primary_decision OR origin_at IS NOT NULL",
                name="ck_opportunity_primary_origin_required",
            ),
            sa.CheckConstraint(
                "event_type NOT IN ('schedule_revision', 'correction') OR parent_event_key IS NOT NULL",
                name="ck_opportunity_revision_parent_required",
            ),
            sa.CheckConstraint(
                "event_type IN ('discovery', 'schedule_revision', 'origin_pending', "
                "'primary_decision', 'correction', 'outcome_release')",
                name="ck_opportunity_event_type",
            ),
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_source_capture_revision_first_observed "
        "ON source_captures(source_id, sha256, source_revision, first_observed_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_opportunity_event_protocol_recorded "
        "ON opportunity_events(protocol_id, recorded_at, id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_opportunity_primary_origin "
        "ON opportunity_events(protocol_id, provider_id, fixture_id, origin_at) "
        "WHERE primary_decision"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_opportunity_primary_fixture "
        "ON opportunity_events(protocol_id, provider_id, fixture_id) "
        "WHERE primary_decision"
    )
    op.execute("""
        CREATE OR REPLACE FUNCTION deny_plan09_evidence_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only';
        END;
        $$
    """)
    op.execute("DROP TRIGGER IF EXISTS source_captures_no_mutation ON source_captures")
    op.execute("""
        CREATE TRIGGER source_captures_no_mutation
        BEFORE UPDATE OR DELETE ON source_captures
        FOR EACH ROW EXECUTE FUNCTION deny_plan09_evidence_mutation()
    """)
    op.execute("DROP TRIGGER IF EXISTS opportunity_events_no_mutation ON opportunity_events")
    op.execute("""
        CREATE TRIGGER opportunity_events_no_mutation
        BEFORE UPDATE OR DELETE ON opportunity_events
        FOR EACH ROW EXECUTE FUNCTION deny_plan09_evidence_mutation()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS opportunity_events_no_mutation ON opportunity_events")
    op.execute("DROP TRIGGER IF EXISTS source_captures_no_mutation ON source_captures")
    op.drop_table("opportunity_events")
    op.drop_table("source_captures")
    op.execute("DROP FUNCTION IF EXISTS deny_plan09_evidence_mutation()")
