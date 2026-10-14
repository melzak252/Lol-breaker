"""Reconcile Plan11 operational tables, defaults and UTC money/quote clocks.

Revision ID: e203c0060011
Revises: e203c0060001

Legacy timestamp-without-time-zone values were written as UTC by the app.
Interpret them explicitly as UTC, never the migration session's timezone.
Offset-bearing quote strings retain their represented instant; naive quote
strings use the same prior UTC interpretation. Invalid clocks or existing
constraint violations abort the transaction rather than discarding evidence or
repairing money. Existing evidence tables/triggers and numeric values are untouched.

This reconciliation is forward-only: downgrade must not erase operational rows
or restore timezone-losing columns.
"""
from alembic import op
import sqlalchemy as sa

revision = "e203c0060011"
down_revision = "e203c0060001"
branch_labels = None
depends_on = None


def _create_operational_tables(bind) -> None:
    tables = set(sa.inspect(bind).get_table_names())
    if "alert_configs" not in tables:
        op.create_table(
            "alert_configs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("min_ev", sa.Float(), nullable=False, server_default="0.05"),
            sa.Column("min_odds", sa.Float(), nullable=False, server_default="1.25"),
            sa.Column("max_odds", sa.Float(), nullable=True),
            sa.Column("cooldown_hours", sa.Float(), nullable=False, server_default="6"),
            sa.Column("discord_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("discord_webhook_url", sa.String(500), nullable=True),
            sa.Column("telegram_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("telegram_bot_token", sa.String(200), nullable=True),
            sa.Column("telegram_chat_id", sa.String(100), nullable=True),
            sa.Column("updated_at", sa.String(50), nullable=True),
        )
    if "value_alert_logs" not in tables:
        op.create_table(
            "value_alert_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("created_at", sa.String(50), nullable=False),
            sa.Column("canonical_match_id", sa.Integer(), nullable=True),
            sa.Column("match_label", sa.String(255), nullable=False),
            sa.Column("league", sa.String(100), nullable=True),
            sa.Column("match_start_at", sa.String(50), nullable=True),
            sa.Column("side", sa.String(10), nullable=True),
            sa.Column("team_name", sa.String(150), nullable=True),
            sa.Column("bookmaker_name", sa.String(100), nullable=True),
            sa.Column("odds", sa.Float(), nullable=True),
            sa.Column("model_prob", sa.Float(), nullable=True),
            sa.Column("market_prob", sa.Float(), nullable=True),
            sa.Column("ev", sa.Float(), nullable=True),
            sa.Column("suggested_stake", sa.Float(), nullable=True),
            sa.Column("channels", sa.String(100), nullable=False),
            sa.Column("status", sa.String(50), nullable=False),
            sa.Column("message_payload", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
        )
        op.create_index("ix_value_alert_logs_created_at", "value_alert_logs", ["created_at"])
        op.create_index("ix_value_alert_logs_canonical_match_id", "value_alert_logs", ["canonical_match_id"])
    if "prop_odds_snapshots" not in tables:
        op.create_table(
            "prop_odds_snapshots",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("bookmaker_id", sa.Integer(), sa.ForeignKey("bookmakers.id"), nullable=False),
            sa.Column("canonical_match_id", sa.Integer(), sa.ForeignKey("canonical_matches.id"), nullable=True),
            sa.Column("market_type", sa.String(50), nullable=False),
            sa.Column("map_number", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("line", sa.Float(), nullable=False),
            sa.Column("target_team", sa.String(200), nullable=True),
            sa.Column("raw_team_a", sa.String(200), nullable=True),
            sa.Column("raw_team_b", sa.String(200), nullable=True),
            sa.Column("odds_over", sa.Float(), nullable=True),
            sa.Column("odds_under", sa.Float(), nullable=True),
            sa.Column("prob_over_novig", sa.Float(), nullable=True),
            sa.Column("prob_under_novig", sa.Float(), nullable=True),
            sa.Column("margin", sa.Float(), nullable=True),
            sa.Column("raw_market_name", sa.String(255), nullable=True),
            sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("is_live", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("source_url", sa.String(500), nullable=True),
            sa.Column("outcomes_payload", sa.Text(), nullable=True),
        )
        for column in ("bookmaker_id", "canonical_match_id", "market_type", "map_number", "line", "target_team", "scraped_at"):
            op.create_index(f"ix_prop_odds_snapshots_{column}", "prop_odds_snapshots", [column])
        op.create_index("ix_prop_odds_match_market_time", "prop_odds_snapshots", ["canonical_match_id", "market_type", "scraped_at"])
        op.create_index("ix_prop_odds_line_search", "prop_odds_snapshots", ["canonical_match_id", "market_type", "line", "scraped_at"])
    if "model_prop_predictions" not in tables:
        op.create_table(
            "model_prop_predictions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("canonical_match_id", sa.Integer(), sa.ForeignKey("canonical_matches.id"), nullable=False),
            sa.Column("map_number", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("model_name", sa.String(100), nullable=False, server_default="kill_spread_distribution"),
            sa.Column("model_version", sa.String(50), nullable=False, server_default="prop-v1.0"),
            sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("expected_total_kills", sa.Float(), nullable=False),
            sa.Column("mu_team_a", sa.Float(), nullable=False),
            sa.Column("mu_team_b", sa.Float(), nullable=False),
            sa.Column("expected_spread", sa.Float(), nullable=False),
            sa.Column("league_avg_kills", sa.Float(), nullable=True),
            sa.Column("pace_category", sa.String(50), nullable=True),
            sa.Column("distribution_json", sa.Text(), nullable=True),
            sa.Column("signals_json", sa.Text(), nullable=True),
        )
        for column in ("canonical_match_id", "map_number", "predicted_at"):
            op.create_index(f"ix_model_prop_predictions_{column}", "model_prop_predictions", [column])
    columns = {column["name"] for column in sa.inspect(bind).get_columns("prop_odds_snapshots")}
    if "outcomes_payload" not in columns:
        op.add_column("prop_odds_snapshots", sa.Column("outcomes_payload", sa.Text(), nullable=True))


def _aware_clock(bind, table: str, column: str) -> None:
    definition = next(item for item in sa.inspect(bind).get_columns(table) if item["name"] == column)
    old_type = definition["type"]
    if isinstance(old_type, sa.DateTime) and old_type.timezone:
        return
    if isinstance(old_type, sa.DateTime):
        using = f'"{column}" AT TIME ZONE \'UTC\''
    elif isinstance(old_type, sa.String):
        # A timezone-qualified ISO clock casts independently of session timezone.
        # Unqualified clocks must not use the session-dependent timestamptz cast.
        using = (
            f'CASE WHEN "{column}" ~ \'[T ][0-9]{{2}}:[0-9]{{2}}.*(Z|[+-][0-9]{{2}}(:?[0-9]{{2}})?)$\' '
            f'THEN "{column}"::timestamptz ELSE "{column}"::timestamp AT TIME ZONE \'UTC\' END'
        )
    else:
        raise RuntimeError(f"{table}.{column}: unsupported legacy clock type {old_type}")
    op.alter_column(table, column, server_default=None)
    op.alter_column(table, column, type_=sa.DateTime(timezone=True), postgresql_using=using)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Plan11 operational reconciliation requires PostgreSQL")
    _create_operational_tables(bind)
    for table in ("bookmakers", "bookmaker_accounts"):
        active = next(item for item in sa.inspect(bind).get_columns(table) if item["name"] == "is_active")
        if isinstance(active["type"], sa.Boolean):
            op.alter_column(table, "is_active", server_default=None)
            op.alter_column(table, "is_active", type_=sa.Integer(), postgresql_using='"is_active"::integer')
        op.alter_column(table, "is_active", server_default="1")
    clocks = {
        "bookmaker_accounts": ("created_at", "updated_at"),
        "bets": ("placed_at", "settled_at", "match_start_time"),
        "bookmaker_wallet_transactions": ("transaction_time",),
        "odds_snapshots": ("scraped_at",),
        "prop_odds_snapshots": ("scraped_at",),
        "model_prop_predictions": ("predicted_at",),
    }
    for table, columns in clocks.items():
        for column in columns:
            _aware_clock(bind, table, column)
    defaults = {
        "bookmaker_accounts": {"currency": "PLN", "opening_balance": "0", "current_balance": "0",
                               "created_at": sa.text("CURRENT_TIMESTAMP"), "updated_at": sa.text("CURRENT_TIMESTAMP")},
        "bets": {"placed_at": sa.text("CURRENT_TIMESTAMP"), "status": "open", "profit": "0", "tax_rate": "0.12", "source": "manual"},
        "bookmaker_wallet_transactions": {"transaction_time": sa.text("CURRENT_TIMESTAMP")},
        "odds_snapshots": {"market_type": "match_winner", "scraped_at": sa.func.now()},
        "prop_odds_snapshots": {"map_number": "1", "scraped_at": sa.func.now(), "is_live": "0"},
        "model_prop_predictions": {"map_number": "1", "model_name": "kill_spread_distribution", "model_version": "prop-v1.0", "predicted_at": sa.func.now()},
        "canonical_predictions": {"prediction_status": "active"},
        "model_ev_signals": {"tax_rate": "0.12", "status": "new"},
    }
    for table, columns in defaults.items():
        for column, default in columns.items():
            op.alter_column(table, column, server_default=default)
    checks = {check["name"] for check in sa.inspect(bind).get_check_constraints("bookmaker_accounts")}
    if "chk_bookmaker_account_balance_non_negative" not in checks:
        op.create_check_constraint("chk_bookmaker_account_balance_non_negative", "bookmaker_accounts", "current_balance >= 0")
    side_checks = sa.inspect(bind).get_check_constraints("bets")
    if not any("side" in check["sqltext"] and "'a'" in check["sqltext"] and "'b'" in check["sqltext"] for check in side_checks):
        op.create_check_constraint("chk_bet_side", "bets", "side IN ('a','b')")


def downgrade() -> None:
    raise RuntimeError("Plan11 operational reconciliation is forward-only; operational data must be retained")
