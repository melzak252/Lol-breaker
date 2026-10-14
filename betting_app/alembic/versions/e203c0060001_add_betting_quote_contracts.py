"""Add evaluation provenance and reconcile serving quote contracts.

Revision ID: e203c0060001
Revises: d4e5f6a7b8c9

Numeric widening, the existing-contract index and removal of the implicit
prematch default are forward-only. Downgrade removes only new signal fields.
Legacy prices/probabilities may already have lost precision; this cannot repair it.
"""
from alembic import op
import sqlalchemy as sa

revision = "e203c0060001"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def _ensure_quote_index(bind) -> None:
    name = "idx_odds_unique_snapshot"
    row = bind.execute(sa.text("""
        SELECT i.indisunique, i.indisvalid, i.indnkeyatts, i.indnatts,
               pg_get_expr(i.indpred, i.indrelid) AS predicate,
               ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY k(attnum, n)
                     JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
                     ORDER BY k.n) AS columns,
               t.relname AS table_name
        FROM pg_class c JOIN pg_namespace ns ON ns.oid=c.relnamespace
        JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_class t ON t.oid=i.indrelid
        WHERE c.relname=:name AND ns.nspname=current_schema()
    """), {"name": name}).mappings().one_or_none()
    if row is not None:
        predicate = "".join((row["predicate"] or "").split())
        expected = {"(market_type='match_winner'::text)",
                    "((market_type)::text='match_winner'::text)",
                    "(market_type='match_winner'::charactervarying)"}
        if not (row["indisunique"] and row["indisvalid"]
                and row["indnkeyatts"] == row["indnatts"] == 3
                and list(row["columns"]) == ["canonical_match_id", "bookmaker_id", "scraped_at"]
                and row["table_name"] == "odds_snapshots" and predicate in expected):
            raise RuntimeError(f"{name}: existing index definition conflicts with quote contract")
        return
    duplicate = bind.execute(sa.text("""
        SELECT 1 FROM odds_snapshots
        WHERE market_type='match_winner' AND canonical_match_id IS NOT NULL
          AND bookmaker_id IS NOT NULL AND scraped_at IS NOT NULL
        GROUP BY canonical_match_id, bookmaker_id, scraped_at HAVING count(*) > 1 LIMIT 1
    """)).first()
    if duplicate is not None:
        raise RuntimeError(f"{name}: duplicate nonnull snapshot keys; no rows modified")
    op.create_index(name, "odds_snapshots", ["canonical_match_id", "bookmaker_id", "scraped_at"],
                    unique=True, postgresql_where=sa.text("market_type='match_winner'"))


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("betting quote contract reconciliation requires PostgreSQL")
    _ensure_quote_index(bind)
    columns = {c["name"] for c in sa.inspect(bind).get_columns("model_ev_signals")}
    if "evaluated_at" not in columns:
        op.add_column("model_ev_signals", sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True))
    if "market_context_json" not in columns:
        op.add_column("model_ev_signals", sa.Column("market_context_json", sa.Text(), nullable=True))
    op.alter_column("odds_snapshots", "is_live", server_default=None)
    serving = {
        "canonical_predictions": ("prob_a", "prob_b"),
        "odds_snapshots": ("odds_a", "odds_b"),
        "model_ev_signals": ("odds", "model_prob", "market_prob", "ev", "tax_rate", "stake_suggestion"),
    }
    for table, names in serving.items():
        types = {c["name"]: c["type"] for c in sa.inspect(bind).get_columns(table)}
        for name in names:
            if isinstance(types[name], sa.Integer):
                op.alter_column(table, name, type_=sa.Double(), postgresql_using=f'"{name}"::double precision')


def downgrade() -> None:
    op.drop_column("model_ev_signals", "market_context_json")
    op.drop_column("model_ev_signals", "evaluated_at")
