"""add best_of to canonical_matches

Revision ID: a8f2c4d7e912
Revises: e1abb483d396
Create Date: 2026-06-05 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a8f2c4d7e912'
down_revision = 'e1abb483d396'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'canonical_matches',
        sa.Column('best_of', sa.SmallInteger(), server_default='1', nullable=True),
    )

    # Populate best_of=3 for Bo3 leagues (everything else stays at default 1)
    # Bo3: LCK, LPL, LEC, LCK Challengers, LCK Road to MSI, CBLOL, LJL,
    #      LCS NA, LCS, TCL, LCP, TJ Sports LoL / LPL, Japan League
    # Exclude NACL / NA Challengers (Bo1) and LPLOL (contains "lpl" substring)
    op.execute("""
        UPDATE canonical_matches
        SET best_of = 3
        WHERE (
            league ILIKE '%lck%'
            OR league ILIKE '%lpl%'
            OR league ILIKE '%lec%'
            OR league ILIKE '%cblo%'
            OR league ILIKE '%ljl%'
            OR league ILIKE '%lcs%'
            OR league ILIKE '%tcl%'
            OR league ILIKE '%lcp%'
            OR league ILIKE '%tj sports%lpl%'
            OR league = 'Japan League'
        )
        AND league NOT ILIKE '%nacl%'
        AND league NOT ILIKE '%na challengers%'
        AND league NOT ILIKE '%lpol%'
    """)


def downgrade() -> None:
    op.drop_column('canonical_matches', 'best_of')
