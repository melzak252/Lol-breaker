"""Retain the historical revision marker without replaying invalid autogenerate output.

Revision ID: 6ff3a3de9d63
Revises: 079e2229e15e
Create Date: 2026-05-31 21:51:10.303024

The initial revision already creates the schema represented by this historical
revision.  The former body was an accidental cross-dialect autogenerate diff:
it dropped tables that the initial revision never created, removed valid
columns and indexes, and attempted unsafe type conversions.  Deployed
databases have already advanced beyond this marker, so preserving the marker
as a no-op gives fresh PostgreSQL databases the same forward migration path
without mutating existing installations.
"""

from typing import Sequence, Union


revision: str = "6ff3a3de9d63"
down_revision: Union[str, Sequence[str], None] = "079e2229e15e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """The preceding baseline already contains this revision's schema."""


def downgrade() -> None:
    """No schema operations were applied by this revision."""
