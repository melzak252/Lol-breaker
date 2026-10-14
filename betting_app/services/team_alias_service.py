"""Scoped team-alias resolution.

Aliases are operational data, not matching code.  The generic normalizer in
``betting_app.core.matching`` still performs deterministic text cleanup (case,
punctuation, suffix removal), but source/league-specific aliases such as
``USE`` or ``BLG`` are resolved here from the ``team_aliases`` table.

This prevents global three-letter aliases from matching unrelated teams in
other tournaments while keeping alias changes auditable and deployable through
database migrations/manual data changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import re
import unicodedata
from typing import Any

from sqlalchemy import inspect, text

from betting_app.core.db import get_session, query_df, transaction
from betting_app.core.matching import normalize_team_name


@dataclass(frozen=True)
class AliasContext:
    """Runtime context used to decide if a scoped alias is applicable."""

    source_system: str | None = None
    league: str | None = None
    tournament: str | None = None
    match_date: date | datetime | str | None = None


@dataclass(frozen=True)
class AliasResolution:
    """Result returned by scoped alias lookups."""

    target_name: str | None
    normalized_target: str | None
    source: str | None
    alias_id: int | None
    confidence: float
    blocked: bool = False


SHORT_ALIAS_MAX_LEN = 3


def alias_lookup_key(raw_name: str) -> str:
    """Normalize an alias *source* key without applying global team aliases.

    Alias-table source keys must not call the global team normalizer because
    any legacy/built-in alias there would silently make a scoped short alias
    global again.  This key performs only deterministic text cleanup.
    """

    ascii_name = unicodedata.normalize("NFKD", raw_name or "").encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^a-z0-9]+", " ", ascii_name.lower())
    ascii_name = re.sub(r"\be\s+sports?\b", " esports ", ascii_name)
    stop_words = {"esports", "esport", "gaming", "team", "lol", "leagueoflegends"}
    return " ".join(token for token in ascii_name.split() if token and token not in stop_words).strip()


def is_short_alias(raw_name: str) -> bool:
    """Return True for collision-prone compact aliases like BLG/HLE/USE."""

    compact = re.sub(r"[^a-z0-9]+", "", (raw_name or "").lower())
    return 0 < len(compact) <= SHORT_ALIAS_MAX_LEN


def _contains_pattern(value: str | None, pattern: str | None) -> bool:
    if not pattern:
        return True
    if not value:
        return False
    return normalize_team_name(pattern) in normalize_team_name(value)


def _date_str(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text_value = str(value)
    return text_value[:10] if text_value else None


def _date_in_range(match_date: date | datetime | str | None, valid_from: str | None, valid_to: str | None) -> bool:
    date_value = _date_str(match_date)
    if valid_from and (not date_value or date_value < valid_from[:10]):
        return False
    if valid_to and (not date_value or date_value > valid_to[:10]):
        return False
    return True


def _row_matches_context(row: dict[str, Any], context: AliasContext, *, strict_short_alias_scope: bool) -> bool:
    source_system = row.get("source_system")
    if source_system and context.source_system and str(source_system).lower() != context.source_system.lower():
        return False
    if source_system and not context.source_system:
        # Source-scoped aliases require a source context.
        return False

    league_pattern = row.get("league_pattern")
    tournament_pattern = row.get("tournament_pattern")
    if not _contains_pattern(context.league, league_pattern):
        return False
    if not _contains_pattern(context.tournament, tournament_pattern):
        return False
    if not _date_in_range(context.match_date, row.get("valid_from"), row.get("valid_to")):
        return False

    if strict_short_alias_scope and is_short_alias(str(row.get("raw_name") or row.get("normalized_name") or "")):
        # Short aliases must have at least one non-source scope.  A source-only
        # BLG/HLE/USE mapping is still too broad across LoL tournaments.
        if not league_pattern and not tournament_pattern and not row.get("valid_from") and not row.get("valid_to"):
            return False
    return True


@lru_cache(maxsize=1)
def _team_alias_columns() -> frozenset[str]:
    """Return available columns for compatibility with pre-migration DBs."""

    session = get_session()
    try:
        return frozenset(column["name"] for column in inspect(session.bind).get_columns("team_aliases"))
    except Exception:
        return frozenset()
    finally:
        session.close()


def clear_alias_schema_cache() -> None:
    """Clear cached schema metadata (used by tests after DB reset)."""

    _team_alias_columns.cache_clear()
    _has_durable_identity_registry.cache_clear()



@lru_cache(maxsize=1)
def _has_durable_identity_registry() -> bool:
    session = get_session()
    try:
        return inspect(session.bind).has_table("source_team_identities")
    except Exception:
        return False
    finally:
        session.close()


def _resolve_durable_identity(
    normalized: str,
    context: AliasContext,
) -> AliasResolution | None:
    if not context.source_system or not _has_durable_identity_registry():
        return None
    rows = query_df(
        """
        SELECT sti.id, sti.canonical_team_id, sti.source_name, sti.confidence,
               sti.competition_scope, sti.valid_from, sti.valid_to,
               ct.canonical_name, ct.normalized_name
        FROM source_team_identities sti
        JOIN canonical_teams ct ON ct.id = sti.canonical_team_id
        WHERE sti.source_system = ?
          AND sti.normalized_source_name = ?
          AND sti.review_status IN ('approved', 'backfilled-name-exact')
        ORDER BY sti.confidence DESC, sti.id
        """,
        (context.source_system.lower(), normalized),
    )
    applicable = [
        row
        for row in rows.to_dict("records")
        if _contains_pattern(
            context.league or context.tournament,
            str(row.get("competition_scope") or "") or None,
        )
        and _date_in_range(
            context.match_date,
            str(row.get("valid_from") or "") or None,
            str(row.get("valid_to") or "") or None,
        )
    ]
    targets = {str(row["normalized_name"]): row for row in applicable}
    if not targets:
        return None
    if len(targets) > 1:
        return AliasResolution(None, None, "durable-ambiguous", None, 0.0)
    row = next(iter(targets.values()))
    return AliasResolution(
        target_name=str(row["canonical_name"]),
        normalized_target=str(row["normalized_name"]),
        source="durable-identity",
        alias_id=None,
        confidence=float(row.get("confidence") or 1.0),
    )

def resolve_scoped_alias(
    raw_name: str,
    *,
    context: AliasContext | None = None,
    strict_short_alias_scope: bool = True,
) -> AliasResolution:
    """Resolve a raw team name through active, context-scoped DB aliases.

    If the table still has the legacy three-column schema, this function falls
    back to the old lookup for non-short aliases only.  Short aliases are not
    resolved without scope metadata.
    """

    context = context or AliasContext()
    normalized = alias_lookup_key(raw_name)
    durable = _resolve_durable_identity(normalized, context)
    if durable is not None:
        return durable
    compact = normalized.replace(" ", "")
    columns = _team_alias_columns()
    if not columns:
        return AliasResolution(None, None, None, None, 0.0)

    select_cols = ["id", "normalized_name", "alias", "source"]
    optional_cols = [
        "raw_name",
        "normalized_alias",
        "source_system",
        "league_pattern",
        "tournament_pattern",
        "valid_from",
        "valid_to",
        "confidence",
        "is_active",
        "is_blocked",
        "review_status",
    ]
    select_cols += [column for column in optional_cols if column in columns]

    active_clause = "AND COALESCE(is_active, 1) = 1" if "is_active" in columns else ""
    order_terms = ["id DESC"]
    if "source_system" in columns:
        order_terms.insert(0, "CASE WHEN source_system IS NOT NULL THEN 0 ELSE 1 END")
    scoped_columns = {"league_pattern", "tournament_pattern", "valid_from", "valid_to"} & columns
    if scoped_columns:
        scope_exprs = [f"{column} IS NOT NULL" for column in sorted(scoped_columns)]
        order_terms.insert(1 if "source_system" in columns else 0, f"CASE WHEN {' OR '.join(scope_exprs)} THEN 0 ELSE 1 END")
    if "confidence" in columns:
        order_terms.insert(-1, "CASE WHEN confidence IS NULL THEN 1 ELSE 0 END")
        order_terms.insert(-1, "confidence DESC")
    rows_df = query_df(
        f"""
        SELECT {', '.join(select_cols)}
        FROM team_aliases
        WHERE normalized_name IN (?, ?)
          {active_clause}
        ORDER BY {', '.join(order_terms)}
        """,
        (normalized, compact),
    )
    if rows_df.empty:
        return AliasResolution(None, None, None, None, 0.0)

    applicable: list[dict[str, Any]] = []
    for row_obj in rows_df.to_dict("records"):
        row = dict(row_obj)
        is_blocked = "is_blocked" in columns and int(row.get("is_blocked") or 0)
        if not _row_matches_context(
            row,
            context,
            strict_short_alias_scope=False if is_blocked else strict_short_alias_scope,
        ):
            continue
        row["_specificity"] = sum(
            bool(row.get(column))
            for column in (
                "source_system",
                "league_pattern",
                "tournament_pattern",
                "valid_from",
                "valid_to",
            )
            if column in columns
        )
        applicable.append(row)

    if not applicable:
        return AliasResolution(None, None, None, None, 0.0)

    best_specificity = max(int(row["_specificity"]) for row in applicable)
    best_rows = [
        row for row in applicable if int(row["_specificity"]) == best_specificity
    ]
    blocked_rows = [
        row for row in best_rows
        if "is_blocked" in columns and int(row.get("is_blocked") or 0)
    ]
    if blocked_rows:
        blocked = min(blocked_rows, key=lambda row: int(row["id"]))
        return AliasResolution(
            None,
            None,
            str(blocked.get("source")),
            int(blocked["id"]),
            0.0,
            blocked=True,
        )

    targets: dict[str, list[dict[str, Any]]] = {}
    for row in best_rows:
        target = str(row.get("alias") or "").strip()
        if not target:
            continue
        normalized_target = (
            str(row.get("normalized_alias") or "").strip()
            or normalize_team_name(target)
        )
        targets.setdefault(normalized_target, []).append(row)

    if len(targets) != 1:
        # Conflicting equally specific active aliases are review data, not a
        # deterministic identity decision based on insertion order.
        return AliasResolution(None, None, "ambiguous", None, 0.0)

    normalized_target, target_rows = next(iter(targets.items()))
    selected = max(
        target_rows,
        key=lambda row: (
            str(row.get("source") or "").startswith("manual"),
            float(row.get("confidence") or 1.0),
            -int(row["id"]),
        ),
    )
    return AliasResolution(
        target_name=str(selected.get("alias") or "").strip(),
        normalized_target=normalized_target,
        source=str(selected.get("source") or "manual"),
        alias_id=int(selected["id"]),
        confidence=float(selected.get("confidence") or 1.0),
    )


def upsert_scoped_alias(
    raw_name: str,
    target_name: str,
    *,
    source: str = "manual",
    source_system: str | None = None,
    league_pattern: str | None = None,
    tournament_pattern: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    confidence: float = 1.0,
    notes: str | None = None,
) -> int:
    """Create/update a scoped alias row and return its id."""

    normalized = alias_lookup_key(raw_name)
    compact_source = source if not source_system and not league_pattern and not tournament_pattern else (
        f"{source}:{source_system or '*'}:{league_pattern or '*'}:{tournament_pattern or '*'}:{valid_from or '*'}:{valid_to or '*'}"
    )
    normalized_alias = normalize_team_name(target_name)
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO team_aliases(
                normalized_name, alias, source, raw_name, normalized_alias,
                source_system, league_pattern, tournament_pattern, valid_from,
                valid_to, confidence, is_active, is_blocked, review_status,
                notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 'approved', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(normalized_name, source) DO UPDATE SET
                alias = excluded.alias,
                raw_name = excluded.raw_name,
                normalized_alias = excluded.normalized_alias,
                source_system = excluded.source_system,
                league_pattern = excluded.league_pattern,
                tournament_pattern = excluded.tournament_pattern,
                valid_from = excluded.valid_from,
                valid_to = excluded.valid_to,
                confidence = excluded.confidence,
                is_active = 1,
                is_blocked = 0,
                review_status = 'approved',
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                normalized,
                target_name,
                compact_source,
                raw_name,
                normalized_alias,
                source_system,
                league_pattern,
                tournament_pattern,
                valid_from,
                valid_to,
                confidence,
                notes,
            ),
        )
        row = connection.execute(
            "SELECT id FROM team_aliases WHERE normalized_name = ? AND source = ?",
            (normalized, compact_source),
        ).fetchone()
        return int(row["id"])
