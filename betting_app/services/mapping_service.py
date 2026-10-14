"""Services for bookmaker-team to GOL.GG-team mapping."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

from betting_app.core.db import query_df, transaction
from betting_app.core.matching import normalize_team_name
from betting_app.core.config import PROJECT_ROOT
from betting_app.services.team_alias_service import (
    AliasContext,
    alias_lookup_key,
    is_short_alias,
    resolve_scoped_alias,
    upsert_scoped_alias,
)


BOOKMAKER_TO_GOLGG_ALIASES = {
    # BRION changed naming/sponsors across bookmakers and seasons.
    "brion": "BRION",
    "hanjin brion": "BRION",
    "oksavingsbank brion": "BRION",
    "ok savingsbank brion": "BRION",
    # Common short forms used by bookmakers.
    "9z": "9z Globant",
    "anyones legend": "Anyone's Legend",
    "arctic pandas": "Arctic Pandas",
    "barca": "Barca eSports",
    "barczaca": "Barczaca",
    "bilibili": "Bilibili",
    "bnk fearx": "FearX",
    "bomba": "BOMBA Team",
    "bubliki": "Bubliki Esports",
    "bulldog": "Bulldog Esports",
    "ccg": "CCG Esports",
    "conviction": "Conviction Esports",
    "dplus": "Dplus KIA",
    "dplus challengers": "Dplus KIA Challengers",
    "dplus kia challengers": "Dplus KIA Challengers",
    "dk challengers": "Dplus KIA Challengers",
    "dwg kia challengers": "Dplus KIA Challengers",
    "dplus kia": "Dplus KIA",
    "dk": "Dplus KIA",
    "deep cross": "Deep Cross Gaming",
    "dn soopers": "DN SOOPers",
    "dn soopers challengers": "DN SOOPers Challengers",
    "soopers": "DN SOOPers",
    "soopers challengers": "DN SOOPers Challengers",
    "kdf challengers": "DN SOOPers Challengers",
    "drx": "DRX",
    # DRX Challengers rebranded to KRX Challengers in GOL.GG (2026)
    "drx challengers": "KRX Challengers",
    "kiwoom drx challengers": "KRX Challengers",
    "krx challengers": "KRX Challengers",
    "drx academy": "KRX Challengers",
    "kiwoom drx academy": "KRX Challengers",
    "edward": "EDward Gaming",
    "eintracht spandau": "Eintracht Spandau",
    "esuba": "Esuba",
    "fc barcelona": "Barca eSports",
    "fearx": "FearX",
    "bnk fearx": "FearX",
    # FearX Youth / Challengers in GOL.GG (2026)
    "fearx challengers": "BNK FEARX Youth",
    "bnk fearx challengers": "BNK FEARX Youth",
    "fearx youth": "BNK FEARX Youth",
    "bnk fearx youth": "BNK FEARX Youth",
    "fearx academy": "BNK FEARX Youth",
    "bnk fearx academy": "BNK FEARX Youth",
    "fluxo w7m": "Fluxo W7M",
    "flyquest": "FlyQuest",
    "forsaken": "Forsaken",
    "g2 nord": "G2 Nord",
    "galions": "Galions",
    "gam": "GAM Esports",
    "gen g": "Gen.G",
    "gen g global academy": "Gen.G Global Academy",
    "geng global academy": "Gen.G Global Academy",
    "gen g challengers": "Gen.G Global Academy",
    "geng challengers": "Gen.G Global Academy",
    "geng": "Gen.G",
    "giantx": "GiantX",
    "gmblers": "GMBLERS",
    "hanwha life": "Hanwha Life Esports",
    "hanwha life challengers": "Hanwha Life Esports Challengers",
    "hanwhalife challengers": "Hanwha Life Esports Challengers",
    "hle challengers": "Hanwha Life Esports Challengers",
    "kt challengers": "KT Rolster Challengers",
    "kt rolster challengers": "KT Rolster Challengers",
    "ktrolster challengers": "KT Rolster Challengers",
    "kt academy": "KT Rolster Challengers",
    "kt rolster academy": "KT Rolster Challengers",
    "jd": "JD Gaming",
    "karmine corp": "Karmine Corp",
    "karmine corp blue": "Karmine Corp Blue",
    "kt rolster": "KT Rolster",
    # KT Rolster Challengers (active 2026) vs legacy KT Challengers
    "kt challengers": "KT Rolster Challengers",
    "kt rolster challengers": "KT Rolster Challengers",
    "kt academy": "KT Rolster Challengers",
    "kt rolster academy": "KT Rolster Challengers",
    "lgd": "LGD Gaming",
    "liquid": "Team Liquid",
    "los": "Los Grandes",
    "los grandes": "Los Grandes",
    "meavedron": "Meavedron",
    "mvk": "MVK",
    "natus vincere": "Natus Vincere",
    "nongshim redforce": "Nongshim RedForce",
    # Nongshim Esports Academy in GOL.GG (2026)
    "nongshim redforce challengers": "Nongshim Esports Academy",
    "nongshim challengers": "Nongshim Esports Academy",
    "ns challengers": "Nongshim Esports Academy",
    "nongshim academy": "Nongshim Esports Academy",
    "ns academy": "Nongshim Esports Academy",
    "nongshim esports academy": "Nongshim Esports Academy",
    "ns red force": "Nongshim RedForce",
    "ok savingsbank brion": "BRION",
    "oksavingsbank brion": "BRION",
    # HANJIN BRION Challengers in GOL.GG (2026)
    "brion challengers": "HANJIN BRION Challengers",
    "hanjin brion challengers": "HANJIN BRION Challengers",
    "ok brion challengers": "HANJIN BRION Challengers",
    "bro challengers": "HANJIN BRION Challengers",
    "ok savingsbank brion challengers": "HANJIN BRION Challengers",
    "red canids": "RED Canids",
    "red canids kalunga": "RED Canids",
    "ronaldo": "Ronaldo Team",
    "ronaldoteam": "Ronaldo Team",
    "secret whales": "Secret Whales",
    "sentinels": "Sentinels",
    "shopify rebellion": "Shopify Rebellion",
    "t1": "T1",
    # T1 Esports Academy in GOL.GG (2026) vs legacy T1 Challengers
    "t1 challengers": "T1 Esports Academy",
    "t1 academy": "T1 Esports Academy",
    "t1 esports academy": "T1 Esports Academy",
    "skt t1 challengers": "T1 Esports Academy",
    "skt t1 academy": "T1 Esports Academy",
    "tes": "Top Esports",
    "thundertalk": "ThunderTalk Gaming",
    "top": "Top Esports",
    "vitality": "Team Vitality",
    "we": "Weibo Gaming",
    "we love": "WLGaming Esports",
    "wlgaming": "WLGaming Esports",
    "ucam club": "UCAM Esports",
    "ucam ec": "UCAM Esports",
    "ucam tokiers": "UCAM Esports",
    "ucam": "UCAM Esports",
    "e wie einfach": "E WIE EINFACH E-SPORTS",
}

_FILE_CANDIDATES_CACHE: list[str] | None = None


def load_golgg_team_candidates() -> list[str]:
    """Load known team names from local GOL.GG data and existing aliases."""

    candidates: set[str] = set()
    try:
        db_teams = query_df("SELECT DISTINCT team_name FROM golgg_teams WHERE team_name IS NOT NULL")
    except Exception:
        db_teams = pd.DataFrame()
    for value in db_teams.get("team_name", []):
        if isinstance(value, str) and value.strip():
            candidates.add(value.strip())
    if candidates:
        alias_df = query_df("SELECT DISTINCT alias FROM team_aliases WHERE alias IS NOT NULL")
        for value in alias_df.get("alias", []):
            if isinstance(value, str) and value.strip():
                candidates.add(value.strip())
        return sorted(candidates)
    if os.getenv("PYTEST_CURRENT_TEST") or "_test" in os.getenv("DATABASE_URL", ""):
        alias_df = query_df("SELECT DISTINCT alias FROM team_aliases WHERE alias IS NOT NULL")
        for value in alias_df.get("alias", []):
            if isinstance(value, str) and value.strip():
                candidates.add(value.strip())
        return sorted(candidates)

    global _FILE_CANDIDATES_CACHE
    if _FILE_CANDIDATES_CACHE is not None:
        return list(_FILE_CANDIDATES_CACHE)

    matches_path = PROJECT_ROOT / "data" / "golgg_matches.json"
    if matches_path.exists():
        with matches_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        top_level_keys = (
            "team1",
            "team2",
            "team1_name",
            "team2_name",
            "sname_t1",
            "sname_t2",
            "won",
            "lost",
            "blue_team",
            "red_team",
        )
        game_level_keys = ("t1_name", "t2_name", "team1_name", "team2_name")
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in top_level_keys:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.add(value.strip())
            for game in row.get("games", []) or []:
                if not isinstance(game, dict):
                    continue
                for key in game_level_keys:
                    value = game.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.add(value.strip())
    alias_df = query_df("SELECT DISTINCT alias FROM team_aliases WHERE alias IS NOT NULL")
    for value in alias_df.get("alias", []):
        if isinstance(value, str) and value.strip():
            candidates.add(value.strip())
    _FILE_CANDIDATES_CACHE = sorted(candidates)
    return sorted(candidates)

def sync_golgg_teams() -> int:
    """Populate the local canonical team table from available GOL.GG data."""

    teams = load_golgg_team_candidates()
    with transaction() as connection:
        for team in teams:
            connection.execute(
                """
                INSERT INTO golgg_teams(team_name, normalized_name)
                VALUES (?, ?)
                ON CONFLICT(team_name) DO UPDATE SET
                    normalized_name = excluded.normalized_name
                """,
                (team, normalize_team_name(team)),
            )
    return len(teams)


def known_golgg_teams() -> pd.DataFrame:
    """Return known canonical team names."""

    return query_df("SELECT * FROM golgg_teams ORDER BY team_name")


def suggest_mapping(
    raw_name: str,
    *,
    source_system: str | None = None,
    league: str | None = None,
    tournament: str | None = None,
    match_date: str | None = None,
) -> tuple[str | None, float, str | None]:
    """Suggest a canonical GOL.GG team for a raw bookmaker name.

    Returns (golgg_name, confidence, source) where source is one of:
    'alias'  — matched via team_aliases table (confidence 1.0)
    'builtin' — matched via BOOKMAKER_TO_GOLGG_ALIASES dict or exact GOL.GG name (confidence 1.0)
    'blocked' — a blocked alias exists, mapping suppressed (None, 0.0)
    None — no mapping found

    Fuzzy auto-mapping is intentionally disabled. Incorrect fuzzy matches can
    poison ratings/features/predictions with another team's historical data, so
    unknown teams must now be mapped explicitly via aliases.
    """

    normalized = normalize_team_name(raw_name)

    scoped = resolve_scoped_alias(
        raw_name,
        context=AliasContext(
            source_system=source_system,
            league=league,
            tournament=tournament,
            match_date=match_date,
        ),
    )
    if scoped.blocked:
        return None, 0.0, "blocked"
    if scoped.target_name:
        return scoped.target_name, scoped.confidence, scoped.source or "alias"

    if is_short_alias(raw_name):
        # Do not let legacy global normalizer/builtin aliases resolve collision-
        # prone abbreviations without an applicable scoped DB alias.
        teams = known_golgg_teams()
        candidates = teams["team_name"].tolist() if not teams.empty else load_golgg_team_candidates()
        raw_key = alias_lookup_key(raw_name)
        for candidate in candidates:
            if alias_lookup_key(candidate) == raw_key:
                return candidate, 1.0, "exact"
        return None, 0.0, None

    # Check for blocked alias first
    blocked = query_df(
        "SELECT 1 FROM team_aliases WHERE normalized_name = ? AND alias = '' AND source = 'blocked' LIMIT 1",
        (normalized,),
    )
    if not blocked.empty:
        return None, 0.0, "blocked"

    confirmed = query_df(
        "SELECT alias FROM team_aliases WHERE normalized_name = ? AND alias IS NOT NULL LIMIT 1",
        (normalized,),
    )
    if not confirmed.empty:
        return str(confirmed.iloc[0]["alias"]), 1.0, "alias"

    alias_target = BOOKMAKER_TO_GOLGG_ALIASES.get(normalized) or BOOKMAKER_TO_GOLGG_ALIASES.get(
        normalized.replace(" ", "")
    )
    if alias_target:
        teams = known_golgg_teams()
        candidates = teams["team_name"].tolist() if not teams.empty else load_golgg_team_candidates()
        for candidate in candidates:
            if normalize_team_name(candidate) == normalize_team_name(alias_target):
                return candidate, 1.0, "builtin"
        return alias_target, 1.0, "builtin"

    teams = known_golgg_teams()
    candidates = teams["team_name"].tolist() if not teams.empty else load_golgg_team_candidates()
    for candidate in candidates:
        if normalize_team_name(candidate) == normalized:
            return candidate, 1.0, "builtin"

    return None, 0.0, None


def upsert_alias(
    raw_name: str,
    golgg_team_name: str,
    source: str = "manual",
    confirmed: bool = True,
    *,
    source_system: str | None = None,
    league_pattern: str | None = None,
    tournament_pattern: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    notes: str | None = None,
) -> int:
    """Create/update a raw-name alias mapping."""

    if source_system or league_pattern or tournament_pattern or valid_from or valid_to or notes:
        return upsert_scoped_alias(
            raw_name,
            golgg_team_name,
            source=source,
            source_system=source_system,
            league_pattern=league_pattern,
            tournament_pattern=tournament_pattern,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=1.0 if confirmed else 0.8,
            notes=notes,
        )

    normalized = normalize_team_name(raw_name)
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO team_aliases(normalized_name, alias, source)
            VALUES (?, ?, ?)
            ON CONFLICT(normalized_name, source) DO UPDATE SET
                alias = excluded.alias
            """,
            (normalized, golgg_team_name, source),
        )
        row = connection.execute(
            "SELECT id FROM team_aliases WHERE normalized_name = ? AND source = ?", (normalized, source)
        ).fetchone()
        return int(row["id"])


def delete_alias(raw_name: str, source: str = "manual") -> bool:
    """Delete a raw-name alias mapping. Returns True if a row was deleted."""
    normalized = normalize_team_name(raw_name)
    with transaction() as connection:
        cursor = connection.execute(
            "DELETE FROM team_aliases WHERE normalized_name = ? AND source = ?",
            (normalized, source),
        )
        return cursor.rowcount > 0


def block_alias(raw_name: str) -> int:
    """Mark a raw team name as blocked/unmapped.

    The alias column is set to empty string to indicate a block rather than a mapping.
    """
    normalized = normalize_team_name(raw_name)
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO team_aliases(normalized_name, alias, source)
            VALUES (?, '', 'blocked')
            ON CONFLICT(normalized_name, source) DO UPDATE SET
                alias = excluded.alias
            """,
            (normalized,),
        )
        row = connection.execute(
            "SELECT id FROM team_aliases WHERE normalized_name = ? AND source = 'blocked'",
            (normalized,),
        ).fetchone()
        return int(row["id"])


def unblock_alias(raw_name: str) -> bool:
    """Remove a blocked alias entry. Returns True if a row was deleted."""
    normalized = normalize_team_name(raw_name)
    with transaction() as connection:
        cursor = connection.execute(
            "DELETE FROM team_aliases WHERE normalized_name = ? AND source = 'blocked'",
            (normalized,),
        )
        return cursor.rowcount > 0


def golgg_name_from_id(golgg_id: int | None) -> str | None:
    """Look up GOL.GG team name by internal database ID."""
    if golgg_id is None:
        return None
    golgg_id = int(golgg_id)  # ensure plain Python int (not numpy.int64)
    with transaction() as connection:
        row = connection.execute(
            "SELECT team_name FROM golgg_teams WHERE id = ?",
            (golgg_id,),
        ).fetchone()
        if row:
            return str(row["team_name"])
    return None


def native_golgg_team_id(
    internal_team_row_id: int | str | None,
    *,
    team_name: str | None = None,
) -> str | None:
    """Resolve a provider team ID without confusing it with our DB row ID.

    ``golgg_teams.id`` is only the local relational key.  The native ID comes
    from ``golgg_teams.team_id`` or, when absent, one unambiguous exact-name
    current-roster source ID.  Names are compared exactly (case-insensitively),
    never through fuzzy aliases.
    """
    provider_id = None
    exact_name = team_name.strip() if isinstance(team_name, str) else ""
    if internal_team_row_id is not None:
        try:
            internal_id = int(internal_team_row_id)
        except (TypeError, ValueError) as error:
            raise ValueError("GOL.GG database team row ID must be an integer.") from error
        with transaction() as connection:
            row = connection.execute(
                "SELECT team_name, team_id FROM golgg_teams WHERE id = ?",
                (internal_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown GOL.GG database team row ID {internal_id}.")
        stored_name = str(row["team_name"] or "").strip()
        if exact_name and stored_name.casefold() != exact_name.casefold():
            raise ValueError(
                f"GOL.GG database team row {internal_id} does not identify {exact_name!r}."
            )
        exact_name = stored_name
        if row["team_id"] not in (None, ""):
            provider_id = str(row["team_id"]).strip()
    if provider_id:
        return provider_id
    if not exact_name:
        return None
    with transaction() as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT team_id
            FROM team_current_roster_players
            WHERE LOWER(TRIM(team_name)) = LOWER(TRIM(?))
              AND team_id IS NOT NULL AND TRIM(team_id) <> ''
            """,
            (exact_name,),
        ).fetchall()
    source_ids = {str(row["team_id"]).strip() for row in rows if row["team_id"]}
    return next(iter(source_ids)) if len(source_ids) == 1 else None


def unmapped_raw_teams() -> pd.DataFrame:
    """Return raw bookmaker names without confirmed canonical mapping."""

    return query_df(
        """
        WITH raw_names AS (
            SELECT raw_team_a AS raw_name FROM odds_snapshots
            UNION
            SELECT raw_team_b AS raw_name FROM odds_snapshots
        )
        SELECT raw_name
        FROM raw_names
        WHERE lower(trim(raw_name)) NOT IN (
            SELECT lower(trim(normalized_name)) FROM team_aliases
        )
        ORDER BY raw_name
        """
    )
