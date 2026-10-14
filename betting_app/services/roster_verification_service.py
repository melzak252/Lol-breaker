"""Roster verification and synchronization service using LoL Fandom (Leaguepedia) and Liquipedia.

Verifies active 5-man rosters for upcoming teams, identifies lineup changes/substitutions,
and updates team_current_roster_players so prediction and rating pipelines operate on verified rosters.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
import html
import json
import logging
import re
import time
from typing import Any, Sequence
import urllib.parse
import urllib.request

from sqlalchemy import text
from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.core.matching import normalize_team_name
from betting_app.services.current_roster_service import (
    clean_player_name, lock_current_roster_teams, upsert_current_roster,
)
from betting_app.services.liquipedia_service import LiquipediaClient

logger = logging.getLogger(__name__)

FANDOM_CARGO_EXPORT_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ROLE_NORM_MAP: dict[str, str] = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "jungler": "JUNGLE",
    "mid": "MID",
    "middle": "MID",
    "bot": "ADC",
    "bottom": "ADC",
    "ad": "ADC",
    "adc": "ADC",
    "support": "SUPPORT",
    "sup": "SUPPORT",
}


def normalize_role(role: str | None) -> str | None:
    """Normalize a raw role string to standard TOP/JUNGLE/MID/ADC/SUPPORT."""
    if not role:
        return None
    cleaned = str(role).strip().lower()
    if cleaned in ROLE_NORM_MAP:
        return ROLE_NORM_MAP[cleaned]
    for key, norm in ROLE_NORM_MAP.items():
        if key in cleaned:
            return norm
    return None
EXPECTED_ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


@dataclass
class VerifiedRosterResult:
    team_name: str
    normalized_team_name: str = ""
    source: str = "auto"
    players: list[dict[str, str]] = field(default_factory=list)
    source_match_date: str | None = None
    notes: str | None = None

    def __post_init__(self):
        if not self.normalized_team_name and self.team_name:
            self.normalized_team_name = normalize_team_name(self.team_name)

class FandomRosterClient:
    """Client for querying Leaguepedia / LoL Fandom Cargo export tables."""

    def __init__(
        self,
        export_url: str = FANDOM_CARGO_EXPORT_URL,
        user_agent: str = DEFAULT_BROWSER_UA,
        timeout: int = 12,
    ):
        self.export_url = export_url
        self.user_agent = user_agent
        self.timeout = timeout
    normalize_role = staticmethod(normalize_role)


    def query_cargo(
        self,
        table: str,
        fields: str,
        where: str | None = None,
        limit: int = 50,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query a Cargo table via Special:CargoExport."""
        params: dict[str, str] = {
            "tables": table,
            "fields": fields,
            "format": "json",
            "limit": str(limit),
        }
        if where:
            params["where"] = where
        if order_by:
            params["order_by"] = order_by

        url = f"{self.export_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
                return json.loads(content)
        except Exception as e:
            logger.warning("LoL Fandom Cargo query failed for table %s: %s", table, e)
            return []
    _query_cargo = query_cargo


    def resolve_team(self, team_name: str) -> tuple[str, str] | None:
        """Resolve canonical team name and OverviewPage on Fandom.

        Returns (canonical_name, overview_page) or None if not found.
        """
        clean = team_name.strip()
        # 1. Direct match on Name, Short, or OverviewPage
        where_exact = f'Name="{clean}" OR Short="{clean}" OR OverviewPage="{clean}"'
        teams = self.query_cargo("Teams", "OverviewPage,Name,Short", where_exact, limit=3)
        if teams:
            return str(teams[0]["Name"]), str(teams[0]["OverviewPage"])

        # 2. Academy/Challengers expansion
        aliases_to_try: list[str] = []
        if "challengers" in clean.lower():
            base = re.sub(r"(?i)\s+challengers", "", clean).strip()
            aliases_to_try.extend([
                f"{base} Esports Academy",
                f"{base} Academy",
                f"{base} Challengers",
            ])
        elif "academy" in clean.lower():
            base = re.sub(r"(?i)\s+academy", "", clean).strip()
            aliases_to_try.extend([
                f"{base} Esports Academy",
                f"{base} Academy",
            ])
        elif clean.lower() == "los heretics":
            aliases_to_try.append("Team Heretics Academy")

        for alias in aliases_to_try:
            where_alias = f'Name="{alias}" OR OverviewPage="{alias}"'
            teams = self.query_cargo("Teams", "OverviewPage,Name,Short", where_alias, limit=3)
            if teams:
                return str(teams[0]["Name"]), str(teams[0]["OverviewPage"])

        # 3. Wildcard LIKE query
        escaped_clean = clean.replace('"', '\\"')
        where_like = f'Name LIKE "%{escaped_clean}%" OR OverviewPage LIKE "%{escaped_clean}%"'
        teams = self.query_cargo("Teams", "OverviewPage,Name,Short", where_like, limit=5)
        if teams:
            # Pick non-academy unless the search specifically requested academy
            is_looking_for_academy = any(w in clean.lower() for w in ("academy", "challengers", "fenix"))
            if not is_looking_for_academy:
                main_teams = [
                    t for t in teams
                    if not any(w in str(t.get("Name", "")).lower() for w in ("academy", "challengers", "junior", "rookie"))
                ]
                if main_teams:
                    return str(main_teams[0]["Name"]), str(main_teams[0]["OverviewPage"])
            return str(teams[0]["Name"]), str(teams[0]["OverviewPage"])

        return None

    def fetch_active_roster(self, team_name: str) -> VerifiedRosterResult | None:
        """Fetch active starting roster of 5 players from LoL Fandom Cargo."""
        resolved = self.resolve_team(team_name)
        if not resolved:
            logger.info("Could not resolve team '%s' on Fandom", team_name)
            return None

        canonical_name, overview_page = resolved

        # Query active players
        where_players = (
            f'(Team="{overview_page}" OR Team="{canonical_name}") AND '
            f'Role IN ("Top", "Jungle", "Mid", "Bot", "ADC", "Support")'
        )
        players_data = self.query_cargo("Players", "ID,Name,Team,Role,IsSubstitute", where_players, limit=40)
        if not players_data:
            return None

        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in players_data:
            r_raw = str(p.get("Role") or "").lower()
            norm = ROLE_NORM_MAP.get(r_raw)
            if norm:
                by_role[norm].append(p)

        # Disambiguate if multiple players share the same role
        need_scoreboard = any(len(candidates) > 1 for candidates in by_role.values())
        recent_starters: dict[str, str] = {}
        latest_match_date: str | None = None

        if need_scoreboard or len(by_role) < 5:
            # Check recent scoreboard matches
            where_sb = f'Team="{overview_page}" OR Team="{canonical_name}"'
            scoreboard = self.query_cargo(
                "ScoreboardPlayers",
                "Name,Role,Team,DateTime_UTC",
                where=where_sb,
                order_by="DateTime_UTC DESC",
                limit=30,
            )
            for s in scoreboard:
                r_raw = str(s.get("Role") or "").lower()
                r_norm = ROLE_NORM_MAP.get(r_raw)
                if r_norm and r_norm not in recent_starters:
                    recent_starters[r_norm] = str(s.get("Name") or "")
                if not latest_match_date and s.get("DateTime UTC"):
                    latest_match_date = str(s.get("DateTime UTC"))

        starting_five: list[dict[str, str]] = []
        for role in EXPECTED_ROLES:
            candidates = by_role.get(role, [])
            if not candidates:
                if role in recent_starters:
                    starter_handle = clean_player_name(recent_starters[role])
                    starting_five.append({
                        "player_id": starter_handle,
                        "player_name": starter_handle,
                        "role": role,
                    })
                continue

            if len(candidates) == 1:
                handle = clean_player_name(candidates[0].get("ID") or candidates[0].get("Name") or "")
                starting_five.append({
                    "player_id": handle,
                    "player_name": handle,
                    "role": role,
                })
            else:
                # Disambiguation: match recent starter from scoreboard
                chosen = None
                if role in recent_starters:
                    st_name = recent_starters[role].lower()
                    for c in candidates:
                        cid = str(c.get("ID") or "").lower()
                        cname = str(c.get("Name") or "").lower()
                        if cid == st_name or cname == st_name:
                            chosen = c
                            break
                if not chosen:
                    # Prefer IsSubstitute == 0 or non-sub
                    non_subs = [c for c in candidates if c.get("IsSubstitute") == 0]
                    chosen = non_subs[0] if non_subs else candidates[0]

                chosen_handle = clean_player_name(chosen.get("ID") or chosen.get("Name") or "")
                starting_five.append({
                    "player_id": chosen_handle,
                    "player_name": chosen_handle,
                    "role": role,
                })

        if len(starting_five) != 5 or len({p["role"] for p in starting_five}) != 5:
            logger.info("Fandom roster for '%s' incomplete: got %d roles", team_name, len(starting_five))
            return None

        return VerifiedRosterResult(
            team_name=canonical_name,
            normalized_team_name=normalize_team_name(team_name),
            source="fandom",
            players=starting_five,
            source_match_date=latest_match_date,
            notes=f"Resolved via LoL Fandom Cargo ({overview_page})",
        )
class LiquipediaRosterAdapter:
    """Adapter wrapping LiquipediaClient for active roster fetching."""

    def __init__(self, client: LiquipediaClient | None = None) -> None:
        self.client = client or LiquipediaClient()

    def fetch_active_roster(self, team_name: str) -> VerifiedRosterResult | None:
        try:
            players = self.client.fetch_active_roster(team_name)
            if not players:
                return None

            by_role: dict[str, Any] = {}
            for p in players:
                if p.role not in by_role:
                    by_role[p.role] = p

            if len(by_role) != 5:
                return None

            formatted = [
                {
                    "player_id": clean_player_name(str(p.player_id or p.player_name)),
                    "player_name": clean_player_name(str(p.player_id or p.player_name)),
                    "role": str(role),
                }
                for role, p in by_role.items()
            ]

            return VerifiedRosterResult(
                team_name=team_name,
                normalized_team_name=normalize_team_name(team_name),
                source="liquipedia",
                players=formatted,
                notes="Resolved via Liquipedia API",
            )
        except Exception as e:
            logger.warning("Liquipedia roster fetch failed for '%s': %s", team_name, e)
            return None


class RosterVerificationService:
    """Service to verify and synchronize team rosters against Fandom and Liquipedia."""

    def __init__(
        self,
        fandom_client: FandomRosterClient | None = None,
        liquipedia_client: LiquipediaRosterAdapter | None = None,
    ):
        self.fandom = fandom_client or FandomRosterClient()
        self.liquipedia = liquipedia_client or LiquipediaRosterAdapter()

    def get_stored_roster(self, session: Session, team_name: str) -> list[dict[str, Any]]:
        """Retrieve the currently stored roster for a normalized team name."""
        normalized = normalize_team_name(team_name)
        try:
            rows = session.execute(
                text("""
                    SELECT player_id, player_name, role, source, source_match_date, updated_at
                    FROM team_current_roster_players
                    WHERE normalized_team_name = :normalized
                    ORDER BY CASE role
                        WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3
                        WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 9 END
                """),
                {"normalized": normalized},
            ).mappings().all()
            return [dict(r) for r in rows]
        except Exception as e:
            if session.get_bind().dialect.name != "sqlite":
                raise
            logger.debug("Could not query team_current_roster_players (table may not exist): %s", e)
            return []
    @staticmethod
    def compare_rosters(
        stored_players: list[dict[str, Any]],
        new_players: list[dict[str, str]],
    ) -> tuple[bool, list[dict[str, str]]]:
        """Compare stored roster against freshly fetched roster.

        Returns (is_different, changes_list).
        """
        stored_by_role = {str(p["role"]).upper(): str(p["player_id"]).lower() for p in stored_players}
        stored_names = {str(p["role"]).upper(): str(p.get("player_name") or p["player_id"]) for p in stored_players}
        new_by_role = {str(p["role"]).upper(): str(p["player_id"]).lower() for p in new_players}
        new_names = {str(p["role"]).upper(): str(p.get("player_name") or p["player_id"]) for p in new_players}

        changes: list[dict[str, str]] = []
        is_different = False

        for role in EXPECTED_ROLES:
            old_id = stored_by_role.get(role)
            new_id = new_by_role.get(role)
            old_name = stored_names.get(role, "<missing>")
            new_name = new_names.get(role, "<missing>")

            if old_id != new_id:
                is_different = True
                changes.append({
                    "role": role,
                    "old_player": old_name,
                    "new_player": new_name,
                    "change_type": "substituted" if old_id else "added",
                })

        return is_different, changes

    def verify_team(
        self,
        team_name: str,
        session: Session,
        source: str = "auto",  # "fandom" | "liquipedia" | "auto"
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Verify a single team roster against online sources."""
        stored = self.get_stored_roster(session, team_name)

        fetched: VerifiedRosterResult | None = None
        # Try Fandom first if auto or fandom
        if source in ("auto", "fandom"):
            fetched = self.fandom.fetch_active_roster(team_name)

        # If Fandom failed or source was explicitly liquipedia, try Liquipedia
        if not fetched and source in ("auto", "liquipedia"):
            fetched = self.liquipedia.fetch_active_roster(team_name)

        if not fetched:
            return {
                "team_name": team_name,
                "status": "failed",
                "reason": (
                    str(self.liquipedia.client.last_error)
                    if source in ("auto", "liquipedia") and self.liquipedia.client.last_error
                    else "Could not fetch 5-player active roster from source"
                ),
                "stored_players": stored,
                "changes": [],
            }
        if not dry_run:
            # Re-read after serialization; a no-change result must not retain
            # this team's transaction lock into the next verification.
            lock_current_roster_teams(session, [team_name])
            stored = self.get_stored_roster(session, team_name)

        is_different, changes = self.compare_rosters(stored, fetched.players)

        if not is_different and not force:
            if not dry_run:
                session.commit()
            return {
                "team_name": team_name,
                "status": "up_to_date",
                "source": fetched.source,
                "players": fetched.players,
                "changes": [],
            }

        # Update required
        if not dry_run:
            upsert_current_roster(
                session,
                team_name=team_name,
                players=fetched.players,
                source=fetched.source,
                source_match_date=fetched.source_match_date or datetime.now(UTC).isoformat(),
                force=force,
            )
            session.commit()

        return {
            "team_name": team_name,
            "status": "updated" if not dry_run else "would_update",
            "source": fetched.source,
            "players": fetched.players,
            "changes": changes,
            "notes": fetched.notes,
        }

    def verify_team_roster(
        self,
        session: Session,
        team_name: str,
        source: str = "auto",
        force: bool = False,
        dry_run: bool = False,
    ) -> tuple[str, list[dict[str, str]]]:
        """Convenience wrapper returning (status, changes)."""
        res = self.verify_team(team_name=team_name, session=session, source=source, force=force, dry_run=dry_run)
        return res["status"], res["changes"]

    def verify_and_sync_rosters(
        self,
        team_names: Sequence[str] | None = None,
        session: Session | None = None,
        source: str = "auto",
        force: bool = False,
        dry_run: bool = False,
        limit: int = 50,
        delay_between_requests: float = 0.25,
    ) -> dict[str, Any]:
        """Verify rosters for upcoming matches or a specific list of teams."""
        own_session = session is None
        sess = session or get_session()

        target_teams = list(dict.fromkeys(
            team.strip() for team in (team_names or ()) if team.strip()
        ))

        try:
            if not team_names:
                # Query distinct upcoming match teams
                rows = sess.execute(
                    text("""
                        SELECT DISTINCT team_a_name, team_b_name
                        FROM canonical_matches
                        WHERE status = 'upcoming'
                        ORDER BY team_a_name
                        LIMIT :limit
                    """),
                    {"limit": limit},
                ).mappings().all()

                found_teams: set[str] = set()
                for r in rows:
                    if r.get("team_a_name"):
                        found_teams.add(str(r["team_a_name"]))
                    if r.get("team_b_name"):
                        found_teams.add(str(r["team_b_name"]))
                target_teams = sorted(found_teams)[:limit]

            total = len(target_teams)
            updated: list[dict[str, Any]] = []
            up_to_date: list[str] = []
            failed: list[dict[str, Any]] = []

            logger.info("Starting roster verification for %d teams (source=%s, dry_run=%s)", total, source, dry_run)

            for idx, team in enumerate(target_teams):
                if idx > 0 and delay_between_requests > 0:
                    time.sleep(delay_between_requests)

                try:
                    res = self.verify_team(
                        team,
                        sess,
                        source=source,
                        force=force,
                        dry_run=dry_run,
                    )
                    status = res.get("status")
                    if status in ("updated", "would_update"):
                        updated.append(res)
                    elif status == "up_to_date":
                        up_to_date.append(team)
                    else:
                        failed.append(res)
                except Exception as e:
                    sess.rollback()
                    logger.error("Error verifying team '%s': %s", team, e)
                    failed.append({"team_name": team, "status": "error", "reason": str(e)})

            logger.info(
                "Roster verification complete: %d total, %d updated, %d up-to-date, %d failed",
                total, len(updated), len(up_to_date), len(failed)
            )

            return {
                "total_teams": total,
                "updated_count": len(updated),
                "up_to_date_count": len(up_to_date),
                "failed_count": len(failed),
                "source": source,
                "dry_run": dry_run,
                "updated": updated,
                "up_to_date": up_to_date,
                "failed": failed,
            }
        finally:
            if own_session:
                sess.close()
