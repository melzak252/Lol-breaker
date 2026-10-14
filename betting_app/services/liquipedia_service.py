"""Client and synchronizer for fetching upcoming matches, Best-of format, and active team rosters from Liquipedia API."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from betting_app.core.db import connect, get_session, transaction
from betting_app.services.canonical_match_service import canonical_team_key
from betting_app.services.current_roster_service import clean_player_name, upsert_current_roster
from betting_app.services.liquipedia_transport import LiquipediaRequestError, query as liquipedia_query

logger = logging.getLogger(__name__)

LIQUIPEDIA_API_URL = "https://liquipedia.net/leagueoflegends/api.php"
DEFAULT_USER_AGENT = "EnsembleLegendsResearch/1.0 (academic research; melzacki.jakubl@gmail.com)"

ROLE_MAP = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "jg": "JUNGLE",
    "mid": "MID",
    "middle": "MID",
    "bot": "ADC",
    "ad": "ADC",
    "adc": "ADC",
    "support": "SUPPORT",
    "sup": "SUPPORT",
}


@dataclass(frozen=True)
class LiquipediaMatch:
    team1: str
    team2: str
    best_of: int | None
    tournament: str | None
    start_time: datetime | None


@dataclass(frozen=True)
class LiquipediaRosterPlayer:
    player_id: str  # Ingame nickname / handle (e.g. Faker)
    player_name: str | None  # Real name (e.g. Lee Sang-hyeok)
    role: str  # TOP, JUNGLE, MID, ADC, SUPPORT


class LiquipediaClient:
    """Client for querying the Liquipedia MediaWiki API."""

    def __init__(
        self,
        api_url: str = LIQUIPEDIA_API_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        wait_for_spacing: bool = False,
    ):
        self.api_url = api_url
        self.user_agent = user_agent
        self.wait_for_spacing = wait_for_spacing
        self.last_error: LiquipediaRequestError | None = None

    def _query_result(
        self, params: dict[str, Any], timeout: int = 15
    ) -> tuple[dict[str, Any] | list[Any] | None, LiquipediaRequestError | None]:
        self.last_error = None
        try:
            data = liquipedia_query(
                self.api_url,
                params,
                self.user_agent,
                timeout=timeout,
                wait_for_spacing=self.wait_for_spacing,
            )
            return data, None
        except LiquipediaRequestError as error:
            self.last_error = error
            logger.error("Liquipedia API request failed for %s: %s", params.get("action"), error)
            return None, error

    def _query(self, params: dict[str, Any], timeout: int = 15) -> dict[str, Any] | list[Any] | None:
        data, _error = self._query_result(params, timeout=timeout)
        return data

    def fetch_recent_and_upcoming_matches(self, limit: int = 50) -> list[LiquipediaMatch]:
        """Fetch matches parsed from the Match/Ticker/Container widget template."""
        params = {
            "action": "expandtemplates",
            "text": f"{{{{#invoke:Lua|invoke|module=Widget/Factory|fn=fromTemplate|widget=Match/Ticker/Container|limit={limit}}}}}",
            "prop": "wikitext",
            "format": "json",
        }
        data = self._query(params)
        if not isinstance(data, dict):
            return []
        html = data.get("expandtemplates", {}).get("wikitext", "")
        if not html:
            return []

        return self._parse_matches_html(html)

    def search_team_page(self, team_name: str) -> str | None:
        """Find the canonical Liquipedia page title for a team using opensearch."""
        params = {
            "action": "opensearch",
            "search": team_name,
            "limit": "5",
            "format": "json",
        }
        data = self._query(params)
        if not isinstance(data, list) or len(data) < 2:
            return None
        titles = data[1]
        if not isinstance(titles, list) or not titles:
            return None
        # Return first title that is not a subpage like /Results
        for title in titles:
            if isinstance(title, str) and "/" not in title:
                return title
        return titles[0] if isinstance(titles[0], str) else None

    def fetch_active_roster(self, team_page_or_name: str) -> list[LiquipediaRosterPlayer]:
        """Fetch and parse active 5-role roster from a team page on Liquipedia."""
        page_title = team_page_or_name.replace(" ", "_")
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
        }
        data, request_error = self._query_result(params)
        if request_error and request_error.kind == "missing_page":
            # Only an API/title-resolution miss may trigger a second query.
            resolved = self.search_team_page(team_page_or_name)
            if resolved and resolved != team_page_or_name:
                params["page"] = resolved.replace(" ", "_")
                data = self._query(params)

        if not data:
            return []

        if not isinstance(data, dict):
            return []

        html = data.get("parse", {}).get("text", {}).get("*", "")
        if not html:
            return []

        return self._parse_roster_html(html)

    def _parse_matches_html(self, html: str) -> list[LiquipediaMatch]:
        matches: list[LiquipediaMatch] = []
        chunks = re.split(r'<div[^>]*class=[\"\'][^\"\']*match-info[\"\'][^>]*>', html)

        for chunk in chunks[1:]:
            ts_m = re.search(r'data-timestamp=[\"\'](\d+)[\"\']', chunk)
            start_time = None
            if ts_m:
                try:
                    start_time = datetime.fromtimestamp(int(ts_m.group(1)), tz=UTC)
                except (ValueError, TypeError):
                    pass

            bo_m = re.search(r'\(Bo(\d)\)', chunk)
            best_of = int(bo_m.group(1)) if bo_m else None

            tourn_m = re.search(r'class=[\"\'][^\"\']*match-info-tournament[^\"\']*[\"\']?[^>]*>(.*?)</div>', chunk, flags=re.DOTALL)
            tourn_name = None
            if tourn_m:
                t_raw = tourn_m.group(1)
                t_raw = re.sub(r'<[^>]+>', '', t_raw)
                t_raw = re.sub(r'\[\[File:[^\]]+\]\]', '', t_raw)
                t_raw = re.sub(r'\[\[[^\|\]]+\|([^\]]+)\]\]', r'\1', t_raw)
                t_clean = t_raw.strip()
                if t_clean:
                    tourn_name = t_clean

            opp_left_m = re.search(r'class=[\"\'][^\"\']*match-info-header-opponent-left[^\"\']*[\"\']?[^>]*>(.*?)</div>', chunk, flags=re.DOTALL)
            opp_candidates = re.findall(r'class=[\"\'][^\"\']*match-info-header-opponent(?:\s+[^\"\']*)?[\"\']?[^>]*>(.*?)</div>', chunk, flags=re.DOTALL)

            opp_right_raw = None
            if len(opp_candidates) > 1:
                opp_right_raw = opp_candidates[1]

            if opp_left_m and opp_right_raw:
                t1_raw = re.sub(r'<[^>]+>', '', opp_left_m.group(1))
                t2_raw = re.sub(r'<[^>]+>', '', opp_right_raw)
                team1 = self._clean_team_text(t1_raw)
                team2 = self._clean_team_text(t2_raw)
                if team1 and team2:
                    matches.append(
                        LiquipediaMatch(
                            team1=team1,
                            team2=team2,
                            best_of=best_of,
                            tournament=tourn_name,
                            start_time=start_time,
                        )
                    )

        return matches

    def _parse_roster_html(self, html: str) -> list[LiquipediaRosterPlayer]:
        """Extract the active players from the first matching active roster table."""
        tables = re.findall(r'<table[^>]*>(.*?)</table>', html, flags=re.DOTALL)
        for t in tables:
            # Active roster tables have 'ID', 'Position' or 'Role', but NOT 'Leave Date'
            if ("Position" in t or "Role" in t) and "Leave Date" not in t:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.DOTALL)
                players: list[LiquipediaRosterPlayer] = []
                for r in rows[1:]:
                    cells = [
                        re.sub(r'&#\d+;', ' ', re.sub(r'<[^>]+>', '', c)).strip()
                        for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.DOTALL)
                    ]
                    if len(cells) >= 3:
                        handle = cells[0].strip()
                        real_name = cells[1].strip() or None
                        raw_pos = cells[2].lower().strip()
                        # Match standard roles (ignore Coach, Sub, Streamer)
                        matched_role = None
                        for key, std_role in ROLE_MAP.items():
                            if key in raw_pos:
                                matched_role = std_role
                                break
                        if matched_role and handle:
                            players.append(
                                LiquipediaRosterPlayer(
                                    player_id=handle,
                                    player_name=real_name,
                                    role=matched_role,
                                )
                            )
                if players:
                    return players
        return []

    @staticmethod
    def _clean_team_text(text: str) -> str:
        text = re.sub(r"\[\[File:[^\]]+\]\]", "", text)
        m = re.search(r"\[\[([^\|\]]+)(?:\|([^\]]+))?\]\]", text)
        if m:
            return m.group(1).strip()
        return text.strip()


def sync_liquipedia_best_of(limit: int = 50) -> dict[str, Any]:
    """Fetch recent/upcoming matches from Liquipedia and update canonical_matches.best_of when missing or divergent."""
    client = LiquipediaClient()
    lq_matches = client.fetch_recent_and_upcoming_matches(limit=limit)
    if not lq_matches:
        return {"fetched": 0, "matched": 0, "updated": 0}

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, normalized_team_a, normalized_team_b, best_of, start_time_normalized, status
            FROM canonical_matches
            WHERE status IN ('upcoming', 'live', 'finished', 'completed')
            ORDER BY id DESC
            LIMIT 500
            """
        ).fetchall()

    canonical_list = [dict(r) for r in rows]
    updated_count = 0
    matched_count = 0

    to_update: list[tuple[int, int]] = []

    for lq in lq_matches:
        if not lq.best_of:
            continue

        lq_k1 = canonical_team_key(lq.team1)
        lq_k2 = canonical_team_key(lq.team2)
        if not lq_k1 or not lq_k2:
            continue

        for cm in canonical_list:
            cm_k1 = canonical_team_key(cm.get("normalized_team_a") or "")
            cm_k2 = canonical_team_key(cm.get("normalized_team_b") or "")

            if (lq_k1 == cm_k1 and lq_k2 == cm_k2) or (lq_k1 == cm_k2 and lq_k2 == cm_k1):
                matched_count += 1
                if cm.get("best_of") != lq.best_of:
                    to_update.append((lq.best_of, int(cm["id"])))
                    cm["best_of"] = lq.best_of
                break

    if to_update:
        with transaction() as conn:
            for bo, cid in to_update:
                conn.execute(
                    "UPDATE canonical_matches SET best_of = ? WHERE id = ?",
                    (bo, cid),
                )
                updated_count += 1

    logger.info(
        "Liquipedia BoN sync: fetched=%d matched=%d updated=%d",
        len(lq_matches),
        matched_count,
        updated_count,
    )
    return {
        "fetched": len(lq_matches),
        "matched": matched_count,
        "updated": updated_count,
    }


def sync_liquipedia_team_rosters(
    team_names: Sequence[str] | None = None,
    *,
    wait_for_spacing: bool = False,
) -> dict[str, Any]:
    """Fetch active rosters for target teams from Liquipedia and update team_current_roster_players.

    If team_names is None, queries upcoming matches with incomplete rosters.
    """
    client = LiquipediaClient(wait_for_spacing=wait_for_spacing)
    session = get_session()
    updated_teams = 0
    failed_teams = 0

    target_teams = list(dict.fromkeys(name.strip() for name in (team_names or ()) if name.strip()))
    if not team_names:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT cm.team_a_name, cm.team_b_name
                FROM canonical_matches cm
                WHERE cm.status = 'upcoming'
                ORDER BY cm.team_a_name
                LIMIT 30
                """
            ).fetchall()
        for r in rows:
            if r.get("team_a_name"):
                target_teams.append(str(r["team_a_name"]))
            if r.get("team_b_name"):
                target_teams.append(str(r["team_b_name"]))
        target_teams = sorted(set(target_teams))

    try:
        for team_name in target_teams:
            roster_players = client.fetch_active_roster(team_name)
            if not roster_players:
                failed_teams += 1
                continue

            # Need 5 distinct roles to form a valid starting roster
            by_role: dict[str, LiquipediaRosterPlayer] = {}
            for p in roster_players:
                if p.role not in by_role:
                    by_role[p.role] = p

            if len(by_role) != 5:
                failed_teams += 1
                continue

            payload = [
                {
                    "player_id": clean_player_name(p.player_id or p.player_name),
                    "player_name": clean_player_name(p.player_id or p.player_name),
                    "role": role,
                }
                for role, p in by_role.items()
            ]

            changed = upsert_current_roster(
                session,
                team_name=team_name,
                players=payload,
                source="liquipedia",
            )
            if changed:
                updated_teams += 1
            session.commit()

        return {
            "total_teams": len(target_teams),
            "updated_teams": updated_teams,
            "failed_teams": failed_teams,
        }
    finally:
        session.close()
