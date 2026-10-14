"""Explicit tournament source synchronization and local bracket snapshots.

Normal reads and simulations do not fetch sources. Explicit synchronization
uses the shared Liquipedia cache/budget or the configured Fandom Cargo endpoint;
neither source is assumed unlimited or timestamp-certified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any
import urllib.parse
import urllib.request

from betting_app.core.matching import normalize_team_name
from betting_app.services.canonical_match_service import canonical_team_key
from betting_app.services.liquipedia_transport import LiquipediaRequestError, query as liquipedia_query
from betting_app.services.tournament_service import (
    SUPPORTED_BRACKETS,
    BracketMatchNode,
    TournamentBracket,
)

logger = logging.getLogger(__name__)

FANDOM_CARGO_EXPORT_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
LIQUIPEDIA_API_URL = "https://liquipedia.net/leagueoflegends/api.php"

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
LIQUIPEDIA_USER_AGENT = "EnsembleLegendsResearch/1.0 (academic research; melzacki.jakubl@gmail.com)"

CACHE_DIR = Path("data/cache/tournaments")
CACHE_TTL_SECONDS = 1800  # 30 minutes

TOURNAMENT_METADATA: dict[str, dict[str, Any]] = {
    "lck_2026_playoffs": {
        "name": "LCK 2026 Season - Playoffs",
        "liquipedia_page": "LCK/2026_Season/Playoffs",
        "fandom_overview": "LCK/2026 Season/Season Playoffs",
        "region": "LCK",
        "format": "double_elimination",
        "teams": ["Gen.G", "Hanwha Life Esports", "T1", "KT Rolster", "Dplus", "BNK FearX"],
        "round_order": [
            "UB_R1_M1", "UB_R1_M2", "UB_R2_M1", "UB_R2_M2",
            "LB_R1", "LB_R2", "UB_Final", "LB_R3", "LB_Final", "Grand_Final",
        ],
    },
    "lec_2026_summer_playoffs": {
        "name": "LEC 2026 Summer - Playoffs",
        "liquipedia_page": "LEC/2026_Season/Summer_Playoffs",
        "fandom_overview": "LEC/2026 Season/Summer Playoffs",
        "region": "LEC",
        "format": "double_elimination",
        "teams": ["Karmine Corp", "GIANTX", "G2 Esports", "Team Vitality", "Natus Vincere", "Movistar KOI"],
        "round_order": [
            "UB_SF1", "UB_SF2",
            "LB_R1_M1", "LB_R1_M2",
            "UB_Final", "LB_SF", "LB_Final", "Grand_Final",
        ],
    },
    "lpl_2026_split3_playoffs": {
        "name": "LPL 2026 Split 3 - Playoffs",
        "liquipedia_page": "LPL/2026_Season/Grand_Finals",
        "fandom_overview": "LPL/2026 Season/Grand Finals",
        "region": "LPL",
        "format": "double_elimination",
        "teams": [
            "Bilibili Gaming",
            "Anyone's Legend",
            "Team WE",
            "JD Gaming",
            "LGD Gaming",
            "Top Esports",
            "Invictus Gaming",
            "Ninjas in Pyjamas",
        ],
        "round_order": [
            "UB_R1_M1", "UB_R1_M2",
            "UB_R2_M1", "UB_R2_M2",
            "LB_R1_M1", "LB_R1_M2",
            "LB_R2_M1", "LB_R2_M2",
            "LB_R3",
            "UB_Final",
            "LB_Final",
            "Grand_Final",
        ],
    },
    "lcs_2026_championship": {
        "name": "LCS 2026 Summer - Playoffs",
        "liquipedia_page": "LCS/2026/Summer/Playoffs",
        "fandom_overview": "LCS/2026 Season/Summer Playoffs",
        "region": "LCS",
        "format": "double_elimination",
        "teams": ["Team Liquid", "LYON", "Cloud9", "FlyQuest", "Sentinels", "Shopify Rebellion"],
        "round_order": [
            "UB_R1_M1", "UB_R1_M2", "UB_R2_M1", "UB_R2_M2",
            "LB_R1", "LB_R2", "UB_Final", "LB_R3", "LB_Final", "Grand_Final",
        ],
    },
}


class LiquipediaBracketService:
    """Synchronizes live tournament brackets from LoL Fandom (Leaguepedia) and Liquipedia."""

    def __init__(
        self,
        fandom_url: str = FANDOM_CARGO_EXPORT_URL,
        liquipedia_url: str = LIQUIPEDIA_API_URL,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.fandom_url = fandom_url
        self.liquipedia_url = liquipedia_url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, tournament_id: str) -> Path:
        return self.cache_dir / f"{tournament_id}.json"

    def read_cached_bracket(self, tournament_id: str) -> dict[str, Any] | None:
        """Read cached bracket snapshot if it exists."""
        cache_file = self._get_cache_file(tournament_id)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cached bracket for %s: %s", tournament_id, e)
            return None

    def load_bracket(self, tournament_id: str) -> dict[str, Any]:
        """Load stored or curated state without making any external request."""
        if tournament_id not in SUPPORTED_BRACKETS:
            raise ValueError(f"Tournament '{tournament_id}' is not supported.")
        bracket = SUPPORTED_BRACKETS[tournament_id]()
        cached = self.read_cached_bracket(tournament_id)
        if cached is None:
            return {
                "ok": True, "bracket": bracket, "source": "curated",
                "status": "curated", "synced_at": None, "updated_matches": 0,
                "message": "Lokalna drabinka; dane źródłowe nie zostały pobrane.",
            }
        self._apply_cached_state(bracket, cached.get("matches", {}))
        fresh = False
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached["synced_at"])).total_seconds()
            fresh = 0 <= age < CACHE_TTL_SECONDS
        except (KeyError, TypeError, ValueError):
            pass
        return {
            "ok": True, "bracket": bracket, "source": cached.get("source", "cache"),
            "status": "cached" if fresh else "stale_cache",
            "synced_at": cached.get("synced_at"),
            "updated_matches": cached.get("updated_matches", 0),
            "message": "Załadowano lokalny stan drabinki bez pobierania danych.",
        }

    def write_cached_bracket(self, tournament_id: str, data: dict[str, Any]) -> None:
        """Save bracket snapshot to disk."""
        cache_file = self._get_cache_file(tournament_id)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to write cached bracket for %s: %s", tournament_id, e)

    def fetch_fandom_cargo_matches(self, overview_page: str, timeout: int = 12) -> list[dict[str, Any]]:
        """Fetch tournament match schedule and outcomes from LoL Fandom Cargo export."""
        params = {
            "tables": "MatchSchedule",
            "fields": "Team1,Team2,Winner,Team1Score,Team2Score,MatchDay,DateTime_UTC,BestOf,Tab",
            "where": f"OverviewPage = '{overview_page}'",
            "format": "json",
            "limit": "250",
            "order_by": "DateTime_UTC ASC",
        }
        url = f"{self.fandom_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parsed: list[dict[str, Any]] = []
                for m in data:
                    t1 = self.clean_team_name(m.get("Team1") or "")
                    t2 = self.clean_team_name(m.get("Team2") or "")
                    if not t1 or not t2 or t1 == "TBD" or t2 == "TBD":
                        continue
                    s1 = int(m["Team1Score"]) if m.get("Team1Score") is not None else None
                    s2 = int(m["Team2Score"]) if m.get("Team2Score") is not None else None
                    winner = None
                    if m.get("Winner") == 1 or m.get("Winner") == "1":
                        winner = t1
                    elif m.get("Winner") == 2 or m.get("Winner") == "2":
                        winner = t2
                    elif s1 is not None and s2 is not None:
                        if s1 > s2 and s1 >= 3:
                            winner = t1
                        elif s2 > s1 and s2 >= 3:
                            winner = t2

                    parsed.append({
                        "team1": t1,
                        "team2": t2,
                        "score1": s1,
                        "score2": s2,
                        "winner": winner,
                        "date": m.get("DateTime UTC"),
                        "tab": m.get("Tab"),
                    })
                return parsed
        except Exception as e:
            logger.warning("Fandom Cargo query failed for '%s': %s", overview_page, e)
            return []

    def fetch_liquipedia_page(self, page_title: str, timeout: int = 15) -> tuple[dict[str, Any] | None, str | None]:
        """Fetch parsed page text and wikitext from Liquipedia MediaWiki API."""
        params = {
            "action": "parse",
            "page": page_title.replace(" ", "_"),
            "prop": "text|wikitext",
            "format": "json",
        }
        try:
            data = liquipedia_query(
                self.liquipedia_url,
                params,
                LIQUIPEDIA_USER_AGENT,
                timeout=timeout,
            )
            if not isinstance(data, dict):
                raise LiquipediaRequestError("Liquipedia bracket API returned a non-object response", kind="response")
            return data, None
        except LiquipediaRequestError as error:
            err_msg = str(error)
            logger.warning("Liquipedia fetch failed for %s: %s", page_title, err_msg)
            return None, err_msg

    @staticmethod
    def clean_team_name(raw: str) -> str:
        """Strip wikitext formatting, flags, and tags from team strings."""
        if not raw:
            return ""
        text = re.sub(r"\{\{[^}]*\}\}", "", raw)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s*\([^)]*\)", "", text)
        text = " ".join(text.split())
        return text

    def parse_bracket_wikitext(self, wikitext: str) -> list[dict[str, Any]]:
        """Extract match results from Liquipedia Bracket wikitext template syntax."""
        match_pattern = re.compile(
            r"(?:r(?P<round>\d+)m(?P<match>\d+)|m(?P<match_alt>\d+))"
            r"(?P<key>team|score|win)(?P<slot>1|2)?\s*=\s*(?P<val>[^\n|]+)",
            re.IGNORECASE,
        )
        slots_by_match: dict[str, dict[str, Any]] = {}
        for m in match_pattern.finditer(wikitext):
            r_num = m.group("round") or "1"
            m_num = m.group("match") or m.group("match_alt") or "1"
            match_id = f"R{r_num}_M{m_num}"
            key = m.group("key").lower()
            slot = m.group("slot") or "1"
            val = m.group("val").strip()

            if match_id not in slots_by_match:
                slots_by_match[match_id] = {
                    "match_id": match_id,
                    "team1": None,
                    "team2": None,
                    "score1": None,
                    "score2": None,
                    "winner": None,
                }

            if key == "team":
                cleaned = self.clean_team_name(val)
                if slot == "1":
                    slots_by_match[match_id]["team1"] = cleaned
                else:
                    slots_by_match[match_id]["team2"] = cleaned
            elif key == "score":
                try:
                    score_val = int(val)
                    if slot == "1":
                        slots_by_match[match_id]["score1"] = score_val
                    else:
                        slots_by_match[match_id]["score2"] = score_val
                except ValueError:
                    pass
            elif key == "win" and val in ("1", "true", "yes"):
                slots_by_match[match_id]["winner_slot"] = int(slot)

        matches: list[dict[str, Any]] = []
        for _, data in slots_by_match.items():
            if data["team1"] and data["team2"]:
                if data.get("winner_slot") == 1:
                    data["winner"] = data["team1"]
                elif data.get("winner_slot") == 2:
                    data["winner"] = data["team2"]
                elif data["score1"] is not None and data["score2"] is not None:
                    if data["score1"] > data["score2"] and data["score1"] >= 3:
                        data["winner"] = data["team1"]
                    elif data["score2"] > data["score1"] and data["score2"] >= 3:
                        data["winner"] = data["team2"]
                matches.append(data)
        return matches

    def parse_bracket_html(self, html: str) -> list[dict[str, Any]]:
        """Parse match nodes, scores, and winners from Liquipedia Bracket HTML."""
        matches: list[dict[str, Any]] = []
        game_blocks = re.findall(
            r'(?:<div[^>]*class=[\"\'][^\"\']*bracket-(?:game|cell|match)[^\"\']*[\"\'][^>]*>|<table[^>]*class=[\"\'][^\"\']*bracket-match[^\"\']*[\"\'][^>]*>)(.*?)'
            r'(?=(?:<div[^>]*class=[\"\'][^\"\']*bracket-(?:game|cell|match)|<table[^>]*class=[\"\'][^\"\']*bracket-match|$))',
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        for block in game_blocks:
            team_names = re.findall(
                r'<span[^>]*class=[\"\'][^\"\']*(?:name|team-template-text|bracket-team)[^\"\']*[\"\'][^>]*>(.*?)</span>',
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            scores = re.findall(
                r'<span[^>]*class=[\"\'][^\"\']*(?:score|bracket-score)[^\"\']*[\"\'][^>]*>(\d+)</span>',
                block,
                flags=re.IGNORECASE,
            )
            cleaned_teams = [
                re.sub(r"<[^>]+>", "", t).strip() for t in team_names if re.sub(r"<[^>]+>", "", t).strip()
            ]
            cleaned_teams = [self.clean_team_name(t) for t in cleaned_teams if t and not t.isdigit()]

            if len(cleaned_teams) >= 2:
                t1, t2 = cleaned_teams[0], cleaned_teams[1]
                s1 = int(scores[0]) if len(scores) >= 1 else None
                s2 = int(scores[1]) if len(scores) >= 2 else None
                winner = None
                if s1 is not None and s2 is not None:
                    if s1 > s2 and s1 >= 3:
                        winner = t1
                    elif s2 > s1 and s2 >= 3:
                        winner = t2
                matches.append({
                    "team1": t1,
                    "team2": t2,
                    "score1": s1,
                    "score2": s2,
                    "winner": winner,
                })
        return matches

    def map_matches_chronologically(
        self,
        bracket: TournamentBracket,
        parsed_matches: list[dict[str, Any]],
        round_order: list[str],
    ) -> tuple[TournamentBracket, int]:
        """Apply parsed matches to bracket nodes in strict chronological / round progression order."""
        sorted_matches = sorted(parsed_matches, key=lambda x: x.get("date") or "")
        assigned_nodes: set[str] = set()
        display_names = {canonical_team_key(t): t for t in bracket.teams}
        updated_count = 0

        for p in sorted_matches:
            p_t1 = canonical_team_key(p["team1"])
            p_t2 = canonical_team_key(p["team2"])
            if p_t1 not in display_names or p_t2 not in display_names:
                continue
            p_winner_raw = p.get("winner")
            p_winner = canonical_team_key(p_winner_raw) if p_winner_raw else None
            p_s1 = p.get("score1")
            p_s2 = p.get("score2")
            if not p_winner and p_s1 is not None and p_s2 is not None:
                if p_s1 >= 3 and p_s1 > p_s2:
                    p_winner = p_t1
                    p_winner_raw = p["team1"]
                elif p_s2 >= 3 and p_s2 > p_s1:
                    p_winner = p_t2
                    p_winner_raw = p["team2"]

            matched_node_id: str | None = None
            for node_id in round_order:
                if node_id in assigned_nodes:
                    continue
                node = bracket.matches.get(node_id)
                if not node:
                    continue
                n_t1 = canonical_team_key(node.team1 or "")
                n_t2 = canonical_team_key(node.team2 or "")

                has_empty_slot = (not node.team1) or (not node.team2)
                if (n_t1 == p_t1 and n_t2 == p_t2) or (n_t1 == p_t2 and n_t2 == p_t1):
                    matched_node_id = node_id
                    break
                elif has_empty_slot and (node.winner is None):
                    known_key = n_t1 if node.team1 else n_t2
                    if known_key in (p_t1, p_t2):
                        matched_node_id = node_id
                        break
            if matched_node_id:
                assigned_nodes.add(matched_node_id)
                node = bracket.matches[matched_node_id]

                if not node.team1:
                    node.team1 = display_names.get(p_t1, p["team1"])
                if not node.team2:
                    node.team2 = display_names.get(p_t2, p["team2"])

                if p_winner:
                    winner_disp = (
                        node.team1 if canonical_team_key(node.team1 or "") == p_winner
                        else node.team2 if canonical_team_key(node.team2 or "") == p_winner
                        else display_names.get(p_winner, p_winner_raw)
                    )
                    node.winner = winner_disp

                    if node.next_match_winner_id and node.next_match_winner_id in bracket.matches:
                        target = bracket.matches[node.next_match_winner_id]
                        if node.next_match_winner_slot == 1:
                            target.team1 = winner_disp
                        elif node.next_match_winner_slot == 2:
                            target.team2 = winner_disp

                    loser_key = p_t2 if p_winner == p_t1 else p_t1
                    loser_disp = (
                        node.team2 if canonical_team_key(node.team1 or "") == p_winner
                        else node.team1 if canonical_team_key(node.team2 or "") == p_winner
                        else display_names.get(loser_key, p["team2"] if p_winner == p_t1 else p["team1"])
                    )
                    if node.next_match_loser_id and node.next_match_loser_id in bracket.matches:
                        loser_target = bracket.matches[node.next_match_loser_id]
                        if node.next_match_loser_slot == 1:
                            loser_target.team1 = loser_disp
                        elif node.next_match_loser_slot == 2:
                            loser_target.team2 = loser_disp

                if p_s1 is not None or p_s2 is not None:
                    n_k1 = canonical_team_key(node.team1 or "")
                    if n_k1 == p_t2:
                        node.score1 = p_s2
                        node.score2 = p_s1
                    else:
                        node.score1 = p_s1
                        node.score2 = p_s2
                updated_count += 1

        return bracket, updated_count

    def sync_bracket(
        self,
        tournament_id: str,
        source: str = "auto",  # "auto" | "fandom" | "liquipedia"
        raw_content: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Synchronize the bracket from LoL Fandom or Liquipedia, falling back cleanly to cache."""
        meta = TOURNAMENT_METADATA.get(tournament_id)
        if not meta or tournament_id not in SUPPORTED_BRACKETS:
            return {
                "ok": False,
                "error": f"Tournament '{tournament_id}' is not supported.",
                "source": "error",
            }

        bracket = SUPPORTED_BRACKETS[tournament_id]()
        now_utc = datetime.now(timezone.utc).isoformat()
        round_order = meta.get("round_order", list(bracket.matches.keys()))

        # Check existing cache if not forced
        cached = self.read_cached_bracket(tournament_id)
        if not force and cached and raw_content is None:
            cached_time = cached.get("synced_at")
            if cached_time:
                try:
                    dt = datetime.fromisoformat(cached_time)
                    age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
                    if age_seconds < CACHE_TTL_SECONDS:
                        self._apply_cached_state(bracket, cached.get("matches", {}))
                        return {
                            "ok": True,
                            "source": cached.get("source", "cache"),
                            "status": cached.get("status", "cached"),
                            "synced_at": cached_time,
                            "message": cached.get("message", "Załadowano aktualny stan z pamięci podręcznej."),
                            "bracket": bracket,
                            "updated_matches": cached.get("updated_matches", 0),
                        }
                except Exception as e:
                    logger.warning("Error parsing cached timestamp: %s", e)

        parsed_matches: list[dict[str, Any]] = []
        sync_source = "fandom_cargo"
        sync_status = "success"
        message = "Pomyślnie zsynchronizowano aktualny stan drabinki."

        if raw_content is not None:
            if "<" in raw_content:
                parsed_matches = self.parse_bracket_html(raw_content)
            else:
                parsed_matches = self.parse_bracket_wikitext(raw_content)
            sync_source = "liquipedia_manual_import"
            message = f"Zaimportowano stan wikitext/HTML ({len(parsed_matches)} meczów)."

        elif source in ("liquipedia", "auto"):
            # Try Liquipedia if explicitly requested or auto
            if source == "liquipedia":
                page_title = meta["liquipedia_page"]
                data, error = self.fetch_liquipedia_page(page_title)
                if data and "parse" in data:
                    parse_obj = data["parse"]
                    wikitext = parse_obj.get("wikitext", {}).get("*", "")
                    html = parse_obj.get("text", {}).get("*", "")
                    if wikitext:
                        parsed_matches = self.parse_bracket_wikitext(wikitext)
                    if not parsed_matches and html:
                        parsed_matches = self.parse_bracket_html(html)
                    sync_source = "liquipedia_api"
                    message = f"Zsynchronizowano stan z Liquipedia ({len(parsed_matches)} meczów)."
                else:
                    logger.info("Liquipedia sync failed (%s), falling back to LoL Fandom Cargo", error)
                    # Fallback to Fandom
                    parsed_matches = self.fetch_fandom_cargo_matches(meta["fandom_overview"])
                    sync_source = "fandom_cargo_fallback"
                    message = f"Liquipedia niedostępna ({error}); użyto LoL Fandom Cargo ({len(parsed_matches)} meczów)."

        if not parsed_matches and raw_content is None and source in ("fandom", "auto"):
            # Fetch from LoL Fandom Cargo
            parsed_matches = self.fetch_fandom_cargo_matches(meta["fandom_overview"])
            sync_source = "fandom_cargo"
            message = f"Pomyślnie zsynchronizowano z LoL Fandom Cargo ({len(parsed_matches)} meczów)."

        updated_count = 0
        if parsed_matches:
            bracket, updated_count = self.map_matches_chronologically(bracket, parsed_matches, round_order)
        else:
            local = self.load_bracket(tournament_id)
            local["ok"] = False
            local["status"] = "cached_fallback" if cached is not None else "unavailable"
            local["message"] = "Nie pobrano nowych danych. Zachowano lokalny stan i czas ostatniej synchronizacji."
            return local

        # Persist to disk cache
        cache_payload = {
            "tournament_id": tournament_id,
            "name": meta["name"],
            "source": sync_source,
            "status": sync_status,
            "synced_at": now_utc,
            "message": message,
            "updated_matches": updated_count,
            "matches": {
                m_id: {
                    "team1": m.team1,
                    "team2": m.team2,
                    "score1": m.score1,
                    "score2": m.score2,
                    "winner": m.winner,
                }
                for m_id, m in bracket.matches.items()
            },
        }
        self.write_cached_bracket(tournament_id, cache_payload)

        return {
            "ok": True,
            "source": sync_source,
            "status": sync_status,
            "synced_at": now_utc,
            "message": message,
            "bracket": bracket,
            "updated_matches": updated_count,
        }

    @staticmethod
    def _apply_cached_state(bracket: TournamentBracket, cached_matches: dict[str, Any]) -> None:
        """Apply stored match states to the bracket nodes."""
        for m_id, m_state in cached_matches.items():
            if m_id in bracket.matches:
                node = bracket.matches[m_id]
                if m_state.get("team1"):
                    node.team1 = m_state["team1"]
                if m_state.get("team2"):
                    node.team2 = m_state["team2"]
                if m_state.get("score1") is not None:
                    node.score1 = m_state["score1"]
                if m_state.get("score2") is not None:
                    node.score2 = m_state["score2"]
                if m_state.get("winner"):
                    node.winner = m_state["winner"]
