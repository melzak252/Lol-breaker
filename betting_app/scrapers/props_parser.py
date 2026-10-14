"""Parser and normalizer for bookmaker in-game prop markets (IDEA-018).

Extracts and normalizes Over/Under total kills, team total kills, kill handicaps,
and game duration lines from Polish bookmaker payloads (STS, Betclic, Superbet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal

MarketType = Literal[
    "total_kills",
    "team_kills",
    "handicap_kills",
    "duration",
    "map_winner",
    "first_blood",
    "first_dragon",
    "first_baron",
    "first_tower",
    "total_maps",
    "map_handicap",
    "race_to_kills",
    "correct_score",
    "first_herald",
    "first_inhibitor",
    "total_dragons",
    "total_barons",
    "total_towers",
]

LINE_RE = re.compile(r"([+-]?\d+(?:[.,]\d+)?)")


def compute_novig_two_way(odds_1: float, odds_2: float) -> tuple[float, float, float]:
    """Compute no-vig fair probabilities and market margin for a two-way market."""
    if odds_1 <= 1.0 or odds_2 <= 1.0:
        return 0.5, 0.5, 0.0
    inv1 = 1.0 / odds_1
    inv2 = 1.0 / odds_2
    total_inv = inv1 + inv2
    margin = round(total_inv - 1.0, 4)
    prob_1 = round(inv1 / total_inv, 4)
    prob_2 = round(inv2 / total_inv, 4)
    return prob_1, prob_2, margin


@dataclass(frozen=True)
class ParsedPropLine:
    """A single normalized bookmaker proposition line."""

    market_type: MarketType
    line: float
    odds_over: float | None = None
    odds_under: float | None = None
    team: str | None = None
    handicap_team: str | None = None
    odds_cover_a: float | None = None
    odds_cover_b: float | None = None
    novig_prob_over: float | None = None
    novig_prob_under: float | None = None
    novig_prob_cover_a: float | None = None
    novig_prob_cover_b: float | None = None
    margin: float | None = None
    raw_market_name: str | None = None
    outcomes_payload: dict[str, Any] | None = None

@dataclass(frozen=True)
class ParsedMatchProps:
    """Aggregated prop lines for a match from a bookmaker."""

    bookmaker: str
    raw_team_a: str
    raw_team_b: str
    map_number: int = 1
    lines: list[ParsedPropLine] = field(default_factory=list)
    source_url: str | None = None

    def get_total_kills_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "total_kills"]

    def get_team_kills_lines(self, team: str | None = None) -> list[ParsedPropLine]:
        lines = [l for l in self.lines if l.market_type == "team_kills"]
        if team:
            t_norm = team.strip().lower()
            return [l for l in lines if l.team and l.team.strip().lower() == t_norm]
        return lines

    def get_handicap_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "handicap_kills"]

    def get_duration_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "duration"]

    def get_first_blood_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_blood"]

    def get_first_dragon_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_dragon"]

    def get_first_baron_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_baron"]

    def get_first_tower_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_tower"]

    def get_map_winner_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "map_winner"]

    def get_total_maps_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "total_maps"]

    def get_map_handicap_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "map_handicap"]
    def get_correct_score_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "correct_score"]

    def get_first_herald_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_herald"]

    def get_first_inhibitor_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "first_inhibitor"]

    def get_race_to_kills_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "race_to_kills"]

    def get_total_dragons_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "total_dragons"]

    def get_total_barons_lines(self) -> list[ParsedPropLine]:
        return [l for l in self.lines if l.market_type == "total_barons"]


def extract_line_number(text: str) -> float | None:
    """Extract decimal line number (e.g. '1. mapa - suma zabójstw (26.5)' -> 26.5)."""
    clean_text = re.sub(r"^\s*\d+\.\s*mapa\s*-\s*", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\bmapa\s*\d+\b", "", clean_text, flags=re.IGNORECASE)

    # 1. Number in parentheses: '(26.5)' or '(-4.5)'
    paren_match = re.search(r"\(([+-]?\d+(?:[.,]\d+)?)\)", clean_text)
    if paren_match:
        try:
            return float(paren_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # 2. Number preceded by keywords: 'Powyżej 26.5', 'Handicap -4.5'
    kw_match = re.search(r"(?:powyżej|poniżej|over|under|więcej|mniej|handicap)\s*([+-]?\d+(?:[.,]\d+)?)", clean_text, re.IGNORECASE)
    if kw_match:
        try:
            return float(kw_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # 3. Explicit decimal number: '28.5'
    dec_match = re.search(r"([+-]?\d+[.,]\d+)", clean_text)
    if dec_match:
        try:
            return float(dec_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # 4. Fallback: integer preceded by + or -
    sign_match = re.search(r"([+-]\d+)", clean_text)
    if sign_match:
        try:
            return float(sign_match.group(1))
        except ValueError:
            pass

    return None

def parse_sts_map_props(data: dict[str, Any]) -> ParsedMatchProps:
    """Parse STS map prop markets from event payload."""
    team_a = data.get("team_a", "")
    team_b = data.get("team_b", "")
    map_number = int(data.get("map_number", 1))
    markets = data.get("markets", [])

    parsed_lines: list[ParsedPropLine] = []

    for m in markets:
        name = str(m.get("name", "")).strip().lower()
        outcomes = m.get("outcomes", [])
        if len(outcomes) < 2:
            continue

        # Total kills (e.g. '1. mapa - suma zabójstw (26.5)')
        if "suma zabójstw" in name and "drużyn" not in name:
            line_val = m.get("line")
            if line_val is None:
                line_val = extract_line_number(name)
            if line_val is not None:
                o_over = None
                o_under = None
                for out in outcomes:
                    lbl = str(out.get("name", "")).lower()
                    odd = float(out.get("odds", 0.0))
                    if "powyżej" in lbl or "over" in lbl or "+" in lbl:
                        o_over = odd
                    elif "poniżej" in lbl or "under" in lbl or "-" in lbl:
                        o_under = odd
                if o_over and o_under:
                    p1, p2, margin = compute_novig_two_way(o_over, o_under)
                    parsed_lines.append(
                        ParsedPropLine(
                            market_type="total_kills",
                            line=float(line_val),
                            odds_over=o_over,
                            odds_under=o_under,
                            novig_prob_over=p1,
                            novig_prob_under=p2,
                            margin=margin,
                            raw_market_name=m.get("name"),
                        )
                    )

        # Team total kills (e.g. '1. mapa - liczba zabójstw drużyny 1 (14.5)')
        elif "liczba zabójstw drużyn" in name or "suma zabójstw drużyn" in name:
            line_val = m.get("line")
            if line_val is None:
                line_val = extract_line_number(name)
            team_target = team_a if "drużyny 1" in name or "team 1" in name else team_b
            if line_val is not None:
                o_over, o_under = None, None
                for out in outcomes:
                    lbl = str(out.get("name", "")).lower()
                    odd = float(out.get("odds", 0.0))
                    if "powyżej" in lbl or "over" in lbl:
                        o_over = odd
                    elif "poniżej" in lbl or "under" in lbl:
                        o_under = odd
                if o_over and o_under:
                    p1, p2, margin = compute_novig_two_way(o_over, o_under)
                    parsed_lines.append(
                        ParsedPropLine(
                            market_type="team_kills",
                            line=float(line_val),
                            odds_over=o_over,
                            odds_under=o_under,
                            team=team_target,
                            novig_prob_over=p1,
                            novig_prob_under=p2,
                            margin=margin,
                            raw_market_name=m.get("name"),
                        )
                    )

        # Handicap kills (e.g. '1. mapa - handicap zabójstw (-4.5)')
        elif "handicap zabójstw" in name:
            line_val = m.get("line")
            if line_val is None:
                line_val = extract_line_number(name)
            if line_val is not None:
                o_a, o_b = None, None
                for out in outcomes:
                    lbl = str(out.get("name", "")).lower()
                    odd = float(out.get("odds", 0.0))
                    if "1" in lbl or team_a.lower() in lbl:
                        o_a = odd
                    elif "2" in lbl or team_b.lower() in lbl:
                        o_b = odd
                if o_a and o_b:
                    p1, p2, margin = compute_novig_two_way(o_a, o_b)
                    parsed_lines.append(
                        ParsedPropLine(
                            market_type="handicap_kills",
                            line=float(line_val),
                            odds_cover_a=o_a,
                            odds_cover_b=o_b,
                            novig_prob_cover_a=p1,
                            novig_prob_cover_b=p2,
                            margin=margin,
                            raw_market_name=m.get("name"),
                        )
                    )

        # Duration (e.g. '1. mapa - czas trwania (32.5)')
        elif "czas trwania" in name:
            line_val = m.get("line")
            if line_val is None:
                line_val = extract_line_number(name)
            if line_val is not None:
                o_over, o_under = None, None
                for out in outcomes:
                    lbl = str(out.get("name", "")).lower()
                    odd = float(out.get("odds", 0.0))
                    if "powyżej" in lbl or "over" in lbl:
                        o_over = odd
                    elif "poniżej" in lbl or "under" in lbl:
                        o_under = odd
                if o_over and o_under:
                    p1, p2, margin = compute_novig_two_way(o_over, o_under)
                    parsed_lines.append(
                        ParsedPropLine(
                            market_type="duration",
                            line=float(line_val),
                            odds_over=o_over,
                            odds_under=o_under,
                            novig_prob_over=p1,
                            novig_prob_under=p2,
                            margin=margin,
                            raw_market_name=m.get("name"),
                        )
                    )

    return ParsedMatchProps(
        bookmaker="STS",
        raw_team_a=team_a,
        raw_team_b=team_b,
        map_number=map_number,
        lines=parsed_lines,
    )


def parse_betclic_map_props(data: dict[str, Any]) -> ParsedMatchProps:
    """Parse Betclic map prop markets from event payload."""
    team_a = data.get("raw_team_a", "")
    team_b = data.get("raw_team_b", "")
    map_number = int(data.get("map_number", 1))
    market_list = data.get("market_list", [])

    parsed_lines: list[ParsedPropLine] = []

    for item in market_list:
        title = str(item.get("title", "")).strip().lower()
        choices = item.get("choices", [])
        if len(choices) < 2:
            continue

        if "liczba zabójstw" in title and "drużyn" not in title:
            line = float(item.get("line", 0.0)) or extract_line_number(title)
            if line:
                o_over = float(choices[0].get("odds", 0.0))
                o_under = float(choices[1].get("odds", 0.0))
                p1, p2, margin = compute_novig_two_way(o_over, o_under)
                parsed_lines.append(
                    ParsedPropLine(
                        market_type="total_kills",
                        line=float(line),
                        odds_over=o_over,
                        odds_under=o_under,
                        novig_prob_over=p1,
                        novig_prob_under=p2,
                        margin=margin,
                        raw_market_name=item.get("title"),
                    )
                )
        elif "handicap" in title:
            line = float(item.get("line", 0.0)) or extract_line_number(title)
            if line is not None:
                o_a = float(choices[0].get("odds", 0.0))
                o_b = float(choices[1].get("odds", 0.0))
                p1, p2, margin = compute_novig_two_way(o_a, o_b)
                parsed_lines.append(
                    ParsedPropLine(
                        market_type="handicap_kills",
                        line=float(line),
                        odds_cover_a=o_a,
                        odds_cover_b=o_b,
                        novig_prob_cover_a=p1,
                        novig_prob_cover_b=p2,
                        margin=margin,
                        raw_market_name=item.get("title"),
                    )
                )

    return ParsedMatchProps(
        bookmaker="Betclic",
        raw_team_a=team_a,
        raw_team_b=team_b,
        map_number=map_number,
        lines=parsed_lines,
    )


def parse_superbet_map_props(data: dict[str, Any]) -> ParsedMatchProps:
    """Parse Superbet map prop markets from event payload."""
    team_a = data.get("match_name_a", "")
    team_b = data.get("match_name_b", "")
    map_number = int(data.get("map_index", 1))
    market_items = data.get("events", [])

    parsed_lines: list[ParsedPropLine] = []

    for ev in market_items:
        market_name = str(ev.get("name", "")).strip().lower()
        selections = ev.get("selections", [])
        if len(selections) < 2:
            continue

        if "suma zabójstw" in market_name:
            line = float(ev.get("line", 0.0)) or extract_line_number(market_name)
            if line:
                o_over = float(selections[0].get("price", 0.0))
                o_under = float(selections[1].get("price", 0.0))
                p1, p2, margin = compute_novig_two_way(o_over, o_under)
                parsed_lines.append(
                    ParsedPropLine(
                        market_type="total_kills",
                        line=float(line),
                        odds_over=o_over,
                        odds_under=o_under,
                        novig_prob_over=p1,
                        novig_prob_under=p2,
                        margin=margin,
                        raw_market_name=ev.get("name"),
                    )
                )

    return ParsedMatchProps(
        bookmaker="Superbet",
        raw_team_a=team_a,
        raw_team_b=team_b,
        map_number=map_number,
        lines=parsed_lines,
    )
