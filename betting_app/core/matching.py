"""Team-name normalization and fuzzy matching."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


STOP_WORDS = {"esports", "esport", "gaming", "team", "lol", "leagueoflegends"}

# Known alias mappings: bookmaker name -> canonical short form.
# Applied after stop-word removal to collapse common name variants.
ALIASES: dict[str, str] = {
    # German Prime League abbreviations
    "e wie einfach": "ewieeinfach",
    "e wie einfach e sports": "ewieeinfach",
    "ewieeinfach e sports": "ewieeinfach",
    "unicorns of love sexy edition": "unicornsoflovese",
    "unicorns of love se": "unicornsoflovese",
    "uol sexy edition": "unicornsoflovese",
    "kaufland hangry knights": "hangryknights",
    "hangry knights": "hangryknights",
    "eintracht frankfurt": "frankfurt",
    "frankfurt": "frankfurt",
    "vfb esports": "vfbesports",
    "vfb esport": "vfbesports",
    "vfbesports": "vfbesports",
    "teamorangegaming": "teamorangegaming",
    "team orange gaming": "teamorangegaming",
    "big": "big",
    "berlin international gaming": "big",
    "berlin international": "big",
    "schalke 04": "schalke04",
    "schalke04 evolution": "schalke04",
    "fc schalke 04": "schalke04",
    # LCK / LCK CL
    "t1": "t1",
    "skt t1": "t1",
    "sktelecom t1": "t1",
    "dwg kia": "damwonkia",
    "damwon kia": "damwonkia",
    "dplus kia": "damwonkia",
    "dplus": "damwonkia",
    "hanwha life esport": "hanwhalife",
    "hanwha life esports": "hanwhalife",
    "hanwha life": "hanwhalife",
    "liiv sandbox": "liivsandbox",
    "sandbox": "liivsandbox",
    "kt rolster": "ktrolster",
    "kt": "ktrolster",
    "gen g": "geng",
    "geng": "geng",
    "generation gaming": "geng",
    # Common suffix/prefix variants
    "secret whales": "secretwhales",
    "team secret whales": "secretwhales",
    "top esport": "topesports",
    "top esports": "topesports",
    "bilibili gaming": "bilibiligaming",
    "bilibili": "bilibiligaming",
    "team we": "we",
    "we": "we",
    "world elite": "we",
    "sdm tigres": "sdmtigres",
    "sdm": "sdmtigres",
    "lyon academy": "lyonacademy",
    "lyon": "lyon",
    "lyon gaming": "lyon",
    "mcon": "mcon",
    "mcon esports": "mcon",
    "the bandits": "thebandits",
    "bandits": "thebandits",
    "deep cross gaming": "deepcrossgaming",
    "deep cross": "deepcrossgaming",
    "misfits gaming": "misfits",
    "misfits": "misfits",
    "g2 esports": "g2",
    "g2": "g2",
    "g2 nord": "g2nord",
    "g2nord": "g2nord",
    # Additional common LoL orgs
    "fnatic": "fnatic",
    "fnc": "fnatic",
    "cloud9": "cloud9",
    "c9": "cloud9",
    "team liquid": "teamliquid",
    "teamliquid": "teamliquid",
    "tl": "teamliquid",
    "100 thieves": "100thieves",
    "100t": "100thieves",
    "evil geniuses": "evilgeniuses",
    "eg": "evilgeniuses",
    "flyquest": "flyquest",
    "fly quest": "flyquest",
    "dignitas": "dignitas",
    "dig": "dignitas",
    "immortals": "immortals",
    "imt": "immortals",
    "nrg": "nrg",
    "nrg esports": "nrg",
    "team orangegaming": "teamorangegaming",
    # Regional / academy name variants observed in bookmaker feeds vs GOL.GG
    "estral e sports": "estral",
    "estral esports": "estral",
    "ub alma mater": "universitat de barcelona",
    "universitat de barcelona": "universitat de barcelona",
    "kabum ilha das lendas": "kabum idl",
    "kabum idl": "kabum idl",
    "solary eclipse": "solary",
    "solary": "solary",
    # These are separate Spanish ERL squads in GOL.GG. Collapsing them to the
    # parent org made GIANTX iTero/Pride fixtures share canonical matches and
    # could attach the wrong roster to an upcoming prediction.
    "giantx academy": "giantx itero",
    "giantx itero": "giantx itero",
    # Keep the main LEC roster distinct from the Spanish academy/Fenix roster.
    # Older mappings collapsed these to "movistar koi", which allowed upcoming
    # main-team matches to inherit academy mappings/ratings and vice versa.
    "mkoi": "movistar koi",
    "movistar koi": "movistar koi",
    "koi academy": "movistar koi fenix",
    "movistar koi academy": "movistar koi fenix",
    "mkf": "movistar koi fenix",
    "movistar koi fenix": "movistar koi fenix",
}

SECONDARY_TEAM_MARKERS = {
    "academy",
    "challengers",
    "fenix",
    "prime",
    "pride",
    "bee",
    "cl",
    "eclipse",
    "youth",
    "junior",
}


def _apply_aliases(tokens: list[str]) -> list[str]:
    """Collapse known alias variants to a single canonical form."""
    joined = " ".join(tokens)
    if joined in ALIASES:
        return ALIASES[joined].split()
    # Try progressively shorter prefixes
    for length in range(len(tokens), 0, -1):
        sub = " ".join(tokens[:length])
        if sub in ALIASES:
            rest = tokens[length:]
            return ALIASES[sub].split() + rest
    return tokens


def normalize_team_name(name: str) -> str:
    """Normalize a bookmaker/GOL.GG team name for matching."""

    ascii_name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[^a-z0-9]+", " ", ascii_name)
    # Bookmakers frequently spell the suffix as "E-Sports" / "E Sports".
    # Treat it as the regular "esports" stop word instead of keeping a stray
    # "e sports" tail that breaks exact alias matching (e.g. Estral E-Sports).
    ascii_name = re.sub(r"\be\s+sports?\b", " esports ", ascii_name)
    tokens = [token for token in ascii_name.split() if token and token not in STOP_WORDS]
    tokens = _apply_aliases(tokens)
    return " ".join(tokens).strip()



def stable_team_id(team_id: object, team_name: object) -> str:
    """Return stable team identifier used by historical ratings and features."""
    if team_id is not None and str(team_id).strip():
        return str(team_id).strip()
    return normalize_team_name(str(team_name or ""))

def similarity(left: str, right: str) -> float:
    """Return normalized fuzzy similarity in [0, 1]."""

    norm_left = normalize_team_name(left)
    norm_right = normalize_team_name(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0
    left_markers = set(norm_left.split()) & SECONDARY_TEAM_MARKERS
    right_markers = set(norm_right.split()) & SECONDARY_TEAM_MARKERS
    token_left = set(norm_left.split())
    token_right = set(norm_right.split())
    token_score = len(token_left & token_right) / max(len(token_left | token_right), 1)
    seq_score = SequenceMatcher(None, norm_left, norm_right).ratio()
    score = max(seq_score, token_score)
    if left_markers != right_markers:
        # Do not let fuzzy string overlap (e.g. "Movistar KOI" vs
        # "Movistar KOI Fenix") cross the default match threshold.
        score = min(score, 0.65)
    return score


def best_match(raw_name: str, candidates: list[str], min_confidence: float = 0.72) -> tuple[str | None, float]:
    """Find the best candidate team name for a raw bookmaker name."""

    if not candidates:
        return None, 0.0
    scored = [(candidate, similarity(raw_name, candidate)) for candidate in candidates]
    candidate, score = max(scored, key=lambda item: item[1])
    if score < min_confidence:
        return None, score
    return candidate, score
