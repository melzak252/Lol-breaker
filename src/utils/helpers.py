import math
from math import comb
import unicodedata

def normalize_name(name: str) -> str:
    """Normalize team names for matching between datasets."""
    if not isinstance(name, str):
        return ""
        
    # Remove accents
    name = unicodedata.normalize('NFD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    
    name = name.lower()
    
    # Common suffixes/prefixes to remove
    removals = [
        " esports", " gaming", " team", " club", " e-sports", " academy", 
        " challengers", " pride", " challengers", " challenger", " cl",
        " rising", " fenix", " blue", " white", " red", " black"
    ]
    for r in removals:
        name = name.replace(r, "")
        
    name = name.replace("'", "").replace("-", " ").replace(".", "").strip()
    
    overrides = {
        "anyone s legend": "anyone s legend", "oksavingsbank brion": "ok brion",
        "heretics": "heretics", "oh my god": "omg", "fearx": "fearx",
        "dn freecs": "kwangdong freecs", "vitality": "vitality", "isurus": "isurus",
        "top": "top", "jd": "jd", "invictus": "invictus",
        "weibo": "weibo", "bilibili": "bilibili", "tt": "tt",
        "secret whales": "secret whales", "talon": "psg talon", "flyquest": "flyquest",
        "sk": "sk", "g2": "g2", "clg": "clg",
        "tsm": "tsm", "red canids": "red canids", "pain": "pain",
        "furia": "furia", "kwangdong freecs": "kwangdong freecs",
        "dn soopers": "kwangdong freecs", "afreeca freecs": "kwangdong freecs",
        "liiv sandbox": "fearx", "sandbox": "fearx", "lsb": "fearx",
        "fredit brion": "ok brion", "brion": "ok brion",
        "dwg kia": "dplus kia", "damwon": "dplus kia", "dk": "dplus kia",
        "team dynamics": "nongshim redforce", "ns redforce": "nongshim redforce",
        "koi": "movistar koi", "rogue": "movistar koi",
        "giants": "giantx", "excel": "giantx", "gx": "giantx"
    }
    return overrides.get(name, name)

def probability_home_win(home_odds: float, away_odds: float) -> float:
    """Calculate implied probability of home win from decimal odds, removing the margin."""
    ph_raw = 1.0 / float(home_odds)
    pa_raw = 1.0 / float(away_odds)
    return ph_raw / (ph_raw + pa_raw)

def match_win_probability(p: float, bon: int) -> float:
    """
    Calculate the probability of winning a Best-of-N series given the probability
    of winning a single game.
    """
    wins_needed = bon // 2 + 1
    probability = 0.0

    for k in range(wins_needed, bon + 1):
        probability += comb(bon, k) * (p ** k) * ((1 - p) ** (bon - k))

    return probability
