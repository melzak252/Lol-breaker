
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import deque

DATA_DIR = Path("data")

_DEFAULT_TEAM_STATS = {
    "win_rate": 0.5, "kills": 12.0, "deaths": 12.0, "gd15": 0.0, "dpm": 1800.0,
    "vspm": 7.0, "towers": 5.0, "nashors": 0.5, "gold": 55000.0, "duration": 1800.0,
}

def _safe_stat(player: dict, key: str) -> float:
    return float(player.get("stats", {}).get(key, 0.0) or 0.0)

def _safe_team_stat(game: dict, stats_key: str, key: str) -> float:
    return float((game.get(stats_key, {}) or {}).get(key, 0.0) or 0.0)

def _update_team_history(team_history: dict, team_id: str, game: dict, window_size: int):
    is_team_1 = str(game.get("t1_id")) == str(team_id)
    win = bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))
    players_key = "t1_players" if is_team_1 else "t2_players"
    stats_key = "t1_stats" if is_team_1 else "t2_stats"
    players = game.get(players_key, {}) or {}
    game_stats = {
        "win_rate": float(win),
        "kills": sum(_safe_stat(p, "kills") for p in players.values()),
        "deaths": sum(_safe_stat(p, "deaths") for p in players.values()),
        "dpm": sum(_safe_stat(p, "dpm") for p in players.values()),
        "vspm": sum(_safe_stat(p, "vspm") for p in players.values()),
        "gd15": sum(_safe_stat(p, "gd@15") for p in players.values()),
        "towers": _safe_team_stat(game, stats_key, "towers"),
        "nashors": _safe_team_stat(game, stats_key, "nashors"),
        "gold": _safe_team_stat(game, stats_key, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    if team_id not in team_history:
        team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)

def main():
    matches_path = DATA_DIR / "golgg_matches.json"
    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    matches.sort(key=lambda m: m["date"])

    team_history = {}
    window_size = 20
    
    stats_2025 = []
    stats_2026 = []

    for match in tqdm(matches):
        m_date = match["date"]
        year = int(m_date.split("-")[0])
        
        if year == 2025 or year == 2026:
            # Collect current rolling stats
            for tid in [str(match.get("t1_id")), str(match.get("t2_id"))]:
                if tid in team_history and team_history[tid]:
                    hist = list(team_history[tid])
                    row = {k: np.mean([h[k] for h in hist]) for k in _DEFAULT_TEAM_STATS.keys()}
                    if year == 2025: stats_2025.append(row)
                    else: stats_2026.append(row)

        # Update history
        t1 = str(match.get("t1_id"))
        t2 = str(match.get("t2_id"))
        for game in match.get("games", []):
            _update_team_history(team_history, t1, game, window_size)
            _update_team_history(team_history, t2, game, window_size)

    df_2025 = pd.DataFrame(stats_2025)
    df_2026 = pd.DataFrame(stats_2026)

    print("\nFeature Distributions (Mean):")
    comparison = pd.DataFrame({
        "2025": df_2025.mean(),
        "2026": df_2026.mean(),
        "Diff %": (df_2026.mean() - df_2025.mean()) / df_2025.mean() * 100
    })
    print(comparison)
    
    print("\nGD15 Zero %:")
    print(f"2025: {(df_2025['gd15'] == 0).mean():.2%}")
    print(f"2026: {(df_2026['gd15'] == 0).mean():.2%}")

    print("\nDPM Zero %:")
    print(f"2025: {(df_2025['dpm'] == 0).mean():.2%}")
    print(f"2026: {(df_2026['dpm'] == 0).mean():.2%}")

if __name__ == "__main__":
    main()
