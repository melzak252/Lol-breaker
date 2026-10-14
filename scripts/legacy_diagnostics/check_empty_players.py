import pandas as pd
from betting_app.core.db import query_df

all_matches = query_df("""
    SELECT match_id, team1_id, team2_id, team1_name, team2_name, date, best_of, draw
    FROM golgg_matches
    WHERE date IS NOT NULL AND draw = 0
    ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
""")

all_games = query_df("""
    SELECT game_id, match_id, team1_win, team2_win, game_duration,
           team1_stats_json, team2_stats_json
    FROM golgg_games
""")
games_by_match = {str(m_id): group for m_id, group in all_games.groupby('match_id')}

all_players = query_df("""
    SELECT game_id, player_id, team_id, stats_json
    FROM golgg_game_players
""")
players_by_game = {str(g_id): group for g_id, group in all_players.groupby('game_id')}

empty_p1 = 0
empty_p2 = 0
total = 0

for _, m in all_matches.iterrows():
    m_id = str(m['match_id'])
    t1_id = str(m['team1_id'])
    t2_id = str(m['team2_id'])
    
    match_games = games_by_match.get(m_id)
    if match_games is None or match_games.empty: continue
    match_games = match_games.sort_values('game_id')
    
    first_game_id = str(match_games.iloc[0]['game_id'])
    game_players = players_by_game.get(first_game_id)
    if game_players is None:
        p1, p2 = [], []
    else:
        p1 = game_players[game_players['team_id'].astype(str) == t1_id]['player_id'].astype(str).tolist()
        p2 = game_players[game_players['team_id'].astype(str) == t2_id]['player_id'].astype(str).tolist()
        
    if not p1: empty_p1 += 1
    if not p2: empty_p2 += 1
    total += 1

print(f"Total matches: {total}")
print(f"Empty p1: {empty_p1}")
print(f"Empty p2: {empty_p2}")
