import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()

# Check how many game_players have empty team_id
c.execute("SELECT COUNT(*) FROM golgg_game_players WHERE team_id IS NULL OR team_id = ''")
empty_team_id = c.fetchone()[0]
print(f"Game players with empty team_id: {empty_team_id}")

# Check how many matches have games but no players for team1 or team2
c.execute("""
    WITH first_games AS (
        SELECT match_id, MIN(CAST(game_id AS INTEGER)) as first_game_id
        FROM golgg_games
        GROUP BY match_id
    ),
    match_teams AS (
        SELECT m.match_id, m.team1_id, m.team2_id, fg.first_game_id
        FROM golgg_matches m
        JOIN first_games fg ON m.match_id = fg.match_id
        WHERE m.draw = 0 AND m.date IS NOT NULL
    ),
    p1_counts AS (
        SELECT mt.match_id, COUNT(gp.player_id) as p1_count
        FROM match_teams mt
        LEFT JOIN golgg_game_players gp ON gp.game_id = mt.first_game_id AND gp.team_id = mt.team1_id
        GROUP BY mt.match_id
    ),
    p2_counts AS (
        SELECT mt.match_id, COUNT(gp.player_id) as p2_count
        FROM match_teams mt
        LEFT JOIN golgg_game_players gp ON gp.game_id = mt.first_game_id AND gp.team_id = mt.team2_id
        GROUP BY mt.match_id
    )
    SELECT 
        COUNT(*) as total_matches,
        SUM(CASE WHEN p1.p1_count = 0 THEN 1 ELSE 0 END) as empty_p1,
        SUM(CASE WHEN p2.p2_count = 0 THEN 1 ELSE 0 END) as empty_p2
    FROM match_teams mt
    JOIN p1_counts p1 ON mt.match_id = p1.match_id
    JOIN p2_counts p2 ON mt.match_id = p2.match_id
""")
res = c.fetchone()
print(f"Total matches: {res[0]}")
print(f"Empty p1: {res[1]}")
print(f"Empty p2: {res[2]}")

conn.close()
