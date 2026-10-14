from betting_app.core.db import transaction

def check_upcoming_data():
    with transaction() as conn:
        rows = conn.execute("""
            SELECT um.id, um.canonical_match_id, um.raw_team_a, um.raw_team_b, 
                   um.team_a_golgg_id, um.team_b_golgg_id, um.last_seen_at
            FROM upcoming_matches um
            ORDER BY um.last_seen_at DESC
            LIMIT 10
        """).fetchall()
        for row in rows:
            print(dict(row))

if __name__ == "__main__":
    check_upcoming_data()
