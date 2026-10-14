from betting_app.core.db import transaction

def check_matches():
    with transaction() as conn:
        cm_count = conn.execute("SELECT COUNT(*) as cnt FROM canonical_matches WHERE status = 'upcoming'").fetchone()['cnt']
        os_count = conn.execute("SELECT COUNT(DISTINCT canonical_match_id) as cnt FROM odds_snapshots").fetchone()['cnt']
        um_count = conn.execute("SELECT COUNT(DISTINCT canonical_match_id) as cnt FROM upcoming_matches").fetchone()['cnt']
        
        print(f"Upcoming canonical matches: {cm_count}")
        print(f"Matches with odds: {os_count}")
        print(f"Matches with upcoming_matches entry: {um_count}")
        
        sql = """
            SELECT cm.id, cm.team_a_name, cm.team_b_name
            FROM canonical_matches cm
            JOIN odds_snapshots os ON os.canonical_match_id = cm.id
            LEFT JOIN LATERAL (
                SELECT team_a_golgg_id, team_b_golgg_id
                FROM upcoming_matches
                WHERE canonical_match_id = cm.id
                ORDER BY last_seen_at DESC
                LIMIT 1
            ) um ON TRUE
            WHERE cm.status = 'upcoming'
            GROUP BY cm.id, um.team_a_golgg_id, um.team_b_golgg_id
        """
        joined = conn.execute(sql).fetchall()
        print(f"Matches meeting prediction criteria: {len(joined)}")
        for m in joined[:5]:
            print(f"ID: {m['id']}, {m['team_a_name']} vs {m['team_b_name']}")

if __name__ == "__main__":
    check_matches()
