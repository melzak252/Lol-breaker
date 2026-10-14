from betting_app.core.db import transaction

def find_teams():
    with transaction() as conn:
        teams = conn.execute("SELECT id, team_name FROM golgg_teams WHERE team_name IN ('T1', 'Gen.G', 'G2 Esports', 'Fnatic', 'Dplus KIA', 'Hanwha Life Esports')").fetchall()
        for t in teams:
            print(f"ID: {t['id']}, Name: {t['team_name']}")

if __name__ == "__main__":
    find_teams()
