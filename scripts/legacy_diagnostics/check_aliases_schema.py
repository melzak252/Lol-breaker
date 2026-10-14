from betting_app.core.db import transaction

def check_team_aliases_schema():
    with transaction() as conn:
        info = conn.execute("PRAGMA table_info(team_aliases)").fetchall()
        for col in info:
            print(f"Column: {col['name']}, Type: {col['type']}")

if __name__ == "__main__":
    check_team_aliases_schema()
