from betting_app.core.db import transaction

def check_aliases_data():
    with transaction() as conn:
        rows = conn.execute("SELECT * FROM team_aliases LIMIT 5").fetchall()
        for row in rows:
            print(dict(row))

if __name__ == "__main__":
    check_aliases_data()
