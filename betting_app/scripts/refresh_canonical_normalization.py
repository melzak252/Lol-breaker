from betting_app.core.db import get_session, query_df
from betting_app.services.canonical_match_service import canonical_team_key
from sqlalchemy import text

def refresh_normalization():
    with get_session() as session:
        rows = session.execute(text("SELECT id, team_a_name, team_b_name, normalized_team_a, normalized_team_b FROM canonical_matches")).fetchall()
        updated = 0
        for r in rows:
            new_a = canonical_team_key(r.team_a_name)
            new_b = canonical_team_key(r.team_b_name)
            if new_a != r.normalized_team_a or new_b != r.normalized_team_b:
                session.execute(text(
                    "UPDATE canonical_matches SET normalized_team_a = :a, normalized_team_b = :b WHERE id = :id"
                ), {"a": new_a, "b": new_b, "id": r.id})
                updated += 1
        session.commit()
        print(f"Updated normalization for {updated} matches.")

if __name__ == "__main__":
    refresh_normalization()
