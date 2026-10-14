
from betting_app.core.db import get_session
from betting_app.models.canonical_match import CanonicalMatch
from betting_app.services.canonical_match_service import canonical_team_key

def refresh():
    with get_session() as session:
        matches = session.query(CanonicalMatch).all()
        updated = 0
        for m in matches:
            new_a = canonical_team_key(m.team_a_name)
            new_b = canonical_team_key(m.team_b_name)
            if m.normalized_team_a != new_a or m.normalized_team_b != new_b:
                print(f"Updating CM {m.id}: {m.normalized_team_a} -> {new_a}, {m.normalized_team_b} -> {new_b}")
                m.normalized_team_a = new_a
                m.normalized_team_b = new_b
                updated += 1
        session.commit()
        print(f"Updated normalization for {updated} matches.")

if __name__ == "__main__":
    refresh()
