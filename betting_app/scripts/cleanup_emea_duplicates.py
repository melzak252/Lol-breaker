"""
Cleanup 3 EMEA Masters duplicate canonical_matches that have wrong dates.
Repoint FK references from expired → survivor, then delete expired rows.
"""
import sys
sys.path.insert(0, '/app')

from betting_app.core.db import get_session
from sqlalchemy import text

PAIRS = {
    59940: 57713,  # E WIE EINFACH E-SPORTS vs Hmble → E WIE EINFACH vs HMBLE
    59968: 52885,  # UCAM Esports vs Hmble → UCAM Esports Club vs Hmble
    60140: 57016,  # Team Heretics Academy vs Forsaken → Forsaken vs Team Heretics Academy
    58464: 60130,  # Karmine Corp Blue vs UCAM Esports Club (June 14) → UCAM Esports vs Karmine Corp Blue (June 13)
    58091: 60130,  # UCAM Esports Club vs Karmine Corp Blue (June 13) → UCAM Esports vs Karmine Corp Blue (June 13)
    58060: 59936,  # UCAM Esports Club vs Misa eSports (June 11) → Misa Esports vs UCAM Esports (June 11)
}

FK_TABLES = [
    ('canonical_predictions', 'canonical_match_id'),
    ('odds_snapshots', 'canonical_match_id'),
    ('upcoming_match_features', 'canonical_match_id'),
    ('upcoming_matches', 'canonical_match_id'),
    ('model_ev_signals', 'canonical_match_id'),
    ('bookmaker_events', 'canonical_match_id'),
]


def repoint_and_delete():
    with get_session() as db:
        for old_id, new_id in PAIRS.items():
            print(f'=== CM {old_id} → Survivor {new_id} ===')
            
            # Phase 1: Repoint FK that WON'T conflict
            for table, col in FK_TABLES:
                if table == 'upcoming_match_features':
                    # Handle unique constraint: try repoint, skip on conflict
                    rows = db.execute(text(
                        f"SELECT * FROM {table} WHERE {col} = :old_id"
                    ), {'old_id': old_id}).fetchall()
                    for row in rows:
                        # Check if survivor already has same (feature_version, ratings_version)
                        conflict = db.execute(text(
                            "SELECT 1 FROM upcoming_match_features "
                            "WHERE canonical_match_id = :new_id "
                            "AND feature_version = :fv AND ratings_version = :rv LIMIT 1"
                        ), {'new_id': new_id, 'fv': row.feature_version, 'rv': row.ratings_version}).scalar()
                        
                        if conflict:
                            # Delete the old one (survivor already has it)
                            db.execute(text(
                                f"DELETE FROM upcoming_match_features WHERE id = :rid"
                            ), {'rid': row.id})
                            print(f'  upcoming_match_features: removed duplicate row {row.id} (survivor already has it)')
                        else:
                            # Repoint
                            db.execute(text(
                                f"UPDATE upcoming_match_features SET {col} = :new_id WHERE id = :rid"
                            ), {'new_id': new_id, 'rid': row.id})
                            print(f'  upcoming_match_features: repointed row {row.id} → {new_id}')
                elif table == 'upcoming_matches':
                    # Handle unique constraint on (bookmaker_match_key)
                    rows = db.execute(text(
                        f"SELECT * FROM {table} WHERE {col} = :old_id"
                    ), {'old_id': old_id}).fetchall()
                    for row in rows:
                        # Check if survivor already has same bookmaker_match_key
                        conflict = db.execute(text(
                            "SELECT 1 FROM upcoming_matches "
                            "WHERE bookmaker_match_key = :key AND canonical_match_id = :new_id LIMIT 1"
                        ), {'key': row.bookmaker_match_key, 'new_id': new_id}).scalar()
                        
                        if conflict:
                            db.execute(text(
                                f"DELETE FROM upcoming_matches WHERE id = :rid"
                            ), {'rid': row.id})
                            print(f'  upcoming_matches: removed duplicate row {row.id} (survivor already has it)')
                        else:
                            db.execute(text(
                                f"UPDATE upcoming_matches SET {col} = :new_id WHERE id = :rid"
                            ), {'new_id': new_id, 'rid': row.id})
                            print(f'  upcoming_matches: repointed row {row.id} → {new_id}')
                else:
                    # Tables without unique constraints on canonical_match_id
                    cnt = db.execute(text(
                        f"UPDATE {table} SET {col} = :new_id WHERE {col} = :old_id"
                    ), {'new_id': new_id, 'old_id': old_id}).rowcount
                    if cnt:
                        print(f'  {table}: repointed {cnt} rows → {new_id}')
            
            # Phase 2: Delete the expired canonical_match
            db.execute(text("DELETE FROM canonical_matches WHERE id = :old_id"), {'old_id': old_id})
            print(f'  DELETED canonical_match {old_id}')
        
        db.commit()
    
    print('\nDone! All 3 EMEA duplicates cleaned up.')


if __name__ == '__main__':
    repoint_and_delete()
