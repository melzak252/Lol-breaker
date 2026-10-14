"""
Naprawia mapowanie canonical_match_id w odds_snapshots.

Problem: fix_canonical_match_id_mapping.py mapował po canonical_key,
ale to nie zadziałało poprawnie — 3963 z 4876 snapshotów ma złe przypisanie.

Rozwiązanie:
1. Wyczyść odds_snapshots w PG
2. Zmigruj z SQLite z mapowaniem opartym na (team_a_name, team_b_name)
"""

import sqlite3
import psycopg2
from pathlib import Path

SQLITE_PATH = Path("/app/data/betting_app.sqlite3")
PG_DSN = "postgresql://betting:betting_local_password@timescaledb:5432/betting"


def main():
    print("=== Naprawianie mapowania odds_snapshots ===\n")
    
    # 1. Wczytaj SQLite canonical_matches (team_a_name, team_b_name)
    print("1. Wczytywanie SQLite canonical_matches...")
    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_cur = sqlite_conn.cursor()
    
    sqlite_cur.execute("""
        SELECT id, team_a_name, team_b_name
        FROM canonical_matches
        WHERE status = 'upcoming'
    """)
    sqlite_matches = {
        (row[1].lower(), row[2].lower()): row[0]  # (team_a, team_b) -> sqlite_id
        for row in sqlite_cur.fetchall()
    }
    print(f"   SQLite: {len(sqlite_matches)} canonical_matches\n")
    
    # 2. Wczytaj PG canonical_matches
    print("2. Wczytywanie PG canonical_matches...")
    pg_conn = psycopg2.connect(PG_DSN)
    pg_cur = pg_conn.cursor()
    
    pg_cur.execute("""
        SELECT id, team_a_name, team_b_name
        FROM canonical_matches
        WHERE status = 'upcoming'
    """)
    pg_matches = {
        (row[1].lower(), row[2].lower()): row[0]  # (team_a, team_b) -> pg_id
        for row in pg_cur.fetchall()
    }
    print(f"   PG: {len(pg_matches)} canonical_matches\n")
    
    # 3. Zbuduj mapowanie: sqlite_id -> pg_id
    print("3. Budowanie mapowania sqlite_id -> pg_id...")
    id_mapping = {}
    for (team_a, team_b), sqlite_id in sqlite_matches.items():
        if (team_a, team_b) in pg_matches:
            pg_id = pg_matches[(team_a, team_b)]
            id_mapping[sqlite_id] = pg_id
    
    print(f"   Zmapowano {len(id_mapping)} meczów\n")
    
    # 4. Wyczyść odds_snapshots w PG
    print("4. Czyszczenie odds_snapshots w PG...")
    pg_cur.execute("DELETE FROM odds_snapshots WHERE market_type = 'match_winner'")
    deleted = pg_cur.rowcount
    pg_conn.commit()
    print(f"   Usunięto {deleted} wierszy\n")
    
    # 5. Zmigruj odds_snapshots z SQLite
    print("5. Migracja odds_snapshots z SQLite...")
    
    sqlite_cur.execute("""
        SELECT 
            os.canonical_match_id,
            os.bookmaker_id,
            os.market_type,
            os.raw_team_a,
            os.raw_team_b,
            os.odds_a,
            os.odds_b,
            os.scraped_at,
            os.source_url
        FROM odds_snapshots os
        WHERE os.market_type = 'match_winner'
        ORDER BY os.scraped_at
    """)
    
    rows = sqlite_cur.fetchall()
    print(f"   SQLite: {len(rows)} snapshotów do migracji\n")
    
    # 6. Wstaw do PG z poprawnym canonical_match_id
    print("6. Wstawianie do PG...")
    inserted = 0
    skipped = 0
    
    # Mapowanie bookmaker_id: SQLite -> PG
    bookmaker_mapping = {
        1: None,    # manual - skip
        2: 2,       # sts
        3: 3,       # betclic
        4: 6,       # fortuna
        391: 4,     # superbet
        392: 5,     # efortuna
        520: 7,     # betfan
        521: 8,     # totalbet
        522: 9,     # lebull
    }
    
    for row in rows:
        sqlite_cm_id = row[0]
        sqlite_bk_id = row[1]
        
        if sqlite_cm_id not in id_mapping:
            skipped += 1
            continue
        
        if sqlite_bk_id not in bookmaker_mapping or bookmaker_mapping[sqlite_bk_id] is None:
            skipped += 1
            continue
        
        pg_cm_id = id_mapping[sqlite_cm_id]
        pg_bk_id = bookmaker_mapping[sqlite_bk_id]
        
        pg_cur.execute("""
            INSERT INTO odds_snapshots (
                canonical_match_id, bookmaker_id, market_type,
                raw_team_a, raw_team_b, odds_a, odds_b,
                scraped_at, source_url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            pg_cm_id,
            pg_bk_id,  # bookmaker_id (mapped)
            row[2],  # market_type
            row[3],  # raw_team_a
            row[4],  # raw_team_b
            row[5],  # odds_a
            row[6],  # odds_b
            row[7],  # scraped_at
            row[8],  # source_url
        ))
        inserted += pg_cur.rowcount
    
    pg_conn.commit()
    print(f"   Wstawiono: {inserted}")
    print(f"   Pominięto: {skipped}\n")
    
    # 7. Weryfikacja
    print("7. Weryfikacja...")
    pg_cur.execute("""
        SELECT COUNT(*) FROM odds_snapshots WHERE market_type = 'match_winner'
    """)
    total = pg_cur.fetchone()[0]
    
    pg_cur.execute("""
        SELECT COUNT(*)
        FROM odds_snapshots os
        JOIN canonical_matches cm ON cm.id = os.canonical_match_id
        WHERE os.market_type = 'match_winner'
          AND (os.raw_team_a != cm.team_a_name OR os.raw_team_b != cm.team_b_name)
    """)
    mismatched = pg_cur.fetchone()[0]
    
    print(f"   Total odds_snapshots: {total}")
    print(f"   Mismatched team names: {mismatched}")
    print(f"   Poprawne: {total - mismatched}\n")
    
    sqlite_conn.close()
    pg_conn.close()
    
    print("=== Zakończono ===")


if __name__ == "__main__":
    main()
