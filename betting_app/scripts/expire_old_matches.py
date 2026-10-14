"""
Skrypt do aktualizacji statusów meczów, które się już odbyły.

Oznacza mecze jako 'expired' jeśli:
1. status='upcoming' AND start_time_normalized < NOW() - grace_period (domyślnie 3h)
2. status='upcoming' AND last_seen_at < NOW() - stale_seen_hours (domyślnie 6h)
   — wyłapuje mecze anulowane/przesunięte, których scrapery już nie widzą

Grace period pozwala na opóźnienia w rozpoczęciu meczu.
Stale seen period wyłapuje mecze, które zniknęły ze scrapera (anulowane/przesunięte).
"""

import argparse
import logging
import psycopg2
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

PG_DSN = "postgresql://betting:betting_local_password@timescaledb:5432/betting"


def expire_old_matches(grace_hours: int = 3, dry_run: bool = False) -> int:
    """
    Oznacza mecze jako 'expired' jeśli czas startu minął.
    
    Args:
        grace_hours: Ile godzin po czasie startu czekać przed oznaczeniem jako expired
        dry_run: Jeśli True, tylko wyświetla co by zrobił
        
    Returns:
        Liczba zaktualizowanych meczów
    """
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    
    try:
        # Znajdź mecze, które powinny być oznaczone jako expired
        cutoff_time = datetime.utcnow() - timedelta(hours=grace_hours)
        cutoff_iso = cutoff_time.isoformat() + "+00:00"
        
        # Sprawdź ile meczów spełnia kryteria
        cur.execute("""
            SELECT COUNT(*) 
            FROM canonical_matches 
            WHERE status = 'upcoming' 
            AND start_time_normalized::timestamp < %s::timestamp
        """, (cutoff_iso,))
        
        count = cur.fetchone()[0]
        log.info(f"Znaleziono {count} meczów do oznaczenia jako 'expired' (cutoff: {cutoff_iso})")
        
        if count == 0:
            return 0
        
        # Wyświetl przykładowe mecze
        cur.execute("""
            SELECT id, team_a_name, team_b_name, start_time_normalized
            FROM canonical_matches 
            WHERE status = 'upcoming' 
            AND start_time_normalized::timestamp < %s::timestamp
            ORDER BY start_time_normalized::timestamp DESC
            LIMIT 10
        """, (cutoff_iso,))
        
        log.info("Przykładowe mecze do oznaczenia:")
        for row in cur.fetchall():
            log.info(f"  ID={row[0]}: {row[1]} vs {row[2]} ({row[3]})")
        
        if dry_run:
            log.info("DRY RUN - nie wprowadzam zmian")
            return count
        
        # Aktualizuj status
        cur.execute("""
            UPDATE canonical_matches 
            SET status = 'expired'
            WHERE status = 'upcoming' 
            AND start_time_normalized::timestamp < %s::timestamp
        """, (cutoff_iso,))
        
        updated = cur.rowcount
        conn.commit()
        
        log.info(f"Zaktualizowano {updated} meczów: status='upcoming' → 'expired'")
        
        return updated
        
    except Exception as e:
        conn.rollback()
        log.error(f"Błąd: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def expire_stale_seen_matches(stale_seen_hours: int = 6, dry_run: bool = False) -> int:
    """
    Oznacza mecze jako 'expired' jeśli nie były widziane przez scrapery
    przez określoną liczbę godzin (last_seen_at jest stare).
    
    To wyłapuje mecze, które zostały anulowane lub przesunięte — scrapery
    przestały je zwracać, więc last_seen_at przestał być aktualizowany.
    
    Args:
        stale_seen_hours: Ile godzin bez widoku scrapera przed oznaczeniem jako expired
        dry_run: Jeśli True, tylko wyświetla co by zrobił
        
    Returns:
        Liczba zaktualizowanych meczów
    """
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    
    try:
        stale_cutoff = datetime.utcnow() - timedelta(hours=stale_seen_hours)
        stale_cutoff_iso = stale_cutoff.isoformat() + "+00:00"
        
        # Znajdź mecze upcoming, których WSZYSTKIE upcoming_matches mają stare last_seen_at
        # (tj. żaden scraper nie widział tego meczu od stale_seen_hours)
        cur.execute("""
            SELECT COUNT(*) 
            FROM canonical_matches cm
            WHERE cm.status = 'upcoming'
            AND NOT EXISTS (
                SELECT 1 FROM upcoming_matches um
                WHERE um.canonical_match_id = cm.id
                AND (um.last_seen_at IS NULL OR um.last_seen_at > %s::timestamp)
            )
        """, (stale_cutoff_iso,))
        
        count = cur.fetchone()[0]
        log.info(f"Znaleziono {count} meczów ze starym last_seen_at do oznaczenia jako 'expired' (stale_cutoff: {stale_cutoff_iso})")
        
        if count == 0:
            return 0
        
        # Wyświetl przykładowe mecze
        cur.execute("""
            SELECT cm.id, cm.team_a_name, cm.team_b_name, cm.start_time_normalized,
                   MAX(um.last_seen_at) as last_seen
            FROM canonical_matches cm
            LEFT JOIN upcoming_matches um ON um.canonical_match_id = cm.id
            WHERE cm.status = 'upcoming'
            AND NOT EXISTS (
                SELECT 1 FROM upcoming_matches um2
                WHERE um2.canonical_match_id = cm.id
                AND (um2.last_seen_at IS NULL OR um2.last_seen_at > %s::timestamp)
            )
            GROUP BY cm.id, cm.team_a_name, cm.team_b_name, cm.start_time_normalized
            ORDER BY cm.start_time_normalized DESC
            LIMIT 10
        """, (stale_cutoff_iso,))
        
        log.info("Przykładowe mecze ze starym last_seen_at:")
        for row in cur.fetchall():
            log.info(f"  ID={row[0]}: {row[1]} vs {row[2]} (start: {row[3]}, last_seen: {row[4]})")
        
        if dry_run:
            log.info("DRY RUN - nie wprowadzam zmian")
            return count
        
        # Aktualizuj status — tylko mecze, których WSZYSTKIE upcoming_matches mają stare last_seen_at
        cur.execute("""
            UPDATE canonical_matches 
            SET status = 'expired'
            WHERE status = 'upcoming'
            AND NOT EXISTS (
                SELECT 1 FROM upcoming_matches um
                WHERE um.canonical_match_id = canonical_matches.id
                AND (um.last_seen_at IS NULL OR um.last_seen_at > %s::timestamp)
            )
        """, (stale_cutoff_iso,))
        
        updated = cur.rowcount
        conn.commit()
        
        log.info(f"Zaktualizowano {updated} meczów ze starym last_seen_at: status='upcoming' → 'expired'")
        
        return updated
        
    except Exception as e:
        conn.rollback()
        log.error(f"Błąd: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Oznacz przeterminowane mecze jako expired")
    parser.add_argument(
        "--grace-hours",
        type=int,
        default=3,
        help="Ile godzin po czasie startu czekać przed oznaczeniem jako expired (domyślnie 3)"
    )
    parser.add_argument(
        "--stale-seen-hours",
        type=int,
        default=0,
        help="Ile godzin bez widoku scrapera przed oznaczeniem jako expired (0 = wyłączone, domyślnie 0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko wyświetl co by zrobił, nie wprowadzaj zmian"
    )
    
    args = parser.parse_args()
    
    total_updated = 0
    
    # 1. Wygaszanie na podstawie czasu startu
    updated = expire_old_matches(
        grace_hours=args.grace_hours,
        dry_run=args.dry_run
    )
    total_updated += updated
    
    # 2. Wygaszanie na podstawie starym last_seen_at (jeśli włączone)
    if args.stale_seen_hours > 0:
        updated = expire_stale_seen_matches(
            stale_seen_hours=args.stale_seen_hours,
            dry_run=args.dry_run
        )
        total_updated += updated
    
    log.info(f"Zakończono. Łącznie zaktualizowano {total_updated} meczów.")


if __name__ == "__main__":
    main()
