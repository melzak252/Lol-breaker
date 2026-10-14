import pandas as pd
import numpy as np
from betting_app.core.db import query_df, get_session
from betting_app.core.matching import normalize_team_name
from sqlalchemy import text
import os
from dotenv import load_dotenv

load_dotenv()

def map_matches():
    # 1. Get all expired and upcoming matches that are NOT mapped yet
    query_canonical = """
    SELECT 
        cm.id as canonical_match_id,
        cm.team_a_name,
        cm.team_b_name,
        cm.start_time_normalized,
        cm.status
    FROM canonical_matches cm
    LEFT JOIN golgg_match_mappings gmm ON cm.id = gmm.canonical_match_id
    WHERE gmm.canonical_match_id IS NULL
    AND cm.status IN ('expired', 'upcoming')
    """
    df_cm = query_df(query_canonical)
    print(f"Found {len(df_cm)} unmapped canonical matches (expired/upcoming).")

    # 2. Get all GOL.GG matches
    query_golgg = "SELECT match_id, team1_name, team2_name, date, team1_win, team2_win FROM golgg_matches"
    df_golgg = query_df(query_golgg)
    print(f"Found {len(df_golgg)} GOL.GG matches.")

    # Normalize names for matching
    df_golgg['norm1'] = df_golgg['team1_name'].apply(normalize_team_name)
    df_golgg['norm2'] = df_golgg['team2_name'].apply(normalize_team_name)

    mappings_found = 0

    with get_session() as session:
        for _, row in df_cm.iterrows():
            match_date = pd.to_datetime(row['start_time_normalized']).date()
            norm_a = normalize_team_name(row['team_a_name'])
            norm_b = normalize_team_name(row['team_b_name'])

            # Find potential matches in GOL.GG within +/- 1 day
            mask = (pd.to_datetime(df_golgg['date']).dt.date >= match_date - pd.Timedelta(days=1)) & \
                   (pd.to_datetime(df_golgg['date']).dt.date <= match_date + pd.Timedelta(days=1))
            
            potentials = df_golgg[mask]

            for _, g_row in potentials.iterrows():
                g1 = g_row['norm1']
                g2 = g_row['norm2']

                # Check if teams match (either order)
                match_direct = (norm_a == g1 and norm_b == g2) or (norm_a == g2 and norm_b == g1)
                
                # Substring match for cases like "T1" vs "T1 Esports"
                match_sub = ( (norm_a in g1 or g1 in norm_a) and (norm_b in g2 or g2 in norm_b) ) or \
                            ( (norm_a in g2 or g2 in norm_a) and (norm_b in g1 or g1 in norm_b) )

                if match_direct or match_sub:
                    print(f"MATCH FOUND: {row['team_a_name']} vs {row['team_b_name']} ({match_date}) "
                          f"-> GOL.GG: {g_row['team1_name']} vs {g_row['team2_name']} ({g_row['date']})")
                    
                    try:
                        session.execute(text("""
                            INSERT INTO golgg_match_mappings (canonical_match_id, golgg_match_id)
                            VALUES (:c_id, :g_id)
                            ON CONFLICT DO NOTHING
                        """), {"c_id": row['canonical_match_id'], "g_id": g_row['match_id']})
                        
                        # If it was expired or upcoming, but we found a result, mark as finished
                        session.execute(text("""
                            UPDATE canonical_matches SET status = 'finished' WHERE id = :id
                        """), {"id": row['canonical_match_id']})
                        
                        mappings_found += 1
                    except Exception as e:
                        print(f"Error inserting mapping: {e}")
                    break # Found a match for this canonical match
        
        session.commit()

    print(f"\nTotal new mappings created: {mappings_found}")

if __name__ == "__main__":
    map_matches()
