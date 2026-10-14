
from betting_app.core.db import query_df
from betting_app.services.mapping_service import suggest_mapping
from betting_app.core.matching import normalize_team_name as _ntn, similarity as _sim
from betting_app.services.canonical_match_service import canonical_team_key as _ctk
from datetime import datetime, timedelta

def debug_mapping(cm_id):
    cm_rows = query_df(f"SELECT * FROM canonical_matches WHERE id = {cm_id}")
    if cm_rows.empty:
        print(f"CM {cm_id} not found.")
        return
    cm = cm_rows.iloc[0]
    print(f"DEBUGGING CM {cm_id}: {cm['team_a_name']} vs {cm['team_b_name']} ({cm['start_time_normalized']}) [{cm['league']}]")
    
    team_a = cm['team_a_name']
    team_b = cm['team_b_name']
    start_time = cm['start_time_normalized']
    
    if not start_time:
        print("No start time, cannot map by date.")
        return

    try:
        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    except ValueError:
        # Handle format like "2026-06-14 00:55:13+00"
        if ' ' in start_time and '+' in start_time:
             dt = datetime.strptime(start_time.split('.')[0], "%Y-%m-%d %H:%M:%S")
        else:
             print(f"Could not parse date: {start_time}")
             return

    date_str = dt.strftime("%Y-%m-%d")
    
    window_start = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    window_end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
    
    print(f"Searching GOL.GG matches between {window_start} and {window_end}")
    
    golgg = query_df(f"""
        SELECT match_id, team1_name, team2_name, date, tournament_name
        FROM golgg_matches
        WHERE date >= '{window_start}' AND date <= '{window_end}'
    """)
    
    print(f"Found {len(golgg)} GOL.GG matches in window.")
    
    norm_a = str(cm['normalized_team_a'] or "")
    norm_b = str(cm['normalized_team_b'] or "")
    
    for _, g in golgg.iterrows():
        g1 = str(g['team1_name'])
        g2 = str(g['team2_name'])
        
        # Simple check
        s1 = (_sim(g1, norm_a) + _sim(g2, norm_b)) / 2
        s2 = (_sim(g1, norm_b) + _sim(g2, norm_a)) / 2
        score = max(s1, s2)
        
        if score > 0.5:
            print(f"  GOLGG {g['match_id']}: {g1} vs {g2} ({g['date']}) -> Score: {score:.3f}")
            print(f"    Sim({g1}, {norm_a})={_sim(g1, norm_a):.3f}, Sim({g2}, {norm_b})={_sim(g2, norm_b):.3f}")
            print(f"    Sim({g1}, {norm_b})={_sim(g1, norm_b):.3f}, Sim({g2}, {norm_a})={_sim(g2, norm_a):.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debug_mapping(int(sys.argv[1]))
    else:
        # Try a few
        for cid in [64551, 64722, 64626]:
            debug_mapping(cid)
            print("-" * 40)
