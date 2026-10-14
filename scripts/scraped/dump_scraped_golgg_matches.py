"""Dump all finished canonical matches mapped to GOL.GG IDs."""

import json
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@timescaledb:5432/betting",
)
engine = create_engine(db_url)

with engine.connect() as conn:
    df = pd.read_sql(
        text("""
        SELECT cm.id as canonical_match_id, gmm.golgg_match_id, cm.team_a_name, cm.team_b_name,
               cm.winner_side, cm.start_time_normalized, cm.league, cm.best_of
        FROM canonical_matches cm
        JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        WHERE cm.status IN ('finished', 'completed') AND cm.winner_side IN ('team_a', 'team_b')
        ORDER BY cm.start_time_normalized ASC
        """),
        conn,
    )

out_file = "/app/data/scraped_golgg_matches.json"
df.to_json(out_file, orient="records", indent=2)
print(f"Dumped {len(df)} mapped matches to {out_file}")
