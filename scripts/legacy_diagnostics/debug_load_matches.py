import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect
from betting_app.scripts.rebuild_ratings import load_matches, load_existing_rating_state

version = "latest-full"
state = load_existing_rating_state(version)
cutoff = state.get("data_cutoff_at")
processed = state.get("processed_match_ids")

print(f"Cutoff: {cutoff}")
print(f"Processed count: {len(processed)}")

matches = load_matches(after_date=cutoff, processed_match_ids=processed)
print(f"Found {len(matches)} new matches")

if matches:
    print(f"First new match: {matches[0].match_id} on {matches[0].match_date}")
