import pandas as pd
from betting_app.core.db import query_df

query = """
SELECT MIN(start_time_normalized), MAX(start_time_normalized), COUNT(*)
FROM canonical_matches
WHERE status IN ('finished', 'completed')
"""
print(query_df(query))
