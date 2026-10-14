from betting_app.core.db import query_df
print(query_df("SELECT status, COUNT(*) FROM canonical_matches GROUP BY status"))
print(query_df("SELECT model_name, COUNT(*) FROM canonical_predictions GROUP BY model_name"))
print(query_df("SELECT COUNT(*) FROM golgg_match_mappings"))
