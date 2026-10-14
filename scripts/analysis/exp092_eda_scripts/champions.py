from pathlib import Path
import pandas as pd,numpy as np,json
P=Path(__file__).resolve().parent;root=P.parents[2]
f=pd.read_csv(root/'data/artifacts/exp086-champion-meta/source-20260908/champion_history.csv');f['year']=pd.to_datetime(f.date).dt.year
exact=int(f.duplicated().sum());unique=f.drop_duplicates();key=['match_id','game_id','team_id','role'];conflict=unique[unique.duplicated(key,keep=False)]
conflict.to_csv(P/'champion_conflicting_slots.csv',index=False)
counts=unique.groupby(['match_id','game_id']).size();bad=counts[counts!=10];bad.to_csv(P/'champion_game_slot_counts.csv')
y=[]
for year,h in unique.groupby('year'):
 y.append(dict(year=year,rows=len(h),games=h.groupby(['match_id','game_id']).ngroups,players=h.player_id.nunique(),champions=h.champion_id.nunique(),patch_missing=h.observed_game_patch.isna().mean(),champion_missing=h.champion_id.isna().mean()))
pd.DataFrame(y).to_csv(P/'champion_year_coverage.csv',index=False)
# Descriptive pool breadth, not current-draft prediction features.
modern=unique[unique.year>=2024];pool=modern.groupby(['player_id','year','role']).agg(games=('game_id','size'),champions=('champion_id','nunique')).reset_index();pool.to_csv(P/'player_champion_pool.csv',index=False)
patch=[]
for (p,role),h in modern.dropna(subset=['observed_game_patch']).groupby(['observed_game_patch','role']):
 z=h.champion_id.value_counts(normalize=True);patch.append(dict(patch=p,role=role,n=len(h),champions=len(z),top3_share=z.head(3).sum(),entropy=float(-(z*np.log(z)).sum())))
pd.DataFrame(patch).to_csv(P/'patch_role_diversity.csv',index=False)
meta={'rows':len(f),'exact_duplicate_rows':exact,'conflicting_slot_rows_after_exact_dedup':len(conflict),'games_not_10_slots':len(bad),'modern_player_year_roles_20plus':int((pool.games>=20).sum()),'median_champions_20plus':float(pool.loc[pool.games>=20,'champions'].median()),'historical_availability_timestamps':False};(P/'champion_metadata.json').write_text(json.dumps(meta,indent=2));print(meta)
