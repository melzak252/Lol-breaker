from pathlib import Path
import pandas as pd
P=Path(__file__).parent
m=pd.read_csv(P/'series_inventory.csv',dtype={'id':str});c=pd.read_csv(P.parents[2]/'data/artifacts/exp086-champion-meta/source-20260908/champion_history.csv',dtype={'match_id':str})
c=c.merge(m[['id','tournament','date']],left_on='match_id',right_on='id',suffixes=('','_series'),validate='many_to_one');c=c[c.date==c.date_series].copy()
# Membership graph, not inferred ordered transfers. Exclude international-event labels.
c['family']=c.tournament.str.extract(r'^(Prime League|Hitpoint Masters|LVP SL|CBLOL|LPL|LCK|LEC|LCS|LFL|TCL|PCS|VCS|LJL|NLC|LCP)',expand=False)
c=c[pd.to_datetime(c.date).dt.year>=2024].dropna(subset=['family']);members=c[['player_id','family']].drop_duplicates();sets=members.groupby('family').player_id.agg(set);rows=[]
for i,a in enumerate(sets.index):
 for b in sets.index[i+1:]:
  common=sets[a]&sets[b]
  if common:rows.append(dict(league_a=a,league_b=b,shared_players=len(common)))
r=pd.DataFrame(rows).sort_values('shared_players',ascending=False);r.to_csv(P/'league_player_bridges.csv',index=False)
print(r.head(12).to_string(index=False))
