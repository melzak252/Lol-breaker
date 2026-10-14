from pathlib import Path
import json,os,hashlib
from collections import Counter,defaultdict
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
P=Path(__file__).resolve().parent;ROOT=P.parents[2];D=ROOT/'data'
# Inventory metadata only. Never read secrets, database payloads or browser debug bodies.
inv=[]
for base,dirs,files in os.walk(D):
 for name in files:
  q=Path(base)/name
  if q.is_symlink():continue
  st=q.stat();rel=q.relative_to(D);inv.append({'path':str(rel),'bytes':st.st_size,'family':rel.parts[0] if len(rel.parts)>1 else 'top-level','extension':q.suffix})
iv=pd.DataFrame(inv);iv.groupby(['family','extension']).agg(files=('bytes','size'),bytes=('bytes','sum')).reset_index().to_csv(P/'inventory_summary.csv',index=False)
# Streaming top-level JSON array, bounded memory, strict parse failures.
def objects(path):
 dec=json.JSONDecoder();buf='';pos=0;eof=False
 with path.open() as fh:
  while True:
   if not eof and len(buf)-pos<1048576:
    buf=buf[pos:]+fh.read(1048576);pos=0;eof=len(buf)==0 or len(buf)<1048576
   while pos<len(buf) and buf[pos] in ' \n\r\t[,':pos+=1
   if pos<len(buf) and buf[pos]==']':return
   try:obj,end=dec.raw_decode(buf,pos)
   except json.JSONDecodeError:
    more=fh.read(1048576)
    if not more:raise
    buf=buf[pos:]+more;pos=0;continue
   pos=end;yield obj
# Repaired full history: derive aggregate coverage, roster continuity, map-side effects.
raw=D/'artifacts/exp090-data-repair/run03/matches.json';series=[];maps=[];stats=defaultdict(Counter);rosters=[]
for m in objects(raw):
 date=str(m.get('date') or '');year=date[:4];games=m.get('games') or []
 series.append({'id':str(m.get('match_id')),'date':date,'year':year,'tournament':m.get('tournament_name'),'best_of':m.get('best_of'),'maps':len(games),'declared_games':m.get('games_played'),'patch':m.get('patch'),'score_sum':(m.get('t1_score') or 0)+(m.get('t2_score') or 0)})
 for i,game in enumerate(games):
  maps.append({'id':str(m.get('match_id')),'game_id':str(game.get('game_id')),'year':year,'date':date,'map_number':i+1,'blue_won':bool(game.get('t1_win')) if game.get('t1_side')=='blue' else bool(game.get('t2_win')) if game.get('t2_side')=='blue' else np.nan,'patch':game.get('patch') or m.get('patch')})
  for side in ['t1','t2']:
   players=game.get(side+'_players') or {};players=players.values() if isinstance(players,dict) else players
   ids=[]
   for player in players:
    if not isinstance(player,dict):continue
    ids.append(str(player.get('player_id')));stats[year]['players']+=1
    stats[year]['champion_present']+=int(bool(player.get('champion_id')))
    for key in ['kills','deaths','dpm','vspm','gd@15','csd@15','xpd@15','solo_kills','kp%']:
     val=(player.get('stats') or {}).get(key);stats[year][key+'_present']+=int(val is not None and val!='')
   if i==0:rosters.append({'id':str(m.get('match_id')),'date':date,'year':year,'team':str(game.get(side+'_id')),'players':tuple(sorted(ids))})
a=pd.DataFrame(series);a.to_csv(P/'series_inventory.csv',index=False);mm=pd.DataFrame(maps)
coverage=[]
for year,h in a.groupby('year'):
 ss=stats[year];n=ss['players'];coverage.append(dict(year=year,series=len(h),maps=int(h.maps.sum()),patch_fraction=h.patch.notna().mean(),game_count_disagree=(h.maps!=h.declared_games).sum(),**{k:v/n if n else None for k,v in ss.items() if k!='players'},player_rows=n))
pd.DataFrame(coverage).to_csv(P/'raw_coverage_year.csv',index=False)
mm.groupby(['year','map_number']).blue_won.agg(['count','mean']).reset_index().to_csv(P/'blue_side_by_year_map.csv',index=False)
r=pd.DataFrame(rosters).sort_values(['team','date','id']);changes=[]
for team,h in r.groupby('team'):
 prev=None;prevdate=None
 for row in h.itertuples():
  good=len(row.players)==5 and len(set(row.players))==5 and 'None' not in row.players
  if good and prev is not None and row.date>prevdate:changes.append({'id':row.id,'year':row.year,'team':team,'retained_players':len(set(row.players)&set(prev))})
  if good:prev=row.players;prevdate=row.date
pd.DataFrame(changes).groupby(['year','retained_players']).size().rename('team_series').reset_index().to_csv(P/'roster_continuity.csv',index=False)
print('RAW DONE',len(a),len(mm),flush=True)
# Feature and prediction archives: independent generation from Attention; do not combine by row order.
f=pd.read_parquet(D/'match_inputs_and_ratings_2020_2026.parquet');pred=pd.read_parquet(D/'model_predictions_2020_2026.parquet');f['year']=pd.to_datetime(f.date).dt.year
assert not f.golgg_match_id.duplicated().any();assert not pred.golgg_match_id.duplicated().any()
profile=[]
for col in f:
 h=f[col];z=pd.to_numeric(h,errors='coerce');profile.append(dict(column=col,n=len(h),missing=int(h.isna().sum()),unique=h.nunique(),zeros=int((z==0).sum()),nonfinite=int((~np.isfinite(z)&z.notna()).sum()),min=z.min(),median=z.median(),max=z.max()))
pd.DataFrame(profile).to_csv(P/'feature_profile.csv',index=False)
family=['elo','gl','ts','os','pl','tm'];rating=[f'{kind}_{fam}' for kind in ['player','team'] for fam in family]
f[rating].corr(method='spearman').to_csv(P/'rating_correlations.csv')
# Missingness encoded as zero: report evidence without assuming zero always means missing.
zero=[]
for year,h in f.groupby('year'):
 for col in [c for c in f if 'rolling_' in c]:zero.append({'year':year,'feature':col,'n':len(h),'zero_fraction':(h[col]==0).mean(),'mean':h[col].mean(),'std':h[col].std()})
pd.DataFrame(zero).to_csv(P/'rolling_by_year.csv',index=False)
# Chronological feature probes. Symmetric differences; no market features or model outputs.
clip=lambda p:np.clip(p,1e-6,1-1e-6)
X=pd.DataFrame(index=f.index)
for col in rating:X[col]=np.log(clip(f[col])/(1-clip(f[col])))
roll=[c.removeprefix('t1_') for c in f if c.startswith('t1_rolling_')]
for col in roll:X[col]=f['t1_'+col]-f['t2_'+col]
X['rest']=f.days_since_last_1-f.days_since_last_2
train=f.year<=2023;val=f.year==2024;test=f.year>=2025
sets={'player_glicko':['player_gl'],'player_six':['player_'+x for x in family],'player_team':rating,'player_team_w20':rating+roll,'player_team_w20_rest':rating+roll+['rest']}
probe=[];coef=[];probe_preds={};selection=[]
for name,cols in sets.items():
 xx=X[cols].to_numpy(float);assert np.isfinite(xx).all()
 scaler=StandardScaler(with_mean=False).fit(xx[train]);xx=scaler.transform(xx)
 candidates=[]
 for C in [.01,.1,1,10]:
  model=LogisticRegression(C=C,fit_intercept=False,max_iter=2000).fit(xx[train],f.loc[train,'y_true']);candidates.append((log_loss(f.loc[val,'y_true'],model.predict_proba(xx[val])[:,1]),C,model))
 score,C,model=min(candidates,key=lambda x:x[0]);selection.append(dict(probe=name,C=C,val_ll=score,train_n=int(train.sum()),val_n=int(val.sum()),test_n=int(test.sum())))
 # Keep fit fixed from 2020–2023, no refit after choosing C on 2024.
 pp=model.predict_proba(xx[test])[:,1];probe_preds[name]=pp
 for year in [2025,2026]:
  mask=f.loc[test,'year'].to_numpy()==year;y=f.loc[test,'y_true'].to_numpy()[mask];p=pp[mask]
  probe.append(dict(probe=name,year=year,n=len(y),ll=log_loss(y,p),brier=brier_score_loss(y,p),auc=roc_auc_score(y,p)))
 for col,w in zip(cols,model.coef_[0]):coef.append(dict(probe=name,feature=col,standardized_weight=w))
pd.DataFrame(probe).to_csv(P/'chronological_feature_probes.csv',index=False);pd.DataFrame(selection).to_csv(P/'probe_selection.csv',index=False);pd.DataFrame(coef).to_csv(P/'probe_coefficients.csv',index=False)
# Coverage joins by IDs only, no attaching probability to unverified sides.
a['in_features']=a.id.isin(f.golgg_match_id.astype(str));od=pd.read_csv(D/'odds.csv',usecols=['golgg_match_id'],dtype=str);a['in_historical_odds']=a.id.isin(od.golgg_match_id)
a.groupby('year').agg(series=('id','size'),features=('in_features','sum'),historical_odds=('in_historical_odds','sum')).to_csv(P/'coverage_funnel.csv')
# Live odds: timing and coverage only. Raw sides require separate canonical resolution.
o=pd.read_parquet(D/'kursy_meczowe.parquet');o['scraped']=pd.to_datetime(o.scraped_at,utc=True,errors='coerce');o['start']=pd.to_datetime(o.canonical_match_start,utc=True,errors='coerce');o['before_start']=(o.scraped<o.start)&o.start.notna();o['month']=o.scraped.dt.strftime('%Y-%m')
o.groupby('bookmaker_name').agg(rows=('id','size'),source_matches=('match_id','nunique'),canonical_matches=('canonical_match_id','nunique'),canonical_missing=('canonical_match_id',lambda h:h.isna().sum()),start_missing=('start',lambda h:h.isna().sum()),before_start=('before_start','sum'),live_rows=('is_live','sum')).to_csv(P/'live_odds_coverage.csv')
o.groupby(['month','bookmaker_name']).size().rename('snapshots').reset_index().to_csv(P/'live_odds_by_month.csv',index=False)
# Unique exact timestamps grouped within source event; not bookmaker updates count.
o=o.sort_values('scraped');keys=['bookmaker_name','match_id','market_type'];o['gap_minutes']=o.groupby(keys).scraped.diff().dt.total_seconds()/60
o.groupby('bookmaker_name').gap_minutes.quantile([.5,.9,.99]).unstack().to_csv(P/'quote_gap_quantiles.csv')
meta={'raw_source':str(raw.relative_to(ROOT)),'raw_series':len(a),'raw_maps':len(mm),'duplicate_series_ids':int(a.id.duplicated().sum()),'duplicate_game_ids':int(mm.game_id.duplicated().sum()),'features_rows':len(f),'prediction_rows':len(pred),'feature_prediction_ids_equal':set(f.golgg_match_id.astype(str))==set(pred.golgg_match_id.astype(str)),'inventory_files':len(iv),'inventory_bytes':int(iv.bytes.sum()),'live_odds_rows':len(o),'live_date_min':str(o.scraped.min()),'live_date_max':str(o.scraped.max()),'source_manifest_status':json.loads((raw.parent/'manifest.json').read_text()).get('status'),'probe_note':'Archived features not independently certified point-in-time. Models fit <=2023; C selected 2024; evaluated 2025+ without refit. No production modifications.'}
(P/'metadata.json').write_text(json.dumps(meta,indent=2))
fig,ax=plt.subplots(1,2,figsize=(13,5));cov=pd.read_csv(P/'raw_coverage_year.csv');modern=cov[pd.to_numeric(cov.year,errors='coerce')>=2014]
for c in ['dpm_present','vspm_present','gd@15_present','champion_present']:ax[0].plot(modern.year,modern[c],label=c.replace('_present',''))
ax[0].set(title='Dostępność statystyk graczy',xlabel='Rok',ylabel='Udział niepustych pól');ax[0].legend();pr=pd.DataFrame(probe).pivot(index='probe',columns='year',values='ll');pr.plot.bar(ax=ax[1]);ax[1].set(title='Chronologiczne próby cech',ylabel='LogLoss',xlabel='');ax[1].tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(P/'eda_overview.png',dpi=150)
print(json.dumps(meta));print(pd.DataFrame(probe).to_string(index=False))
