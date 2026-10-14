from pathlib import Path
import pandas as pd,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
P=Path(__file__).parent;f=pd.read_parquet(P.parents[2]/'data/match_inputs_and_ratings_2020_2026.parquet');f['year']=pd.to_datetime(f.date).dt.year
families=['elo','gl','ts','os','pl','tm'];ratings=[f'{k}_{x}' for k in ['player','team'] for x in families];X=pd.DataFrame(index=f.index)
for col in ratings:X[col]=np.log(f[col]/(1-f[col]))
roll=[c.removeprefix('t1_') for c in f if c.startswith('t1_rolling_')]
for col in roll:X[col]=f['t1_'+col]-f['t2_'+col]
X['uncertainty_x_gl']=f.player_gl_rd_avg1.add(f.player_gl_rd_avg2)*X.player_gl
X['experience_x_gl']=np.log1p(f.roster_min_prior_series)*X.player_gl
sets={'ratings':ratings,'ratings_winrate':ratings+['rolling_win_rate'],'ratings_economy':ratings+['rolling_gd15','rolling_gold'],'ratings_combat':ratings+['rolling_kills','rolling_deaths','rolling_dpm'],'ratings_objectives':ratings+['rolling_towers','rolling_nashors'],'ratings_vision_duration':ratings+['rolling_vspm','rolling_duration'],'ratings_w20':ratings+roll,'ratings_uncertainty_experience':ratings+['uncertainty_x_gl','experience_x_gl']}
tr=f.year<=2023;va=f.year==2024;te=f.year>=2025;preds={};out=[]
for name,cols in sets.items():
 x=X[cols].to_numpy();assert np.isfinite(x).all();x=StandardScaler(with_mean=False).fit(x[tr]).transform(x);best=None
 for C in [.01,.1,1,10]:
  m=LogisticRegression(C=C,fit_intercept=False,max_iter=2000).fit(x[tr],f.loc[tr,'y_true']);score=log_loss(f.loc[va,'y_true'],m.predict_proba(x[va])[:,1])
  if best is None or score<best[0]:best=(score,C,m)
 pp=best[2].predict_proba(x[te])[:,1];preds[name]=pp
 for yr in [2025,2026]:
  mask=f.loc[te,'year'].to_numpy()==yr;out.append(dict(probe=name,year=yr,n=int(mask.sum()),C=best[1],val_ll=best[0],test_ll=log_loss(f.loc[te,'y_true'].to_numpy()[mask],pp[mask])))
pd.DataFrame(out).to_csv(P/'targeted_feature_probes.csv',index=False)
h=f.loc[te,['date','golgg_match_id','y_true','year']].copy();y=h.y_true.to_numpy();base=-np.log(np.where(y==1,preds['ratings'],1-preds['ratings']));rng=np.random.default_rng(92341);rows=[]
for name,pp in preds.items():
 d=-np.log(np.where(y==1,pp,1-pp))-base;z=pd.DataFrame({'month':h.date.astype(str).str[:7].to_numpy(),'d':d}).groupby('month').d.agg(['sum','count']).to_numpy();ix=rng.integers(len(z),size=(5000,len(z)));a=z[ix].sum(axis=1);ci=np.quantile(a[:,0]/a[:,1],[.025,.975]);rows.append(dict(probe=name,n=len(h),delta=d.mean(),ci_low=ci[0],ci_high=ci[1]));h[name]=pp
pd.DataFrame(rows).to_csv(P/'probe_paired_uncertainty.csv',index=False);h.to_parquet(P/'probe_predictions.parquet',index=False)
print(pd.DataFrame(out).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
