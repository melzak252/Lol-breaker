from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent.parent);args=ap.parse_args();P=Path(__file__).resolve().parent
f=pd.read_parquet(args.assets/'inputs/paired_history.parquet');f=f[f.year>=2024].merge(pd.read_csv(args.assets/'inputs/backup_crosscheck.csv',dtype={'golgg_match_id':str})[['golgg_match_id','backup_status']],on='golgg_match_id',validate='one_to_one')
loss=lambda y,p: -np.log(np.where(np.asarray(y)==1,np.asarray(p),1-np.asarray(p)))
assert f.golgg_match_id.is_unique
for c in ['p_full','market_close','market_open','p_no_w20','p_no_organization','baseline_full']:
 assert f[c].between(0,1,inclusive='neither').all()
 f['ll_'+c]=loss(f.y,f[c])
f['delta']=f.ll_p_full-f.ll_market_close
books=['betclic','betfan','efortuna','lv_bet','sts','superbet']
cols=[f'market_{b}_close' for b in books]
f['book_range']=f[cols].max(axis=1)-f[cols].min(axis=1)
f['model_conf']=np.maximum(f.p_full,1-f.p_full);f['model_correct']=(f.p_full>=.5)==f.y
f['movement']=abs(f.market_close-f.market_open)
f['novice']=pd.cut(f.min_player_series,[-1,0,9,29,np.inf],labels=['0','1–9','10–29','30+'])
f['w20_support']=pd.cut(f.w20_min_valid_games,[-1,0,9,19,np.inf],labels=['0','1–9','10–19','20+'])
f['confidence']=pd.cut(f.model_conf,[.499999,.6,.7,.8,.9,1],labels=['50–60%','60–70%','70–80%','80–90%','90–100%'])
f['dispersion']=pd.cut(f.book_range,[-1,.03,.08,.15,1],labels=['≤3pp','3–8pp','8–15pp','>15pp'])
f['movement_band']=pd.cut(f.movement,[-1,.02,.05,.1,1],labels=['≤2pp','2–5pp','5–10pp','>10pp'])
f['year_format']=f.year.astype(str)+' / BO'+f.best_of.astype(str)
f['league_family']=f.tournament_name.str.extract(r'^(EMEA Masters|EU Masters|Prime League|Hitpoint Masters|LVP SL|CBLOL|LPL|LCK|LEC|LCS|LFL|TCL|PCS|VCS|LJL|NLC|LCP)',expand=False).fillna('Other / unmapped')
g=f[f.backup_status=='backup_has_prices'].copy();rng=np.random.default_rng(20260912)
def ci(h,d):
 if h.date.str[:7].nunique()<6:return np.array([np.nan,np.nan])
 z=pd.DataFrame({'month':h.date.str[:7].to_numpy(),'v':np.asarray(d)}).groupby('month').v.agg(['sum','count']).to_numpy();ix=rng.integers(len(z),size=(3000,len(z)));a=z[ix].sum(axis=1);return np.quantile(a[:,0]/a[:,1],[.025,.975])
rows=[]
for col in ['year','year_format','league_family','best_of','novice','unknown_players','new_organization','w20_support','confidence','dispersion','movement_band','tournament_name']:
 for key,h in g.groupby(col,observed=True,dropna=False):
  low,high=ci(h,h.delta)
  rows.append(dict(dimension=col,group=str(key),n=len(h),months=h.date.str[:7].nunique(),model_ll=h.ll_p_full.mean(),market_ll=h.ll_market_close.mean(),delta=h.delta.mean(),contribution=h.delta.sum()/len(g),ci_low=low,ci_high=high,confidence=h.model_conf.mean(),accuracy=h.model_correct.mean()))
s=pd.DataFrame(rows);s.to_csv(P/'subgroups.csv',index=False)
# Favorite-oriented calibration preserves team-side symmetry; Wilson intervals for empirical wins.
cal=[]
for name in ['p_full','market_open','market_close']:
 conf=np.maximum(g[name],1-g[name]); correct=((g[name]>=.5)==g.y).astype(int)
 for lo,hi in zip([.5,.6,.7,.8,.9],[.6,.7,.8,.9,1.000001]):
  mask=(conf>=lo)&(conf<hi);n=int(mask.sum());q=correct[mask].mean();z=1.96;den=1+z*z/n;center=(q+z*z/(2*n))/den;half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
  cal.append(dict(model=name,lo=lo,hi=min(hi,1),n=n,predicted=conf[mask].mean(),observed=q,ci_low=center-half,ci_high=center+half))
pd.DataFrame(cal).to_csv(P/'calibration.csv',index=False)
# Fit symmetric temperature on 2024 only, assess frozen transform in 2025 and 2026.
train=g[g.year==2024];test=g[g.year>=2025];assert train.date.max()<test.date.min()
T=minimize_scalar(lambda t:loss(train.y,expit(logit(train.p_full)/t)).mean(),bounds=(.5,3),method='bounded').x
out=[]
for label,h in [('2025+',test),('2025',test[test.year==2025]),('2026',test[test.year==2026])]:
 for name,p in [('full',h.p_full),('temperature_2024',expit(logit(h.p_full)/T)),('no_w20',h.p_no_w20),('no_w20_no_org',h.p_no_organization),('linear_full',h.baseline_full)]:
  l=loss(h.y,p);d=l-h.ll_p_full;lo,hi=ci(h,d)
  out.append(dict(period=label,variant=name,n=len(h),ll=l.mean(),delta_vs_full=d.mean(),ci_low=lo,ci_high=hi,delta_vs_market=(l-h.ll_market_close).mean(),brier=np.mean((h.y-p)**2),max_loss=l.max()))
pd.DataFrame(out).to_csv(P/'chronological_variants.csv',index=False)
# Annual descriptive ablations and support interaction, no new fit.
abl=[]
for dim in ['year','best_of','w20_support','novice']:
 for key,h in g.groupby(dim,observed=True):
  for name in ['p_no_w20','p_no_organization','baseline_full']:
   d=h['ll_'+name]-h.ll_p_full;lo,hi=ci(h,d)
   abl.append(dict(dimension=dim,group=str(key),variant=name,n=len(h),delta_vs_full=d.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(abl).to_csv(P/'ablation_subgroups.csv',index=False)
# Cross-check league/year and W20 to avoid treating overlapping slices as additive.
inter=[]
for (league,year),h in g.groupby(['league_family','year']):
 inter.append(dict(league=league,year=int(year),n=len(h),delta=h.delta.mean(),no_w20_delta=(h.ll_p_no_w20-h.ll_p_full).mean()))
pd.DataFrame(inter).to_csv(P/'league_year.csv',index=False)
# Source audit flags selected without outcome or model agreement.
flags=[]
for timing in ['open','close']:
 for book in books:
  a=f[f'odds1_{book}_{timing}'];b=f[f'odds2_{book}_{timing}'];valid=a.notna()&b.notna()&(a>1)&(b>1);inv=1/a+1/b
  peers=f[[f'market_{bb}_{timing}' for bb in books if bb!=book]];med=peers.median(axis=1);p=f[f'market_{book}_{timing}'];difference=abs(p-med)
  tests={'inverse_sum_below_1':valid&(inv<1-1e-10),'inverse_sum_above_1_30':valid&(inv>1.30),'outlier_vs_3_peers_25pp':valid&(peers.notna().sum(axis=1)>=3)&(difference>.25)}
  for flag,mask in tests.items():
   for i in f.index[mask]: flags.append(dict(golgg_match_id=f.at[i,'golgg_match_id'],date=f.at[i,'date'],team1=f.at[i,'team1'],team2=f.at[i,'team2'],backup_status=f.at[i,'backup_status'],book=book,timing=timing,flag=flag,raw_odds1=a[i],raw_odds2=b[i],inverse_sum=inv[i],p_aligned=p[i],peer_median=med[i],difference=difference[i]))
flags=pd.DataFrame(flags);flags.to_csv(P/'data_audit_queue.csv',index=False)
flags.groupby(['backup_status','timing','flag']).agg(rows=('golgg_match_id','size'),matches=('golgg_match_id','nunique')).to_csv(P/'data_audit_counts.csv')
# Flag sensitivity does not identify which book is wrong.
flag_ids=set(flags.golgg_match_id);h=g[~g.golgg_match_id.isin(flag_ids)]
pd.DataFrame([dict(cohort='primary_without_all_flagged_matches',n=len(h),delta=h.delta.mean(),ci_low=ci(h,h.delta)[0],ci_high=ci(h,h.delta)[1])]).to_csv(P/'flag_sensitivity.csv',index=False)
# Rank review cases by paired loss, keep metadata; not a rule to drop rows.
g.sort_values('delta',ascending=False).head(100)[['golgg_match_id','date','team1','team2','tournament_name','best_of','p_full','market_open','market_close','y','delta','min_player_series','unknown_players','new_organization','w20_min_valid_games','book_range']].to_csv(P/'100_model_errors_for_review.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
cc=pd.DataFrame(cal)
for name,label in [('p_full','Model'),('market_close','Rynek close')]:
 h=cc[cc.model==name];axes[0].plot(h.predicted,h.observed,'o-',label=label)
axes[0].plot([.5,1],[.5,1],'--',color='gray');axes[0].set(xlabel='Średnia pewność faworyta',ylabel='Rzeczywisty udział zwycięstw',title='Kalibracja: 3755 meczów');axes[0].legend()
h=s[s.dimension=='best_of'];axes[1].bar(h.group,h.delta);axes[1].axhline(0,color='gray');axes[1].set(xlabel='Format BO',ylabel='LogLoss model − rynek close',title='Różnica według formatu serii')
fig.tight_layout();fig.savefig(P/'diagnostyka_modelu.png',dpi=160)
meta={'n':len(g),'train_n':len(train),'test_n':len(test),'temperature_fit_2024':float(T),'train_max':train.date.max(),'test_min':test.date.min(),'flagged_unique_matches':int(flags.golgg_match_id.nunique()),'notes':'Exploratory subgroups; CI no multiplicity adjustment. No correction of source values. Calibration fits only 2024.'};(P/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta));print(s[s.dimension.isin(['best_of','novice','w20_support','confidence','new_organization','dispersion','movement_band'])].to_string(index=False));print(pd.DataFrame(out).to_string(index=False));print(flags.groupby(['backup_status','flag']).golgg_match_id.nunique())
