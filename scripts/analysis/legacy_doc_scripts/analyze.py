from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
f=pd.read_parquet(P/'inputs/paired_history.parquet');f=f[f.year>=2024].merge(pd.read_csv(P/'inputs/backup_crosscheck.csv',dtype={'golgg_match_id':str})[['golgg_match_id','backup_status']],on='golgg_match_id',validate='one_to_one')
books=['betclic','betfan','efortuna','lv_bet','sts','superbet'];rng=np.random.default_rng(20260911)
for timing in ['open','close']:
 for book in books:
  o1=f[f'odds1_{book}_{timing}'];o2=f[f'odds2_{book}_{timing}'];valid=o1.notna()&o2.notna()&(o1>1)&(o2>1)
  f[f'a_{book}_{timing}']=np.where(valid,np.where(f.orientation==1,o1,o2),np.nan)
  f[f'b_{book}_{timing}']=np.where(valid,np.where(f.orientation==1,o2,o1),np.nan)
 ac=[f'a_{b}_{timing}' for b in books];bc=[f'b_{b}_{timing}' for b in books]
 f[f'best_a_{timing}']=f[ac].max(axis=1);f[f'best_b_{timing}']=f[bc].max(axis=1)
 f[f'best_a_book_{timing}']=f[ac].idxmax(axis=1).str.removeprefix('a_').str.removesuffix('_'+timing)
 f[f'best_b_book_{timing}']=f[bc].idxmax(axis=1).str.removeprefix('b_').str.removesuffix('_'+timing)
 a=f[f'best_a_{timing}'];b=f[f'best_b_{timing}'];f[f'best_probability_{timing}']=b/(a+b);f[f'best_inverse_sum_{timing}']=1/a+1/b
 f[f'median_a_{timing}']=f[ac].median(axis=1);f[f'median_b_{timing}']=f[bc].median(axis=1)
 assert np.allclose(f[f'best_probability_{timing}'],(1/a)/(1/a+1/b))
 # No-vig values are synthetic benchmark only, not the executable break-even threshold 1/odds.
assert f.golgg_match_id.is_unique
loss=lambda y,p:-np.log(np.where(y==1,p,1-p))
def ci(g,delta):
 z=pd.DataFrame({'month':g.date.str[:7].to_numpy(),'delta':delta}).groupby('month').delta.agg(['sum','count']).to_numpy();ix=rng.integers(len(z),size=(5000,len(z)));a=z[ix].sum(axis=1);return np.quantile(a[:,0]/a[:,1],[.025,.975])
fullcommon=f.copy()
for b in books:
 fullcommon=fullcommon[fullcommon[f'market_{b}_open'].notna()&fullcommon[f'market_{b}_close'].notna()]
cohorts={'all_4374':f,'backup_has_prices':f[f.backup_status=='backup_has_prices'],'backup_missing_prices':f[f.backup_status=='backup_missing_prices'],'all_six_common':fullcommon[fullcommon.backup_status=='backup_has_prices']}
rows=[];tails=[];bands=[];quant=[];offers=[];series=[];plotdata={}
for cohort,g in cohorts.items():
 for book in books+['consensus','max_odds']:
  h=g if book in ['consensus','max_odds'] else g[g[f'market_{book}_open'].notna()&g[f'market_{book}_close'].notna()]
  if not len(h):continue
  for timing in ['open','close']:
   col=f'market_{book}_{timing}' if book in books else f'market_{timing}' if book=='consensus' else f'best_probability_{timing}'
   p=h[col].to_numpy();y=h.y.to_numpy();lm=loss(y,h.p_full.to_numpy());lb=loss(y,p);delta=lm-lb;n=len(h);bounds=ci(h,delta)
   rows.append(dict(cohort=cohort,book=book,timing=timing,n=n,model_ll=lm.mean(),book_ll=lb.mean(),delta=delta.mean(),ci_low=bounds[0],ci_high=bounds[1],model_better=(delta<0).mean(),model_max=lm.max(),book_max=lb.max()))
   mx=np.maximum(lm,lb)
   for lo,hi in [(0,np.log(2)),(np.log(2),1.2),(1.2,2.3),(2.3,100)]:
    mask=(mx>lo)&(mx<=hi);bands.append(dict(cohort=cohort,book=book,timing=timing,band=f'{lo:.3f}-{hi}',n=int(mask.sum()),contribution=delta[mask].sum()/n))
   for frac in [.01,.05,.1]:
    k=int(np.ceil(n*frac));mask=np.ones(n,dtype=bool);mask[np.argsort(mx)[-k:]]=False;tails.append(dict(cohort=cohort,book=book,timing=timing,fraction=frac,remaining_delta=delta[mask].mean()))
   for label,v in [('model',lm),('book',lb)]:
    for q in [.5,.9,.95,.99,1]:quant.append(dict(cohort=cohort,book=book,timing=timing,distribution=label,quantile=q,loss=np.quantile(v,q)))
   if cohort=='backup_has_prices' and timing=='close':plotdata[book]=(lm,lb)
 for timing in ['open','close']:
  a=g[f'best_a_{timing}'];b=g[f'best_b_{timing}'];inv=g[f'best_inverse_sum_{timing}']
  offers.append(dict(cohort=cohort,timing=timing,n=len(g),mixed_book_fraction=(g[f'best_a_book_{timing}']!=g[f'best_b_book_{timing}']).mean(),inverse_sum_lt1=int((inv<1-1e-10).sum()),inverse_sum_median=inv.median(),median_uplift_a=(a/g[f'median_a_{timing}']-1).median(),median_uplift_b=(b/g[f'median_b_{timing}']-1).median()))
for name,v in [('summary',rows),('tail_removal',tails),('loss_bands',bands),('quantiles',quant),('best_offer_diagnostics',offers)]:pd.DataFrame(v).to_csv(P/f'{name}.csv',index=False)
keep=['golgg_match_id','date','team1','team2','best_of','y','p_full','backup_status']+[c for c in f if c.startswith('best_') and c!='best_of']
f[keep].to_csv(P/'best_available_odds.csv',index=False);f[keep].to_parquet(P/'best_available_odds.parquet',index=False)
fig,axes=plt.subplots(2,4,figsize=(16,8))
for ax,(book,(a,b)) in zip(axes.flat,plotdata.items()):
 for label,v,color in [('Attention',a,'#2563eb'),('Rynek',b,'#ea580c')]:
  v=np.sort(v);ax.step(v,(len(v)-np.arange(len(v)))/len(v),label=label,color=color)
 ax.set(title=f'{book} · N={len(a)}',yscale='log',xlabel='LogLoss meczu',ylabel='Udział ≥ próg');ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Close: osobne firmy i MAX — kursy obecne w starszym backupie\nKażdy panel porównuje model i rynek na tych samych meczach; próby między panelami mogą się różnić.');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(P/'indywidualne_ogony.png',dpi=160)
s=pd.DataFrame(rows);print(s.query("cohort=='backup_has_prices'")[['book','timing','n','model_ll','book_ll','delta','ci_low','ci_high']].to_string(index=False));print(pd.DataFrame(offers).to_string(index=False));print('COMMON',len(cohorts['all_six_common']))
