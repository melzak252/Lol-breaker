"""Strict joins and descriptive evidence for the corrected thesis replay.

This module does not select models. Archived OPEN/CLOSE labels do not establish
quote-time information availability; all market comparisons are retrospective.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src.analysis.thesis_metrics import _metrics
from src.analysis.metrics import binary_log_loss_vector


def _unique(frame):
    d=frame.copy()
    if d.golgg_match_id.isna().any():
        raise ValueError('Missing match ID')
    d['golgg_match_id']=d.golgg_match_id.astype(str)
    if d.golgg_match_id.duplicated().any():
        raise ValueError('Duplicate match ID')
    return d.set_index('golgg_match_id',drop=False)


def join_prediction_streams(metadata, streams, probability='p_calibrated'):
    """Require exact coverage, labels and dates, without joining by row position."""
    d=_unique(metadata)
    for name,frame in streams.items():
        p=_unique(frame)
        if set(p.index)!=set(d.index):
            raise ValueError(f'Forecast coverage differs: {name}')
        p=p.loc[d.index]
        if not np.array_equal(d.y_true.to_numpy(),p.y_true.to_numpy()):
            raise ValueError(f'Forecast label disagreement: {name}')
        if not np.array_equal(pd.to_datetime(d.date).to_numpy(),pd.to_datetime(p.date).to_numpy()):
            raise ValueError(f'Forecast date disagreement: {name}')
        for side in ('team1_id','team2_id'):
            if side in p and side in d and not d[side].astype(str).equals(p[side].astype(str)):
                raise ValueError(f'Forecast identity disagreement: {name}')
        v=p[probability].to_numpy(dtype=float)
        if not (np.isfinite(v)&(v>=0)&(v<=1)).all():
            raise ValueError(f'Invalid probabilities: {name}')
        d[f'p_{name}']=v
    return d.reset_index(drop=True)


def _name(x):
    return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''


def attach_market(metadata, odds):
    """Align independently mapped sides; result conflicts veto rather than orient."""
    d=_unique(metadata).reset_index(drop=True)
    o=_unique(odds).reset_index(drop=True)
    d=d.merge(o,on='golgg_match_id',how='left',validate='one_to_one',suffixes=('','_odds'))
    a,b=d.team1_name.map(_name),d.team2_name.map(_name)
    c,e=d.golgg_team1.map(_name),d.golgg_team2.map(_name)
    same=(a==c)&(b==e)&(a!='')&(a!=b)
    swapped=(a==e)&(b==c)&(a!='')&(a!=b)
    valid=(same|swapped)
    d['market_alignment']=np.select([same,swapped],['same','swapped'],default='unresolved')
    day_ok=(pd.to_datetime(d.date)-pd.to_datetime(d.golgg_date)).abs().dt.days<=1
    # bool strings must never use Python's bool('False').
    result=d.t1_win.map({True:1,False:0,'True':1,'False':0,'true':1,'false':0,1:1,0:0})
    oriented=np.where(swapped,1-result,result)
    outcome_ok=oriented==d.y_true
    valid &= day_ok & outcome_ok
    d['market_identity_valid']=valid
    for horizon,left,right in [('open','avg_open_home','avg_open_away'),('close','avg_odds_home','avg_odds_away')]:
        x=pd.to_numeric(d[left],errors='coerce'); y=pd.to_numeric(d[right],errors='coerce')
        prices=np.isfinite(x)&np.isfinite(y)&(x>1)&(y>1)
        raw=y/(x+y)
        d[f'market_{horizon}']=np.where(valid&prices,np.where(swapped,1-raw,raw),np.nan)
    audit={'target_rows':len(d),'mapped_rows':int(d.golgg_team1.notna().sum()),
           'identity_valid':int(valid.sum()),'outcome_conflicts':int(((same|swapped)&~outcome_ok).sum()),
           'date_conflicts':int(((same|swapped)&~day_ok).sum()),
           'open_rows':int(d.market_open.notna().sum()),'close_rows':int(d.market_close.notna().sum()),
           'open_close_common':int(d[['market_open','market_close']].notna().all(axis=1).sum()),
           'quote_time_certified':False}
    return d,audit


def block_difference(delta, blocks, draws=5000, seed=20260921):
    """Paired cluster intervals for two explicitly different estimands."""
    v=np.asarray(delta,float); _,idx=np.unique(np.asarray(blocks,str),return_inverse=True)
    count=np.bincount(idx); sums=np.bincount(idx,weights=v); means=sums/count
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(count),size=(draws,len(count)))
    weighted=sums[sampled].sum(axis=1)/count[sampled].sum(axis=1)
    equal=means[sampled].mean(axis=1)
    lo,hi=np.quantile(weighted,[.025,.975]) if len(count)>1 else (np.nan,np.nan)
    elo,ehi=np.quantile(equal,[.025,.975]) if len(count)>1 else (np.nan,np.nan)
    return dict(n=len(v),blocks=len(count),delta=float(v.mean()),ci_lower=float(lo),ci_upper=float(hi),
                equal_block_delta=float(means.mean()),equal_block_ci_lower=float(elo),equal_block_ci_upper=float(ehi),
                bootstrap_fraction_nonnegative=float((weighted>=0).mean()),draws=draws,seed=seed)


def evaluate_thesis(panel, out_dir, primary='p_logistic', draws=5000):
    """Score every stream on full and identical market cohorts; save slices/CIs."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    d=_unique(panel).reset_index(drop=True)
    if not d.y_true.isin([0,1]).all(): raise ValueError('Nonbinary target')
    models=[c for c in d if c.startswith('p_')]
    metric_rows=[]; bins=[]; comparisons=[]; slices=[]
    cohorts={'sports':d,'market_common':d[d[['market_open','market_close']].notna().all(axis=1)]}
    for cohort,frame in cohorts.items():
        if frame.empty: continue
        columns=models+(['market_open','market_close'] if cohort=='market_common' else [])
        y=frame.y_true.to_numpy(float)
        losses={}
        for model in columns:
            p=frame[model].to_numpy(float)
            if not (np.isfinite(p)&(p>=0)&(p<=1)).all(): raise ValueError(f'Invalid {model}')
            m,rel=_metrics(y,p)
            for n_bins in [5,15]:
                ix=np.minimum((p*n_bins).astype(int),n_bins-1)
                m[f'ece_{n_bins}']=float(sum(abs((p[ix==b]-y[ix==b]).sum())/len(y) for b in range(n_bins)))
            metric_rows.append(dict(cohort=cohort,model=model,**m))
            bins.extend(dict(cohort=cohort,model=model,**r) for r in rel)
            losses[model]={'logloss':binary_log_loss_vector(y,p),'brier':(p-y)**2}
        references=[primary]+[c for c in ['p_player_gl','market_open','market_close'] if c in columns and c!=primary]
        for reference in references:
            for candidate in columns:
                if candidate==reference: continue
                for metric in ['logloss','brier']:
                    delta=losses[candidate][metric]-losses[reference][metric]
                    blocks={'month':pd.to_datetime(frame.date).dt.strftime('%Y-%m').to_numpy()}
                    if 'competition' in frame:
                        # Names plus year are a sensitivity proxy, not certified edition IDs.
                        blocks['tournament_year_proxy']=(pd.to_datetime(frame.date).dt.year.astype(str)+':'+frame.competition.fillna('unknown').astype(str)).to_numpy()
                    for unit,labels in blocks.items():
                        comparisons.append(dict(cohort=cohort,candidate=candidate,reference=reference,metric=metric,
                                                blocking=unit,**block_difference(delta,labels,draws)))
        dimensions={'year':pd.to_datetime(frame.date).dt.year.astype(str)}
        for col in ['BoN','competition_tier']:
            if col in frame: dimensions[col]=frame[col].fillna('unknown').astype(str)
        for dimension,labels in dimensions.items():
            for label in labels.unique():
                mask=(labels==label).to_numpy()
                for model in columns:
                    m,_=_metrics(y[mask],frame.loc[mask,model].to_numpy(float))
                    slices.append(dict(cohort=cohort,dimension=dimension,slice=label,model=model,**m))
    for name,rows in [('metrics',metric_rows),('reliability',bins),('comparisons',comparisons),('slices',slices)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    d.to_parquet(out/'paired_predictions.parquet',index=False)
    summary={'status':'HISTORICAL_DEVELOPMENT_ONLY','cohorts':{k:len(v) for k,v in cohorts.items()},
             'models':models,'primary':primary,'draws':draws,'ece_bins':'equal width 10 primary; 5/15 sensitivity',
             'probability_policy':'numeric logloss clip 1e-15, original probabilities for Brier/ECE',
             'limitations':['Repeated historical model development, not prospective confirmation.',
                            'First-map roster known scenario; precise release and quote times uncertified.',
                            'Bootstrap intervals are unadjusted for model selection and multiple comparisons.',
                            'Tournament-year blocking is a sensitivity proxy, not verified edition IDs.',
                            'No betting profitability or deployed-model certification.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
