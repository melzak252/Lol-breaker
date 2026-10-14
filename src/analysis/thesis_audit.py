"""Reproducible corrected thesis experiment, exposed by run_model_benchmark.

All paths are explicit. Historical artifacts are inputs, never output defaults.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from math import comb
from pathlib import Path
import hashlib
import json
import multiprocessing
import platform

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.analysis.thesis_models import (RATING_NAMES, FROZEN_PARAMS, default_candidate_grid,
    select_development_configuration, replay_monthly, positive_slope_calibration)
from src.analysis.thesis_evaluation import join_prediction_streams, attach_market, evaluate_thesis
from src.analysis.thesis_identity import align_identity_names

BASE_FEATURES=[f'player_{x}' for x in RATING_NAMES]+[
    'player_elo_min1','player_elo_min2','player_gl_max1','player_gl_max2',
    'player_gl_rd_avg1','player_gl_rd_avg2',
    *[f'player_{x}_sigma_avg{s}' for x in ('ts','os','pl','tm') for s in (1,2)]]
STATS=('win_rate','kills','deaths','gd15','dpm','vspm','towers','nashors','gold','duration')


def feature_variants():
    variants={f'w{w}':BASE_FEATURES+[f't{s}_rolling_{stat}_w{w}' for stat in STATS for s in (1,2)]
              +[f'player_{x}_binom_series' for x in RATING_NAMES] for w in (3,5,10,20,50)}
    full=variants['w20']
    variants['player_only']=[f for f in full if not f.startswith('t1_') and not f.startswith('t2_')]
    variants['glicko_context']=[f for f in full if f.startswith(('player_gl','t1_','t2_'))]
    variants['no_player_gl']=[f for f in full if f!='player_gl']
    variants['no_glicko']=[f for f in full if not f.startswith('player_gl')]
    variants['player_team']=full+[f'team_{x}' for x in RATING_NAMES]+[f'team_{x}_binom_series' for x in RATING_NAMES]
    return variants


def add_series_features(data):
    d=data.copy()
    for scope in ('player','team'):
        for family in RATING_NAMES:
            p=d[f'{scope}_{family}'].to_numpy(float); out=p.copy()
            for n in (3,5):
                adjusted=sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(n//2+1,n+1))
                out=np.where(d.BoN.to_numpy()==n,adjusted,out)
            d[f'{scope}_{family}_binom_series']=out
    return d


def _hash(path):
    with Path(path).open('rb') as h: return hashlib.file_digest(h,'sha256').hexdigest()


def _job(feature_path, features, family, params, name, output):
    out=Path(output)/'models'/name;out.mkdir(parents=True,exist_ok=False)
    data=add_series_features(pd.read_parquet(feature_path))
    def progress(r):
        print(json.dumps({'variant':name,'origin':r['origin'],'train':r['train_rows'],'test':r['test_rows']}),flush=True)
    predictions,receipts=replay_monthly(data,features,family,params,start='2024-01-01',
                                       calibration_start='2024-01-01',progress=progress)
    predictions.to_parquet(out/'predictions.parquet',index=False)
    (out/'fit_receipts.json').write_text(json.dumps(receipts,separators=(',',':'))+'\n')
    (out/'complete.json').write_text(json.dumps({'variant':name,'folds':len(receipts),
        'prediction_sha256':_hash(out/'predictions.parquet'),'receipt_sha256':_hash(out/'fit_receipts.json')}))
    return name


def rating_stream(data, name):
    """Compare a projected map-rating baseline with identical past-only calibration."""
    frame=data[pd.to_datetime(data.date)>=pd.Timestamp('2024-01-01')].copy()
    frame['p_raw']=np.clip(frame[name+'_binom_series'],1e-6,1-1e-6)
    rows=[]
    for origin,group in frame.groupby(pd.to_datetime(frame.date).dt.to_period('M'),sort=True):
        prior=frame[(pd.to_datetime(frame.date)<origin.start_time)&(pd.to_datetime(frame.result_day)<origin.start_time)]
        available=len(prior)>=200 and prior.y_true.nunique()==2
        slope=positive_slope_calibration(prior.p_raw.to_numpy(),prior.y_true.to_numpy()) if available else 1.
        out=group[['golgg_match_id','date','result_day','y_true','p_raw']].copy()
        out['p_calibrated']=expit(slope*logit(out.p_raw.to_numpy()))
        out['calibration_slope']=slope;out['origin']=str(origin);out['cal_rows']=len(prior)
        rows.append(out)
    return pd.concat(rows,ignore_index=True)


def run_thesis_audit(feature_path, market_path, identities_path, output_dir, workers=2):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    data=add_series_features(pd.read_parquet(feature_path))
    features=feature_variants()
    paths=[Path(feature_path),Path(market_path),Path(identities_path),Path(__file__),
           *[Path(__file__).with_name(name) for name in ('thesis_models.py','thesis_evaluation.py',
                'thesis_identity.py','thesis_metrics.py','metrics.py')],
           Path(__file__).parents[1]/'models'/'team_order.py']
    protocol={'started_at':datetime.now(timezone.utc).isoformat(),'sources':{str(p):_hash(p) for p in paths},
              'python':platform.python_version(),'candidates':default_candidate_grid(),'features':features,
              'development':'2023, strict earlier full sports history TRAIN; label release before2024',
              'evaluation':'monthly2024 onward; historical development, not untouched holdout',
              'calibration':'positive zero-intercept slope; only prior2024+ OOS released labels; min200',
              'primary_rule':'lowest2023 development logloss across declared family candidates',
              'ablation_rule':'LR selected2023 parameters fixed across W3/5/10/20/50,context,Glicko andteam variants',
              'probability_interpretation':'map-strength proxy; binomial IID approximation supplied once as a feature',
              'workers':workers}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    selection=select_development_configuration(data,features['w20'])
    winner=min(selection['candidates'],key=lambda r:r['logloss'])
    selection['primary_family']=winner['family']
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    protocol['selection_sha256']=_hash(out/'selection.json')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'selection_complete':True,'primary_family':winner['family']}),flush=True)
    jobs=[(family,family,features['w20'],params) for family,params in selection['selected'].items()]
    jobs += [(f'lr_{name}','logistic',columns,selection['selected']['logistic']) for name,columns in features.items() if name!='w20']
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_job,str(feature_path),columns,family,params,name,str(out)) for name,family,columns,params in jobs]
        for future in as_completed(futures): print(json.dumps({'completed_variant':future.result()}),flush=True)
    evaluation=data[pd.to_datetime(data.date)>=pd.Timestamp('2024-01-01')].copy()
    streams={name:pd.read_parquet(out/'models'/name/'predictions.parquet') for name,_,_,_ in jobs}
    # Raw and calibrated main-family streams are both reported: no calibration cherry-picking.
    streams.update({f'{family}_raw':streams[family].assign(p_calibrated=streams[family].p_raw) for family in selection['selected']})
    for scope in ('player','team'):
        for family in RATING_NAMES:
            name=f'{scope}_{family}'
            p=rating_stream(data,name)
            directory=out/'models'/name;directory.mkdir(parents=True)
            p.to_parquet(directory/'predictions.parquet',index=False)
            streams[name]=p
    identities=pd.DataFrame(json.loads(Path(identities_path).read_text()))
    evaluation=align_identity_names(evaluation,identities)
    evaluation['competition']=evaluation.tournament_name
    panel=join_prediction_streams(evaluation,streams)
    odds=pd.read_csv(market_path,dtype={'golgg_match_id':str})
    panel,market_audit=attach_market(panel,odds)
    (out/'market_audit.json').write_text(json.dumps(market_audit,indent=2)+'\n')
    summary=evaluate_thesis(panel,out/'evaluation',primary='p_'+winner['family'])
    if any(_hash(p)!=digest for p,digest in protocol['sources'].items()):
        raise RuntimeError('Input/source changed during execution')
    (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
