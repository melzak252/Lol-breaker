"""Predeclared thesis ablations using corrected W20 chronology and fixed LR.

These variants diagnose the corrected W20 recipe; they are not reproductions
of the historical W50 masking experiment. No evaluation score selects a fit.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.analysis.thesis_audit import BASE_FEATURES, feature_variants, add_series_features
from src.analysis.thesis_models import replay_monthly, positive_slope_calibration


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _ids_hash(values):
    return hashlib.sha256(json.dumps(sorted(map(str,values)),separators=(',',':')).encode()).hexdigest()


def supplement_variants():
    features=feature_variants()['w20']
    team_context=['team_elo_r1','team_elo_r2','team_gl_r1','team_gl_rd1','team_gl_r2','team_gl_rd2']
    team_context += [f'team_{family}_{stat}{side}' for family in ('ts','os','pl','tm') for side in (1,2) for stat in ('mu','sigma')]
    return {
        'no_binomial':dict(features=[f for f in features if not f.endswith('_binom_series')],mask_columns=[],mask_rate=0.),
        'full_player_team':dict(features=feature_variants()['player_team']+team_context,mask_columns=[],mask_rate=0.),
        'mask_base10':dict(features=features,mask_columns=list(BASE_FEATURES),mask_rate=.10),
        'mask_playergl10':dict(features=features,mask_columns=['player_gl'],mask_rate=.10),
        'mask_playergl30':dict(features=features,mask_columns=['player_gl'],mask_rate=.30),
    }


def mask_past_training(data, columns, origin, rate, seed):
    """Mask only released TRAIN cells, keyed by stable chronological ID order."""
    if not 0<=rate<=1:
        raise ValueError('Mask rate must lie in [0,1]')
    out=data.copy()
    eligible=(pd.to_datetime(out.date)<pd.Timestamp(origin))&(pd.to_datetime(out.result_day)<pd.Timestamp(origin))
    train=out.loc[eligible].sort_values(['date','golgg_match_id'],kind='stable')
    mask=np.random.default_rng(seed).random((len(train),len(columns)))<rate
    values=train[columns].mask(mask)
    out.loc[train.index,columns]=values
    receipt=dict(seed=seed,mask_columns=columns,mask_rate=rate,eligible_rows=len(train),
                 masked_cells=int(mask.sum()),train_ids_sha256=_ids_hash(train.golgg_match_id),
                 mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return out,receipt


def calibrate_oos(predictions,min_samples=200):
    """Rebuild calibration from genuine monthly OOS predictions, never TRAIN."""
    d=predictions.copy().sort_values(['date','golgg_match_id'],kind='stable')
    rows=[];receipts=[]
    for origin,test in d.groupby('origin',sort=True):
        prior=d[(pd.to_datetime(d.date)<pd.Timestamp(origin))&(pd.to_datetime(d.result_day)<pd.Timestamp(origin))
                &(pd.to_datetime(d.date)>=pd.Timestamp('2024-01-01'))]
        available=len(prior)>=min_samples and prior.y_true.nunique()==2
        slope=positive_slope_calibration(prior.p_raw.to_numpy(),prior.y_true.to_numpy()) if available else 1.
        out=test.copy();out['p_calibrated']=expit(slope*logit(out.p_raw.to_numpy()))
        out['calibration_slope']=slope;out['calibrator_available']=available
        rows.append(out)
        receipts.append(dict(origin=origin,cal_rows=len(prior),cal_ids=prior.golgg_match_id.tolist(),
                             cal_ids_sha256=_ids_hash(prior.golgg_match_id),calibration_slope=slope,
                             calibrator_available=bool(available),
                             cal_max_date=prior.date.max().isoformat() if len(prior) else None,
                             cal_max_release=prior.result_day.max().isoformat() if len(prior) else None))
    return pd.concat(rows,ignore_index=True),receipts


def _run_variant(feature_path,params,name,config,out_dir):
    out=Path(out_dir)/name;out.mkdir(parents=True,exist_ok=False)
    data=add_series_features(pd.read_parquet(feature_path))
    effective={**params,'mask_rate':0.0}
    family=config.get('family','logistic')
    def progress(r):
        print(json.dumps({'supplement':name,'origin':r['origin'],'train':r['train_rows']}),flush=True)
    if config['mask_rate']==0:
        predictions,receipts=replay_monthly(data,config['features'],family,effective,
                                            start='2024-01-01',calibration_start='2024-01-01',progress=progress)
    else:
        predictions=[];receipts=[]
        end=pd.to_datetime(data.date).max().to_period('M').end_time
        for index,origin in enumerate(pd.date_range('2024-01-01',end,freq='MS')):
            next_origin=origin+pd.offsets.MonthBegin(1)
            if not ((pd.to_datetime(data.date)>=origin)&(pd.to_datetime(data.date)<next_origin)).any():
                continue
            masked,mask_receipt=mask_past_training(data,config['mask_columns'],origin,config['mask_rate'],20260921+index)
            p,rs=replay_monthly(masked,config['features'],family,effective,
                                start=str(origin.date()),end=str(next_origin.date()),
                                calibration_start='2024-01-01',progress=progress)
            rs[0]['masking']=mask_receipt
            receipts.extend(rs);predictions.append(p)
        predictions,cal_receipts=calibrate_oos(pd.concat(predictions,ignore_index=True))
        for fit,cal in zip(receipts,cal_receipts,strict=True):
            if fit['origin']!=cal['origin']:
                raise RuntimeError('Calibration origin mismatch')
            fit.update(cal)
    predictions.to_parquet(out/'predictions.parquet',index=False)
    (out/'fit_receipts.json').write_text(json.dumps(receipts,separators=(',',':'))+'\n')
    result=dict(name=name,rows=len(predictions),folds=len(receipts),
                predictions_sha256=_hash(out/'predictions.parquet'),receipts_sha256=_hash(out/'fit_receipts.json'))
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run_supplements(feature_path,selection_path,out_dir,workers=2):
    if workers not in (1,2):
        raise ValueError('Use at most two supplement workers')
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(Path(selection_path).read_text())
    params=selection['selected']['logistic']
    variants=supplement_variants()
    primary=selection['primary_family']
    variants['primary_no_context']=dict(features=feature_variants()['player_only'],mask_columns=[],mask_rate=0.,family=primary)
    variants['primary_no_binomial']=dict(features=variants['no_binomial']['features'],mask_columns=[],mask_rate=0.,family=primary)
    paths=[Path(feature_path),Path(selection_path),Path(__file__),
           Path(__file__).with_name('thesis_models.py'),Path(__file__).with_name('thesis_audit.py'),
           Path(__file__).parents[1]/'models'/'team_order.py']
    protocol=dict(started_at=datetime.now(timezone.utc).isoformat(),sources={str(p):_hash(p) for p in paths},
                  variants=variants,params=params,primary_family=primary,primary_params=selection['selected'][primary],workers=workers,
                  status='PREDECLARED_BEFORE_SUPPLEMENT_FITS',
                  interpretation='Corrected W20 diagnostic ablations; original masking recipe used W50 without binomial.',
                  selection='LR and primary family parameters selected on2023 only; no supplementary selection.',
                  training='All strict earlier dates and released labels; monthly2024+; test inputs unmasked.',
                  calibration='Only strict earlier released2024+ genuine OOS; positive zero-intercept slope; min200.',
                  mask_seed='20260921 plus months since2024-01; stable date+ID TRAIN order')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_run_variant,str(feature_path),selection['selected'][config.get('family','logistic')],name,config,str(output)) for name,config in variants.items()]
        for future in as_completed(futures):
            result=future.result();results.append(result);print(json.dumps({'supplement_completed':result}),flush=True)
    if any(_hash(p)!=digest for p,digest in protocol['sources'].items()):
        raise RuntimeError('Supplement input or source changed during fits')
    (output/'COMPLETE.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',required=True);parser.add_argument('--selection',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    run_supplements(args.features,args.selection,args.output,args.workers)
