"""Verify completed thesis streams, align IDs, and regenerate final evidence.

This is reporting only: no training, tuning, filtering by forecast loss or changes
of selected family. Both supplement and main runs must be complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.thesis_audit import add_series_features,rating_stream,feature_variants
from src.analysis.thesis_identity import align_identity_names
from src.analysis.thesis_evaluation import attach_market,join_prediction_streams,evaluate_thesis,block_difference
from src.analysis.metrics import binary_log_loss_vector


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_evidence(main_run,supplements,features,market,identities,main_names,supplement_names):
    """Validate immutable input/selection and per-stream receipts before scoring."""
    main_run,supplements=Path(main_run),Path(supplements)
    main=json.loads((main_run/'protocol.json').read_text())
    supplement=json.loads((supplements/'protocol.json').read_text())
    checked={}
    # Data must match the original protocol. Code provenance stays recorded in
    # that protocol; historical code need not remain at its working-tree path.
    for path in map(Path,(features,market,identities)):
        pinned=main['sources'].get(str(path.resolve()),main['sources'].get(str(path)))
        if pinned is None or _sha(path)!=pinned:
            raise ValueError(f'Changed or unpinned finalizer input: {path}')
        checked[str(path.resolve())]=pinned
    feature_hash=_sha(features)
    if not any(Path(p).name=='features.parquet' and digest==feature_hash for p,digest in supplement['sources'].items()):
        raise ValueError('Supplement uses a different or unpinned feature input')
    selection_path=main_run/'selection.json';selection_hash=_sha(selection_path)
    pins=[digest for path,digest in supplement['sources'].items() if Path(path).name=='selection.json']
    if not pins or any(digest!=selection_hash for digest in pins):
        raise ValueError('Main and supplement selection hashes disagree')
    if 'selection_sha256' in main and main['selection_sha256']!=selection_hash:
        raise ValueError('Resumed main selection hash disagrees')
    selected=json.loads(selection_path.read_text())
    if (supplement['params']!=selected['selected']['logistic']
        or supplement['primary_family']!=selected['primary_family']
        or supplement['primary_params']!=selected['selected'][selected['primary_family']]):
        raise ValueError('Supplement parameters disagree with selected main parameters')
    streams=[]
    for root,names,keys in ((main_run/'models',main_names,('prediction_sha256','receipt_sha256')),
                            (supplements,supplement_names,('predictions_sha256','receipts_sha256'))):
        for name in names:
            directory=root/name;completion=directory/'complete.json'
            if not completion.is_file():raise ValueError(f'Incomplete variant: {name}')
            receipt=json.loads(completion.read_text());files={}
            for filename,key in zip(('predictions.parquet','fit_receipts.json'),keys,strict=True):
                path=directory/filename
                if not path.is_file() or receipt.get(key)!=_sha(path):
                    raise ValueError(f'Changed completed evidence: {name}/{filename}')
                files[filename]=receipt[key]
            streams.append(dict(name=name,directory=str(directory.resolve()),files=files,completion_sha256=_sha(completion)))
    return dict(status='PASS',inputs=checked,selection_sha256=selection_hash,streams=streams,
                main_protocol_sha256=_sha(main_run/'protocol.json'),
                supplement_protocol_sha256=_sha(supplements/'protocol.json'),
                finalizer_source_sha256=_sha(__file__))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('features','market','identities','main-run','supplements','output-dir'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-market-rows',type=int,default=None)
    a=ap.parse_args()
    selection=json.loads((a.main_run/'selection.json').read_text())
    expected=list(selection['selected'])+[f'lr_{x}' for x in feature_variants() if x!='w20']
    supplement_names=['no_binomial','full_player_team','mask_base10','mask_playergl10','mask_playergl30','primary_no_context','primary_no_binomial']
    verification=verify_evidence(a.main_run,a.supplements,a.features,a.market,a.identities,expected,supplement_names)
    streams={}
    for name in expected:
        directory=a.main_run/'models'/name
        if not (directory/'complete.json').exists():raise ValueError(f'Incomplete variant: {name}')
        streams[name]=pd.read_parquet(directory/'predictions.parquet')
    for name in supplement_names:
        directory=a.supplements/name
        if not (directory/'complete.json').exists():raise ValueError(f'Incomplete supplement: {name}')
        streams['supp_'+name]=pd.read_parquet(directory/'predictions.parquet')
    data=add_series_features(pd.read_parquet(a.features))
    for family in selection['selected']:
        streams[family+'_raw']=streams[family].assign(p_calibrated=streams[family].p_raw)
    for scope in ('player','team'):
        for family in ('elo','gl','ts','os','pl','tm'):
            name=scope+'_'+family
            streams[name]=rating_stream(data,name)
    targets=data[pd.to_datetime(data.date)>=pd.Timestamp('2024-01-01')].copy()
    targets=align_identity_names(targets,pd.DataFrame(json.loads(a.identities.read_text())))
    targets['competition']=targets.tournament_name
    panel=join_prediction_streams(targets,streams)
    panel,audit=attach_market(panel,pd.read_csv(a.market,dtype={'golgg_match_id':str}))
    if audit['outcome_conflicts'] or audit['date_conflicts']: raise ValueError('Market conflict after independent ID alignment')
    if a.expected_market_rows is not None and audit['open_close_common']!=a.expected_market_rows:
        raise ValueError('Cohort differs from requested independent market audit')
    summary=evaluate_thesis(panel,a.output_dir,primary='p_'+selection['primary_family'])
    (a.output_dir/'input_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (a.output_dir/'market_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    # Controlled ablations and H1 use their own matched reference, not the primary family.
    pairs=[('p_team_'+f,'p_player_'+f) for f in ('elo','gl','ts','os','pl','tm')]
    pairs += [(c,'p_logistic') for c in panel if c.startswith('p_lr_') or c.startswith('p_supp_') and 'primary_' not in c]
    pairs += [('p_supp_primary_no_context','p_'+selection['primary_family']),('p_supp_primary_no_binomial','p_'+selection['primary_family'])]
    pairs += [('p_'+f,'p_'+f+'_raw') for f in selection['selected']]
    rows=[]
    for cohort,df in [('sports',panel),('market_common',panel[panel[['market_open','market_close']].notna().all(axis=1)])]:
        y=df.y_true.to_numpy(float); blocks=pd.to_datetime(df.date).dt.strftime('%Y-%m')
        for candidate,reference in pairs:
            for metric in ('logloss','brier'):
                delta=(binary_log_loss_vector(y,df[candidate])-binary_log_loss_vector(y,df[reference])) if metric=='logloss' else ((df[candidate].to_numpy()-y)**2-(df[reference].to_numpy()-y)**2)
                rows.append(dict(cohort=cohort,candidate=candidate,reference=reference,metric=metric,blocking='month',**block_difference(delta,blocks)))
    pd.DataFrame(rows).to_csv(a.output_dir/'controlled_comparisons.csv',index=False)
    for name in [scope+'_'+family for scope in ('player','team') for family in ('elo','gl','ts','os','pl','tm')]:
        destination=a.output_dir/'rating_predictions';destination.mkdir(exist_ok=True)
        streams[name].to_parquet(destination/f'{name}.parquet',index=False)
    (a.output_dir/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
