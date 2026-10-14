"""Resume frozen thesis fits in fresh spawned processes after worker interruption.

No model or configuration selection is performed. Completed earlier variants are
verified and copied; unfinished variants fit from the same immutable feature file.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.thesis_audit import _job,feature_variants


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    old=json.loads((a.previous/'protocol.json').read_text())
    for source,digest in old['sources'].items():
        if sha(source)!=digest: raise ValueError(f'Changed pinned source: {source}')
    feature=next(x for x in old['sources'] if x.endswith('features.parquet'))
    selection=json.loads((a.previous/'selection.json').read_text())
    variants=feature_variants()
    jobs=[(family,family,variants['w20'],params) for family,params in selection['selected'].items()]
    jobs += [(f'lr_{name}','logistic',columns,selection['selected']['logistic']) for name,columns in variants.items() if name!='w20']
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'models').mkdir()
    shutil.copyfile(a.previous/'selection.json',a.output_dir/'selection.json')
    protocol={**old,'resumed_at':datetime.now(timezone.utc).isoformat(),'previous':str(a.previous),
              'execution_change':'spawn context instead of fork; selection unchanged',
              'resume_source_sha256':sha(__file__),'workers':a.workers,
              'selection_sha256':sha(a.previous/'selection.json')}
    (a.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    pending=[];reused=[]
    for name,family,columns,params in jobs:
        olddir=a.previous/'models'/name
        if (olddir/'complete.json').exists():
            c=json.loads((olddir/'complete.json').read_text())
            if sha(olddir/'predictions.parquet')!=c['prediction_sha256'] or sha(olddir/'fit_receipts.json')!=c['receipt_sha256']:
                raise ValueError(f'Changed completed evidence: {name}')
            shutil.copytree(olddir,a.output_dir/'models'/name)
            reused.append(name)
        else:pending.append((name,family,columns,params))
    print(json.dumps({'reused':reused,'pending':[j[0] for j in pending]}),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_job,feature,columns,family,params,name,str(a.output_dir)) for name,family,columns,params in pending]
        for f in as_completed(futures): print(json.dumps({'completed_variant':f.result()}),flush=True)
    for source,digest in old['sources'].items():
        if sha(source)!=digest:raise ValueError(f'Changed pinned source: {source}')
    (a.output_dir/'FITS_COMPLETE.json').write_text(json.dumps({'variants':[j[0] for j in jobs],
         'reused':reused,'refitted':[j[0] for j in pending],'selection_unchanged':True},indent=2)+'\n')


if __name__=='__main__':main()
