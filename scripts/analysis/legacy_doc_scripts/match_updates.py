"""Controlled Elo/Glicko-2 map-vs-match update experiment on verified VCT."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO=Path('/home/melzak/dev/inzynierka')
sys.path.insert(0,str(REPO/'docs/assets/exp-093-crossgame-team'))
from experiment import DailyRatings,evaluate,bootstrap


def paired_replay(maps,series,progress=False):
    maps=maps.copy();series=series.copy()
    maps['date']=pd.to_datetime(maps.date);series['date']=pd.to_datetime(series.date)
    maps=maps.sort_values(['date','map_id']);series=series.sort_values(['date','series_id'])
    sm=series.assign(score_a=series.y,score_b=1-series.y,map_key='__series__')
    sg=dict(tuple(sm.groupby('date',sort=True)));mg=list(maps.groupby('date',sort=True))
    map_engine=DailyRatings();match_engine=DailyRatings()
    match_engine.models={n:m for n,m in match_engine.models.items() if n.startswith('elo') or n=='map_elo'}
    pointer=0;history=None;outputs={'map_updates':[],'match_updates':[]}
    for ix,(day,queries) in enumerate(series.groupby('date',sort=True)):
        cutoff=day-pd.Timedelta(days=2)
        while pointer<len(mg) and mg[pointer][0]<=cutoff:
            old,block=mg[pointer];map_engine.learn(block,old);match_engine.learn(sg[old],old)
            history=old;pointer+=1
        for r in queries.itertuples():
            before=map_engine.predict(r.a,r.b,'__query__',cutoff)
            whole=match_engine.predict(r.a,r.b,'__query__',cutoff)
            after=dict(before)
            for col in ['p_elo16','p_elo32','p_elo64','p_glicko2']:after[col]=whole[col]
            meta={'series_id':r.series_id,'history_max_date':history.strftime('%Y-%m-%d') if history is not None else ''}
            outputs['map_updates'].append({**meta,**before});outputs['match_updates'].append({**meta,**after})
        if progress and ix%100==0:print(day.date(),len(outputs['map_updates']),flush=True)
    return {k:pd.DataFrame(v) for k,v in outputs.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=REPO/'data/artifacts/exp093-crossgame-team/20260912'
    s=pd.read_parquet(root/'valorant_series/replay.parquet')
    d=pd.read_parquet(root/'valorant/replay.parquet');d.series_id=d.series_id.astype(str)
    d=d[d.series_id.isin(s.series_id)].copy()
    # Align every map to the verified series orientation, independently of score.
    align=s.set_index('series_id');reverse=d.a!=d.series_id.map(align.a)
    for a,b in [('a','b'),('score_a','score_b')]:d.loc[reverse,[a,b]]=d.loc[reverse,[b,a]].to_numpy()
    d.loc[reverse,'y']=1-d.loc[reverse,'y']
    assert (d.a==d.series_id.map(align.a)).all() and (d.b==d.series_id.map(align.b)).all()
    original_features=[c for c in s if c.startswith(('p_','f_','history_'))]
    s=s.drop(columns=original_features)
    predictions=paired_replay(d,s,True);summaries={}
    for variant,features in predictions.items():
        out=args.output/variant;out.mkdir()
        frame=s.merge(features,on='series_id',validate='one_to_one')
        frame.to_parquet(out/'replay.parquet',index=False)
        summaries[variant]=evaluate(frame,'valorant',out,series=True)
    old=pd.read_parquet(args.output/'map_updates/test_predictions.parquet').sort_values('series_id').reset_index(drop=True)
    new=pd.read_parquet(args.output/'match_updates/test_predictions.parquet').sort_values('series_id').reset_index(drop=True)
    assert old.series_id.equals(new.series_id) and old.y.equals(new.y)
    paired=new[['series_id','date','y']].copy();comparisons=[]
    for n in ['elo16','elo32','elo64','glicko2','consensus','stack_ratings','stack_w20']:
        paired[n+'_map']=old[n];paired[n+'_match']=new[n]
        comparisons.append(bootstrap(paired,n+'_match',n+'_map'))
    old_choice=summaries['map_updates']['selected_on_validation'];new_choice=summaries['match_updates']['selected_on_validation']
    comparisons.append(bootstrap(paired,new_choice+'_match',old_choice+'_map'))
    paired.to_parquet(args.output/'paired_test_predictions.parquet',index=False)
    (args.output/'summary.json').write_text(json.dumps({'series':len(s),'maps':len(d),'match_target':'one binary outcome, independent of map count',
        'changed_families':['Elo K16/32/64','Glicko2'],'unchanged':'Other rating families and W20 still receive maps; all predictions before whole matches',
        'history':'Identical verified VCT history in both variants; not the broad EXP-093 map history',
        'summaries':summaries,'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
