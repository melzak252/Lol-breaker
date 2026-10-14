"""Release-corrected six-family thesis features; immutable, odds-independent export.

Historical first-map rosters are an explicit known-lineup scenario. Rating outputs
are engineered rating signals; their map/series calibration is not assumed here.
"""
from collections import defaultdict, deque
from copy import copy
from dataclasses import replace
from datetime import date
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

WINDOWS=(3,5,10,20,50)
STATS=('win','kills','deaths','gd15','dpm','vspm','towers','nashors','gold','duration')


def rolling_snapshot(state, team, side, windows=WINDOWS):
    history=list(state.histories.get(team,()))
    out={}
    for window in windows:
        rows=history[-window:]
        complete=bool(rows) and all(r is not None and all(k in r and np.isfinite(r[k]) for k in STATS) for r in rows)
        out[f't{side}_w{window}_count']=len(rows)
        out[f't{side}_w{window}_complete']=int(complete)
        out[f't{side}_w{window}_full']=int(complete and len(rows)==window)
        for stat in STATS:
            name='win_rate' if stat=='win' else stat
            out[f't{side}_rolling_{name}_w{window}']=float(np.mean([r[stat] for r in rows])) if complete else np.nan
    return out


def rating_snapshot(state, row):
    a,b=str(row.team1_id),str(row.team2_id)
    ra,rb=list(row.roster_a),list(row.roster_b)
    day=date.fromisoformat(str(row.date))
    gl=state.manager.systems['gl']; restored=[]
    try:
        for ratings, dates, ids in ((gl.team_ratings,gl.team_last_played,[a,b]),(gl.player_ratings,gl.player_last_played,sorted(set(ra+rb)))):
            for ident in ids:
                existed=ident in ratings
                original=ratings[ident] if existed else ratings.default_factory()
                restored.append((ratings,ident,original,existed))
                ratings[ident]=copy(original)
                if ident in dates: gl.apply_time_decay(ratings[ident],dates[ident],day)
        out=state.manager.predict_match(a,b,ra,rb)
    finally:
        for ratings,ident,original,existed in restored:
            if existed: ratings[ident]=original
            else: ratings.pop(ident,None)
    for side,team in ((1,a),(2,b)):
        out[f'days_since_last_{side}']=(day-state.manager.last_match_date[team]).days if team in state.manager.last_match_date else 30
        out.update(rolling_snapshot(state,team,side))
    out['days_diff']=out['days_since_last_1']-out['days_since_last_2']
    out['BoN']=int(row.best_of); out['y_true']=int(row.y)
    return out


def replay_features(series, metadata, release_days, state, snapshot=rating_snapshot):
    if metadata.golgg_match_id.astype(str).duplicated().any(): raise ValueError('duplicate target ID')
    pending=[]
    for item in series:
        release=max(max(g.day for g in item.games),date.fromisoformat(release_days.get(item.identifier,item.games[-1].day.isoformat())))
        pending.append(replace(item,games=[*item.games[:-1],replace(item.games[-1],day=release)]))
    pending.sort(key=lambda item:(item.games[-1].day,item.order))
    pos=0; output={}
    for day,group in metadata.groupby('date',sort=True):
        cutoff=date.fromisoformat(str(day))
        while pos<len(pending) and pending[pos].games[-1].day<cutoff:
            state.consume(pending[pos]); pos+=1
        for idx,row in group.iterrows():
            output[idx]={**row.to_dict(),**snapshot(state,row),'feature_source_max_day':state.last_day.isoformat() if state.last_day else None}
        if len(output)%2000<len(group): print(f'Features {len(output)}/{len(metadata)}; consumed {pos}',flush=True)
    result=pd.DataFrame([output[idx] for idx in metadata.index])
    if any(pd.notna(s) and s>=str(d) for s,d in zip(result.feature_source_max_day,result.date)):
        raise ValueError('feature source reaches target day')
    return result


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def build_thesis_features(raw_source, metadata_path, release_path, output_dir):
    from scripts.export_prospective_features import iter_matches
    from scripts.prospective_sports_features import _prepare_history
    from scripts.prospective_sports_features import ChronologicalFeatureState
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    sources=[Path(raw_source),Path(metadata_path),Path(release_path),Path(__file__)]
    initial={str(p.resolve()):sha(p) for p in sources}
    (output/'STARTED.json').write_text(json.dumps({'sources':initial,'windows':WINDOWS},indent=2))
    metadata=pd.read_parquet(metadata_path)
    release=json.loads(Path(release_path).read_text())
    identities={}
    def stream():
        from scripts.prospective_sports_features import _series_team_ids
        from src.utils.golgg_schema import initial_wins
        for raw in iter_matches(Path(raw_source)):
            a,b=_series_team_ids(raw)
            first=raw['games'][0]
            names={str(first['t1_id']):first.get('t1_name',''),str(first['t2_id']):first.get('t2_name','')}
            initial=initial_wins(raw)
            score_a=initial.get(a,0)+sum(int(g['t1_win']) if str(g['t1_id'])==a else int(g['t2_win']) for g in raw['games'])
            score_b=initial.get(b,0)+len(raw['games'])-(score_a-initial.get(a,0))
            identities[str(raw['match_id'])]={'match_id':int(raw['match_id']),'date':raw['date'],'name_1':names[a],'name_2':names[b],'team1_id':a,'team2_id':b,'score_1':score_a,'score_2':score_b,'t1_win':score_a>score_b,'t2_win':score_b>score_a,'draw':score_a==score_b}
            yield raw
    series,audit=_prepare_history(stream(),date.max)
    (output/'identities.json').write_text(json.dumps(list(identities.values())))
    print(f'Prepared {len(series)} historical series',flush=True)
    state=ChronologicalFeatureState()
    state.histories=defaultdict(lambda:deque(maxlen=max(WINDOWS)))
    result=replay_features(series,metadata,release,state)
    result.to_parquet(output/'features.parquet',index=False)
    dependencies={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and (name.startswith('scripts.') or name.startswith('src.ratings') or name.startswith('src.utils.golgg_schema')) and Path(path).is_file():
            dependencies[str(Path(path).resolve())]=sha(path)
    receipt={'status':'COMPLETE','rows':len(result),'history_series':len(series),'sources':initial,'dependencies':dependencies,'output_sha256':sha(output/'features.parquet'),'windows':WINDOWS,'audit':audit,'policy':'All historical series, no odds filtering; corrected full-series release strictly before target date; first-map known-roster scenario; original six-family fixed recipe; partial windows allowed with counts, missing historical stats remain NaN; no map-probability interpretation imposed.'}
    (output/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('raw-source','metadata-path','release-path','output-dir'): parser.add_argument('--'+arg,required=True)
    args=parser.parse_args()
    build_thesis_features(**vars(args))
