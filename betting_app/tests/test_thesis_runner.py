import numpy as np
import pandas as pd
import pytest
from src.analysis.thesis_audit import feature_variants, add_series_features


def test_series_features_use_map_proxy_once_and_preserve_complements():
    d=pd.DataFrame({'BoN':[1,3,5]})
    for scope in ('player','team'):
        for family in ('elo','gl','ts','os','pl','tm'): d[f'{scope}_{family}']=[.7,.7,.7]
    x=add_series_features(d)
    assert np.allclose(x.player_gl_binom_series,[.7,.784,.83692])
    for c in d:
        if c!='BoN': d[c]=1-d[c]
    y=add_series_features(d)
    assert np.allclose(x.player_gl_binom_series+y.player_gl_binom_series,1)


def test_ablation_removes_whole_glicko_family_and_window_substitutes_both_sides():
    variants=feature_variants()
    assert not any('player_gl' in x for x in variants['no_glicko'])
    assert len(variants['w3'])==len(variants['w20'])==46
    assert sum('_w3' in x for x in variants['w3'])==20
    assert 'team_gl' in variants['player_team']


@pytest.mark.parametrize('conflict',[False,True])
def test_main_runner_aligns_source_names_to_prediction_ids_and_validates_labels(tmp_path,monkeypatch,conflict):
    import json
    from concurrent.futures import Future
    import src.analysis.thesis_audit as runner
    feature=tmp_path/'features.parquet';market=tmp_path/'odds.csv';identities=tmp_path/'identities.json'
    data=pd.DataFrame({'golgg_match_id':['1','2'],'date':['2024-01-02','2024-01-03'],
                       'result_day':['2024-01-02','2024-01-03'],'y_true':[0,1],'BoN':[1,3],
                       'team1_id':['b','a'],'team2_id':['a','b'],'tournament_name':['L','L']})
    for scope in ('player','team'):
        for family in runner.RATING_NAMES:data[f'{scope}_{family}']=[.3,.7]
    data.to_parquet(feature,index=False)
    identities.write_text(json.dumps([{'match_id':i,'team1_id':'a','team2_id':'b','name_1':'Alpha',
                                      'name_2':'Beta','t1_win':False if conflict and i==1 else True} for i in (1,2)]))
    pd.DataFrame({'golgg_match_id':['1','2'],'golgg_team1':['Alpha']*2,'golgg_team2':['Beta']*2,
                  'golgg_date':['2024-01-02','2024-01-03'],'t1_win':[True,True],
                  'avg_open_home':[1.5,1.5],'avg_open_away':[3.,3.],
                  'avg_odds_home':[1.5,1.5],'avg_odds_away':[3.,3.]}).to_csv(market,index=False)
    selection={'selected':{'logistic':{}},'candidates':[{'family':'logistic','params':{},'logloss':.6}]}
    monkeypatch.setattr(runner,'select_development_configuration',lambda *a,**k:selection)
    contexts=[]
    class FakeExecutor:
        def __init__(self,**kwargs):contexts.append(kwargs.get('mp_context'))
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,function,feature_path,columns,family,params,name,output):
            destination=__import__('pathlib').Path(output)/'models'/name;destination.mkdir(parents=True)
            p=data[['golgg_match_id','date','result_day','y_true']].copy();p['p_raw']=[.3,.7];p['p_calibrated']=p.p_raw
            p.to_parquet(destination/'predictions.parquet',index=False)
            future=Future();future.set_result(name);return future
    monkeypatch.setattr(runner,'ProcessPoolExecutor',FakeExecutor)
    captured={}
    def evaluate(panel,*args,**kwargs):captured['panel']=panel;return {'passed':True}
    monkeypatch.setattr(runner,'evaluate_thesis',evaluate)
    output=tmp_path/'result'
    if conflict:
        with pytest.raises(ValueError,match='Source outcome conflict'):
            runner.run_thesis_audit(feature,market,identities,output)
        return
    runner.run_thesis_audit(feature,market,identities,output)
    assert captured['panel'].market_open.notna().all()
    np.testing.assert_allclose(captured['panel'].market_open,[1/3,2/3])
    assert captured['panel'].market_alignment.tolist()==['swapped','same']
    assert contexts[0].get_start_method()=='spawn'
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['selection_sha256']==runner._hash(output/'selection.json')
