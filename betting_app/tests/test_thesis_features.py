from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
import pandas as pd
import pytest
from src.analysis.thesis_features import replay_features, rolling_snapshot

@dataclass
class Game:
    day: date
@dataclass
class Series:
    identifier: str
    order: tuple
    games: list
class State:
    last_day = None
    def __init__(self): self.seen=[]
    def consume(self,item): self.last_day=item.games[-1].day; self.seen.append(item.identifier)

def test_release_day_strictly_excludes_same_day_and_future_and_preserves_order():
    series=[Series('1',(0,1),[Game(date(2020,1,1))]),Series('2',(0,2),[Game(date(2020,1,2))])]
    meta=pd.DataFrame({'golgg_match_id':['b','a','c'],'date':['2020-01-03','2020-01-02','2020-01-04']})
    result=replay_features(series,meta,{'1':'2020-01-03'},State(),lambda s,r:{'seen':','.join(s.seen)})
    assert result.seen.tolist()==['2','','2,1']
    assert result.golgg_match_id.tolist()==['b','a','c']

def test_missing_stat_inside_window_is_explicit_not_zero_imputed():
    state=SimpleNamespace(histories={'a':[{'win':1,'kills':9},None,{'win':0,'kills':3}]})
    out=rolling_snapshot(state,'a',1,(3,5))
    assert out['t1_w3_complete']==0
    assert pd.isna(out['t1_rolling_kills_w3'])
    assert out['t1_w5_count']==3

def test_duplicate_target_id_rejected():
    meta=pd.DataFrame({'golgg_match_id':['1','1'],'date':['2020-01-01']*2})
    with pytest.raises(ValueError,match='duplicate'):
        replay_features([],meta,{},State(),lambda s,r:{})

def test_complete_partial_window_means_use_observed_games_only():
    from src.analysis.thesis_features import STATS
    state=SimpleNamespace(histories={'a':[{k:2 for k in STATS},{k:4 for k in STATS}]})
    result=rolling_snapshot(state,'a',1,(3,))
    assert result['t1_rolling_kills_w3']==3
    assert result['t1_w3_complete']==1
    assert result['t1_w3_full']==0

def test_future_release_mutation_cannot_change_earlier_features():
    meta=pd.DataFrame({'golgg_match_id':['a'],'date':['2020-01-02']})
    first=Series('1',(0,1),[Game(date(2020,1,1))])
    future=Series('2',(0,2),[Game(date(2020,1,3))])
    a=replay_features([first,future],meta,{},State(),lambda s,r:{'seen':','.join(s.seen)})
    b=replay_features([first],meta,{},State(),lambda s,r:{'seen':','.join(s.seen)})
    pd.testing.assert_frame_equal(a,b)

def test_actual_six_family_snapshot_is_symmetric_and_does_not_decay_persistent_state():
    from scripts.prospective_sports_features import ChronologicalFeatureState
    from src.analysis.thesis_features import rating_snapshot
    state=ChronologicalFeatureState()
    row=SimpleNamespace(team1_id='A',team2_id='B',roster_a=list('abcde'),roster_b=list('fghij'),date='2020-02-01',best_of=3,y=1)
    forward=rating_snapshot(state,row)
    reverse=rating_snapshot(state,SimpleNamespace(**{**vars(row),'team1_id':'B','team2_id':'A','roster_a':row.roster_b,'roster_b':row.roster_a,'y':0}))
    for system in ['elo','gl','ts','os','pl','tm']:
        assert forward['player_'+system]+reverse['player_'+system]==pytest.approx(1)
        assert forward['team_'+system]+reverse['team_'+system]==pytest.approx(1)
    assert state.manager.last_match_date=={}
    assert not state.manager.systems['gl'].player_ratings
