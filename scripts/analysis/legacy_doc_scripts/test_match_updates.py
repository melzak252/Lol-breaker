import pandas as pd
import numpy as np
from match_updates import paired_replay


def fixture(score):
    maps=[]
    for i,y in enumerate(score):
        maps.append(dict(map_id=str(i),series_id='1',date='2021-01-01',a='a',b='b',y=y,score_a=13 if y else 7,score_b=7 if y else 13,map_key='x'))
    maps.append(dict(map_id='next',series_id='2',date='2021-01-03',a='a',b='b',y=0,score_a=7,score_b=13,map_key='x'))
    series=pd.DataFrame([dict(series_id='1',date='2021-01-01',a='a',b='b',y=1),dict(series_id='2',date='2021-01-03',a='a',b='b',y=0)])
    return pd.DataFrame(maps),series


def test_match_win_has_same_weight_for_2_0_and_2_1():
    short=paired_replay(*fixture([1,1]));long=paired_replay(*fixture([1,0,1]))
    for col in ['p_elo16','p_elo32','p_elo64','p_glicko2']:
        assert np.isclose(short['match_updates'].iloc[1][col],long['match_updates'].iloc[1][col])
    assert not np.isclose(short['map_updates'].iloc[1].p_elo64,long['map_updates'].iloc[1].p_elo64)


def test_only_elo_and_glicko_change_and_current_result_is_unseen():
    maps,series=fixture([1,1]);result=paired_replay(maps,series)
    assert result['match_updates'].iloc[0].p_elo64==.5
    assert result['match_updates'].iloc[1].p_elo64>.5
    for col in ['p_trueskill','p_openskill','p_plackett_luce','p_thurstone','f_w20_win']:
        np.testing.assert_allclose(result['map_updates'][col],result['match_updates'][col])


def test_match_update_is_symmetric():
    maps,series=fixture([1,1]);original=paired_replay(maps,series)
    for frame in [maps,series]:
        frame[['a','b']]=frame[['b','a']];frame.y=1-frame.y
    maps[['score_a','score_b']]=maps[['score_b','score_a']]
    swapped=paired_replay(maps,series)
    for variant in original:
        cols=[c for c in original[variant] if c.startswith('p_')]
        np.testing.assert_allclose(original[variant][cols]+swapped[variant][cols],1,atol=1e-8)
