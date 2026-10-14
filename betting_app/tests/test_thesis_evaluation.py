import numpy as np
import pandas as pd
import pytest

from src.analysis.thesis_evaluation import join_prediction_streams, attach_market, block_difference


def metadata():
    return pd.DataFrame({'golgg_match_id':['1','2'], 'date':['2024-01-01','2024-02-01'],
                         'y_true':[1,0], 'team1_name':['Alpha','Beta'], 'team2_name':['Beta','Alpha']})


def test_join_rejects_disagreeing_labels_instead_of_self_comparison():
    prediction = metadata().assign(p_calibrated=[.6,.4])
    prediction.loc[0,'y_true'] = 0
    with pytest.raises(ValueError, match='label'):
        join_prediction_streams(metadata(), {'lr':prediction})


def test_join_rejects_duplicates_and_missing_forecasts():
    p=metadata().assign(p_calibrated=[.6,.4])
    for malformed in [pd.concat([p,p.iloc[:1]]), p.iloc[:1]]:
        with pytest.raises(ValueError):
            join_prediction_streams(metadata(), {'lr':malformed})


def test_market_alignment_uses_names_not_outcomes_and_rejects_invalid_prices():
    odds=pd.DataFrame({'golgg_match_id':['1','2'], 'golgg_team1':['Beta','Beta'],
                      'golgg_team2':['Alpha','Alpha'], 'golgg_date':['2024-01-01','2024-02-01'],
                      't1_win':[False,False], 'avg_open_home':[3.,1.], 'avg_open_away':[1.5,2.],
                      'avg_odds_home':[3.,2.], 'avg_odds_away':[1.5,2.]})
    d,audit=attach_market(metadata(),odds)
    assert d.loc[0,'market_open']==pytest.approx(2/3)
    assert np.isnan(d.loc[1,'market_open'])
    changed=odds.copy(); changed.loc[0,'t1_win']=True
    d2,_=attach_market(metadata(),changed)
    assert np.isnan(d2.loc[0,'market_open']) # conflict excludes, never flips to agree


def test_bootstrap_keeps_match_and_equal_block_estimands_distinct():
    delta=np.array([1.,1.,1.,-1.]); blocks=np.array(['a','a','a','b'])
    result=block_difference(delta,blocks,draws=5000,seed=5)
    assert result['delta']==pytest.approx(.5)
    assert result['equal_block_delta']==pytest.approx(0.)
    assert result['blocks']==2
    assert 'p_value' not in result
    assert result['ci_lower']<=.5<=result['ci_upper']
