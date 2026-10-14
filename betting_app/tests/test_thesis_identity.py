import pandas as pd
import pytest
from src.analysis.thesis_identity import align_identity_names


def test_names_follow_model_ids_even_when_header_order_is_opposite():
    m=pd.DataFrame({'golgg_match_id':['9'],'team1_id':['b'],'team2_id':['a'],'y_true':[0]})
    ids=pd.DataFrame({'match_id':[9],'team1_id':['a'],'team2_id':['b'],
                      'name_1':['Alpha'],'name_2':['Beta'],'t1_win':[True]})
    out=align_identity_names(m,ids)
    assert out.team1_name.tolist()==['Beta']
    assert out.team2_name.tolist()==['Alpha']


def test_identity_join_rejects_mismatched_ids_and_labels():
    m=pd.DataFrame({'golgg_match_id':['9'],'team1_id':['a'],'team2_id':['b'],'y_true':[0]})
    ids=pd.DataFrame({'match_id':[9],'team1_id':['a'],'team2_id':['b'],
                     'name_1':['Alpha'],'name_2':['Beta'],'t1_win':[True]})
    with pytest.raises(ValueError,match='outcome'): align_identity_names(m,ids)
    ids['team1_id']='c'
    with pytest.raises(ValueError,match='identity'): align_identity_names(m,ids)
