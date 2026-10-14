import numpy as np
import pandas as pd
from scripts.research.build_thesis_descriptive import orient_market, gd15_development, quote_metrics

def test_market_orientation_uses_exact_team_ids_not_outcome():
 panel=pd.DataFrame({'golgg_match_id':['1'],'team1_id':['B'],'team2_id':['A'],'y_true':[0]})
 odds=pd.DataFrame({'golgg_match_id':['1'],'avg_open_home':[2.0],'avg_open_away':[4.0]})
 identity=pd.DataFrame({'match_id':[1],'team1_id':['A'],'team2_id':['B']})
 aligned=orient_market(panel,odds,identity)
 assert aligned.avg_open_home.iloc[0]==4
 assert aligned.avg_open_away.iloc[0]==2
 panel['y_true']=1
 pd.testing.assert_frame_equal(aligned[['avg_open_home','avg_open_away']],orient_market(panel,odds,identity)[['avg_open_home','avg_open_away']])

def test_gd15_development_ignores_evaluation_and_unreleased_labels():
 f=pd.DataFrame({'date':['2023-01-01','2023-02-01','2024-01-01','2023-12-31'],'result_day':['2023-01-01','2023-02-01','2024-01-01','2024-01-01'],'t1_rolling_gd15_w20':[1,0,100,200],'t2_rolling_gd15_w20':[0,1,0,0],'y_true':[1,0,0,0]})
 d=gd15_development(f)
 assert len(d)==2
 assert d.delta.tolist()==[1,-1]

def test_decimal_one_and_nonfinite_quotes_are_excluded():
 d=quote_metrics(np.array([2.,1.,np.nan]),np.array([2.,3.,2.]),np.array([1,0,1]))
 assert d['n']==1
 assert d['logloss']==np.log(2)
