"""Regression contracts for thesis reporting and validation-only selection."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[2]
def module(path):
 spec=importlib.util.spec_from_file_location('test_'+Path(path).stem,ROOT/path)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def test_allmodels_saved_predictions_compute_ece():
 m=module('scripts/06_metamodel/06ab_w20_binomial_all_models_bootstrap.py')
 result=m.evaluate_predictions(pd.DataFrame({'variant':['a','a'],'y_true':[0,1],'y_prob':[.2,.8]}))
 assert result.iloc[0].ece==pytest.approx(.2)

def test_rating_selection_ignores_test_results_and_preserves_whole_row(tmp_path):
 m=module('scripts/05_ratingi_baseline/05c_optimize_rating_families.py');m.OUTPUT_DIR=tmp_path
 data=pd.DataFrame({'system':['elo','elo'],'tune_2021_2023_logloss':[.5,.6],'test_2024_plus_logloss':[.9,.1],'k_player':[16,48],'initial_rating':[1500,1500]})
 first=m.save_family_summary(data);data['test_2024_plus_logloss']=[.01,.99];second=m.save_family_summary(data)
 assert first.iloc[0].k_player==second.iloc[0].k_player==16

def test_player_team_interval_targets_match_weighted_mean():
 m=module('scripts/06_metamodel/06ad_player_vs_player_team_w20_binomial.py');m.N_BOOTSTRAPS=10000
 rows=[]
 for month,n,p in [('2024-01',100,.9),('2024-02',1,.1),('2024-03',1,.1)]:
  for i in range(n):
   for v,q in [(m.BASELINE_VARIANT,.5),('augmented',p)]:rows.append(dict(golgg_match_id=f'{month}-{i}',month=month,y_true=1,variant=v,probability=q))
 res=m.monthly_block_bootstrap(pd.DataFrame(rows)).iloc[0]
 assert res.ci_low < res.delta_logloss_variant_minus_baseline < res.ci_high
 # 19/27 possible resamples contain the large favorable month.
 assert .68 < res.bootstrap_fraction_delta_le_zero < .74
 assert 'p_one_sided_variant_better' not in res.index

@pytest.mark.parametrize('fault',['target','date','duplicate'])
def test_final_merge_rejects_source_metadata_conflicts(fault):
 m=module('scripts/06_metamodel/06ai_final_symmetric_calibrated_market_comparison.py')
 base=pd.DataFrame({'golgg_match_id':['1'],'date':['2024-01-01'],'y_true':[1]})
 source=pd.DataFrame({'golgg_match_id':['1'],'date':['2024-01-01'],'y_true':[1],'p':[.7]})
 if fault=='target':source['y_true']=0
 if fault=='date':source['date']='2024-01-02'
 if fault=='duplicate':source=pd.concat([source,source],ignore_index=True)
 with pytest.raises(ValueError):m.merge_checked_predictions(base,source,['p'],'test source')
