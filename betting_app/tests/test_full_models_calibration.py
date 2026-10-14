import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

script=Path(__file__).resolve().parents[2]/'scripts/research/full_models_analysis.py'
spec=importlib.util.spec_from_file_location('full_models_analysis_test',script)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_symmetric_mapping_preserves_complements_even_for_asymmetric_calibrator():
    x=np.linspace(.01,.99,99)
    g=lambda p: .1+.7*p
    np.testing.assert_allclose(mod.symmetric_map(g,x)+mod.symmetric_map(g,1-x),1,atol=1e-15)


def test_calibration_month_origin_excludes_unreleased_and_current_outcomes():
    n=220
    p=np.linspace(.05,.95,n)
    data=pd.DataFrame({'golgg_match_id':np.arange(n+2).astype(str),
      'date':pd.to_datetime(['2024-01-02']*n+['2024-02-02']*2),
      'result_day':pd.to_datetime(['2024-01-02']*(n-1)+['2024-03-01']+['2024-02-02']*2),
      'y_true':np.r_[np.arange(n)%2,0,1], 'p_raw':np.r_[p,.3,.7]})
    predictions,receipts=mod.calibration_replay(data)
    for method,group in predictions.groupby('method'):
        assert set(group.golgg_match_id)==set(data.golgg_match_id)
        assert not group.golgg_match_id.duplicated().any()
        aligned=data.set_index('golgg_match_id').loc[group.golgg_match_id]
        np.testing.assert_array_equal(group.y_true,aligned.y_true)
        np.testing.assert_array_equal(group.date,aligned.date)
    assert receipts[-1]['cal_rows']==219
    changed=data.copy();changed.loc[changed.result_day>='2024-02-01','y_true']=1-changed.loc[changed.result_day>='2024-02-01','y_true']
    other,_=mod.calibration_replay(changed)
    np.testing.assert_allclose(predictions.probability,other.probability)
    symmetric=predictions[predictions.method.isin(['identity','positive_slope','symmetric_platt','symmetric_isotonic'])]
    assert symmetric.symmetry_error.max()<1e-12
