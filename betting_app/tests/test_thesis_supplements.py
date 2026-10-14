import numpy as np
import pandas as pd

from src.analysis.thesis_supplements import mask_past_training, calibrate_oos, supplement_variants


def test_masking_does_not_change_test_or_unreleased_rows():
    d=pd.DataFrame({'golgg_match_id':['a','b','c'],'date':pd.to_datetime(['2023-01-01','2024-01-03','2023-01-01']),
                    'result_day':pd.to_datetime(['2023-01-01','2024-01-03','2024-02-01']),
                    'player_gl':[.3,.7,.4],'other':[1,2,3]})
    masked,receipt=mask_past_training(d,['player_gl'],'2024-01-01',1.,42)
    assert pd.isna(masked.loc[0,'player_gl'])
    pd.testing.assert_frame_equal(d.iloc[1:],masked.iloc[1:])
    assert masked.other.equals(d.other)
    assert receipt['masked_cells']==1


def test_masking_is_independent_of_future_rows_and_input_order():
    d=pd.DataFrame({'golgg_match_id':['b','a'],'date':pd.to_datetime(['2023-01-01']*2),
                    'result_day':pd.to_datetime(['2023-01-01']*2),'player_gl':[.3,.7]})
    first,_=mask_past_training(d,['player_gl'],'2024-01-01',.5,42)
    future=d.iloc[:1].assign(golgg_match_id='f',date=pd.Timestamp('2025-01-01'))
    changed,_=mask_past_training(pd.concat([future,d.iloc[::-1]],ignore_index=True),['player_gl'],'2024-01-01',.5,42)
    pd.testing.assert_series_equal(first.set_index('golgg_match_id').player_gl.sort_index(),
                                   changed.set_index('golgg_match_id').loc[['a','b']].player_gl)


def test_calibration_excludes_same_month_and_unreleased_outcomes():
    d=pd.DataFrame({'golgg_match_id':['a','b','c','d'],'date':pd.to_datetime(['2024-01-02','2024-01-03','2024-02-02','2024-02-03']),
                    'result_day':pd.to_datetime(['2024-01-02','2024-03-01','2024-02-02','2024-02-03']),
                    'origin':['2024-01-01']*2+['2024-02-01']*2,'y_true':[0,1,0,1], 'p_raw':[.3,.7,.4,.6]})
    p,r=calibrate_oos(d,min_samples=1)
    assert r[1]['cal_ids']==['a']
    assert not r[1]['calibrator_available'] # only one released class
    changed=d.copy();changed.loc[changed.date>='2024-02-01','y_true']=1-changed.loc[changed.date>='2024-02-01','y_true']
    p2,_=calibrate_oos(changed,min_samples=1)
    np.testing.assert_array_equal(p.p_calibrated,p2.p_calibrated)


def test_variant_contracts():
    v=supplement_variants()
    assert len(v['full_player_team']['features'])==80
    assert len(v['no_binomial']['features'])==40
    assert v['mask_playergl10']['mask_columns']==['player_gl']
    assert len(v['mask_base10']['mask_columns'])==20
