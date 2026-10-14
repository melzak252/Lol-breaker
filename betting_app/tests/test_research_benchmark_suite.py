import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Standard import after installation; staging path is provided by test runner.
from src.analysis.research_benchmark import validate_frame, attach_candidate, paired_delta

def frame():
    return pd.DataFrame({'golgg_match_id':['1','2'], 'date':['2025-01-02','2025-02-02'], 'result_day':['2025-01-02','2025-02-02'], 'feature_source_max_day':['2025-01-01','2025-02-01'], 'team1_id':['a','b'], 'team2_id':['b','a'], 'best_of':[3,3], 'y':[1,0], 'p':[.7,.3]})

def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        validate_frame(pd.concat([frame(),frame()]), ['p'])

def test_future_features_rejected():
    f=frame();f.loc[0,'feature_source_max_day']='2025-01-02'
    with pytest.raises(ValueError,match='history'):
        validate_frame(f,['p'])

def test_invalid_probability_not_clipped():
    f=frame();f.loc[0,'p']=1.1
    with pytest.raises(ValueError,match='probability'):
        validate_frame(f,['p'])

def test_candidate_must_cover_all_and_match_sides():
    base=frame();candidate=frame();candidate['train_end']='2023-12-31';candidate['calibration_end']='2024-12-31'
    with pytest.raises(ValueError,match='coverage'):
        attach_candidate(base,candidate.iloc[:1],'p')
    candidate.loc[0,'team1_id']='wrong'
    with pytest.raises(ValueError,match='team1_id'):
        attach_candidate(base,candidate,'p')

def test_candidate_future_calibration_rejected():
    c=frame();c['train_end']='2023-12-31';c['calibration_end']='2026-01-01'
    with pytest.raises(ValueError,match='calibration_end'):
        attach_candidate(frame(),c,'p')

def test_candidate_order_is_by_id():
    c=frame();c['train_end']='2023-12-31';c['calibration_end']='2024-12-31'
    out=attach_candidate(frame(),c.iloc[::-1],'p')
    np.testing.assert_array_equal(out['candidate'],[.7,.3])


def test_online_updates_after_calibration_need_explicit_released_state_policy():
    candidate = frame()
    candidate["train_end"] = candidate["feature_source_max_day"]
    candidate["calibration_end"] = "2024-12-31"
    candidate["update_policy"] = "released_daily_frozen_map_periods"
    with pytest.raises(ValueError, match="train_end"):
        attach_candidate(frame(), candidate, "p")
    accepted = attach_candidate(frame(), candidate, "p", online_updates=True)
    np.testing.assert_array_equal(accepted["candidate"], candidate["p"])
    candidate.loc[0, "train_end"] = candidate.loc[0, "date"]
    with pytest.raises(ValueError, match="train_end"):
        attach_candidate(frame(), candidate, "p", online_updates=True)


def test_online_attachment_requires_named_update_schedule():
    candidate = frame()
    candidate["train_end"] = candidate["feature_source_max_day"]
    candidate["calibration_end"] = "2024-12-31"
    with pytest.raises(ValueError, match="update_policy"):
        attach_candidate(frame(), candidate, "p", online_updates=True)

def test_single_block_has_no_invented_interval():
    f=frame();f['date']='2025-01-02'
    out=paired_delta(f,np.array([.1,.3]),20)
    assert out['blocks']==1 and out['ci95'] is None
    assert out['equal_block_delta']==pytest.approx(.2)

def test_extreme_forecasts_use_same_unclipped_loss_in_headline_and_delta():
    from src.analysis.model_benchmark import compute_benchmark_summary
    from src.analysis.research_benchmark import losses
    y=np.array([0,1]);p=np.array([.99999,.00001])
    result=compute_benchmark_summary(y,p,probability_epsilon=None)
    assert result.log_loss==pytest.approx(losses(y,p).mean())
    assert result.log_loss>11

def test_doctor_detects_changed_frozen_input(tmp_path):
    import json,hashlib
    from src.analysis.research_benchmark import run_suite
    data=tmp_path/'input.bin';data.write_bytes(b'original')
    config=tmp_path/'manifest.json'
    config.write_text(json.dumps({'schema_version':1,'name':'test','input':{'path':'input.bin','sha256':hashlib.sha256(b'original').hexdigest()}}))
    assert run_suite(config,tmp_path,doctor=True)['status']=='READY'
    data.write_bytes(b'changed')
    assert run_suite(config,tmp_path,doctor=True)['status']=='MISSING_OR_CHANGED_INPUTS'
    with pytest.raises(ValueError,match='missing/changed'):
        run_suite(config,tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_existing_output_is_not_overwritten(tmp_path):
    import json,hashlib
    from src.analysis.research_benchmark import run_suite
    data=tmp_path/'input.bin';data.write_bytes(b'original')
    config=tmp_path/'manifest.json';config.write_text(json.dumps({'schema_version':1,'name':'test','input':{'path':'input.bin','sha256':hashlib.sha256(b'original').hexdigest()}}))
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('preserve')
    with pytest.raises(FileExistsError):
        run_suite(config,tmp_path,out)
    assert (out/'keep').read_text()=='preserve'
