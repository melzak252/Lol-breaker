"""Immutable historical panels must not manufacture clocks or side identities."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from src.analysis.c0_betting_development import build_panel
from src.analysis import c0_betting_development as development


def frames():
    control=pd.DataFrame([{'golgg_match_id':'1','team1_id':'10','team2_id':'20','best_of':3,'y':1,
        'date':'2026-04-30','result_day':'2026-04-30','feature_source_max_day':'2026-04-29','p_full_full':.7}])
    market=control.drop(columns='p_full_full').assign(fixture_id=100,control__full__full=.7)
    rows=[]
    for book,oa,ob in [('a',3.,1.5),('b',2.5,1.6)]:
        rows.append({'fixture_id':100,'golgg_match_id':'1','team1_id':'10','team2_id':'20','best_of':3,'y':1,
            'horizon':'T-1h','strict_prior_day':True,'provider_team1_is_a':True,'bookmaker':book,
            'evaluation_cutoff_at':'2026-04-30T11:00:00Z','start_cutoff_at':'2026-04-30T12:00:00Z',
            'quote_at':'2026-04-30T10:00:00Z','outcome_1_quote_at':'2026-04-30T10:00:00Z',
            'outcome_2_quote_at':'2026-04-30T10:00:00Z','valid_until_exclusive_at':None,
            'odds_1':oa,'odds_2':ob,'outcome_1_source_index':0,'outcome_2_source_index':1,
            'source_path':'synthetic.json','source_sha256':'0'*64})
    return control,market,pd.DataFrame(rows)


def test_late_training_label_is_retained_but_not_used_for_fit():
    control,market,quotes=frames()
    control['result_day']='2026-05-01'
    market['result_day']='2026-05-01'
    panel,_=build_panel(control,market,quotes)
    assert panel.iloc[0].phase=='TRAIN'
    assert not panel.iloc[0].fit_eligible
    assert panel.iloc[0].p_C0==.7
    assert panel.iloc[0].financial_eligible


def test_future_latest_quote_withholds_market_without_older_state_fallback():
    control,market,quotes=frames()
    quotes.loc[1,'outcome_1_quote_at']='2026-04-30T11:00:01Z'
    panel,_=build_panel(control,market,quotes)
    assert np.isnan(panel.iloc[0].q)
    assert panel.iloc[0].financial_eligible
    assert not panel.iloc[0].market_eligible
    assert panel.iloc[0].p_C0==.7


def test_provider_side_binding_is_applied_once_and_identity_conflicts_fail():
    control,market,quotes=frames()
    quotes['provider_team1_is_a']=False
    panel,_=build_panel(control,market,quotes)
    assert panel.iloc[0].odds_a==1.6
    assert panel.iloc[0].odds_b==3.
    assert panel.iloc[0].q==pytest.approx(np.mean([3/4.5,2.5/4.1]))
    quotes.loc[0,'team1_id']='wrong'
    with pytest.raises(ValueError,match='identity'):
        build_panel(control,market,quotes)


def test_future_native_source_cannot_pass_scenario_chronology():
    control,market,quotes=frames()
    control['feature_source_max_day']='2026-04-30'
    market['feature_source_max_day']='2026-04-30'
    with pytest.raises(ValueError,match='feature'):
        build_panel(control,market,quotes)


def test_integral_quote_format_is_numeric_but_fractional_formats_conflict():
    control,market,quotes=frames()
    quotes["best_of"]=quotes["best_of"].astype(float)
    quotes["y"]=quotes["y"].astype(float)
    panel,_=build_panel(control,market,quotes)
    assert panel.iloc[0].financial_eligible
    assert panel.iloc[0].p_C0==.7
    quotes.loc[0,"best_of"]=3.5
    with pytest.raises(ValueError,match="identity"):
        build_panel(control,market,quotes)


def test_unreleased_assessment_result_is_withheld_not_scored_as_known():
    control,market,quotes=frames()
    control["result_day"]="2026-10-06"
    market["result_day"]="2026-10-06"
    control["feature_source_max_day"]="2026-08-01"
    market["feature_source_max_day"]="2026-08-01"
    quotes["evaluation_cutoff_at"]="2026-08-02T11:00:00Z"
    quotes["start_cutoff_at"]="2026-08-02T12:00:00Z"
    for column in ("quote_at","outcome_1_quote_at","outcome_2_quote_at"):
        quotes[column]="2026-08-02T10:00:00Z"
    panel,_=build_panel(control,market,quotes)
    assert panel.iloc[0].phase=="ASSESS"
    assert pd.isna(panel.iloc[0].y)
    assert not panel.iloc[0].financial_eligible
    assert panel.iloc[0].p_C0==.7


def test_provider_string_identifiers_are_not_coerced_to_database_ids():
    control,market,quotes=frames()
    market["fixture_id"]="id1803998567498334"
    quotes["fixture_id"]="id1803998567498334"
    panel,_=build_panel(control,market,quotes)
    assert panel.iloc[0].fixture_id=="id1803998567498334"
    assert panel.iloc[0].row_id=="id1803998567498334:1"
    assert panel.iloc[0].financial_eligible


def test_reassessment_refuses_existing_artifacts_before_loading_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "PROJECT_ROOT", tmp_path)
    protocol = {"run_id": "completed"}
    protocol_path = tmp_path/"protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    binding = {"kind": "assessment_only_cash_accounting_review", "additional_formal_fits": 0,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "protocol_file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    binding_path = tmp_path/"binding.json"
    binding_path.write_text(json.dumps(binding))
    saved = tmp_path/"data/05_model_input/c0_betting_development/completed/reviewed_assessment_denominator.parquet"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"previously sealed assessment")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="immutable"):
        development.reassess(protocol_path, binding_path, research_root=tmp_path/"unavailable-archive")
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
