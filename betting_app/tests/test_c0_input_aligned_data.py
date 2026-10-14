"""Consumer-visible prior-input causality and denominator regressions."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.c0_input_aligned_data import build_prior_targets, prepare_input_aligned


def panel():
    return pd.DataFrame([dict(row_id="f:100", golgg_match_id="100", fixture_id="f",
        team1_id=1, team2_id=2, tournament_name="LCK Summer", phase="ASSESS",
        scenario_decision_at="2026-07-10T12:00:00+00:00", best_of=3,
        roster_a=["actual-a"+str(i) for i in range(5)],
        roster_b=["actual-b"+str(i) for i in range(5)], y=1,
        p_C0=.7, native_admitted=True, market_eligible=True, q=.6,
        odds_a=2.41, odds_b=1.8, quotes=[{"quote_id":"immutable-q"}],
        financial_eligible=True, fit_eligible=True, primary_decision=True,
        label_release_day="2026-07-11")])


def event(team, day, *, context="LCK Summer", roster=None, source="10"):
    return dict(team_id=team, effective_release_day=day, competition_context=context,
        source_match_id=source, source_series_order=(0, int(source)), map_index=1,
        roster=roster if roster is not None else [str(team)+"p"+str(i) for i in range(5)])


def test_future_and_same_utc_day_sources_cannot_change_prior_target():
    original = panel()
    base = [event(1,"2026-07-08"), event(2,"2026-07-09")]
    expected = build_prior_targets(original, base)
    got = build_prior_targets(original, base+[event(1,"2026-07-10",source="11"),
        event(2,"2026-07-11",source="12")])
    assert got.prior_roster_a.tolist() == expected.prior_roster_a.tolist()
    assert got.prior_roster_b.tolist() == expected.prior_roster_b.tolist()
    assert got.prior_source_a.iloc[0]["source_match_id"] == "10"
    assert got.prior_source_b.iloc[0]["effective_release_day"] == "2026-07-09"
    assert original.roster_a.iloc[0][0] == "actual-a0"


def test_exact_context_newer_wrong_context_never_displaces_correct_unit():
    got = build_prior_targets(panel(), [event(1,"2026-07-08"), event(2,"2026-07-08"),
        event(1,"2026-07-09",context="LCK Academy",source="20")])
    assert got.prior_roster_a.iloc[0] == ["1p"+str(i) for i in range(5)]
    assert got.prior_source_a.iloc[0]["competition_context"] == "LCK Summer"
    assert got.native_absence.iloc[0] is None


@pytest.mark.parametrize("roster,reason", [(["x"]*5,"prior_roster_a_invalid"),
    (["x","y"],"prior_roster_a_invalid"),
    (["2p"+str(i) for i in range(5)],"prior_rosters_overlap")])
def test_latest_incomplete_duplicate_or_overlap_withholds_without_older_fallback(roster,reason):
    got = build_prior_targets(panel(), [event(1,"2026-07-07"),event(2,"2026-07-08"),
        event(1,"2026-07-09",roster=roster,source="30")])
    assert got.native_absence.iloc[0] == reason
    assert not got.native_admitted.iloc[0]
    assert np.isnan(got.p_C0.iloc[0])
    assert got.prior_source_a.iloc[0]["source_match_id"] == "30"
    assert got.financial_eligible.iloc[0]
    assert got.market_eligible.iloc[0]


def test_missing_exact_unit_never_uses_actual_roster_or_organization_and_retains_cash():
    original = panel()
    got = build_prior_targets(original, [event(1,"2026-07-08",context="LCK Academy"),
        event(2,"2026-07-08")])
    assert len(got) == len(original)
    assert got.native_absence.iloc[0] == "prior_roster_a_missing"
    assert np.isnan(got.p_C0.iloc[0])
    for name in ("row_id","phase","quotes","q","odds_a","odds_b","y",
                 "financial_eligible","fit_eligible","label_release_day"):
        assert got[name].tolist() == original[name].tolist()


def test_side_orientation_and_canonical_player_strings_are_preserved():
    original = panel()
    original.loc[0,"team1_id"], original.loc[0,"team2_id"] = 2,1
    got = build_prior_targets(original,[event(1,"2026-07-08",roster=list(range(5))),
        event(2,"2026-07-08",roster=list(range(5,10)))])
    assert got.prior_roster_a.iloc[0] == [str(i) for i in range(5,10)]
    assert got.prior_roster_b.iloc[0] == [str(i) for i in range(5)]
    assert got.odds_a.iloc[0] == 2.41 and got.y.iloc[0] == 1


def test_absent_native_support_does_not_change_unmatured_cash_exclusion():
    original = panel()
    original.loc[0,"financial_eligible"] = False
    got = build_prior_targets(original,[])
    assert not got.financial_eligible.iloc[0]
    assert not got.native_admitted.iloc[0]
    assert np.isnan(got.p_C0.iloc[0])


def test_preparation_rejects_changed_bound_source_before_native_loading(tmp_path):
    source = tmp_path/"input.json"
    source.write_text("changed")
    protocol = {"schema":"c0_input_aligned_v1","run_id":"consumer-smoke",
        "sources":{"raw_history":{"root":"repository","path":str(source),"sha256":"0"*64}},
        "code_sha256":{}}
    with pytest.raises(ValueError, match="(?i)(hash|changed|source)"):
        prepare_input_aligned(protocol,research_root=tmp_path,smoke_limit=1)


def test_utc_release_boundary_uses_normalized_day_not_local_calendar():
    original = panel()
    original.loc[0,"scenario_decision_at"] = "2026-07-10T00:15:00+02:00"
    got = build_prior_targets(original,[event(1,"2026-07-08"),event(2,"2026-07-08"),
        event(1,"2026-07-09",source="20")])
    assert got.prior_source_a.iloc[0]["source_match_id"] == "10"


def test_uncomputed_prior_input_cannot_reuse_saved_actual_probability_or_clock():
    original = panel()
    original["p_full_full"] = .7
    original["generated_at_utc"] = "2026-09-29T12:00:00Z"
    original["feature_source_max_day"] = "2026-07-09"
    got = build_prior_targets(original,[event(1,"2026-07-08"),event(2,"2026-07-08")])
    assert got.prior_roster_available.iloc[0]
    assert not got.native_admitted.iloc[0]
    assert np.isnan(got.p_C0.iloc[0]) and np.isnan(got.p_full_full.iloc[0])
    assert got.generated_at_utc.iloc[0] is None
    assert got.feature_source_max_day.iloc[0] is None


def test_smoke_stratification_is_supported_phase_bo_and_outcome_independent():
    from src.analysis.c0_input_aligned_data import _smoke_indices
    rows = []
    for index, (phase,bo,available) in enumerate([
        ("TRAIN",1,False),("TRAIN",1,True),("CAL",3,True),("ASSESS",5,True),
        ("ASSESS",1,False),("ASSESS",3,True),("ASSESS",1,True)]):
        rows.append(dict(row_id=str(index),phase=phase,best_of=bo,prior_roster_available=available,
            scenario_decision_at=f"2026-07-{index+1:02d}T12:00:00Z",y=index%2,p_C0=.9))
    candidates = pd.DataFrame(rows)
    chosen = _smoke_indices(candidates,6)
    supported = candidates.loc[chosen].query("prior_roster_available")
    assert set(supported.phase) == {"TRAIN","CAL","ASSESS"}
    assert set(supported.best_of) == {1,3,5}
    assert (~candidates.loc[chosen].prior_roster_available).any()
    candidates["y"] = 1-candidates.y
    candidates["p_C0"] = np.nan
    assert _smoke_indices(candidates,6) == chosen


@pytest.mark.parametrize("release_field", ["label_release_day", "effective_label_release_day"])
def test_released_target_is_forbidden_before_roster_query_and_retains_market_cash(monkeypatch, release_field):
    from src.analysis import c0_input_aligned_data as data
    original = panel()
    original[release_field] = "2026-07-09"
    original["p_full_full"] = .7
    events = [event(1,"2026-07-08"), event(2,"2026-07-08"),
        event(1,"2026-07-09",source="100"), event(2,"2026-07-09",source="100")]

    def forbidden_query(*args, **kwargs):
        pytest.fail("A released target must not reach prior-roster selection")

    monkeypatch.setattr(data.replay, "_select_prior_event", forbidden_query)
    got = build_prior_targets(original, events)
    row = got.iloc[0]
    assert not row.forecast_eligible
    assert row.forecast_absence == "target_already_released_before_cutoff"
    assert row.native_absence == row.forecast_absence
    assert row.native_absence_reasons == [row.forecast_absence]
    assert not row.native_admitted and not row.prior_roster_available
    assert np.isnan(row.p_C0) and np.isnan(row.p_full_full)
    assert row.prior_source_a is None and row.prior_source_b is None
    assert row.target_effective_label_release_day == "2026-07-09"
    assert row.target_native_effective_release_day == "2026-07-09"
    for name in ("row_id","fixture_id","golgg_match_id","scenario_decision_at","phase",
                 "quotes","q","odds_a","odds_b","y","market_eligible","financial_eligible",
                 "fit_eligible","primary_decision",release_field):
        assert got[name].tolist() == original[name].tolist()
    assert events[-1]["source_match_id"] == "100"


@pytest.mark.parametrize("label_day,native_day", [
    ("2026-07-11","2026-07-09"), ("2026-07-09","2026-07-11")])
def test_either_independent_target_release_forbids_even_without_self_roster_source(label_day,native_day):
    original = panel()
    original["effective_label_release_day"] = label_day
    got = build_prior_targets(original, [event(1,"2026-07-08"),event(2,"2026-07-08")],
        target_native_release_days={"100":native_day})
    assert not got.forecast_eligible.iloc[0]
    assert not got.prior_roster_available.iloc[0]
    assert got.target_native_effective_release_day.iloc[0] == native_day
    assert got.target_effective_label_release_day.iloc[0] == label_day
    assert got.market_eligible.iloc[0] and got.financial_eligible.iloc[0]
    assert got.q.iloc[0] == .6


def test_same_utc_day_target_remains_unknown_and_prior_query_never_selects_target():
    original = panel()
    original["effective_label_release_day"] = "2026-07-10"
    events = [event(1,"2026-07-08"),event(2,"2026-07-08"),
        event(1,"2026-07-10",source="100"),event(2,"2026-07-10",source="100")]
    got = build_prior_targets(original,events,target_native_release_days={"100":"2026-07-10"})
    row = got.iloc[0]
    assert row.forecast_eligible and row.forecast_absence is None
    assert row.prior_roster_available
    assert row.prior_source_a["source_match_id"] == row.prior_source_b["source_match_id"] == "10"
    assert row.target_native_effective_release_day == "2026-07-10"
    assert row.target_effective_label_release_day == "2026-07-10"


def test_target_eligibility_is_outcome_independent_and_uses_utc_cutoff_day():
    original = panel()
    original["scenario_decision_at"] = "2026-07-10T00:15:00+02:00"
    original["effective_label_release_day"] = "2026-07-09"
    events = [event(1,"2026-07-08"),event(2,"2026-07-08")]
    unknown = build_prior_targets(original,events,target_native_release_days={"100":"2026-07-09"})
    assert unknown.forecast_eligible.iloc[0]
    original["effective_label_release_day"] = "2026-07-08"
    expected = build_prior_targets(original,events,target_native_release_days={"100":"2026-07-09"})
    original["y"] = 1-original.y
    original["p_C0"] = .1
    got = build_prior_targets(original,events,target_native_release_days={"100":"2026-07-09"})
    for name in ("forecast_eligible","forecast_absence","native_absence",
                 "target_effective_label_release_day","target_native_effective_release_day"):
        assert got[name].tolist() == expected[name].tolist()
    assert not got.forecast_eligible.iloc[0]


def test_eligibility_propagates_to_saved_actual_and_fresh_panels_without_mutating_sources():
    from src.analysis.c0_input_aligned_data import _apply_forecast_eligibility
    original = pd.concat([panel(),panel()],ignore_index=True)
    original.loc[1,"row_id"] = "g:101"
    original.loc[1,"golgg_match_id"] = "101"
    original["effective_label_release_day"] = ["2026-07-09","2026-07-10"]
    releases = {"100":"2026-07-09","101":"2026-07-10"}
    actual = _apply_forecast_eligibility(original,releases)
    fresh_source = original.copy()
    fresh_source.loc[0,"market_eligible"] = False
    fresh = _apply_forecast_eligibility(fresh_source,releases)
    prior = build_prior_targets(original,[event(1,"2026-07-08"),event(2,"2026-07-08")],
        target_native_release_days=releases)
    for frame in (actual,fresh,prior):
        assert frame.forecast_eligible.tolist() == [False,True]
        assert frame.forecast_absence.iloc[0] == "target_already_released_before_cutoff"
        assert not frame.native_admitted.iloc[0] and np.isnan(frame.p_C0.iloc[0])
        assert frame.native_absence.iloc[0] == frame.forecast_absence.iloc[0]
        assert frame.financial_eligible.sum() == 2
        assert frame.quotes.tolist() == original.quotes.tolist()
        assert frame.q.tolist() == original.q.tolist()
        assert frame.odds_a.tolist() == original.odds_a.tolist()
        assert frame.odds_b.tolist() == original.odds_b.tolist()
        assert frame.y.tolist() == original.y.tolist()
    assert actual.p_C0.iloc[1] == .7 and actual.native_admitted.iloc[1]
    assert original.p_C0.tolist() == [.7,.7] and original.native_admitted.all()
    assert not fresh.market_eligible.iloc[0]


def test_forbidden_target_remains_legitimate_prior_history_for_later_other_fixture():
    original = pd.concat([panel(),panel()],ignore_index=True)
    original.loc[1,"row_id"] = "g:101"
    original.loc[1,"golgg_match_id"] = "101"
    original.loc[1,"scenario_decision_at"] = "2026-07-12T12:00:00Z"
    original["effective_label_release_day"] = ["2026-07-09","2026-07-13"]
    events = [event(1,"2026-07-08"),event(2,"2026-07-08"),
        event(1,"2026-07-09",source="100"),event(2,"2026-07-09",source="100")]
    got = build_prior_targets(original,events,
        target_native_release_days={"100":"2026-07-09","101":"2026-07-13"})
    assert got.forecast_eligible.tolist() == [False,True]
    assert got.prior_roster_available.tolist() == [False,True]
    assert got.prior_source_a.iloc[1]["source_match_id"] == "100"
    assert got.prior_source_b.iloc[1]["source_match_id"] == "100"
    assert len(got) == 2 and got.financial_eligible.sum() == 2


def test_worker_blocks_released_target_before_native_state_and_exports_zero_tensor_slots(monkeypatch,tmp_path):
    from datetime import date
    from pathlib import Path
    from types import SimpleNamespace
    from src.analysis import c0_input_aligned_data as data
    from betting_app.core.models import native_c0

    opportunities = pd.concat([panel() for _ in range(1668)],ignore_index=True)
    opportunities["row_id"] = [f"f:{index}" for index in range(1668)]
    opportunities["phase"] = ["TRAIN"]*775+["CAL"]*370+["ASSESS"]*523
    # The provider label alone would allow these queries; corrected native
    # release independently reveals that this target is already known.
    opportunities["effective_label_release_day"] = "2026-07-11"
    raw_path,release_path = tmp_path/"raw.json",tmp_path/"release.json"
    protocol = {"run_id":"regression","sources":{
        "raw_history":{"path":str(raw_path),"sha256":"raw-sha"},
        "release_map":{"path":str(release_path),"sha256":"release-sha"}},
        "code_sha256":{}}
    prepared = {"release_path":str(release_path),"roster_events":[
        event(1,"2026-07-09",source="100"),event(2,"2026-07-09",source="100")],
        "series":[SimpleNamespace(identifier="100",order=(0,100),
            games=[SimpleNamespace(day=date(2026,7,9))])],
        "cursor":0,"state":None,"history_audit":{},"release_audit":{}}

    class NeverQueryNative:
        def load(self):
            return {"name":"Causal-C0","version":"c0-native-2026-w32-e12-v1"}

        def _load_history(self):
            return raw_path,"raw-sha","release-sha",prepared

        def _validate_request(self,*args,**kwargs):
            pytest.fail("Forbidden target reached native request validation")

        def _advance_history(self,*args,**kwargs):
            pytest.fail("Forbidden target advanced native history")

        def _raw_snapshot(self,*args,**kwargs):
            pytest.fail("Forbidden target built native tensors")

    monkeypatch.setenv("BETTING_APP_C0_WORKER","1")
    monkeypatch.setattr(native_c0,"NativeC0Adapter",NeverQueryNative)
    monkeypatch.setattr(data,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(data,"_verify",lambda *args:None)
    monkeypatch.setattr(data,"_source_path",lambda reference,root:Path(reference["path"]))
    monkeypatch.setattr(data,"_load",lambda *args:(pd.DataFrame(index=range(3103)),None,None))
    monkeypatch.setattr(data,"build_panel",lambda *args,**kwargs:(opportunities.copy(),{}))
    monkeypatch.setattr(data,"_attach_start",lambda frame,quotes:frame.assign(start_at="2026-07-10T13:00:00Z"))
    receipt = data._prepare_worker(protocol,tmp_path,None)
    assert receipt["provider_denominator"] == 1668
    assert receipt["native_available_count"] == receipt["forecast_eligible_count"] == 0
    assert receipt["npz_row_ids"] == []
    assert receipt["forecast_absence_counts"] == {"target_already_released_before_cutoff":1668}
    assert prepared["cursor"] == 0 and prepared["state"] is None
    paths = data._paths("regression")
    with np.load(paths["04_feature"]/"native_snapshots.npz") as arrays:
        assert arrays["row_id"].size == 0
        for key,shape in data.TENSOR_SHAPES.items():
            assert arrays[key].shape == (0,*shape)
    assert pd.read_parquet(paths["04_feature"]/"feature_rows.parquet").empty
    for name in ("prior_panel","actual_lineup_panel","fresh300s_prior_panel"):
        saved = pd.read_parquet(paths["05_model_input"]/(name+".parquet"))
        assert len(saved) == 1668 and saved.financial_eligible.sum() == 1668
        assert not saved.forecast_eligible.any() and not saved.native_admitted.any()
        assert saved.p_C0.isna().all()
        for field in ("row_id","phase","quotes","q","odds_a","odds_b","y","market_eligible",
                      "financial_eligible","fit_eligible","primary_decision","effective_label_release_day"):
            assert saved[field].map(data._json).tolist() == opportunities[field].map(data._json).tolist()
        if name == "actual_lineup_panel":
            assert saved.actual_saved_p_C0.tolist() == opportunities.p_C0.tolist()
