"""Consumer-visible quote-origin and exact blend regressions (no external delivery)."""
from datetime import UTC, datetime, timedelta

import pytest

from betting_app.api.routers.matches import (
    _historical_consensus, _historical_prediction, _pick_snapshot,
)
from betting_app.core.models.engine import blend_market_probability


def test_fixed50_uses_logit_shrinkage_and_full_precision():
    p = blend_market_probability(.8, .4, alpha=.5, temperature=1, blending_mode="logit_shrinkage")
    assert p == pytest.approx(.6202041028867288, abs=1e-12)
    assert p * 3 * .88 - 1 == pytest.approx(.637338831620964, abs=1e-12)


def test_mid_is_temporal_and_invalid_latest_state_is_not_revived():
    start = datetime(2026, 10, 6, 12, tzinfo=UTC)
    rows = [dict(id=i, scraped_at=start-timedelta(minutes=m), is_live=live)
            for i, m, live in [(1, 60, 0), (2, 59, 0), (3, 29, 1), (4, 1, 0)]]
    assert _pick_snapshot(rows, "mid", start_at=start)["id"] == 2
    assert _pick_snapshot(rows[:3], "close", start_at=start)["is_live"] == 1


def test_consensus_uses_latest_state_each_book_and_excludes_later_peers(monkeypatch):
    from betting_app.api.routers import matches
    monkeypatch.setattr(matches, "_align", lambda row, *args: (row["odds_a"], row["odds_b"]))
    t = datetime(2026, 10, 6, 11, tzinfo=UTC)
    rows = [dict(id=1, bookmaker_id=1, scraped_at=t, is_live=0, odds_a=3., odds_b=2.),
            dict(id=2, bookmaker_id=2, scraped_at=t, is_live=0, odds_a=2., odds_b=4.),
            dict(id=3, bookmaker_id=2, scraped_at=t+timedelta(seconds=1), is_live=0, odds_a=9., odds_b=2.)]
    assert _historical_consensus(rows, t, "Alpha", "Beta") == pytest.approx((.4+2/3)/2)
    rows.append(dict(id=4, bookmaker_id=2, scraped_at=t, is_live=1, odds_a=None, odds_b=None))
    assert _historical_consensus(rows, t, "Alpha", "Beta") == pytest.approx(.4)


def test_pre_quote_base_survives_post_start_active_prediction():
    t = datetime(2026, 10, 6, 11, tzinfo=UTC)
    rows = [dict(id=1, canonical_match_id=4, model_name="EXP081", model_version="exact", predicted_at=t,
                 data_cutoff_at=t-timedelta(hours=1), prediction_status="superseded", prob_a=.8),
            dict(id=2, canonical_match_id=4, model_name="EXP081", model_version="exact", predicted_at=t+timedelta(hours=2),
                 data_cutoff_at=t, prediction_status="active", prob_a=.2)]
    assert _historical_prediction(rows, origin=t, start_at=t+timedelta(hours=1), model_name="EXP081", model_version="exact")["id"] == 1
    assert _historical_prediction(rows, origin=t, start_at=t+timedelta(hours=1), model_name="Causal-C0", model_version="exact") is None


def _context(now, *, odds=3., age=10):
    from betting_app.services.prematch_quote_service import PrematchMarketContext, PrematchQuote
    quotes = tuple(PrematchQuote(i, 4, i, f"book{i}", now-timedelta(seconds=age),
                                "Alpha", "Beta", odds, 1.45, None, None) for i in (1, 2))
    return PrematchMarketContext(4, "alpha", "beta", "LCK", "upcoming",
                                 now+timedelta(hours=1), now, quotes, 1.45/(odds+1.45), ())


def _signal_rows(now, context):
    from betting_app.services.prematch_quote_service import serialize_market_context
    name, version = "Causal-C0", "c0-native-2026-w32-e12-v1"
    row = dict(id=8, prediction_id=9, prediction_match_id=4, canonical_prediction_id=9,
               canonical_match_id=4, bookmaker_id=1, odds_snapshot_id=1, side="a", odds=3.,
               model_name=name, model_version=version, model_prob=.4, prob_a=.4, prob_b=.6,
               market_prob=context.prob_a, ev=.056, evaluated_at=now, status="new",
               prediction_status="active", predicted_at=now-timedelta(seconds=20),
               data_cutoff_at=now-timedelta(minutes=1), team_a_name="Alpha", team_b_name="Beta",
               league="LCK", start_time_normalized=context.start_at.isoformat(), bookmaker_name="book1",
               diagnostics_json={"uncertainty_required":False,"rating_disagreement":0.,"best_of":3},
               market_context_json=serialize_market_context(context, model_name=name, model_version=version))
    return row


def test_predictions_admit_before_limit_and_preserve_unrounded_ev_and_clocks(monkeypatch):
    from betting_app.api.routers import predictions
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    context = _context(now)
    valid = _signal_rows(now, context)
    legacy = dict(valid, id=7, ev=9., evaluated_at=None)
    monkeypatch.setattr(predictions, "datetime", Clock)
    monkeypatch.setattr(predictions, "query_df", lambda *args: [legacy, valid])
    monkeypatch.setattr(predictions, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    response = predictions.list_predictions(min_ev=0, max_ev=None, min_books=2,
        model_name=valid["model_name"], model_version=valid["model_version"], limit=1, db=object())
    assert response.total == 1
    signal = response.signals[0]
    assert signal.ev == pytest.approx(.056, abs=1e-12)
    assert signal.odds_snapshot_id == 1
    assert signal.evaluated_at == now.isoformat()
    assert signal.quote_scraped_at == (now-timedelta(seconds=10)).isoformat()


@pytest.mark.parametrize("change", ["mutated", "stale", "withdrawn"])
def test_predictions_withhold_changed_operational_context(monkeypatch, change):
    from dataclasses import replace
    from betting_app.api.routers import predictions
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return now
    saved = _context(now)
    row = _signal_rows(now, saved)
    current = _context(now, odds=3.01) if change == "mutated" else (
        _context(now, age=301) if change == "stale" else replace(saved, quotes=(), prob_a=None))
    monkeypatch.setattr(predictions, "datetime", Clock)
    monkeypatch.setattr(predictions, "query_df", lambda *args: [row])
    monkeypatch.setattr(predictions, "load_prematch_quotes", lambda *args, **kwargs: {4:current})
    response = predictions.list_predictions(min_ev=0, max_ev=None, min_books=1,
        model_name=row["model_name"], model_version=row["model_version"], limit=50, db=object())
    assert response.total == 0


def test_historical_requested_exp081_contract_never_borrows_current_c0(monkeypatch):
    import json
    from betting_app.api.routers import matches
    exact = dict(base_model_name="EXP081", base_model_version="sealed", alpha=.5,
                 temperature=1., blending_mode="logit_shrinkage")
    monkeypatch.setattr(matches, "query_df", lambda *args: [
        dict(model_params_json=json.dumps(exact), feature_schema_json="{}")])
    assert matches._historical_blend_contract(object(), "Hybrid-Operational-Market", "exp081-sealed") == exact
    monkeypatch.setattr(matches, "query_df", lambda *args: [])
    assert matches._historical_blend_contract(object(), "Hybrid-Operational-Market", "exp081-sealed") is None


def test_diagnostic_old_prices_cannot_build_recommendation():
    from betting_app.api.routers.matches import _build_match_recommendation
    from betting_app.api.schemas import BookmakerOddsRow
    recommendation = _build_match_recommendation(
        "Alpha", "Beta", False, [BookmakerOddsRow(bookmaker="book", canonical_odds_a=3.,
        canonical_odds_b=1.45, quote_status="diagnostic")], .4, .6, .4, .6)
    assert not recommendation.has_value
    assert recommendation.verdict == "no_odds"


def test_display_selector_normalizes_offsets_before_rank_and_null_clock_withholds_book():
    from betting_app.api.routers.matches import _latest_display_states
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    rows = [dict(bookmaker_id=1, odds_snapshot_id=1, scraped_at="2026-10-06T12:50:00+02:00"),
            dict(bookmaker_id=1, odds_snapshot_id=2, scraped_at="2026-10-06T10:55:00+00:00")]
    assert _latest_display_states(rows, now)[0]["odds_snapshot_id"] == 2
    rows.append(dict(bookmaker_id=1, odds_snapshot_id=3, scraped_at=None))
    assert _latest_display_states(rows, now) == []


@pytest.mark.parametrize("contract_present", [True, False])
def test_results_endpoint_quote_origin_consensus_and_missing_contract_visible_prices(monkeypatch, contract_present):
    import json
    from betting_app.api.routers import matches
    t = datetime(2026, 10, 6, 11, tzinfo=UTC)
    match = dict(canonical_match_id=4, team_a_name="Alpha", team_b_name="Beta", league="LCK",
                 start_time_normalized=(t+timedelta(hours=1)).isoformat(), status="finished")
    contract = dict(base_model_name="EXP081", base_model_version="sealed", alpha=.5,
                    temperature=1., blending_mode="logit_shrinkage")
    states = [dict(odds_snapshot_id=i, canonical_match_id=4, bookmaker_id=book, bookmaker=f"book{book}",
                   raw_team_a="Alpha", raw_team_b="Beta", odds_a=oa, odds_b=ob, is_live=0,
                   scraped_at=at, league="LCK", start_time_normalized=match["start_time_normalized"])
              for i, book, oa, ob, at in [(1,1,3.,2.,t), (2,2,2.,4.,t), (3,2,9.,2.,t+timedelta(seconds=10))]]
    base = dict(id=9, canonical_match_id=4, model_name="EXP081", model_version="sealed", prob_a=.8,
                prob_b=.2, predicted_at=t-timedelta(minutes=1), data_cutoff_at=t-timedelta(hours=1))
    def query(db, sql, params=None):
        if "model_artifacts" in sql:
            return [dict(model_params_json=json.dumps(contract), feature_schema_json="{}")] if contract_present else []
        if "FROM odds_snapshots" in sql:
            return states
        if "FROM canonical_predictions" in sql:
            return [base] if params["name"] == "EXP081" else []
        return [match]
    monkeypatch.setattr(matches, "query_df", query)
    response = matches.list_results(days_back=30, model_name="Hybrid-Operational-Market",
                                   model_version="exp081-sealed", odds_mode="open", db=object())
    item = response.results[0]
    side = item.bookmaker_ev_details["book1"].side_a
    assert side.odds == 3.
    assert side.market_prob == pytest.approx((.4+2/3)/2)
    assert item.probability_basis == "historical_quote_recomputation"
    if contract_present:
        p = blend_market_probability(.8, (.4+2/3)/2, alpha=.5, temperature=1., blending_mode="logit_shrinkage")
        assert side.model_prob == pytest.approx(p, abs=1e-12)
    else:
        assert side.model_prob is None and side.ev is None and side.kelly is None
        assert item.best_odds_a is not None
        assert item.probability_unavailable_reason == "blend_contract_missing_or_conflicting"


def test_alerts_validate_before_cooldown_and_limit_and_dry_run_never_delivers(monkeypatch):
    from types import SimpleNamespace
    from betting_app.services import alert_service as alerts
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return now
    context = _context(now)
    valid = dict(_signal_rows(now, context), signal_id=8, feature_status="ready")
    rows = [dict(valid, signal_id=i+20, evaluated_at=None, ev=9.) for i in range(30)] + [valid]
    cfg = SimpleNamespace(is_enabled=True, min_ev=.05, min_odds=1.25, max_odds=12., cooldown_hours=6.,
                          discord_webhook_url="https://delivery.invalid", discord_enabled=True,
                          telegram_bot_token=None, telegram_chat_id=None, telegram_enabled=False)
    logged = []
    db = SimpleNamespace(add=logged.append, commit=lambda: None)
    cooldowns = []
    def forbidden(*args, **kwargs):
        raise AssertionError("External delivery forbidden")
    monkeypatch.setattr(alerts, "datetime", Clock)
    monkeypatch.setattr(alerts, "get_or_create_alert_config", lambda db: cfg)
    monkeypatch.setattr(alerts, "query_df", lambda *args: rows)
    monkeypatch.setattr(alerts, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    monkeypatch.setattr(alerts, "is_alert_on_cooldown", lambda *args: cooldowns.append(args) or False)
    monkeypatch.setattr(alerts, "send_discord_notification", forbidden)
    monkeypatch.setattr(alerts, "send_telegram_notification", forbidden)
    response = alerts.scan_and_dispatch_ev_alerts(db, force_dry_run=True)
    assert response["dispatched"] == 1
    assert len(cooldowns) == 1 and len(logged) == 1
    assert response["alerts"][0]["status"] == "simulated"


def test_parlays_reject_clockless_legacy_hybrid_before_any_leg(monkeypatch):
    from types import SimpleNamespace
    from betting_app.services import parlay_service as parlays
    now = datetime.now(UTC)
    context = _context(now)
    legacy = dict(id=1, canonical_match_id=4, model_name=parlays.ACTIVE_HYBRID_NAME,
                  model_version=parlays.ACTIVE_HYBRID_VERSION, prediction_status="active",
                  prob_a=.8, prob_b=.2, predicted_at=None, data_cutoff_at=None)
    db = SimpleNamespace(execute=lambda *args: SimpleNamespace(mappings=lambda: [legacy]))
    monkeypatch.setattr(parlays, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    response = parlays.find_parlay_recommendations(db)
    assert response.top_parlay is None


@pytest.mark.parametrize("change", ["mutated", "stale", "withdrawn"])
def test_parlays_withhold_changed_context_before_leg_construction(monkeypatch, change):
    from dataclasses import replace
    from types import SimpleNamespace
    from betting_app.services import parlay_service as parlays
    from betting_app.services.prematch_quote_service import serialize_market_context
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return now
    saved = _context(now)
    current = _context(now, odds=3.01) if change == "mutated" else (
        _context(now, age=301) if change == "stale" else replace(saved, quotes=(), prob_a=None))
    prediction = dict(id=9, canonical_match_id=4, model_name=parlays.ACTIVE_HYBRID_NAME,
                      model_version=parlays.ACTIVE_HYBRID_VERSION, prediction_status="active",
                      predicted_at=now-timedelta(seconds=20), data_cutoff_at=now-timedelta(minutes=1),
                      prob_a=.8, prob_b=.2, diagnostics_json={"market_context":serialize_market_context(saved)})
    db = SimpleNamespace(execute=lambda *args: SimpleNamespace(mappings=lambda: [prediction]))
    monkeypatch.setattr(parlays, "datetime", Clock)
    monkeypatch.setattr(parlays, "load_prematch_quotes", lambda *args, **kwargs: {4:current})
    monkeypatch.setattr(parlays, "hybrid_prediction_contract_is_current", lambda *args, **kwargs: (True,None))
    assert parlays.find_parlay_recommendations(db).top_parlay is None


def test_mutated_stored_hybrid_is_ignored_and_pure_base_current_market_blended_once(monkeypatch):
    from dataclasses import replace
    from betting_app.api.routers import matches
    from betting_app.services.prematch_quote_service import serialize_market_context
    hybrid = matches.get_active_hybrid()
    now = datetime(2026, 10, 6, 11, tzinfo=UTC)
    current = _context(now)
    saved = replace(current, quotes=tuple(replace(q, odds_a=2.) for q in current.quotes), prob_a=1.45/3.45)
    common = dict(canonical_match_id=4, prediction_status="active", predicted_at=now-timedelta(seconds=20),
                  data_cutoff_at=now-timedelta(minutes=1))
    rows = [dict(common, id=1, model_name=hybrid.base_model.name, model_version=hybrid.base_model.version,
                 prob_a=.8, prob_b=.2, diagnostics_json={}),
            dict(common, id=2, model_name=hybrid.hybrid_model_name, model_version=hybrid.hybrid_model_version,
                 prob_a=.9, prob_b=.1, diagnostics_json={"market_context":serialize_market_context(saved)})]
    monkeypatch.setattr(matches, "load_prematch_quotes", lambda *args, **kwargs: {4:current})
    monkeypatch.setattr(matches, "hybrid_prediction_contract_is_current", lambda *args, **kwargs: (True,None))
    serving = matches._serving_predictions(object(), rows, current)
    expected = blend_market_probability(.8,current.prob_a,alpha=hybrid.alpha,
                                       temperature=hybrid.temperature,blending_mode=hybrid.blending_mode)
    assert serving["hybrid_prob_a"] == pytest.approx(expected,abs=1e-12)
    assert serving["hybrid_prob_b"] == pytest.approx(1-expected,abs=1e-12)


def test_all_matching_surfaces_handle_unresolved_orientation_without_indexing_none(monkeypatch):
    from dataclasses import replace
    from betting_app.api.routers import matches
    now = datetime.now(UTC)
    context = replace(_context(now), quotes=(), prob_a=None)
    match = dict(id=4, canonical_match_id=4, team_a_name="Alpha", team_b_name="Beta",
                 normalized_team_a="alpha", normalized_team_b="beta", league="LCK", status="upcoming",
                 best_of=3, start_time_normalized=context.start_at.isoformat())
    odds = dict(match, odds_snapshot_id=1, bookmaker_id=1, bookmaker="book1",
                raw_team_a="Unrelated", raw_team_b="Other", odds_a=3., odds_b=2.,
                is_live=0, scraped_at=now-timedelta(seconds=10))
    prediction = dict(id=9, canonical_match_id=4, model_name="Causal-C0",
                      model_version="c0-native-2026-w32-e12-v1", prob_a=.4, prob_b=.6, predicted_at=now)
    def query(db, sql, params=None):
        if "FROM odds_snapshots" in sql:
            return [dict(odds)]
        if "FROM canonical_matches" in sql:
            return [dict(match)]
        return []
    monkeypatch.setattr(matches, "query_df", query)
    monkeypatch.setattr(matches, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    monkeypatch.setattr(matches, "suggest_mapping", lambda *args: (None,0.,None))
    monkeypatch.setattr(matches, "_load_roster_overrides", lambda *args: {})
    assert matches.list_matches(db=object()).matches == []
    detail = matches.match_detail(4,db=object())
    assert detail.odds[0].canonical_odds_a is None
    assert detail.odds[0].quote_status == "unavailable"
    history = matches.odds_history(4,db=object())
    assert history[0].canonical_odds_a is None and history[0].canonical_odds_b is None
    def history_query(db, sql, params=None):
        if "FROM canonical_predictions" in sql: return [prediction]
        return query(db,sql,params)
    monkeypatch.setattr(matches, "query_df", history_query)
    points = matches.prediction_history(4,db=object())
    assert points[0].avg_odds_a is None and points[0].ev_a is None


@pytest.mark.parametrize("saved", [None, [], "invalid"])
def test_parlays_withhold_nonobject_market_context_without_aborting(monkeypatch, saved):
    from types import SimpleNamespace
    from betting_app.services import parlay_service as parlays
    now = datetime.now(UTC)
    context = _context(now)
    prediction = dict(id=9, canonical_match_id=4, model_name=parlays.ACTIVE_HYBRID_NAME,
                      model_version=parlays.ACTIVE_HYBRID_VERSION, prediction_status="active",
                      prob_a=.8, prob_b=.2, predicted_at=now-timedelta(seconds=20),
                      data_cutoff_at=now-timedelta(minutes=1), diagnostics_json={"market_context":saved})
    db = SimpleNamespace(execute=lambda *args: SimpleNamespace(mappings=lambda: [prediction]))
    monkeypatch.setattr(parlays, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    response = parlays.find_parlay_recommendations(db)
    assert response.top_parlay is None and response.parlays == []


@pytest.mark.parametrize("replacement", ["price", "withdrawn"])
def test_detail_withholds_whole_market_when_display_read_observes_upsert(monkeypatch, replacement):
    from betting_app.api.routers import matches
    now = datetime.now(UTC)
    context = _context(now)
    match = dict(id=4, canonical_match_id=4, team_a_name="Alpha", team_b_name="Beta",
                 normalized_team_a="alpha", normalized_team_b="beta", league="LCK", status="upcoming",
                 best_of=3, start_time_normalized=context.start_at.isoformat())
    rows = [dict(match, odds_snapshot_id=q.odds_snapshot_id, bookmaker_id=q.bookmaker_id,
                 bookmaker=q.bookmaker, raw_team_a=q.raw_team_a, raw_team_b=q.raw_team_b,
                 odds_a=q.odds_a, odds_b=q.odds_b, is_live=0, scraped_at=q.scraped_at, source_url=None)
            for q in context.quotes]
    def query(db, sql, params=None):
        if "FROM odds_snapshots" in sql:
            return [dict(row) for row in rows]
        if "FROM canonical_matches" in sql:
            return [dict(match)]
        return []
    monkeypatch.setattr(matches, "query_df", query)
    monkeypatch.setattr(matches, "load_prematch_quotes", lambda *args, **kwargs: {4:context})
    monkeypatch.setattr(matches, "suggest_mapping", lambda *args: ("mapped",1.,None))
    monkeypatch.setattr(matches, "_load_roster_overrides", lambda *args: {})
    assert all(row.quote_status == "actionable" for row in matches.match_detail(4,db=object()).odds)
    rows[0].update(odds_a=3.01) if replacement == "price" else rows[0].update(
        odds_a=None, odds_b=None, is_live=1)
    response = matches.match_detail(4,db=object())
    assert all(row.quote_status != "actionable" for row in response.odds)
    assert all(row.ev_a is None and row.ev_b is None and row.kelly_a is None for row in response.odds)
    assert response.odds[0].canonical_odds_a == (3.01 if replacement == "price" else None)
