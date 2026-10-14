from datetime import UTC, datetime, timedelta
import pandas as pd
import pytest
from betting_app.services import prematch_quote_service as service

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)

def install(monkeypatch, states):
    match = dict(id=1, normalized_team_a='Alpha', normalized_team_b='Beta', league='LCK', status='scheduled', start_time_normalized=(NOW+timedelta(hours=1)).isoformat())
    monkeypatch.setattr(service, 'query_df', lambda sql, params: pd.DataFrame([match] if 'FROM canonical_matches' in sql else states))
    monkeypatch.setattr(service, 'align_snapshot_odds', lambda a,b,ra,rb,oa,ob,**kw: (oa,ob) if (a,b)==(ra,rb) else None)

def quote(id=1, **kw):
    return dict(id=id,canonical_match_id=1,bookmaker_id=1,bookmaker='Book',raw_team_a='Alpha',raw_team_b='Beta',odds_a=3.,odds_b=2.,is_live=0,scraped_at=NOW-timedelta(seconds=300),source_url=None,offer_url=None,**kw)

def test_newest_invalid_state_does_not_revive_old(monkeypatch):
    old=quote(); new=quote(2); new.update(scraped_at=NOW,odds_a=None)
    install(monkeypatch,[old,new]); ctx=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert not ctx.quotes
    assert ctx.rejections[0].reason=='quote_price_invalid'

def test_exact_freshness_boundary_and_mutated_same_id(monkeypatch):
    state=quote(); install(monkeypatch,[state])
    ctx=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert ctx.prob_a==pytest.approx(.4)
    saved=service.serialize_market_context(ctx)
    assert service.stored_market_context_is_current(saved,evaluation_context=ctx,current_context=ctx)==(True,None)
    state['odds_a']=3.1
    changed=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert not service.stored_market_context_is_current(saved,evaluation_context=changed,current_context=changed)[0]
    state['odds_a']=3.
    assert not service.load_prematch_quotes([1],decision_at=NOW+timedelta(microseconds=1))[1].quotes

def test_prediction_arrival_orders_and_offset_ranking():
    row=dict(id=1,model_name='S',model_version='v',prediction_status='active',data_cutoff_at=(NOW-timedelta(minutes=2)).isoformat(),predicted_at=(NOW-timedelta(minutes=1)).isoformat())
    assert service.prediction_time_is_actionable(row,decision_at=NOW,start_at=NOW+timedelta(hours=1),model_name='S',model_version='v')==(True,None)
    future=dict(row,id=2,predicted_at=(NOW+timedelta(seconds=1)).isoformat())
    assert not service.prediction_time_is_actionable(future,decision_at=NOW,start_at=NOW+timedelta(hours=1),model_name='S',model_version='v')[0]
    same=dict(row,id=3,predicted_at=(NOW-timedelta(minutes=1)).astimezone(__import__('datetime').timezone(timedelta(hours=2))).isoformat())
    assert service.latest_predictions([row,same])[0]['id']==3

@pytest.mark.parametrize('age',[True,-1,float('nan'),float('inf')])
def test_invalid_age_rejected(age):
    with pytest.raises(ValueError): service.load_prematch_quotes([1],decision_at=NOW,max_age_seconds=age)

@pytest.mark.parametrize('change,reason',[
    ({'is_live':1},'quote_live_or_unknown'),
    ({'is_live':None},'quote_live_or_unknown'),
    ({'scraped_at':None},'quote_clock_unknown'),
    ({'raw_team_a':'Unrelated'},'participant_identity_unresolved'),
])
def test_latest_withdrawal_never_revives_previous(monkeypatch,change,reason):
    old=quote(); new=quote(2); new.update({'scraped_at':NOW,**change})
    install(monkeypatch,[old,new])
    context=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert context.prob_a is None
    assert context.rejections[0].reason==reason

def test_book_mean_and_peer_mutation(monkeypatch):
    one=quote(); two=quote(2); two.update(bookmaker_id=2,bookmaker='Peer',odds_a=2.,odds_b=4.)
    install(monkeypatch,[one,two])
    context=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert context.prob_a==pytest.approx((.4+2/3)/2)
    saved=service.serialize_market_context(context,model_name='S',model_version='v')
    two['odds_b']=3.
    current=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert not service.stored_market_context_is_current(saved,evaluation_context=current,current_context=current)[0]

@pytest.mark.parametrize('prediction_delay',[-120,-10])
def test_signal_allows_either_quote_or_forecast_first(monkeypatch,prediction_delay):
    state=quote(); state['scraped_at']=NOW-timedelta(seconds=60)
    install(monkeypatch,[state])
    context=service.load_prematch_quotes([1],decision_at=NOW)[1]
    prediction=dict(id=7,canonical_match_id=1,model_name='S',model_version='v',prediction_status='active',
        predicted_at=(NOW+timedelta(seconds=prediction_delay)).isoformat(),
        data_cutoff_at=(NOW-timedelta(seconds=150)).isoformat(),prob_a=.4,prob_b=.6,diagnostics_json='{}')
    signal=dict(canonical_prediction_id=7,canonical_match_id=1,odds_snapshot_id=1,bookmaker_id=1,
        side='a',odds=3.,model_prob=.4,market_prob=.4,status='new',evaluated_at=NOW,
        market_context_json=service.serialize_market_context(context,model_name='S',model_version='v'))
    assert service.signal_is_actionable(signal,prediction,evaluation_context=context,current_context=context)==(True,None)
    signal['evaluated_at']=None
    assert not service.signal_is_actionable(signal,prediction,evaluation_context=context,current_context=context)[0]

def test_bare_market_override_rejected_before_prediction():
    from betting_app.services.upcoming_inference_service import evaluate_bayesian_market_hybrid
    with pytest.raises(ValueError,match='Bare market'):
        evaluate_bayesian_market_hybrid({'market_prob_a':.4},None)

def test_actual_database_market_isolation_and_same_clock_id_order():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    engine=create_engine('sqlite:///:memory:')
    with Session(engine) as session:
        session.execute(text('CREATE TABLE canonical_matches(id INTEGER, normalized_team_a TEXT, normalized_team_b TEXT, league TEXT,status TEXT,start_time_normalized TEXT)'))
        session.execute(text('CREATE TABLE bookmakers(id INTEGER,name TEXT)'))
        session.execute(text('CREATE TABLE odds_snapshots(id INTEGER,canonical_match_id INTEGER,bookmaker_id INTEGER,market_type TEXT,raw_team_a TEXT,raw_team_b TEXT,odds_a REAL,odds_b REAL,is_live INTEGER,scraped_at TEXT,source_url TEXT,offer_url TEXT)'))
        session.execute(text("INSERT INTO canonical_matches VALUES(1,'Alpha','Beta','LCK','upcoming',:start)"),{'start':(NOW+timedelta(hours=1)).isoformat()})
        session.execute(text("INSERT INTO bookmakers VALUES(1,'Book')"))
        insert=text("INSERT INTO odds_snapshots VALUES(:id,1,1,:market,'Alpha','Beta',3.,2.,:live,:clock,NULL,NULL)")
        for id,market,live in [(1,'match_winner',0),(2,'map_winner',1)]:
            session.execute(insert,dict(id=id,market=market,live=live,clock=NOW.isoformat()))
        context=service.load_prematch_quotes([1],decision_at=NOW,session=session)[1]
        assert [q.odds_snapshot_id for q in context.quotes]==[1]
        session.execute(insert,dict(id=3,market='match_winner',live=1,clock=NOW.isoformat()))
        context=service.load_prematch_quotes([1],decision_at=NOW,session=session)[1]
        assert not context.quotes
        assert context.rejections[0].odds_snapshot_id==3
        session.execute(text("UPDATE canonical_matches SET start_time_normalized=NULL"))
        assert service.load_prematch_quotes([1],decision_at=NOW,session=session)[1].rejections[0].reason=='start_unknown'
    engine.dispose()

def test_missing_market_retains_diagnosed_sports_probability(monkeypatch):
    from types import SimpleNamespace
    from betting_app.services import upcoming_inference_service as producer
    from betting_app.core.models.contract import ModelSpec, HybridSpec, UnifiedPredictionResult
    base=ModelSpec('S','v','features')
    hybrid=HybridSpec(base_model=base,alpha=.5,temperature=1.,hybrid_model_name='H',blending_mode='logit_shrinkage')
    spec=SimpleNamespace(name='H',version='h-v',feature_version='features',family='bayesian_market_hybrid')
    sports=UnifiedPredictionResult(1,.8,.2,best_of=3,model_name='S',model_version='v',diagnostics={'data_cutoff_at':(NOW-timedelta(hours=1)).isoformat()})
    context=service.PrematchMarketContext(1,'Alpha','Beta','LCK','upcoming',NOW+timedelta(hours=1),NOW,(),None,(service.QuoteRejection(None,None,'quote_missing'),))
    monkeypatch.setattr(producer,'get_hybrid_spec',lambda _:hybrid)
    monkeypatch.setattr(producer.PredictionEngine,'predict_from_features',lambda *args:sports)
    monkeypatch.setattr(producer,'load_prematch_quotes',lambda *args,**kwargs:{1:context})
    result=producer.evaluate_bayesian_market_hybrid({'canonical_match_id':1},spec)
    assert result.prob_a==.8
    assert result.diagnostics['p_market_novig'] is None
    assert result.diagnostics['market_features_used'] is False
    assert result.diagnostics['market_unavailable_reason']=='market_missing'

def test_empty_candidates_stale_only_evaluated_model_signals(monkeypatch):
    from contextlib import contextmanager
    from sqlalchemy import create_engine,text
    from sqlalchemy.orm import Session
    from betting_app.core.db import _ConnectionWrapper
    from betting_app.services import upcoming_inference_service as producer
    engine=create_engine('sqlite:///:memory:')
    with Session(engine) as session:
        session.execute(text('CREATE TABLE canonical_predictions(id INTEGER,model_name TEXT,model_version TEXT)'))
        session.execute(text('CREATE TABLE model_ev_signals(id INTEGER,canonical_prediction_id INTEGER,status TEXT)'))
        session.execute(text("INSERT INTO canonical_predictions VALUES(1,'S','v'),(2,'other','v')"))
        session.execute(text("INSERT INTO model_ev_signals VALUES(1,1,'new'),(2,2,'new')"))
        @contextmanager
        def transaction():
            yield _ConnectionWrapper(session)
            session.commit()
        monkeypatch.setattr(producer,'transaction',transaction)
        monkeypatch.setattr(producer,'query_df',lambda sql,params=None:pd.DataFrame([{'open_stake':0.}]) if 'FROM bets' in sql else pd.DataFrame())
        assert producer.generate_model_ev_signals(model_name='S',model_version='v')==[]
        assert session.execute(text('SELECT status FROM model_ev_signals ORDER BY id')).scalars().all()==['stale','new']
    engine.dispose()

def test_exact_hybrid_registration_rejects_parameter_reuse(monkeypatch):
    from contextlib import contextmanager
    from sqlalchemy import create_engine,text
    from sqlalchemy.orm import Session
    from betting_app.core.db import _ConnectionWrapper
    from betting_app.core.models.contract import ModelSpec,HybridSpec
    from betting_app.services import upcoming_inference_service as producer
    engine=create_engine('sqlite:///:memory:')
    with Session(engine) as session:
        session.execute(text('CREATE TABLE model_artifacts(id INTEGER PRIMARY KEY,model_name TEXT,model_version TEXT,feature_schema_json TEXT,model_params_json TEXT,status TEXT,UNIQUE(model_name,model_version))'))
        @contextmanager
        def transaction():
            yield _ConnectionWrapper(session)
        monkeypatch.setattr(producer,'transaction',transaction)
        base=ModelSpec('S','v','features')
        original=HybridSpec(base_model=base,hybrid_model_name='H',alpha=.5,temperature=1.,blending_mode='logit_shrinkage')
        first=producer.register_hybrid_model(spec=original,base_version='v',version='hv')
        assert producer.register_hybrid_model(spec=original,base_version='v',version='hv')==first
        conflicting=HybridSpec(base_model=base,hybrid_model_name='H',alpha=.6,temperature=1.,blending_mode='logit_shrinkage')
        with pytest.raises(ValueError,match='Conflicting'):
            producer.register_hybrid_model(spec=conflicting,base_version='v',version='hv')
        import json
        saved=json.loads(session.execute(text('SELECT model_params_json FROM model_artifacts')).scalar_one())
        assert (saved['base_model_name'],saved['base_model_version'],saved['alpha'])==('S','v',.5)
    engine.dispose()

def test_context_retains_actual_canonical_display_names(monkeypatch):
    match=dict(id=1,team_a_name='Alpha Display',team_b_name='Beta Display',
        normalized_team_a='alpha',normalized_team_b='beta',status='upcoming',
        start_time_normalized=(NOW+timedelta(hours=1)).isoformat())
    monkeypatch.setattr(service,'query_df',lambda sql,params:pd.DataFrame([match]) if 'FROM canonical_matches' in sql else pd.DataFrame())
    context=service.load_prematch_quotes([1],decision_at=NOW)[1]
    assert (context.team_a_name,context.team_b_name)==('Alpha Display','Beta Display')


def test_hybrid_market_input_must_precede_hybrid_issuance(monkeypatch):
    params = dict(base_model_name='S',base_model_version='v',alpha=.5,
                  temperature=1.,blending_mode='logit_shrinkage')
    artifact = dict(model_params_json=params,feature_schema_json={'base_model':'S/v'})
    monkeypatch.setattr(service,'query_df',lambda *args,**kwargs:pd.DataFrame([artifact]))
    origin = NOW-timedelta(seconds=10)
    prediction = dict(model_name='H',model_version='hv',canonical_match_id=1,
        predicted_at=NOW,data_cutoff_at=origin,diagnostics_json=dict(
            params,base_data_cutoff_at=NOW-timedelta(minutes=1),
            market_context=dict(model_name='H',model_version='hv',
                base_model_name='S',base_model_version='v',decision_at=origin)))
    assert service.hybrid_prediction_contract_is_current(prediction)==(True,None)
    prediction['predicted_at']=NOW-timedelta(seconds=20)
    prediction['data_cutoff_at']=NOW-timedelta(seconds=30)
    assert service.hybrid_prediction_contract_is_current(prediction)==(False,'hybrid_base_clock_invalid')
