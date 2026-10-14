"""Operational, as-of quote admission. This is not authentic G0 provenance."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from numbers import Integral
from typing import Any
from betting_app.core.db import query_df
from betting_app.services.canonical_match_service import align_snapshot_odds

@dataclass(frozen=True)
class PrematchQuote:
    odds_snapshot_id: int
    canonical_match_id: int
    bookmaker_id: int
    bookmaker: str
    scraped_at: datetime
    raw_team_a: str
    raw_team_b: str
    odds_a: float
    odds_b: float
    source_url: str | None
    offer_url: str | None

@dataclass(frozen=True)
class QuoteRejection:
    bookmaker_id: int | None
    odds_snapshot_id: int | None
    reason: str

@dataclass(frozen=True)
class PrematchMarketContext:
    canonical_match_id: int
    team_a_name: str
    team_b_name: str
    league: str | None
    status: str | None
    start_at: datetime | None
    decision_at: datetime
    quotes: tuple[PrematchQuote, ...]
    prob_a: float | None
    rejections: tuple[QuoteRejection, ...]

    @property
    def book_count(self) -> int:
        return len(self.quotes)


def parse_aware_utc(value: Any) -> datetime | None:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 1 else None
    except (TypeError, ValueError, OverflowError):
        return None


def load_prematch_quotes(canonical_match_ids: Sequence[int], *, decision_at: datetime, max_age_seconds: float = 300.0, session=None) -> dict[int, PrematchMarketContext]:
    decision = parse_aware_utc(decision_at)
    if not isinstance(decision_at, datetime) or decision is None:
        raise ValueError('decision_at must be timezone-aware')
    if isinstance(max_age_seconds, bool):
        raise ValueError('invalid quote age limit')
    try:
        age_limit = float(max_age_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid quote age limit') from exc
    if not math.isfinite(age_limit) or age_limit < 0:
        raise ValueError('invalid quote age limit')
    if isinstance(canonical_match_ids, (str, bytes)) or not isinstance(canonical_match_ids, Sequence):
        raise ValueError('match IDs must be a sequence of positive integers')
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in canonical_match_ids):
        raise ValueError('match IDs must be positive integers')
    ids = list(dict.fromkeys(int(x) for x in canonical_match_ids))
    if not ids:
        return {}
    marks = ','.join('?' for _ in ids)
    query_options = {'session': session} if session is not None else {}
    matches = {int(r['id']): r for r in query_df(f'SELECT * FROM canonical_matches WHERE id IN ({marks})', tuple(ids), **query_options).to_dict('records')}
    states = query_df(f"SELECT os.*, b.name AS bookmaker FROM odds_snapshots os LEFT JOIN bookmakers b ON b.id=os.bookmaker_id WHERE os.market_type='match_winner' AND os.canonical_match_id IN ({marks})", tuple(ids), **query_options).to_dict('records')
    selected: dict[tuple[int, int], tuple[datetime | None, dict]] = {}
    for row in states:
        key = (int(row['canonical_match_id']), int(row['bookmaker_id']))
        clock = parse_aware_utc(row.get('scraped_at'))
        if clock is not None and clock > decision:
            continue
        prior = selected.get(key)
        # Any unorderable observation makes the book unavailable, rather than reviving history.
        if prior is None or clock is None or (prior[0] is not None and (clock, int(row['id'])) > (prior[0], int(prior[1]['id']))):
            selected[key] = (clock, row)
    result = {}
    for mid in ids:
        match = matches.get(mid, {})
        start = parse_aware_utc(match.get('start_time_normalized'))
        status = match.get('status')
        a = str(match.get('team_a_name') or match.get('normalized_team_a') or '')
        b = str(match.get('team_b_name') or match.get('normalized_team_b') or '')
        common = 'match_missing' if not match else 'start_unknown' if start is None else 'match_started' if decision >= start or str(status).lower() in {'finished','completed','live','in_progress','started','cancelled','canceled'} else None
        quotes, rejects = [], []
        for (fixture, book), (clock, row) in selected.items():
            if fixture != mid:
                continue
            reason = common
            oa, ob = _price(row.get('odds_a')), _price(row.get('odds_b'))
            aligned = None
            if reason is None:
                if clock is None: reason = 'quote_clock_unknown'
                elif row.get('is_live') is None or isinstance(row.get('is_live'), str) or row.get('is_live') != 0: reason = 'quote_live_or_unknown'
                elif (decision-clock).total_seconds() > age_limit: reason = 'quote_stale'
                elif oa is None or ob is None: reason = 'quote_price_invalid'
                else:
                    aligned = align_snapshot_odds(a,b,str(row.get('raw_team_a') or ''),str(row.get('raw_team_b') or ''),oa,ob,league=match.get('league'),source_system='bookmaker',match_date=start.date())
                    if aligned is None: reason = 'participant_identity_unresolved'
            if reason:
                rejects.append(QuoteRejection(book, int(row['id']), reason))
            else:
                quotes.append(PrematchQuote(int(row['id']),mid,book,str(row.get('bookmaker') or ''),clock,str(row.get('raw_team_a') or ''),str(row.get('raw_team_b') or ''),float(aligned[0]),float(aligned[1]),row.get('source_url'),row.get('offer_url')))
        quotes.sort(key=lambda q:(q.bookmaker_id,q.odds_snapshot_id))
        if not quotes and not rejects:
            rejects.append(QuoteRejection(None,None,common or 'quote_missing'))
        q = sum((1/x.odds_a)/(1/x.odds_a+1/x.odds_b) for x in quotes)/len(quotes) if quotes else None
        result[mid] = PrematchMarketContext(mid,a,b,match.get('league'),status,start,decision,tuple(quotes),q,tuple(rejects))
    return result


def latest_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest = {}
    for item in rows:
        row = dict(item)
        clock = parse_aware_utc(row.get('predicted_at'))
        key = (row.get('canonical_match_id'),row.get('model_name'),row.get('model_version'))
        rank = (clock is not None, clock or datetime.min.replace(tzinfo=UTC), int(row['id']))
        if key not in latest or rank > latest[key][0]: latest[key] = (rank,row)
    return [pair[1] for pair in latest.values()]


def prediction_time_is_actionable(row, *, decision_at, start_at, model_name, model_version):
    get = row.get if isinstance(row, Mapping) else lambda k,d=None:getattr(row,k,d)
    if get('model_name') != model_name or get('model_version') != model_version: return False,'prediction_identity_mismatch'
    if get('prediction_status') != 'active': return False,'prediction_inactive'
    cutoff, predicted, decision, start = map(parse_aware_utc,(get('data_cutoff_at'),get('predicted_at'),decision_at,start_at))
    if any(x is None for x in (cutoff,predicted,decision,start)): return False,'prediction_clock_unknown'
    if not cutoff <= predicted <= decision < start: return False,'prediction_clock_invalid'
    return True,None


def serialize_market_context(context, *, model_name=None, model_version=None, base_model_name=None, base_model_version=None):
    return {'schema':'prematch_market_context_v1','version':1,'canonical_match_id':context.canonical_match_id,'team_a_name':context.team_a_name,'team_b_name':context.team_b_name,'league':context.league,'status':context.status,'decision_at':context.decision_at.isoformat(),'start_at':context.start_at.isoformat() if context.start_at else None,'model_name':model_name,'model_version':model_version,'base_model_name':base_model_name,'base_model_version':base_model_version,'sources':[{'odds_snapshot_id':q.odds_snapshot_id,'canonical_match_id':q.canonical_match_id,'bookmaker_id':q.bookmaker_id,'bookmaker':q.bookmaker,'scraped_at':q.scraped_at.isoformat(),'raw_team_a':q.raw_team_a,'raw_team_b':q.raw_team_b,'odds_a':q.odds_a,'odds_b':q.odds_b,'source_url':q.source_url,'offer_url':q.offer_url,'market_type':'match_winner','is_live':0} for q in sorted(context.quotes,key=lambda q:(q.bookmaker_id,q.odds_snapshot_id))],'prob_a':context.prob_a,'book_count':len(context.quotes),'rejections':[{'bookmaker_id':r.bookmaker_id,'odds_snapshot_id':r.odds_snapshot_id,'reason':r.reason} for r in context.rejections]}


def _saved(value):
    if isinstance(value,str):
        try: value=json.loads(value)
        except (ValueError,TypeError): return None
    return value if isinstance(value,Mapping) else None


def stored_market_context_is_current(saved, *, evaluation_context, current_context):
    saved = _saved(saved)
    if not saved or saved.get('schema') != 'prematch_market_context_v1' or saved.get('version') != 1: return False,'market_context_missing'
    if not saved.get('sources') or saved.get('prob_a') is None: return False,'market_context_unavailable'
    if parse_aware_utc(saved.get('decision_at')) != evaluation_context.decision_at: return False,'market_evaluation_mismatch'
    fields = ('canonical_match_id','team_a_name','team_b_name','league','status','start_at','sources','prob_a','book_count')
    for context in (evaluation_context,current_context):
        actual = serialize_market_context(context)
        if any(saved.get(k) != actual[k] for k in fields): return False,'market_context_changed'
        if not context.start_at or context.decision_at >= context.start_at: return False,'match_started'
        if any(not 0 <= (context.decision_at-q.scraped_at).total_seconds() <= 300 for q in context.quotes): return False,'quote_stale'
    if current_context.decision_at < evaluation_context.decision_at: return False,'market_evaluation_future'
    return True,None


def signal_is_actionable(signal, prediction, *, evaluation_context, current_context, session=None):
    sg = signal.get if isinstance(signal,Mapping) else lambda k,d=None:getattr(signal,k,d)
    pg = prediction.get if isinstance(prediction,Mapping) else lambda k,d=None:getattr(prediction,k,d)
    saved = _saved(sg('market_context_json'))
    evaluated = parse_aware_utc(sg('evaluated_at'))
    if evaluated is None or evaluated != evaluation_context.decision_at: return False,'signal_evaluation_unknown'
    if sg('status') != 'new': return False,'signal_inactive'
    if sg('canonical_prediction_id') != pg('id') or sg('canonical_match_id') != pg('canonical_match_id') or sg('canonical_match_id') != current_context.canonical_match_id: return False,'signal_lineage_mismatch'
    if not saved or saved.get('model_name') != pg('model_name') or saved.get('model_version') != pg('model_version'): return False,'signal_identity_mismatch'
    valid,reason=prediction_time_is_actionable(prediction,decision_at=evaluated,start_at=evaluation_context.start_at,model_name=saved['model_name'],model_version=saved['model_version'])
    if not valid: return valid,reason
    valid,reason=stored_market_context_is_current(saved,evaluation_context=evaluation_context,current_context=current_context)
    if not valid: return valid,reason
    diagnostics=_saved(pg('diagnostics_json')) or {}
    if diagnostics.get('market_context'):
        forecast=_saved(diagnostics['market_context'])
        if not forecast: return False,'forecast_market_context_missing'
        clock=parse_aware_utc(forecast.get('decision_at'))
        if clock is None: return False,'forecast_market_context_missing'
        if forecast.get('model_name') != pg('model_name') or forecast.get('model_version') != pg('model_version'):
            return False,'forecast_identity_mismatch'
        valid,reason=hybrid_prediction_contract_is_current(prediction, session=session)
        if not valid: return valid,reason
        if clock > parse_aware_utc(pg('predicted_at')):
            return False,'forecast_clock_invalid'
        ctx=load_prematch_quotes([current_context.canonical_match_id],decision_at=clock,session=session)[current_context.canonical_match_id]
        valid,reason=stored_market_context_is_current(forecast,evaluation_context=ctx,current_context=current_context)
        if not valid: return valid,reason
    elif diagnostics.get('base_model_name') or diagnostics.get('market_features_used') or str(pg('model_name') or '').startswith('Hybrid'): return False,'forecast_market_context_missing'
    side=sg('side')
    if side not in ('a','b'): return False,'signal_side_invalid'
    quote=next((q for q in current_context.quotes if q.odds_snapshot_id==sg('odds_snapshot_id') and q.bookmaker_id==sg('bookmaker_id')),None)
    if quote is None or sg('odds') != getattr(quote,'odds_'+side): return False,'execution_quote_changed'
    if sg('model_prob') != pg('prob_'+side): return False,'signal_probability_changed'
    expected_market = current_context.prob_a if side=='a' else 1-current_context.prob_a
    if sg('market_prob') != expected_market:
        expected_market = _persisted_real_probability(expected_market, column='market_prob', session=session)
        if sg('market_prob') != expected_market: return False,'signal_market_changed'
    return True,None


def hybrid_prediction_contract_is_current(prediction, *, session=None):
    """Check the persisted exact hybrid/base contract, never today's default."""
    pg = prediction.get if isinstance(prediction,Mapping) else lambda k,d=None:getattr(prediction,k,d)
    diagnostics = _saved(pg('diagnostics_json')) or {}
    context = _saved(diagnostics.get('market_context'))
    if not context:
        return False,'forecast_market_context_missing'
    options = {'session':session} if session is not None else {}
    artifacts = query_df(
        'SELECT * FROM model_artifacts WHERE model_name=? AND model_version=?',
        (pg('model_name'),pg('model_version')), **options,
    ).to_dict('records')
    if len(artifacts) != 1:
        return False,'hybrid_contract_missing'
    params = _saved(artifacts[0].get('model_params_json')) or {}
    schema = _saved(artifacts[0].get('feature_schema_json')) or {}
    declared = schema.get('base_model')
    base_name,base_version = params.get('base_model_name'),params.get('base_model_version')
    if not base_name and not base_version and isinstance(declared,str) and '/' in declared:
        base_name,base_version = declared.split('/',1)
    if not base_name or not base_version or (declared is not None and declared != f'{base_name}/{base_version}'):
        return False,'hybrid_base_identity_missing'
    if context.get('model_name') != pg('model_name') or context.get('model_version') != pg('model_version'):
        return False,'forecast_identity_mismatch'
    for key,value in (('base_model_name',base_name),('base_model_version',base_version)):
        if context.get(key) != value or diagnostics.get(key) != value:
            return False,'hybrid_base_identity_mismatch'
    for key in ('alpha','temperature','blending_mode'):
        if key not in params or diagnostics.get(key) != params[key]:
            return False,'hybrid_parameters_mismatch'
    try:
        alpha,temperature=params['alpha'],params['temperature']
        if isinstance(alpha,bool) or isinstance(temperature,bool) or not math.isfinite(alpha) or not 0<=alpha<=1 or not math.isfinite(temperature) or temperature<=0 or params['blending_mode'] not in ('linear','logit_shrinkage'):
            return False,'hybrid_parameters_invalid'
    except (TypeError,ValueError):
        return False,'hybrid_parameters_invalid'
    origin = parse_aware_utc(context.get('decision_at'))
    base_cutoff = parse_aware_utc(diagnostics.get('base_data_cutoff_at'))
    cutoff = parse_aware_utc(pg('data_cutoff_at'))
    issued = parse_aware_utc(pg('predicted_at'))
    if (origin is None or base_cutoff is None or cutoff is None or issued is None
            or not base_cutoff <= origin <= cutoff <= issued):
        return False,'hybrid_base_clock_invalid'
    base_id=diagnostics.get('base_prediction_id')
    if base_id is not None:
        bases=query_df('SELECT * FROM canonical_predictions WHERE id=?',(base_id,),**options).to_dict('records')
        if len(bases)!=1 or bases[0].get('canonical_match_id')!=pg('canonical_match_id'):
            return False,'hybrid_base_lineage_missing'
        valid,reason=prediction_time_is_actionable(bases[0],decision_at=origin,start_at=parse_aware_utc(context.get('start_at')),model_name=base_name,model_version=base_version)
        if not valid: return valid,reason
    return True,None


def _persisted_real_probability(value, *, column, session=None):
    """Match a PostgreSQL REAL scalar's storage representation, not a tolerance.

    The complete source values and double-precision q are already checked in
    the serialized context. SQLite REAL and PostgreSQL DOUBLE stay unchanged.
    """
    from sqlalchemy import REAL, inspect, text
    from betting_app.core.db import get_session
    own_session = session is None
    bound_session = session if session is not None else get_session()
    try:
        bind = bound_session.get_bind()
        if bind.dialect.name != 'postgresql':
            return value
        columns = inspect(bind).get_columns('model_ev_signals')
        native_type = next((item['type'] for item in columns if item['name'] == column), None)
        if isinstance(native_type, REAL):
            # PostgreSQL emits REAL's shortest decimal representation; the
            # driver then parses it as a Python float, not a widened float32.
            return bound_session.execute(text("SELECT CAST(:value AS REAL)"),
                                         {"value": value}).scalar_one()
        return value
    finally:
        if own_session:
            bound_session.close()
