"""Paired loss-only historical assessment; no fitting or outcome-based exclusions."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import xlogy

from src.analysis.c0_betting_development_evaluation import _probability, _value_summary
from src.analysis.c0_input_aligned_study import evaluate_input_aligned

_REFERENCES = ('S', 'S-cal', 'M', 'M-cal', 'M-shape', 'Fixed50', 'H')
_ARMS = {'H-likelihood': ('H-price', 'likelihood'),
         'H-return_mse': ('H-price', 'return_mse'),
         'M-likelihood': ('M-price', 'likelihood'),
         'M-return_mse': ('M-price', 'return_mse')}
_CONTROLS = ('H-likelihood', 'M-return_mse', 'M-likelihood')
_CONFIDENCE = 1 - .05 / 3


def _price(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and value > 1)


def _missing(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _two_prices(row):
    return (_price(row.odds_a) and _price(row.odds_b)
            and all(any(_price(quote.get('odds_' + side)) for quote in row.quotes)
                    for side in ('a', 'b')))


class _MarketEligibleHead:
    """Market admission is independent of native admission or price display."""

    def __init__(self, head):
        self.head = head

    def predict_details(self, frame):
        if not frame.market_eligible.all():
            raise ValueError('market_not_eligible')
        return self.head.predict_details(frame)


def _risk(row, probability):
    weight = .5 * .88**2 * (float(row.odds_a)**2 + float(row.odds_b)**2)
    value = weight * (float(probability) - float(row.y))**2
    if not np.isfinite(weight) or not np.isfinite(value):
        raise ValueError('nonfinite derived return-MSE')
    return float(value)


def _validate(panel, arms, references):
    if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        raise ValueError('arms must contain exactly the four EV-loss arms')
    if not isinstance(references, Mapping) or set(references) != set(_REFERENCES):
        raise ValueError('reference_heads must contain exactly the seven frozen recipes')
    for name, (recipe, objective) in _ARMS.items():
        if arms[name].recipe_id != recipe or arms[name].objective != objective:
            raise ValueError('wrong recipe/objective for arm ' + name)
    for recipe, head in references.items():
        if head.recipe_id != recipe:
            raise ValueError('wrong frozen reference recipe ' + recipe)
    required = {'row_id', 'month', 'p_C0', 'q', 'y', 'native_admitted',
                'financial_eligible', 'odds_a', 'odds_b', 'quotes'}
    if not required.issubset(panel.columns):
        raise ValueError('panel lacks required EV-loss columns: ' + ', '.join(sorted(required-set(panel.columns))))
    if panel.row_id.isna().any() or panel.row_id.duplicated().any():
        raise ValueError('row_id must be unique and nonmissing')
    for column in ('primary_decision', 'native_admitted', 'financial_eligible',
                   'forecast_eligible', 'market_eligible', 'fit_eligible'):
        if column in panel and not panel[column].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError(column + ' must contain explicit booleans')
    for column in ('p_C0', 'q', 'y'):
        if not panel[column].map(lambda v: _missing(v) or (_probability(v) and (column != 'y' or v in (0, 1)))).all():
            raise ValueError('invalid probability/label in ' + column)
    for column in ('odds_a', 'odds_b'):
        if not panel[column].map(lambda v: _missing(v) or _price(v)).all():
            raise ValueError('invalid execution price in ' + column)
    for quotes in panel.quotes:
        if not isinstance(quotes, (list, tuple)) or any(not isinstance(q, Mapping) for q in quotes):
            raise ValueError('quotes must be a sequence of quote mappings')
        for quote in quotes:
            for side in ('a', 'b'):
                price = quote.get('odds_' + side)
                if not _missing(price) and not _price(price):
                    raise ValueError('invalid quote price')
            if not isinstance(quote.get('bookmaker'), str):
                raise ValueError('quote bookmaker must be explicit')
    for row in panel.itertuples():
        if _price(row.odds_a) and _price(row.odds_b):
            with np.errstate(over='ignore', invalid='ignore'):
                weight = .5 * .88**2 * (np.float64(row.odds_a)**2 + np.float64(row.odds_b)**2)
            if not np.isfinite(weight):
                raise ValueError('nonfinite derived return-MSE weight')


def _profile_key(row, name):
    if name in ('source', 'BO'):
        columns = ('source', 'source_label', 'league', 'competition') if name == 'source' else ('best_of', 'bo', 'BO')
        for column in columns:
            value = row.get(column)
            if value is not None and not pd.isna(value):
                return str(value)
        return 'unknown'
    if name == 'confidence':
        value = row.p_C0
        if not _probability(value):
            return 'unknown'
        value = max(value, 1-value)
        return '[.5,.6)' if value < .6 else '[.6,.8)' if value < .8 else '[.8,.9)' if value < .9 else '[.9,1]'
    if name == 'price':
        if not _price(row.odds_a) or not _price(row.odds_b):
            return 'unknown'
        value = max(row.odds_a, row.odds_b)
        return '(1,2]' if value <= 2 else '(2,3]' if value <= 3 else '(3,5]' if value <= 5 else '(5,infinity)'
    if not _probability(row.p_C0) or not _probability(row.q):
        return 'unknown'
    value = abs(row.p_C0-row.q)
    return '[0,.05]' if value <= .05 else '(.05,.1]' if value <= .1 else '(.1,.2]' if value <= .2 else '(.2,1]'


def _return_reports(frame, report, months, samples):
    ledgers = {recipe: {r['row_id']: r for r in report['rows'] if r['recipe'] == recipe}
               for recipe in (*_REFERENCES, 'M-price', 'H-price')}
    scores, profiles, extra = {}, {}, {}
    for recipe, ledger in ledgers.items():
        observations = []
        for _, row in frame.iterrows():
            entry = ledger[row.row_id]
            scoring = raw = None
            if (row.primary_decision and row.forecast_eligible and row.market_eligible
                    and row.financial_eligible and _probability(row.y) and row.y in (0, 1)
                    and _two_prices(row) and _probability(entry['probability'])):
                scoring = _risk(row, entry['probability'])
                raw_probability = entry.get('raw_probability')
                if not _probability(raw_probability):
                    raise ValueError('valid scoring output lacks valid raw probability')
                raw = _risk(row, raw_probability)
                observations.append((row.row_id, row.month, raw, scoring))
            entry.update(return_mse=scoring, raw_return_mse=raw)
        scores[recipe] = {'count': len(observations), 'row_ids': [r[0] for r in observations],
                         'raw': _value_summary([(r[1], r[2]) for r in observations], months, samples),
                         'scoring': _value_summary([(r[1], r[3]) for r in observations], months, samples)}
        extra_ids = [row.row_id for _, row in frame.iterrows() if row.primary_decision
                     and row.forecast_eligible and row.market_eligible and not row.native_admitted
                     and _probability(ledger[row.row_id]['probability'])]
        extra[recipe] = {'count': len(extra_ids), 'row_ids': extra_ids}
        profiles[recipe] = {}
        for dimension in ('source', 'BO', 'confidence', 'price', 'disagreement'):
            groups = {'unknown': []}
            for _, row in frame.iterrows():
                groups.setdefault(_profile_key(row, dimension), []).append(row)
            bins = {}
            for name, rows in sorted(groups.items()):
                selected = [ledger[row.row_id] for row in rows if ledger[row.row_id]['selected']]
                eligible = [row for row in rows if row.financial_eligible]
                bins[name] = {'count': len(rows), 'row_ids': [r.row_id for r in rows],
                    'coverage_count': sum(_probability(ledger[r.row_id]['probability']) for r in rows),
                    'absence_count': sum(not _probability(ledger[r.row_id]['probability']) for r in rows),
                    'selected_count': len(selected), 'financial_eligible_count': len(eligible),
                    'cash_profit': float(sum(r['profit'] for r in selected)),
                    'return_mse': _value_summary([(r.month, ledger[r.row_id]['return_mse']) for r in rows], months, samples),
                    'overstatement': _value_summary([(r['month'], r['overstatement']) for r in selected], months, samples),
                    'ROI': _value_summary([(r['month'], r['profit']) for r in selected], months, samples),
                    'profit_per_opportunity': _value_summary([(r.month, ledger[r.row_id]['profit']) for r in eligible], months, samples)}
            profiles[recipe][dimension] = bins
    report.update(return_scores=scores, profiles=profiles, market_extra_coverage=extra)
    return ledgers


def evaluate_ev_loss(panel, arms, reference_heads, draws=5000, seed=20261006):
    """Score saved heads on one pre-prediction population and shared month draws."""
    _validate(panel, arms, reference_heads)
    frame = panel.reset_index(drop=True).copy()
    for column in ('primary_decision', 'forecast_eligible', 'market_eligible'):
        if column not in frame:
            frame[column] = True
    frame.financial_eligible &= frame.primary_decision
    reports = {}
    for objective in ('likelihood', 'return_mse'):
        heads = {**reference_heads, 'H-price': arms['H-' + objective], 'M-price': arms['M-' + objective]}
        heads = {recipe: (_MarketEligibleHead(head) if recipe not in ('S', 'S-cal') else head)
                 for recipe, head in heads.items()}
        reports[objective] = evaluate_input_aligned(frame, heads, draws=draws, seed=seed)
        for entry in reports[objective]['rows']:
            entry['objective'] = objective
    months = reports['return_mse']['bootstrap']['months']
    samples = np.random.default_rng(seed).integers(0, len(months), (draws, len(months))) if months else np.empty((draws, 0), int)
    ledgers = {objective: _return_reports(frame, report, months, samples) for objective, report in reports.items()}
    common = frame.primary_decision & frame.forecast_eligible & frame.native_admitted & frame.market_eligible & frame.financial_eligible
    for column in ('p_C0', 'q', 'y'):
        common &= frame[column].map(_probability)
    common &= frame.y.isin([0, 1]) & frame.apply(_two_prices, axis=1)
    matched = frame.loc[common]
    candidate = ledgers['return_mse']['H-price']
    contrasts = {}
    for control in _CONTROLS:
        recipe, objective = _ARMS[control]
        baseline = ledgers[objective][recipe]
        risk, raw_risk, ll, brier = [], [], [], []
        for _, row in matched.iterrows():
            p, q = candidate[row.row_id]['probability'], baseline[row.row_id]['probability']
            if not _probability(p) or not _probability(q):
                raise ValueError('head unavailable on declared common risk population: ' + str(row.row_id))
            risk.append((row.month, _risk(row, p)-_risk(row, q)))
            raw_risk.append((row.month, _risk(row, candidate[row.row_id]['raw_probability'])-_risk(row, baseline[row.row_id]['raw_probability'])))
            y = float(row.y)
            ll.append((row.month, float(-xlogy(y, p)-xlogy(1-y, 1-p)+xlogy(y, q)+xlogy(1-y, 1-q))))
            brier.append((row.month, float((p-y)**2-(q-y)**2)))
        summary = _value_summary(risk, months, samples, _CONFIDENCE)
        ll_summary = _value_summary(ll, months, samples, _CONFIDENCE)
        brier_summary = _value_summary(brier, months, samples, _CONFIDENCE)
        contrasts[control] = {**summary, 'candidate': 'H-return_mse', 'control': control,
            'row_ids': matched.row_id.tolist(), 'delta_return_mse': summary['point'],
            'equal_month_delta_return_mse': summary['equal_month_point'],
            'raw_return_mse': _value_summary(raw_risk, months, samples, _CONFIDENCE),
            'LL': ll_summary, 'Brier': brier_summary, 'delta_LL': ll_summary['point'],
            'delta_Brier': brier_summary['point']}
    gates = dict(reports['return_mse']['gates'])
    for name, contrast in contrasts.items():
        gates['return_mse_vs_' + name] = contrast['interval'] is not None and contrast['interval'][1] < 0
    for name in ('LL', 'Brier'):
        point = contrasts['H-likelihood']['delta_' + name]
        gates['nonworse_' + name + '_vs_H-likelihood'] = point is not None and point <= 0
    reasons = [name for name, passed in gates.items() if not passed]
    return {'schema': 'c0_ev_loss_assessment_v1', 'arms': reports,
        'rows': [row for report in reports.values() for row in report['rows']],
        'risk_contrasts': contrasts, 'common_risk_row_ids': matched.row_id.tolist(),
        'gates': gates, 'reasons': reasons, 'primary_candidate': 'H-return_mse',
        'decision': 'DEVELOPMENT_CANDIDATE' if not reasons else 'NOT_ADOPTED',
        'evidence_class': 'exposed_historical_development',
        'bootstrap': {'draws': int(draws), 'seed': int(seed), 'months': months,
            'confidence_level': _CONFIDENCE, 'adjustment': 'Bonferroni', 'comparators': 3,
            'paired_block_basis': 'full_declared_preselection_month_universe',
            'undefined_policy': 'any_undefined_draw_nulls_interval_no_finite_subset'},
        'limits': reports['return_mse']['limits']}
