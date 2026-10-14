#!/usr/bin/env python3
"""Lock the academic protocol and audit the complete offline readiness inventory.

This investigation does not forecast, fit, scrape, or certify a future sample.
Outputs are exclusive-create and all inspected input bytes are hashed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import variance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_tournament_forecasts import (
    LEDGER_COHORT, _ledger_average, _ledger_identity, _ledger_safe,
)

DEFAULT_READINESS = ROOT / 'data/artifacts/tournament-evaluation-20260909/readiness/readiness_manifest.json'
DEFAULT_SCORES = ROOT / 'data/artifacts/tournament-daily-evaluation-20260909/lck-daily-frozen-available-control-scores.json'
DEFAULT_OUTPUT = ROOT / 'data/artifacts/tournament-academic-validation-20260909/academic_cohort'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_date(value):
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def protocol(boundary, input_hashes):
    return {
        'version': 'academic_protocol-20260909-v1',
        'status': 'locked_for_future_confirmation_not_a_completed_validation',
        'sport': 'League of Legends', 'timezone': 'UTC',
        'input_hashes': input_hashes,
        'scope': {
            'inventory': 'all 234 local phases; exclude with reasons, never successful-subset selection',
            'primary_estimand': 'edition-balanced paired Bernoulli qualification LogLoss EXP081 frozen minus identical-state fair_series, at phase-start post_draw origins, among unresolved canonical destination/team targets',
            'primary_scope': 'marginal qualification; not a claim of the full tournament joint distribution',
            'secondary_axes': ['whole_edition_start', 'next_round', 'remaining_tournament', 'format_stage_risk'],
            'scope_field': 'forecast_scope; phase_start never relabeled whole_edition_start',
            'whole_distribution_claim': 'requires every structurally applicable target family below, coherent joint outputs, empirical held-out calibration and proper-score gates; marginal success alone is insufficient',
        },
        'targets': {
            'qualification_marginal': {'unit': 'destination/team Bernoulli', 'scores': ['log_loss', 'binary_brier']},
            'championship': {'unit': 'one categorical champion', 'scores': ['categorical_log_loss', 'multiclass_brier']},
            'finishing_positions': {'unit': 'official ordered tied-place bands per team', 'scores': ['normalized_ranked_probability_score', 'categorical_log_loss', 'multiclass_brier'], 'ties': 'retain official tied bands; never order tied teams'},
            'finalist_and_node_reach': {'unit': 'team participation in actual final or predeclared named node', 'scores': ['binary_log_loss', 'binary_brier'], 'not': 'champion top-k ranking'},
            'matchup': {'unit': 'unordered pair occupying a named node; absent-node is an explicit outcome where applicable', 'scores': ['categorical_log_loss', 'multiclass_brier']},
            'joint_qualifier_set': {'unit': 'canonical complete set per destination', 'scores': ['categorical_log_loss', 'multiclass_brier'], 'not': 'product of dependent team marginals'},
            'joint_finalist_pair': {'unit': 'canonical actual finalist pair', 'scores': ['categorical_log_loss', 'multiclass_brier']},
            'next_round_series': {'unit': 'legal known matchup or integrated unknown-draw round outcome', 'scores': ['log_loss', 'binary_brier']},
            'joint_paths': {'unit': 'predeclared upset-count, repeat-encounter and lower-bracket champion-path categories', 'scores': ['categorical_log_loss', 'multiclass_brier']},
        },
        'target_gate': {
            'structurally_not_applicable': 'source-backed format rule and reason required (e.g. qualification-only phase has no champion)',
            'unsupported_or_missing': 'block corresponding claim; whole-tournament trust remains inconclusive if any applicable target/phase link/joint distribution is missing',
            'probability_integrity': ['legal support', 'normalization', 'marginal_joint_consistency', 'slot_mass_conservation', 'structural_zeros_distinct_from_zero_hits'],
            'no_pooled_score': 'never average binary, multiclass, RPS, series and joint proper scores into one metric',
        },
        'forecast_modes': {
            'required_field': 'forecast_mode', 'default': None,
            'values': ['pre_draw', 'post_draw', 'daily_rollforward', 'post_round'],
            'pre_draw': 'integrate only genuine declared unknown draw; deterministic seed graph is not pre_draw',
            'post_draw': 'source-available published pairing or deterministic_spec before target start',
            'daily_rollforward': 'one UTC-midnight origin per active day; observation_day_end=cutoff date minus one day; no same-day facts or invented timestamps',
            'post_round': 'all predeclared parallel block results completed and available, before next target start',
        },
        'strata': {
            'never_pool': ['forecast_mode', 'forecast_scope', 'frozen_vs_refreshed', 'retrospective_vs_prospective', 'proxy_vs_announced_roster', 'prior_day_vs_timestamp_certified', 'target_family'],
            'frozen': 'edition-start weights/calibrator and pair matrix fixed; update only known tournament facts',
            'refreshed': 'same edition-start weights/calibrator; update real observed ratings/W20/rest/roster using completed available history; missing transitions exclude, no frozen fallback',
            'proxy_roster': 'prior-observed five-role roster, explicitly retrospective research only',
            'announced_roster': 'source hash and announcement available_at at or before origin',
            'tracks': ['captured_prospective', 'retrospective_reconstruction', 'synthetic_mechanics_only'],
        },
        'controls': {
            'fair_series': 'primary topology control p(series)=0.5; identical entrants, state, draw/choice and score/tiebreak policy',
            'operational': 'exact currently routed operational artifact/recipe hash locked before admission; eligible chronological fold and feature contract required',
            'EXP039': 'exact frozen thesis artifact only where training/availability predates origin; newly trained analogous weights separately named and never called frozen EXP039',
            'EXP081': 'primary candidate; chronological fit and later separate calibration both strictly before edition first origin; weights fixed entire edition',
            'Linear79': 'secondary named recipe control when eligible',
            'flat': 'descriptive non-topological marginals only; not a joint/path competitor',
            'matching': 'exact edition/phase/origin/target/information-stratum intersection per contrast; retain per-model coverage and union exclusions; no default-model substitution',
            'model_identity_gate': 'immutable artifact hashes and input provenance mandatory before future admission; protocol lock is not a claim those artifacts were supplied',
        },
        'exclusions': ['unverified_edition_boundary_or_stage_transition', 'missing_canonical_entrants_or_seeds', 'missing_rule_draw_choice_provenance', 'missing_origin_roster_or_feature_parity', 'incomplete_history_transition', 'model_fit_or_calibration_after_origin', 'missing_model_identity_or_prior_fold', 'missing_availability_for_strict_mode', 'invalid_or_missing_outcome_link', 'unsupported_applicable_target', 'already_resolved_target', 'MC_unresolved_log_score', 'cancelled_or_uncompleted_for_scoring'],
        'weighting': {
            'primary': 'equal teams within destination; equal destinations within phase-start origin; equal origins within edition; equal editions, separately per stratum',
            'daily': 'one scheduled midnight origin per day, including no-match days; equal targets within destination, destinations within origin, daily origins within edition; equal editions, never treat days as independent',
            'horizon': 'separate known-at-origin scheduled horizon bins [0,1),[1,3),[3,7),[7,infinity) days; unknown schedules are an explicit unknown bin, not realized-outcome dates',
            'horizon_sensitivity': 'equal nonempty horizon-bin weight within edition vs equal-origin primary; report missing-bin coverage; never merge modes',
            'phase_balance_sensitivity': 'equal phase weight within edition; do not allow long leagues to dominate',
            'resolved_mask': 'remove already determined targets identically for all methods; report counts',
        },
        'inference': {
            'unit': 'paired edition score delta, retaining all correlated phases/teams/checkpoints',
            'blocks': 'group intact editions by verified edition start month; chronological moving calendar blocks',
            'bootstrap_resamples': 5000, 'seed': 20260909, 'confidence': 0.95,
            'block_lengths_months': [1, 2, 3],
            'primary_block_length_months': 1,
            'method': 'resample chronological calendar-month blocks with replacement including empty months; carry whole editions and recompute equal-edition delta; inspect overlapping/calendar-spanning league sensitivity at lengths 2 and 3',
            'sparse': 'fewer than two independent nonempty blocks: variance/CI/power not estimable; two blocks is algebraic minimum, never an adequacy certificate',
            'superiority': 'paired LogLoss upper CI below zero; mandatory Brier/calibration corroboration and disclosed harm/sensitivity; no ranking from entropy',
            'multiplicity': 'one declared marginal primary test at two-sided alpha .05; Holm familywise .05 across all secondary formal target x control x mode x scope x information-stratum contrasts registered before label access; unregistered slices descriptive; report entire tested family',
            'controls_family': ['EXP081_minus_fair', 'EXP081_minus_operational', 'EXP081_minus_EXP039', 'refreshed_minus_frozen'],
        },
        'calibration': {
            'fit_data': 'earlier development only; never refit on confirmation outputs',
            'bins': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
            'tournament_margin': .05,
            'gate': 'simultaneous 95% edition/month-block max-deviation bands for bin observed-minus-predicted error fully inside [-.05,.05] over all claimed target/stratum/bin cells; unsupported, single-block or degenerate cells are inconclusive, never pass',
            'series_ECE_max': .030, 'series_slope_interval': [.85, 1.15],
            'series_gate': 'upper 95% ECE bound <=.030 and slope 95% interval within [.85,1.15]; report intercept, Brier decomposition and separation; existing project promotion gates remain additional',
            'report': ['bin_forecasts', 'independent_editions', 'month_blocks', 'high_confidence_0.8_to_1', 'longshots', 'sharpness', 'unsupported_ranges'],
        },
        'numerical_precision_and_sensitivity': {
            'initial_simulations': 20000, 'maximum_simulations': 200000,
            'schedule': 'double samples to cap independent of observed labels',
            'marginal_Wilson_half_width_max': .005,
            'oracle': 'exact enumeration/dynamic programming on supported small states; use paired compatible streams and independent seed replication',
            'zero_hits': 'retain raw hits; exact or predeclared rare-event estimate required for decisive LogLoss; unresolved is explicit, true impossible observed outcome is infinite, no epsilon rescue',
            'joint_precision': 'marginal half-width does not certify joint LogLoss; inspect supported observed joint-category probability precision separately without label-driven sampling',
            'assumptions': ['uniform_vs_source_supported_draw_policy', 'legal_choice_policy', 'score_tiebreak_policy', 'proxy_vs_announced_roster', 'frozen_vs_refreshed', 'independent_series_vs_one_coherent_latent_state_per_rollout'],
            'sensitivity_gate': 'report valid variants and unavailable prerequisites; do not fabricate latent state or future game statistics; material sign reversal or unresolved numerical error prevents robust claim',
        },
        'power': {
            'effect_grid_absolute_LogLoss': [.01, .02, .05], 'target_power': .8,
            'method': 'estimate sample variance of edition-balanced paired score deltas from development editions; estimate dependence/design effect from start-month blocks and block-length sensitivity; approximate required independent editions ceil((z_(1-alpha/2)+z_.8)^2*s_delta^2/delta^2), then inflate for block dependence and multiplicity; confirm with centered empirical block-bootstrap power simulation',
            'prerequisites': ['at_least_two_finite_matched_independent_development_editions', 'verified_edition_boundaries', 'multiple_calendar_blocks_for_dependence', 'locked_effect_size_and_comparison_family', 'finite_joint_or_marginal_scores_for_the_specific_estimand'],
            'no_certificate': 'phase count, team count, MC samples and one pilot cannot estimate edition-block variance; no arbitrary required N or power guarantee',
            'stopping': 'no outcome-based stopping; size/horizon must be fixed from independent development variance before confirmation labels are accessed; until then accrue sealed eligible forecasts only and make no powered claim',
        },
        'untouched_reservation': {
            'strictly_after_UTC': boundary,
            'status': 'rule_reserved_no_future_forecasts_or_labels_collected_by_this_audit',
            'inclusion': 'all newly starting whole editions in the seven inventory families after boundary and actual protocol/model lock, with no phase/result previously inspected; first origin before first match, canonical graph/rules/entrants and timestamp-certified features/announcements; register every candidate and reason-coded exclusion before outcomes',
            'families': ['worlds', 'msi', 'lcs', 'lck', 'lpl', 'lec', 'eu_masters'],
            'exclude': 'every current inventory edition including scheduled phases is development/inspected; never recycle current pilot, related phases of an underway edition, or previously accessed labels as untouched',
            'capture': 'persist hash-linked protocol/model/source/state/forecast ledger at real predicted_at before targets; labels separately sealed until fixed analysis stop; no retrospective backdating',
            'boundary_update': 'if any outcomes beyond boundary were already inspected before operational enrollment, advance boundary with append-only amendment before admission; never reuse inspected editions',
            'promotion': False,
        },
    }


def pilot_blocks(scores):
    """Keep every eligible paired target; unavailable scores block the full estimand."""
    cohorts = collections.defaultdict(lambda: collections.defaultdict(dict))
    edition_starts = {}
    for row in scores['target_scores']:
        group = tuple(row[field] for field in LEDGER_COHORT)
        key = _ledger_identity(row)
        indexed = cohorts[group][row['model']]
        if key in indexed:
            raise ValueError(f'duplicate pilot target/model: {(key, row["model"])}')
        indexed[key] = row
        edition, started = row['tournament_id'], row['edition_start_at']
        if edition in edition_starts and edition_starts[edition] != started:
            raise ValueError(f'{edition}: inconsistent edition start')
        edition_starts[edition] = started

    def score_value(row):
        if row is None:
            return None
        status = row['metric_status'].get('log_loss')
        if status == 'infinite':
            return math.inf
        value = row['scores'].get('log_loss')
        return value if status == 'scored' and isinstance(value, (int, float)) and math.isfinite(value) else None

    result = []
    for group, models in sorted(cohorts.items()):
        for model in sorted(set(models) & {'exp081', 'exp081:frozen'}):
            targets, references = models[model], models.get('fair_series', {})
            origins = collections.defaultdict(list)
            model_statuses, reference_statuses = collections.Counter(), collections.Counter()
            eligible = matched = finite = resolved = pending = 0
            for key in sorted(targets.keys() | references.keys()):
                row, ref = targets.get(key), references.get(key)
                present = row if row is not None else ref
                if row is not None and ref is not None and (
                        row['resolved'] != ref['resolved'] or row['observed'] != ref['observed']):
                    raise ValueError(f'{key}: pilot reference outcome/resolution mismatch')
                if present['resolved']:
                    resolved += 1
                    continue
                if present['observed'] is None:
                    pending += 1
                    continue
                eligible += 1
                matched += row is not None and ref is not None
                model_statuses[row['metric_status'].get('log_loss', 'unavailable_metric')
                               if row is not None else 'missing_model'] += 1
                reference_statuses[ref['metric_status'].get('log_loss', 'unavailable_metric')
                                   if ref is not None else 'missing_reference'] += 1
                left, right = score_value(row), score_value(ref)
                is_finite = left is not None and right is not None and math.isfinite(left) and math.isfinite(right)
                finite += is_finite
                origins[key[:3]].append((left, right, left-right if is_finite else None))
            editions = collections.defaultdict(list)
            for (edition, _, _), values in origins.items():
                editions[edition].append(tuple(_ledger_average(list(column)) for column in zip(*values)))
            edition_values = {
                edition: tuple(_ledger_average(list(column)) for column in zip(*values))
                for edition, values in sorted(editions.items())}
            deltas = [values[2] for values in edition_values.values()]
            months = {edition: edition_starts[edition][:7] if edition_starts[edition] is not None else None
                      for edition in edition_values}
            n_months = len({month for month in months.values() if month is not None})
            unknown_months = sum(month is None for month in months.values())
            complete = eligible > 0 and finite == eligible
            estimable = complete and len(deltas) >= 2 and n_months >= 2 and unknown_months == 0
            delta_status = (
                'unavailable_no_observed_targets' if not eligible
                else 'unavailable_reference' if reference_statuses['missing_reference']
                else 'unavailable_model' if model_statuses['missing_model']
                else 'unavailable_unscorable' if any(
                    status not in ('scored', 'infinite')
                    for status in (*model_statuses, *reference_statuses))
                else 'unavailable_infinite' if model_statuses['infinite'] or reference_statuses['infinite']
                else 'scored' if complete else 'unavailable_unscorable')
            result.append(dict(zip(LEDGER_COHORT, group)) | {
                'model': model, 'reference': 'fair_series',
                'forecast_targets': len(targets.keys() | references.keys()),
                'eligible_targets': eligible, 'matched_targets': matched,
                'paired_finite_targets': finite, 'resolved_targets': resolved, 'pending_targets': pending,
                'model_score_status_counts': dict(sorted(model_statuses.items())),
                'reference_score_status_counts': dict(sorted(reference_statuses.items())),
                'origins': len(origins), 'candidate_edition_blocks': len(edition_values),
                'start_month_blocks': n_months, 'unknown_edition_month_blocks': unknown_months,
                'edition_balanced_log_loss_delta': _ledger_average(deltas), 'delta_status': delta_status,
                'model_mean_log_loss': _ledger_average([values[0] for values in edition_values.values()]),
                'reference_mean_log_loss': _ledger_average([values[1] for values in edition_values.values()]),
                'edition_values': [{
                    'tournament_id': edition, 'edition_start_month': months[edition],
                    'edition_start_at': edition_starts[edition],
                    'edition_start_status': ('unverified_whole_edition_boundary'
                                             if months[edition] is None else 'provided_timestamp'),
                    'model_mean': values[0], 'reference_mean': values[1], 'delta': values[2]}
                    for edition, values in edition_values.items()],
                'edition_delta_sample_variance': variance(deltas) if estimable else None,
                'power_status': 'variance_available_requires_verified_block_dependence' if estimable
                                else 'not_estimable_unscorable_or_infinite_cohort' if eligible and not complete
                                else 'not_estimable_insufficient_independent_edition_month_blocks',
                'required_N': None, 'scope': 'inspected_development_not_verified_independent_editions'})
    return _ledger_safe(result)


def run(readiness_path, scores_path, output_dir):
    inventory = json.loads(readiness_path.read_text())
    scores = json.loads(scores_path.read_text())
    phases = inventory['phases']
    if len(phases) != 234 or len({p['phase_id'] for p in phases}) != len(phases):
        raise ValueError('expected exactly 234 unique inventory phase IDs')
    hashes = {'readiness_manifest': digest(readiness_path), 'development_scores': digest(scores_path), 'investigation_script': digest(Path(__file__))}
    source_integrity = []
    for item in inventory['inputs']:
        path = Path(item['path'])
        actual = digest(path) if path.is_file() else None
        source_integrity.append({'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'recorded_sha256': item.get('sha256'), 'current_sha256': actual, 'matches': actual == item.get('sha256')})
    source_csv = Path(inventory['source_tables']['series_results.csv']['path'])
    hashes['current_series_results'] = digest(source_csv)
    with source_csv.open(newline='') as stream:
        series = list(csv.DictReader(stream))
    completed_dates = [iso_date(x['source_date']) for x in series if x['status'] == 'completed']
    valid_completed = [x for x in completed_dates if x is not None]
    max_completed = max(valid_completed)
    retrievals = [dt.datetime.fromisoformat(x['retrieved_at'].replace('Z', '+00:00')) for x in series if x['retrieved_at']]
    latest_retrieval = max(retrievals)
    # Entire current session date is inspected; reserve beyond its UTC end too.
    latest_inspected_day = max(dt.date(2026, 9, 9), max_completed, latest_retrieval.date())
    boundary = (latest_inspected_day + dt.timedelta(days=1)).isoformat() + 'T00:00:00+00:00'
    frozen = protocol(boundary, hashes)
    phase_rows = []
    for p in phases:
        phase_rows.append({'phase_id': p['phase_id'], 'candidate_edition_id': p['tournament_id'],
                           'family': p['source_event']['family'], 'year': p['source_event']['year'],
                           'status': p['event_status'], 'start_date': p['source_event'].get('start_date'),
                           'end_date': p['source_event'].get('end_date'), 'profiles': p['profile_ids'],
                           'historically_certified': p['historical_certification']['eligible'],
                           'reconstruction_eligible': p['reconstruction']['eligible'],
                           'axes': p['axes'], 'source_gaps': p['source_gaps'],
                           'future_confirmation_eligible': False, 'reservation_exclusion': 'already_inspected_inventory'})
    grouped = collections.defaultdict(list)
    for row in phase_rows:
        grouped[row['candidate_edition_id']].append(row)
    candidates = []
    for edition in inventory['editions']:
        members = grouped[edition['tournament_id']]
        starts = [iso_date(x['start_date']) for x in members]
        valid = [x for x in starts if x is not None]
        candidates.append({'candidate_edition_id': edition['tournament_id'], 'title': edition['title'],
                           'family': edition['family'], 'year': edition['year'],
                           'phase_ids': sorted(x['phase_id'] for x in members),
                           'candidate_start_month': min(valid).isoformat()[:7] if valid else None,
                           'all_start_dates_valid': len(valid) == len(starts),
                           'whole_edition_verified': edition['whole_edition_verified'],
                           'reasons': edition['reasons']})
    if set(grouped) != {x['candidate_edition_id'] for x in candidates}:
        raise ValueError('edition/phase membership does not reconcile')
    axes = {}
    for axis in sorted({a for p in phases for a in p['axes']}):
        records = [p['axes'][axis] for p in phases]
        axes[axis] = {'eligible_phases': sum(x['eligible'] for x in records), 'excluded_phases': sum(not x['eligible'] for x in records),
                      'reason_counts': dict(sorted(collections.Counter(reason for x in records for reason in set(x['reasons'])).items()))}
    months = collections.Counter(x['candidate_start_month'] for x in candidates if x['candidate_start_month'])
    exclusions = [{'phase_id': p['phase_id'], 'axis': axis, **record} for p in phases for axis, record in p['axes'].items() if not record['eligible']]
    stored_exclusions = inventory['exclusions']
    if {(x['phase_id'], x['axis']) for x in exclusions} != {(x['phase_id'], x['axis']) for x in stored_exclusions}:
        raise ValueError('stored and recomputed exclusion keys differ')
    audit = {
        'version': 'academic_cohort-20260909-v1', 'qualification': 'offline_inventory_and_inspected_development_only',
        'protocol_sha256': hashlib.sha256(canonical(frozen)).hexdigest(), 'protocol_digest_serialization': 'UTF-8 sorted keys compact separators ensure_ascii=false no trailing newline',
        'inputs': {'readiness': str(readiness_path), 'development_scores': str(scores_path), 'hashes': hashes},
        'source_integrity': source_integrity,
        'coverage': {'phases': len(phases), 'status_counts': dict(collections.Counter(p['event_status'] for p in phases)),
                     'family_phase_counts': dict(collections.Counter(p['source_event']['family'] for p in phases)),
                     'historically_certified_phases': sum(p['historical_certification']['eligible'] for p in phases),
                     'reconstruction_eligible_phases': sum(p['reconstruction']['eligible'] for p in phases),
                     'eligible_origins': len(inventory['eligible_origins']), 'phase_axis_exclusions': len(exclusions),
                     'candidate_title_clusters': len(candidates), 'verified_whole_editions': sum(x['whole_edition_verified'] for x in candidates),
                     'candidate_start_months': len(months), 'candidate_month_counts': dict(sorted(months.items())),
                     'candidate_independence_warning': 'title clusters are not verified independent editions; different titles may be stages of one edition; month counts are not certified inferential blocks',
                     'modern_2024_onward_phases': sum(int(p['source_event']['year']) >= 2024 for p in phases),
                     'series_rows': len(series), 'completed_rows': len(completed_dates), 'completed_missing_or_invalid_source_dates': sum(x is None for x in completed_dates),
                     'source_joined_rows': inventory['source_tables']['series_results.csv']['joined_rows'], 'source_unjoined_rows': inventory['source_tables']['series_results.csv']['unjoined_rows']},
        'axes': axes, 'phases': phase_rows, 'candidate_editions': candidates, 'exclusions': exclusions,
        'inspected_time': {'latest_valid_completed_source_date': max_completed.isoformat(), 'latest_source_retrieval': latest_retrieval.isoformat(),
                           'session_inspection_date': '2026-09-09', 'reservation_strictly_after': boundary,
                           'source_date_quality_warning': 'invalid/day-only source dates do not certify intraday availability; retrieval is inspection time, not original announcement'},
        'power': {'status': 'not_estimable', 'required_N': None, 'reason': 'zero verified eligible whole editions; inspected pilot provides one edition/month only per contrast',
                  'method': frozen['power'], 'development_contrasts': pilot_blocks(scores)},
        'untouched': {'eligible_collected_editions': 0, 'forecasts_collected_by_this_audit': 0, 'results_collected_by_this_audit': 0,
                      'reservation': frozen['untouched_reservation']},
        'verdict': {'whole_tournament_trust': 'inconclusive_missing_eligible_editions_targets_controls_and_untouched_confirmation', 'production_qualified': False},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'academic_protocol.json').write_bytes(canonical(frozen))
    (output_dir / 'academic_cohort.json').write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + '\n')
    summary = {k: audit[k] for k in ('protocol_sha256', 'coverage', 'inspected_time', 'verdict')}
    summary['power_status'] = audit['power']['status']
    summary['development_contrasts'] = audit['power']['development_contrasts']
    summary['output_dir'] = str(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', type=Path, default=DEFAULT_READINESS)
    parser.add_argument('--development-scores', type=Path, default=DEFAULT_SCORES)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.readiness.resolve(), args.development_scores.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
