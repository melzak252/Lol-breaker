#!/usr/bin/env python3
"""Offline real-series EXP081 reconstruction; no tournament simulation or training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.export_prospective_features import iter_matches
from scripts.prospective_sports_features import (
    ChronologicalFeatureState, FEATURE_CONTRACT, FEATURE_VERSION, _prepare_history,
    _series_team_ids,
)
from src.models.tournament_prediction import (
    BUNDLE_VERSION, file_digest, json_digest, predict_exp081_pairs,
)

ART = ROOT / 'data/artifacts'
DEFAULT_OUTPUT = ART / 'real-series-model-comparison-20260909/exp081'


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = ART / 'golgg-database-recovery-20260909/matches.json'
    old_source = ART / 'golgg-header-repair-20260908-validated/matches.json'
    model_path = ART / 'tournament-evaluation-20260909/exp081-chronological-v2-20260606/model.json'
    matrix_path = ART / 'tournament-evaluation-20260909/pairs-phase_start.json'
    readiness_path = ART / 'tournament-evaluation-20260909/readiness/readiness_manifest.json'
    series_path = ART / 'leaguepedia-tournament-rules-20260908/series_results.csv'
    mapping_path = ART / 'leaguepedia-tournament-rules-20260908/golgg_mapping.csv'
    pilot_path = ART / 'tournament-evaluation-20260909/lck-pilot-inputs/outcomes.json'
    recovery_path = ART / 'golgg-database-recovery-20260909/manifest.json'
    inputs = {str(p.relative_to(ROOT)): file_digest(p) for p in (
        source, old_source, model_path, matrix_path, readiness_path,
        series_path, mapping_path, pilot_path, recovery_path,
    )}
    artifact, frozen = load(model_path), load(matrix_path)
    model_hash = inputs[str(model_path.relative_to(ROOT))]
    model_id = artifact['model_version']
    eligible_day = artifact['eligible_from'][:10]
    ready = load(readiness_path)
    source_rows = pd.read_csv(series_path, dtype=str).fillna('').to_dict('records')
    mappings = {r['wiki_series_id']: r for r in pd.read_csv(mapping_path, dtype=str).fillna('').to_dict('records')}
    phases = {}
    for phase in ready['phases']:
        for number in phase['series_evidence']['source_row_numbers']:
            phases.setdefault(number, []).append(phase['phase_id'])
    pilot = load(pilot_path)
    pilot_sources = {r['wiki_series_id']: r for r in pilot['source_series']}
    target_ids = {r['golgg_match_id'] for r in mappings.values()
                  if r['status'] == 'matched' and r['golgg_date'] >= eligible_day}
    target_ids.add('79275')
    # Only identity/date fields enter fixture construction. Outcome reconciliation
    # is deliberately deferred until predictions and their digest are on disk.
    target_records = {str(r['match_id']): r for r in iter_matches(source)
                      if str(r['match_id']) in target_ids}
    reference_names = {str(r['match_id']): (r.get('sname_t1'), r.get('sname_t2'))
                       for r in iter_matches(old_source) if str(r['match_id']) in target_ids}
    fixtures, coverage = [], []
    for number, raw in enumerate(source_rows, 2):
        wid = raw['wiki_series_id']
        mapped = mappings.get(wid, {})
        row = {
            'series_id': wid, 'wiki_series_id': wid, 'phase_id': ';'.join(phases.get(number, [])),
            'tournament': raw['event_title'], 'phase': raw['stage'],
            'source_date': raw['source_date'], 'wiki_source_date': raw['source_date'],
            'scheduled_at_utc': raw['scheduled_at_utc'] or None,
            'forecast_cutoff': None, 'team_a': None, 'team_b': None,
            'best_of': int(raw['best_of']) if raw['best_of'].isdigit() else None,
            'golgg_match_id': mapped.get('golgg_match_id') or None,
            'model_identifier': model_id, 'model_sha256': model_hash,
            'probability_a': None, 'y_true': None, 'information_policy': 'refreshed_prior_day',
            'eligibility': False, 'status': 'excluded', 'reason': None,
            'availability_certified': False,
        }
        reason = None
        if raw['status'] != 'completed':
            reason = 'source_series_not_completed'
        elif not raw['source_date']:
            reason = 'missing_source_calendar_date'
        elif (mapped.get('golgg_date') or raw['scheduled_at_utc'][:10] or raw['source_date']) < eligible_day:
            reason = 'no_compatible_chronological_fold_before_series; June6_fold_contains_later_training_or_calibration'
        elif wid not in pilot_sources and mapped.get('status') != 'matched':
            reason = 'no_unambiguous_archived_golgg_identity_mapping:' + mapped.get('status', 'absent')
        else:
            mid = '79275' if wid in pilot_sources and pilot_sources[wid]['node_id'] == 'upper' else mapped.get('golgg_match_id')
            record = target_records.get(mid)
            if record is None:
                reason = 'mapped_series_absent_from_recovered_source'
            else:
                t1, t2 = _series_team_ids(record)
                row.update(golgg_match_id=mid, source_date=record['date'])
                if wid in pilot_sources:
                    canonical = pilot_sources[wid]['canonical_mapping']
                    wiki_a, wiki_b = canonical[raw['team1']], canonical[raw['team2']]
                else:
                    original_names = reference_names.get(mid)
                    recovered_names = (record.get('sname_t1'), record.get('sname_t2'))
                    reversed_names = False
                    if original_names != recovered_names:
                        if original_names == recovered_names[::-1]:
                            reversed_names = True
                        else:
                            reason = 'recovered_header_identity_cannot_be_aligned_to_archived_mapping'
                    swapped = (mapped.get('side_swapped') == 'True') != reversed_names
                    wiki_a, wiki_b = (t2, t1) if swapped else (t1, t2)
                if set((wiki_a, wiki_b)) != set((t1, t2)):
                    reason = 'recovered_team_ids_disagree_with_archived_fixture'
                if reason is None:
                    row.update(team_a=min(t1, t2), team_b=max(t1, t2),
                               forecast_cutoff=record['date'], cutoff_precision='calendar_day',
                               wiki_team_a_id=wiki_a, wiki_team_b_id=wiki_b)
                    fixtures.append(row)
        row['reason'] = reason
        coverage.append(row)
    save(out / 'fixtures_without_outcomes.json', fixtures)
    frozen_probs, frozen_provenance = predict_exp081_pairs(
        artifact, frozen, teams=sorted(frozen['rosters']), best_ofs=[5],
        cutoff=frozen['cutoff'], mode='chronological',
    )
    frozen_rows = []
    for row in fixtures:
        if row['series_id'] not in pilot_sources:
            continue
        item = dict(row)
        item.update(information_policy='frozen_phase_start', forecast_cutoff=frozen['cutoff'],
                    cutoff_precision='timestamp', probability_a=frozen_probs[item['team_a'], item['team_b'], 5],
                    status='predicted_retrospective_development_diagnostic', reason='frozen_pre_event_matrix; retrospective_model_and_incomplete_source',
                    feature_history_max_at=frozen_provenance['feature_history_max_at'])
        frozen_rows.append(item)
    save(out / 'frozen_predictions_before_outcomes.json', frozen_rows)
    save(out / 'frozen_provenance.json', frozen_provenance)
    end_day = date.fromisoformat(max(r['source_date'] for r in fixtures)) + timedelta(days=1)
    compact, history_audit = _prepare_history(iter_matches(source), end_day)
    state = ChronologicalFeatureState()
    position = 0
    features, native_sigmas, failed = [], [], []
    for row in sorted(fixtures, key=lambda r: (r['source_date'], r['series_id'])):
        cutoff = date.fromisoformat(row['source_date'])
        while position < len(compact) and compact[position].games[-1].day < cutoff:
            state.consume(compact[position])
            position += 1
        teams = [row['team_a'], row['team_b']]
        try:
            rosters = {team: state.latest_rosters[team][1] for team in teams}
            frame = state.pairs(rosters, cutoff, [row['best_of']])
            native = state.manager.predict_match(*teams, rosters[teams[0]], rosters[teams[1]])
            # The midnight is solely the exporter's technical calendar boundary,
            # never an invented source match/publication timestamp.
            boundary = cutoff.isoformat() + 'T00:00:00+00:00'
            bundle = {
                'version': BUNDLE_VERSION, 'cutoff': boundary,
                'feature_version': FEATURE_VERSION, 'feature_contract': FEATURE_CONTRACT,
                'feature_contract_sha256': json_digest(FEATURE_CONTRACT),
                'rows': frame.to_dict('records'), 'audit': state.audit(cutoff),
                'rosters': rosters, 'roster_policy': 'previous_observed_roster_proxy',
                'previous_roster_observations': {t: {'source_day': state.latest_rosters[t][0][0].isoformat(),
                    'series_order': list(state.latest_rosters[t][0][1]), 'game_index': state.latest_rosters[t][0][2]} for t in teams},
                'input_hashes': {'recovered_history': inputs[str(source.relative_to(ROOT))]},
                'prepared_at': datetime.now(timezone.utc).isoformat(),
            }
            probabilities, provenance = predict_exp081_pairs(
                artifact, bundle, teams=teams, best_ofs=[row['best_of']],
                cutoff=boundary, mode='chronological',
            )
            row.update(probability_a=probabilities[*teams, row['best_of']],
                       status='predicted_retrospective_development_diagnostic',
                       reason='strict_prior_day_features; retrospective_model; recovered_history_missing309_global_series',
                       feature_history_max_at=provenance['feature_history_max_at'],
                       feature_bundle_sha256=json_digest(bundle))
            save(out / 'pair_bundles' / (row['series_id'] + '.json'), bundle)
            features.append({**frame.iloc[0].to_dict(), **{k: row[k] for k in (
                'series_id', 'phase_id', 'golgg_match_id', 'source_date', 'forecast_cutoff', 'information_policy')},
                'roster_policy': bundle['roster_policy']})
            native_sigmas.append({
                **{key: row[key] for key in ('series_id', 'team_a', 'team_b', 'forecast_cutoff')},
                'player_tm_sigma_avg1': native['player_tm_sigma_avg1'],
                'player_tm_sigma_avg2': native['player_tm_sigma_avg2'],
            })
        except (KeyError, ValueError) as exc:
            row.update(status='excluded', reason='strict_prior_day_export_failed:' + str(exc))
            failed.append({'series_id': row['series_id'], 'reason': row['reason']})
    pd.DataFrame(features).to_csv(out / 'refreshed_features.csv', index=False)
    pd.DataFrame(native_sigmas).to_csv(out / 'refreshed_native_tm_sigma.csv', index=False)
    save(out / 'refreshed_predictions_before_outcomes.json', coverage)
    prediction_hashes = {name: file_digest(out / name) for name in (
        'frozen_predictions_before_outcomes.json', 'refreshed_predictions_before_outcomes.json',
        'refreshed_features.csv', 'refreshed_native_tm_sigma.csv')}
    save(out / 'feature_audit.json', {
        'outputs': {name: prediction_hashes[name] for name in (
            'refreshed_features.csv', 'refreshed_native_tm_sigma.csv')},
        'ratings_version': FEATURE_VERSION,
        'same_day_policy': 'strictly earlier completed source calendar days',
        'roster_policy': 'previous_observed_roster_proxy',
        'information_policy': 'frozen_weights__refreshed_day_start_features__previous_observed_roster_proxy__retrospective',
        'supplemental_tm_sigma_source': 'native RatingManager.predict_match at the identical prior-day state and roster; no imputation',
        'promotion_blockers': ['development history', 'retrospective model creation',
            '309 missing global source series', 'uncertified source and roster publication'],
    })
    # Outcome join starts only after prediction files have been finalized.
    raw_lookup = {r['wiki_series_id']: r for r in source_rows}
    reconciliation = []
    for row in [*coverage, *frozen_rows]:
        if row['probability_a'] is None:
            continue
        raw = raw_lookup[row['series_id']]
        winner_side = raw['winner_side']
        if winner_side not in ('1', '2'):
            row.update(status='unscorable', reason='source_has_no_binary_winner')
            continue
        winner = row['wiki_team_a_id'] if winner_side == '1' else row['wiki_team_b_id']
        row['y_true'] = int(row['team_a'] == winner)
        record = target_records[row['golgg_match_id']]
        t1, t2 = _series_team_ids(record)
        wiki_scores = {row['wiki_team_a_id']: int(raw['team1_score']), row['wiki_team_b_id']: int(raw['team2_score'])}
        recovered_scores = {t1: record['t1_score'], t2: record['t2_score']}
        check = {
            'series_id': row['series_id'], 'golgg_match_id': row['golgg_match_id'],
            'wiki_source_date': raw['source_date'], 'wiki_scheduled_at_utc': raw['scheduled_at_utc'] or None,
            'golgg_source_date': record['date'], 'wiki_scores': wiki_scores,
            'recovered_scores': recovered_scores, 'observed_maps': len(record['games']),
            'score_agrees': wiki_scores == recovered_scores,
            'map_count_agrees': sum(wiki_scores.values()) == len(record['games']),
            'winner_agrees': max(recovered_scores, key=recovered_scores.get) == winner,
            'utc_scheduled_day_agrees': not raw['scheduled_at_utc'] or raw['scheduled_at_utc'][:10] == record['date'],
            'raw_calendar_day_agrees': raw['source_date'] == record['date'],
            'recovery_provenance': record.get('recovery_provenance'),
            'outcome_authority': 'archived_Leaguepedia_series_result; recovered_GOLGG_is_crosscheck_only',
        }
        reconciliation.append(check)
        if not all(check[k] for k in ('score_agrees', 'map_count_agrees', 'winner_agrees', 'utc_scheduled_day_agrees')):
            row.update(status='prediction_with_outcome_source_discrepancy', reason='see outcome_reconciliation.json; never promoted to validation')
    save(out / 'series_rows.json', [*coverage, *frozen_rows])
    pd.DataFrame([*coverage, *frozen_rows]).to_csv(out / 'series_rows.csv', index=False)
    save(out / 'outcome_reconciliation.json', reconciliation)
    report = {
        'qualification': 'development_only_retrospective_reconstruction_not_out_of_time_validation',
        'probability_unit': 'actual_series_win; not tournament_advancement',
        'training': artifact['provenance']['fold'], 'model_sha256': model_hash,
        'model_created_at': artifact.get('created_at'), 'model_eligible_from': artifact['eligible_from'],
        'source_inventory_series': len(coverage), 'source_inventory_phases': len(ready['phases']),
        'refreshed_candidates': len(fixtures), 'refreshed_predicted': len(features),
        'frozen_lck_predicted': len(frozen_rows),
        'coverage_status_counts': dict(Counter(r['status'] for r in coverage)),
        'coverage_exclusion_counts': dict(Counter(r['reason'] for r in coverage if r['probability_a'] is None)),
        'strict_export_failures': failed, 'strict_recovered_history_audit': history_audit,
        'recovery_completeness': load(recovery_path), 'input_sha256': inputs,
        'prediction_files_before_outcome_join_sha256': prediction_hashes,
        'script_sha256': file_digest(Path(__file__)),
        'limitations': [
            'All inspected history is development data; no untouched holdout or superiority claim.',
            'Frozen and refreshed policies are separate rows and must never be pooled.',
            'Chronological-v2 feature semantics match model; the different recovered source coverage remains diagnostic.',
            'Recovered history has 309 globally missing series; no missing transition is fabricated or suppressed during strict parsing.',
            'Source calendar days are not publication timestamps; only strictly earlier completed source days enter features.',
            'Model created after series; historical roster proxies are not certified announcements.',
            'No original EXP081 frozen weights were substituted for chronological-v2 weights.',
        ],
    }
    save(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('source_inventory_series', 'source_inventory_phases', 'refreshed_candidates', 'refreshed_predicted', 'frozen_lck_predicted', 'coverage_status_counts', 'coverage_exclusion_counts')}, indent=2))


if __name__ == '__main__':
    main()
