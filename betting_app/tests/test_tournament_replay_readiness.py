"""Replay audit follows archived exact team IDs, not display-name aliases."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_tournament_replay_readiness import audit
from scripts.score_tournament_forecasts import digest


def _inputs(tmp_path):
    history, snapshots = tmp_path/'matches.json', tmp_path/'snapshots.csv'
    requests, replay_audit = tmp_path/'requests.jsonl', tmp_path/'audit.json'
    empty = tmp_path/'empty'
    empty.mkdir()
    history.write_text(json.dumps([{
        'match_id': '1', 'date': '2024-01-14', 'tournament_name': 'Fixture', 'best_of': 1,
        'sname_t1': 'Rogue', 'sname_t2': 'TH', 'won': 'Team Heretics',
        'games': [{'game_id': '1', 't1_id': '2172', 't2_id': '2171',
                   't1_win': False, 't2_win': True}],
    }]))
    pd.DataFrame([{'golgg_match_id': '1', 'date': '2024-01-14', 'tournament': 'Fixture',
                   'team1_id': '2172', 'team2_id': '2171', 'best_of': 1, 'y_true': 0}]).to_csv(snapshots, index=False)
    requests.write_text('')
    replay_audit.write_text(json.dumps({'source': {'sha256': digest(history)},
                                       'outputs': {'snapshots.csv': digest(snapshots)}}))
    return SimpleNamespace(output_dir=tmp_path/'output', snapshots=snapshots, history=history,
                           requests=requests, replay_audit=replay_audit, event_root=[empty],
                           cache_dir=empty, from_date='2024-01-01')


def test_exact_map_winner_aligns_despite_abbreviated_series_display_name(tmp_path):
    result = audit(_inputs(tmp_path))
    assert result['source_aligned_rows'] == 1
    assert result['full_event_replay_eligible_bundles'] == 0


def test_unrelated_archive_cannot_be_used_for_snapshot_readiness(tmp_path):
    args = _inputs(tmp_path)
    args.history.write_text('[]')
    with pytest.raises(ValueError, match='originating replay audit'):
        audit(args)


def _manifest_inputs(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    phases = [
        {'phase_id': 'groups', 'title': 'Cup/Main Event', 'family': 'worlds',
         'year': 2024, 'stage': 'group_stage', 'status': 'completed',
         'start_date': '2024-01-01', 'end_date': '2024-01-03', 'source': {}},
        {'phase_id': 'finals', 'title': 'Cup/Main Event', 'family': 'worlds',
         'year': 2024, 'stage': 'playoffs', 'status': 'completed',
         'start_date': '2024-01-04', 'end_date': '2024-01-05', 'source': {}},
    ]
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))
    pd.DataFrame([{'event_title': 'Cup/Main Event', 'stage': 'playoffs',
                   'wiki_series_id': 's1', 'source_date': '2024-01-05',
                   'scheduled_at_utc': '', 'status': 'completed',
                   'team1': 'A', 'team2': 'B', 'best_of': 5,
                   'validation_errors': '[]'}]).to_csv(bundle / 'series_results.csv', index=False)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'phase_count': 2, 'profiles': [], 'phase_rules': []}))
    return bundle, catalog, phases


def test_inventory_retains_each_phase_and_does_not_call_title_group_whole_edition(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, _ = _manifest_inputs(tmp_path)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert {p['phase_id'] for p in result['phases']} == {'groups', 'finals'}
    assert result['coverage']['completed_phases'] == 2
    assert len(result['editions']) == 1
    assert result['editions'][0]['whole_edition_verified'] is False
    assert all(not p['axes']['axis1']['eligible'] for p in result['phases'])
    assert result['phases'][1]['series_evidence']['joined_rows'] == 1
    assert not result['eligible_origins']


def test_source_family_disambiguates_shared_title_without_using_results(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    other = {**phases[1], 'phase_id': 'regional-finals', 'family': 'lec'}
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, [*phases, other])))
    rows = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False)
    rows['family'] = 'worlds'
    rows.to_csv(bundle / 'series_results.csv', index=False)

    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])

    evidence = {phase['phase_id']: phase['series_evidence']['joined_rows']
                for phase in result['phases']}
    assert evidence == {'groups': 0, 'finals': 1, 'regional-finals': 0}
    assert result['unjoined_series'] == []


def test_inventory_reports_malformed_dates_missing_columns_and_failed_joins(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phases[0]['start_date'] = 'not-a-date'
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))
    pd.DataFrame([{'event_title': 'Unknown', 'wiki_series_id': 'orphan'}]).to_csv(
        bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert 'invalid_start_date' in result['phases'][0]['reasons']
    assert 'stage' in result['source_tables']['series_results.csv']['missing_columns']
    assert result['unjoined_series'][0]['wiki_series_id'] == 'orphan'
    assert len(result['phases']) == 2


def test_duplicate_phase_identity_is_rejected_instead_of_losing_coverage(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases + phases[:1])))
    with pytest.raises(ValueError, match='Duplicate'):
        build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])


def _archive_phase_pages(bundle, phases, formats, *, splits=None):
    """Minimal archived MediaWiki responses, with only pre-play identity/rules."""
    pages = []
    for pageid, (title, format_text) in enumerate(formats.items(), 1):
        members = [phase for phase in phases if phase['title'] == title]
        fields = f"|CM_StandardLeague=Fixture League\n|CM_Year={members[0]['year']}\n"
        if splits and title in splits:
            fields += f"|split={splits[title]}\n"
        pages.append({'pageid': pageid, 'title': title, 'revisions': [{
            'revid': pageid + 100, 'slots': {'main': {'content':
                '{{Infobox Tournament\n' + fields + '}}\n'
                '== Overview ==\n=== Format ===\n' + format_text +
                '\n=== Results ===\n[[Other Edition|A realized winner]]'}}}]})
        for phase in members:
            phase['source'] = {'pageid': pageid, 'revid': pageid + 100,
                               'raw_path': str(bundle / 'raw' / 'pages.json')}
    (bundle / 'raw').mkdir(exist_ok=True)
    (bundle / 'raw' / 'pages.json').write_text(json.dumps({'query': {'pages': pages}}))
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))


def test_ambiguous_identity_is_not_resolved_by_winner_score_or_placing(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phases.append({**phases[1], 'phase_id': 'other-finals', 'family': 'lec',
                   'winner': 'A', 'score': '3-0', 'placement': 1})
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))
    rows = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False)
    rows['winner_side'], rows['team1_score'], rows['team2_score'] = '1', '3', '0'
    rows.to_csv(bundle / 'series_results.csv', index=False)
    first = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    rows['winner_side'], rows['team1_score'], rows['team2_score'] = '2', '0', '3'
    rows.to_csv(bundle / 'series_results.csv', index=False)
    second = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    for result in (first, second):
        assert result['source_tables']['series_results.csv']['joined_rows'] == 0
        assert result['unjoined_series'][0]['reason'] == 'ambiguous_phase_join'
        assert result['unjoined_series'][0]['candidate_phase_ids'] == ['finals', 'other-finals']


def test_composite_source_phase_keeps_swiss_and_playoffs_under_one_identity(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    composite = {**phases[0], 'phase_id': 'main', 'stage': 'main',
                 'end_date': '2024-01-05', 'family': 'eu_masters',
                 'format_summary': 'Swiss followed by single-elimination playoffs.'}
    _archive_phase_pages(bundle, [composite], {
        composite['title']: '* Swiss Stage\n* Sixteen teams advance to single-elimination Playoffs'})
    base = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False).iloc[0].to_dict()
    pd.DataFrame([{**base, 'wiki_series_id': 'swiss', 'stage': 'swiss',
                   'families': '["eu_masters"]', 'source_date': '2024-01-02'},
                  {**base, 'wiki_series_id': 'knockout', 'stage': 'playoffs',
                   'families': '["eu_masters"]'}]).to_csv(bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert result['phases'][0]['series_evidence']['source_row_numbers'] == [2, 3]
    assert {row['phase_id'] for row in result['series_reconciliation']} == {'main'}
    assert {row['mapping']['method'] for row in result['series_reconciliation']} == {
        'archived_composite_phase'}
    assert result['coverage']['duplicate_series_assignments'] == 0
    # A mechanics label or a result alone cannot stand in for the archived format.
    (bundle / 'raw' / 'pages.json').unlink()
    unresolved = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert len(unresolved['unjoined_series']) == 2


def test_group_and_rumble_source_labels_resolve_only_consistent_phase_boundaries(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phases[0].update(family='msi', format_summary='Initial group stage: three groups.')
    phases[1].update(family='msi', stage='group_stage',
                     format_summary='Rumble stage: one double round robin.')
    _archive_phase_pages(bundle, phases, {
        phases[0]['title']: '* Stage 1 (Groups)\n* Stage 2 (Rumble)'})
    base = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False).iloc[0].to_dict()
    pd.DataFrame([
        {**base, 'wiki_series_id': 'initial', 'stage': 'group_stage', 'family': 'msi',
         'source_date': '2024-01-02', 'tab': 'Groups Day 2'},
        {**base, 'wiki_series_id': 'rumble', 'stage': 'group_stage', 'family': 'msi',
         'source_date': '2024-01-04', 'tab': 'Rumble Day 1'},
        {**base, 'wiki_series_id': 'conflict', 'stage': 'group_stage', 'family': 'msi',
         'source_date': '2024-01-02', 'tab': 'Rumble Day 1'},
    ]).to_csv(bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert {row['wiki_series_id']: row['phase_id'] for row in result['series_reconciliation']} == {
        'initial': 'groups', 'rumble': 'finals', 'conflict': None}
    assert result['unjoined_series'][0]['reason'] == 'phase_label_date_conflict'


@pytest.mark.parametrize(('source_date', 'source_time', 'zone', 'dst', 'scheduled', 'local', 'day'), [
    ('2024-06-12', '23:00', 'PST', 'yes', '2024-06-13T06:00:00+00:00',
     '2024-06-12', '2024-06-13'),
    ('00:00', '2024-01-05', 'CET', 'no', '2024-01-04T23:00:00+00:00',
     '2024-01-05', '2024-01-05'),
])
def test_zoned_event_days_preserve_raw_dates_without_certifying_completion(
        tmp_path, source_date, source_time, zone, dst, scheduled, local, day):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, _ = _manifest_inputs(tmp_path)
    (bundle / 'time_offsets.json').write_text(json.dumps({'yes': {'PST': 7}, 'no': {'CET': -1}}))
    rows = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False)
    rows['source_date'], rows['source_time'] = source_date, source_time
    rows['source_timezone'], rows['source_dst'], rows['scheduled_at_utc'] = zone, dst, scheduled
    rows.to_csv(bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    evidence = result['series_reconciliation'][0]['event_day']
    assert evidence['raw']['source_date'] == source_date
    assert evidence['normalized_local_event_day'] == local
    assert evidence['scheduled_utc_event_day'] == scheduled[:10]
    assert evidence['conservative_event_day'] == day
    assert evidence['completion_at'] is None and evidence['available_at'] is None
    assert evidence['completion_certified'] is False
    assert not result['eligible_origins']


def test_conflicting_zoned_timestamp_does_not_supply_a_release_day(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, _ = _manifest_inputs(tmp_path)
    (bundle / 'time_offsets.json').write_text('{"yes": {"PST": 7}}')
    rows = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False)
    rows['source_time'], rows['source_timezone'], rows['source_dst'] = '23:00', 'PST', 'yes'
    rows['scheduled_at_utc'] = '2024-01-05T23:00:00+00:00'
    rows.to_csv(bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    evidence = result['series_reconciliation'][0]['event_day']
    assert evidence['conservative_event_day'] is None
    assert 'scheduled_timestamp_disagrees_with_archived_zone' in evidence['issues']


def test_archived_format_links_group_phases_but_not_distinct_splits(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phases[0]['title'] = 'Cup/Spring Groups'
    phases[1]['title'] = 'Cup/Spring Finals'
    summer = {**phases[1], 'phase_id': 'summer', 'title': 'Cup/Summer Finals'}
    phases.append(summer)
    _archive_phase_pages(bundle, phases, {
        phases[0]['title']: '* Top two advance to [[Cup/Spring Finals|Finals]].',
        phases[1]['title']: '* Seeds come from [[Cup/Spring Groups|Groups]].\n'
                            '* Winner also qualifies for [[Cup/Summer Finals|Summer]].',
        summer['title']: '* Single elimination.',
    }, splits={phase['title']: 'Summer' if phase['phase_id'] == 'summer' else 'Spring'
               for phase in phases})
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    editions = {frozenset(edition['phase_ids']) for edition in result['editions']}
    assert editions == {frozenset({'groups', 'finals'}), frozenset({'summer'})}
    assert all(not edition['whole_edition_verified'] for edition in result['editions'])
    assert all(edition['edition_start_at'] is None for edition in result['editions'])
    assert all(not edition['cluster_start_is_edition_start'] for edition in result['editions'])
    assert result['phases'][1]['phase_start_date'] == '2024-01-04'
    assert any(link['reason'] == 'different_archived_split'
               for link in result['unresolved_phase_links'])


def test_duplicate_series_identity_is_excluded_without_duplicate_assignment(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, _ = _manifest_inputs(tmp_path)
    rows = pd.read_csv(bundle / 'series_results.csv', keep_default_na=False)
    pd.concat([rows, rows], ignore_index=True).to_csv(bundle / 'series_results.csv', index=False)
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert result['source_tables']['series_results.csv']['joined_rows'] == 0
    assert len(result['unjoined_series']) == 2
    assert {row['reason'] for row in result['unjoined_series']} == {'duplicate_series_identity'}


def test_changed_archive_is_hashed_and_cannot_authorize_a_composite_join(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phase = {**phases[0], 'stage': 'main', 'format_summary': 'Groups followed by playoffs.'}
    _archive_phase_pages(bundle, [phase], {phase['title']: '* Groups\n* Playoffs'})
    raw_path = bundle / 'raw' / 'pages.json'
    original_hash = digest(raw_path)
    (bundle / 'source_manifest.json').write_text(json.dumps({
        'files': {'raw/pages.json': {'sha256': original_hash}}}))
    raw_path.write_text(raw_path.read_text() + '\n')
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    assert result['source_tables']['series_results.csv']['joined_rows'] == 0
    assert any(change.get('expected_sha256') == original_hash
               and change.get('sha256') == digest(raw_path)
               for change in result['source_changes'])


def test_archived_team_aliases_corrobate_group_identity_without_seeding(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, _ = _manifest_inputs(tmp_path)
    (bundle / 'team_aliases.json').write_text(json.dumps({
        name: {'resolved': True, 'canonical_title': canonical}
        for name, canonical in [('A', 'Team Alpha'), ('ALPHA', 'Team Alpha'),
                                ('B', 'Team Beta'), ('BETA', 'Team Beta')]}))
    (bundle / 'source_group_definitions.jsonl').write_text(json.dumps({
        'title': 'Data:Cup/Main Event', 'revid': 10, 'source_ordinal': 1,
        'fields': {'groups': 'Group A,Group B', 'Group A': 'ALPHA,BETA', 'Group B': 'C,D'}}))
    result = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[])
    row = result['series_reconciliation'][0]
    assert row['teams']['team1']['canonical_title'] == 'Team Alpha'
    assert [match['group'] for match in row['group_membership_evidence']] == ['Group A']
    assert row['group_membership_used_as_seed_or_qualifier'] is False
    assert result['phases'][1]['reconstruction']['eligible'] is False


def test_prior_phase_seed_evidence_is_research_candidate_without_prestart_capture(tmp_path):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    bundle, catalog, phases = _manifest_inputs(tmp_path)
    phases[1]['participant_count'] = 2
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))
    pilot_dir = tmp_path / 'pilot'
    pilot_dir.mkdir()
    # A distinct, earlier phase is required; never use target-phase placements.
    phases[0]['title'] = 'Cup/Groups'
    (bundle / 'events.jsonl').write_text('\n'.join(map(json.dumps, phases)))
    payload = {'source_phase': phases[1], 'team_ids_in_seed_order': ['1', '2'],
               'team_names': {'1': 'Team A', '2': 'Team B'},
               'prior_standings': [
                   {'event_title': 'Cup/Groups', 'placement': '1', 'team': 'TEAM A'},
                   {'event_title': 'Cup/Groups', 'placement': '2', 'team': 'Team B'}],
               'outcomes_used_for_features_or_simulation': False}
    path = pilot_dir / 'seeding-and-outcome-evidence.json'
    path.write_text(json.dumps(payload))
    manifest = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[pilot_dir])
    candidate = manifest['pilot_candidates'][0]
    assert candidate['status'] == 'phase_start_reconstruction_candidate'
    assert candidate['historical_certification'] is False
    assert candidate['eligible_forecast'] is False
    assert manifest['phases'][1]['reconstruction']['captured_forecast_required'] is False
    payload['prior_standings'][0]['event_title'] = 'Cup/Main Event'
    path.write_text(json.dumps(payload))
    manifest = build_readiness_manifest(bundle, catalog=catalog, pilot_roots=[pilot_dir])
    assert manifest['pilot_candidates'][0]['status'] == 'excluded'


def _pilot_sources(tmp_path):
    from scripts.audit_tournament_replay_readiness import DEFAULT_BUNDLE, DEFAULT_PILOT
    bundle = tmp_path / 'pilot-bundle'
    bundle.mkdir()
    evidence = json.loads(DEFAULT_PILOT.read_text())
    (bundle / 'events.jsonl').write_text((DEFAULT_BUNDLE / 'events.jsonl').read_text())
    (bundle / 'team_aliases.json').write_text((DEFAULT_BUNDLE / 'team_aliases.json').read_text())
    rows = pd.read_csv(DEFAULT_BUNDLE / 'series_results.csv', keep_default_na=False)
    rows = rows[rows.event_title == evidence['source_phase']['title']]
    rows.to_csv(bundle / 'series_results.csv', index=False)
    return bundle, rows


def test_pilot_compiles_source_routes_and_does_not_invent_a_champion(tmp_path):
    from scripts.audit_tournament_replay_readiness import compile_pilot_inputs
    bundle, _ = _pilot_sources(tmp_path)
    result = compile_pilot_inputs(tmp_path / 'compiled', bundle=bundle)
    outcome = json.loads((tmp_path / 'compiled' / 'outcomes.json').read_text())
    assert result['scope'] == 'phase_start'
    assert result['qualification'] == 'reconstructed_not_certified'
    assert not any(t['target_kind'] == 'champion' for t in result['targets'])
    assert {r['round']: r['winner'] for r in outcome['result_rows']} == {
        'r1': '2802', 'r2': '2806', 'r3': '2804', 'upper': '2805', 'last': '2809'}
    assert outcome['qualified_team_ids'] == ['2805', '2809']
    for origin in result['origins']:
        state = json.loads((tmp_path / 'compiled' / origin['state']).read_text())
        assert all(row['completed_date'] < origin['cutoff'][:10] for row in state['completed'])
        assert all('completed_at' not in row and 'available_at' not in row for row in state['completed'])
    with pytest.raises(FileExistsError):
        compile_pilot_inputs(tmp_path / 'compiled', bundle=bundle)


def test_pilot_rejects_unmapped_teams_and_illegal_source_scores(tmp_path):
    from scripts.audit_tournament_replay_readiness import compile_pilot_inputs
    bundle, rows = _pilot_sources(tmp_path)
    rows.loc[rows.index[0], 'team1'] = 'Unverified Academy Alias'
    rows.to_csv(bundle / 'series_results.csv', index=False)
    with pytest.raises(ValueError, match='canonical'):
        compile_pilot_inputs(tmp_path / 'bad-team', bundle=bundle)
    rows.loc[rows.index[0], 'team1'] = 'Dplus Kia'
    rows.loc[rows.index[0], 'team1_score'] = '2'
    rows.to_csv(bundle / 'series_results.csv', index=False)
    with pytest.raises(ValueError, match='score'):
        compile_pilot_inputs(tmp_path / 'bad-score', bundle=bundle)


