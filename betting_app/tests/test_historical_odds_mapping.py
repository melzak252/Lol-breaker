"""Historical odds must be linked by independent identities, never by outcomes."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[2] / 'scripts/03_dane_pipeline/00_map_matches.py'
spec = importlib.util.spec_from_file_location('historical_odds_mapper', PATH)
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)


def match(mid, first, second, score=(2, 0)):
    return {'match_id': mid, 'date': '2025-02-01', 'name_1': first, 'name_2': second,
            'score_1': score[0], 'score_2': score[1], 't1_win': score[0] > score[1],
            't2_win': score[1] > score[0], 'draw': score[0] == score[1]}


def quote(first, second, url='https://example.test/fixture', score=(2, 0)):
    row = {'date': '2025-02-01 18:00:00', 'home_team': first, 'away_team': second,
           'homeResult': score[0], 'awayResult': score[1], 'url': url, 'tournament': 'test',
           'avg_odds_home': 1.5, 'avg_odds_away': 3.0, 'avg_open_home': 1.4, 'avg_open_away': 3.2}
    for book in ('betclic', 'betfan', 'efortuna', 'lv_bet', 'sts', 'superbet', 'fuksiarz'):
        row.update({f'odds1_{book}_close': 1.5, f'odds2_{book}_close': 3.0,
                    f'odds1_{book}_open': 1.4, f'odds2_{book}_open': 3.2})
    return row


def run(tmp_path, games, prices, aliases=None, input_format='csv', tail='', snapshot_bytes=None):
    raw = tmp_path / f'raw.{input_format}'; golgg = tmp_path / 'matches.json'
    output = tmp_path / 'mapped.csv'; alias_path = tmp_path / 'aliases.json'
    if input_format == 'csv':
        pd.DataFrame(prices).to_csv(raw, index=False)
    elif input_format == 'json':
        raw.write_text(json.dumps(prices))
    else:
        raw.write_text(''.join(json.dumps(row) + '\n' for row in prices) + tail)
    golgg.write_text(json.dumps(games))
    alias_payload = aliases if aliases and aliases.get('reviewed') is True else {
        'reviewed': True, 'aliases': aliases or {}}
    alias_path.write_text(json.dumps(alias_payload))
    before = json.loads(alias_path.read_text())
    options = {} if snapshot_bytes is None else {'snapshot_bytes': snapshot_bytes}
    mapper.map_matches(raw, golgg, output, alias_path, **options)
    assert json.loads(alias_path.read_text()) == before
    return pd.read_csv(output)


@pytest.mark.parametrize('first,second', [('Unrelated A', 'Unrelated B'), ('Alpha', 'Unrelated B')])
def test_matching_scores_do_not_establish_fixture_identity(tmp_path, first, second):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote(first, second)])
    assert result.empty


def test_academy_and_main_squad_are_not_the_same_identity(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta'), match(2, 'Alpha Academy', 'Beta')],
                 [quote('Alpha Academy', 'Beta')])
    assert result.golgg_match_id.tolist() == [2]


def test_repeated_same_day_pair_is_not_resolved_by_score_or_input_order(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta'), match(2, 'Alpha', 'Beta', (0, 2))],
                 [quote('Alpha', 'Beta'), quote('Alpha', 'Beta', url='https://example.test/other', score=(0, 2))])
    assert result.empty


def test_duplicate_raw_candidates_do_not_consume_a_single_fixture(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')],
                 [quote('Alpha', 'Beta'), quote('Alpha', 'Beta', url='https://example.test/other')])
    assert result.empty


def test_reversed_identity_swaps_every_quote_and_retains_raw_provenance(tmp_path):
    result = run(tmp_path, [match(1, 'Beta', 'Alpha', (0, 2))], [quote('Alpha', 'Beta')])
    row = result.iloc[0]
    assert row.avg_odds_home == 3.0 and row.avg_odds_away == 1.5
    assert row.avg_open_home == 3.2 and row.avg_open_away == 1.4
    for book in ('betclic', 'betfan', 'efortuna', 'lv_bet', 'sts', 'superbet', 'fuksiarz'):
        assert row[f'odds1_{book}_close'] == 3.0
        assert row[f'odds2_{book}_close'] == 1.5
        assert row[f'odds1_{book}_open'] == 3.2
        assert row[f'odds2_{book}_open'] == 1.4
    assert row.source_home_team == 'Alpha' and row.source_away_team == 'Beta'
    assert bool(row.source_sides_swapped)


def test_legacy_learned_aliases_are_not_silently_treated_as_reviewed(tmp_path):
    path = tmp_path / 'aliases.json'
    path.write_text(json.dumps({'alpha': 'unrelated'}))
    with pytest.raises(ValueError, match='reviewed'):
        mapper.load_aliases(path)


def test_both_adjacent_dates_are_considered_before_resolving_identity(tmp_path):
    games = [match(1, 'Alpha', 'Beta'), match(2, 'Alpha', 'Beta')]
    games[0]['date'] = '2025-01-31'
    games[1]['date'] = '2025-02-02'
    result = run(tmp_path, games, [quote('Alpha', 'Beta')])
    assert result.empty


def test_result_contradiction_rejects_but_never_changes_identity(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote('Alpha', 'Beta', score=(0, 2))])
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['rows'][0]['candidate_ids'] == [1]
    assert audit['rows'][0]['status'] == 'result_mismatch'


def test_matching_winner_does_not_clear_conflicting_series_score(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta', score=(2, 0))],
                 [quote('Alpha', 'Beta', score=(2, 1))])
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['rows'][0]['candidate_ids'] == [1]
    assert audit['rows'][0]['status'] == 'score_mismatch'


def test_reviewed_alias_can_align_both_independent_identities(tmp_path):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote('A', 'B')],
                 aliases={'A': 'Alpha', 'B': 'Beta'})
    assert result.golgg_match_id.tolist() == [1]


def test_existing_mapping_evidence_is_never_overwritten(tmp_path):
    run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote('Alpha', 'Beta')])
    output = tmp_path / 'mapped.csv'
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        mapper.map_matches(tmp_path / 'raw.csv', tmp_path / 'matches.json', output, tmp_path / 'aliases.json')
    assert output.read_bytes() == before


def test_canonical_golgg_names_and_scores_follow_series_team_ids_not_header_order(tmp_path):
    game = {
        'match_id': '1', 'date': '2025-02-01',
        'sname_t1': 'Alpha', 'sname_t2': 'Beta',
        't1_id': '20', 't2_id': '10', 't1_score': 0, 't2_score': 2,
        't1_win': False, 't2_win': True, 'draw': False,
        'games': [
            {'t1_id': '10', 't2_id': '20', 't1_name': 'Alpha', 't2_name': 'Beta',
             't1_win': True, 't2_win': False},
            {'t1_id': '20', 't2_id': '10', 't1_name': 'Beta', 't2_name': 'Alpha',
             't1_win': False, 't2_win': True},
        ],
    }
    result = run(tmp_path, [game], [quote('Alpha', 'Beta')])
    row = result.iloc[0]
    assert (row.golgg_team1, row.golgg_team2) == ('Beta', 'Alpha')
    assert (row.t1_score, row.t2_score) == (0, 2)
    assert (row.avg_open_home, row.avg_open_away) == (3.2, 1.4)
    assert bool(row.source_sides_swapped)


@pytest.mark.parametrize('input_format', ['json', 'jsonl'])
def test_normalized_download_string_scores_map_as_numbers(tmp_path, input_format):
    price = quote('Alpha', 'Beta', score=('2', '0'))
    price.update(event_id='abcd1234',
                 url='https://www.oddsportal.com/esports/league-of-legends/test/alpha-beta-abcd1234/',
                 date='2025-02-01T18:00:00+00:00', odds_status='available')
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format=input_format)
    assert result.golgg_match_id.tolist() == [1]
    assert result.source_event_id.tolist() == ['abcd1234']


def test_reversal_preserves_additional_bookmaker_prices_and_sidewise_change_clocks(tmp_path):
    price = quote('Alpha', 'Beta')
    price.update(odds1_rainbet_open=1.7, odds2_rainbet_open=2.8,
                 odds1_rainbet_close=1.9, odds2_rainbet_close=2.5,
                 odds1_rainbet_open_at='2025-01-31T12:00:00+00:00',
                 odds2_rainbet_open_at='2025-01-31T12:01:00+00:00',
                 odds1_rainbet_close_at='2025-02-01T17:30:00+00:00',
                 odds2_rainbet_close_at='2025-02-01T17:31:00+00:00')
    result = run(tmp_path, [match(1, 'Beta', 'Alpha', (0, 2))], [price])
    row = result.iloc[0]
    for period in ('open', 'close'):
        for suffix in ('', '_at'):
            assert row[f'odds1_rainbet_{period}{suffix}'] == price[f'odds2_rainbet_{period}{suffix}']
            assert row[f'odds2_rainbet_{period}{suffix}'] == price[f'odds1_rainbet_{period}{suffix}']


def test_missing_counterpart_quote_remains_missing_after_reversal(tmp_path):
    price = quote('Alpha', 'Beta')
    price.update(odds1_rainbet_open=1.7,
                 odds1_rainbet_open_at='2025-01-31T12:00:00+00:00')
    row = run(tmp_path, [match(1, 'Beta', 'Alpha', (0, 2))], [price]).iloc[0]
    assert pd.isna(row.odds1_rainbet_open)
    assert pd.isna(row.odds1_rainbet_open_at)
    assert row.odds2_rainbet_open == 1.7
    assert row.odds2_rainbet_open_at == '2025-01-31T12:00:00+00:00'


def test_timezone_aware_start_uses_utc_day_instead_of_local_calendar_day(tmp_path):
    games = [match(1, 'Alpha', 'Beta'), match(2, 'Alpha', 'Beta')]
    games[0]['date'] = '2025-01-31'
    price = quote('Alpha', 'Beta')
    price['date'] = '2025-02-01T00:30:00+02:00'
    result = run(tmp_path, games, [price])
    assert result.golgg_match_id.tolist() == [1]
    assert result.odds_date.tolist() == ['2025-01-31']
    assert result.date_offset_days.tolist() == [0]


def test_live_journal_excludes_incomplete_final_record_and_binds_complete_prefix(tmp_path):
    prices = [quote('Alpha', 'Beta')]
    complete = ''.join(json.dumps(row) + '\n' for row in prices).encode()
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], prices,
                 input_format='jsonl', tail='{"home_team":"unfinished')
    assert result.golgg_match_id.tolist() == [1]
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    import hashlib
    assert audit['source']['complete_prefix_bytes'] == len(complete)
    assert audit['source']['sha256'] == hashlib.sha256(complete).hexdigest()
    assert audit['source']['incomplete_tail_bytes'] == len(b'{"home_team":"unfinished')


def test_explicit_snapshot_boundary_excludes_later_complete_download_records(tmp_path):
    prices = [quote('Alpha', 'Beta'), quote('Gamma', 'Delta')]
    boundary = len((json.dumps(prices[0]) + '\n').encode())
    result = run(tmp_path, [match(1, 'Alpha', 'Beta'), match(2, 'Gamma', 'Delta')],
                 prices, input_format='jsonl', snapshot_bytes=boundary)
    assert result.golgg_match_id.tolist() == [1]
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['source']['complete_prefix_bytes'] == boundary
    assert audit['source']['rows'] == 1


def test_malformed_complete_journal_record_is_not_treated_as_pending_tail(tmp_path):
    with pytest.raises(ValueError, match='JSONL'):
        run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote('Alpha', 'Beta')],
            input_format='jsonl', tail='{"broken":}\n')
    assert not (tmp_path / 'mapped.csv').exists()


def test_failed_download_never_becomes_a_quote_fixture_from_legacy_prices(tmp_path):
    price = quote('Alpha', 'Beta')
    price.update(odds_status='failed', legacy_row=quote('Alpha', 'Beta'))
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price])
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['rows'][0]['status'] == 'source_odds_unavailable'


@pytest.mark.parametrize('scores', [(-1, 0), (2.5, 0), ('unknown', '0')])
def test_invalid_scores_are_audited_without_selecting_another_fixture(tmp_path, scores):
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [quote('Alpha', 'Beta', score=scores)])
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['rows'][0]['candidate_ids'] == [1]
    assert audit['rows'][0]['status'] == 'invalid_source_result'


@pytest.mark.parametrize('source,target', [
    ('Barca eSports', 'Barça eSports'),
    ('Besiktas Esports', 'Beşiktaş Esports'),
    ('Koln Gaming', 'Köln Gaming'),
])
def test_accented_spelling_matches_without_removing_squad_identity(tmp_path, source, target):
    result = run(tmp_path, [match(1, target, 'Beta'), match(2, target + ' Academy', 'Beta')],
                 [quote(source, 'Beta')])
    assert result.golgg_match_id.tolist() == [1]


def scoped_rule(rule_id, source, target, participant_id, start='2025-01-01', end='2025-12-31'):
    return {'id': rule_id, 'source_name': source, 'target_name': target,
            'source_participant_id': str(participant_id), 'valid_from': start, 'valid_to': end,
            'evidence': ['https://identity.example.test/announcement']}


def with_archive_identity(price, home_id, away_id):
    return {**price, 'source_archive_row': {
        'home-name': price['home_team'], 'away-name': price['away_team'],
        'home-participant-id': home_id, 'away-participant-id': away_id}}


def reviewed_rules(*rules):
    return {'reviewed': True, 'aliases': {}, 'scoped_aliases': list(rules)}


def test_same_display_name_resolves_two_distinct_source_participant_entities(tmp_path):
    prices = [with_archive_identity(quote('Shared Label', 'Beta'), 101, 201),
              with_archive_identity(quote('Shared Label', 'Delta'), 102, 202)]
    aliases = reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101),
                             scoped_rule('regional', 'Shared Label', 'Gamma', 102))
    result = run(tmp_path, [match(1, 'Alpha', 'Beta'), match(2, 'Gamma', 'Delta')],
                 prices, aliases=aliases, input_format='jsonl')
    assert result.golgg_match_id.tolist() == [1, 2]


@pytest.mark.parametrize('participant_id', [None, 999])
def test_scoped_name_relation_never_applies_without_its_source_participant_id(tmp_path, participant_id):
    price = quote('Shared Label', 'Beta')
    if participant_id is not None:
        price = with_archive_identity(price, participant_id, 201)
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)))
    assert result.empty


@pytest.mark.parametrize('start,end,expected', [
    ('2025-02-01', '2025-02-01', [1]),
    ('2025-02-02', '2025-12-31', []),
    ('2025-01-01', '2025-01-31', []),
])
def test_scoped_alias_validity_is_bounded_by_source_utc_day(tmp_path, start, end, expected):
    price = with_archive_identity(quote('Shared Label', 'Beta'), 101, 201)
    price['date'] = '2025-02-02T00:30:00+02:00'
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101, start, end)))
    assert result.golgg_match_id.tolist() == expected


def test_scoped_alias_keeps_direct_historical_label_as_an_identity_variant(tmp_path):
    price = with_archive_identity(quote('Alpha', 'Beta'), 101, 201)
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('old-label', 'Alpha', 'Gamma', 101)))
    assert result.golgg_match_id.tolist() == [1]


def test_participant_ids_follow_current_price_order_when_archive_order_is_reversed(tmp_path):
    price = quote('Shared Label', 'Beta')
    price['source_archive_row'] = {
        'home-name': 'Beta', 'away-name': 'Shared Label',
        'home-participant-id': 201, 'away-participant-id': 101,
    }
    result = run(tmp_path, [match(1, 'Beta', 'Alpha', (0, 2))], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)))
    row = result.iloc[0]
    assert bool(row.source_sides_swapped)
    assert (row.avg_open_home, row.avg_open_away) == (3.2, 1.4)


def test_archive_ids_do_not_identify_changed_unknown_source_participants(tmp_path):
    price = with_archive_identity(quote('Shared Label', 'Beta'), 101, 201)
    price['source_archive_row']['home-name'] = 'Unrelated'
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)))
    assert result.empty


def test_alias_variants_with_two_fixtures_remain_ambiguous_regardless_of_scores(tmp_path):
    price = with_archive_identity(quote('Shared Label', 'Beta'), 101, 201)
    aliases = reviewed_rules(scoped_rule('first', 'Shared Label', 'Alpha', 101),
                             scoped_rule('second', 'Shared Label', 'Gamma', 101))
    result = run(tmp_path, [match(1, 'Alpha', 'Beta'), match(2, 'Gamma', 'Beta', (0, 2))],
                 [price], aliases=aliases, input_format='jsonl')
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert set(audit['rows'][0]['candidate_ids']) == {1, 2}
    assert audit['rows'][0]['status'] == 'ambiguous_golgg_fixture'


def test_aliases_cannot_resolve_uncertain_source_orientation_from_winner(tmp_path):
    price = with_archive_identity(quote('Shared Home', 'Shared Away'), 101, 201)
    aliases = reviewed_rules(
        scoped_rule('home-first', 'Shared Home', 'Alpha', 101),
        scoped_rule('home-second', 'Shared Home', 'Beta', 101),
        scoped_rule('away-first', 'Shared Away', 'Alpha', 201),
        scoped_rule('away-second', 'Shared Away', 'Beta', 201),
    )
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price],
                 aliases=aliases, input_format='jsonl')
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert audit['rows'][0]['status'] == 'ambiguous_source_orientation'


def test_alias_and_direct_quotes_competing_for_one_fixture_are_both_withheld(tmp_path):
    alias_price = with_archive_identity(quote('Shared Label', 'Beta'), 101, 201)
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')],
                 [alias_price, quote('Alpha', 'Beta', url='https://example.test/other')],
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)),
                 input_format='jsonl')
    assert result.empty
    audit = json.loads((tmp_path / 'mapped.audit.json').read_text())
    assert [row['status'] for row in audit['rows']] == ['ambiguous_odds_fixture'] * 2


@pytest.mark.parametrize('competition,expected', [
    ('League of Legends LEC 2025', [1]),
    ('League of Legends Prime League 2025', []),
])
def test_scoped_alias_does_not_cross_competition_divisions(tmp_path, competition, expected):
    price = with_archive_identity(quote('Shared Label', 'Beta'), 101, 201)
    price['tournament'] = competition
    rule = {**scoped_rule('main-only', 'Shared Label', 'Alpha', 101), 'tournament_pattern': 'LEC'}
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price],
                 aliases=reviewed_rules(rule), input_format='jsonl')
    assert result.golgg_match_id.tolist() == expected


def test_combined_snapshot_preserves_participant_bound_identity_resolution(tmp_path):
    price = {**quote('Shared Label', 'Beta'), 'source_identity': {
        'source_archive_home_name': 'Shared Label', 'source_archive_away_name': 'Beta',
        'source_home_participant_id': 101, 'source_away_participant_id': 201,
    }}
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price], input_format='jsonl',
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)))
    assert result.golgg_match_id.tolist() == [1]


def test_csv_participant_columns_allow_scoped_aliases_without_nested_archive_payload(tmp_path):
    price = {**quote('Shared Label', 'Beta'), 'source_home_participant_id': 101,
             'source_away_participant_id': 201}
    result = run(tmp_path, [match(1, 'Alpha', 'Beta')], [price],
                 aliases=reviewed_rules(scoped_rule('main', 'Shared Label', 'Alpha', 101)))
    assert result.golgg_match_id.tolist() == [1]
