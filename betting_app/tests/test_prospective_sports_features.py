"""Canonical raw feature reconstruction must not consume present/future results."""
from datetime import date
from copy import deepcopy
import pandas as pd
import pytest
from betting_app.tests.test_siamese_research_dataset import match

ROSTERS = {'a': [str(i) for i in range(5)], 'b': [str(i) for i in range(5, 10)]}


def test_future_outcomes_and_input_order_cannot_change_frozen_features():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01'), match(2, '2024-01-02', winner=False), match(3, '2024-01-03')]
    first, audit = build_pair_features(history, ROSTERS, date(2024, 1, 3), [1, 3, 5])
    changed = deepcopy(history)
    changed[2] = match(3, '2024-01-03', winner=False, kills=900)
    changed.append(match(4, '2025-01-01', winner=False, kills=999))
    second, _ = build_pair_features(reversed(changed), ROSTERS, date(2024, 1, 3), [1, 3, 5])
    pd.testing.assert_frame_equal(first, second)
    assert first.t1_rolling_kills.tolist() == [10., 10., 10.]
    assert first.days_since_last_1.tolist() == [1, 1, 1]
    assert audit['used_series'] == 2


def test_incomplete_recent_stats_are_not_skipped_or_imputed():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01'), match(2, '2024-01-02')]
    del history[0]['games'][0]['t1_stats']['gold']
    with pytest.raises(ValueError):
        build_pair_features(history, ROSTERS, date(2024, 1, 3), [3])


def test_unobserved_announced_player_is_not_replaced_by_last_roster():
    from scripts.prospective_sports_features import build_pair_features
    rosters = deepcopy(ROSTERS)
    rosters['a'][0] = 'new-unobserved-player'
    with pytest.raises(ValueError):
        build_pair_features([match(1, '2024-01-01')], rosters, date(2024, 1, 2), [3])


def test_same_day_numeric_match_order_matches_chronological_replay():
    from scripts.prospective_sports_features import build_pair_features
    numeric = [match(10, '2024-01-02', winner=False), match(2, '2024-01-02')]
    chronological = [match(2, '2024-01-01'), match(10, '2024-01-02', winner=False)]
    actual, _ = build_pair_features(numeric, ROSTERS, date(2024, 1, 3), [3])
    expected, _ = build_pair_features(chronological, ROSTERS, date(2024, 1, 3), [3])
    pd.testing.assert_frame_equal(actual, expected)


def test_excluded_days_do_not_require_usable_outcomes_or_identities():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01')]
    expected, _ = build_pair_features(history, ROSTERS, date(2024, 1, 2), [1])
    actual, audit = build_pair_features(
        history + [{'date': '2024-01-02', 'draw': object()}, {'date': '2024-01-03', 'games': None}],
        ROSTERS, date(2024, 1, 2), [1],
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert audit['exclusions']['current_day'] == 1
    assert audit['exclusions']['future_day'] == 1
    assert pd.Timestamp(actual.iloc[0].feature_history_max_at) == pd.Timestamp('2024-01-02T00:00:00Z')


def test_reversed_game_sides_preserve_team_aligned_results_and_history():
    from scripts.prospective_sports_features import build_pair_features
    original = [match(1, '2024-01-01'), match(2, '2024-01-02', winner=False)]
    swapped = deepcopy(original)
    for item in swapped:
        game = item['games'][0]
        for field in ('id', 'win', 'players', 'stats'):
            game[f't1_{field}'], game[f't2_{field}'] = game[f't2_{field}'], game[f't1_{field}']
    expected, _ = build_pair_features(original, ROSTERS, date(2024, 1, 3), [3])
    actual, _ = build_pair_features(swapped, ROSTERS, date(2024, 1, 3), [3])
    pd.testing.assert_frame_equal(actual, expected)


def test_substitute_receives_only_games_actually_played():
    from scripts.prospective_sports_features import build_pair_features
    series = match(1, '2024-01-01')
    series.update(best_of=3, t1_score=2, t2_score=1)
    series['players_1'] = ROSTERS['a']  # Legacy top-level roster is not authoritative.
    loss = match(2, '2024-01-01', winner=False)['games'][0]
    win = match(3, '2024-01-01')['games'][0]
    loss['t1_players']['0']['player_id'] = 'substitute'
    win['t1_players']['0']['player_id'] = 'substitute'
    series['games'].extend([loss, win])
    announced = deepcopy(ROSTERS)
    announced['a'][0] = 'substitute'
    actual, audit = build_pair_features([series], announced, date(2024, 1, 2), [3])
    assert audit['player_history_counts']['substitute']['games'] == 2
    assert audit['player_history_counts']['0']['games'] == 1
    assert actual.iloc[0].t1_rolling_win_rate == pytest.approx(2 / 3)
    absent = deepcopy(series)
    for game in absent['games'][1:]:
        game['t1_players']['0']['player_id'] = '0'
    with pytest.raises(ValueError):
        build_pair_features([absent], announced, date(2024, 1, 2), [3])


def _third_team_match(mid, day):
    item = match(mid, day)
    item['t2_id'] = 'c'
    item['games'][0]['t2_id'] = 'c'
    for index, player in enumerate(item['games'][0]['t2_players'].values()):
        player['player_id'] = str(index + 10)
    return item


def test_all_pairs_and_formats_share_one_decayed_team_state():
    import numpy as np
    from scripts.prospective_sports_features import build_pair_features
    from src.models.symmetric_series import _REQUIRED_BASE_FIELDS
    announced = {**ROSTERS, 'c': [str(i) for i in range(10, 15)]}
    history = [match(1, '2024-01-01'), _third_team_match(2, '2024-01-02')]
    actual, _ = build_pair_features(history, announced, date(2024, 2, 1), [5, 1, 3, 3])
    assert set(actual) == _REQUIRED_BASE_FIELDS | {'team_a', 'team_b', 'best_of', 'feature_history_max_at'}
    assert list(actual[['team_a', 'team_b', 'best_of']].itertuples(index=False, name=None)) == [
        (a, b, bo) for a, b in [('a', 'b'), ('a', 'c'), ('b', 'c')] for bo in [1, 3, 5]
    ]
    assert np.isfinite(actual[sorted(_REQUIRED_BASE_FIELDS)].to_numpy()).all()
    for team in announced:
        states = []
        for _, row in actual.iterrows():
            side = 1 if row.team_a == team else 2 if row.team_b == team else None
            if side:
                states.append([row[f'team_gl_rd{side}'], row[f'player_gl_rd_avg{side}'], row[f'days_since_last_{side}']])
        assert all(state == states[0] for state in states)
    isolated, _ = build_pair_features(history, ROSTERS, date(2024, 2, 1), [3])
    row = actual[(actual.team_a == 'a') & (actual.team_b == 'b') & (actual.best_of == 3)]
    pd.testing.assert_frame_equal(row.reset_index(drop=True), isolated)


def test_rating_probabilities_complement_when_lexical_team_order_reverses():
    from scripts.prospective_sports_features import build_pair_features
    from src.models.symmetric_series import _PROBABILITY_FIELDS
    history = [match(1, '2024-01-01'), _third_team_match(2, '2024-01-04')]
    original, _ = build_pair_features(history, ROSTERS, date(2024, 2, 1), [3])
    relabeled = deepcopy(history)
    for item in relabeled:
        item['t1_id'] = 'z'
        for game in item['games']:
            game['t1_id'] = 'z'
    reversed_pair, _ = build_pair_features(
        relabeled, {'z': ROSTERS['a'], 'b': ROSTERS['b']}, date(2024, 2, 1), [3],
    )
    for field in _PROBABILITY_FIELDS:
        assert 0 <= original.iloc[0][field] <= 1
        assert original.iloc[0][field] + reversed_pair.iloc[0][field] == pytest.approx(1, abs=1e-12)


def test_missing_history_only_stops_blocking_after_twenty_newer_games():
    from datetime import timedelta
    from scripts.prospective_sports_features import build_pair_features
    history = [match(i + 1, (date(2024, 1, 1) + timedelta(days=i)).isoformat()) for i in range(21)]
    history[0]['games'][0]['t1_stats']['gold'] = None
    with pytest.raises(ValueError):
        build_pair_features(history[:20], ROSTERS, date(2024, 2, 1), [3])
    actual, _ = build_pair_features(history, ROSTERS, date(2024, 2, 1), [3])
    assert actual.iloc[0].t1_rolling_gold == 50000


@pytest.mark.parametrize('duplicate_kind', ['match', 'game'])
def test_duplicate_prior_records_cannot_double_count_ratings(duplicate_kind):
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01'), match(2, '2024-01-02')]
    if duplicate_kind == 'match':
        history[1]['match_id'] = history[0]['match_id']
    else:
        history[1]['games'][0]['game_id'] = history[0]['games'][0]['game_id']
    with pytest.raises(ValueError):
        build_pair_features(history, ROSTERS, date(2024, 1, 3), [3])


def test_player_display_name_cannot_replace_missing_historical_id():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01')]
    player = history[0]['games'][0]['t1_players']['0']
    player['player_name'] = player.pop('player_id')
    with pytest.raises(ValueError):
        build_pair_features(history, ROSTERS, date(2024, 1, 2), [3])


def test_unfinished_prior_series_cannot_initialize_announced_rosters():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2024-01-01')]
    history[0]['best_of'] = 3
    with pytest.raises(ValueError):
        build_pair_features(history, ROSTERS, date(2024, 1, 2), [3])


def test_nested_future_game_cannot_enter_a_prior_dated_series():
    from scripts.prospective_sports_features import build_pair_features
    earlier = match(1, '2024-01-01')
    later = match(2, '2024-01-02', kills=999)
    later['games'][0]['date'] = '2024-01-03'
    later['draw'] = 'not_yet_known'
    expected, _ = build_pair_features([earlier], ROSTERS, date(2024, 1, 3), [3])
    actual, audit = build_pair_features([earlier, later], ROSTERS, date(2024, 1, 3), [3])
    pd.testing.assert_frame_equal(actual, expected)
    assert audit['used_series'] == 1




def test_missing_duplicate_match_ids_preserve_exact_game_identity_features():
    from scripts.prospective_sports_features import build_pair_features
    original = match(1, '2024-01-01')
    missing = deepcopy(original)
    del missing['t1_id'], missing['t2_id']
    expected, _ = build_pair_features([original], ROSTERS, date(2024, 1, 2), [3])
    actual, audit = build_pair_features([missing], ROSTERS, date(2024, 1, 2), [3])
    pd.testing.assert_frame_equal(actual, expected)
    assert audit['normalization']['series_with_game_derived_team_ids'] == 1
    assert 't1_id' not in missing and 't2_id' not in missing


def test_one_missing_match_id_uses_known_opponent_not_game_position():
    from scripts.prospective_sports_features import build_pair_features
    original = match(1, '2024-01-01')
    missing = deepcopy(original)
    del missing['t1_id']
    game = missing['games'][0]
    for field in ('id', 'win', 'players', 'stats'):
        game[f't1_{field}'], game[f't2_{field}'] = game[f't2_{field}'], game[f't1_{field}']
    expected, _ = build_pair_features([original], ROSTERS, date(2024, 1, 2), [3])
    actual, _ = build_pair_features([missing], ROSTERS, date(2024, 1, 2), [3])
    pd.testing.assert_frame_equal(actual, expected)


def test_later_game_identity_cannot_be_hidden_by_first_game_resolution():
    from scripts.prospective_sports_features import build_pair_features
    item = match(1, '2024-01-01')
    del item['t1_id'], item['t2_id']
    item['best_of'] = 3
    item['t1_score'] = 2
    later = deepcopy(item['games'][0])
    later.update(game_id='2', t2_id='unrelated-exact-id')
    item['games'].append(later)
    with pytest.raises(ValueError):
        build_pair_features([item], ROSTERS, date(2024, 1, 2), [3])


def test_conflicting_explicit_match_id_aliases_are_not_precedence_resolved():
    from scripts.prospective_sports_features import build_pair_features
    item = match(1, '2024-01-01')
    item['tid_1'] = 'contradictory-team'
    with pytest.raises(ValueError):
        build_pair_features([item], ROSTERS, date(2024, 1, 2), [3])


def test_unoriented_header_cannot_reverse_exact_game_results():
    from scripts.prospective_sports_features import build_pair_features
    original = match(1, '2024-01-01')
    item = deepcopy(original)
    del item['t1_id'], item['t2_id']
    item.update(t1_score=0, t2_score=1, t1_win=False, t2_win=True)
    expected, _ = build_pair_features([original], ROSTERS, date(2024, 1, 2), [3])
    actual, _ = build_pair_features([item], ROSTERS, date(2024, 1, 2), [3])
    pd.testing.assert_frame_equal(actual, expected)


def test_missing_duplicate_ids_do_not_authorize_impossible_score_totals():
    from scripts.prospective_sports_features import build_pair_features
    item = match(1, '2024-01-01')
    del item['t1_id'], item['t2_id']
    item['t1_score'], item['t2_score'] = 0, 0
    with pytest.raises(ValueError):
        build_pair_features([item], ROSTERS, date(2024, 1, 2), [3])


def test_explicit_header_identity_makes_reversed_score_a_contradiction():
    from scripts.prospective_sports_features import build_pair_features
    item = match(1, '2024-01-01')
    item.update(t1_score=0, t2_score=1, t1_win=False, t2_win=True)
    with pytest.raises(ValueError):
        build_pair_features([item], ROSTERS, date(2024, 1, 2), [3])


def test_draw_flag_cannot_hide_duplicate_opponents_before_rating_replay():
    from scripts.prospective_sports_features import build_pair_features
    valid = match(1, '2024-01-01')
    corrupt = match(2, '2024-01-02')
    del corrupt['t1_id'], corrupt['t2_id']
    corrupt['draw'] = True
    corrupt['games'][0]['t2_id'] = corrupt['games'][0]['t1_id']
    with pytest.raises(ValueError):
        build_pair_features([valid, corrupt], ROSTERS, date(2024, 1, 3), [3])


def test_draw_flag_cannot_hide_missing_declared_games():
    from scripts.prospective_sports_features import build_pair_features
    valid = match(1, '2024-01-01')
    corrupt = match(2, '2024-01-02')
    corrupt.update(draw=True, best_of=4, games_played=4, t1_score=2, t2_score=2, t1_win=False, t2_win=False)
    second = deepcopy(corrupt['games'][0])
    second.update(game_id='3', t1_win=False, t2_win=True)
    corrupt['games'].append(second)
    with pytest.raises(ValueError):
        build_pair_features([valid, corrupt], ROSTERS, date(2024, 1, 3), [3])


def test_compact_series_supports_weighted_bo5_initial_wins():
    from scripts.prospective_sports_features import _compact_series, _history_helpers
    item = match(1, '2024-01-01')
    second = deepcopy(item['games'][0])
    second['game_id'] = '2'
    item['games'].append(second)
    item.update(best_of=5, t1_score=2, t2_score=0, t1_win=True, t2_win=False, initial_wins={'a': 1})
    helpers = _history_helpers()
    series = _compact_series(item, (1,), helpers, set(), [date(2024, 1, 1), date(2024, 1, 1)])
    assert len(series.games) == 2


def test_compact_series_rejects_extra_map_after_weighted_bo5_clincher():
    from scripts.prospective_sports_features import _compact_series, _history_helpers
    item = match(1, '2024-01-01')
    for gid in ('2', '3'):
        g = deepcopy(item['games'][0])
        g['game_id'] = gid
        item['games'].append(g)
    item.update(best_of=5, t1_score=2, t2_score=0, t1_win=True, t2_win=False, initial_wins={'a': 1})
    helpers = _history_helpers()
    with pytest.raises(ValueError, match="historical series contains a game after its deciding win"):
        _compact_series(item, (1,), helpers, set(), [date(2024, 1, 1)] * 3)


def test_training_and_all_pair_export_share_prior_rosters_decay_and_w20():
    import numpy as np
    from scripts.prospective_sports_features import build_pair_features, build_chronological_training_rows
    from scripts.train_and_tune_siamese_series import build_training_features
    from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

    history = [match(1, '2023-01-01'), match(2, '2023-01-02', winner=False)]
    targets = [match(10, '2023-02-10'), match(11, '2023-02-10', winner=False)]
    # A realized debut on the target day must not enter the prediction roster.
    targets[0]['games'][0]['t1_players']['0']['player_id'] = 'target-debut'
    trained, audit = build_chronological_training_rows(history + targets)
    exported, _ = build_pair_features(history + targets, ROSTERS, date(2023, 2, 10), [1])
    fields = sorted(_REQUIRED_BASE_FIELDS)
    for mid in ('10', '11'):
        row = trained.loc[trained.golgg_match_id == mid]
        np.testing.assert_array_equal(row[fields].to_numpy(), exported[fields].to_numpy())
        x_train, names = build_training_features(row)
        x_pair, pair_names = build_training_features(exported.assign(BoN=1))
        assert names == pair_names
        np.testing.assert_array_equal(x_train, x_pair)
    assert audit['roster_policy'] == 'previous_observed_roster_proxy'


def test_training_snapshot_is_invariant_to_target_roster_result_and_later_history():
    from scripts.prospective_sports_features import build_chronological_training_rows
    from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

    history = [match(1, '2023-01-01'), match(2, '2023-01-09')]
    original, _ = build_chronological_training_rows(history)
    altered = deepcopy(history)
    altered[1] = match(2, '2023-01-09', winner=False, kills=900)
    altered[1]['games'][0]['t1_players']['0']['player_id'] = 'future-player'
    altered.append(match(3, '2023-02-01', winner=False))
    changed, _ = build_chronological_training_rows(altered)
    fields = sorted(_REQUIRED_BASE_FIELDS)
    pd.testing.assert_frame_equal(
        original.loc[original.golgg_match_id == '2', fields].reset_index(drop=True),
        changed.loc[changed.golgg_match_id == '2', fields].reset_index(drop=True),
    )


def test_roster_transfer_changes_player_signal_without_copying_team_probability():
    from scripts.prospective_sports_features import build_pair_features
    history = [match(1, '2023-01-01'), _third_team_match(2, '2023-01-02')]
    ordinary, _ = build_pair_features(history, ROSTERS, date(2023, 2, 1), [3])
    transferred = deepcopy(ROSTERS)
    transferred['a'][0] = '10'
    changed, _ = build_pair_features(history, transferred, date(2023, 2, 1), [3])
    assert changed.iloc[0].team_elo == ordinary.iloc[0].team_elo
    assert changed.iloc[0].player_elo != ordinary.iloc[0].player_elo


def test_valid_draw_updates_history_in_both_training_and_export():
    from scripts.prospective_sports_features import build_pair_features, build_chronological_training_rows
    tied = match(1, '2023-01-01', kills=20)
    tied.update(best_of=2, draw=True, t1_score=1, t2_score=1, t1_win=False, t2_win=False)
    tied['games'].append(match(2, '2023-01-01', winner=False, kills=40)['games'][0])
    future = match(3, '2023-01-02')
    trained, audit = build_chronological_training_rows([tied, future])
    exported, _ = build_pair_features([tied], ROSTERS, date(2023, 1, 2), [1])
    assert trained.golgg_match_id.tolist() == ['3']
    assert trained.iloc[0].t1_rolling_kills == exported.iloc[0].t1_rolling_kills == 30
    assert audit['state_only_series'] == 1


def test_overnight_series_is_released_only_after_completion_day():
    from scripts.prospective_sports_features import build_pair_features, build_chronological_training_rows
    prior = match(1, '2023-01-01')
    overnight = match(2, '2023-01-02', kills=100)
    overnight.update(best_of=3, t1_score=2)
    last = match(3, '2023-01-04', kills=200)['games'][0]
    last['date'] = '2023-01-04'
    overnight['games'].append(last)
    during, after = match(4, '2023-01-03', winner=False), match(5, '2023-01-05')
    rows, _ = build_chronological_training_rows([prior, overnight, during, after])
    fields = ['team_elo', 'player_gl', 't1_rolling_kills', 'days_since_last_1']
    for target in (during, after):
        pair, _ = build_pair_features([prior, overnight, during, after], ROSTERS,
                                      date.fromisoformat(target['date']), [1])
        actual = rows.loc[rows.golgg_match_id == target['match_id'], fields].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, pair[fields])
    assert rows.loc[rows.golgg_match_id == '4', 't1_rolling_kills'].item() == 10
