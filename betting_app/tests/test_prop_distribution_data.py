"""Missingness and units must survive standalone prop source extraction."""
import math
import numpy as np

from src.features.prop_distribution_data import extract_map_labels, symmetric_features


def test_zero_kills_is_valid_and_missing_duration_stays_missing():
    labels = extract_map_labels({'t1_stats': {'kills': 0}, 't2_stats': {'kills': 3}, 'game_duration': None})
    assert labels['kills'] == 3
    assert math.isnan(labels['duration_minutes'])


def test_duration_seconds_are_not_mislabeled_minutes():
    labels = extract_map_labels({'t1_stats': {'kills': None}, 't2_stats': {'kills': 3}, 'game_duration': 1800})
    assert math.isnan(labels['kills'])
    assert labels['duration_minutes'] == 30.


def test_symmetric_summaries_preserve_shared_pace_and_side_swap():
    players = np.arange(2 * 2 * 5 * 3, dtype=float).reshape(2, 2, 5, 3)
    teams = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    context = np.array([[1., 0., 0.], [0., 1., 0.]])
    baseline = np.array([[.3, .2], [.7, .8]])
    features, names = symmetric_features(players, teams, context, baseline)
    swapped, swapped_names = symmetric_features(players[:, ::-1], teams[:, ::-1], context, 1. - baseline)
    assert names == swapped_names
    assert np.allclose(features, swapped, equal_nan=True)
    changed, _ = symmetric_features(players + 10., teams, context, baseline)
    assert not np.allclose(features, changed, equal_nan=True)


def test_invalid_counts_and_nonpositive_duration_are_not_imputed():
    labels = extract_map_labels({'t1_stats': {'kills': -1}, 't2_stats': {'kills': 3}, 'game_duration': 0})
    assert math.isnan(labels['kills'])
    assert labels['kills_reason'] == 't1_kills_invalid_count'
    assert math.isnan(labels['duration_minutes'])
    assert labels['duration_reason'] == 'duration_nonpositive'


def test_missingness_features_preserve_side_symmetry():
    players = np.ones((1, 2, 5, 2))
    players[0, 0, :, 0] = np.nan
    teams = np.ones((1, 2, 2))
    teams[0, 1, 1] = np.nan
    a, names = symmetric_features(players, teams, np.array([[1., 0., 0.]]), np.array([[.2, .7]]))
    b, _ = symmetric_features(players[:, ::-1], teams[:, ::-1], np.array([[1., 0., 0.]]), np.array([[.8, .3]]))
    assert any('missing' in name for name in names)
    np.testing.assert_allclose(a, b, equal_nan=True)


def _source_series(identifier, day, *, best_of=1, kills=(10, 5), duration=1800, teams=('A', 'B')):
    roles = ('TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT')
    game = {
        'game_id': 'g' + identifier, 'match_id': identifier, 'date': day,
        't1_id': teams[0], 't2_id': teams[1], 't1_win': True, 't2_win': False,
        't1_players': {role: {'player_id': 'a' + str(i), 'name': 'same display name'} for i, role in enumerate(roles)},
        't2_players': {role: {'player_id': 'b' + str(i), 'name': 'same display name'} for i, role in enumerate(roles)},
        't1_stats': {'kills': kills[0]}, 't2_stats': {'kills': kills[1]},
        'game_duration': duration,
    }
    return {'match_id': identifier, 'date': day, 'best_of': best_of, 't1_id': teams[0],
            't2_id': teams[1], 'games': [game], 'games_played': 1}


def _write_bank(tmp_path, rows, releases=None):
    import hashlib
    import json
    import pandas as pd
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(rows))
    bank = tmp_path / 'bank'
    bank.mkdir()
    releases = releases or {}
    metadata = pd.DataFrame([
        {'match_id': r['match_id'], 'date': r['date'], 'release_day': releases.get(r['match_id'], r['date']),
         'best_of': r['best_of'], 'team_a': 'A', 'team_b': 'B',
         'roster_a': ['a' + str(i) for i in range(5)], 'roster_b': ['b' + str(i) for i in range(5)],
         'tournament': 'fixture', 'max_history_release_day': None}
        for r in rows if r['best_of'] in (1, 3, 5)])
    metadata.to_parquet(bank / 'metadata.parquet', index=False)
    n = len(metadata)
    np.savez(bank / 'arrays.npz', players=np.zeros((n, 2, 5, 1)), teams=np.zeros((n, 2, 1)),
             context=np.column_stack([metadata.best_of.eq(b).to_numpy(float) for b in (1, 3, 5)]),
             baseline_series=np.full((n, 2), .5))
    bank.joinpath('audit.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                                    'target_rows': n}))
    return source, bank


def test_future_label_perturbation_does_not_change_earlier_pace(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01'), _source_series('2', '2024-01-02'),
            _source_series('3', '2024-01-03')]
    source, bank = _write_bank(tmp_path, rows)
    first = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'out1')['dataset_path'])
    rows[-1]['games'][0]['t1_stats']['kills'] = 90
    rows[-1]['games'][0]['game_duration'] = 4800
    other = tmp_path / 'other'
    other.mkdir()
    source2, bank2 = _write_bank(other, rows)
    second = pd.read_parquet(build_distribution_dataset(source2, bank2, tmp_path / 'out2')['dataset_path'])
    features = [c for c in first if c.startswith('x_')]
    np.testing.assert_allclose(first[features], second[features], equal_nan=True)
    assert first.iloc[2].kills != second.iloc[2].kills


def test_same_day_and_delayed_releases_are_withheld_and_bo2_is_history(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01', best_of=2),
            _source_series('2', '2024-01-02', kills=(20, 10)),
            _source_series('3', '2024-01-03'), _source_series('4', '2024-01-04')]
    source, bank = _write_bank(tmp_path, rows, {'2': '2024-01-03'})
    frame = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'out')['dataset_path'])
    row = frame.set_index('match_id')
    assert row.loc['2', 'x_pace_player_kills_for_mean_level'] == 7.5
    assert row.loc['3', 'x_pace_player_kills_for_mean_level'] == 7.5
    assert row.loc['3', 'max_pace_release_day'] == '2024-01-01'
    assert row.loc['4', 'max_pace_release_day'] == '2024-01-03'
    assert row.loc['3', 'x_pace_player_kills_for_count_mean_level'] == 1.


def test_team_side_is_aligned_by_id_not_name_and_rosters_must_match(tmp_path):
    import pandas as pd
    import pytest
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01', teams=('B', 'A'), kills=(4, 20)),
            _source_series('2', '2024-01-02')]
    # These rosters are deliberately attached to the opposite blue/red side.
    game = rows[0]['games'][0]
    game['t1_players'], game['t2_players'] = game['t2_players'], game['t1_players']
    source, bank = _write_bank(tmp_path, rows)
    frame = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'out')['dataset_path'])
    assert frame.iloc[1]['x_pace_player_kills_for_mean_level'] == 12
    assert frame.iloc[1]['x_pace_player_kills_for_mean_gap'] == 16
    metadata = pd.read_parquet(bank / 'metadata.parquet')
    metadata.at[0, 'roster_a'] = ['wrong'] * 5
    metadata.to_parquet(bank / 'metadata.parquet', index=False)
    with pytest.raises(ValueError, match='roster'):
        build_distribution_dataset(source, bank, tmp_path / 'bad')


def test_missing_side_stats_are_skipped_without_zero_history(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01', kills=(None, 5)), _source_series('2', '2024-01-02')]
    source, bank = _write_bank(tmp_path, rows)
    frame = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'out')['dataset_path'])
    assert math.isnan(frame.iloc[0].kills)
    assert frame.iloc[0].kills_reason == 't1_kills_missing'
    assert frame.iloc[1]['x_pace_player_kills_for_count_mean_level'] == .5
    assert math.isnan(frame.iloc[1]['x_pace_player_total_kills_mean_level'])


def test_duplicate_source_ids_and_changed_source_hash_reject(tmp_path):
    import json
    import pytest
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01')]
    source, bank = _write_bank(tmp_path, rows)
    source.write_text(json.dumps(rows + rows))
    with pytest.raises(ValueError, match='source.*hash'):
        build_distribution_dataset(source, bank, tmp_path / 'bad')
    import hashlib
    audit = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'target_rows': 1}
    (bank / 'audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='duplicate.*match'):
        build_distribution_dataset(source, bank, tmp_path / 'duplicate')


def test_extreme_valid_labels_are_not_clipped_and_remakes_are_absent():
    game = _source_series('1', '2024-01-01', kills=(90, 80), duration=7200)['games'][0]
    labels = extract_map_labels(game)
    assert labels['kills'] == 170
    assert labels['duration_minutes'] == 120
    game['remake'] = True
    labels = extract_map_labels(game)
    assert math.isnan(labels['kills'])
    assert math.isnan(labels['duration_minutes'])
    assert labels['kills_reason'] == 'map_excluded:remake'


def test_all_canonical_rows_survive_source_absence_and_illegal_lengths(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01', best_of=3)]
    source, bank = _write_bank(tmp_path, rows)
    metadata = pd.read_parquet(bank / 'metadata.parquet')
    missing = metadata.iloc[0].copy()
    missing['match_id'] = 'absent'
    metadata = pd.concat([metadata, missing.to_frame().T], ignore_index=True)
    metadata['best_of'] = metadata.best_of.astype(int)
    metadata.to_parquet(bank / 'metadata.parquet', index=False)
    import json
    audit = json.loads((bank / 'audit.json').read_text())
    audit['target_rows'] = 2
    (bank / 'audit.json').write_text(json.dumps(audit))
    np.savez(bank / 'arrays.npz', players=np.zeros((2, 2, 5, 1)), teams=np.zeros((2, 2, 1)),
             context=np.tile([0., 1., 0.], (2, 1)), baseline_series=np.full((2, 2), .5))
    receipt = build_distribution_dataset(source, bank, tmp_path / 'out')
    frame = pd.read_parquet(receipt['dataset_path'])
    assert frame.match_id.tolist() == ['1', 'absent']
    assert math.isnan(frame.iloc[0].maps_played)
    assert frame.iloc[0].maps_reason == 'illegal_terminal_map_count'
    assert math.isnan(frame.iloc[1].kills)
    assert frame.iloc[1].kills_reason == 'source_series_absent'
    assert receipt['audit']['source_absent_rows'] == 1


def test_equal_display_names_do_not_merge_player_history(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01'), _source_series('2', '2024-01-02')]
    for index, player in enumerate(rows[1]['games'][0]['t1_players'].values()):
        player['player_id'] = 'new' + str(index)
    source, bank = _write_bank(tmp_path, rows)
    metadata = pd.read_parquet(bank / 'metadata.parquet')
    metadata.at[1, 'roster_a'] = ['new' + str(i) for i in range(5)]
    metadata.to_parquet(bank / 'metadata.parquet', index=False)
    frame = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'out')['dataset_path'])
    assert math.isnan(frame.iloc[1]['x_pace_player_kills_for_mean_level'])
    assert frame.iloc[1]['x_pace_player_kills_for_count_mean_level'] == .5


def test_explicit_zero_map_advantages_do_not_erase_completed_series(tmp_path):
    import pandas as pd
    from src.features.prop_distribution_data import build_distribution_dataset
    rows = [_source_series('1', '2024-01-01')]
    rows[0]['map_advantage'] = {'A': 0, 'B': 0}
    source, bank = _write_bank(tmp_path, rows)
    frame = pd.read_parquet(build_distribution_dataset(source, bank, tmp_path / 'zero_advantage')['dataset_path'])
    assert frame.iloc[0]['maps_played'] == 1
    assert frame.iloc[0]['maps_reason'] == 'valid'
