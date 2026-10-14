"""Chronology and identity invariants for the actual-composition diagnostic."""
import copy
import numpy as np
import pandas as pd
import pytest

from src.features.draft_prop_data import prepare_draft_data


def _map(mid, day, champs, release=None):
    row = dict(match_id=mid, game_id=mid, map_index=1, date=day, map_date=day,
               release_day=release or day, tournament='Synthetic phase', best_of=3,
               t1_id='A', t2_id='B', t1_side='blue', t2_side='red',
               canonical=True, kills=30., duration_minutes=32., draft_valid=True,
               draft_absence=None, champion_repeat_from_prior=False)
    for team, offset in [('t1', 0), ('t2', 5)]:
        for role in range(5):
            row[f'{team}_player_{role}'] = f'p{offset + role}'
            row[f'{team}_champion_{role}'] = str(champs[offset + role])
    return row


def _base(mid, day, release=None):
    return dict(match_id=mid, date=day, release_day=release or day, best_of=3,
                team_a='A', team_b='B', roster_a=np.array([f'p{i}' for i in range(5)]),
                roster_b=np.array([f'p{i}' for i in range(5, 10)]),
                tournament='Synthetic phase', kills=30., duration_minutes=32.,
                maps_played=2., x_signal=1.)


def test_same_day_and_corrected_release_do_not_enter_champion_history():
    maps = pd.DataFrame([_map('old', '2025-06-01', range(10)),
                         _map('held', '2025-06-02', range(10)),
                         _map('target', '2025-06-03', range(10))])
    base = pd.DataFrame([_base('old', '2025-06-01'),
                         _base('held', '2025-06-02', '2025-06-03'),
                         _base('target', '2025-06-03')])
    frame, schema, audit = prepare_draft_data(maps, base)
    target = frame.loc[frame.match_id == 'target'].iloc[0]
    assert target['d_player_logcount_a_0'] == np.log1p(1)
    assert target['draft_history_max_release_day'] == '2025-06-01'


def test_assessment_future_mutation_cannot_change_earlier_features():
    maps = pd.DataFrame([_map('train', '2025-06-01', range(10)),
                         _map('early', '2026-01-01', range(10)),
                         _map('future', '2026-01-02', range(10))])
    base = pd.DataFrame([_base('train', '2025-06-01'), _base('early', '2026-01-01'),
                         _base('future', '2026-01-02')])
    original, schema, _ = prepare_draft_data(maps, base)
    changed = maps.copy(deep=True)
    changed.loc[changed.match_id == 'future', ['kills', 'duration_minutes']] = [200., 120.]
    changed.loc[changed.match_id == 'future', 't1_champion_0'] = '999'
    replay, replay_schema, _ = prepare_draft_data(changed, base)
    columns = schema['shared_features'] + schema['draft_features']
    np.testing.assert_array_equal(original.loc[original.match_id == 'early', columns].to_numpy(),
                                  replay.loc[replay.match_id == 'early', columns].to_numpy())
    assert schema['champion_vocabulary'] == replay_schema['champion_vocabulary']


def test_cold_features_and_schema_preserve_input_order_and_objects():
    maps = pd.DataFrame([_map('later', '2025-06-02', range(10)),
                         _map('cold', '2025-06-01', range(10))])
    base = pd.DataFrame([_base('later', '2025-06-02'),
                         _base('cold', '2025-06-01')], index=[8, 3])
    before_maps, before_base = copy.deepcopy(maps), copy.deepcopy(base)
    frame, schema, audit = prepare_draft_data(maps, base)
    pd.testing.assert_frame_equal(maps, before_maps)
    pd.testing.assert_frame_equal(base, before_base)
    cold = frame.loc[frame.match_id == 'cold'].iloc[0]
    assert pd.isna(cold.p_global_kills_mean)
    assert pd.isna(cold.d_kills_mean_a_0)
    assert cold.d_player_logcount_a_0 == cold.d_role_frequency_a_0 == 0
    assert cold.draft_player_champ_min_count == 0
    assert cold.draft_patch == 'UNKNOWN'
    assert cold.p_patch_missing == 1
    assert pd.isna(cold.p_patch_major)


def test_team_and_player_set_alignment_preserves_played_roles_not_sorted_slots():
    source = _map('target', '2025-06-01', range(10))
    source['t1_id'], source['t2_id'] = 'B', 'A'
    for role in range(5):
        source[f't1_player_{role}'] = f'p{9-role}'
        source[f't2_player_{role}'] = f'p{4-role}'
    base = pd.DataFrame([_base('target', '2025-06-01')])
    frame, schema, _ = prepare_draft_data(pd.DataFrame([source]), base)
    assert frame.iloc[0].draft_valid
    for role in range(5):
        assert frame.iloc[0][f'd_champion_a_{role}'] == schema['champion_vocabulary'][str(5+role)]
        assert frame.iloc[0][f'd_champion_b_{role}'] == schema['champion_vocabulary'][str(role)]


def test_player_champion_experience_crosses_roles_but_role_history_does_not():
    old = _map('old', '2025-06-01', range(10))
    target = _map('target', '2025-06-02', range(10))
    for kind in ('player', 'champion'):
        target[f't1_{kind}_0'], target[f't1_{kind}_1'] = (
            target[f't1_{kind}_1'], target[f't1_{kind}_0'])
    frame, _, _ = prepare_draft_data(pd.DataFrame([old, target]),
                                    pd.DataFrame([_base('target', '2025-06-02')]))
    row = frame.iloc[0]
    assert row.d_player_logcount_a_0 == np.log1p(1)
    assert row.d_role_logcount_a_0 == 0
    assert row.d_role_frequency_a_0 == 0
    assert row.d_kills_mean_a_0 == 30
    assert row.d_role_frequency_a_2 == .5


def test_whole_series_release_withholds_early_map_and_keeps_source_only_bo2():
    first = _map('bo2', '2025-06-01', range(10), '2025-06-01')
    first['best_of'] = 2
    second = _map('bo2', '2025-06-01', range(10), '2025-06-03')
    second.update(game_id='bo2-second', map_index=2, map_date='2025-06-03', best_of=2)
    maps = pd.DataFrame([second, _map('early', '2025-06-02', range(10)),
                         first, _map('late', '2025-06-04', range(10))])
    frame, _, audit = prepare_draft_data(maps, pd.DataFrame([
        _base('early', '2025-06-02'), _base('late', '2025-06-04')]))
    early, late = frame.iloc[0], frame.iloc[1]
    assert early.d_player_logcount_a_0 == 0
    assert late.d_player_logcount_a_0 == np.log1p(3)
    assert late.draft_history_max_release_day == '2025-06-03'
    assert audit['history_source_coverage']['source_only_maps'] == 2
    assert audit['history_source_coverage']['bo2_maps'] == 2


def test_train_codebook_requires_closed_valid_target_not_history_or_future():
    maps = pd.DataFrame([
        _map('source-only', '2025-06-01', range(100, 110)),
        _map('train', '2025-06-02', range(10)),
        _map('unreleased', '2025-06-30', range(20, 30), '2025-07-01'),
        _map('cal', '2025-07-01', range(30, 40)),
        _map('assess', '2026-01-01', range(40, 50))])
    base = pd.DataFrame([
        _base('train', '2025-06-02'),
        _base('unreleased', '2025-06-30', '2025-07-01'),
        _base('cal', '2025-07-01'), _base('assess', '2026-01-01')])
    frame, schema, audit = prepare_draft_data(maps, base)
    assert set(schema['champion_vocabulary']) == {str(i) for i in range(10)}
    assert (frame.loc[frame.match_id != 'train', schema['categorical_features']] == -1).all().all()
    assert audit['partition_valid_counts'] == {'TRAIN': 1, 'CAL': 1, 'ASSESS': 1}
    assert audit['unknown_champions']['CAL']['slots'] == 10


@pytest.mark.parametrize('corruption', ['duplicate_champion', 'missing_champion',
                                      'duplicate_player', 'overlap', 'roster', 'team'])
def test_invalid_compositions_are_retained_without_history_or_codebook_entries(corruption):
    bad = _map('bad', '2025-06-01', range(10))
    if corruption == 'duplicate_champion':
        bad['t2_champion_4'] = bad['t1_champion_0']
    elif corruption == 'missing_champion':
        bad['t1_champion_0'] = None
    elif corruption == 'duplicate_player':
        bad['t1_player_1'] = bad['t1_player_0']
    elif corruption == 'overlap':
        bad['t2_player_0'] = bad['t1_player_0']
    elif corruption == 'roster':
        bad['t1_player_0'] = 'other-player'
    else:
        bad['t1_id'] = 'other-team'
    # A roster mismatch is a target-composition failure; genuinely valid played
    # roles still remain source history, unlike intrinsically invalid rosters.
    maps = pd.DataFrame([bad, _map('next', '2025-06-02', range(10, 20))])
    frame, schema, audit = prepare_draft_data(maps, pd.DataFrame([
        _base('bad', '2025-06-01'), _base('next', '2025-06-02')]))
    assert len(frame) == 2
    assert not frame.iloc[0].draft_valid
    assert isinstance(frame.iloc[0].draft_absence, str)
    assert (frame.iloc[0][schema['categorical_features']] == -1).all()
    assert frame.iloc[1].p_global_kills_mean == 30
    assert frame.iloc[1].d_player_logcount_a_0 == 0
    assert set(schema['champion_vocabulary']) == {str(i) for i in range(10, 20)}
    assert audit['partition_absence_counts']['TRAIN'] == 1


def test_missing_source_and_declared_absence_are_not_dropped():
    invalid = _map('invalid', '2025-06-02', range(10))
    invalid.update(draft_valid=False, draft_absence='missing role evidence')
    frame, _, audit = prepare_draft_data(pd.DataFrame([invalid]), pd.DataFrame([
        _base('missing', '2025-06-01'), _base('invalid', '2025-06-02')]))
    assert frame.match_id.tolist() == ['missing', 'invalid']
    assert not frame.draft_valid.any()
    assert frame.draft_absence.tolist() == ['source_first_map_absent', 'missing role evidence']
    assert audit['partition_absence_counts']['TRAIN'] == 2


def test_role_means_shrink_observed_labels_toward_prior_global_not_current_labels():
    old = _map('old', '2025-06-01', range(10))
    invalid = _map('invalid', '2025-06-02', range(10))
    invalid.update(kills=70., duration_minutes=52., draft_valid=False,
                   draft_absence='invalid composition')
    target = _map('target', '2025-06-03', range(10))
    target.update(kills=999., duration_minutes=999., patch='14.2')
    frame, _, _ = prepare_draft_data(pd.DataFrame([old, invalid, target]),
                                    pd.DataFrame([_base('target', '2025-06-03')]))
    row = frame.iloc[0]
    assert row.p_global_kills_mean == 50
    assert row.p_global_duration_mean == 42
    assert row.d_kills_mean_a_0 == (30 + 20*50)/21
    assert row.d_duration_mean_a_0 == (32 + 20*42)/21
    assert row.d_player_logcount_a_0 == np.log1p(1)
    assert (row.p_patch_major, row.p_patch_minor, row.p_patch_missing) == (14, 2, 0)


def test_invalid_map_labels_do_not_become_global_or_role_statistics():
    old = _map('old', '2025-06-01', range(10))
    old.update(kills=-1., duration_minutes=0.)
    target = _map('target', '2025-06-02', range(10))
    frame, _, _ = prepare_draft_data(pd.DataFrame([old, target]),
                                    pd.DataFrame([_base('target', '2025-06-02')]))
    row = frame.iloc[0]
    assert row.d_player_logcount_a_0 == np.log1p(1)
    assert pd.isna(row.p_global_kills_mean)
    assert pd.isna(row.d_kills_mean_a_0)
    assert pd.isna(row.d_duration_mean_a_0)


@pytest.mark.parametrize('corruption', ['duplicate_game', 'duplicate_order', 'order_gap',
                                      'release', 'map_date', 'date', 'identity'])
def test_source_identity_order_and_chronology_corruption_is_rejected(corruption):
    first = _map('old', '2025-06-01', range(10))
    second = _map('target', '2025-06-02', range(10))
    if corruption == 'duplicate_game':
        second['game_id'] = first['game_id']
    elif corruption == 'duplicate_order':
        second.update(match_id='old', date=first['date'])
    elif corruption == 'order_gap':
        second['map_index'] = 2
    elif corruption == 'release':
        second['release_day'] = '2025-06-01'
    elif corruption == 'map_date':
        second['map_date'] = 'not-a-date'
    elif corruption == 'date':
        second['date'] = '2025-02-30'
    else:
        second['game_id'] = ' target '
    with pytest.raises(ValueError):
        prepare_draft_data(pd.DataFrame([first, second]),
                           pd.DataFrame([_base('target', '2025-06-02')]))


def test_read_only_same_day_queries_do_not_create_experience():
    maps = pd.DataFrame([_map('one', '2025-06-01', range(10)),
                         _map('two', '2025-06-01', range(10)),
                         _map('next', '2025-06-02', range(10))])
    frame, _, _ = prepare_draft_data(maps, pd.DataFrame([
        _base('two', '2025-06-01'), _base('one', '2025-06-01'),
        _base('next', '2025-06-02')]))
    assert (frame.iloc[:2].d_player_logcount_a_0 == 0).all()
    np.testing.assert_allclose(np.expm1(frame.iloc[2].d_player_logcount_a_0), 2., rtol=0, atol=1e-14)


@pytest.mark.parametrize('corruption', ['duplicate_match', 'overlap', 'duplicate_player',
                                      'team', 'release', 'identity'])
def test_canonical_identity_and_chronology_corruption_is_rejected(corruption):
    target = _base('target', '2025-06-02')
    rows = [target]
    if corruption == 'duplicate_match':
        rows.append(copy.deepcopy(target))
    elif corruption == 'overlap':
        target['roster_b'][0] = target['roster_a'][0]
    elif corruption == 'duplicate_player':
        target['roster_a'][1] = target['roster_a'][0]
    elif corruption == 'team':
        target['team_b'] = target['team_a']
    elif corruption == 'release':
        target['release_day'] = '2025-06-01'
    else:
        target['roster_a'][0] = ' p0 '
    with pytest.raises(ValueError):
        prepare_draft_data(pd.DataFrame([_map('target', '2025-06-02', range(10))]),
                           pd.DataFrame(rows))


def test_later_raw_series_release_overrides_earlier_canonical_release():
    first = _map('held', '2025-06-01', range(10))
    second = _map('held', '2025-06-01', range(10), '2025-06-03')
    second.update(game_id='held-second', map_index=2, map_date='2025-06-03')
    maps = pd.DataFrame([first, second, _map('target', '2025-06-03', range(10))])
    frame, _, _ = prepare_draft_data(maps, pd.DataFrame([
        _base('held', '2025-06-01'), _base('target', '2025-06-03')]))
    assert frame.iloc[1].d_player_logcount_a_0 == 0
    assert pd.isna(frame.iloc[1].p_global_kills_mean)


def test_each_statistic_uses_its_own_valid_label_support():
    old = _map('old', '2025-06-01', range(10))
    old.update(kills=30.5, duration_minutes=32.)
    target = _map('target', '2025-06-02', range(10))
    frame, _, _ = prepare_draft_data(pd.DataFrame([old, target]),
                                    pd.DataFrame([_base('target', '2025-06-02')]))
    row = frame.iloc[0]
    assert pd.isna(row.p_global_kills_mean)
    assert pd.isna(row.d_kills_mean_a_0)
    assert row.p_global_duration_mean == row.d_duration_mean_a_0 == 32
    assert row.d_role_logcount_a_0 == np.log1p(1)


def test_formal_training_partition_uses_later_raw_series_release():
    from src.models.draft_prop_runner import partition_rows
    old = _map('closed', '2025-06-29', range(10))
    first = _map('held', '2025-06-30', range(10))
    second = _map('held', '2025-06-30', range(10), '2025-07-01')
    second.update(game_id='held-second', map_index=2, map_date='2025-07-01')
    frame, _, audit = prepare_draft_data(
        pd.DataFrame([old, first, second]),
        pd.DataFrame([_base('held', '2025-06-30'), _base('closed', '2025-06-29')]))
    assert audit['partition_valid_counts']['TRAIN'] == 1
    assert partition_rows(frame)['TRAIN'].tolist() == [False, True]
