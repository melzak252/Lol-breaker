import numpy as np
import pandas as pd
import pytest

from src.models.champion_history_runner import _align_c0_reference


def fixture():
    native = pd.DataFrame({'golgg_match_id': ['x', 'y'], 'y': [1, 0], 'best_of': [3, 5],
                           'w32_e12__full': [.8, .7]})
    metadata = pd.DataFrame({'golgg_match_id': ['y', 'x'], 'team1_id': ['C', 'A'],
                             'team2_id': ['D', 'B'], 'y': [0, 1], 'best_of': [5, 3]})
    target = pd.DataFrame({'match_id': ['x', 'y'], 'team_a': ['A', 'D'],
                           'team_b': ['B', 'C'], 'winner': [1, 1], 'best_of': [3, 5]})
    return native, metadata, target


def test_reversed_participants_complement_margin_without_outcome_selection():
    probability, reversed_ = _align_c0_reference(*fixture())
    np.testing.assert_array_equal(reversed_, [False, True])
    np.testing.assert_allclose(probability, [.8, .3], atol=1e-15)


def test_changed_outcome_is_rejected_not_used_to_choose_probability_orientation():
    native, metadata, target = fixture()
    target.loc[1, 'winner'] = 0
    with pytest.raises(ValueError, match='winner orientation or labels'):
        _align_c0_reference(native, metadata, target)


def test_wrong_participant_is_rejected_even_if_saved_outcome_agrees():
    native, metadata, target = fixture()
    target.loc[0, 'team_a'] = 'UNRELATED'
    with pytest.raises(ValueError, match='participant'):
        _align_c0_reference(native, metadata, target)


def test_duplicate_reference_identity_is_not_silently_resolved():
    native, metadata, target = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        _align_c0_reference(pd.concat([native, native.iloc[:1]]), metadata, target)


def test_native_label_lineage_and_format_must_match_metadata():
    native, metadata, target = fixture()
    metadata.loc[0, 'y'] = 1
    with pytest.raises(ValueError, match='native label'):
        _align_c0_reference(native, metadata, target)
    native, metadata, target = fixture()
    native.loc[0, 'best_of'] = 5
    with pytest.raises(ValueError, match='format'):
        _align_c0_reference(native, metadata, target)


def test_singleton_score_reference_exports_coherent_winner_length_and_play_probabilities(tmp_path):
    from src.models.champion_history_model import HistoryPrediction
    from src.models.champion_history_runner import _save_prediction

    prediction = HistoryPrediction(np.array([[.4, .2, 0., .1, .3, 0.]]), np.array([3]))
    series = pd.DataFrame({'row_index': [0], 'match_id': ['x'], 'date': ['2026-01-08'],
                           'best_of': [3], 'winner': [1], 'score_index': [0]})
    _save_prediction(prediction, series, pd.DataFrame(), tmp_path, 'reference')
    terminal = pd.read_parquet(tmp_path / 'reference_terminal.parquet').iloc[0]
    assert terminal.p_win == pytest.approx(.6)
    assert terminal.length_p_0 == pytest.approx(.5)
    assert terminal.length_p_1 == pytest.approx(.5)
    assert terminal.length_p_2 == 0.
    assert terminal.map_3_play_probability == pytest.approx(.5)
    assert terminal.map_4_play_probability == terminal.map_5_play_probability == 0.
    components = pd.read_parquet(tmp_path / 'reference_components.parquet')
    assert components.statistics_head_available.eq(False).all()
