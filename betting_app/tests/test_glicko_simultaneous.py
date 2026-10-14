"""Opponent updates must use the same pre-game state, independent of sides."""
import pytest
from glicko2 import Player
from src.ratings.glicko import GlickoRating


def test_team_updates_match_independent_pre_game_calculation():
    system = GlickoRating()
    system.team_ratings['A'] = Player(rating=1730, rd=82)
    system.team_ratings['B'] = Player(rating=1490, rd=155)
    expected_a = Player(rating=1730, rd=82)
    expected_b = Player(rating=1490, rd=155)
    expected_a.update_player([1490], [155], [0])
    expected_b.update_player([1730], [82], [1])
    system.update_team('A', 'B', 0, 1)
    for team, expected in [('A', expected_a), ('B', expected_b)]:
        actual = system.get_team_rating(team)
        assert (actual.rating, actual.rd, actual.vol) == pytest.approx((expected.rating, expected.rd, expected.vol))
    reversed_system = GlickoRating()
    reversed_system.team_ratings['A'] = Player(rating=1730, rd=82)
    reversed_system.team_ratings['B'] = Player(rating=1490, rd=155)
    reversed_system.update_team('B', 'A', 1, 0)
    for team in ['A', 'B']:
        actual = reversed_system.get_team_rating(team)
        expected = system.get_team_rating(team)
        assert (actual.rating, actual.rd, actual.vol) == pytest.approx((expected.rating, expected.rd, expected.vol))


def test_pre_fix_persisted_state_cannot_resume_under_corrected_updates():
    from betting_app.scripts.rebuild_ratings import rating_from_row
    with pytest.raises(ValueError):
        rating_from_row('gl',GlickoRating(),{}, {'rating':1600.0,'rd':80.0,'volatility':0.06})


def test_persisted_corrected_state_preserves_next_game_prediction():
    import json
    from betting_app.scripts.rebuild_ratings import rating_from_row, unpack_rating
    original, restored = GlickoRating(), GlickoRating()
    original.update_team('A','B',1,0)
    for team in ('A','B'):
        state = json.loads(json.dumps(unpack_rating(original.get_team_rating(team))[3]))
        restored.team_ratings[team] = rating_from_row('gl',restored,{},state)
    original.update_team('A','B',0,1)
    restored.update_team('A','B',0,1)
    assert restored.predict_team_win_prob('A','B') == pytest.approx(original.predict_team_win_prob('A','B'),abs=1e-12)
