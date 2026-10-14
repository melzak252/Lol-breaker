from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.ratings.family_calibrated_glicko2 import (
    FamilyCalibratedGlicko2,
    RatingEvent,
    series_win_probability,
)


def event(
    event_id: str,
    *,
    event_date: date = date(2025, 1, 1),
    team_a_id: str = "team-a",
    team_b_id: str = "team-b",
    players_a: tuple[str, ...] = ("a1", "a2"),
    players_b: tuple[str, ...] = ("b1", "b2"),
    family_a: str = "LCK",
    family_b: str = "LPL",
    tier_a: str = "major",
    tier_b: str = "major",
    scores: tuple[int, ...] = (1, 0, 1),
    best_of: int | None = None,
) -> RatingEvent:
    return RatingEvent(
        event_id=event_id,
        event_date=event_date,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        players_a=players_a,
        players_b=players_b,
        family_a=family_a,
        family_b=family_b,
        tier_a=tier_a,
        tier_b=tier_b,
        scores=scores,
        best_of=best_of,
    )


def reverse_sides(value: RatingEvent) -> RatingEvent:
    return RatingEvent(
        event_id=value.event_id,
        event_date=value.event_date,
        team_a_id=value.team_b_id,
        team_b_id=value.team_a_id,
        players_a=tuple(reversed(value.players_b)),
        players_b=tuple(reversed(value.players_a)),
        family_a=value.family_b,
        family_b=value.family_a,
        tier_a=value.tier_b,
        tier_b=value.tier_a,
        scores=tuple(1 - score for score in reversed(value.scores)),
        best_of=value.best_of,
    )


def test_same_date_event_side_and_player_permutations_are_invariant() -> None:
    first = event("02", players_a=("a2", "a1"), scores=(1, 0, 1))
    second = event(
        "01",
        team_a_id="team-c",
        team_b_id="team-d",
        players_a=("c2", "c1"),
        players_b=("d2", "d1"),
        family_a="LEC",
        family_b="LCS",
        scores=(0, 1, 0),
    )

    canonical = FamilyCalibratedGlicko2()
    canonical_predictions = canonical.process_period([first, second])

    permuted = FamilyCalibratedGlicko2()
    permuted_predictions = permuted.process_period(
        [
            replace(second, players_a=tuple(reversed(second.players_a)), scores=(0, 0, 1)),
            replace(first, players_b=tuple(reversed(first.players_b)), scores=(0, 1, 1)),
        ]
    )
    assert permuted_predictions == canonical_predictions
    assert permuted.to_state() == canonical.to_state()

    reversed_engine = FamilyCalibratedGlicko2()
    reversed_predictions = reversed_engine.process_period(
        [reverse_sides(second), reverse_sides(first)]
    )
    assert reversed_engine.to_state() == canonical.to_state()
    for event_id, probability in canonical_predictions.items():
        assert reversed_predictions[event_id] == pytest.approx(1.0 - probability)


def test_domestic_same_family_result_does_not_update_offsets() -> None:
    engine = FamilyCalibratedGlicko2()
    engine.process_period(
        [event("domestic", family_a="LCK", family_b="LCK", scores=(1, 1, 0))]
    )

    family = engine.get_family_state("LCK")
    tier = engine.get_tier_state("major")
    assert family.mean == 0.0
    assert family.variance == 150.0**2
    assert tier.mean == 0.0
    assert tier.variance == 100.0**2

    initial_bridge_date = family.last_bridge_date
    engine.process_period(
        [
            event(
                "later-domestic",
                event_date=date(2025, 2, 1),
                players_a=("c1", "c2"),
                players_b=("d1", "d2"),
                family_a="LEC",
                family_b="LEC",
                scores=(1,),
            )
        ]
    )
    advanced_family = engine.get_family_state("LCK")
    assert advanced_family.variance > family.variance
    assert advanced_family.last_bridge_date == initial_bridge_date

def test_shared_domestic_location_mean_and_variance_cancel_from_matchup() -> None:
    seed = FamilyCalibratedGlicko2()
    seed.process_period(
        [event("seed", family_a="LCK", family_b="LCK", scores=(1,))]
    )
    ordinary = FamilyCalibratedGlicko2.from_state(seed.to_state())
    shifted_payload = json.loads(json.dumps(seed.to_state()))
    shifted_payload["families"]["LCK"]["mean"] = 700.0
    shifted_payload["families"]["LCK"]["variance"] = 9_000_000.0
    shifted_payload["tiers"]["major"]["mean"] = -250.0
    shifted_payload["tiers"]["major"]["variance"] = 4_000_000.0
    shifted = FamilyCalibratedGlicko2.from_state(shifted_payload)
    follow_up = event(
        "follow-up",
        event_date=date(2025, 1, 2),
        family_a="LCK",
        family_b="LCK",
        scores=(0,),
    )

    assert ordinary.process_period([follow_up]) == shifted.process_period([follow_up])
    for player_id in ordinary.player_ids:
        assert ordinary.get_player_state(player_id) == shifted.get_player_state(player_id)


def test_same_tier_cross_family_match_cancels_shared_tier_uncertainty() -> None:
    low_tier_variance = FamilyCalibratedGlicko2(initial_tier_deviation=0.0)
    high_tier_variance = FamilyCalibratedGlicko2(initial_tier_deviation=10_000.0)
    domestic = event(
        "domestic-seed",
        family_a="LCK",
        family_b="LCK",
        scores=(1,),
    )
    low_tier_variance.process_period([domestic])
    high_tier_variance.process_period([domestic])
    bridge = event(
        "same-tier-bridge",
        event_date=date(2025, 1, 2),
        family_a="LCK",
        family_b="LPL",
        tier_a="major",
        tier_b="major",
        scores=(1,),
    )

    low_prediction = low_tier_variance.process_period([bridge])
    high_prediction = high_tier_variance.process_period([bridge])
    assert low_prediction == high_prediction
    for player_id in low_tier_variance.player_ids:
        assert low_tier_variance.get_player_state(
            player_id
        ) == high_tier_variance.get_player_state(player_id)


def test_cross_family_result_moves_offsets_and_centers_total_location() -> None:
    engine = FamilyCalibratedGlicko2()
    engine.process_period(
        [
            event(
                "bridge",
                family_a="LCK",
                tier_a="major",
                family_b="VCS",
                tier_b="minor_top_level",
                scores=(1, 1, 0),
            )
        ]
    )

    lck = engine.get_family_location("LCK", "major")
    vcs = engine.get_family_location("VCS", "minor_top_level")
    assert lck.mean > 0.0
    assert vcs.mean < 0.0
    assert lck.mean + vcs.mean == pytest.approx(0.0, abs=1e-12)
    assert engine.get_family_state("LCK").mean > 0.0
    assert engine.get_tier_state("major").mean > 0.0
    assert engine.get_family_state("VCS").mean < 0.0
    assert engine.get_tier_state("minor_top_level").mean < 0.0

    ranking = engine.get_player_ranking("a1")
    assert ranking.raw_rating == engine.get_player_state("a1").rating
    assert ranking.rating == pytest.approx(ranking.raw_rating + lck.mean)
    assert ranking.rd > ranking.raw_rd


def test_transfer_keeps_player_skill_and_adds_no_synthetic_result() -> None:
    engine = FamilyCalibratedGlicko2()
    engine.process_period(
        [
            event(
                "before-transfer",
                players_a=("transfer-1", "transfer-2"),
                players_b=("opponent-1", "opponent-2"),
                family_a="LCK",
                family_b="LCK",
                scores=(1,),
            )
        ]
    )
    prior_rating = engine.get_player_state("transfer-1").rating

    engine.process_period(
        [
            event(
                "after-transfer",
                event_date=date(2025, 1, 2),
                players_a=("transfer-2", "transfer-1"),
                players_b=("new-opponent-1", "new-opponent-2"),
                family_a="LPL",
                family_b="LPL",
                scores=(0,),
            )
        ]
    )

    assert prior_rating != 1500.0
    assert engine.get_player_state("transfer-1") == engine.get_player_state("transfer-2")
    assert engine.get_player_state("transfer-1").rating != 1500.0
    assert engine.get_player_affiliation("transfer-1") == ("LPL", "major")
    assert engine.get_player_games("transfer-1") == 2


def test_returning_player_inflates_from_own_last_activity_across_daily_periods() -> None:
    inactive = FamilyCalibratedGlicko2()
    active = FamilyCalibratedGlicko2()
    start = date(2025, 1, 1)
    opening = event("initial", event_date=start, scores=(1,))
    inactive.process_period([opening])
    active.process_period([opening])
    stored_inactive = inactive.get_player_state("a1")
    initial_ranking = inactive.get_player_ranking("a1")

    for day in range(1, 32):
        unrelated = event(
            f"unrelated-{day:02d}",
            event_date=start + timedelta(days=day),
            players_a=("c1", "c2"),
            players_b=("d1", "d2"),
            family_a="LEC",
            family_b="LEC",
            scores=(day % 2,),
        )
        inactive.process_period([unrelated])
        active.process_period(
            [
                replace(
                    unrelated,
                    players_a=("a1", "a2"),
                    players_b=("b1", "b2"),
                    family_a="LCK",
                    family_b="LPL",
                )
            ]
        )

    projected_ranking = inactive.get_player_ranking("a1")
    assert inactive.get_player_state("a1") == stored_inactive
    assert projected_ranking.raw_rating == stored_inactive.rating
    assert projected_ranking.raw_rd > stored_inactive.rd
    assert projected_ranking.rd > initial_ranking.rd
    restored_inactive = FamilyCalibratedGlicko2.from_state(inactive.to_state())
    assert restored_inactive.get_player_state("a1") == stored_inactive
    assert restored_inactive.get_player_ranking("a1") == projected_ranking

    returning = event(
        "return",
        event_date=start + timedelta(days=32),
        scores=(1, 0),
    )
    inactive.process_period([returning])
    active.process_period([returning])
    assert inactive.get_player_state("a1").rd > active.get_player_state("a1").rd


def test_unknown_affiliation_is_player_only_and_roundtrips() -> None:
    engine = FamilyCalibratedGlicko2()
    predictions = engine.process_period(
        [
            event(
                "unknown-side",
                family_a="unknown",
                tier_a="unknown",
                family_b="unknown",
                tier_b="unknown",
                scores=(1,),
            )
        ]
    )

    assert predictions["unknown-side"] == pytest.approx(0.5)
    assert engine.family_ids == ()
    assert engine.tier_ids == ()
    assert engine.get_player_affiliation("a1") is None
    assert engine.get_player_state("a1").rating > 1500.0
    unknown_ranking = engine.get_player_ranking("a1")
    assert unknown_ranking.family == unknown_ranking.tier == "unknown"
    assert unknown_ranking.rd > unknown_ranking.raw_rd
    restored = FamilyCalibratedGlicko2.from_state(engine.to_state())
    assert restored.to_state() == engine.to_state()

    known = FamilyCalibratedGlicko2()
    known.process_period(
        [
            event(
                "known-prior",
                players_a=("known-player",),
                players_b=("known-opponent",),
                family_a="LCK",
                family_b="LCK",
                scores=(1,),
            )
        ]
    )
    known.process_period(
        [
            event(
                "unknown-later",
                event_date=date(2025, 1, 2),
                players_a=("known-player",),
                players_b=("unaffiliated-opponent",),
                family_a="unknown",
                tier_a="unknown",
                family_b="unknown",
                tier_b="unknown",
                scores=(0,),
            )
        ]
    )
    assert known.get_player_affiliation("known-player") == ("LCK", "major")
    assert known.family_ids == ("LCK",)


def test_same_day_conflicting_affiliations_preserve_prior_or_remain_unknown() -> None:
    first_order = FamilyCalibratedGlicko2()
    second_order = FamilyCalibratedGlicko2()
    prior = event(
        "prior",
        players_a=("known",),
        players_b=("prior-opponent",),
        family_a="LCK",
        family_b="LCK",
        scores=(1,),
    )
    first_order.process_period([prior])
    second_order.process_period([prior])
    conflict_a = event(
        "a",
        event_date=date(2025, 1, 2),
        players_a=("known", "new"),
        players_b=("a-opponent",),
        family_a="LPL",
        family_b="LPL",
        scores=(1,),
    )
    conflict_b = event(
        "b",
        event_date=date(2025, 1, 2),
        team_a_id="team-c",
        team_b_id="team-d",
        players_a=("new", "known"),
        players_b=("b-opponent",),
        family_a="LEC",
        family_b="LEC",
        scores=(0,),
    )

    first_order.process_period([conflict_a, conflict_b])
    second_order.process_period([conflict_b, conflict_a])
    assert first_order.to_state() == second_order.to_state()
    assert first_order.get_player_affiliation("known") == ("LCK", "major")
    assert first_order.get_player_affiliation("new") is None


def test_state_roundtrip_is_deterministic_and_continues_identically() -> None:
    engine = FamilyCalibratedGlicko2()
    engine.process_period([event("first")])
    payload = engine.to_state()

    encoded = json.dumps(payload, sort_keys=True)
    restored = FamilyCalibratedGlicko2.from_state(json.loads(encoded))
    assert restored.to_state() == payload
    assert restored.get_player_rankings() == engine.get_player_rankings()

    next_event = event(
        "second",
        event_date=date(2025, 2, 5),
        team_a_id="team-b",
        team_b_id="team-a",
        players_a=("b1", "b2"),
        players_b=("a1", "a2"),
        family_a="LPL",
        family_b="LCK",
        scores=(1, 1),
    )
    assert restored.process_period([next_event]) == engine.process_period([next_event])
    assert restored.to_state() == engine.to_state()


@pytest.mark.parametrize(
    "events",
    [
        [],
        [event("duplicate"), event("duplicate")],
        [event("one-date"), event("other-date", event_date=date(2025, 1, 2))],
        [event("empty-a", players_a=())],
        [event("duplicate-player", players_a=("a1", "a1"))],
        [event("overlap", players_b=("a1", "b2"))],
        [event("same-team", team_b_id="team-a")],
        [event("unknown-family-only", family_a="unknown")],
        [event("empty-family", family_a="")],
        [event("invented-family", family_a="Atlantis")],
        [event("unknown-tier-only", tier_a="unknown")],
        [event("invented-tier", tier_a="elite")],
        [event("empty-scores", scores=())],
        [event("draw-score", scores=(1, 2))],
        [event("boolean-score", scores=(True,))],
    ],
)
def test_invalid_periods_are_rejected_atomically(events: list[RatingEvent]) -> None:
    engine = FamilyCalibratedGlicko2()
    before = engine.to_state()

    with pytest.raises((TypeError, ValueError)):
        engine.process_period(events)

    assert engine.to_state() == before


def test_periods_must_advance_strictly_after_commit() -> None:
    engine = FamilyCalibratedGlicko2()
    first = event("first")
    engine.process_period([first])
    committed = engine.to_state()

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.process_period([replace(first, event_id="same-date")])

    assert engine.to_state() == committed


def test_series_win_probability_combinatorics() -> None:
    probs = [0.0, 0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.0]
    for p in probs:
        p_bo1 = series_win_probability(p, 1)
        p_bo3 = series_win_probability(p, 3)
        p_bo5 = series_win_probability(p, 5)

        # Exact polynomial check
        assert p_bo1 == pytest.approx(p)
        assert p_bo3 == pytest.approx(p * p * (3.0 - 2.0 * p))
        assert p_bo5 == pytest.approx(p * p * p * (10.0 - 15.0 * p + 6.0 * p * p))

        # Exact anti-symmetry: P(p) + P(1-p) == 1.0
        assert p_bo1 + series_win_probability(1.0 - p, 1) == pytest.approx(1.0, abs=1e-12)
        assert p_bo3 + series_win_probability(1.0 - p, 3) == pytest.approx(1.0, abs=1e-12)
        assert p_bo5 + series_win_probability(1.0 - p, 5) == pytest.approx(1.0, abs=1e-12)

        # Boundary values
        if p == 0.0:
            assert p_bo1 == 0.0 and p_bo3 == 0.0 and p_bo5 == 0.0
        elif p == 1.0:
            assert p_bo1 == 1.0 and p_bo3 == 1.0 and p_bo5 == 1.0
        elif p == 0.5:
            assert p_bo1 == 0.5 and p_bo3 == 0.5 and p_bo5 == 0.5

    # Monotonic increase for favored team (p > 0.5)
    for p in [0.51, 0.60, 0.70, 0.85]:
        p1 = series_win_probability(p, 1)
        p3 = series_win_probability(p, 3)
        p5 = series_win_probability(p, 5)
        assert p < p3 < p5, f"Expected monotonic increase for p={p}: {p} < {p3} < {p5}"

    # Monotonic decrease for underdog (p < 0.5)
    for p in [0.15, 0.30, 0.40, 0.49]:
        p1 = series_win_probability(p, 1)
        p3 = series_win_probability(p, 3)
        p5 = series_win_probability(p, 5)
        assert p > p3 > p5, f"Expected monotonic decrease for underdog p={p}: {p} > {p3} > {p5}"

    # Input validations
    for invalid_p in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            series_win_probability(invalid_p, 3)
    for invalid_bo in [0, -1, 2, "3", None]:
        with pytest.raises((ValueError, TypeError)):
            series_win_probability(0.6, invalid_bo)  # type: ignore[arg-type]


def test_process_period_anti_symmetry_across_tiers_and_formats() -> None:
    engine_forward = FamilyCalibratedGlicko2(regional_discount=0.70)
    engine_reversed = FamilyCalibratedGlicko2(regional_discount=0.70)

    # Seed players with domestic activity in LCK (major) and LFL (regional)
    seed_events = [
        event("seed-lck", family_a="LCK", family_b="LCK", players_a=("lck1", "lck2"), players_b=("lck3", "lck4"), scores=(1, 1)),
        event("seed-lfl", family_a="LFL", family_b="LFL", tier_a="regional", tier_b="regional", players_a=("lfl1", "lfl2"), players_b=("lfl3", "lfl4"), scores=(1, 1)),
    ]
    engine_forward.process_period(seed_events)
    engine_reversed.process_period(seed_events)

    # Test matches in Bo1, Bo3, Bo5
    match_date = date(2025, 1, 2)
    cross_events = [
        event("bo1-major-vs-reg", event_date=match_date, family_a="LCK", family_b="LFL", tier_a="major", tier_b="regional", players_a=("lck1", "lck2"), players_b=("lfl1", "lfl2"), scores=(1,), best_of=1),
        event("bo3-major-vs-reg", event_date=match_date, family_a="LCK", family_b="LFL", tier_a="major", tier_b="regional", players_a=("lck1", "lck2"), players_b=("lfl1", "lfl2"), scores=(1, 0, 1), best_of=3),
        event("bo5-major-vs-reg", event_date=match_date, family_a="LCK", family_b="LFL", tier_a="major", tier_b="regional", players_a=("lck1", "lck2"), players_b=("lfl1", "lfl2"), scores=(1, 1, 1), best_of=5),
    ]
    reversed_cross_events = [reverse_sides(e) for e in cross_events]

    forward_preds = engine_forward.process_period(cross_events)
    reversed_preds = engine_reversed.process_period(reversed_cross_events)

    for e in cross_events:
        f_prob = forward_preds[e.event_id]
        r_prob = reversed_preds[e.event_id]
        assert f_prob + r_prob == pytest.approx(1.0, abs=1e-12), f"Anti-symmetry violated for {e.event_id}"


def test_regional_tier_discount_shifts_major_vs_lower_tier_win_rate() -> None:
    match_date = date(2025, 1, 2)

    # Prepare two identical engines: one undiscounted (gamma=1.0) and one with gamma=0.70
    undiscounted_engine = FamilyCalibratedGlicko2(regional_discount=1.0)
    discounted_engine = FamilyCalibratedGlicko2(regional_discount=0.70)

    # Give both major and regional teams the exact same winning record against 1500 opponents
    day1_undisc = [
        event("d1-lck", event_date=date(2025, 1, 1), family_a="LCK", family_b="LCK", tier_a="major", tier_b="major", players_a=("m1", "m2"), players_b=("opp_m1", "opp_m2"), scores=(1, 1)),
        event("d1-lfl", event_date=date(2025, 1, 1), family_a="LFL", family_b="LFL", tier_a="regional", tier_b="regional", players_a=("r1", "r2"), players_b=("opp_r1", "opp_r2"), scores=(1, 1)),
    ]
    day1_disc = [
        event("d1-lck", event_date=date(2025, 1, 1), family_a="LCK", family_b="LCK", tier_a="major", tier_b="major", players_a=("m1", "m2"), players_b=("opp_m1", "opp_m2"), scores=(1, 1)),
        event("d1-lfl", event_date=date(2025, 1, 1), family_a="LFL", family_b="LFL", tier_a="regional", tier_b="regional", players_a=("r1", "r2"), players_b=("opp_r1", "opp_r2"), scores=(1, 1)),
    ]
    undiscounted_engine.process_period(day1_undisc)
    discounted_engine.process_period(day1_disc)

    # Check raw ratings are identical
    assert undiscounted_engine.get_player_state("m1").rating == pytest.approx(discounted_engine.get_player_state("m1").rating)
    assert undiscounted_engine.get_player_state("r1").rating == pytest.approx(discounted_engine.get_player_state("r1").rating)
    raw_m = undiscounted_engine.get_player_state("m1").rating
    raw_r = undiscounted_engine.get_player_state("r1").rating
    assert raw_m == pytest.approx(raw_r)  # Identical skill

    # Predict head-to-head match: Major vs Regional
    head_to_head = event("cross-match", event_date=match_date, family_a="LCK", family_b="LFL", tier_a="major", tier_b="regional", players_a=("m1", "m2"), players_b=("r1", "r2"), scores=(1, 0, 1), best_of=3)

    pred_undiscounted = undiscounted_engine.predict_event(head_to_head)
    pred_discounted = discounted_engine.predict_event(head_to_head)

    # Undiscounted (symmetric ratings, zero bridge mean) should be ~0.50
    assert pred_undiscounted == pytest.approx(0.50, abs=0.01)
    # Discounted (gamma=0.70) shifts in favor of major team significantly
    assert pred_discounted > 0.60
    assert pred_discounted > pred_undiscounted

    # Monotonicity with gamma: smaller gamma (stronger discount) -> higher win prob for major
    gammas = [0.90, 0.70, 0.50]
    preds = []
    for g in gammas:
        eng = FamilyCalibratedGlicko2(gamma=g)
        eng.process_period(day1_disc)
        p = eng.predict_event(head_to_head)
        preds.append(p)
    assert preds[0] < preds[1] < preds[2]


def test_regional_discount_configuration_and_serialization() -> None:
    engine = FamilyCalibratedGlicko2(regional_discount=0.65)
    assert engine.regional_discount == 0.65
    assert engine.gamma == 0.65

    engine_gamma = FamilyCalibratedGlicko2(gamma=0.75)
    assert engine_gamma.regional_discount == 0.75
    assert engine_gamma.gamma == 0.75

    for invalid in [0.0, -0.5, 1.5]:
        with pytest.raises(ValueError):
            FamilyCalibratedGlicko2(regional_discount=invalid)
        with pytest.raises(ValueError):
            FamilyCalibratedGlicko2(gamma=invalid)

    # Roundtrip serialization
    e = FamilyCalibratedGlicko2(regional_discount=0.70)
    state = e.to_state()
    assert state["parameters"]["regional_discount"] == 0.70
    restored = FamilyCalibratedGlicko2.from_state(state)
    assert restored.regional_discount == 0.70


def test_missing_affiliations_and_substitutes() -> None:
    engine = FamilyCalibratedGlicko2(regional_discount=0.70)
    day1 = [
        event("starter-game", players_a=("starter", "p2"), players_b=("b1", "b2"), scores=(1,)),
    ]
    engine.process_period(day1)
    starter_rating = engine.get_player_state("starter").rating

    # Day 2: Substitute plays instead of starter
    day2 = [
        event("sub-game", event_date=date(2025, 1, 2), players_a=("substitute", "p2"), players_b=("b1", "b2"), scores=(0,)),
    ]
    engine.process_period(day2)

    # Substitute does not inherit starter's rating; starter's rating remains unchanged
    assert engine.get_player_state("starter").rating == starter_rating
    assert engine.get_player_state("substitute").rating != starter_rating
    assert engine.get_player_games("starter") == 1
    assert engine.get_player_games("substitute") == 1

    # Unknown affiliations match: neutral, no discount applied
    unknown_event = event("unknown-affil", event_date=date(2025, 1, 3), family_a="unknown", family_b="unknown", tier_a="unknown", tier_b="unknown", players_a=("p2", "starter"), players_b=("b1", "b2"), scores=(1,))
    rev_unknown = reverse_sides(unknown_event)
    p_forward = engine.predict_event(unknown_event)
    p_rev = engine.predict_event(rev_unknown)
    assert p_forward + p_rev == pytest.approx(1.0, abs=1e-12)

