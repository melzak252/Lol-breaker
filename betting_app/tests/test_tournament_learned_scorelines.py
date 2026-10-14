"""Behavioral contract for the TRAIN-only conditional stopped-score law."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.analysis.tournament_metrics import (
    compute_scoreline_log_loss,
    compute_series_length_rps,
)
from src.models.tournament_scorelines import LearnedConditionalScorelineModel


@pytest.fixture(scope="module")
def closed_train() -> pd.DataFrame:
    """Distinct series: strong winners sweep, weak winners usually drop maps.

    Alternating the winning side makes the explanatory variable the *winner's*
    probability rather than the probability of whichever side occupies column A.
    Each format contains every legal losing-map count and each row has its own
    date, identity and probability; repeated copies do not amplify the signal.
    """
    rows = []
    for best_of in (3, 5):
        target = best_of // 2 + 1
        for group, base_probability in enumerate((0.52, 0.68, 0.86)):
            for offset in range(12):
                p_winner = base_probability + offset * 0.006
                if group == 0:
                    losing_maps = target - 1 if offset < 10 else offset % target
                elif group == 2:
                    losing_maps = 0 if offset < 10 else 1 + offset % (target - 1)
                else:
                    losing_maps = offset % target
                y = offset % 2
                day = pd.Timestamp("2024-02-01") + pd.Timedelta(days=len(rows))
                rows.append({
                    "match_id": f"closed-train-{len(rows)}",
                    "p_c0": p_winner if y else 1.0 - p_winner,
                    "y": y,
                    "best_of": best_of,
                    "wins_a": target if y else losing_maps,
                    "wins_b": losing_maps if y else target,
                    "date": day.isoformat(),
                    "result_day": (day + pd.Timedelta(days=1)).isoformat(),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def learned(closed_train):
    return LearnedConditionalScorelineModel.fit(closed_train, cutoff="2025-01-01")


@pytest.mark.parametrize("best_of", [3, 5])
def test_fit_learns_winner_strength_dependent_sweeps(learned, best_of):
    weak = learned.losing_map_pmf(0.55, best_of)
    strong = learned.losing_map_pmf(0.92, best_of)
    assert strong[0] > weak[0] + 0.15
    assert strong[-1] < weak[-1] - 0.15
    assert sum(i * p for i, p in enumerate(strong)) < sum(
        i * p for i, p in enumerate(weak)
    )
    assert sum(weak) == pytest.approx(1.0, abs=1e-12)
    assert sum(strong) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("p", [0.001, 0.173, 0.5, 0.827, 0.999])
def test_exact_score_law_preserves_series_probability_and_terminal_support(learned, best_of, p):
    target = best_of // 2 + 1
    pmf = learned.scoreline_pmf(p, best_of)
    support = {(target, lost) for lost in range(target)} | {
        (lost, target) for lost in range(target)
    }
    assert set(pmf) == support
    assert all(math.isfinite(value) and value > 0.0 for value in pmf.values())
    assert sum(pmf.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(pmf[(target, lost)] for lost in range(target)) == pytest.approx(p, abs=1e-12)
    assert sum(pmf[(lost, target)] for lost in range(target)) == pytest.approx(1.0 - p, abs=1e-12)
    winner_a = learned.losing_map_pmf(p, best_of)
    winner_b = learned.losing_map_pmf(1.0 - p, best_of)
    for lost in range(target):
        assert pmf[(target, lost)] == pytest.approx(p * winner_a[lost], abs=1e-12)
        assert pmf[(lost, target)] == pytest.approx((1.0 - p) * winner_b[lost], abs=1e-12)
    reversed_pmf = learned.scoreline_pmf(1.0 - p, best_of)
    assert {(b, a): value for (a, b), value in reversed_pmf.items()} == pytest.approx(pmf, abs=1e-12)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_boundary_probabilities_keep_exact_zero_for_impossible_winner(learned, best_of, p):
    target = best_of // 2 + 1
    pmf = learned.scoreline_pmf(p, best_of)
    assert set(pmf) == {(target, lost) for lost in range(target)} | {
        (lost, target) for lost in range(target)
    }
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pmf.values())
    assert sum(pmf[(target, lost)] for lost in range(target)) == p
    assert sum(pmf[(lost, target)] for lost in range(target)) == 1.0 - p
    impossible = [(lost, target) if p else (target, lost) for lost in range(target)]
    assert all(pmf[score] == 0.0 for score in impossible)
    boundary = learned.losing_map_pmf(p, best_of)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in boundary)
    assert sum(boundary) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.137, 0.5, 0.863, 1.0])
def test_bo1_has_only_deterministic_conditional_single_map(learned, p):
    assert learned.scoreline_pmf(p, 1) == {(1, 0): p, (0, 1): 1.0 - p}
    assert list(learned.losing_map_pmf(p, 1)) == [1.0]
    assert learned.series_length_pmf(p, 1) == {1: 1.0}


@pytest.mark.parametrize("date,result_day", [
    ("2025-01-02", "2025-01-03"),
    ("2025-01-01", "2025-01-02"),
    ("2024-12-29", "2025-01-01"),
    ("2024-12-29", "2025-01-03"),
])
def test_future_or_unreleased_scores_cannot_change_fit(closed_train, learned, date, result_day):
    extra = {
        "match_id": "excluded-series", "p_c0": 0.98, "y": 1,
        "best_of": 5, "wins_a": 3, "wins_b": 0,
        "date": date, "result_day": result_day,
    }
    before = LearnedConditionalScorelineModel.fit(
        pd.concat([closed_train, pd.DataFrame([extra])], ignore_index=True),
        cutoff="2025-01-01",
    )
    extra.update(y=0, wins_a=2, wins_b=3)
    after = LearnedConditionalScorelineModel.fit(
        pd.concat([closed_train, pd.DataFrame([extra])], ignore_index=True),
        cutoff="2025-01-01",
    )
    for best_of in (3, 5):
        assert before.pooled_distributions[best_of] == pytest.approx(learned.pooled_distributions[best_of])
        assert after.pooled_distributions[best_of] == pytest.approx(learned.pooled_distributions[best_of])
        for p in (0.17, 0.55, 0.91):
            expected = learned.scoreline_pmf(p, best_of)
            assert before.scoreline_pmf(p, best_of) == pytest.approx(expected, abs=1e-12)
            assert after.scoreline_pmf(p, best_of) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("best_of,wins_a,wins_b,y", [
    (3, 1, 1, 1), (3, 2, 2, 1), (3, 3, 0, 1),
    (5, 3, 3, 1), (5, 2, 1, 1), (5, 3, -1, 1),
    (5, 3, 1.5, 1), (5, 3, 1, 0), (1, 1, 1, 1),
    (7, 4, 0, 1),
])
def test_fit_rejects_nonterminal_scores_and_unsupported_formats(closed_train, best_of, wins_a, wins_b, y):
    frame = closed_train.astype({"wins_a": float, "wins_b": float}).copy()
    frame.loc[0, ["best_of", "wins_a", "wins_b", "y"]] = [best_of, wins_a, wins_b, y]
    with pytest.raises(ValueError):
        LearnedConditionalScorelineModel.fit(frame, cutoff="2025-01-01")


@pytest.mark.parametrize("best_of", [0, 2, 7])
def test_prediction_rejects_unsupported_formats(learned, best_of):
    for predict in (learned.scoreline_pmf, learned.losing_map_pmf, learned.series_length_pmf):
        with pytest.raises(ValueError):
            predict(0.7, best_of)


def test_saved_coefficients_preserve_likelihoods_after_json_roundtrip(learned, tmp_path):
    path = tmp_path / "learned-scoreline.json"
    path.write_text(json.dumps(learned.to_dict(), allow_nan=False), encoding="utf-8")
    restored = LearnedConditionalScorelineModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    for best_of in (1, 3, 5):
        for p in (0.0, 0.173, 0.5, 0.827, 1.0):
            expected = learned.scoreline_pmf(p, best_of)
            actual = restored.scoreline_pmf(p, best_of)
            assert actual == pytest.approx(expected, abs=1e-12)
            for score, probability in expected.items():
                if probability > 0.0:
                    assert -math.log(actual[score]) == pytest.approx(-math.log(probability), abs=1e-10)
            assert restored.series_length_pmf(p, best_of) == pytest.approx(
                learned.series_length_pmf(p, best_of), abs=1e-12
            )


@pytest.mark.parametrize("best_of", [3, 5])
def test_exact_score_and_length_scoring_use_learned_law_not_pooled_train(learned, closed_train, best_of):
    target = best_of // 2 + 1
    train = closed_train.loc[closed_train.best_of == best_of]
    losing = train[["wins_a", "wins_b"]].min(axis=1)
    empirical = [float((losing == lost).mean()) for lost in range(target)]
    assert learned.pooled_distributions[best_of] == pytest.approx(empirical)
    p = 0.92
    pooled_score = {(target, lost): p * empirical[lost] for lost in range(target)}
    pooled_score.update({(lost, target): (1.0 - p) * empirical[lost] for lost in range(target)})
    pooled_length = {target + lost: empirical[lost] for lost in range(target)}
    exact = learned.scoreline_pmf(p, best_of)
    lengths = learned.series_length_pmf(p, best_of)
    assert set(lengths) == set(range(target, best_of + 1))
    for length, probability in lengths.items():
        assert probability == pytest.approx(
            sum(mass for score, mass in exact.items() if sum(score) == length), abs=1e-12
        )
    sweep = (target, 0)
    assert compute_scoreline_log_loss(exact, sweep) == pytest.approx(-math.log(exact[sweep]))
    assert compute_scoreline_log_loss(exact, sweep) < compute_scoreline_log_loss(pooled_score, sweep)
    expected_rps = sum(
        (sum(prob for length, prob in lengths.items() if length <= boundary) - 1.0) ** 2
        for boundary in range(target, best_of)
    ) / (target - 1)
    assert compute_series_length_rps(lengths, target, best_of) == pytest.approx(expected_rps)
    assert compute_series_length_rps(lengths, target, best_of) < compute_series_length_rps(
        pooled_length, target, best_of
    )


def test_later_scoring_cutoff_cannot_admit_cal_releases_into_closed_train(closed_train):
    frame = closed_train.copy()
    frame.loc[0, "date"] = "2024-12-31T00:00:00"
    frame.loc[0, "result_day"] = "2025-01-02T00:00:00"
    closed = LearnedConditionalScorelineModel.fit(frame, cutoff="2025-01-01")
    later = LearnedConditionalScorelineModel.fit(frame, cutoff="2025-02-01")
    for p in (.05, .5, .95):
        assert later.scoreline_pmf(p, 3) == pytest.approx(
            closed.scoreline_pmf(p, 3), abs=1e-12)


def test_saved_scoreline_train_cannot_extend_into_cal_releases(learned):
    payload = learned.to_dict()
    payload["provenance"]["cutoff"] = "2025-02-01T00:00:00+00:00"
    payload["provenance"]["max_result_day"] = "2025-01-02T00:00:00+00:00"
    with pytest.raises(ValueError):
        LearnedConditionalScorelineModel.from_dict(payload)
