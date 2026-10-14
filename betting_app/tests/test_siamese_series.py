"""Experimental Siamese research-model tests with explicit synthetic parameters.

These tests never load or depend on the retired operational model artifact.
"""

from __future__ import annotations
import numpy as np
import pytest

from src.models.siamese_series import SiameseMember, SiameseSeriesModel
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS, build_feature_mapping



def _generate_synthetic_snapshots() -> tuple[dict[str, float], dict[str, float]]:
    """Build a realistic snapshot for Team A vs Team B and its swapped equivalent."""
    # Team A is stronger: higher Elo, Glicko, Win rate, GD15
    snapshot_a: dict[str, float] = {}

    # Team ratings
    snapshot_a["team_elo_r1"] = 1650.0
    snapshot_a["team_elo_r2"] = 1450.0
    snapshot_a["team_gl_r1"] = 1700.0
    snapshot_a["team_gl_r2"] = 1480.0
    snapshot_a["team_gl_rd1"] = 65.0
    snapshot_a["team_gl_rd2"] = 85.0

    # Player metrics
    snapshot_a["player_elo_min1"] = 1520.0
    snapshot_a["player_elo_min2"] = 1380.0
    snapshot_a["player_gl_max1"] = 1820.0
    snapshot_a["player_gl_max2"] = 1600.0
    snapshot_a["player_gl_rd_avg1"] = 60.0
    snapshot_a["player_gl_rd_avg2"] = 75.0

    # Rest days
    snapshot_a["days_since_last_1"] = 4.0
    snapshot_a["days_since_last_2"] = 7.0

    # Bayesian systems
    for sys_name in ("ts", "os", "pl", "tm"):
        snapshot_a[f"team_{sys_name}_mu1"] = 32.0
        snapshot_a[f"team_{sys_name}_mu2"] = 24.0
        snapshot_a[f"team_{sys_name}_sigma1"] = 3.2
        snapshot_a[f"team_{sys_name}_sigma2"] = 4.1
        snapshot_a[f"player_{sys_name}_sigma_avg1"] = 3.0
        snapshot_a[f"player_{sys_name}_sigma_avg2"] = 3.8

    # Win probabilities from rating systems (Team A favored: ~65-70%)
    for sys_name in ("elo", "gl", "ts", "os", "pl", "tm"):
        snapshot_a[f"team_{sys_name}"] = 0.68
        snapshot_a[f"player_{sys_name}"] = 0.64

    # W20 rolling stats
    w20_stats_a = {
        "win_rate": 0.65,
        "kills": 15.2,
        "deaths": 9.8,
        "gd15": 1250.0,
        "dpm": 2300.0,
        "vspm": 8.5,
        "towers": 7.8,
        "nashors": 0.85,
        "gold": 62000.0,
        "duration": 1950.0,
    }
    w20_stats_b = {
        "win_rate": 0.45,
        "kills": 10.5,
        "deaths": 14.2,
        "gd15": -800.0,
        "dpm": 1850.0,
        "vspm": 6.8,
        "towers": 4.5,
        "nashors": 0.35,
        "gold": 54000.0,
        "duration": 2050.0,
    }
    for k, v in w20_stats_a.items():
        snapshot_a[f"t1_rolling_{k}"] = v
    for k, v in w20_stats_b.items():
        snapshot_a[f"t2_rolling_{k}"] = v

    # Construct exact swapped snapshot_b
    snapshot_b: dict[str, float] = {}
    for f in _REQUIRED_BASE_FIELDS:
        if f.startswith("t1_rolling_"):
            stat = f[len("t1_rolling_"):]
            snapshot_b[f] = snapshot_a[f"t2_rolling_{stat}"]
        elif f.startswith("t2_rolling_"):
            stat = f[len("t2_rolling_"):]
            snapshot_b[f] = snapshot_a[f"t1_rolling_{stat}"]
        elif f.endswith("1"):
            prefix = f[:-1]
            snapshot_b[f] = snapshot_a[f"{prefix}2"]
        elif f.endswith("2"):
            prefix = f[:-1]
            snapshot_b[f] = snapshot_a[f"{prefix}1"]
        elif f in ("team_elo", "player_elo", "team_gl", "player_gl", "team_ts", "player_ts", "team_os", "player_os", "team_pl", "player_pl", "team_tm", "player_tm"):
            snapshot_b[f] = 1.0 - snapshot_a[f]
        else:
            raise KeyError(f"unhandled field in swapping: {f}")

    return snapshot_a, snapshot_b


def _synthetic_model() -> SiameseSeriesModel:
    snapshot, _ = _generate_synthetic_snapshots()
    feature_names = tuple(build_feature_mapping(snapshot, best_of=3))
    feature_count = len(feature_names)
    members = []
    for _ in range(5):
        w1 = np.zeros((feature_count, 32))
        b1 = np.zeros(32)
        w2 = np.zeros((32, 16))
        b2 = np.zeros(16)
        w3 = np.zeros((16, 1))
        # Exercise a signed synthetic signal while keeping the model independent
        # of the retired packaged artifact.
        w1[feature_names.index("team_elo_logit"), 0] = 1.0
        w1[feature_names.index("team_elo_logit_bo3"), 0] = 1.0
        w1[feature_names.index("team_elo_logit_bo5"), 0] = 1.0
        w2[0, 0] = 1.0
        w3[0, 0] = 1.0
        members.append(SiameseMember(w1, b1, w2, b2, w3, 0.0, 1.0))
    return SiameseSeriesModel(
        feature_names=feature_names,
        means=np.zeros(feature_count),
        scales=np.ones(feature_count),
        members=tuple(members),
    )


def test_explicit_synthetic_model_schema_and_members() -> None:
    model = _synthetic_model()
    assert len(model.feature_names) == len(model.means) == len(model.scales) == 79
    assert len(model.members) == 5
    assert model.members[0].w1.shape == (79, 32)
    assert model.members[0].w2.shape == (32, 16)
    assert model.members[0].w3.shape == (16, 1)


def test_exact_machine_anti_symmetry() -> None:
    model = _synthetic_model()

    snap_a, snap_b = _generate_synthetic_snapshots()

    for bo in (1, 3, 5):
        p_a = model.predict(snap_a, best_of=bo)
        p_b = model.predict(snap_b, best_of=bo)

        assert p_a + p_b == pytest.approx(1.0, abs=1e-12)
        assert p_a > 0.5
        assert p_b < 0.5

        p_mean_a, sigma_a, p_low_a, opposite_a = model.predict_with_uncertainty(snap_a, best_of=bo)
        p_mean_b, sigma_b, p_low_b, opposite_b = model.predict_with_uncertainty(snap_b, best_of=bo)

        assert sigma_a == pytest.approx(sigma_b, abs=1e-12)
        assert sigma_a == 0.0
        assert p_low_a == p_mean_a
        assert p_low_b == p_mean_b
        assert opposite_a == pytest.approx(p_low_b, abs=1e-12)
        assert opposite_b == pytest.approx(p_low_a, abs=1e-12)


@pytest.mark.parametrize("kappa", [-1.0, float("nan"), float("inf")])
def test_invalid_risk_penalty_is_rejected(kappa) -> None:
    model = _synthetic_model()
    snapshot, _ = _generate_synthetic_snapshots()
    with pytest.raises(ValueError):
        model.predict_with_uncertainty(snapshot, best_of=3, kappa=kappa)


def test_zero_risk_penalty_preserves_both_mean_probabilities() -> None:
    model = _synthetic_model()
    snapshot, _ = _generate_synthetic_snapshots()
    mean, _, low_a, low_b = model.predict_with_uncertainty(snapshot, best_of=3, kappa=0.0)
    assert low_a == mean
    assert low_b == pytest.approx(1.0 - mean)
    assert low_b <= 1.0 - mean


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_ensemble_member_cannot_produce_a_risk_score(invalid) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    model = replace(
        _synthetic_model(),
        members=(SimpleNamespace(forward_anti_symmetric=lambda _: invalid),),
    )
    snapshot, _ = _generate_synthetic_snapshots()
    with pytest.raises(ValueError):
        model.predict_with_uncertainty(snapshot, best_of=3)


def test_format_dynamics() -> None:
    model = _synthetic_model()
    snap_a, _ = _generate_synthetic_snapshots()

    p_bo1 = model.predict(snap_a, best_of=1)
    p_bo3 = model.predict(snap_a, best_of=3)
    p_bo5 = model.predict(snap_a, best_of=5)
    assert 0.0 < p_bo1 < 1.0
    assert p_bo3 > p_bo1
    assert p_bo5 > p_bo1

