"""Exact historical TrueSkill-probability correction, never a production change.

Inputs are explicit, hashed frozen artifacts. Only the ``player_ts`` snapshot
formula changes; state transitions and all other thesis features are replayed
through their existing implementations. Series projection remains the caller's
responsibility through ``thesis_audit.add_series_features``.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Sequence
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
from trueskill import Rating

from src.analysis.thesis_features import WINDOWS, rating_snapshot, replay_features, sha
from src.analysis.thesis_metrics import _metrics
from src.ratings.trueskill_rating import TrueSkillRating


PINNED_FEATURE_SHAPE = (40_636, 201)


def standard_trueskill_probability(
    rating: TrueSkillRating, team_a: Sequence[Rating], team_b: Sequence[Rating]
) -> float:
    """Return P(sum performance A > sum performance B), without mutating states.

    Teams contain current ``Rating`` objects, not IDs. There is one independent
    performance-noise variance beta**2 per player; tau belongs to the unchanged
    update rule, not this pre-match denominator. No inverse CDF is involved.
    """
    if not len(team_a) or not len(team_b):
        raise ValueError("TrueSkill probability requires two nonempty teams")
    delta = sum(player.mu for player in team_a) - sum(player.mu for player in team_b)
    variance = (
        sum(player.sigma ** 2 for player in team_a)
        + sum(player.sigma ** 2 for player in team_b)
        + (len(team_a) + len(team_b)) * rating.ts_model.beta ** 2
    )
    return float(rating.ts_model.cdf(delta / math.sqrt(variance)))


def check_trueskill_invariants() -> dict:
    """Exercise analytical formulas, symmetry, cold starts and state purity.

    The bundled TrueSkill CDF is approximate, including at zero. Independent
    math.erfc comparisons therefore use 1e-7 absolute tolerance; equality with
    the same backend and the legacy one-on-one formula uses 1e-14.
    This callable does not run on import and uses only disposable rating state.
    """
    rating = TrueSkillRating(beta=12.5, tau=0.166)
    a = [rating.ts_model.create_rating(mu=28 + i, sigma=5 + i / 2) for i in range(5)]
    b = [rating.ts_model.create_rating(mu=22 + i, sigma=6 + i / 3) for i in range(5)]
    rating.player_ratings.update({f"a{i}": value for i, value in enumerate(a)})
    rating.player_ratings.update({f"b{i}": value for i, value in enumerate(b)})
    rating.team_ratings["a"] = rating.ts_model.create_rating(mu=31, sigma=4)
    cold_a = [rating.ts_model.create_rating() for _ in range(5)]
    cold_b = [rating.ts_model.create_rating() for _ in range(5)]

    def state_values():
        return (
            tuple((key, id(value), value.mu, value.sigma)
                  for key, value in rating.player_ratings.items()),
            tuple((key, id(value), value.mu, value.sigma)
                  for key, value in rating.team_ratings.items()),
            tuple((id(value), value.mu, value.sigma) for value in cold_a + cold_b),
            (rating.ts_model.mu, rating.ts_model.sigma, rating.ts_model.beta,
             rating.ts_model.tau, rating.ts_model.draw_probability),
        )

    before = state_values()
    mean_difference = sum(p.mu for p in a) / 5 - sum(p.mu for p in b) / 5
    mean_variances = sum(p.sigma ** 2 for p in a) / 5 + sum(p.sigma ** 2 for p in b) / 5
    legacy_z = mean_difference / math.sqrt(mean_variances + 2 * rating.ts_model.beta ** 2)
    expected = float(rating.ts_model.cdf(math.sqrt(5) * legacy_z))
    observed = standard_trueskill_probability(rating, a, b)
    swapped = standard_trueskill_probability(rating, b, a)
    one_on_one = standard_trueskill_probability(rating, a[:1], b[:1])
    legacy_one_on_one = rating._expected_score(a[:1], b[:1])
    cold = standard_trueskill_probability(rating, cold_a, cold_b)
    unequal = standard_trueskill_probability(rating, a, b[:3])
    unequal_z = (sum(p.mu for p in a) - sum(p.mu for p in b[:3])) / math.sqrt(
        sum(p.sigma ** 2 for p in a + b[:3]) + 8 * rating.ts_model.beta ** 2
    )
    errors = {
        "equal_team_analytical_error": abs(observed - expected),
        "independent_normal_cdf_error": abs(observed - 0.5 * math.erfc(-math.sqrt(5) * legacy_z / math.sqrt(2))),
        "swap_symmetry_error": abs(observed + swapped - 1),
        "one_on_one_error": abs(one_on_one - legacy_one_on_one),
        "cold_start_error": abs(cold - 0.5),
        "unequal_team_analytical_error": abs(unequal - 0.5 * math.erfc(-unequal_z / math.sqrt(2))),
    }
    for name, error in errors.items():
        tolerance = 1e-14 if name in ("equal_team_analytical_error", "one_on_one_error") else 1e-7
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError(f"TrueSkill invariant failed: {name}={error}, tolerance={tolerance}")
    if before != state_values():
        raise AssertionError("TrueSkill probability mutated rating states")
    return {"status": "PASS", **errors, "cold_start_probability": cold,
            "standard_probability": observed, "legacy_probability": rating._expected_score(a, b),
            "rating_states_unchanged": True}


def _dependency_hashes() -> dict[str, str]:
    paths = set()
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(("scripts.", "src.", "trueskill", "openskill")):
            path = Path(path).resolve()
            if path.is_file():
                paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}


def _json_finite(value):
    """Use JSON null for undefined descriptive metrics, never NaN literals."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _raw_diagnostics(original: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    dates = pd.to_datetime(original.date)
    diagnostics = {}
    for name, mask in (("all_history", np.ones(len(original), dtype=bool)),
                       ("since2024", (dates >= pd.Timestamp("2024-01-01")).to_numpy())):
        if not mask.any():
            continue
        labels = original.y_true.to_numpy(dtype=int)[mask]
        cohort = {}
        for label, frame in (("original", original), ("standard", corrected)):
            probabilities = frame.player_ts.to_numpy(dtype=float)[mask]
            metrics, bins = _metrics(labels, probabilities)
            cohort[label] = {"metrics": metrics, "reliability_bins": bins,
                             "probability_min": float(probabilities.min()),
                             "probability_max": float(probabilities.max()),
                             "probability_mean": float(probabilities.mean()),
                             "probability_std": float(probabilities.std()),
                             "mean_absolute_distance_from_half": float(np.mean(abs(probabilities - 0.5)))}
        diagnostics[name] = cohort
    return diagnostics


def replay_trueskill_correction(
    raw_source, metadata_path, release_path, original_feature_path, output_dir
) -> Path:
    """Replay frozen history and export only after exact baseline identity proof.

    Refuses existing output directories and mismatched pinned shape/schema/data.
    The returned Path is ``output_dir/features.parquet``. STARTED.json without a
    COMPLETE feature_change_summary.json denotes a failed/incomplete run and is
    deliberately not reusable. No production methods or updates are patched.
    """
    from scripts.export_prospective_features import iter_matches
    from scripts.prospective_sports_features import ChronologicalFeatureState, _prepare_history

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "raw_source": Path(raw_source), "metadata": Path(metadata_path),
        "release_days": Path(release_path), "original_features": Path(original_feature_path),
        "correction_module": Path(__file__),
    }
    sources = {str(path.resolve()): sha(path) for path in paths.values()}
    # Initialize the existing transition implementation before freezing its
    # dynamically imported dependencies; no historical outcomes are consumed.
    state = ChronologicalFeatureState()
    state.histories = defaultdict(lambda: deque(maxlen=max(WINDOWS)))
    dependencies = _dependency_hashes()
    helper_path = str(Path(state.helpers.__file__).resolve())
    dependencies[helper_path] = sha(helper_path)
    invariants = check_trueskill_invariants()
    protocol = {
        "sources": sources, "input_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "dependencies": dependencies, "python": platform.python_version(),
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "trueskill", "openskill"
        )},
        "required_shape": PINNED_FEATURE_SHAPE, "windows": WINDOWS,
        "invariant_checks": invariants,
        "changed_feature": "player_ts",
        "formula": "cdf((sum(mu_A)-sum(mu_B))/sqrt(sum(sigma_A^2)+sum(sigma_B^2)+(n_A+n_B)*beta^2))",
        "policy": "Existing full-series release, strictly prior target day, first-map known-roster scenario; unchanged rating updates; no inverse-CDF reconstruction; no model tuning.",
    }
    (output / "STARTED.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    original = pd.read_parquet(original_feature_path)
    metadata = pd.read_parquet(metadata_path)
    if original.shape != PINNED_FEATURE_SHAPE:
        raise ValueError(f"Pinned feature shape must be {PINNED_FEATURE_SHAPE}, got {original.shape}")
    if not original.columns.is_unique or "player_ts" not in original:
        raise ValueError("Pinned feature columns must be unique and contain player_ts")
    if not metadata.index.is_unique or len(metadata) != len(original):
        raise ValueError("Target metadata must have unique index and the pinned row count")
    release = json.loads(Path(release_path).read_text())
    series, audit = _prepare_history(iter_matches(Path(raw_source)), date.max)
    baseline_probabilities = {}

    def corrected_snapshot(current_state, row):
        snapshot = rating_snapshot(current_state, row)
        baseline_probabilities[row.name] = snapshot["player_ts"]
        rating = current_state.manager.systems["ts"]
        # Match RatingManager's existing empty-roster semantics. The baseline
        # snapshot has already materialized all these states, so reads are pure.
        players_a = list(row.roster_a) or [f"dummy_{row.team1_id}"]
        players_b = list(row.roster_b) or [f"dummy_{row.team2_id}"]
        snapshot["player_ts"] = standard_trueskill_probability(
            rating, [rating.player_ratings[player] for player in players_a],
            [rating.player_ratings[player] for player in players_b],
        )
        return snapshot

    corrected = replay_features(series, metadata, release, state, snapshot=corrected_snapshot)
    baseline = corrected.copy()
    baseline["player_ts"] = [baseline_probabilities[index] for index in metadata.index]
    # No sorting, merging, rounding, tolerance or copying pinned values over a
    # mismatch: every replayed original field must prove identity independently.
    pd.testing.assert_frame_equal(baseline, original, check_exact=True, check_dtype=True,
                                  check_like=False, obj="frozen baseline replay")
    unchanged = [column for column in original.columns if column != "player_ts"]
    pd.testing.assert_frame_equal(corrected[unchanged], original[unchanged],
                                  check_exact=True, check_dtype=True, check_like=False,
                                  obj="unchanged corrected features")
    old = original.player_ts.to_numpy(dtype=float)
    new = corrected.player_ts.to_numpy(dtype=float)
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("TrueSkill forecasts must all be finite")
    if ((old < 0) | (old > 1) | (new < 0) | (new > 1)).any():
        raise ValueError("TrueSkill forecasts must lie in [0, 1]")
    drift = np.abs(new - old)
    changed = int(np.count_nonzero(new != old))
    if not changed:
        raise AssertionError("The standard formula unexpectedly changed no frozen forecasts")
    diagnostics = _raw_diagnostics(original, corrected)
    # Catch input or implementation changes during the potentially long replay.
    for path, digest in {**sources, **dependencies}.items():
        if sha(path) != digest:
            raise ValueError(f"Frozen replay input/dependency changed: {path}")
    final_dependencies = {**dependencies, **_dependency_hashes()}
    destination = output / "features.parquet"
    corrected.to_parquet(destination, index=False)
    summary = {
        **protocol, "status": "COMPLETE", "dependencies": final_dependencies,
        "rows": len(corrected), "columns": list(corrected.columns),
        "history_series": len(series), "history_audit": audit,
        "baseline_identity": {"exact": True, "rows": len(original),
                              "columns": len(original.columns), "check_dtype": True,
                              "absolute_tolerance": 0, "relative_tolerance": 0},
        "changed_columns": ["player_ts"], "changed_rows": changed,
        "changed_counts": {column: changed if column == "player_ts" else 0 for column in original.columns},
        "max_absolute_drift": float(drift.max()), "mean_absolute_drift": float(drift.mean()),
        "unchanged_columns": len(unchanged), "unchanged_columns_max_absolute_drift": 0,
        "threshold_half_disagreements": int(np.count_nonzero((new >= 0.5) != (old >= 0.5))),
        "raw_forecast_diagnostics": diagnostics,
        "diagnostic_scope": "Descriptive raw player_ts against series y_true, using existing thesis metrics; not calibrated or binomial-projected forecasts and not evidence of prospective performance.",
        "output_sha256": sha(destination),
    }
    (output / "feature_change_summary.json").write_text(
        json.dumps(_json_finite(summary), indent=2, allow_nan=False) + "\n"
    )
    return destination
