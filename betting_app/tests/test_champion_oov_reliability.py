"""W1 RED contracts. Synthetic arrays only: no database, source scan, or fit.

Initialization/prediction checks are deliberately not training trajectories.
The separate w1_contract.json declares scientific decisions required before fits.
"""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.champion_oov_reliability import (
    build_identity_state,
    inventory_history_support,
    stage0_support_gate,
)
from src.models.champion_oov_reliability import (
    CompositionPreprocessor,
    assemble_terminal_ensemble,
    build_initial_model,
    compose_identity,
    held_channel_codes,
    held_identity,
    identity_residual_weights,
)
from src.models.champion_history_model import (
    DEFAULT_CONFIG,
    HistoryPrediction,
    HistoryPreprocessor,
    HistorySeedModel,
    _HistoryNetwork,
)

CHANNELS = ("champion", "patch", "competition")
ARRAY_CHANNELS = {"champion": "champions", "patch": "patches", "competition": "competitions"}
HISTORY_NAMES = (
    "kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm", "gd@15",
    "csd@15", "xpd@15", "kp%", "dmg%", "kills_for", "kills_against",
    "total_kills", "duration_minutes", "prior_opponent_strength_mean",
    "prior_opponent_strength_sd", "played_age_days", "release_age_days",
)


def _arrays():
    rng = np.random.default_rng(91)
    shape = (3, 2, 5, 16)
    arrays = {
        "history": rng.normal(size=shape + (20,)).astype(np.float32),
        "mask": np.ones(shape, dtype=bool),
        "roles": rng.integers(0, 6, shape, dtype=np.int64),
        "players": rng.normal(size=(3, 2, 5, 2)).astype(np.float32),
        "role_probs": np.full((3, 2, 5, 6), 1 / 6, dtype=np.float32),
        "context": rng.normal(size=(3, 9)).astype(np.float32),
        "best_of": np.array([1, 3, 5]),
    }
    for key in ARRAY_CHANNELS.values():
        arrays[key] = np.full(shape, 2, dtype=np.int64)
    arrays["mask"][0] = False
    arrays["history"][0] = np.nan
    return arrays


def _unheld_id(channel):
    return next(str(i) for i in range(1, 100) if int.from_bytes(
        hashlib.sha256(f"{channel}:{i}".encode("utf-8")).digest(), "big") % 5 != 0)


def _schema():
    return {"history_feature_names": list(HISTORY_NAMES),
            **{f"{channel}_vocabulary": {_unheld_id(channel): 2} for channel in CHANNELS}}


def _identity_state(arrays):
    # Prepared per-token support sidecars share exactly the inherited token axes.
    return {"n_prior": {channel: np.full(arrays["mask"].shape, 20, dtype=np.int64)
                        for channel in CHANNELS}}


@pytest.mark.parametrize("channel", CHANNELS)
def test_entity_partition_is_exact_sha256_channel_colon_id_modulo_five(channel):
    identities = ["103", "114", "14", "competition with spaces", "patch:14.1"]
    expected = {identity: int.from_bytes(hashlib.sha256(
        f"{channel}:{identity}".encode("utf-8")).digest(), "big") % 5 == 0
        for identity in identities}
    assert {identity: held_identity(channel, identity) for identity in identities} == expected
    assert {identity: held_identity(channel, identity) for identity in reversed(identities)} == expected
    with pytest.raises(ValueError, match="channel"):
        held_identity("target_patch", "103")
    for missing in (None, ""):
        with pytest.raises(ValueError, match="identity"):
            held_identity(channel, missing)


def test_h_routes_every_occurrence_of_held_identity_to_unk_without_touching_missing_padding():
    channel = "champion"
    held = next(str(i) for i in range(1, 100) if int.from_bytes(
        hashlib.sha256(f"{channel}:{i}".encode()).digest(), "big") % 5 == 0)
    vocabulary = {held: 2, _unheld_id(channel): 3}
    codes = np.array([2, 3, 2, 0, 1, 2, 2])
    mask = np.array([True, True, True, True, True, False, True])
    before = codes.copy()
    routed = held_channel_codes(channel, codes, vocabulary, mask)
    np.testing.assert_array_equal(routed, [1, 3, 1, 0, 1, 2, 1])
    order = np.array([6, 2, 4, 0, 5, 3, 1])
    np.testing.assert_array_equal(held_channel_codes(channel, codes[order], vocabulary, mask[order]), routed[order])
    np.testing.assert_array_equal(codes, before)


def test_support_shrink_is_prior_count_over_prior_count_plus_twenty_with_zero_unknown_residual():
    channel = "champion"
    known = _unheld_id(channel)
    held = next(str(i) for i in range(1, 100) if int.from_bytes(
        hashlib.sha256(f"{channel}:{i}".encode()).digest(), "big") % 5 == 0)
    ids = np.array([known, known, known, "natural-oov", None, held, known], dtype=object)
    counts = np.array([0, 20, 60, 1000, 1000, 1000, 1000])
    mask = np.array([True, True, True, True, True, True, False])
    result = identity_residual_weights(channel, ids, counts, mask, {known: 2, held: 3}, held_out=True)
    np.testing.assert_allclose(result, [0, .5, .75, 0, 0, 0, 0], atol=0, rtol=0)
    control = identity_residual_weights(channel, ids, counts, mask, {known: 2, held: 3}, held_out=False)
    assert control[5] == pytest.approx(1000 / 1020)
    np.testing.assert_array_equal(counts, [0, 20, 60, 1000, 1000, 1000, 1000])
    with pytest.raises(ValueError, match="prior"):
        identity_residual_weights(channel, ids, counts - 1, mask, {known: 2}, held_out=False)


@pytest.mark.parametrize("dimension", [8, 4])
def test_composition_preserves_actual_channel_width_plus_supported_residual_and_padding_is_zero(dimension):
    phi = np.arange(5 * 36, dtype=float).reshape(5, 36) / 100
    residual = np.full((5, dimension), 4.)
    weight = np.arange(dimension * 36, dtype=float).reshape(dimension, 36) / 1000
    bias = np.arange(dimension, dtype=float) / 10
    before = [x.copy() for x in (phi, residual, weight, bias)]
    result = compose_identity(phi, residual, np.array([20, 60, 20, 20, 0]),
                              np.array([True, True, True, False, True]),
                              np.array([True, False, True, True, False]),
                              np.array([False, False, True, False, False]), weight, bias,
                              missing=np.array([False, False, False, False, True]))
    expected = phi @ weight.T + bias
    expected[0] += 2
    expected[3] = 0
    expected[4] += 4  # missing identity retains the UNGATED legacy code0 residual
    np.testing.assert_allclose(result, expected, atol=1e-12)
    for original, snapshot in zip((phi, residual, weight, bias), before):
        np.testing.assert_array_equal(original, snapshot)
    with pytest.raises(ValueError, match="36"):
        compose_identity(phi[:, :-1], residual, np.ones(5), np.ones(5, bool),
                         np.ones(5, bool), np.zeros(5, bool), weight, bias)


def test_phi36_has_exact_existing_channels_stat_masks_five_roles_and_train_only_normalization():
    arrays = _arrays()
    arrays["history"][1] = 2
    arrays["history"][2] = 1000  # never influence TRAIN normalization
    arrays["history"][1, 0, 0, -1, 0] = np.nan
    arrays["roles"][1, 0, 0, -1] = 3
    arrays["roles"][1, 0, 1, -1] = 0  # unknown role: five zeros, not a sixth bit
    before = copy.deepcopy(arrays)
    pre = CompositionPreprocessor.fit(arrays, np.array([0, 1]), history_feature_names=HISTORY_NAMES)
    phi = pre.transform(arrays)
    assert phi.shape == arrays["mask"].shape + (36,)
    np.testing.assert_array_equal(phi[0], 0)
    np.testing.assert_array_equal(phi[1, 0, 0, -1, :19], 0)
    np.testing.assert_array_equal(phi[1, 0, 0, -1, 19:31], [0] + [1] * 11)
    np.testing.assert_array_equal(phi[1, 0, 0, -1, 31:], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(phi[1, 0, 1, -1, 31:], 0)
    changed = copy.deepcopy(arrays)
    changed["history"][..., 17] = 99999  # sd is NOT the one opponent-strength phi field
    np.testing.assert_array_equal(phi, pre.transform(changed))
    assert np.all(phi[2, ..., :19] > 0)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    with pytest.raises(ValueError, match="feature"):
        CompositionPreprocessor.fit(arrays, np.array([1]), history_feature_names=HISTORY_NAMES[:-1])


def test_identity_state_uses_unique_released_source_events_not_repeated_windows_or_future_records():
    observations = pd.DataFrame([
        {"observation_id": "g1:p1", "match_id": "s1", "date": "2025-06-28", "release_day": "2025-06-29",
         "champion": "103", "patch": "14.1", "competition": "Exact Cup", "history_eligible": True},
        {"observation_id": "g2:p1", "match_id": "s2", "date": "2025-06-29", "release_day": "2025-06-30",
         "champion": "103", "patch": "14.1", "competition": "Exact Cup", "history_eligible": True},
        {"observation_id": "g3:p1", "match_id": "s3", "date": "2025-06-29", "release_day": "2025-07-02",
         "champion": "future", "patch": "future", "competition": "Future Cup", "history_eligible": True},
        {"observation_id": "g4:p1", "match_id": "s4", "date": "2025-06-27", "release_day": "2025-06-28",
         "champion": "excluded", "patch": "excluded", "competition": "Excluded Cup", "history_eligible": False},
    ])
    targets = pd.DataFrame({"match_id": ["t1", "t2"], "date": ["2025-06-30", "2025-07-01"]})
    result = build_identity_state(observations, targets, train_end="2025-07-01")
    assert set(result["vocabularies"]["champion"]) == {"103"}
    assert result["targets"]["t1"]["champion"] == {"103": 1}
    assert result["targets"]["t2"]["champion"] == {"103": 2}
    assert result == build_identity_state(observations.iloc[::-1], targets.iloc[::-1], train_end="2025-07-01")
    with pytest.raises(ValueError, match="duplicate"):
        build_identity_state(pd.concat([observations, observations.iloc[:1]]), targets, train_end="2025-07-01")
    bad = observations.copy()
    bad.loc[0, "release_day"] = "2025-06-27"
    with pytest.raises(ValueError, match="release"):
        build_identity_state(bad, targets, train_end="2025-07-01")


def test_stage0_counts_independent_fixtures_missing_and_cold_separately_from_token_totals():
    arrays = _arrays()
    arrays["mask"][:] = False
    arrays["mask"][0, 0, 0, :3] = True
    arrays["mask"][1, 0, 0, :2] = True
    arrays["champions"][:] = 1  # padding must not manufacture OOV fixtures
    arrays["champions"][0, 0, 0, :3] = [1, 1, 0]
    arrays["champions"][1, 0, 0, :2] = [2, 0]
    arrays["patches"][0, 0, 0, 0] = 1
    arrays["competitions"][1, 0, 0, 0] = 1
    series = pd.DataFrame({"row_index": [0, 1, 2], "match_id": ["a", "b", "c"],
                           "date": ["2026-01-08", "2026-02-01", "2026-03-01"],
                           "split": ["ASSESS"] * 3})
    raw = {channel: np.full(arrays["mask"].shape, _unheld_id(channel), dtype=object) for channel in CHANNELS}
    raw["champion"][0, 0, 0, :3] = ["natural-new", "natural-new", None]
    raw["champion"][1, 0, 0, :2] = [_unheld_id("champion"), None]
    raw["patch"][0, 0, 0, 0] = "new-patch"
    raw["competition"][1, 0, 0, 0] = "new-exact-competition"
    inventory = inventory_history_support(series, arrays, canonical_identities=raw, schema=_schema())
    rows = inventory["series"].set_index("match_id")
    assert rows.loc["a", "valid_tokens"] == 3
    assert rows.loc["a", "champion_oov_tokens"] == 2
    assert rows.loc["a", "natural_champion_oov"]
    assert rows.loc["a", "champion_missing_tokens"] == 1
    assert rows.loc["b", "seen_champion_history"]
    assert rows.loc["b", "champion_missing_tokens"] == 1
    assert rows.loc["c", "cold_history"]
    assert not rows.loc["c", "natural_champion_oov"]
    assert not rows.loc["c", "seen_champion_history"]
    assert inventory["assessment"]["natural_champion_oov"]["series_count"] == 1
    assert inventory["assessment"]["natural_champion_oov"]["months"] == ["2026-01"]
    assert inventory["month_universe"] == ["2026-01", "2026-02", "2026-03"]
    assert rows.loc["a", "patch_oov_tokens"] == 1
    assert rows.loc["b", "competition_oov_tokens"] == 1
    missing_raw = copy.deepcopy(raw)
    missing_raw["champion"][0, 0, 0, 0] = None
    with pytest.raises(ValueError, match="identity"):
        inventory_history_support(series, arrays, canonical_identities=missing_raw, schema=_schema())


@pytest.mark.parametrize("group,count,months", [
    ("natural_champion_oov", 199, 6), ("natural_champion_oov", 200, 5),
    ("seen_champion_history", 199, 6), ("seen_champion_history", 200, 5),
])
def test_stage0_requires_both_oov_and_seen_fixture_and_month_floors(group, count, months):
    inventory = {"assessment": {name: {"series_count": 200,
                 "months": [f"2026-{i:02d}" for i in range(1, 7)]}
                 for name in ("natural_champion_oov", "seen_champion_history")}}
    components = {"available": True, "valid_masks": True, "phi_dimension": 36}
    assert stage0_support_gate(inventory, components)["status"] == "SUPPORT_PASS"
    inventory["assessment"][group] = {"series_count": count, "months": [f"2026-{i:02d}" for i in range(1, months + 1)]}
    result = stage0_support_gate(inventory, components)
    assert result["status"] == "CLOSED_NO_FITS"
    assert any(group in reason for reason in result["reasons"])
    assert result["allowed_neural_fits"] == 0


@pytest.mark.parametrize("components", [
    {"available": False, "valid_masks": True, "phi_dimension": 36},
    {"available": True, "valid_masks": False, "phi_dimension": 36},
    {"available": True, "valid_masks": True, "phi_dimension": 40},
])
def test_stage0_closes_without_component_or_mask_contract_even_when_fixture_support_passes(components):
    inventory = {"assessment": {name: {"series_count": 200,
                 "months": [f"2026-{i:02d}" for i in range(1, 7)]}
                 for name in ("natural_champion_oov", "seen_champion_history")}}
    assert stage0_support_gate(inventory, components)["status"] == "CLOSED_NO_FITS"


def test_terminal_temperature_is_applied_to_each_seed_before_average_without_bo_reconversion():
    pmfs = [np.array([[.81, 0, 0, .19, 0, 0], [.09, .01, 0, .4, .5, 0]]),
            np.array([[.64, 0, 0, .36, 0, 0], [.3, .3, 0, .1, .3, 0]]),
            np.array([[.36, 0, 0, .64, 0, 0], [.6, .1, 0, .2, .1, 0]])]
    members = [HistoryPrediction(p, np.array([1, 3])) for p in pmfs]
    result = assemble_terminal_ensemble(members, temperature=2.)
    tempered = [np.sqrt(p) / np.sqrt(p).sum(axis=1, keepdims=True) for p in pmfs]
    expected = np.mean(tempered, axis=0)
    np.testing.assert_allclose(result.score_probs, expected, atol=1e-14)
    np.testing.assert_allclose(result.p_win, expected[:, :3].sum(1), atol=1e-14)
    assert len(result.components) == 3
    for component, p in zip(result.components, tempered):
        np.testing.assert_allclose(component.score_probs, p, atol=1e-14)
    wrong = np.sqrt(np.mean(pmfs, axis=0))
    wrong /= wrong.sum(axis=1, keepdims=True)
    assert not np.allclose(result.score_probs, wrong)
    for bad in (0, -1, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="temperature"):
            assemble_terminal_ensemble(members, temperature=bad)
    with pytest.raises(ValueError, match="three"):
        assemble_terminal_ensemble(members[:2], temperature=1.)


@pytest.mark.parametrize("arm", ["H", "C", "HC"])
def test_initial_predictions_have_legal_bo_pmf_side_swap_and_cold_mask_invariance_without_input_mutation(arm):
    arrays = _arrays()
    snapshot = copy.deepcopy(arrays)
    model = build_initial_model(arrays, np.array([0, 1]), seed=20261007, arm=arm,
                                schema=_schema(), identity_state=_identity_state(arrays), config=DEFAULT_CONFIG)
    original = model.predict(arrays, identity_state=_identity_state(arrays))
    side_keys = {"history", "mask", "roles", "players", "role_probs", *ARRAY_CHANNELS.values()}
    swapped = {key: value[:, ::-1].copy() if key in side_keys else value.copy() for key, value in arrays.items()}
    reversed_prediction = model.predict(swapped, identity_state=_identity_state(swapped))
    np.testing.assert_allclose(reversed_prediction.score_probs, original.score_probs[:, [3, 4, 5, 0, 1, 2]], atol=2e-6)
    np.testing.assert_allclose(original.p_win + reversed_prediction.p_win, 1, atol=2e-6)
    np.testing.assert_allclose(original.score_probs.sum(1), 1, atol=1e-9)
    assert np.isfinite(original.score_probs).all()
    legal = np.tile(np.arange(3), 2)[None, :] <= (arrays["best_of"] // 2)[:, None]
    assert np.all(original.score_probs[~legal] == 0)
    poisoned = copy.deepcopy(arrays)
    poisoned["history"][0] = 1e6
    for key in (*ARRAY_CHANNELS.values(), "roles"):
        poisoned[key][0] = 1
    np.testing.assert_array_equal(model.predict(poisoned, identity_state=_identity_state(poisoned)).score_probs[0], original.score_probs[0])
    missing = copy.deepcopy(arrays)
    for key in ARRAY_CHANNELS.values():
        missing[key][1] = 0
    assert np.isfinite(model.predict(missing, identity_state=_identity_state(missing)).score_probs).all()
    permuted = {key: value[:, :, [4, 1, 3, 0, 2]].copy() if key in side_keys else value.copy() for key, value in arrays.items()}
    np.testing.assert_allclose(model.predict(permuted, identity_state=_identity_state(permuted)).score_probs, original.score_probs, atol=2e-6)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], snapshot[key])


def test_h_unheld_initial_backbone_matches_old_seed_stream_by_observable_predictions_not_hashes():
    arrays = _arrays()
    indices = np.array([0, 1])
    config = dict(DEFAULT_CONFIG)
    vocab_sizes = dict.fromkeys(ARRAY_CHANNELS.values(), 3)
    preprocessing = HistoryPreprocessor.fit(arrays, indices, config["batch_size"])
    # No fit; this is the original constructor stream only. Prop bias cannot
    # change terminal predictions and is intentionally not compared here.
    with torch.random.fork_rng():
        torch.manual_seed(20261007)
        backbone = _HistoryNetwork((20, 2, 9), vocab_sizes, config)
    backbone.eval()
    reference = HistorySeedModel(config=config, seed=20261007, champion_enabled=True,
                                 multitask=True, dimensions=(20, 2, 9), vocab_sizes=vocab_sizes,
                                 preprocessing=preprocessing, network=backbone)
    h = build_initial_model(arrays, indices, seed=20261007, arm="H", schema=_schema(),
                            identity_state=_identity_state(arrays), config=config)
    np.testing.assert_allclose(h.predict(arrays, identity_state=_identity_state(arrays)).score_probs, reference.predict(arrays).score_probs, atol=2e-6)
    c = build_initial_model(arrays, indices, seed=20261007, arm="C", schema=_schema(),
                            identity_state=_identity_state(arrays), config=config)
    hc = build_initial_model(arrays, indices, seed=20261007, arm="HC", schema=_schema(),
                             identity_state=_identity_state(arrays), config=config)
    # Same seed, same active capacity and all IDs unheld => identical predictions.
    np.testing.assert_array_equal(c.predict(arrays, identity_state=_identity_state(arrays)).score_probs, hc.predict(arrays, identity_state=_identity_state(arrays)).score_probs)

    assert sum(p.numel() for p in c.network.parameters()) == sum(p.numel() for p in h.network.parameters()) + 592
    assert sum(p.numel() for p in hc.network.parameters()) == sum(p.numel() for p in c.network.parameters())
    missing = copy.deepcopy(arrays)
    unknown = copy.deepcopy(arrays)
    for key in ARRAY_CHANNELS.values():
        missing[key][:] = 0
        unknown[key][:] = 1
    expected_missing = reference.predict(missing).score_probs
    np.testing.assert_allclose(h.predict(missing, identity_state=_identity_state(missing)).score_probs, expected_missing, atol=2e-6)
    np.testing.assert_array_equal(c.predict(missing, identity_state=_identity_state(missing)).score_probs, hc.predict(missing, identity_state=_identity_state(missing)).score_probs)
    assert not np.allclose(c.predict(missing, identity_state=_identity_state(missing)).score_probs[1:], c.predict(unknown, identity_state=_identity_state(unknown)).score_probs[1:], atol=1e-8)
    assert not np.allclose(h.predict(missing, identity_state=_identity_state(missing)).score_probs[1:], h.predict(unknown, identity_state=_identity_state(unknown)).score_probs[1:], atol=1e-8)


@pytest.mark.parametrize("arm", ["C", "HC"])
def test_variable_support_sidecars_follow_explicit_side_player_and_target_axes(arm):
    arrays = _arrays()
    shape = arrays["mask"].shape
    levels = np.array([0, 1, 5, 20, 60, 100, 1000], dtype=np.int64)
    positions = np.arange(np.prod(shape)).reshape(shape)
    state = {"n_prior": {
        channel: levels[(positions + offset) % len(levels)]
        for offset, channel in enumerate(CHANNELS)
    }}
    before = copy.deepcopy(state)
    model = build_initial_model(
        arrays, np.array([0, 1]), seed=20261007, arm=arm,
        schema=_schema(), identity_state=state, config=DEFAULT_CONFIG,
    )
    original = model.predict(arrays, identity_state=state)
    side_keys = {"history", "mask", "roles", "players", "role_probs", *ARRAY_CHANNELS.values()}
    swapped = {
        key: value[:, ::-1].copy() if key in side_keys else value.copy()
        for key, value in arrays.items()
    }
    swapped_state = {"n_prior": {
        channel: counts[:, ::-1].copy() for channel, counts in state["n_prior"].items()
    }}
    reversed_prediction = model.predict(swapped, identity_state=swapped_state)
    np.testing.assert_allclose(
        reversed_prediction.score_probs,
        original.score_probs[:, [3, 4, 5, 0, 1, 2]], atol=2e-6, rtol=0,
    )
    # An unpermuted sidecar is observably different: constant-count fixtures
    # cannot establish that the causal support actually follows player tokens.
    wrong = model.predict(swapped, identity_state=state)
    assert not np.allclose(
        wrong.score_probs[1:], reversed_prediction.score_probs[1:], atol=1e-8, rtol=0,
    )
    order = [4, 1, 3, 0, 2]
    permuted = {
        key: value[:, :, order].copy() if key in side_keys else value.copy()
        for key, value in arrays.items()
    }
    permuted_state = {"n_prior": {
        channel: counts[:, :, order].copy() for channel, counts in state["n_prior"].items()
    }}
    np.testing.assert_allclose(
        model.predict(permuted, identity_state=permuted_state).score_probs,
        original.score_probs, atol=2e-6, rtol=0,
    )
    for row in (2, 0, 1):
        subset = {key: value[row:row + 1].copy() for key, value in arrays.items()}
        subset_state = {"n_prior": {
            channel: counts[row:row + 1].copy() for channel, counts in state["n_prior"].items()
        }}
        np.testing.assert_allclose(
            model.predict(subset, identity_state=subset_state).score_probs,
            original.score_probs[row:row + 1], atol=2e-6, rtol=0,
        )
        np.testing.assert_allclose(
            model.predict(arrays, np.array([row]), identity_state=state).score_probs,
            original.score_probs[row:row + 1], atol=2e-6, rtol=0,
        )
    with pytest.raises(ValueError, match="identity_state"):
        model.predict(arrays)
    malformed = copy.deepcopy(state)
    malformed["n_prior"]["champion"] = malformed["n_prior"]["champion"][:, :, :, :-1]
    with pytest.raises(ValueError, match="prior"):
        model.predict(arrays, identity_state=malformed)
    subset = {key: value[1:2].copy() for key, value in arrays.items()}
    with pytest.raises(ValueError, match="prior"):
        model.predict(subset, identity_state=state)
    for channel in CHANNELS:
        np.testing.assert_array_equal(state["n_prior"][channel], before["n_prior"][channel])
