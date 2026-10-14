"""Consumer invariants for the frozen-native, role-aware series correction."""
from io import BytesIO

import pytest
import torch

from src.models.c0_meta import C0MetaCorrection


PLAYER_FIELDS = ("profile", "player_stats", "player_roles", "player_champions", "player_valid")
META_FIELDS = ("meta_stats", "meta_champions", "meta_support", "meta_patch")
VOCABULARY_SIZE = 7


def inputs(batch_size=6):
    generator = torch.Generator().manual_seed(61007)
    values = {
        "profile": torch.randn(batch_size, 2, 5, 96, generator=generator),
        "player_stats": torch.randn(batch_size, 2, 5, 40, generator=generator),
        "player_roles": torch.softmax(torch.randn(batch_size, 2, 5, 5, generator=generator), -1),
        "player_champions": torch.softmax(torch.randn(batch_size, 2, 5, VOCABULARY_SIZE, generator=generator), -1),
        "player_valid": torch.ones(batch_size, 2, 5, dtype=torch.bool),
        "meta_stats": torch.randn(batch_size, 2, 5, 40, generator=generator),
        "meta_champions": torch.softmax(torch.randn(batch_size, 2, 5, VOCABULARY_SIZE, generator=generator), -1),
        "meta_support": torch.rand(batch_size, 2, 5, 2, generator=generator),
        "meta_patch": torch.rand(batch_size, 2, 5, 2, generator=generator),
        "base_probability": torch.linspace(.05, .95, batch_size),
        "best_of": torch.tensor(([1, 3, 5, 7, 9, 11] * batch_size)[:batch_size]),
    }
    return values


def active_model(use_meta=True):
    torch.manual_seed(117)
    model = C0MetaCorrection(VOCABULARY_SIZE, use_meta=use_meta)
    # A real nonzero learned state, not a mocked forward implementation.
    with torch.no_grad():
        model.readout.weight.copy_(torch.linspace(-.7, .9, model.readout.weight.numel()).reshape_as(model.readout.weight))
    return model


def clone_inputs(values):
    return {name: value.clone() for name, value in values.items()}


@pytest.mark.parametrize("use_meta", [False, True])
def test_zero_initialized_head_preserves_native_series_probability_for_every_best_of(use_meta):
    model = C0MetaCorrection(VOCABULARY_SIZE, use_meta=use_meta)
    values = inputs()
    torch.testing.assert_close(model(values), torch.logit(values["base_probability"]), rtol=0, atol=0)
    torch.testing.assert_close(model.predict_proba(values), values["base_probability"], rtol=0, atol=6e-8)
    endpoints = clone_inputs(values)
    endpoints["base_probability"][:2] = torch.tensor([0., 1.])
    torch.testing.assert_close(model.predict_proba(endpoints)[:2], endpoints["base_probability"][:2], rtol=0, atol=0)
    changed_bo = clone_inputs(values)
    changed_bo["best_of"][:] = 1
    torch.testing.assert_close(active_model(use_meta)(values), active_model(use_meta)(changed_bo), rtol=0, atol=0)


@pytest.mark.parametrize("use_meta", [False, True])
def test_swapping_teams_complements_series_probability_and_negates_logit(use_meta):
    model = active_model(use_meta)
    values = inputs()
    swapped = clone_inputs(values)
    for name in PLAYER_FIELDS:
        swapped[name] = values[name].flip(1)
    swapped["base_probability"] = 1 - values["base_probability"]
    # The meta axis is global/local context, not team A/team B.
    torch.testing.assert_close(model(values), -model(swapped), rtol=1e-6, atol=5e-7)
    torch.testing.assert_close(model.predict_proba(values) + model.predict_proba(swapped), torch.ones(6), rtol=0, atol=1e-7)


@pytest.mark.parametrize("use_meta", [False, True])
def test_arbitrary_independent_player_permutations_do_not_change_series_logits(use_meta):
    model = active_model(use_meta)
    values = inputs()
    reordered = clone_inputs(values)
    for name in PLAYER_FIELDS:
        reordered[name][:, 0] = values[name][:, 0, [4, 1, 3, 0, 2]]
        reordered[name][:, 1] = values[name][:, 1, [2, 4, 0, 3, 1]]
    torch.testing.assert_close(model(values), model(reordered), rtol=1e-6, atol=3e-7)


def test_logit_correction_is_bounded_even_with_large_learned_readout():
    model = active_model()
    with torch.no_grad():
        model.readout.weight.mul_(100000.)
    values = inputs()
    delta = model(values) - torch.logit(values["base_probability"])
    assert torch.isfinite(delta).all()
    assert (delta.abs() <= .500001).all()
    assert (delta.abs() > .1).any()


@pytest.mark.parametrize("use_meta", [False, True])
def test_unknown_players_are_masked_and_both_cold_is_exact_native(use_meta):
    model = active_model(use_meta)
    values = inputs()
    values["player_valid"][:, :, 2:] = False
    dirty = clone_inputs(values)
    for name in PLAYER_FIELDS[:-1]:
        dirty[name][:, :, 2:] = float("nan")
    torch.testing.assert_close(model(values), model(dirty), rtol=0, atol=0)
    cold = clone_inputs(dirty)
    cold["player_valid"][:] = False
    for name in PLAYER_FIELDS[:-1]:
        cold[name][:] = float("nan")
    torch.testing.assert_close(model(cold), torch.logit(cold["base_probability"]), rtol=0, atol=0)
    cold["meta_stats"] *= 1000
    torch.testing.assert_close(model(cold), torch.logit(cold["base_probability"]), rtol=0, atol=0)


def test_meta_has_nonzero_differential_gradient_for_equal_role_mixtures_but_different_players():
    model = active_model()
    values = inputs(batch_size=1)
    # A context-only common offset would cancel here: the sides have identical
    # role mixtures, champion frequencies, and recent summaries, but different
    # frozen individual profiles.
    for name in ("player_stats", "player_roles", "player_champions"):
        values[name][:, 1] = values[name][:, 0]
    values["meta_stats"].requires_grad_(True)
    gradient = torch.autograd.grad(model(values).sum(), values["meta_stats"])[0]
    assert gradient.abs().sum() > 1e-6
    perturbed = clone_inputs(values)
    perturbed["meta_stats"] = perturbed["meta_stats"] + .2
    assert (model(values) - model(perturbed)).abs().max() > 1e-6
    # If player profiles also become identical, shared context must cancel.
    identical = clone_inputs(values)
    identical["profile"][:, 1] = identical["profile"][:, 0]
    torch.testing.assert_close(model(identical), torch.logit(identical["base_probability"]), rtol=0, atol=0)


@pytest.mark.parametrize("field", META_FIELDS)
def test_every_supplied_meta_channel_participates_in_the_learned_correction(field):
    model = active_model()
    values = inputs()
    values[field].requires_grad_(True)
    gradient = torch.autograd.grad(model(values).sum(), values[field])[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 1e-6


def test_player_only_ignores_all_meta_inputs():
    model = active_model(use_meta=False)
    values = inputs()
    changed = clone_inputs(values)
    for name in META_FIELDS:
        changed[name][:] = float("nan")
    torch.testing.assert_close(model(values), model(changed), rtol=0, atol=0)
    exported = model.export_role_meta(changed)
    assert exported.shape == (6, 2, 5, 16)
    torch.testing.assert_close(exported, torch.zeros_like(exported), rtol=0, atol=0)


def test_role_meta_export_and_real_state_dict_serialization_roundtrip():
    model = active_model()
    values = inputs()
    exported = model.export_role_meta(values)
    assert exported.shape == (6, 2, 5, 16)
    assert not exported.requires_grad
    assert torch.isfinite(exported).all()
    changed = clone_inputs(values)
    changed["meta_stats"] += .5
    assert (exported - model.export_role_meta(changed)).abs().max() > 1e-6
    checkpoint = BytesIO()
    torch.save({"vocabulary_size": VOCABULARY_SIZE, "use_meta": True, "state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = C0MetaCorrection(saved["vocabulary_size"], saved["use_meta"])
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(values), model(values), rtol=0, atol=0)
    torch.testing.assert_close(restored.export_role_meta(values), exported, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["profile", "player_stats", "player_roles", "player_champions", "player_valid", "meta_stats", "meta_champions", "meta_support", "meta_patch", "base_probability", "best_of"])
def test_misaligned_input_shapes_are_rejected(field):
    values = inputs()
    values[field] = values[field][..., :1]
    with pytest.raises(ValueError):
        C0MetaCorrection(VOCABULARY_SIZE, True)(values)


def test_support_mask_requires_boolean_values():
    values = inputs()
    values["player_valid"] = values["player_valid"].float()
    with pytest.raises(ValueError):
        C0MetaCorrection(VOCABULARY_SIZE, True)(values)
