"""Consumer contracts for the fixed-window, activity-aware C0 correction."""
from io import BytesIO

import pytest
import torch

from src.models.c0_meta_stable import StableC0MetaCorrection


VOCABULARY_SIZE = 7
PLAYER_FIELDS = (
    "profile", "player_stats", "player_roles", "player_champions",
    "player_valid", "player_activity",
)
META_FIELDS = ("meta_stats", "meta_champions", "meta_support", "meta_patch")


def inputs(batch_size=6):
    generator = torch.Generator().manual_seed(61007)
    activity = torch.zeros(batch_size, 2, 5, 8)
    activity[..., :3] = torch.rand(batch_size, 2, 5, 3, generator=generator)
    activity[..., 3:] = 1.
    return {
        "profile": torch.randn(batch_size, 2, 5, 96, generator=generator),
        "player_stats": torch.randn(batch_size, 2, 5, 40, generator=generator),
        "player_roles": torch.softmax(torch.randn(batch_size, 2, 5, 5, generator=generator), -1),
        "player_champions": torch.softmax(torch.randn(batch_size, 2, 5, VOCABULARY_SIZE, generator=generator), -1),
        "player_valid": torch.ones(batch_size, 2, 5, dtype=torch.bool),
        "player_activity": activity,
        "meta_stats": torch.randn(batch_size, 2, 5, 40, generator=generator),
        "meta_champions": torch.softmax(torch.randn(batch_size, 2, 5, VOCABULARY_SIZE, generator=generator), -1),
        "meta_support": torch.rand(batch_size, 2, 5, 2, generator=generator),
        "meta_patch": torch.rand(batch_size, 2, 5, 2, generator=generator),
        "base_probability": torch.linspace(.05, .95, batch_size),
        "best_of": torch.tensor(([1, 3, 5, 7, 9, 11] * batch_size)[:batch_size]),
    }


def clone_inputs(values):
    return {name: value.clone() for name, value in values.items()}


def active_model(use_meta=True):
    torch.manual_seed(117)
    model = StableC0MetaCorrection(VOCABULARY_SIZE, use_meta)
    with torch.no_grad():
        model.readout.weight.copy_(torch.linspace(-.7, .9, 16).reshape_as(model.readout.weight))
    return model


def routed_model(input_channel):
    """Use a genuine learned state to isolate one player-encoder channel."""
    model = StableC0MetaCorrection(VOCABULARY_SIZE, False)
    with torch.no_grad():
        model.player_encoder.weight.zero_()
        model.player_encoder.bias.zero_()
        model.player_encoder.weight[0, input_channel] = 1.
        model.readout_hidden.weight.zero_()
        model.readout_hidden.bias.zero_()
        model.readout_hidden.weight[0, 0] = 1.
        model.readout.weight.zero_()
        model.readout.weight[0, 0] = 1.
    return model


@pytest.mark.parametrize("use_meta", [False, True])
@pytest.mark.parametrize("long_support", [False, True])
def test_native_profile_is_retained_without_recent_support(use_meta, long_support):
    model = active_model(use_meta)
    values = inputs()
    values["player_activity"][..., 0] = 0.
    values["player_activity"][..., 4] = 0.
    values["player_activity"][..., 5] = float(long_support)
    values["player_activity"][..., 6] = 1.
    values["profile"].requires_grad_(True)
    gradient = torch.autograd.grad(model(values).sum(), values["profile"])[0]
    assert gradient.abs().sum() > 1e-6
    changed = clone_inputs(values)
    changed["profile"] = torch.zeros_like(values["profile"])
    assert (model(values) - model(changed)).abs().max() > 1e-6


@pytest.mark.parametrize("use_meta", [False, True])
def test_long_supported_player_never_uses_unsupported_native_profile(use_meta):
    model = active_model(use_meta)
    values = inputs()
    values["player_activity"][..., 4] = 0.
    values["player_activity"][..., 5] = 1.
    values["player_activity"][..., 6] = 0.
    values["profile"].requires_grad_(True)
    gradient = torch.autograd.grad(model(values).sum(), values["profile"])[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
    dirty = clone_inputs(values)
    dirty["profile"][:] = float("nan")
    torch.testing.assert_close(model(values), model(dirty), rtol=0, atol=0)
    changed = clone_inputs(values)
    changed["player_stats"] += .5
    assert (model(values) - model(changed)).abs().max() > 1e-6


@pytest.mark.parametrize("use_meta", [False, True])
@pytest.mark.parametrize("identity_known", [False, True])
def test_no_usable_representation_preserves_base_for_cold_and_known_identities(use_meta, identity_known):
    model = active_model(use_meta)
    values = inputs()
    values["player_valid"][:] = False
    values["player_activity"].zero_()
    values["player_activity"][..., 3] = float(identity_known)
    values["player_activity"][..., 7] = float(identity_known)
    for field in ("profile", "player_stats", "player_roles", "player_champions"):
        values[field][:] = float("nan")
    torch.testing.assert_close(model(values), torch.logit(values["base_probability"]), rtol=0, atol=0)
    values["player_activity"][:] = float("nan")
    torch.testing.assert_close(model(values), torch.logit(values["base_probability"]), rtol=0, atol=0)


def test_activity_flags_distinguish_native_supported_inactivity_from_unknown_history():
    # Both identities have a usable native profile but only A has raw prior
    # history, so the history-known flag must reach the learned correction.
    model = routed_model(149 + 3)
    values = inputs(batch_size=1)
    values["player_activity"].zero_()
    values["player_activity"][..., 6:8] = 1.
    values["player_activity"][:, 0, :, 3] = 1.
    assert model(values).item() > torch.logit(values["base_probability"]).item()
    unknown = clone_inputs(values)
    unknown["player_activity"][..., 3] = 0.
    torch.testing.assert_close(model(unknown), torch.logit(unknown["base_probability"]), rtol=0, atol=0)


@pytest.mark.parametrize("use_meta", [False, True])
def test_swap_permutation_bounded_residual_and_input_nonmutation(use_meta):
    model = active_model(use_meta)
    values = inputs()
    values["player_valid"][:, 0, 4] = False
    values["player_activity"][:, 1, 2, 6] = 0.
    original = clone_inputs(values)
    result = model(values)
    delta = result - torch.logit(values["base_probability"])
    assert (delta.abs() <= .500001).all()
    assert (delta.abs() > 1e-6).any()
    swapped = clone_inputs(values)
    for name in PLAYER_FIELDS:
        swapped[name] = values[name].flip(1)
    swapped["base_probability"] = 1 - values["base_probability"]
    torch.testing.assert_close(result, -model(swapped), rtol=1e-6, atol=5e-7)
    reordered = clone_inputs(values)
    for name in PLAYER_FIELDS:
        reordered[name][:, 0] = values[name][:, 0, [4, 1, 3, 0, 2]]
        reordered[name][:, 1] = values[name][:, 1, [2, 4, 0, 3, 1]]
    torch.testing.assert_close(result, model(reordered), rtol=1e-6, atol=3e-7)
    for name in values:
        torch.testing.assert_close(values[name], original[name], rtol=0, atol=0)


@pytest.mark.parametrize("use_meta", [False, True])
def test_zero_initialized_head_and_target_best_of_do_not_reproject_series_probability(use_meta):
    values = inputs()
    model = StableC0MetaCorrection(VOCABULARY_SIZE, use_meta)
    torch.testing.assert_close(model(values), torch.logit(values["base_probability"]), rtol=0, atol=0)
    torch.testing.assert_close(model.predict_proba(values), values["base_probability"], rtol=0, atol=6e-8)
    values["base_probability"][:2] = torch.tensor([0., 1.])
    torch.testing.assert_close(model.predict_proba(values)[:2], values["base_probability"][:2], rtol=0, atol=0)
    learned = active_model(use_meta)
    changed = clone_inputs(values)
    changed["best_of"][:] = 1
    torch.testing.assert_close(learned(values), learned(changed), rtol=0, atol=0)


def test_player_only_branch_ignores_nan_meta_and_exports_zero_role_embeddings():
    model = active_model(False)
    values = inputs()
    dirty = clone_inputs(values)
    for name in META_FIELDS:
        dirty[name][:] = float("nan")
    torch.testing.assert_close(model(values), model(dirty), rtol=0, atol=0)
    exported = model.export_role_meta(dirty)
    assert exported.shape == (6, 2, 5, 16)
    torch.testing.assert_close(exported, torch.zeros_like(exported), rtol=0, atol=0)


@pytest.mark.parametrize("use_meta", [False, True])
def test_public_serialized_state_replays_forecasts_and_role_meta(use_meta):
    model = active_model(use_meta)
    values = inputs()
    exported = model.export_role_meta(values)
    assert not exported.requires_grad
    checkpoint = BytesIO()
    torch.save({"vocabulary_size": VOCABULARY_SIZE, "use_meta": use_meta, "state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = StableC0MetaCorrection(saved["vocabulary_size"], saved["use_meta"])
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(values), model(values), rtol=0, atol=0)
    torch.testing.assert_close(restored.export_role_meta(values), exported, rtol=0, atol=0)
    if use_meta:
        changed = clone_inputs(values)
        changed["meta_stats"] += .5
        assert (exported - model.export_role_meta(changed)).abs().max() > 1e-6


@pytest.mark.parametrize("field", PLAYER_FIELDS + META_FIELDS + ("base_probability", "best_of"))
def test_misaligned_input_shapes_are_rejected(field):
    values = inputs()
    values[field] = values[field][..., :1]
    with pytest.raises(ValueError):
        StableC0MetaCorrection(VOCABULARY_SIZE, True)(values)


@pytest.mark.parametrize("activity", [None, torch.zeros(6, 2, 5, 8, dtype=torch.int64), torch.zeros(6, 2, 5, 8, dtype=torch.float64)])
def test_activity_requires_tensor_with_model_float_dtype(activity):
    values = inputs()
    values["player_activity"] = activity
    with pytest.raises(ValueError, match="player_activity"):
        StableC0MetaCorrection(VOCABULARY_SIZE, True)(values)


def test_mapping_and_boolean_support_schema_are_inherited():
    model = StableC0MetaCorrection(VOCABULARY_SIZE, True)
    with pytest.raises(ValueError, match="mapping"):
        model([])
    values = inputs()
    values["player_valid"] = values["player_valid"].float()
    with pytest.raises(ValueError, match="boolean"):
        model(values)
