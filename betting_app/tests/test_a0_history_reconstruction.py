"""Observed-only reconstruction must not invent targets or reveal hidden fields."""
import pytest
import torch

from src.models.a0_history_reconstruction import mask_history_fields, reconstruction_loss


def history_fixture():
    values = torch.zeros(1, 2, 5, 16, 33)
    valid = torch.zeros(1, 2, 5, 16, dtype=torch.bool)
    valid[..., :2] = True
    values[..., :2, 8] = 0.0  # A genuinely measured zero remains a target.
    values[..., :2, 9] = 3.0
    values[..., :2, 20:22] = 1.0
    values[..., :2, 25] = 4.0  # Age/context must survive corruption.
    values[0, 0, 0, 1, 21] = 0.0
    values[0, 0, 0, 1, 9] = float('nan')
    values[..., 2:, :] = float('nan')  # Padding is never evidence.
    return {'history': values, 'mask': valid, 'players': torch.zeros(1, 2, 5, 21)}


def test_hidden_observed_zero_is_target_but_native_missing_and_padding_are_not():
    original = history_fixture()
    chosen = torch.zeros(1, 2, 5, dtype=torch.bool)
    chosen[0, 0, 0] = True
    masked, targets, native, artificial = mask_history_fields(original, chosen)
    assert artificial.sum().item() == 3
    assert native.sum().item() == 39
    assert targets[0, 0, 0, 0, 0].item() == 0.0
    assert masked['history'][0, 0, 0, :2, 8:10].eq(0).all()
    assert masked['history'][0, 0, 0, :2, 20:22].eq(0).all()
    assert masked['history'][0, 0, 0, :2, 25].eq(4).all()
    assert masked['history'][0, 1, 0, 0, 9].item() == 3.0
    assert torch.isfinite(masked['history']).all()
    assert torch.equal(masked['mask'], original['mask'])
    assert original['history'][0, 0, 0, 0, 9].item() == 3.0
    assert not artificial[0, 0, 0, 1, 1]
    assert not native[..., 2:, :].any()


def test_reconstruction_commutes_with_side_and_corruption_swap():
    original = history_fixture()
    chosen = torch.tensor([[[True, False, True, False, True], [False, True, False, True, False]]])
    first = mask_history_fields(original, chosen)
    swapped = {key: value.flip(1) for key, value in original.items()}
    second = mask_history_fields(swapped, chosen.flip(1))
    assert torch.equal(first[0]['history'].flip(1), second[0]['history'])
    for left, right in zip(first[1:], second[1:]):
        assert torch.equal(left.flip(1), right)


def test_loss_ignores_unobserved_cells_and_their_gradients():
    predicted = torch.tensor([1., 4., float('nan')], requires_grad=True)
    target = torch.tensor([0., float('nan'), float('nan')])
    hidden = torch.tensor([True, False, False])
    loss = reconstruction_loss(predicted, target, hidden)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert torch.equal(predicted.grad, torch.tensor([1., 0., 0.]))
    with pytest.raises(ValueError, match='observed|empty|hidden'):
        reconstruction_loss(predicted, target, torch.zeros(3, dtype=torch.bool))


def test_corruption_rejects_broadcasted_player_masks_and_nonfinite_observations():
    original = history_fixture()
    with pytest.raises(ValueError, match='shape'):
        mask_history_fields(original, torch.ones(1, 2, 1, dtype=torch.bool))
    original['history'][0, 0, 0, 0, 8] = float('nan')
    with pytest.raises(ValueError, match='nonfinite|finite'):
        mask_history_fields(original, torch.ones(1, 2, 5, dtype=torch.bool))
