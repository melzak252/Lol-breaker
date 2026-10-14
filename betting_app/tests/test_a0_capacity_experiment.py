"""Native width and uninterrupted trajectory behavioral contracts."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.a0_capacity_experiment import (
    ParametricCommonMapResidual, ParametricNonlinearHistoryModel,
    soft_terminal_loss,
)


@pytest.fixture(scope="module")
def backend():
    from src.models.a0_uncertainty import _activate_archive_backend, _resolve_research_root
    return _activate_archive_backend(_resolve_research_root(None))[0]


def inputs(rows=4):
    generator = torch.Generator().manual_seed(37)
    return {
        'players': torch.randn(rows, 2, 5, 21, generator=generator),
        'context': torch.randn(rows, 2, 70, generator=generator),
        'history': torch.randn(rows, 2, 5, 16, 33, generator=generator),
        'mask': torch.rand(rows, 2, 5, 16, generator=generator) > .4,
    }


@pytest.mark.parametrize('width,total', [(16,2798), (32,6606), (64,17294)])
def test_width_preserves_shapes_symmetry_and_padding_invariance(backend,width,total):
    network = ParametricCommonMapResidual(width)
    assert sum(p.numel() for p in network.parameters()) + 1 == total
    x = inputs()
    z, auxiliary = network(x, return_aux=True)
    assert z.shape == (4,)
    assert auxiliary.shape == (4,2,5,12)
    swapped = {key:value.flip(1) for key,value in x.items()}
    reverse, reverse_aux = network(swapped, return_aux=True)
    torch.testing.assert_close(reverse, -z, atol=1e-6, rtol=0)
    torch.testing.assert_close(reverse_aux, auxiliary.flip(1))
    hidden = {key:value.clone() for key,value in x.items()}
    hidden['history'][~hidden['mask']] = float('nan')
    torch.testing.assert_close(network(hidden),z)
    cold = {key:torch.zeros_like(value) for key,value in x.items()}
    torch.testing.assert_close(network(cold).sigmoid(), torch.full((4,),.5),atol=1e-4,rtol=0)


@pytest.mark.parametrize('seed',[20260910,20260911,20260912])
def test_width32_initialization_and_forward_are_exact_native(backend,seed):
    native = backend['nonlinear_history'].NonlinearHistoryModel(None,SimpleNamespace(seen_champions=np.array([0])),seed=seed)
    model = ParametricNonlinearHistoryModel(None,None,seed=seed,width=32)
    assert list(model.network.state_dict()) == list(native.network.state_dict())
    for key,value in native.network.state_dict().items():
        assert torch.equal(model.network.state_dict()[key],value),key
    assert torch.equal(model.eta,native.eta)
    x=inputs()
    torch.testing.assert_close(model.network(x),native.network(x),atol=0,rtol=0)


def synthetic_batch(self,raw,data,ids,ix,bo,w20=True,org=True,rng=None):
    x = {key:value[ix] for key,value in data.items()}
    # Exercise the gate RNG without changing synthetic inputs.
    rng.random(len(ix))
    return x,torch.zeros(len(ix))


def test_checkpoint_keeps_optimizer_rng_and_uninterrupted_weights(backend,tmp_path):
    model=ParametricNonlinearHistoryModel(None,None,width=32,epochs=2)
    native=backend['nonlinear_history'].NonlinearHistoryModel(None,SimpleNamespace(seen_champions=np.array([0])),epochs=2)
    model.batch=MethodType(synthetic_batch,model)
    native.batch=MethodType(synthetic_batch,native)
    x=inputs()
    ix=np.arange(4);bo=np.array([0,1,2,1]);a=np.array([1,2,1,0]);b=np.array([0,1,3,2])
    targets=np.ones((4,2,5,12),dtype=np.float32);mask=np.ones_like(targets,dtype=bool)
    snapshots=model.fit_trajectory({},x,None,ix,bo,a,b,targets,mask,
        checkpoint_dir=tmp_path,checkpoint_epochs=(1,2),device='cpu')
    from src.models.a0_target_objectives import fit_beta_target_on_device
    fit_beta_target_on_device(native,{},x,None,ix,bo,a,b,targets,mask,device='cpu')
    for key,value in native.network.state_dict().items():
        torch.testing.assert_close(model.network.state_dict()[key],value,atol=0,rtol=0)
    first=torch.load(tmp_path/'checkpoint_epoch1.pt',weights_only=False)
    last=torch.load(tmp_path/'checkpoint_epoch2.pt',weights_only=False)
    for key,value in snapshots[1].network.state_dict().items():
        assert torch.equal(first['network_state_dict'][key],value)
    assert all(float(state['step'])==1 for state in first['optimizer_state_dict']['state'].values())
    assert all(float(state['step'])==2 for state in last['optimizer_state_dict']['state'].values())
    assert first['numpy_rng_state'] != last['numpy_rng_state']
    assert torch.equal(first['torch_rng_state'],last['torch_rng_state'])
    assert snapshots[1].epoch_diagnostics[-1]['finite_gradients']
    assert not torch.equal(snapshots[1].network.head[0].weight,snapshots[2].network.head[0].weight)
    assert snapshots[1].train_losses == model.train_losses[:1]


def test_soft_terminal_teacher_normalization_and_detachment(backend):
    z=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True)
    eta=torch.tensor(-2.,dtype=torch.float64,requires_grad=True)
    bo=torch.tensor([1,3]);teacher=torch.zeros(2,6,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        teacher[0,[0,3]]=.5
        teacher[1,[0,1,3,4]]=.25
    loss=soft_terminal_loss(z,bo,eta,teacher)
    from src.models.a0_target_objectives import beta_objective
    expected=[]
    for row,rounds in enumerate((1,2)):
        values=[]
        for side in (0,1):
            for loser in range(rounds):
                a,b=(rounds,loser) if side==0 else (loser,rounds)
                values.append(beta_objective(z[row:row+1],bo[row:row+1],[a],[b],eta)[0])
        expected.append(torch.stack(values).mean())
    torch.testing.assert_close(loss,torch.stack(expected))
    loss.mean().backward()
    assert teacher.grad is None
    assert torch.isfinite(z.grad).all() and torch.isfinite(eta.grad)
    invalid=teacher.detach().clone();invalid[0,1]=.1
    with pytest.raises(ValueError,match='legal|mass|normal'):
        soft_terminal_loss(z,bo,eta,invalid)
