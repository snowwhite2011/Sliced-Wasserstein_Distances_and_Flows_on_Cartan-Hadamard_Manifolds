import torch
import torch
import numpy as np
from hswfs_new.tsw import SPDTSW
from tqdm.auto import trange
from torchinterp1d import Interp1d


def get_potential(x_proj, target_proj):
    n_projs = x_proj.shape[0]
    device = x_proj.device
    percentiles = torch.linspace(0, 1, 100)

    L = []
    for k in torch.__version__.split("."):
        try:
            L.append(int(k))
        except:
            L.append(0)

    torch_version = tuple(L)
    # torch_version = tuple([int(k) for k in torch.__version__.split(".")])

    quantiles_x = torch.tensor(np.percentile(x_proj.cpu(), percentiles*100,
                                             axis=1),
                               device=device, dtype=torch.float32).T

    if torch_version >= (1, 13, 0):
        cdf_x = Interp1d.apply(quantiles_x, percentiles.to(device), x_proj).detach()
    else:
        cdf_x = Interp1d()(quantiles_x, percentiles.to(device), x_proj).detach()

    quantiles_target = torch.tensor(np.percentile(target_proj.cpu(),
                                                  percentiles*100, axis=1),
                                    device=device, dtype=torch.float32).T

    if torch_version >= (1, 13, 0):
        x_transported = Interp1d.apply(percentiles.to(device).repeat(n_projs, 1),
                                       quantiles_target, cdf_x).detach()
    else:
        x_transported = Interp1d()(percentiles.to(device).repeat(n_projs, 1),
                                   quantiles_target, cdf_x).detach()

    return x_proj - x_transported


def chtswf(x0, n_epochs, dataiter, manifold, dtype = torch.float64, random_state = 123, p = 2, delta = 1., mass_division = "distance_based", tauk=1e-1, n_projs=50, n_trees = 10, device = 'cuda', projs=None):
    print(mass_division)
    spdtsw = SPDTSW(manifold.d, ntrees=n_trees, nlines=int(n_projs/ n_trees), device=device, dtype=float, random_state=random_state, p=p, delta=delta, mass_division=mass_division,)
    device = x0.device

    L = [x0.cpu().clone()]
    xk = x0.clone()

    pbar = trange(n_epochs)

    for k in pbar:
        target = next(dataiter)
        if len(target) == 2:  # list (samples, labels)
            target = next(dataiter)[0].detach()
        else:
            target = target.detach()
        nabla_SW = spdtsw.spdtsw(xk, target)

        xk = manifold.exp(xk, - tauk * nabla_SW)
        L.append(xk.cpu().clone())

    return L

def chswf(x0, n_epochs, dataiter, manifold, tauk=1e-1, n_projs=50, projs=None):
    device = x0.device

    L = [x0.cpu().clone()]
    xk = x0.clone()

    pbar = trange(n_epochs)

    for k in pbar:
        target = next(dataiter)
        if len(target) == 2:  # list (samples, labels)
            target = next(dataiter)[0].detach()
        else:
            target = target.detach()

        if projs is None:
            v = manifold.sample_geodesics(n_projs).type(target.dtype).to(device)
        else:
            v = projs

        target_proj = manifold.proj(target, v)
        xk_proj = manifold.proj(xk, v)

        d_potential = get_potential(xk_proj, target_proj)
        nabla_proj = manifold.grad_proj(xk, v)
        print("d_potential[:, :, None] shape", d_potential[:, :, None].shape)
        print("nabla_prj shape:", nabla_proj.shape)
        print((d_potential[:, :, None] * nabla_proj).shape)
        nabla_SW = (d_potential[:, :, None] * nabla_proj).mean(dim=0)

        xk = manifold.exp(xk, - tauk * nabla_SW)
        L.append(xk.cpu().clone())

    return L
