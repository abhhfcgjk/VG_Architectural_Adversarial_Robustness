import numpy as np
import torch


def mix_up(x, y, alpha=0.1, device='cuda'):
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    x = (1-lam)*x + lam*x[index, :]
    y = (y, y[index], (torch.ones(batch_size) * lam).to(device))
    return x, y
