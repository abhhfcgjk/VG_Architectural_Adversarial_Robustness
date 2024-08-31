import numpy as np
import torch

from typing import List, Union, Tuple

from icecream import ic

class MixData:
    def __init__(self, gamma=0.1, device='cuda'):
        self.gamma = gamma
        self.device = device
        self.__lam = np.random.beta(gamma, gamma)
        self.shape_label: torch.Tensor
        self.texture_label: torch.Tensor
        self.debiased_label: torch.Tensor
        self.criterion = None

    def calculate_loss(self, prediction, label):
        assert self.criterion is not None
        self.loss_vector = (1. - label[2])*self.criterion(prediction, label[0]) \
              + label[2]*self.criterion(prediction, label[1])
        return self.loss_vector.mean()

    def __call__(self, x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]):
        batch_size = x[0].size()[0]
        indexs = torch.randperm(batch_size).to(self.device)
        x[0][:] = (1-self.__lam)*x[0][:] + self.__lam*x[0][indexs, :]
        y[1][:,:batch_size] = y[0][:,:batch_size]
        y[2][:,:batch_size] = self.__lam
        return x, y
