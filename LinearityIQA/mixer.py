import numpy as np
import torch

from typing import List, Union

from icecream import ic

class MixData:
    def __init__(self, gamma=0.1, device='cuda'):
        self.gamma = gamma
        self.device = device
        self.__lam = np.random.beta(gamma, gamma)
        self.shape_label: torch.Tensor
        self.texture_label: torch.Tensor
        self.debiased_label: torch.Tensor
        self.__criterion = None

    def calculate_loss(self, prediction):
        assert self.__criterion
        self.loss_vector = (1. - self.__lam)*self.__criterion(prediction,self.shape_label) + self.__lam*self.__criterion(prediction, self.texture_label)
        return self.loss_vector.mean()

    def __call__(self, x: torch.Tensor, y: Union[List,torch.Tensor]):
        batch_size = x.size()[0]
        indexs = torch.randperm(batch_size).to(self.device)
        x = (1-self.__lam)*x + self.__lam*x[indexs, :]
        y= y[0]
        self.shape_label = y
        self.texture_label = y[indexs]
        self.debiased_label = (torch.ones(batch_size) * self.__lam).to(self.device)
        return x, self.debiased_label

    @property
    def criterion(self):
        return self.__criterion
    
    @criterion.setter
    def criterion(self, value):
        self.__criterion = value
