"""
UAP: Universal adversarial perturbations.

A universal adversarial perturbation (UAP) is a single perturbation that can fool a model on a large number of inputs.
This attack generates a UAP by accumulating gradients over a dataset.
"""

from typing import Callable, Iterable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List

from .base import Attacker

class UAP(Attacker):
    """Cumulative UAP Attacker.

    :param model: The model to be attacked.
    :param device: The device on which to train the attack.
    :param loss_computer: Loss function to be used for the attack.
    :param eps: Upper bound for linf norm of perturbations.
    :param lr: Learning rate to update the perturbation.
    :param n_epochs: Number of epochs to train the attack.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_computer,
        device: torch.device = "cuda",
        eps: float = 10.0,
        lr: float = 0.0001,
        iters: int = 10,
        **kwargs
    ) -> None:
        super().__init__(model)
        self.eps = eps / 255
        self.lr = lr
        self.n_epochs = iters
        self.loss_computer = loss_computer
        self.device = device

    def generate(self, dataloader: Iterable) -> None:
        data_batch = next(iter(dataloader))
        # print(len(data_batch), data_batch)
        image_size = data_batch[0]['data'].shape[1:]
        self.perturbation = torch.zeros(image_size, device=self.device)
        self.perturbation.unsqueeze_(0)

        self.model.eval()
        self.perturbation.requires_grad = True
        optimizer = torch.optim.Adam([self.perturbation], lr=self.lr)

        for _ in tqdm(range(self.n_epochs)):
            for data_batch in dataloader:
                # print(len(data_batch))
                inputs = data_batch[0]['data']
                inputs = inputs.to(self.device)

                attacked_inputs = inputs + self.perturbation
                attacked_inputs.clamp_(0.0, 1.0)
                outputs = self.model(attacked_inputs)
                loss = -self.loss_computer(outputs, None)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.perturbation.data.clamp_(-self.eps, self.eps)

        self.perturbation.detach_()

    def run(self, inputs: Tensor, target: Tensor) -> Tensor:
        if self.perturbation is None:
            raise ValueError("UAP perturbation is not generated yet. Call `generate` method first.")

        attacked_inputs = inputs + self.perturbation
        attacked_inputs.clamp_(0.0, 1.0)
        return attacked_inputs
    
    # """Linearity"""
    # def loss_fn(self, output):
    #     loss = 1 - (output[-1] * self.k[0] + self.b[0]) / self.metric_range
    #     return torch.mean(loss)