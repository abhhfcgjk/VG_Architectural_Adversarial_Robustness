import torch
from typing import List
from torch.autograd import Variable
import numpy as np
import cv2
from scipy import ndimage
from torchvision import transforms
from torch import Tensor
from tqdm import tqdm
from .base import Attacker
from torch import optim


class Color(Attacker):
    def __init__(self, 
                 model,
                #  loss_computer,
                 iters: int = 30,
                 device="cuda",
                 metric_range=100,
                 *args, **kwargs):
        super().__init__(model)
        self.device = device
        self.iters = iters
        # self.loss_computer = loss_computer
        self.metric_range = metric_range

    @staticmethod
    def quantization(x):
        x_quan = torch.round(x * 255) / 255
        return x_quan

    # picecwise-linear color filter
    @staticmethod
    def CF(img, param, pieces):

        param = param[:, :, None, None]
        color_curve_sum = torch.sum(param, 4) + 1e-30
        total_image = img * 0
        for i in range(pieces):
            total_image += (
                torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
            )
        total_image *= pieces / color_curve_sum
        return total_image

    def step(self, optimizer, inputs, Paras, pieces, metric_range, device):
        const = 0.5
        batch_size = inputs.shape[0]
        Paras.data = torch.clamp(Paras.data, min=0)
        Paras_sum = torch.sum(Paras.view(batch_size, 3, -1), dim=2, keepdim=True)

        # regularization on the adjustment
        l2 = torch.sum(
            (((Paras / Paras_sum - 1 / pieces) ** 2)).view(batch_size, -1), dim=1
        )

        adv = self.CF(inputs, Paras, pieces)
        adv[adv > 1] = 1
        adv[adv < 0] = 0
        score = (
            self.model(adv.to(device))
        )
        l2_loss = (const * l2).sum()
        sign = 1
        loss = torch.mean(1 - score * sign / metric_range + l2_loss.to(device))
        optimizer.zero_grad()
        loss.backward()
        Paras.grad = torch.nan_to_num(Paras.grad)
        optimizer.step()
        inputs.data.clamp_(0.0, 1.0)

        return self.quantization(adv).detach(), score.detach(), loss.detach()

    def run(self, inputs: Tensor, target: Tensor):
        pieces = 64
        compress_image = Variable(inputs.clone().to(self.device), requires_grad=False)

        batch_size = compress_image.shape[0]
        best_score = -100000
        o_best_adversary = compress_image.clone()

        Paras = torch.ones(batch_size, 3, pieces).to(self.device) * 1 / pieces
        Paras.requires_grad = True
        optimizer = optim.Adam([Paras], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(self.iters):
            # print(iteration, best_score)
            # perform the adversary
            adv, score, loss = self.step(
                optimizer,
                compress_image,
                Paras,
                pieces,
                self.metric_range,
                self.device,
            )
            if (score.max() > best_score):
                best_score = score.max()
                o_best_adversary = adv.clone()

        res_image = (o_best_adversary).data.clamp_(min=0, max=1)
        return res_image