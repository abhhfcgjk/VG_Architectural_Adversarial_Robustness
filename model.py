from collections import OrderedDict
from unittest.mock import seal

import torch
import torch.nn as nn
import torchvision as tv
from torchvision.ops import RoIPool, RoIAlign
import numpy as np
from torch.nn import ReLU, SiLU, ELU, GELU
from Cayley import CayleyBlockPool
from pathlib import Path


from inceptionresnetv2 import inceptionresnetv2

from activ import swap_all_activations, ReLU_ELU, ReLU_SiLU
from pruning.pruning import l1_prune, ln_prune, pls_prune, displs_prune, hsic_prune


class KonCept512(nn.Module):
    def print_sparcity(self):
        """only for resnet"""
        print("SPARCITY")
        p_list = self.prune_parameters
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))
    def __init__(self, num_classes, **kwargs):
        super(KonCept512, self).__init__()

        self.Activ = None
        self.db_model_dir = kwargs.get('db_model')
        self.is_cayley1 = kwargs.get('cayley1', False)
        self.is_cayley2 = kwargs.get('cayley2', False)
        self.prune_amount = kwargs.get('prune', 0.0)

        base_model = inceptionresnetv2(num_classes=1000, pretrained="imagenet")
        self.base = nn.Sequential(*list(base_model.children())[:-1])

        if self.is_cayley1:
            print(self)
            self.cayley1 = CayleyBlockPool(in_channels=2080, intermed_channels=1500, 
                                          stride=1, padding=0, kernel_size=3)
            self.base = nn.Sequential(*list(self.base.children())[:-3],
                                      self.cayley1,
                                      *list(self.base.children())[-3:])
        elif self.is_cayley2:
            print(self)
            self.cayley2 = CayleyBlockPool(in_channels=2080, intermed_channels=1500, 
                                          stride=1, padding=0, kernel_size=3)
            self.base = nn.Sequential(*list(self.base.children())[:-2],
                                      self.cayley2,
                                      *list(self.base.children())[-2:])

        self.__set_activation(activation=kwargs.get('activation'))

        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            self.Activ(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            self.Activ(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            self.Activ(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        # if self.prune_amount > 0:
        #     self.__load_pretrained()
        #     self.prune(amount=kwargs['prune'], 
        #                         prtype=kwargs['prune_type'],
        #                         width=256,
        #                         height=192,
        #                         images_count=100,
        #                         kernel=1)
        #     self.print_sparcity()

    def load_pretrained(self, path):
        new_state_dict = {}
        self.db_model_dir = Path(self.db_model_dir)
        # checkpoint = torch.load(self.db_model_dir / 'koncept-activation=relu.pth')['model']
        checkpoint = torch.load(path)
        for key, value in checkpoint.items():
            new_key = key.replace('model.', '')  # Adjust as necessary
            new_state_dict[new_key] = value
        self.load_state_dict(new_state_dict, strict=False)

    def prune(self, amount, prtype='l1', **kwargs):
        self.prune_parameters: tuple
        self.pruning_type = prtype

        if prtype == 'l1':
            self.prune_parameters = l1_prune(self, amount)
        elif prtype == 'pls':
            self.prune_parameters = pls_prune(self, amount, **kwargs)
        elif prtype == 'displs':
            self.prune_parameters = displs_prune(self, amount, **kwargs)
        elif prtype == 'hsic':
            self.prune_parameters = hsic_prune(self, amount, **kwargs)
        elif prtype == 'l2':
            self.prune_parameters = ln_prune(self, amount, 2)

    def __set_activation(self, activation: str = "relu"):
        if activation == "Fsilu":
            swap_all_activations(self.base, ReLU, SiLU)
            self.Activ = SiLU
        elif activation == "Felu":
            swap_all_activations(self.base, ReLU, ELU)
            self.Activ = ELU
        elif activation == "Fgelu":
            swap_all_activations(self.base, ReLU, GELU)
            self.Activ = GELU
        elif activation == "Frelu_elu":
            swap_all_activations(self.base, ReLU, ReLU_ELU)
            self.Activ = ReLU_ELU
        elif activation == "Frelu_silu":
            swap_all_activations(self.base, ReLU, ReLU_SiLU)
            self.Activ = ReLU_SiLU
        else:
            activation = "relu"
            self.Activ = ReLU
        print(f"Activation: {activation}")

    def forward(self, x):
        x = self.base(x)
        x = nn.functional.avg_pool2d(x, x.size()[-2:])
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ImageNormalizer(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
        std: tuple[float, float, float] = [0.229, 0.224, 0.225],
        persistent: bool = True,
    ):
        super(ImageNormalizer, self).__init__()

        self.register_buffer(
            "mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=persistent
        )
        self.register_buffer(
            "std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=persistent
        )

    def forward(self, inputs: torch.Tensor):
        return (inputs - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
    std: tuple[float, float, float] = [0.229, 0.224, 0.225],
):
    # layers = OrderedDict([("crop", tv.transforms.Resize([691, 921])), ("normalize", ImageNormalizer(mean, std)), ("model", model)])
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)
