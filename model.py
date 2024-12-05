from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision as tv
from torchvision.ops import RoIPool, RoIAlign
import numpy as np
from torch.nn import ReLU, SiLU, ELU, GELU

from inceptionresnetv2 import inceptionresnetv2

from activ import swap_all_activations, ReLU_ELU, ReLU_SiLU


class KonCept512(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(KonCept512, self).__init__()

        self.Activ = None

        base_model = inceptionresnetv2(num_classes=1000, pretrained="imagenet")
        self.base = nn.Sequential(*list(base_model.children())[:-1])

        self.__set_activation(activation=kwargs.get('activation'))

        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            self.Activ(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            self.Activ(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            self.Activ(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

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
        else:
            activation = "relu"
            self.Activ = ReLU
        print(f"Activation: {activation}")

    def forward(self, x):
        x = self.base(x)
        x = nn.functional.avg_pool2d(x, x.size()[-2:])
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
