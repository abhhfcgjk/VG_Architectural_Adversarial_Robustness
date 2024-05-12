import models_train.Linearity as Linearity
import models_train.KonCept512 as KonCept512

from models_train.VOneNet import get_model

from torch import nn
import torch
from torchvision import models


from models_train.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU

from typing import Tuple, Union, List, Literal

from icecream import ic

_MODELS = Literal["Linearity", "KonCept"]


class Identity(nn.Module):
    def forward(self, x):
        return x


class Wrap(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.module = features
        self.model = features[0]
        self.index = 0
        self.__resnet50layers_count = 4
        self.layers = [self.model[-1].layer1, self.model[-1].layer2, self.model[-1].layer3, self.model[-1].layer4]
        self.it = [self.model[0], self.model[1]] + self.layers
        self.len = 6

    def __getitem__(self, item):
        if item > 5:
            raise IndexError('index error in wrap')
        elif item < 2:
            return self.model[item]
        else:
            return self.layers[item - 2]

    def __len__(self):
        return self.len

    def __str__(self):
        return self.model.__str__()

    def __iter__(self):
        return iter(self.it)

    def __next__(self):
        if self.index < 2:
            self.index += 1
            return self.model[self.index - 1]
        else:
            self.index += 1
            return self.layers[self.index - 3]

    def forward(self, input):
        # for module in self.model:
        # input = self.module(input)
        return self.module(input)



class IQA(nn.Module):
    @staticmethod
    def extract_adversarial_state_dict(adversarial_path: str):
        sd = torch.load(adversarial_path)['model']
        
        for key in list(sd.keys()):
            if ('attacker' in key) and not ('new_mean' in key or 'new_std' in key):
                sd[key.replace('module.attacker.model.', '')] = sd.pop(key)
            else:
                sd.pop(key)
        return sd

        
    @staticmethod
    def extract_shapa_texture_debiased_state_dict(path: str):
        sd = torch.load(path)["state_dict"]

        for key in list(sd.keys()):
            if 'module' in key:
                sd[key.replace('module.', '')] = sd.pop(key)
        return sd

    def __init__(self, arch, *args, **kwargs):
        super(IQA, self).__init__()
        self.features = None
        self.arch = arch
        if arch == 'wideresnet50':
            features = list(torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).children())
        elif arch == 'vonenet50':
            features = list(get_model(model_arch='resnet50', pretrained=True,
                                      map_location='cuda' if torch.cuda.is_available() else 'cpu').children())
            # features[0][-1].avgpool = Identity()
            # features[0][-1].fc = Identity()
        elif arch == "advresnet50":
            adversarial_path = './LinearityIQA/adversarial_trained/resnet50_imagenet_linf_4.pt'
            adversarial_resnet = models.__dict__[arch.replace('adv', '')]()

            adversarial_state_dict = self.extract_adversarial_state_dict(adversarial_path)

            adversarial_resnet.load_state_dict(adversarial_state_dict)
            features = list(adversarial_resnet.children())
        elif ("debiased" in arch) or ("shape" in arch) or ("texture" in arch):
            assert "resnet50" in arch
            model_type = arch.replace("resnet50", "")
            if model_type == "shape" or model_type == "texture":
                _path = f'./style_transfer_checkpoints/res50-{model_type}-biased.pth'
            elif model_type == "debiased":
                _path = f'./style_transfer_checkpoints/res50-{model_type}.pth'
            else:
                raise TypeError(f"No model type {model_type}.")
            
            resnet_model = models.__dict__["resnet50"]()
            _state_dict = self.extract_shapa_texture_debiased_state_dict(_path)
            resnet_model.load_state_dict(_state_dict, strict=False)

            features = list(resnet_model.children())
        else:
            resnet_model = models.__dict__[arch](pretrained=True)

            features = list(resnet_model.children())
        
        self._base_model_features = features

    def get_features(self, features) -> Tuple[List[Union[int, int]], nn.Module]:
        if self.arch != "vonenet50":
            assert self.__class__.__name__ in _MODELS

            if self.__class__.__name__ == "Linearity":
                features = features[:-2]
            elif self.__class__.__name__ == "KonCept":
                features = features[:-1]

        elif self.arch == "vonenet50":
            if self.__class__.__name__ == "Linearity":
                features[0][-1].avgpool = Identity()
            features[0][-1].fc = Identity()
        
        if self.arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif self.arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif self.arch == 'vonenet50':
            self.id1 = 4
            self.id2 = 5
            in_features = [1024, 2048]
        elif 'res' in self.arch:
            self.id1 = 6
            self.id2 = 7
            if 'resnet18' in self.arch or 'resnet34' in self.arch:
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else:
            raise NotImplementedError(f'The arch {self.arch} is not implemented!')
        
        if self.arch == 'vonenet50':
            return in_features, Wrap(features)
        else:
            return in_features, nn.Sequential(*features)
        
    def get_activation_module(self, activation: str) -> nn.Module:
        if activation == 'silu':
            '''Change activations only in Linearity blocks to SiLU'''
            Activ = nn.SiLU
        elif activation == 'relu_silu':
            '''Change activations only in Linearity blocks to ReLU_SiLU'''
            Activ = ReLU_SiLU
        elif activation == 'Fsilu':
            '''All activations changed to SiLU'''
            ReLU_to_SILU(self.features)
            Activ = nn.SiLU
        elif activation == 'Frelu_silu':
            '''All activations changed to ReLU_SiLU class'''
            ReLU_to_ReLUSiLU(self.features)
            Activ = ReLU_SiLU
        else:
            '''Default activation function'''
            Activ = nn.ReLU
        return Activ

def IQAModel(model_name: str, *args, **kwargs):
    ic(model_name)
    if model_name=="Linearity":
        return Linearity.Linearity(*args, **kwargs)
    elif model_name=="KonCept":
        return KonCept512.KonCept(*args, **kwargs)
    raise NameError(f"No {model_name} model.")
