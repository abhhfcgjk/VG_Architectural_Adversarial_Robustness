from models_train.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU, swap_all_activations, ReLU_ELU, ReLU_GELU
from models_train.VOneNet import get_model
from models_train.inceptionresnet import inceptionresnetv2
from models_train.Cayley import get_lipschitz_model, swap_conv_to_lipschitz
from models_train import LipSim

from torch import nn
import torch
from torchvision import models

from typing import Tuple, Union, List, Literal

from icecream import ic
import os
import yaml

_MODELS = ["Linearity", "KonCept"]

YAML_PATH = './path_config.yaml'

class Identity(nn.Module):
    def forward(self, x):
        return x


class Wrap(nn.Module):
    """Wrap to train with VOneNet"""
    def __init__(self, features, layers_count=4):
        super().__init__()
        self.module = features
        self.model = features[0]
        self.index = 0
        # self.__resnet50layers_count = 4
        self.layers_count = layers_count
        self.layers = []
        for i in range(1, layers_count+1):
            # self.layers = [self.model[-1].layer1, self.model[-1].layer2, self.model[-1].layer3, self.model[-1].layer4]
            self.layers.append(getattr(self.model[-1], f'layer{i}'))
        self.it = [self.model[0], self.model[1]] + self.layers
        self.len = self.layers_count + 2

    def __getitem__(self, item):
        if item > self.len-1:
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
        # if arch == "inceptionresnet":
        #     return

        if arch == 'wideresnet50':
            features = list(torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).children())
        elif arch == 'wideresnet101':
            features = list(torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True).children())
        elif arch == 'vonenet50':
            features = list(get_model(model_arch='resnet50', pretrained=True,
                                      map_location='cuda' if torch.cuda.is_available() else 'cpu').children())
        
        elif arch == 'vonenet101':
            server_mnt = "~/mnt/dione/28i_mel"
            destination_path = os.path.expanduser(server_mnt)
            weightsdir =os.path.join(destination_path, 'vonenet_resnet101.pth.tar')
            ic(weightsdir)
            features = list(get_model(model_arch='resnet101', pretrained=True,weightsdir=weightsdir,
                                      map_location='cuda' if torch.cuda.is_available() else 'cpu').children(),
                                      )
            ic(features)
        
        elif arch == 'lipsim':
            server_mnt = "~/mnt/dione/28i_mel"
            destination_path = os.path.expanduser(server_mnt)
            weightsdir =os.path.join(destination_path, 'model.ckpt-1.pth')
            # features = list(LipSim.L2LipschitzNetwork(1).children())
            # ic(len(features))
            lipsim = LipSim.L2LipschitzNetwork(1, depth=20,depth_linear=7, num_channels=45, n_features=1024, conv_size=5)
            ckpt = torch.load(weightsdir)['model_state_dict']
            for key in list(ckpt.keys()):
                ckpt[key.replace('module.model.', '')] = ckpt[key]
                del ckpt[key]
            lipsim.load_state_dict(ckpt)
            features = LipSim.extract_features(lipsim, step_back=2)
            ic(features)
        elif arch == 'lipsim2':
            server_mnt = "~/mnt/dione/28i_mel"
            destination_path = os.path.expanduser(server_mnt)
            weightsdir =os.path.join(destination_path, 'model.ckpt-1.pth')
            # features = list(LipSim.L2LipschitzNetwork(1).children())
            # ic(len(features))
            lipsim = LipSim.L2LipschitzNetwork(1, depth=20,depth_linear=7, num_channels=45, n_features=1024, conv_size=5)
            ckpt = torch.load(weightsdir)['model_state_dict']
            for key in list(ckpt.keys()):
                ckpt[key.replace('module.model.', '')] = ckpt[key]
                del ckpt[key]
            lipsim.load_state_dict(ckpt)
            features = LipSim.extract_features(lipsim, step_back=None)
            ic(features)

        elif arch == "advresnet50":
            # adversarial_path = './LinearityIQA/adversarial_trained/resnet50_imagenet_linf_4.pt'
            with open(YAML_PATH, 'r') as file:
                yaml_file = yaml.safe_load(file)
            adversarial_path = yaml_file['checkpoints']['adversarial']
            # adversarial_path = yaml.dump()
            adversarial_resnet = models.__dict__[arch.replace('adv', '')]()

            adversarial_state_dict = self.extract_adversarial_state_dict(adversarial_path)

            adversarial_resnet.load_state_dict(adversarial_state_dict)
            features = list(adversarial_resnet.children())
        elif arch == 'advresnet101':
            adv_resnet = models.__dict__['resnet101']()
            server_mnt = "~/mnt/dione/28i_mel"
            destination_path = os.path.expanduser(server_mnt)
            # weightsdir =os.path.join(destination_path, 'linearity-apgd-ssim-8.pth')
            weightsdir =os.path.join(destination_path, 'apgd_ssim_eps=2.pth')
            sd = adv_resnet.state_dict()
            ckpt = torch.load(weightsdir)['model']
            # print(ckpt)
            # for key in list(ckpt.keys()):
            #     if not key in list(sd.keys()):
            #         del ckpt[key]
            ic(len(ckpt.keys()))
            ic(len(sd.keys()))
            adv_resnet.load_state_dict(ckpt)
        elif arch == 'debiasedresnet101':
            assert "resnet101" in arch
            
            with open(YAML_PATH, 'r') as file:
                yaml_file = yaml.safe_load(file)
            model_type = arch.replace("resnet", "")
            
            _path = yaml_file['checkpoints'][model_type]
            resnet_model = models.__dict__["resnet101"]()
            _state_dict = self.extract_shapa_texture_debiased_state_dict(_path)
            resnet_model.load_state_dict(_state_dict, strict=False)

            features = list(resnet_model.children())
        elif ("debiased" in arch) or ("shape" in arch) or ("texture" in arch):
            assert "resnet50" in arch
            with open(YAML_PATH, 'r') as file:
                yaml_file = yaml.safe_load(file)
            model_type = arch.replace("resnet50", "")
            # if model_type == "shape" or model_type == "texture":
            #     _path = f'./style_transfer_checkpoints/res50-{model_type}-biased.pth'
            # elif model_type == "debiased":
            #     _path = f'./style_transfer_checkpoints/res50-{model_type}.pth'
            # else:
            #     raise TypeError(f"No model type {model_type}.")
            _path = yaml_file['checkpoints'][model_type]
            
            resnet_model = models.__dict__["resnet50"]()
            _state_dict = self.extract_shapa_texture_debiased_state_dict(_path)
            resnet_model.load_state_dict(_state_dict, strict=False)

            features = list(resnet_model.children())
        elif arch == "inceptionresnet":
            features = list(inceptionresnetv2(num_classes=1000, pretrained='imagenet').children())
            ic("inceptionresnet")
        else:
            resnet_model = models.__dict__[arch](pretrained=True)

            features = list(resnet_model.children())
        
        self._base_model_features = features
        ic(len(self._base_model_features))

    def get_features(self, features) -> Tuple[List[Union[int, int]], nn.Module]:

        assert self.__class__.__name__ in _MODELS
        if self.arch == 'lipsim' or self.arch=='lipsim2':
            features = features
        elif self.arch == "inceptionresnet":
            features = features[:-1]
        elif self.arch != "vonenet50" and self.arch != "vonenet101":
            if self.__class__.__name__ == "Linearity":
                features = features[:-2]
            elif self.__class__.__name__ == "KonCept":
                features = features[:-1]

        elif self.arch == "vonenet50" or self.arch == "vonenet101":
            if self.__class__.__name__ == "Linearity":
                features[0][-1].avgpool = Identity()
            features[0][-1].fc = Identity()
        
        # if self.arch == 'alexnet':
        #     in_features = [256, 256]
        #     self.id1 = 9
        #     self.id2 = 12
        #     features = features[0]
        # elif self.arch == 'vgg16':
        #     in_features = [512, 512]
        #     self.id1 = 23
        #     self.id2 = 30
        #     features = features[0]
        if self.arch == 'vonenet50' or self.arch == 'vonenet101':
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
        elif self.arch == "inceptionresnet":
            in_features = [1024, 3072]
        elif self.arch=='lipsim':
            self.id1 = 15
            self.id2 = 20
            in_features = [45, 45]
        elif self.arch=='lipsim2':
            self.id1 = 20 # 12
            self.id2 = 22
            in_features = [45, 45]
        else:
            raise NotImplementedError(f'The arch {self.arch} is not implemented!')
        
        if self.arch == 'vonenet50':
            return in_features, Wrap(features)
        elif self.arch == 'vonenet101':
            return in_features, Wrap(features, layers_count=4)
        elif self.arch == 'lipsim' or self.arch=='lipsim2':
            ic(features.__len__())
            ic(nn.Sequential(*features).__len__())
            return in_features, nn.Sequential(*features)
        else:
            return in_features, nn.Sequential(*features)
        
    def get_activation_module(self, activation: str) -> nn.Module:
        if activation == 'silu':
            '''Change activations only in Linearity blocks to SiLU'''
            Activ = nn.SiLU
        elif activation == 'relu_silu':
            '''Change activations only in Linearity blocks to ReLU_SiLU'''
            Activ = ReLU_SiLU
        elif activation == 'relu_elu':
            '''Change activations only in Linearity blocks to ReLU_ELU'''
            Activ = ReLU_ELU
        elif activation == 'relu_gelu':
            '''Change activations only in Linearity blocks to ReLU_GELU'''
            Activ = ReLU_GELU
        elif activation == 'Fsilu':
            '''All activations changed to SiLU'''
            ReLU_to_SILU(self.features)
            Activ = nn.SiLU
        elif activation == 'Frelu_silu':
            '''All activations changed to ReLU_SiLU class'''
            ReLU_to_ReLUSiLU(self.features)
            Activ = ReLU_SiLU
        elif activation == 'Frelu_elu':
            '''All activations changed to ReLU_ELU class'''
            swap_all_activations(self.features, nn.ReLU, ReLU_ELU)
            Activ = ReLU_ELU
        elif activation == 'Frelu_gelu':
            '''All activations changed to ReLU_GELU class'''
            swap_all_activations(self.features, nn.ReLU, ReLU_GELU)
            Activ = ReLU_GELU
        elif activation == "gelu":
            '''All activations changed to GELU'''
            Activ = nn.GELU
        elif activation == "elu":
            '''All activations changed to ELU'''
            Activ = nn.ELU
        elif activation == "Fgelu":
            '''All activations changed to GELU'''
            swap_all_activations(self.features, nn.ReLU, nn.GELU)
            Activ = nn.GELU
        elif activation == "Felu":
            '''All activations changed to ELU'''
            swap_all_activations(self.features, nn.ReLU, nn.ELU)
            Activ = nn.ELU
        else:
            '''Default activation function'''
            Activ = nn.ReLU
        return Activ