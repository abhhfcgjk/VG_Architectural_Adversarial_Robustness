from collections import OrderedDict
from unittest.mock import seal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.ops import RoIPool, RoIAlign
import numpy as np
from torch.nn import ReLU, SiLU, ELU, GELU
from pathlib import Path
import torchvision.models as models
import os

from activ import swap_all_activations, ReLU_ELU, ReLU_SiLU
from pruning.pruning import l1_prune, ln_prune, pls_prune, displs_prune, hsic_prune
from resnet_modify  import resnet50 as resnet_modifyresnet
from transformers import Transformer
from posencode import PositionEmbeddingSine
from Cayley import CayleyBlockPool

class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()

class TReS(nn.Module):
    def print_sparcity(self):
        """only for resnet"""
        print("SPARCITY")
        p_list = self.prune_parameters
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))
    def __init__(self, nheadt, num_encoder_layerst, dim_feedforwardt, /,
                 **kwargs):
        super(TReS, self).__init__()

        self.Activ = None
        self.db_model_dir = kwargs.get('db_model')
        self.is_cayley1 = kwargs.get('cayley1', False)
        self.is_cayley2 = kwargs.get('cayley2', False)
        self.prune_amount = kwargs.get('prune', 0.0)
        self.device = 'cuda:0'
        
        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)

        dim_modelt = 3840
        modelpretrain = models.resnet50(pretrained=True)
        torch.save(modelpretrain.state_dict(), 'modelpretrain')
        self.base = resnet_modifyresnet()
        self.base.load_state_dict(torch.load('modelpretrain'), strict=True)
        self.dim_modelt = dim_modelt
        os.remove("modelpretrain")
        # self.n_regtokens = 4
        # self.regtokens = torch.nn.Parameter(
        #         1e-2 * torch.randn(1, self.n_regtokens, self.dim_modelt)
        #     )

        if self.is_cayley1:
            # print(self)
            self.cayley1 = CayleyBlockPool(in_channels=2080, intermed_channels=1040, 
                                          stride=1, padding=0, kernel_size=3)
            # self.base = nn.Sequential(*list(self.base.children())[:-3],
            #                           self.cayley1,
            #                           *list(self.base.children())[-3:])
        elif self.is_cayley2:
            # print(self)
            self.cayley2 = CayleyBlockPool(in_channels=2080, intermed_channels=1040, 
                                          stride=1, padding=0, kernel_size=3)
            # self.base = nn.Sequential(*list(self.base.children())[:-2],
            #                           self.cayley2,
            #                           *list(self.base.children())[-2:])

        self.__set_activation(activation=kwargs.get('activation'))

        nheadt = nheadt
        num_encoder_layerst = num_encoder_layerst
        dim_feedforwardt = dim_feedforwardt
        ddropout = 0.5
        normalize = True
        self.model = Transformer(d_model=dim_modelt,nhead=nheadt,
                                 num_encoder_layers=num_encoder_layerst,
                                 dim_feedforward=dim_feedforwardt,
                                 normalize_before=normalize,
                                 dropout = ddropout)
        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)
        self.fc2 = nn.Linear(dim_modelt, self.base.fc.in_features) 
        self.fc = nn.Linear(self.base.fc.in_features*2, 1) 
        self.ReLU = self.Activ()
        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))
        self.drop2d = nn.Dropout(p=0.1)
        self.consistency = nn.L1Loss()

    def load_pretrained(self, path):
        new_state_dict = {}
        self.db_model_dir = Path(self.db_model_dir)
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
        self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
        self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
        out,layer1,layer2,layer3,layer4 = self.base(x) 

        layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
        layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
        layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
        layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
        layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)

        out_t_c = self.model(layers,self.pos_enc)
        out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg7(layer4)
        layer4_o = torch.flatten(layer4_o,start_dim=1)
        predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))

        fout,flayer1,flayer2,flayer3,flayer4 = self.base(torch.flip(x, [3])) 
        flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
        flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
        flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
        flayer4_t =            self.L2pooling_l4(F.normalize(flayer4,dim=1, p=2))
        flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)

        fout_t_c = self.model(flayers,self.pos_enc)
        fout_t_o = torch.flatten(self.avg7(fout_t_c),start_dim=1)
        fout_t_o = (self.fc2(fout_t_o))
        flayer4_o = self.avg7(flayer4)
        flayer4_o = torch.flatten(flayer4_o,start_dim=1)
        fpredictionQA =  (self.fc(torch.flatten(torch.cat((fout_t_o,flayer4_o),dim=1),start_dim=1)))

        consistloss1 = self.consistency(out_t_c,fout_t_c.detach())
        consistloss2 = self.consistency(layer4,flayer4.detach())
        consistloss = 1*(consistloss1+consistloss2)
        return predictionQA, consistloss

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
