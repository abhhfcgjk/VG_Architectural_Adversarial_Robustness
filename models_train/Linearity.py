import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn import quantized as nq
from torch.nn.utils.fusion import fuse_conv_bn_weights, fuse_conv_bn_eval

# from models_train.Dlayer import D1Layer, D2Layer

from typing import List, Tuple, Any, Union

from models_train.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU
from models_train.SE import SqueezeExcitation
from models_train.VOneNet import get_model
from models_train.Cayley import CayleyBlock, CayleyBlockPool

from models_train.baseIQAmodel import IQA
from models_train import swap_convs

from icecream import ic

def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P + 1):
        pool_size = [np.int32(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1 = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class Linearity(IQA):
    @staticmethod
    def print_sparcity(prune_list: Tuple):
        """only for resnet"""
        print("SPARCITY")
        p_list = prune_list
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))

    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, 
                 P6=1, P7=1, activation='relu', 
                 **kwargs):
        super(Linearity, self).__init__(arch)
        
        self.pool = pool
        self.use_bn_end = use_bn_end
        # self.pruning = pruning
        # self.dlayer = dlayer
        # self.gabor = gabor
        # self.cayley = cayley
        # self.cayley_pool = cayley_pool
        # self.cayley_pair = cayley_pair

        self.pruning = kwargs.get('pruning', 0.0)
        self.gabor = kwargs.get('gabor', False)
        self.cayley = kwargs.get('cayley', False)
        self.cayley_pool = kwargs.get('cayley_pool', False)
        self.cayley_pair = kwargs.get('cayley_pair', False)
        self.is_quantize = kwargs.get('quantize', False)
        # self.arch = arch
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        ic(self._base_model_features.__len__())
        in_features, self.features = self.get_features(self._base_model_features)
        if self.gabor:
            swap_convs.swap_to_gabor(self.features)
        if self.is_quantize:
            swap_convs.swap_to_quntized(self.features)

        Activ = self.get_activation_module(activation)
        if self.arch == 'lipsim' or self.arch=='lipsim2':
            self.lipsim_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=(16,32), stride=(2,2), padding=(0,0)),
                nn.MaxPool2d(kernel_size=(18,42), stride=(1,1), dilation=(1,2), padding=0)
            )

        # if self.is_quantize:
        #     self.quant = QuantStub()
        #     self.dequant = DeQuantStub()

        if self.cayley:
            self.cayley_block6 = CayleyBlockPool(512, 200, stride=1, padding=0, kernel_size=3)
        if self.cayley_pool:
            self.cayley_block6 = CayleyBlockPool(1024, 200, stride=1, padding=0, kernel_size=3)
        if self.cayley_pair:
            self.cayley_conv4 = CayleyBlockPool(1024, 200, stride=1, padding=0, kernel_size=3)
            self.cayley_conv5 = CayleyBlockPool(2048, 200, stride=1, padding=0, kernel_size=3)
            # self.cayley_block7 = CayleyBlock(2048, 800, stride=(1,1), padding=(4,2), kernel_size=(2,3))
        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6 + 1)]), 1024),
                                nn.BatchNorm1d(1024),
                                nn.Linear(1024, 256),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 64),
                                nn.BatchNorm1d(64), Activ())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7 + 1)]), 1024),
                                nn.BatchNorm1d(1024),
                                nn.Linear(1024, 256),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 64),
                                nn.BatchNorm1d(64), Activ())

        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    @staticmethod
    def fuse_all_conv_bn(model):
        """
        Fuses all consecutive Conv2d and BatchNorm2d layers.
        License: Copyright Zeeshan Khan Suri, CC BY-NC 4.0
        """
        stack = []
        for name, module in model.named_children(): # immediate children
            if list(module.named_children()): # is not empty (not a leaf)
                Linearity.fuse_all_conv_bn(module)
                
            if isinstance(module, nn.BatchNorm2d):
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))

    # def fuse_model(self):
    #     assert self.is_quantize, "This method is available only in quantization mode."
    #     print("FUSE")
    #     print(list(self.named_children()))
    #     def _help(model):
    #         fuse_list = []
    #         for name, module in model.named_children():
    #             if isinstance(module, (nn.Sequential, models.resnet.Bottleneck)):
    #                 _help(module)
    #             elif not isinstance(module, (QuantStub, DeQuantStub)):
    #                 # print(name)
    #                 if isinstance(module, nn.Conv2d):
    #                     fuse_list.append(name)
    #                 # fuse_list.append([n for n, m in module.named_children()])
    #         # print(fuse_list)
    #         # if len(fuse_list) > 0:
    #         print(model)
    #         torch.ao.quantization.fuse_modules_qat(model, fuse_list, inplace=True)
    #         for name in fuse_list:
    #             print(f"Quantize {name}.")
    #     _help(list(self.modules())[0])

    #     # self.fuse_all_conv_bn(self.features)
    #     print("FUSE END")

    def extract_features(self, x):
        f, pq = [], []

        # ic(self.features)
        ic(len(self.features))
        # ic(self.features[2])
        if self.arch == 'lipsim' or self.arch=='lipsim2':
            x = self.lipsim_pool(x)
        
        for ii, model in enumerate(self.features):
            ic(ii)
            ic(x.shape)
            
            ic(model)
            if ii==self.id1 and self.cayley:
                ic(x.shape)
                x = self.cayley_block6(x)
            x = model(x)
            
            ic(x.shape)
            if ii == self.id1:
                # if self.cayley:
                #     x = self.cayley_block6(x)
                if self.cayley_pool:
                    ic(x.shape)
                    x = self.cayley_block6(x)
                
                x6 = x
                if self.cayley_pair:
                    x6 = self.cayley_conv4(x6)
                ic(x6.shape)
                x6 = SPSP(x6, P=self.P6, method=self.pool)
                ic(x6.shape)
                x6 = self.dr6(x6)
                ic(x6.shape)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = x
                if self.cayley_pair:
                    x7 = self.cayley_conv5(x7)
                ic(x7.shape)
                x7 = SPSP(x7, P=self.P7, method=self.pool)
                ic(x7.shape)
                x7 = self.dr7(x7)
                ic(x7.shape)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)
        ic(f.shape)
        return f, pq

    def forward(self, x):
        # if self.is_quantize:
        #     x = self.quant(x)
        
        f, pq = self.extract_features(x)
        s = self.regression(f)
        
        # if self.is_quantize:
        #     s = self.dequant(s)
        pq.append(s)

        return pq