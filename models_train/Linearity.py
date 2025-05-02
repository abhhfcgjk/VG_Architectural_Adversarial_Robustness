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
from models_train.swap_convs import swap_to_quntized
from models_train.pruning import PLSPrune, l1_prune, pls_prune, ln_prune, displs_prune, hsic_prune

from models_train.baseIQAmodel import IQA
from models_train import swap_convs
from orthogonium import BcopRkoConv2d

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
    def print_sparcity(self):
        """only for resnet"""
        print("SPARCITY")
        if not hasattr(self, 'prune_parameters'):
            def __help(model):
                ans = []
                for name, layer in model.named_children():
                    if isinstance(layer, nn.Conv2d):
                        ans.append((layer, 'weight'))
                    else:
                        ans += __help(layer)
                return ans
            convs = __help(self)
            # print(self.convs)
            print(convs)
            p_list = convs
        else:
            p_list = self.prune_parameters
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

        # self.pruning = kwargs.get('prune', 0.0)
        # self.width_prune = kwargs.get('width_prune')
        # self.height_prune = kwargs.get('height_prune')
        # self.pls_images = kwargs.get('pls_images')
        # self.kernel_prune = kwargs.get('kernel_prune')

        self.aoc = kwargs.get('aoc', False)
        self.gabor = kwargs.get('gabor', False)
        self.cayley = kwargs.get('cayley', False)
        self.cayley_pool = kwargs.get('cayley_pool', False)
        self.cayley_pair = kwargs.get('cayley_pair', False)
        self.cayley1 = kwargs.get('cayley1', False)
        self.cayley2 = kwargs.get('cayley2', False)
        self.cayley3 = kwargs.get('cayley3', False)
        self.cayley4 = kwargs.get('cayley4', False)
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

        if self.cayley1:
            self.cayley_block1 = CayleyBlockPool(3, 3, stride=1, padding=0, kernel_size=3)
        if self.cayley2:
            self.cayley_block2 = CayleyBlockPool(64, 64, stride=1, padding=0, kernel_size=3)
        if self.cayley3:
            self.cayley_block3 = CayleyBlockPool(256, 256, stride=1, padding=0, kernel_size=3)
        if self.cayley4:
            self.cayley_block4 = CayleyBlockPool(512, 128, stride=1, padding=0, kernel_size=3)
        
        self.id_cl1 = 0
        self.id_cl2 = 4
        self.id_cl3 = 5
        self.id_cl4 = 5

        if self.cayley:
            self.cayley_block6 = CayleyBlockPool(512, 200, stride=1, padding=0, kernel_size=3)
        if self.cayley_pool:
            self.cayley_block6 = CayleyBlockPool(1024, 200, stride=1, padding=0, kernel_size=3)
        if self.aoc:
            self.aoc_block = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
                BcopRkoConv2d(512, 1024, kernel_size=3, stride=1, padding=0)
                )
        
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

    def quantize(self, precision):
        swap_to_quntized(self.model, precision=precision, full_copy=True)
        self.model = self.model.to(self.device)

    def prune(self, amount, prtype='l1', **kwargs):
        self.prune_parameters: tuple
        self.pruning_type = prtype

        if prtype == 'l1':
            self.prune_parameters = l1_prune(self, amount)
        elif prtype == 'l2':
            self.prune_parameters = ln_prune(self, amount, 2)
        elif prtype == 'pls':
            self.prune_parameters = pls_prune(self, amount, **kwargs)
                                        # width=self.width_prune, 
                                        # height=self.height_prune, 
                                        # images_count=self.pls_images, 
                                        # kernel=self.kernel_prune) # 120, 90
        elif prtype == 'displs':
            self.prune_parameters = displs_prune(self, amount, **kwargs)
        elif prtype == 'hsic':
            self.prune_parameters = hsic_prune(self, amount, **kwargs)

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
                print('cayley:', x.shape)
                x = self.cayley_block6(x)
            if ii==self.id_cl1 and self.cayley1:
                # print(self.cayley_block1)
                print('cayley1:', x.shape)
                x = self.cayley_block1(x)
            if ii==self.id_cl2 and self.cayley2:
                # print(self.cayley_block2)
                print('cayley2:', x.shape)
                x = self.cayley_block2(x)
            if ii==self.id_cl3 and self.cayley3:
                # print(self.cayley_block3)
                print('cayley3:', x.shape)
                x = self.cayley_block3(x)
            if ii==self.id_cl4 and self.cayley4:
                # print(model)
                for bn, (_, layer) in enumerate(model.named_children()):
                    # print(layer)
                    if bn==2:
                        print('cayley4:', x.shape)
                        x = self.cayley_block4(x)
                    x = layer(x)
            else:
                x = model(x)
            # x = model(x)
            
            ic(x.shape)
            if ii == self.id1:
                # if self.cayley:
                #     x = self.cayley_block6(x)
                if self.cayley_pool:
                    # print('cayley_pool:', x.shape)
                    x = self.cayley_block6(x)
                if self.aoc:
                    x = self.aoc_block(x)
                x6 = x
                if self.cayley_pair:
                    print('cayley_pair:', x.shape)
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