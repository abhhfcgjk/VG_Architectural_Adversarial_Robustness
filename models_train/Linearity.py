import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from models_train.Dlayer import D1Layer

from typing import List, Tuple, Any, Union

from models_train.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU
from models_train.SE import SqueezeExcitation
from models_train.VOneNet import get_model

from models_train.baseIQAmodel import IQA

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
            # print("SP:",x.shape)
            a = F.avg_pool2d(x, pool_size)
            # print("SP:",a.shape)
            pool_features.append(a.view(batch_size, -1))  # average pooling
            # print("SP:",pool_features[-1].shape)
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

    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1, activation='relu', dlayer=None,
                 pruning=0.0):
        super(Linearity, self).__init__(arch)
        
        self.pruning = pruning
        self.pool = pool
        self.use_bn_end = use_bn_end
        self.dlayer = dlayer
        # self.arch = arch
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #

        in_features, self.features = self.get_features(self._base_model_features)

        Activ = self.get_activation_module(activation)

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


        self.d_in = nn.Linear(64*2, 200)
        self.d_h = nn.Linear(200, 200)
        self.d_out = nn.Linear(200, 64*2)
        self.d_layer = D1Layer(8, 16)
        # self.d_layer.train = True

    def extract_features(self, x):
        f, pq = [], []

        for ii, model in enumerate(self.features):
            x = model(x)

            if ii == self.id1:
                x6 = x
                x6 = SPSP(x6, P=self.P6, method=self.pool)
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = x
                x7 = SPSP(x7, P=self.P7, method=self.pool)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)
        ic(f.shape)
        if self.dlayer=='d1':
            e, eq_loss = self.d_layer(f)
            h1 = self.d_in(e)
            h2 = self.d_h(h1)
            f = self.d_out(h2)
        else:
            eq_loss = 0
        # f = f.float()
        # ic(f)
        # ic(eq_loss)
        return f, pq, eq_loss


    def forward(self, x):
        f, pq, eq_loss = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        if self.training:
            return pq, eq_loss
        else:
            return pq