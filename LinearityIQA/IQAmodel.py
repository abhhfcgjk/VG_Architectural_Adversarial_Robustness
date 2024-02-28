import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock
import numpy as np
from .activ import ReLU_SiLU
from .SE import SqueezeExcitation
# from torchvision.ops import SqueezeExcitation
# from . import wide_resnet34

def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P+1):
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
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1, activation='relu', se=False):
        super(IQAModel, self).__init__()
        # self.wd_ratio = 0
        self.is_se = se
        self.layers = []
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        if arch=='wideresnet50':
            features = list(torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).children())[:-2]

        else:
            features = list(models.__dict__[arch](pretrained=True).children())[:-2]
        # print(type(features))
        # print(features)
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            # elif arch == 'wideresnet34':
            #     in_features = [int(round(256*self.width_factor)), int(round(512*self.width_factor))]
            else:
                in_features = [1024, 2048]
        else: 
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)

        if activation=='relu':
            Activ = nn.ReLU
        elif activation=='silu':
            Activ = nn.SiLU
        elif activation=='relu_silu':
            Activ = ReLU_SiLU
        
        if self.is_se:
            self.se6 = SqueezeExcitation(input_channels=in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]),
                                        squeeze_channels=in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]),
                                        activation=nn.SiLU)
            self.se7 = SqueezeExcitation(input_channels=in_features[1] * c * sum([p * p for p in range(1, self.P6+1)]),
                                        squeeze_channels=in_features[1] * c * sum([p * p for p in range(1, self.P6+1)]),
                                        activation=nn.SiLU)

        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), Activ())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]), 1024),
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

    def extract_features(self, x):
        f, pq = [], []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.id1:
                x6 = SPSP(x, P=self.P6, method=self.pool)
                if self.is_se:
                    x6 = self.se6(x6)
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = SPSP(x, P=self.P7, method=self.pool)
                if self.is_se:
                    x7 = self.se7(x7)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq

