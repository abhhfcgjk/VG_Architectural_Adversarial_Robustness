import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock
import numpy as np


if __name__=='IQAmodel':
    from activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU
    from SE import SqueezeExcitation
    from VOneNet import get_model
    # from ComModel import ...
    from ComModel import (LinearAttentionBlock,
                                       self_correlation,
                                       ProjectorBlock,
                                       LinearWithChannel,
                                       resnet18,
                                       resnet34,
                                       resnet50)
else:
    from LinearityIQA.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU
    from LinearityIQA.SE import SqueezeExcitation

    from LinearityIQA.VOneNet import get_model

    from LinearityIQA.ComModel import (LinearAttentionBlock,
                                       self_correlation,
                                       ProjectorBlock,
                                       LinearWithChannel,
                                       resnet18,
                                       resnet34,
                                       resnet50)

class Identity(nn.Module):
    def forward(self, x):
        return x

class wrap(nn.Module):
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
            return self.layers[item-2]

    def __len__(self):
        return self.len
    
    def __str__(self):
        return self.model.__str__()

    def __iter__(self):
        return iter(self.it)
    def __next__(self):
        if self.index < 2:
            self.index += 1
            return self.model[self.index-1]
        else:
            self.index += 1
            return self.layers[self.index-3]
    def forward(self, input):
        # for module in self.model:
        input = self.module(input)
        return input


# model = models.resnet18(pretrained=False)
# model.fc = Identity()
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
# print(output.shape)

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
            print("SP:",x.shape)
            a = F.avg_pool2d(x, pool_size)
            print("SP:",a.shape)
            pool_features.append(a.view(batch_size, -1))  # average pooling
            print("SP:",pool_features[-1].shape)
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    __arches = {"resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50}
    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1, activation='relu', se=False):
        super(IQAModel, self).__init__()
        # self.wd_ratio = 0
        self.is_se = se
        self.layers = []
        self.pool = pool
        self.use_bn_end = use_bn_end
        self.arch = arch
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #

        
        
        if arch=='wideresnet50':
            features = list(torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).children())[:-2]
        elif arch=='vonenet50':
            features = list(get_model(model_arch='resnet50', pretrained=True, map_location='cuda').children())
            features[0][-1].avgpool = Identity()
            features[0][-1].fc = Identity()
        elif 'rartfa' in arch:
            # self.features = RARTFA(arch=arch.replace("rartfa", ''), pretrained=True)
            self.features = list(self.__arches[arch.replace('rartfa', '')](pretrained=True).children())[:-2]
            # print("HERERERE")
            # quit()
            self.md = self.__arches[arch.replace('rartfa', '')](pretrained=True)
            self.md.fc = Identity()
            self.md.avgpool = Identity()
            # quit()
            # self.features = list(self.md.children())
            # print(self.md.children())
            # quit()
            # return
        else:
            features = list(models.__dict__[arch](pretrained=True).children())[:-2]

        # print(features)
        # quit()

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
        elif arch=='vonenet50':
            self.id1 = 4
            self.id2 = 5
            in_features = [1024, 2048]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if 'resnet18' in arch or 'resnet34' in arch:
                in_features = [256, 512]
            # elif arch == 'wideresnet34':
            #     in_features = [int(round(256*self.width_factor)), int(round(512*self.width_factor))]
            else:
                in_features = [1024, 2048]
        
        else:
            print('The arch is not implemented!')
            quit()

        if arch=='vonenet50':
            self.features = wrap(features)
        elif 'rartfa' in arch:
            pass
        else:
            self.features = nn.Sequential(*features)
        # print(len(self.features))

        
        if activation=='silu':
            Activ = nn.SiLU
        elif activation=='relu_silu':
            Activ = ReLU_SiLU
        else:
            Activ = nn.ReLU

        if activation=='Fsilu':
            ReLU_to_SILU(self.features)
            Activ = nn.SiLU
        elif activation=='Frelu_silu':
            ReLU_to_ReLUSiLU(self.features)
            Activ = ReLU_SiLU
        # print(self.features)

        if 'rartfa' in self.arch:
            self.attn1 = LinearAttentionBlock(512, normalize_attn=False)
            self.attn2 = LinearAttentionBlock(512, normalize_attn=False)
            self.attn3 = LinearAttentionBlock(512, normalize_attn=False)

            self.proj1 = ProjectorBlock(64, 512)
            self.proj2 = ProjectorBlock(128, 512)
            self.proj3 = ProjectorBlock(256, 512)

            # self.proj11 = ProjectorBlock(64, 64)

            self.corr1 = self_correlation(10, 64)
            self.corr2 = self_correlation(10, 128)
            self.corr3 = self_correlation(10, 256)

            self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0, bias=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


            self.classify = nn.Linear(300, 1) #20-1200,10
        # else:
        if self.is_se:
            self.se6 = SqueezeExcitation(input_channels=in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]),
                                        squeeze_channels=4,
                                        activation=Activ)
            self.se7 = SqueezeExcitation(input_channels=in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]),
                                        squeeze_channels=4,
                                        activation=Activ)

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
                if self.is_se:
                    x6 = self.se6(x)
                else:
                    x6 = x
                # print(x6.shape)
                x6 = SPSP(x6, P=self.P6, method=self.pool)
                # print(x6.shape)
                x6 = self.dr6(x6)
                # print(x6.shape)
                # quit()
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                if self.is_se:
                    x7 = self.se7(x)
                else:
                    x7 = x
                x7 = SPSP(x7, P=self.P7, method=self.pool)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def exec_rartfa(self, x, gg=None):
        # print(x.size())
        x = self.md(x)
        l1, l2, l3, l4, _ = self.md.layer_results.values()
        # print(l1.size(), l2.size(), l3.size(), l4.size(), _.size())
        # print(l4)
        # quit()
        if gg == None:
            gg = self.dense(l4)
        gg = self.avgpool(gg)
        # print(l1.size(), l2.size(), l3.size(), l4.size(), gg.size())
        c1, g1 = self.attn1(self.proj1(l1), gg)
        print("ATTN:", c1.shape)
        out1 = self.corr1(l1, c1)

        c2, g2 = self.attn2(self.proj2(l2), gg)
        print("ATTN:", c2.shape)
        out2 = self.corr2(l2, c2)

        c3, g3 = self.attn3(self.proj3(l3), gg)
        out3 = self.corr3(l3, c3)

        g = torch.cat((out1, out2, out3), dim=1)
        print("OUT SHAPE:",out1.shape, out2.shape, out3.shape, g.shape)
        g = g.view(g.size(0), -1)
        print("G|||||",g.shape)
        out = self.classify(g)

        return out, c1, c2

    def forward(self, x):
        if 'rartfa' in self.arch:
            # pq = self.features(x)
            s, _, _ = self.exec_rartfa(x)
            # print(pq.size())
            pq.append(s)
        else:
            f, pq = self.extract_features(x)
            s = self.regression(f)
            pq.append(s)

        return pq

