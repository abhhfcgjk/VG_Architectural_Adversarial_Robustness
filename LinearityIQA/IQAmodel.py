import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torchvision import models
from torchvision.models.resnet import BasicBlock
import numpy as np


from typing import List

if __name__=='IQAmodel':
    from activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU, PruneConv
    from SE import SqueezeExcitation
    from VOneNet import get_model
else:
    from LinearityIQA.activ import ReLU_SiLU, ReLU_to_SILU, ReLU_to_ReLUSiLU, PruneConv
    from LinearityIQA.SE import SqueezeExcitation

    from LinearityIQA.VOneNet import get_model


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
            # print("SP:",x.shape)
            a = F.avg_pool2d(x, pool_size)
            # print("SP:",a.shape)
            pool_features.append(a.view(batch_size, -1))  # average pooling
            # print("SP:",pool_features[-1].shape)
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    @staticmethod
    def get_prune_features(model: nn.Module) -> List:

        prune_params_list = []

        for name, module in model.named_children():
            if hasattr(module, 'weight'):
                prune_params_list.append((module, 'weight'))
            else:
                prune_params_list += IQAModel.get_prune_features(module)
        return prune_params_list


    @staticmethod
    def print_sparcity(model: nn.Module, prune_list: List = None):
        """only for resnet"""
        p_list = []
        if prune_list:
            p_list = prune_list
        else:
            p_list = IQAModel.get_prune_list(model)
        
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0))/float(getattr(module, attr).nelement())
            if percentage > 0.0:
                print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                             getattr(module, attr).shape, percentage))

    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1, activation='relu', se=False, pruning=0.0):
        super(IQAModel, self).__init__()
        # self.wd_ratio = 0
        self.pruning = pruning
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
            features = list(get_model(model_arch='resnet50', pretrained=True, map_location='cuda' if torch.cuda.is_available() else 'cpu').children())
            features[0][-1].avgpool = Identity()
            features[0][-1].fc = Identity()
        
        else:
            resnet_model = models.__dict__[arch](pretrained=True)
            # print(list(resnet_model.named_children()))
            # print(list(resnet_model.named_children()).__len__())

            # for label, module in resnet_model.named_children():
            #     if 'layer' in label:
            #         print(module[0].conv1)
            # from PIL import Image
            # from pathlib import Path
            # from torchvision.transforms.functional import resize, to_tensor
            # im_path = list(Path('./KonIQ-10k/').iterdir())[0]
            # im = to_tensor(Image.open(im_path).convert('RGB'))
            # im = im.unsqueeze(0)
            # print(im)
            # print(resnet_model(im))
            

            # quit()
            print(self.pruning)
            # if self.pruning is not None and self.pruning > 0:
            #     prune_parameters = tuple(self.get_prune_features(resnet_model))

            #     prune.global_unstructured(
            #         prune_parameters,
            #         pruning_method=prune.L1Unstructured,
            #         amount=self.pruning,
            #     )

            #     for module, param in prune_parameters:
            #         prune.remove(module, param)

            #     IQAModel.print_sparcity(resnet_model, prune_parameters)

            if self.pruning is not None and self.pruning>0:
                COLAB = False
                h=90
                w=120
                t_count = 100
                if COLAB:
                    PruneConv.apply(resnet_model, '',self.pruning, train_count=5,
                                    dataset_labels_path='./VG_Architectural_Adversarial_Robustness/LinearityIQA/data/KonIQ-10kinfo.mat',
                                    dataset_path='./drive/MyDrive/KonIQ-10k', is_resize=True,
                                    resize_height=96, resize_width=128)
                else:
                    PruneConv.apply(resnet_model, 'weight',self.pruning, train_count=t_count, is_resize=True,
                                    resize_height=h, resize_width=w)
                # PruneConv.remove('weight')
                prune_parameters = []
                for i in range(len(PruneConv.convs)):
                    prune_parameters.append((PruneConv.convs[i], 'weight'))
        
                for i in range(len(PruneConv.convs)):
                    # print(name)
                    module = PruneConv.convs[i]
                    # print(module.weight)                    
                    prune.remove(module, 'weight')
                # PruneConv.remove('weight')
                IQAModel.print_sparcity(resnet_model, prune_parameters)
                torch.save(resnet_model.state_dict(), f"pruned/{arch}_{h}x{w}_amount={self.pruning}_imgs={t_count}")
            features = list(resnet_model.children())[:-2]
            


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
            else:
                in_features = [1024, 2048]
        
        else:
            print('The arch is not implemented!')
            quit()

        if arch=='vonenet50':
            self.features = wrap(features)
        
        else:
            self.features = nn.Sequential(*features)

        
        if activation=='silu':
            '''Change activations only in Linearity blocks to SiLU'''
            Activ = nn.SiLU
        elif activation=='relu_silu':
            '''Change activations only in Linearity blocks to ReLU_SiLU'''
            Activ = ReLU_SiLU
        elif activation=='Fsilu':
            '''All activations changed to SiLU'''
            ReLU_to_SILU(self.features)
            Activ = nn.SiLU
        elif activation=='Frelu_silu':
            '''All activations changed to ReLU_SiLU class'''
            ReLU_to_ReLUSiLU(self.features)
            Activ = ReLU_SiLU
        else:
            '''Default activation function'''
            Activ = nn.ReLU

        
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
                x6 = SPSP(x6, P=self.P6, method=self.pool)
                x6 = self.dr6(x6)
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

    

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq

