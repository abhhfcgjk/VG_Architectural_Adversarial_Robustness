from typing import List, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIPool, RoIAlign
import numpy as np
from torch.nn import ReLU, SiLU, ELU, GELU
import torch.nn.functional as F

from icecream import ic
ic.disable()

from Cayley import CayleyBlockPool
# from activ import swap_all_activations, ReLU_ELU, ReLU_SiLU
import activ
from pruning.pruning import l1_prune, ln_prune, pls_prune, displs_prune, hsic_prune


def weight_init(net): 
    for m in net.modules():    
        if isinstance(m, nn.Conv2d):         
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.
        self.num_class = 39
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)    
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)          
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
        return X


class DBCNN(torch.nn.Module):
    def print_sparcity(self):
        print("SPARCITY")
        p_list = self.prune_parameters
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))


    def __init__(self, scnn_root, config, **options):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.prune_amount = options.get('prune', 0.0)
        self.db_model_dir = config.get('db_model')

        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
        
        if options.get('cayley', None) or options.get('cayley2', None):
            self.cayley = CayleyBlockPool(in_channels=512, intermed_channels=200)
            self.features1 = nn.Sequential(*list(self.features1.children())[:-2], 
                                            self.cayley, 
                                            *list(self.features1.children())[-2:])
        elif options.get('cayley3', None) or options.get('cayley4', None):
            self.cayley = CayleyBlockPool(in_channels=512, intermed_channels=200)
            self.features1 = nn.Sequential(*list(self.features1.children())[:-6], 
                                            self.cayley, 
                                            *list(self.features1.children())[-6:])

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
              
        scnn.load_state_dict(torch.load(scnn_root))
        self.features2 = scnn.module.features
        if options.get('cayley2',None):
            self.cayley2 = CayleyBlockPool(in_channels=128, intermed_channels=100)
            self.features2 = nn.Sequential(*list(self.features2.children())[:-3], 
                                            self.cayley2, 
                                            *list(self.features2.children())[-3:])
        elif options.get('cayley4',None):
            self.cayley4 = CayleyBlockPool(in_channels=64, intermed_channels=64)
            self.features2 = nn.Sequential(*list(self.features2.children())[:-9], 
                                            self.cayley4, 
                                            *list(self.features2.children())[-9:])
        
        # Linear classifier.
        self.fc = torch.nn.Linear(512*128, 1)
        
        self.__set_activation(options.get('activation', None))

        if self.prune_amount > 0:
            self.__load_pretrained()
            self.prune(amount=options['prune'], 
                      prtype=options['prune_type'],
                      width=256,
                      height=192,
                      images_count=100,
                      kernel=1)
            self.print_sparcity()

        if options['fc'] == True:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
        
    def __load_pretrained(self):
        new_state_dict = {}
        # self.db_model_dir = Path(self.db_model_dir)
        checkpoint = torch.load(self.db_model_dir)['model']
        for key, value in checkpoint.items():
            new_key = key.replace('model.', '')  # Adjust as necessary
            new_state_dict[new_key] = value
        self.load_state_dict(new_state_dict, strict=False)

    def __set_activation(self, activ_function):
        if activ_function=='Frelu_elu':
            activ.swap_all_activations(self.features1, nn.ReLU, activ.ReLU_ELU)
            # activ.swap_all_activations(self.features2, nn.ReLU, activ.ReLU_ELU)
            self.Activ = activ.ReLU_ELU
        elif activ_function=='Frelu_silu':
            activ.swap_all_activations(self.features1, nn.ReLU, activ.ReLU_SiLU)
            # activ.swap_all_activations(self.features2, nn.ReLU, activ.ReLU_SiLU)
            self.Activ = activ.ReLU_SiLU
        elif activ_function=='Felu':
            activ.swap_all_activations(self.features1, nn.ReLU, nn.ELU)
            # activ.swap_all_activations(self.features2, nn.ReLU, nn.ELU)
            self.Activ = nn.ELU
        elif activ_function=='Fsilu':
            activ.swap_all_activations(self.features1, nn.ReLU, nn.SiLU)
            # activ.swap_all_activations(self.features2, nn.ReLU, nn.SiLU)
            self.Activ = nn.SiLU
        elif activ_function=='Fgelu':
            activ.swap_all_activations(self.features1, nn.ReLU, nn.GELU)
            # activ.swap_all_activations(self.features2, nn.ReLU, nn.GELU)
            self.Activ = nn.GELU
        else:
            # activ.swap_all_activations(self.features1, nn.ReLU, nn.ReLU)
            self.Activ = nn.ReLU

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

    def _euclidian_mapping(self, B):
        #ic(torch.sqrt(torch.abs(B)))
        B_mul = torch.sign(B)*torch.sqrt(torch.abs(B))
        #ic(B_mul)
        
        mapped_B = B_mul / (torch.norm(B_mul) + 1e-8)
        return mapped_B

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128        
        
        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2,(H,W))

        X1 = X1.view(N, 512, H*W)
        X2 = X2.view(N, 128, H*W)  
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H*W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512*128)
        X = torch.sqrt(torch.abs(X) + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 1)
        return X


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

