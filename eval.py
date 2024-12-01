import os
from typing import List, Tuple

import torch
import torchvision
import torch.nn as nn
from SCNN import SCNN
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
from torch.nn.utils import prune
import numpy as np
from collections import OrderedDict
import yaml

import activ
from VOneNet import get_model
from Cayley import CayleyBlockPool

from tqdm import tqdm

# from icecream import ic

YAML_PATH = "./path_config.yaml"

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    # import accimage
    try:
        # return accimage.Image(path)
        pass
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
    

class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        if options.get('backbone', None)=='vonenet':

            with open(YAML_PATH, 'r') as file:
                yaml_conf = yaml.safe_load(file)
            weights_dir = yaml_conf['checkpoints']['vonenet-vgg16']
            model = get_model(
                model_arch='vgg16', 
                pretrained=True, 
                weightsdir=weights_dir,
                map_location='cuda')
            
            self.features1 = nn.Sequential(OrderedDict([
                ('vone_block', model.module.vone_block),
                ('bottleneck', model.module.bottleneck),
                ('model', model.module.model.features[:-1])
            ]))

            print(self.features1)
            
            # self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
            # print(self.features1)

        else:
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
            activ.swap_all_activations(self.features1, nn.ReLU, nn.ReLU)
            self.Activ = nn.ReLU

    def _euclidian_mapping(self, B):
        B_mul = torch.sign(B)*torch.sqrt(torch.abs(B))
        mapped_B = B_mul / (torch.norm(B_mul) + 1e-8)
        return mapped_B

    def forward(self, X):
        """Forward pass of the network."""
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
        X = torch.bmm(X1, torch.transpose(X2, 1, 2))
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512*128)
        X = self._euclidian_mapping(X + 1e-8)
        X = self.fc(X)
        assert X.size() == (N, 1)
        return X



class DBCNNManager(object):
    
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        # Network.
        self._options = options
        self._path = path
        self._net = torch.nn.DataParallel(DBCNN(self._path['scnn_root'], self._options), device_ids=[0]).cuda()

        print(self._net)
        
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
            
           
        if self._options['dataset'] == 'koniq-10k':
            import Koniq_10k
            train_data = Koniq_10k.Koniq_10k(
                    root=self._path['koniq-10k'], loader = default_loader, index = self._options['train_index'],
                    transform=test_transforms)
            test_data = Koniq_10k.Koniq_10k(
                    root=self._path['koniq-10k'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        else:
            raise AttributeError('Only support LIVE and LIVEC right now!')
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)

        modelpath = self._path['ckpt']# os.path.join(pwd,'db_models',('net_params' + '_best' + '.pkl'))
        ckpt = torch.load(modelpath)
        self._net.load_state_dict(ckpt['model'])

    def _consitency(self):
        # self._net.train(False)
        self._net.eval()
        num_total = 0
        pscores = []
        tscores = []
        for X, y in self._test_loader:
            # Data.
            # X = torch.tensor(X.cuda())
            # y = torch.tensor(y.cuda())
            X = X.clone().detach().requires_grad_(True).cuda()
            y = y.clone().detach().requires_grad_(True).cuda()

            # Prediction.
            score = self._net(X)
            pscores = pscores +  score[0].cpu().tolist()
            tscores = tscores + y.cpu().tolist()

            self.test_max = max(pscores)
            self.test_min = min(pscores)
            
            num_total += y.size(0)
        test_srcc, _ = stats.spearmanr(pscores,tscores)
        test_plcc, _ = stats.pearsonr(pscores,tscores)
        self.srcc = test_srcc
        # self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Train DB-CNN for BIQA.')
    parser.add_argument('--path', type=str, help="Path to checkpoints")

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='koniq-10k',
                        help='dataset: live|csiq|tid2013|livec|mlive|koniq-10k')
    
    parser.add_argument('--backbone', default='vgg16', type=str, help='Basemodel: vgg16|vonenet')

    parser.add_argument('--prune', dest='prune', type=float,
                        default=0, help='Pruning percentage.')
    parser.add_argument('--prune_epochs', dest='prune_epochs', type=int,
                        default=5, help='Pruning epochs.')
    parser.add_argument('--prune_type', dest='prune_type', type=str,
                        default='l2', help='Pruning type.')
    parser.add_argument('--prune_lr', dest='prune_lr', type=float,
                        default=1e-6, help='Pruning learning rate.')
    
    parser.add_argument('--iter', dest='iter', type=int, default=0, help='Current train iteration')

    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function in VGG16. relu|relu_elu')
    parser.add_argument('--tune_iters', dest='tune_iters', type=int,
                        default=1, help='Iters for tune')
    parser.add_argument('--gradient_regularization', '-gr', action='store_true',
                        help='Use gradient regularization')
    parser.add_argument('--cayley', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley2', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley3', action='store_true',
                        help='Use cayley block before conv block')
    parser.add_argument('--cayley4', action='store_true',
                        help='Use cayley block before conv block in VGG16 and SCNN')
    parser.add_argument('--debug', action='store_true',
                        help='DEBUG')
    
    
    args = parser.parse_args()

    options = {
        'batch_size': 8,
        'dataset':args.dataset,
        'fc': True,
        'backbone': args.backbone,
        'cayley': args.cayley,
        'cayley2': args.cayley2,
        'cayley3': args.cayley3,
        'cayley4': args.cayley4,
        'prune': args.prune,
        'prune_lr': args.prune_lr,
        'prune_epochs': args.prune_epochs,
        'prune_type': args.prune_type,
        'activation': args.activation,
        'train_index': [],
        'test_index': [],
        'gradient_regularization': args.gradient_regularization,
    }
    
    path = {
        'koniq-10k': os.path.join('dataset', 'KonIQ-10k'),
        'ckpt': args.path,
        'scnn_root': os.path.join('pretrained_scnn','scnn.pkl'),
    }
    
    if options['dataset'] == 'koniq-10k':
        index = list(range(0,10073))
    else:
        raise KeyError 

    train_index = index[0:round(0.8*len(index))]
    test_index = index[round(0.8*len(index)):len(index)]
    options['train_index'] = train_index
    options['test_index'] = test_index
    options['fc'] = True
    manager = DBCNNManager(options, path)
    srcc, plcc = manager._consitency()
    print(srcc, plcc, manager.test_min, manager.test_max)
    # torch.save(checkpoints, path['ckpt'])

if __name__ == '__main__':
    main()
