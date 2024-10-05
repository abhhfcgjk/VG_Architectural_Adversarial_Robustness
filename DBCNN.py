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

from Cayley import CayleyBlockPool
from VOneNet import get_model
import activ

from tqdm import tqdm

from icecream import ic

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
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

        if options['backbone']=='vonenet50':
            model = get_model(model_arch='resnet50', pretrained=True, weightsdir=None,
                                map_location='cuda')
            print(model)
            self.features1 = nn.Sequential(
                        model.module.vone_block,
                        model.module.bottleneck,
                        *list(model.module.model.children())[:-2],
                        nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1))

        else:
            # Convolution and pooling layers of VGG-16.
            self.features1 = torchvision.models.vgg16(pretrained=True).features
            self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
        
        if options['cayley'] or options['cayley2']:
            self.cayley = CayleyBlockPool(in_channels=512, intermed_channels=200)
            self.features1 = nn.Sequential(*list(self.features1.children())[:-2], 
                                            self.cayley, 
                                            *list(self.features1.children())[-2:])
        elif options['cayley3']:
            self.cayley = CayleyBlockPool(in_channels=512, intermed_channels=200)
            self.features1 = nn.Sequential(*list(self.features1.children())[:-6], 
                                            self.cayley, 
                                            *list(self.features1.children())[-6:])
            
        if options['activation']=='relu_elu':
            # activ.swap_all_activations(self.features1, nn.ReLU, activ.ReLU_ELU)
            self.setup_activation(activ.ReLU_ELU)
        else:
            pass
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()

        
              
        scnn.load_state_dict(torch.load(scnn_root))
        self.features2 = scnn.module.features
        if options['cayley2']:
            self.cayley2 = CayleyBlockPool(in_channels=128, intermed_channels=100)
            # print(self.features2)
            self.features2 = nn.Sequential(*list(self.features2.children())[:-3], 
                                            self.cayley2, 
                                            *list(self.features2.children())[-3:])
            # print(self.features2)
        
        # Linear classifier.
        self.fc = torch.nn.Linear(512*128, 1)
        
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

    def setup_activation(self, activ_function):
        activ.swap_all_activations(self.features1, nn.ReLU, activ_function)

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]


        X1 = self.features1(X)
        # ic(self.features1)
        
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
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
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
        self._options = options
        self._path = path
        self.cayley_flag = 'cayley' if options['cayley'] else ''
        self.cayley_flag2 = 'cayley2' if options['cayley2'] else ''
        self.cayley_flag3 = 'cayley3' if options['cayley3'] else ''
        self.backbone_flag = 'vonenet50' if options['backbone']=='vonenet50' else ''
        self.activation_flag = self._options['activation']
        
        # Network.
        self._net = torch.nn.DataParallel(DBCNN(self._path['scnn_root'], self._options), device_ids=[0]).cuda()
        if self._options['fc'] == False:
            self._net.load_state_dict(torch.load(path['fc_root']))

        
        print(self._net)

        if self._options['pruning']>0:
            sparsity_features = self.l1_prune(amount=self._options['pruning'])
            self.print_sparcity(sparsity_features)
        # Criterion.
        self._criterion = torch.nn.MSELoss().cuda()

        # Solver.
        if self._options['fc'] == True:
            self._solver = torch.optim.SGD(
                    self._net.module.fc.parameters(), lr=self._options['base_lr'],
                    momentum=0.9, weight_decay=self._options['weight_decay'])
        else:
            self._solver = torch.optim.Adam(
                    self._net.module.parameters(), lr=self._options['base_lr'],
                    weight_decay=self._options['weight_decay'])

        
        if (self._options['dataset'] == 'live') | (self._options['dataset'] == 'livec'):
            if self._options['dataset'] == 'live':
                crop_size = 432
            else:
                crop_size = 448
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])
        elif (self._options['dataset'] == 'csiq') | (self._options['dataset'] == 'tid2013'):
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'mlive':
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((570,960)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'koniq-10k':
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((498,664)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])
            
            
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
            
           
        if self._options['dataset'] == 'live':  
            import LIVEFolder
            train_data = LIVEFolder.LIVEFolder(
                    root=self._path['live'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = LIVEFolder.LIVEFolder(
                    root=self._path['live'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'livec':
            import LIVEChallengeFolder
            train_data = LIVEChallengeFolder.LIVEChallengeFolder(
                    root=self._path['livec'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = LIVEChallengeFolder.LIVEChallengeFolder(
                    root=self._path['livec'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'koniq-10k':
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

    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_epoch = 0
        print('Epoch\tTrain loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self._options['epochs']):
            epoch_loss = []
            pscores = []
            tscores = []
            num_total = 0
            for X, y in tqdm(self._train_loader, total=len(self._train_loader)):
                # Data.
                X = X.clone().detach().requires_grad_(True).cuda()
                y = y.clone().detach().requires_grad_(True).cuda()

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y.view(len(score),1).detach())
                epoch_loss.append(loss.item())
                # Prediction.
                num_total += y.size(0)
                pscores = pscores +  score.cpu().tolist()
                tscores = tscores + y.cpu().tolist()
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_srcc, _ = stats.spearmanr(pscores,tscores)
            test_srcc, test_plcc = self._consitency(self._test_loader)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_epoch = t + 1
                print('*', end='')
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd,'fc_models',(
                        self.cayley_flag \
                        + self.cayley_flag2 \
                        + self.cayley_flag3 \
                        + self.backbone_flag\
                        + self.activation_flag + 'net_params' + '_best' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd,'db_models',(
                        self.cayley_flag \
                        + self.cayley_flag2 \
                        + self.cayley_flag3 \
                        + self.backbone_flag\
                        + self.activation_flag + 'net_params' + '_best' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t%4.4f' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))           

        print('Best at epoch %d, test srcc %f' % (best_epoch, best_srcc))
        return best_srcc

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0
        pscores = []
        tscores = []
        for X, y in data_loader:
            # Data.
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
        self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc
    
    def print_sparcity(prune_list: Tuple):
        """only for resnet"""
        print("SPARCITY")
        p_list = prune_list
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))

    def get_prune_features(self) -> List:
        prune_params_list = []

        for name, module in self._net.named_children():
            if isinstance(module, (nn.Conv2d)):
                prune_params_list.append((module, 'weight'))
            else:
                prune_params_list += self.get_prune_features(module)
        return prune_params_list

    def l1_prune(self, amount=0.1) -> Tuple:
        # if amount <= 0:
        #     return None

        prune_params = tuple(self.get_prune_features())
        prune.global_unstructured(
            parameters=prune_params,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        for module, name in prune_params:
            prune.remove(module, name)
        return prune_params




def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='koniq-10k',
                        help='dataset: live|csiq|tid2013|livec|mlive|koniq-10k')
    
    parser.add_argument('--backbone', default='vgg16', type=str, help='Basemodel: vgg16|vonenet50')
    parser.add_argument('--pruning', dest='pruning', type=float,
                        default=0, help='Pruning percentage.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function in VGG16.')
    parser.add_argument('--tune_iters', dest='tune_iters', type=int,
                        default=1, help='Iters for tune')
    parser.add_argument('--cayley', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley2', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley3', action='store_true',
                        help='Use cayley block before conv block')
    
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset':args.dataset,
        'fc': [],
        'backbone': args.backbone,
        'cayley': args.cayley,
        'cayley2': args.cayley2,
        'cayley3': args.cayley3,
        'pruning': args.pruning,
        'activation': args.activation,
        'train_index': [],
        'test_index': []
    }
    cayley_status = 'cayley' if args.cayley else ''
    cayley_status2 = 'cayley2' if args.cayley2 else ''
    cayley_status3 = 'cayley3' if args.cayley3 else ''
    activation_status = args.activation
    backbone_status = 'vonenet50' if args.backbone=='vonenet50' else ''
    path = {
        'koniq-10k': os.path.join('dataset', 'KonIQ-10k'),
        'ckpt': f'DBCNN-cayley={args.cayley}'\
                f'-cayley2={args.cayley2}'\
                f'-cayley3={args.cayley3}'\
                f'-activation={args.activation}.pt',

        'live': os.path.join('dataset','databaserelease2'),
        'csiq': os.path.join('dataset','CSIQ'),
        'tid2013': os.path.join('dataset','TID2013'),
        'livec': os.path.join('dataset','ChallengeDB_release'),
        'mlive': os.path.join('dataset','LIVEmultidistortiondatabase'),
        'fc_model': os.path.join('fc_models'),
        'scnn_root': os.path.join('pretrained_scnn','scnn.pkl'),
        'fc_root': os.path.join('fc_models', 
                                f'{cayley_status}{cayley_status2}'\
                                f'{cayley_status3}{backbone_status}{activation_status}'\
                                'net_params_best.pkl'),
        'db_model': os.path.join('db_models'),
        
    }
    
    
    if options['dataset'] == 'live':          
        index = list(range(0,29))
    elif options['dataset'] == 'csiq':
        index = list(range(0,30))
    elif options['dataset'] == 'tid2013':   
        index = list(range(0,25))
    elif options['dataset'] == 'mlive':
        index = list(range(0,15))
    elif options['dataset'] == 'livec':
        index = list(range(0,1162))
    elif options['dataset'] == 'koniq-10k':
        index = list(range(0,10073))
    
    

    tune_iters = args.tune_iters # 2 # 10
    lr_backup = options['base_lr']
    srcc_all = np.zeros((1,tune_iters),dtype=np.float32)
    
    

    for i in range(0,tune_iters):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
    
        options['train_index'] = train_index
        options['test_index'] = test_index
        #train the fully connected layer only
        options['fc'] = True
        options['base_lr'] = 1e-3
        manager = DBCNNManager(options, path)
        best_srcc = manager.train()
    
        #fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        manager = DBCNNManager(options, path)
        best_srcc = manager.train()
        
        checkpoints = {
            'model': manager._net.state_dict(),
            'SRCC': best_srcc,
            'max': manager.test_max,
            'min': manager.test_min
        }
        torch.save(checkpoints, path['ckpt'])

        srcc_all[0][i] = best_srcc
        
    srcc_mean = np.mean(srcc_all)
    print(srcc_all)
    print('average srcc:%4.4f' % (srcc_mean))  
    return best_srcc


if __name__ == '__main__':
    main()
