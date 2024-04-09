from typing import Any
import torch
# from torch.nn import Module, ReLU, SiLU, Conv2d
from torch import nn
import torch.nn.functional as F
from torch import Tensor, exp
from torch.autograd import Function
from torch.nn.utils.prune import BasePruningMethod, _validate_pruning_amount
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize
import numpy as np
from sklearn.cross_decomposition import PLSRegression

import os
from typing import Tuple, List

from PIL import Image
import h5py


class Activaion_forward_ReLU_backward_SiLU(Function):
    @staticmethod
    def forward(ctx, x, inplace):
        result = F.relu(x, inplace=inplace)
        dx = 1/(1+exp(-x)) + x*exp(-x)/(1+exp(-x))**2
        ctx.save_for_backward(dx)
        ctx.inplace = inplace
        return result
    @staticmethod
    def backward(ctx, grad_output):
        dx, = ctx.saved_tensors
        result = grad_output*dx
        inplace = ctx.inplace
        return result, None

class ReLU_SiLU(nn.Module):
    """
    Activation function is ReLU in forward.
    Activation function is SiLU in backward.
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU_SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        result = Activaion_forward_ReLU_backward_SiLU.apply(input, self.inplace)
        return result

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def ReLU_to_SILU(model):
    """Swap ReLU activation to SiLU."""
    for name,layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(model, name, nn.SiLU())
        else:
            ReLU_to_SILU(layer)

def ReLU_to_ReLUSiLU(model):
    """Swap ReLU activation to ReLU_SiLU"""
    for name,layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(model, name, ReLU_SiLU())
        else:
            ReLU_to_ReLUSiLU(layer)


class PruneDataLoader(Dataset):
    def __init__(self, train_count=20, dataset_path='./KonIQ-10k/', dataset_labels_path='./data/KonIQ-10kinfo.mat',
                 is_resize=True, resize_height=498, resize_width=664):
        self.dataset_path = dataset_path
        self.dataset_labels_path = dataset_labels_path
        self.resize = is_resize
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.images_count = train_count
        Info = h5py.File(self.dataset_labels_path, 'r')
        index = Info['index'][:, 0]
        ref_ids = Info['ref_ids'][0, :]
        index = index[0: self.images_count]

        self.imgs_indexs = []
        for i in range(len(ref_ids)):
            if ref_ids[i] in index:
                self.imgs_indexs.append(i)
        print("# PRUNE images: {}".format(len(self.imgs_indexs)))

        self.label = Info['subjective_scores'][0, self.imgs_indexs].astype(np.float32)
        self.im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.imgs_indexs]
        self.ims = []
        for im_name in self.im_names:
            im = Image.open(os.path.join(self.dataset_path, im_name)).convert('RGB')
            if self.resize:  # resize or not?
                im = resize(im, (self.resize_height, self.resize_width))  # h, w
            im = to_tensor(im)
            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im = im.unsqueeze(0)
            self.ims.append(im)
    def __len__(self):
        return len(self.imgs_indexs)
    def __getitem__(self, index) -> Any:
        im = self.ims[index]
        # im = to_tensor(im)
        # im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # im = im.unsqueeze(0)
        label = self.label[index]
        return im, label
    def __iter__(self):
        return zip(self.ims, self.label)

class FeatureExtractor(nn.Module):
    def __init__(self, layers: List):
        super(FeatureExtractor, self).__init__()
        self.layers = layers
        for layer in self.layers:
            for block in range(len(layer)):
                for conv in layer[block].children():
                    if isinstance(conv, nn.Conv2d):
                        print(conv)
        self.feature_maps = nn.ModuleList([nn.Sequential(conv, nn.AvgPool2d((1, 1)))
                              for layer in self.layers 
                              for block in range(len(layer)) 
                              for conv in layer[block].children()
                              if isinstance(conv, nn.Conv2d)])
    def forward(self, x):
        feature_maps = [x:=feature_map(x) for feature_map in self.feature_maps]
        return feature_maps

class PruneConv(BasePruningMethod):
    # PRUNING_TYPE = 'unstructured'

    def __init__(self, amount, c=2):
        _validate_pruning_amount(amount)
        self.amount = amount
        self.c = c
        
    def compute_mask(self, t, default_mask):
        return super().compute_mask(t, default_mask)
    
    @staticmethod
    def flatten(features):
        print(len(features))
        # for f in features:
        #     print(f.shape)
        n_samples = features[0].shape[0]
        conv_count = len(features)
        print('N samples:',n_samples)
        X = None
        for idx in range(conv_count):
            if X is None:
                # print(features[idx].shape)
                X = features[idx].reshape(n_samples, -1)
                # print(X.shape)
                PruneConv.idx_score_layer.append((0, X.shape[1]-1))
            else:
                tmp = features[idx].reshape(n_samples, -1)
                PruneConv.idx_score_layer.append((X.shape[1], X.shape[1] + tmp.shape[1]-1))
                X = np.hstack((X, tmp))
        # print(X.shape)
        X = np.array(X)
        return X
    
    @staticmethod
    def get_layer_features(model, feature_maps, loader):
        convs_count = len(feature_maps)
        X = [None for _ in range(convs_count)]
        y = None
        for im, label in loader:
            x = model.conv1(im)
            out = [x:=feature(x) for feature in feature_maps]
            out = [item.detach().numpy() for item in out]
            if X[0] is not None:
                X = [np.vstack((X[i], out[i])) for i in range(convs_count)]
                y = np.vstack((y, np.array(label)))
                print(y)
            else:
                X = out
                y = np.array(label)
        return X, y
    @classmethod
    def _load_data(cls, *args, **kwargs) -> Tuple[List, List]:
        data_loader = PruneDataLoader(train_count=kwargs.get('train_count'))
        # dataset = DataLoader(data_loader, batch_size=kwargs.get('train_count'),
                            #  shuffle=False, num_workers=1, pin_memory=False)
        return data_loader

    @classmethod
    def apply(cls, model, name, amount, c=2,
              importance_scores=None, /,
              train_count=20, dataset_path='./KonIQ-10k/', 
              dataset_labels_path='./data/KonIQ-10kinfo.mat', resize=True, 
              resize_height=498, resize_width=664):
        prune_loader = PruneConv._load_data(cls, train_count=train_count, dataset_path=dataset_path,
                             dataset_labels_path=dataset_labels_path, resize=resize,
                             resize_height=resize_height, resize_width=resize_width)
        
        layers = [module for label, module in model.named_children() if 'layer' in label]
        PruneConv.idx_score_layer = []
        conv1 = model.conv1
        feature_maps = nn.ModuleList([nn.Sequential(conv, nn.AvgPool2d((1, 1)))
                              for layer in layers 
                              for block in range(len(layer)) 
                              for conv in layer[block].children()
                              if isinstance(conv, nn.Conv2d)])

        
        # convs_count = len(feature_maps)
        # X = [None for _ in range(convs_count)]
        # y = None
        # for im, label in prune_loader:
        #     x = conv1(im)
        #     out = [x:=feature(x) for feature in feature_maps]
        #     out = [item.detach().numpy() for item in out]
        #     if X[0] is not None:
        #         X = [np.vstack((X[i], out[i])) for i in range(convs_count)]
        #         y = np.vstack((y, np.array(label)))
        #         print(y)
        #     else:
        #         X = out
        #         y = np.array(label)
        X, y = PruneConv.get_layer_features(model, feature_maps, prune_loader)
        print(len(X), len(X[0]), len(feature_maps))
        X = PruneConv.flatten(X)
        print(X.shape, y.shape)

        pls_model = PLSRegression(n_components=c, scale=True)
        pls_model.fit(X, y)
        

        # return super(PruneConv, cls).apply(module, name, amount=amount, c=c, importance_scores=importance_scores)
    