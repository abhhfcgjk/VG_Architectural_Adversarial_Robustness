from typing import Any
import torch
# from torch.nn import Module, ReLU, SiLU, Conv2d
from torch import nn
import torch.nn.functional as F
from torch import Tensor, exp
from torch.autograd import Function
from torch.nn.utils.prune import BasePruningMethod, _validate_pruning_amount, _validate_pruning_amount_init, remove
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize
import numpy as np
import numpy.typing as npt
from sklearn.cross_decomposition import PLSRegression

import os
from typing import Tuple, List

from PIL import Image
import h5py
from tqdm import tqdm


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
        # resize_height =12
        # resize_width=12
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


class PLSEssitimator:
    def __init__(self, model, arch='resnet', n_components=2, device='cuda'):
        assert 'resnet' in arch
        self.device = 'cuda' if (device=='cuda' and torch.cuda.is_available()) else 'cpu'
        self.n_comp = n_components
        self.idx_score_layer = []
        self.model = model
        self.layers = [module for label, module in self.model.named_children() 
                       if 'layer' in label]
        self.convs = [conv for layer in self.layers 
                      for block in range(len(layer)) 
                      for conv in layer[block].children()
                      if isinstance(conv, nn.Conv2d)]
        self.feature_maps = nn.ModuleList([
                            nn.Sequential(conv, nn.AvgPool2d((1, 1))) for conv in self.convs
                            ])
        

    def flatten(self, features) -> npt.ArrayLike:
        # print(len(features))
        # print(features[0].shape)
        # for f in features:
        #     print(f.shape)
        n_samples = features[0].shape[0]
        conv_count = len(features)
        
        # print('N samples:',n_samples)
        X = None
        for idx in range(conv_count):
            if X is None:
                # print(features[idx].shape)
                X = features[idx].reshape(n_samples, -1)
                # print(X.shape)
                self.idx_score_layer.append((0, X.shape[1]-1))
            else:
                tmp = features[idx].reshape(n_samples, -1)
                self.idx_score_layer.append((X.shape[1], X.shape[1] + tmp.shape[1]-1))
                X = np.hstack((X, tmp))
            # print('features',features[idx].shape, X.shape, PruneConv.idx_score_layer[-1][1]-PruneConv.idx_score_layer[-1][0]+1)
        # print(X.shape)
        X = np.array(X)
        return X
    

    def VIP(self, x, y, pls_model):
        t = Tensor(pls_model.x_scores_).to(self.device)
        w = Tensor(pls_model.x_weights_).to(self.device)
        q = Tensor(pls_model.y_loadings_).to(self.device)

        m, p = x.shape
        _, h = t.shape
        
        vips = np.zeros((p,))
        print(vips.shape)
        # s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        s = torch.diag(torch.mm(torch.mm(torch.mm(t.t(), t), q.t()), q)).reshape(h, -1)
        print('S',s.shape, s.t())
        total_s = torch.sum(s)
        print(total_s)

        for i in tqdm(range(p), total=p):
            weight = Tensor([(w[i, j] / torch.linalg.norm(w[:, j])).to(self.device) ** 2 
                             for j in range(h)]).to(self.device)
            weight = weight.unsqueeze(1)
            # print(weight.shape)
            #vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
            elem = torch.sqrt(p * (torch.mm(s.t(), weight).to(self.device)) / total_s).to(self.device)
            vips[i] = elem.detach().cpu().numpy()
        # vips = vips.detach().numpy()
        print('vips len: ', len(vips), vips[0])
        return vips

    def get_layer_features(self, model, loader: Dataset):

        convs_count = len(self.feature_maps)
        X = [None for _ in range(convs_count)]
        y = None
        for im, label in loader:
            x = model.conv1(im)
            out = [x:=feature(x) for feature in self.feature_maps]
            out = [item.detach().cpu().numpy() for item in out]
            if X[0] is not None:
                X = [np.vstack((X[i], out[i])) for i in range(convs_count)]
                y = np.vstack((y, np.array(label)))
                print(y)
            else:
                X = out
                y = np.array(label)
        return X, y

    def get_prune_idxs(self, amount=0.1):
        low_bound = self.find_closer_th(amount)
        output = []
        for i in range(len(self.score_layer)):
            score_filters = self.score_layer[i]
            idxs = np.where(score_filters <= low_bound)[0]
            if len(idxs)==len(score_filters):
                print(f"Warning: All filters at layer [{i}]")
                idxs = []
            output.append((i, idxs))
        return output
    
    def find_closer_th(self, percentage):
        scores = None
        print("score layer:")
        for i in range(len(self.score_layer)):
            # print('SCORE filter: ', len(score_layer))
            if scores is None:
                scores = self.score_layer[i]
            else:
                scores = np.concatenate((scores, self.score_layer[i])) #np.hstack
        total = scores.shape[0]
        print("scores shape: ", scores.shape)
        
        esstimations = np.zeros((total))
        for i in range(total):
            th = scores[i]
            # print(np.where(scores<=th))
            destin = len(np.where(scores <= th)[0])/total
            # print('Destin: ', destin)
            esstimations[i] = abs(percentage - destin)
        
        th = scores[np.argmin(esstimations)]
        return th

    def _generate_score_layer(self, X, y) -> None:
        scores = self.VIP(X, y, self.pls)
        self.score_layer = []
        for idx, conv in enumerate(self.convs):
            n_filters = conv.weight.shape[0]
            print(conv)
            print('weight shape', conv.weight.shape)
            
            begin, end = self.idx_score_layer[idx]
            print('end-begin: ', end-begin+1)
            score_layer = scores[begin:end+1]

            features_filter = (end-begin+1)//n_filters
            print('features_filter:',features_filter, n_filters)
            score_filters = np.zeros((n_filters))
            for filter_idx in range(n_filters):
                score_filters[filter_idx] = np.mean(score_layer[filter_idx:filter_idx+features_filter])
            self.score_layer.append(score_filters)

    def fit(self, X, y):
        self.pls = PLSRegression(n_components=self.n_comp, scale=True)
        self.pls.fit(X, y)

        self._generate_score_layer(X, y)

class PruneConv(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    SAVE_PATH = "prune/resnet"
    def __init__(self, amount, c=2):
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.c = c
        
    def compute_mask(self, t, default_mask):
        mask = t
        return mask

    @classmethod
    def _load_data(cls, *args, **kwargs) -> Tuple[List, List]:
        # print(args, kwargs)
        data_loader = PruneDataLoader(**kwargs)
        # dataset = DataLoader(data_loader, batch_size=kwargs.get('train_count'),
                            #  shuffle=False, num_workers=1, pin_memory=False)
        return data_loader

    @classmethod
    def apply(cls, model, name, amount, c=2,
              importance_scores=None, /,
              train_count=20, dataset_path='./LinearityIQA/KonIQ-10k/', 
              dataset_labels_path='./LinearityIQA/data/KonIQ-10kinfo.mat', is_resize=True, 
              resize_height=498, resize_width=664):
        prune_loader = cls._load_data(cls, train_count=train_count, dataset_path=dataset_path,
                             dataset_labels_path=dataset_labels_path, is_resize=is_resize,
                             resize_height=resize_height, resize_width=resize_width)
        
        pls_prune = PLSEssitimator(model)
        cls.convs = pls_prune.convs

        X, y = pls_prune.get_layer_features(model, prune_loader)
        X = pls_prune.flatten(X)

        pls_prune.fit(X, y)

        cls.prune_idxs = pls_prune.get_prune_idxs(amount=amount)

        cls.pruned_convs = []
        for i in range(len(pls_prune.convs)):
            idxs = [item for item in cls.prune_idxs if item[0]==i]
            if len(idxs)!=0:
                idxs = idxs[0][1]
            module = pls_prune.convs[i]
            importance_scores = np.ones_like(module.weight.detach().cpu().numpy())
            importance_scores[idxs, :] = 0
            importance_scores = torch.from_numpy(importance_scores)
            super(PruneConv, cls).apply(module, name, amount=amount, c=c, importance_scores=importance_scores)

        return cls
    
    @classmethod
    def remove(cls, name: str):
        for i in range(len(cls.pruned_convs)):
            module = cls.pruned_convs[i]
            remove(module, name)


def PLS_prune_resnet(model):
    pass