from torch import nn
from torch import Tensor
import pandas as pd

import torch
from torch.nn.utils.prune import BasePruningMethod, _validate_pruning_amount_init
from torch.nn.utils import prune
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression

import os
from typing import Tuple, List, Any
import copy


# import inceptionresnetv2
from PIL import Image
# import h5py
from tqdm import tqdm

from .projection import PLSRegressionCUDA, ADMM, PwoA


class PruneDataLoader(Dataset):
    def __init__(self, train_count=20, dataset_path='./KonIQ-10k/', dataset_labels_path='./data/KonIQ-10kinfo.mat',
                 is_resize=True, is_crop=False, height=498, width=664):
        self.dataset_path = dataset_path
        self.dataset_labels_path = dataset_labels_path
        self.resize = is_resize
        self.crop = is_crop
        self.height = height
        self.width = width
        self.images_count = train_count

        Info = pd.read_csv(self.dataset_labels_path)
        Info = Info[Info['set']=='train'].head(self.images_count)
        self.label = Info['MOS'].to_numpy().astype(np.float64)
        self.im_names = Info['image_name'].to_numpy()

        # Transformations
        transform_list = []
        if self.crop:
            transform_list.append(v2.RandomCrop((self.height, self.width)))
        elif self.resize:
            transform_list.append(v2.Resize((self.height, self.width)))
        transform_list += [v2.ToTensor(), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = v2.Compose(transform_list)

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im = Image.open(os.path.join(self.dataset_path, im_name)).convert('RGB')
        im = self.transform(im)
        label = self.label[index]
        return im, label

from torch.optim.lr_scheduler import _LRScheduler

class EveryThirdEpochScheduler(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, last_epoch=-1):
        """
        Custom learning rate scheduler that decreases the learning rate by a
        factor every third epoch.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            factor (float): Factor by which to multiply the learning rate.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.factor = factor
        super(EveryThirdEpochScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Check if the epoch is a multiple of 3 (excluding 0)
        if (self.last_epoch + 1) % 3 == 0 and self.last_epoch >= 0:
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

class HSICEstimator:
    def __init__(self, model, amount=0.1, batch_size=8, lr=1e-4, epochs=5, **kwargs):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.amount = amount
        self.model = copy.deepcopy(model)
        self.pretrained = model # pre-trained model

        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.scheduler = EveryThirdEpochScheduler(self.optimizer, factor=0.1)
        self.pretrained.eval()

        self.masks = {}
        self.current_epoch = 0
        self.end_epoch = epochs
        self.convs = self.__get_convs_list()

        self.admm = ADMM(self.convs, self.amount, admm_epochs=self.epochs)
        self.pwoa = PwoA(self.convs)

        self.loss_fn = lambda X, y, out, out_pre, hidden, epoch, batch_idx: self.admm(epoch, batch_idx) + self.pwoa(X, y, out, out_pre, hidden)

    def fit(self):
        self._prune_loop()

    def set_data_loader(self, *args, **kwargs):
        self.data_loader = DataLoader(PruneDataLoader(**kwargs), 
                                      batch_size=self.batch_size, 
                                      num_workers=4,
                                      shuffle=True)
        # self.data_loader = PruneDataLoader(**kwargs)

    def _init_hook(self):
        # print(len(self.hooks) if hasattr(self, 'hooks') else None)
        assert ((not hasattr(self, 'hooks'))or(hasattr(self, 'hooks') and len(self.hooks)==0)), 'Exists active hooks'
        hidden_outputs = []

        def hook_fn(module, input, output):
            hidden_outputs.append(output)

        # Register hooks for all layers
        self.hooks = []
        for name, module in self.model.named_modules():
            # Add hooks for specific layer types (e.g., nn.Conv2d, nn.Linear)
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
        return hidden_outputs

    def _close_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _prune_step(self, step, data):
        hidden = self._init_hook()
        X, y = data
        X = X.cuda()
        y = y.cuda()
        # print(X.shape, y.shape, len(data))
        output = self.model(X)
        output_pre = self.pretrained(X)
        loss = self.loss_fn(X, y, output, output_pre, hidden, self.current_epoch, step)
        self._close_hook()
        loss.backward()
    
    def _prune_loop(self):
        while self.current_epoch < self.end_epoch:
            self.model.train()
            for step, data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                self._prune_step(step, data)
            self.scheduler.step()
            self.current_epoch += 1

    def get_prune_idxs(self):
        return self.admm.idxs

    def __get_convs_list(self):
        def __help(model):
            ans = []
            for _, layer in model.named_children():
                if isinstance(layer, nn.Conv2d):
                    ans.append(layer)
                else:
                    ans += __help(layer)
            return ans
        convs = __help(self.model)
        # print(self.convs)
        return convs

class HSICPrune(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    prune_idxs: List
    convs: List

    def __init__(self, amount, c=2):
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        mask = t
        return mask.to('cuda')

    @classmethod
    def apply(cls, model, name, amount,
              importance_scores=None, /, 
              batch_size=8, epochs=10, learning_rate=1e-4,
              train_count=2000, dataset_path='KonIQ-10k/1024x768',
              dataset_labels_path='KonIQ-10k/koniq10k_distributions_sets.csv', **kwargs):
        hsic_prune = HSICEstimator(model, amount, batch_size, lr=learning_rate, epochs=epochs)
        hsic_prune.set_data_loader(train_count=train_count, dataset_path=dataset_path,
                                   dataset_labels_path=dataset_labels_path, 
                                   is_resize=True, is_crop=False, 
                                   height=384, width=512)
        hsic_prune.fit()
        cls.prune_idxs = hsic_prune.get_prune_idxs()
        cls.convs = hsic_prune.convs
        for i in range(len(hsic_prune.convs)):
            idxs = [item for item in cls.prune_idxs if item[0] == i]
            if len(idxs) != 0:
                idxs = idxs[0][1]
            module = hsic_prune.convs[i]
            shape = module.weight.shape
            weight2d = module.weight.reshape(shape[0], -1)
            importance_scores = np.ones_like(weight2d.detach().cpu().numpy())
            importance_scores[:, idxs] = 0
            importance_scores = importance_scores.reshape(shape)
            importance_scores = torch.from_numpy(importance_scores)
            super(HSICPrune, cls).apply(module, name, amount, importance_scores=importance_scores)
        return cls

class PLSEsitimator:
    def __init__(self, model, n_components=2, kernel=None, discriminative=False, device='cuda'):
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.n_comp = n_components
        self.kernel = kernel
        self.idx_score_layer = []
        self.model = model
        self.convs = []
        self.discriminative = discriminative

        # self.convs = self.__get_convs_list()
        self.__get_convs_list()
        print(len(self.convs))
        self.feature_maps = self.convs

    def __get_convs_list(self):
        def __help(model):
            ans = []
            for _, layer in model.named_children():
                if isinstance(layer, (nn.Conv2d)):
                    ans.append(layer)
                else:
                    ans += __help(layer)
            return ans
        self.convs = __help(self.model)
        print(self.convs)
        return self.convs

    def flatten(self, features) -> npt.ArrayLike:
        n_samples = features[0].shape[0]
        conv_count = len(features)

        X = None
        for idx in range(conv_count):
            if X is None:
                X = features[idx].reshape(n_samples, -1)
                self.idx_score_layer.append((0, X.shape[1] - 1))
            else:
                tmp = features[idx].reshape(n_samples, -1)
                self.idx_score_layer.append((X.shape[1], X.shape[1] + tmp.shape[1] - 1))
                X = np.hstack((X, tmp))
        X = np.array(X)
        return X

    def VIP(self, x, y, pls_model):
        t = pls_model.T
        w = pls_model.W
        q = pls_model.Q

        m, p = x.shape
        _, h = t.shape
        s = torch.diag(torch.mm(t.T @ t, q.T @ q)).reshape(h, -1)  # (h, 1)
        total_s = torch.sum(s)

        # Normalize weights for each component
        norm_w = torch.linalg.norm(w, dim=0, keepdim=True)  # (1, h)
        normalized_w = (w / norm_w) ** 2  # (p, h)

        # Compute VIP scores for all features simultaneously
        vip_scores = torch.sqrt(
            p * (normalized_w @ s / total_s).squeeze()
        )  # (p,)

        self.vips = vip_scores
        return self.vips
    
    def __register_hooks(self, hook_fn):
        self.hooks = []
        layers_to_hook = self.feature_maps
        for layer in layers_to_hook:
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def __remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_layer_features(self, loader: Dataset) -> Tuple[List, List]:
        outs = []
        def __hook_fn(module, input, output):
            outs.append(output)
        self.__register_hooks(__hook_fn)
        convs_count = len(self.feature_maps)
        X = [None for _ in range(convs_count)]
        y = None
        for im, label in tqdm(loader, total=len(loader)):
            im = im.to(self.device)
            # x = model[0](im)
            x = im
            _ = self.model(x)
            out = [item.detach().cpu().numpy() for item in outs]
            if X[0] is not None:
                X = [np.vstack((X[i], out[i])) for i in range(convs_count)]
                y = np.vstack((y, np.array(label)))
                # print(y)
            else:
                X = out
                y = np.array(label)
        self.__remove_hooks()
        return X, y

    def _pooling_module(self, features):
        if self.kernel is not None:
            return [F.max_pool2d(x, kernel_size=self.kernel) if x.shape[-1]>=self.kernel else x for x in features]
        return features

    def get_prune_idxs(self, amount=0.1) -> List:
        self.low_bound = self.find_closer_th(amount)
        output = []
        for i in range(len(self.score_layer)):
            score_filters = self.score_layer[i]
            idxs = np.where(score_filters <= self.low_bound)[0]
            if len(idxs) == len(score_filters):
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
                scores = np.concatenate((scores, self.score_layer[i]))  #np.hstack
        total = scores.shape[0]
        # print("scores shape: ", scores.shape)

        esstimations = np.zeros((total))
        for i in range(total):
            th = scores[i]
            destin = len(np.where(scores <= th)[0]) / total
            esstimations[i] = abs(percentage - destin)

        th = scores[np.argmin(esstimations)]
        return th

    def _generate_score_layer(self, X, y) -> None:
        scores = self.VIP(X, y, self.pls).detach().cpu().numpy()
        self.score_layer = []
        self.score_max = 0
        for idx, conv in enumerate(self.convs):
            n_filters = conv.weight.shape[0]
            begin, end = self.idx_score_layer[idx]
            # print('end-begin: ', end-begin+1)
            score_layer = scores[begin:end + 1]

            features_filter = (end - begin + 1) // n_filters
            # print('features_filter:',features_filter, n_filters)
            score_filters = np.zeros((n_filters))
            for filter_idx in range(n_filters):
                score_filters[filter_idx] = np.mean(score_layer[filter_idx:filter_idx + features_filter])
            self.score_max = max(self.score_max, max(score_filters))
            self.score_layer.append(score_filters)

    def _use_discriminative_score(self):
        self.dis = np.zeros((len(self.convs),))
        print(len(self.score_layer), len(self.convs), len(self.idx_score_layer))
        for idx, conv in reversed(list(enumerate(self.convs, start=0))):
            print(idx)
            if idx==0:
                break
            begin, end = self.idx_score_layer[idx]
            score_layer = self.vips[begin:end + 1]
            self.dis[idx] = 1/self.__cv(score_layer)

            begin, end = self.idx_score_layer[idx-1]
            score_layer = self.vips[begin:end + 1]
            self.dis[idx-1] = 1/self.__cv(score_layer)

            if self.dis[idx] < self.dis[idx-1]:
                continue
            else:
                """Do not prune current layer"""
                n_filters = conv.weight.shape[0]
                score_filters = np.ones((n_filters))
                for filter_idx in range(n_filters):
                    score_filters[filter_idx] = 1.
                self.score_layer[idx] = score_filters

    def fit(self, X, y):
        self.pls_model = PLSRegressionCUDA(n_components=self.n_comp)
        self.pls_model.fit(X, y)
        self.pls = self.pls_model
        # self.pls = PLSGPU(self.pls_model, batch_size=X.shape[0])

        self._generate_score_layer(X, y)
        if self.discriminative:
            self._use_discriminative_score()

    @staticmethod
    def __cv(x):
        return torch.std(x)/torch.mean(x)

class PLSPrune(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    prune_idxs: List
    convs: List

    def __init__(self, amount, c=2):
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.c = c

    def compute_mask(self, t, default_mask):
        mask = t
        return mask.to('cuda')

    @classmethod
    def _load_data(cls, *args, **kwargs) -> PruneDataLoader:
        data_loader = DataLoader(PruneDataLoader(**kwargs), 
                                 batch_size=1
                                 )
        return data_loader

    @classmethod
    def apply(cls, model, name, amount, c=2,
              importance_scores=None, /, discriminative=False,
              train_count=100, kernel=1, dataset_path='KonIQ-10k/1024x768',
              dataset_labels_path='KonIQ-10k/koniq10k_distributions_sets.csv', is_resize=True, is_crop=False,
              height=384, width=512):
        prune_loader = cls._load_data(cls, train_count=train_count, dataset_path=dataset_path,
                                      dataset_labels_path=dataset_labels_path, 
                                      is_resize=is_resize, is_crop=is_crop, 
                                      height=height, width=width)
        pls_prune = PLSEsitimator(model, kernel=kernel, n_components=2, discriminative=discriminative)
        cls.convs = pls_prune.convs

        X, y = pls_prune.get_layer_features(prune_loader)
        X = pls_prune.flatten(X)
        # y = y.unsqueeze(1)
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        print('SHAPE: ', X.shape, y.shape)
        print(X.dtype, y.dtype)
        pls_prune.fit(X, y)

        cls.prune_idxs = pls_prune.get_prune_idxs(amount=amount)
        cls.pruned_convs = []
        for i in range(len(pls_prune.convs)):
            idxs = [item for item in cls.prune_idxs if item[0] == i]
            if len(idxs) != 0:
                idxs = idxs[0][1]
            module = pls_prune.convs[i]
            importance_scores = np.ones_like(module.weight.detach().cpu().numpy())
            importance_scores[idxs, :] = 0
            importance_scores = torch.from_numpy(importance_scores)
            super(PLSPrune, cls).apply(module, name, amount, c=c, importance_scores=importance_scores)
        return cls

def hsic_prune(model: nn.Module, amount, /, 
               learning_rate=1e-4, epochs=10, 
               width=664, height=498, images_count=2000, **kwargs) -> Tuple:
    resnet_model = model.base.cuda()
    h = height  #16#90
    w = width  #24#120
    t_count = images_count
    HSICPrune.apply(resnet_model, 'weight', amount, 
                    learning_rate=learning_rate, epochs=epochs,
                    train_count=t_count, 
                    is_resize=True, is_crop=False,
                    height=h, width=w)

    prune_parameters = []
    for i in range(len(HSICPrune.convs)):
        prune_parameters.append((HSICPrune.convs[i], 'weight'))

    if kwargs.get('permanent', False):
        prune_parameters = tuple(prune_parameters)
        for i in range(len(HSICPrune.convs)):
            module = HSICPrune.convs[i]
            prune.remove(module, 'weight')
    return prune_parameters

def displs_prune(model: nn.Module, amount, /, 
                 width=332, height=249, images_count=100, kernel=None, 
                 **kwargs) -> Tuple:
    resnet_model = nn.Sequential(*list(model.base.children())[:-1]).to(kwargs.get('device', 'cuda'))
    h = height  #16#90
    w = width  #24#120
    t_count = images_count
    PLSPrune.apply(resnet_model, 'weight', amount, 
                   discriminative=True, train_count=t_count, 
                   is_resize=True, is_crop=False,
                   height=h, width=w, kernel=kernel)

    prune_parameters = []
    for i in range(len(PLSPrune.convs)):
        prune_parameters.append((PLSPrune.convs[i], 'weight'))

    if kwargs.get('permanent', False):
        prune_parameters = tuple(prune_parameters)
        for i in range(len(PLSPrune.convs)):
            module = PLSPrune.convs[i]
            prune.remove(module, 'weight')
    return prune_parameters

def pls_prune(model: nn.Module, amount, /, 
              width=332, height=249, images_count=100, kernel=None, 
              **kwargs) -> Tuple:
    print(model)
    resnet_model = nn.Sequential(*list(model.base.children())[:-1]).to(kwargs.get('device', 'cuda'))
    h = height  #16#90
    w = width  #24#120
    t_count = images_count
    PLSPrune.apply(resnet_model, 'weight', amount, 
                   discriminative=False, train_count=t_count, 
                   is_resize=True, is_crop=False,
                   height=h, width=w, kernel=kernel)

    prune_parameters = []
    for i in range(len(PLSPrune.convs)):
        prune_parameters.append((PLSPrune.convs[i], 'weight'))

    if kwargs.get('permanent', False):
        prune_parameters = tuple(prune_parameters)
        for i in range(len(PLSPrune.convs)):
            module = PLSPrune.convs[i]
            prune.remove(module, 'weight')
    return prune_parameters


def get_prune_features(model: nn.Module) -> List:
    prune_params_list = []

    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d)):
            prune_params_list.append((module, 'weight'))
        else:
            prune_params_list += get_prune_features(module)
    return prune_params_list


def l1_prune(model: nn.Module, amount: float, **kwargs) -> Tuple:
    prune_params = tuple(get_prune_features(model))
    prune.global_unstructured(
        parameters=prune_params,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    if kwargs.get('permanent', False):
        for module, name in prune_params:
            prune.remove(module, name)
    print(list(model.named_buffers()))
    return prune_params


def ln_prune(model: nn.Module, amount: float, n: int, **kwargs) -> Tuple:
    prune_params = tuple(get_prune_features(model))
    for module, name in prune_params:
        prune.ln_structured(module, name, amount=amount, n=n, dim=0)

    if kwargs.get('permanent', False):
        for module, name in prune_params:
            prune.remove(module, name)
    return prune_params
