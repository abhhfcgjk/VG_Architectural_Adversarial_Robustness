from torch import nn
from torch import Tensor

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

from PIL import Image
import h5py
from tqdm import tqdm
from icecream import ic

from .projection import PLSGPU, ADMM, PwoA
from .VOneNet.modules import GFB, VOneBlock

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
        transform_list = []
        if self.crop:
            transform_list.append(v2.RandomCrop((self.height, self.width)))
        elif self.resize:
            transform_list.append(v2.Resize((self.height, self.width)))
        transform_list += [v2.ToTensor(), 
                      v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        transform = v2.Compose(transform_list)
        for im_name in self.im_names:
            im = Image.open(os.path.join(self.dataset_path, im_name)).convert('RGB')
            
            # if self.resize:  # resize or not?
                # im = resize(im, (self.resize_height, self.resize_width))  # h, w
            # im = to_tensor(im)
            # im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im = transform(im)
            # im = im.unsqueeze(0)
            self.ims.append(im)

    def __len__(self):
        return len(self.imgs_indexs)

    def __getitem__(self, index):
        im = self.ims[index]
        label = self.label[index]
        return im, label

    def __iter__(self):
        return zip(self.ims, self.label)

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
              train_count=2000, dataset_path='./KonIQ-10k/',
              dataset_labels_path='./data/KonIQ-10kinfo.mat', is_resize=True, is_crop=False,
              height=498, width=664):
        hsic_prune = HSICEstimator(model, amount, batch_size, lr=learning_rate, epochs=epochs)
        hsic_prune.set_data_loader(train_count=train_count, dataset_path=dataset_path,
                                   dataset_labels_path=dataset_labels_path, 
                                   is_resize=is_resize, is_crop=is_crop, 
                                   height=height, width=width)
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
    def __init__(self, model, arch='resnet', n_components=2, kernel=None, discriminative=False, device='cuda'):
        assert 'resnet' in arch
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.n_comp = n_components
        self.kernel = kernel
        self.idx_score_layer = []
        self.model = model
        self.convs = []
        self.discriminative = discriminative
        ic(self.model)

        # self.convs = self.__get_convs_list()
        self.__get_convs_list()
        print(len(self.convs))

        # features = []
        # ic(len(self.convs))
        # for i, conv in enumerate(self.convs):
        #     # ic(conv.weight.shape)
        #     features.append(nn.Sequential(conv))
        # self.feature_maps = nn.ModuleList(features)
        self.feature_maps = self.convs
        ic(self.feature_maps)

    def __get_convs_list(self):
        # def __help(model):
        #     ans = []
        #     for _, layer in model.named_children():
        #         if isinstance(layer, (nn.Conv2d, GFB)):
        #             ans.append(layer)
        #         else:
        #             ans += __help(layer)
        #     return ans
        # self.convs = __help(self.model)
        # print(self.convs)
        # return self.convs

        for layer in self.model:
            # ic(type(layer))
            if isinstance(layer, (nn.Sequential, nn.Conv2d, VOneBlock)):
                if isinstance(layer, nn.Conv2d):
                    layer = nn.Sequential(nn.Sequential(layer))
                elif isinstance(layer, VOneBlock):
                    layer = nn.Sequential(nn.Sequential(layer.simple_conv_q0))
                for block in layer:
                    ic("elem", block)
                    for conv in block.children():
                        if isinstance(conv, (nn.Conv2d, GFB)):
                            self.convs.append(conv)
        # print(self.convs)
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
        t = Tensor(pls_model.x_scores_).to(self.device)
        w = Tensor(pls_model.x_weights_).to(self.device)
        q = Tensor(pls_model.y_loadings_).to(self.device)

        m, p = x.shape
        _, h = t.shape

        self.vips = np.zeros((p,))
        print(self.vips.shape)
        s = torch.diag(torch.mm(torch.mm(torch.mm(t.t(), t), q.t()), q)).reshape(h, -1)
        print('S', s.shape, s.t())
        total_s = torch.sum(s)
        print(total_s)

        for i in tqdm(range(p), total=p):
            weight = Tensor([(w[i, j] / torch.linalg.norm(w[:, j])).to(self.device) ** 2
                             for j in range(h)]).to(self.device)
            weight = weight.unsqueeze(1)
            elem = torch.sqrt(p * (torch.mm(s.t(), weight).to(self.device)) / total_s).to(self.device)
            self.vips[i] = elem.detach().cpu().numpy()
        return self.vips

    def get_layer_features(self, loader: Dataset) -> Tuple[List, List]:
        convs_count = len(self.feature_maps)
        X = [None for _ in range(convs_count)]
        y = None
        for im, label in tqdm(loader, total=len(loader)):
            im = im.to(self.device)
            # x = model[0](im)
            x = im
            out = [x := feature(x) for feature in self.feature_maps]
            
            # out = self._pooling_module(out)
            out = [item.detach().cpu().numpy() for item in out]
            if X[0] is not None:
                X = [np.vstack((X[i], out[i])) for i in range(convs_count)]
                y = np.vstack((y, np.array(label)))
                # print(y)
            else:
                X = out
                y = np.array(label)
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
        scores = self.VIP(X, y, self.pls)
        self.score_layer = []
        self.score_max = 0
        for idx, conv in enumerate(self.convs):
            n_filters = conv.weight.shape[0]
            # print(conv)
            # print('weight shape', conv.weight.shape)

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
        self.dis = np.zeros((100,))
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
        self.pls_model = PLSRegression(n_components=self.n_comp, scale=True)
        self.pls_model.fit(X, y)
        self.pls = self.pls_model
        # self.pls = PLSGPU(self.pls_model, batch_size=X.shape[0])

        self._generate_score_layer(X, y)
        if self.discriminative:
            self._use_discriminative_score()

    @staticmethod
    def __cv(x):
        return np.std(x)/np.mean(x)

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
        data_loader = PruneDataLoader(**kwargs)
        return data_loader

    @classmethod
    def apply(cls, model, name, amount, c=2,
              importance_scores=None, /, discriminative=False,
              train_count=20, kernel=1, dataset_path='./KonIQ-10k/',
              dataset_labels_path='./data/KonIQ-10kinfo.mat', is_resize=True, is_crop=False,
              height=498, width=664):
        prune_loader = cls._load_data(cls, train_count=train_count, dataset_path=dataset_path,
                                      dataset_labels_path=dataset_labels_path, 
                                      is_resize=is_resize, is_crop=is_crop, 
                                      height=height, width=width)
        pls_prune = PLSEsitimator(model, kernel=kernel, n_components=2, discriminative=discriminative)
        cls.convs = pls_prune.convs

        X, y = pls_prune.get_layer_features(prune_loader)
        X = pls_prune.flatten(X)
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
    resnet_model = model.features
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

    prune_parameters = tuple(prune_parameters)
    for i in range(len(HSICPrune.convs)):
        module = HSICPrune.convs[i]
        prune.remove(module, 'weight')
    return prune_parameters

def displs_prune(model: nn.Module, amount, /, width=120, height=90, images_count=50, kernel=None) -> Tuple:
    resnet_model = model.features
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

    prune_parameters = tuple(prune_parameters)
    for i in range(len(PLSPrune.convs)):
        module = PLSPrune.convs[i]
        prune.remove(module, 'weight')
    return prune_parameters

def pls_prune(model: nn.Module, amount, /, width=120, height=90, images_count=50, kernel=None) -> Tuple:
    print(model)
    resnet_model = model.features
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

    prune_parameters = tuple(prune_parameters)
    for i in range(len(PLSPrune.convs)):
        module = PLSPrune.convs[i]
        prune.remove(module, 'weight')
    return prune_parameters


def get_prune_features(model: nn.Module) -> List:
    prune_params_list = []

    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, GFB)):
            prune_params_list.append((module, 'weight'))
        else:
            prune_params_list += get_prune_features(module)
    return prune_params_list


def l1_prune(model: nn.Module, amount: float) -> Tuple:
    prune_params = tuple(get_prune_features(model))
    prune.global_unstructured(
        parameters=prune_params,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for module, name in prune_params:
        prune.remove(module, name)
    return prune_params


def ln_prune(model: nn.Module, amount: float, n: int) -> Tuple:
    prune_params = tuple(get_prune_features(model))
    for module, name in prune_params:
        prune.ln_structured(module, name, amount=amount, n=n, dim=0)
        prune.remove(module, name)
    return prune_params
