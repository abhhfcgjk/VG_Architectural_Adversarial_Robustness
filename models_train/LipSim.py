# import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Normalize, v2
from torchvision.transforms.functional import pil_to_tensor

def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class SDPConvLin(nn.Module):

    def __init__(self, cin, cout, kernel_size=3):
        super(SDPConvLin, self).__init__()

        self.activation = nn.ReLU(inplace=False)

        self.kernel = nn.Parameter(torch.empty(cin, cout, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.randn(cin))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        batch_size, cout, x_size, x_size = x.shape
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(torch.sqrt(t))
        x = t[None, :, None, None] * x
        out = F.conv_transpose2d(x, self.kernel, padding=1) + self.bias[None, :, None, None]
        return out


class SDPLin(nn.Module):

    def __init__(self, cin, cout, bias=True):
        super(SDPLin, self).__init__()

        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(cout))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.q = nn.Parameter(torch.rand(cin))

    def forward(self, x):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        wtw = self.weight.T @ self.weight
        t = torch.abs(q_inv * wtw * q).sum(1)
        t = torch.sqrt(safe_inv(t))
        W = self.weight * t
        out = F.linear(x, W, self.bias)
        return out


class SDPBasedLipschitzConvLayer(nn.Module):

    def __init__(self, cin, inner_dim, kernel_size=3, stride=1):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.padding = kernel_size // 2

        self.kernel = nn.Parameter(torch.randn(inner_dim, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        t = t.reshape(1, -1, 1, 1)
        res = F.conv2d(x, self.kernel, padding=1)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.conv_transpose2d(res, self.kernel, padding=1)
        out = x - res
        return out


class SDPBasedLipschitzLinearLayer(nn.Module):

    def __init__(self, cin, inner_dim):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, cin))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out


class PaddingChannels(nn.Module):

    def __init__(self, ncout, ncin=3, mode="zero"):
        super(PaddingChannels, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x
            return out


class PoolingLinear(nn.Module):

    def __init__(self, ncin, ncout, agg="mean"):
        super(PoolingLinear, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.agg = agg

    def forward(self, x):
        if self.agg == "trunc":
            return x[:, :self.ncout]
        k = 1. * self.ncin / self.ncout
        out = x[:, :self.ncout * int(k)]
        out = out.view(x.shape[0], self.ncout, -1)
        if self.agg == "mean":
            out = np.sqrt(k) * out.mean(axis=2)
        elif self.agg == "max":
            out, _ = out.max(axis=2)
        return out
    



class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class Projection(nn.Module):

    def __init__(self, n_classes):
        super(Projection, self).__init__()
        self.v = 1 / np.sqrt(n_classes)

    def forward(self, x):
        return torch.clamp(x, -self.v, self.v)


class L2LipschitzNetwork(nn.Module):

    def __init__(self, n_classes, depth=30, num_channels=30, depth_linear=5, n_features=2048, conv_size=5):
        super(L2LipschitzNetwork, self).__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.depth_linear = depth_linear
        self.n_features = n_features
        self.conv_size = conv_size
        self.n_classes = n_classes
        # self.config = config

        imsize = 224
        self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

        layers = []
        for _ in range(self.depth):  # config, input_size, cin, cout, kernel_size=3, epsilon=1e-6
            layers.append(
                SDPBasedLipschitzConvLayer(self.num_channels, self.conv_size)
            )
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        self.convs = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]
        in_channels = self.num_channels * 14 * 14
        for _ in range(self.depth_linear):
            layers_linear.append(
                SDPBasedLipschitzLinearLayer(in_channels, self.n_features)
            )
        self.linear = nn.Sequential(*layers_linear)
        self.base = nn.Sequential(*[self.conv1, self.convs, self.linear])
        self.last = PoolingLinear(in_channels, self.n_classes, agg="trunc")

    def forward(self, x):
        # x = self.base(x)
        ic('fst', x.shape)
        # for model in self.base:
        #     x = model(x)

        x = self.base(x)
        x = self.last(x)
        ic(self.last)
        ic(x.shape)
        return x


# class LipSimBase(nn.Module):
#     def __init__(self, )


from typing import List

def extract_features(lipsim, step_back=2) -> List:
    features = [lipsim.base[0]]

    for model in lipsim.base[1].children():
        # features += [ for m in model]
        # for m in model.children():
        #     features.append(m)
        features.append(model)
    if step_back:
        features = features[:-step_back]
    return features

class Identity(nn.Module):
    def forward(self, x):
        return x

if __name__=='__main__':
    # from IQAdataset import get_data_loaders
    from PIL import Image
    import os
    from icecream import ic
    # loader = get_data_loaders()
    server_mnt = "~/mnt/dione/28i_mel"
    destination_path = os.path.expanduser(server_mnt)
    weightsdir =os.path.join(destination_path, 'model.ckpt-1.pth')
    transform = v2.Compose([
        v2.Resize((224, 224)), #224 # 498x664
        v2.ToDtype(torch.float32),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im = pil_to_tensor(Image.open('../KonIQ-10k/826373.jpg').convert('RGB')).unsqueeze(0)
    im = transform(im)

    lipsim = L2LipschitzNetwork(1, depth=20,depth_linear=7, num_channels=45, n_features=1024, conv_size=5)
    # ic(lipsim)


    ckpt = torch.load(weightsdir)['model_state_dict']
    for key in list(ckpt.keys()):
        ckpt[key.replace('module.model.', '')] = ckpt[key]
        del ckpt[key]
    lipsim.load_state_dict(ckpt)


    features = extract_features(lipsim)
    ic(features)

    x = im
    
    for model in features:
        x = model(x)
        ic(x.shape)
    print('iter', x.shape)

    # x = im
    # x = F.max_pool2d(x, kernel_size=(16,32), stride=(2,2), padding=(0,0))
    # ic(x.shape)
    # x = F.max_pool2d(x, kernel_size=(18,42), stride=(1,1), dilation=(1,2), padding=0)
    # ic(x.shape)
    # print(list(model.children())[4])

    lipsim.base[-1] = Identity()
    lipsim.last = Identity()
    print(lipsim(im).shape)


