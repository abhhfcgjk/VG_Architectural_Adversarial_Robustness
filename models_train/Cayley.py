import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from typing import Optional, List, Tuple, Union

from icecream import ic

# Extend this class to get emulated striding (for stride 2 only)
class StridedConv(nn.Module):
    def __init__(self, *args, **kwargs):
        striding = False
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            kwargs['stride'] = 1
            striding = True
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
            args = tuple(args)
        else: # handles OSSN case
            if len(args) == 3:
                kwargs['padding'] = args[2] // 2
            else:
                kwargs['padding'] = kwargs['kernel_size'] // 2
        super().__init__(*args, **kwargs)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if striding:
            self.register_forward_pre_hook(lambda _, x: \
                    einops.rearrange(x[0], downsample, k1=2, k2=2))  
        
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

class CayleyConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.eval:
            self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True).cuda())
        else:
            self.register_parameter('alpha', None)
        # self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True).cuda())


    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
    
    def forward(self, x):
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        ic(x.shape)
        ic(n, n*(n//2+1),cin, batches)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0)
        ic(xfft.shape)
        xfft = xfft.reshape(n * (n // 2 + 1), cin, batches)
        ic(xfft.shape)
        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
        yfft = (cayley(self.alpha * wfft / wfft.norm()) @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y



class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True).cuda())
        self.alpha.data = self.weight.norm()

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
        self.Q = None
            
    def forward(self, X):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        return F.linear(X, self.Q if self.training else self.Q.detach(), self.bias)
    
class PrintConv(nn.Conv2d):
    def forward(self, X):
        out = super().forward(X)
        ic(out.shape)
        return out

class CayleyBlock(nn.Module):
    __constants__ =['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}
    def __init__(self, in_channels=2048, intermed_channels=1024, stride=(1,1), padding=(8,4), kernel_size=(2,4)):
        super(CayleyBlock, self).__init__()
        
        # self.conv_in = nn.ConvTranspose2d(in_channels, intermed_channels, stride=stride, kernel_size=(3,4), padding=(1,1))
        self.conv_in = nn.Conv2d(in_channels, intermed_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.conv_cayley = CayleyConv(intermed_channels, intermed_channels, stride=(1,1), kernel_size=3)
        self.conv_out = nn.Conv2d(intermed_channels, in_channels, kernel_size=3, stride=stride)
    
    def forward(self, X):
        ic("Cayley(")
        ic(X.shape)
        x = self.conv_in(X)
        ic(x.shape)
        out = self.conv_cayley(x)
        ic(out.shape)
        out = self.conv_out(out)
        ic(out.shape)
        ic(")cayley")
        return out
    
class CayleyBlockPool(nn.Module):
    __constants__ =['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}
    def __init__(self, in_channels=2048, intermed_channels=1024, stride=(1,1), padding=(0,0), kernel_size=3):
        super(CayleyBlockPool, self).__init__()
        
        # self.h = height
        # self.w = weight

        self.conv_in = nn.Conv2d(in_channels, intermed_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.conv_cayley = CayleyConv(intermed_channels, in_channels, stride=(1,1), kernel_size=3)
        # self.conv_out = nn.AvgPool2d(kernel_size=3, stride=stride)
    
    def pool_to_square(self, x, h, w):
        if w > h:
            delta = w - h
            up = int(np.ceil(h*0.02)+1)
            ic(up, delta)
            kernel_size = (up, up+delta)
            return F.max_pool2d(x, kernel_size=kernel_size, stride=1)
        elif w < h:
            delta = h - w
            up = int(np.ceil(w*0.02))
            kernel_size = (delta+up, up)
            return F.max_pool2d(x, kernel_size=kernel_size)
        return x

    def forward(self, X):
        ic("Cayley(")
        # ic(X.shape)
        # x = self.conv_in(X)
        ic(X.shape)
        _, _, h, w = X.shape
        x = self.pool_to_square(X, h, w)
        ic(x.shape)
        x = self.conv_in(x)
        ic(x.shape)
        out = self.conv_cayley(x)
        ic(out.shape)
        # out = self.conv_out(out)
        # ic(out.shape)
        ic(")cayley")
        return out

def swap_conv_to_lipschitz(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            print(name, layer)
            print(layer.in_channels, layer.out_channels,layer.kernel_size, layer.padding,layer.stride)
            ort_conv = PrintConv(layer.in_channels, layer.out_channels,
                                    kernel_size=layer.kernel_size[0], padding=layer.padding,
                                    stride=layer.stride)
            
            print(ort_conv)
            setattr(model, name, ort_conv)

        else:
            swap_conv_to_lipschitz(layer)

def get_lipschitz_model(model):
    return swap_conv_to_lipschitz(model)
