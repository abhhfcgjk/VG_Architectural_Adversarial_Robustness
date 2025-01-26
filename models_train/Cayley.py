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

# class IterativeInverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, psi=1., eps=1e-3, max_iters=50):
#         """
#         Forward pass for matrix inversion using Newton-Schulz iteration.
#         """
#         n = A.size(-1)
#         I = torch.eye(n, dtype=A.dtype, device=A.device)
#         X = A.transpose(1, 2) / torch.norm(torch.bmm(A,A.transpose(1, 2)), p='fro')
#         # X = psi* I
#         ctx.save_for_backward(A, X, I)
#         ctx.num_iters = max_iters
#         residual = 0
#         res0 = 10000
#         X_best = X
#         # for _ in range(num_iters):
#             # X = 2 * X - X @ A @ X
#         for i in range(max_iters):
#             X_new = 2 * X - torch.bmm(X, torch.bmm(A, X))
#             residual = torch.norm(torch.bmm(A, X_new) - I, dim=(1, 2)).max()# torch.norm(A @ X_new - I)  # Residual error
#             # print(residual)
#             if res0 > residual:
#                 X_best = X_new
#                 res0 = residual
#             elif res0 < eps and residual > res0:
#                 X = X_best
#                 ctx.num_iters = i + 1
#                 break
#             else:
#                 X = X_new
#         ctx.save_for_backward(A, X)
#         print("Iterative inverse: {} iterations, error {}".format(ctx.num_iters, residual))
#         return X
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for matrix inversion.
#         """
#         A, X = ctx.saved_tensors
#         # print(A.shape, X.shape, grad_output.shape)
#         num_iters = ctx.num_iters
#         grad_A = torch.zeros_like(A)
#         for _ in range(num_iters):
#             grad_A += -grad_output @ X.transpose(1, 2) @ A.transpose(1, 2) - grad_output.transpose(1, 2) @ X.transpose(1, 2) @ A
#         return grad_A, None, None, None

class IterativeInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, psi=1.0, eps=1e-4, max_iters=50):
        """
        Forward pass for matrix inversion using Newton-Schulz iteration.
        """
        n = A.size(-1)
        I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
        # X = A.transpose(-1, -2) / torch.norm(A, p='fro', dim=(-1, -2), keepdim=True).square()
        #X = A.transpose(1, 2) / torch.norm(torch.bmm(A,A.transpose(1, 2)), p='fro')
        X = psi* I

        for i in range(max_iters):
            AX = torch.bmm(A, X)
            residual = torch.norm(AX - I, dim=(-1, -2)).max().item()
            if residual < eps:
                break
            X.copy_(2 * X - torch.bmm(X, AX))
        
        ctx.save_for_backward(A, X)
        ctx.num_iters = i
        #print("Iterative inverse: {} iterations, error {}".format(ctx.num_iters, residual))
        return X
    # @staticmethod
    # def forward(ctx, A, psi=1.0, eps=1e-4, max_iters=50):
    #     n = A.size(-1)
    #     I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
        
    #     ctx.save_for_backward(A, I)

    #     chunks = torch.chunk(A, chunks=A.shape[0] // 10, dim=0)
    #     inv_chunks = []

    #     for chunk in chunks:
    #         X = psi * I

    #         for i in range(max_iters):
    #             with torch.no_grad():  # Save memory by disabling gradient tracking
    #                 AX = torch.bmm(chunk, X)
    #                 residual = torch.norm(AX - I, dim=(-1, -2)).max().item()
    #                 if residual < eps:
    #                     break
    #                 X.copy_(2 * X - torch.bmm(X, AX))  # In-place update to save memory

    #         inv_chunks.append(X)

    #     X = torch.cat(inv_chunks, dim=0)
    #     ctx.save_for_backward(A, X)
    #     ctx.num_iters = i
    #     return X
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for matrix inversion.
        """
        A, X = ctx.saved_tensors
        I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device).expand_as(A)
        grad_A = torch.zeros_like(A)
        # Gradients from the iterative steps
        for _ in range(ctx.num_iters):
            AX = torch.bmm(A, X)
            grad_A -= torch.bmm(grad_output, AX.transpose(-1, -2)) + torch.bmm(grad_output.transpose(-1, -2), AX)
        return grad_A, None, None, None

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    V_herm = V.conj().transpose(1, 2) @ V
    U_skew = U - U.conj().transpose(1, 2)
    A = U_skew + V_herm

    # batch_size = U_skew.shape[0]
    # results = []
    # for i in range(batch_size):
    #     matrix = U_skew[i]
    #     is_toeplitz = True
    #     # Check if each row is a shifted version of the previous one
    #     for j in range(1, matrix.shape[0]):
    #         if not torch.all(matrix[j] == torch.roll(matrix[j-1], shifts=1)):
    #             is_toeplitz = False
    #             break
    #     results.append(is_toeplitz)
    # print(results)

    psi = 1 #0 - 1j #torch.tensor(0 + 1j, dtype=torch.complex64)
    iIpA = torch.inverse(I + A)
    # iIpA = torch.linalg.cholesky(I+A)
    # psi = 2. / (1. + torch.norm(U_skew.conj(), p='fro') + torch.norm(V_herm.conj(), p='fro'))
    # mu = U_skew.abs().sum(dim=1).max(dim=1).values
    # nu = V_herm.abs().sum(dim=1).max(dim=1).values
    
    #print(torch.norm(U @ U.conj().transpose(1, 2)))
    #psi = -2j * (mu / (mu*mu + (1+nu)*(1+nu))).reshape(-1, 1, 1)
    #psi = (2. / (1. + mu + nu)).reshape(-1, 1, 1)

    # psi = 2. / (1. + torch.norm(U_skew, p='fro') + torch.norm(V_herm, p='fro'))
    # print(psi)
    # iIpA = IterativeInverse.apply(I + A, psi, 1e-4, 200)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

class CayleyConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.eval:
            self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float64, requires_grad=True).cuda())
        else:
            self.register_parameter('alpha', None)
    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
    def forward(self, x):
        x = x.to(torch.float64)
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)]\
                .reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0)
        xfft = xfft.reshape(n * (n // 2 + 1), cin, batches)
        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n))\
                   .reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
        wfft = wfft.to(torch.complex64)
        xfft = xfft.to(torch.complex64)
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


if __name__=='__main__':
    A = torch.rand(4, 4, dtype=torch.float64, requires_grad=True).cuda() + 4 * torch.eye(4).cuda()
    X = IterativeInverse.apply(A, 10)  # Forward pass with 10 iterations
    loss = torch.sum(X)
    print(X)
    print(torch.inverse(A))
    print(loss)
    loss.backward()  # Backpropagate
    print(A.grad)