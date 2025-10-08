import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F
import numpy as np
from AOC.fast_block_ortho_conv import (
    transpose_kernel,
    fast_matrix_conv,
)


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class AOLReparametrizer(nn.Module):
    def __init__(self, nb_features, groups):
        super(AOLReparametrizer, self).__init__()
        self.nb_features = nb_features
        self.groups = groups
        self.q = nn.Parameter(torch.ones(nb_features, 1, 1, 1))

    def forward(self, kernel):
        ktk = fast_matrix_conv(
            transpose_kernel(kernel, self.groups, flip=True), kernel, self.groups
        )
        ktk = torch.abs(ktk)
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(torch.sqrt(t))
        t = t.reshape(-1, 1, 1, 1)
        kernel = kernel * t
        return kernel


class AOLConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        """
        Almost-Orthogonal Convolution layer. This layer implements the method proposed in [1] to enforce
        almost-orthogonality. While orthogonality is not enforced, the lipschitz constant of the layer
        is guaranteed to be less than 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding size. Default is 0.
            dilation (int or tuple, optional): Dilation rate. Default is 1.
            groups (int, optional): Number of groups. Default is 1.
            bias (bool, optional): Whether to include a learnable bias. Default is True.
            padding_mode (str, optional): Padding mode. Default is "zeros".
            device (torch.device, optional): Device to store the layer parameters. Default is None.
            dtype (torch.dtype, optional): Data type to store the layer parameters. Default is None.


        References:
            `[1] Prach, B., & Lampert, C. H. (2022).
                   "Almost-orthogonal layers for efficient general-purpose lipschitz networks."
                   ECCV.`<https://arxiv.org/abs/2208.03160>`_
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        parametrize.register_parametrization(
            self,
            "weight",
            AOLReparametrizer(
                out_channels,
                groups=groups,
            ),
        )


class AOLFurierConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        """
        Almost-Orthogonal Convolution layer. This layer implements the method proposed in [1] to enforce
        almost-orthogonality. While orthogonality is not enforced, the lipschitz constant of the layer
        is guaranteed to be less than 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding size. Default is 0.
            dilation (int or tuple, optional): Dilation rate. Default is 1.
            groups (int, optional): Number of groups. Default is 1.
            bias (bool, optional): Whether to include a learnable bias. Default is True.
            padding_mode (str, optional): Padding mode. Default is "zeros".
            device (torch.device, optional): Device to store the layer parameters. Default is None.
            dtype (torch.dtype, optional): Data type to store the layer parameters. Default is None.


        References:
            `[1] Prach, B., & Lampert, C. H. (2022).
                   "Almost-orthogonal layers for efficient general-purpose lipschitz networks."
                   ECCV.`<https://arxiv.org/abs/2208.03160>`_
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        if self.eval:
            self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float64, requires_grad=True).cuda())
        else:
            self.register_parameter('alpha', None)
        parametrize.register_parametrization(
            self,
            "weight",
            AOLReparametrizer(
                out_channels,
                groups=groups,
            ),
        )
    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)
    def forward(self, X):
        _, _, h, w = X.shape
        x = F.adaptive_avg_pool2d(X, min(h,w))

        x = x.to(torch.float64)
        weight = self.weight.to(torch.float64)
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)]\
                .reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0)
        xfft = xfft.reshape(n * (n // 2 + 1), cin, batches)
        wfft = self.shift_matrix * torch.fft.rfft2(weight, (n, n))\
                .reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
        if self.alpha is None:
            self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
        wfft = wfft.to(torch.complex64)
        xfft = xfft.to(torch.complex64)
        yfft = ((self.alpha * wfft / wfft.norm()) @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y

class AOLConvTranspose2D(nn.ConvTranspose2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        """
        Almost-Orthogonal Convolution layer. This layer implements the method proposed in [1] to enforce
        almost-orthogonality. While orthogonality is not enforced, the lipschitz constant of the layer
        is guaranteed to be less than 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding size. Default is 0.
            output_padding (int or tuple, optional): Additional size added to the output shape. Default is 0.
            groups (int, optional): Number of groups. Default is 1.
            bias (bool, optional): Whether to include a learnable bias. Default is True.
            dilation (int or tuple, optional): Dilation rate. Default is 1.
            padding_mode (str, optional): Padding mode. Default is "zeros".
            device (torch.device, optional): Device to store the layer parameters. Default is None.
            dtype (torch.dtype, optional): Data type to store the layer parameters. Default is None.


        References:
            `[1] Prach, B., & Lampert, C. H. (2022).
                   "Almost-orthogonal layers for efficient general-purpose lipschitz networks."
                   ECCV.`<https://arxiv.org/abs/2208.03160>`_
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # Register the same AOLReparametrizer
        parametrize.register_parametrization(
            self,
            "weight",
            AOLReparametrizer(in_channels, groups=groups),
        )