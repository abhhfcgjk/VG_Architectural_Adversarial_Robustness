import scipy.special as sci
from torch import nn
import torch.nn.functional as F
from torch import Tensor, exp, sqrt
from torch.autograd import Function
import numpy as np

class Activaion_forward_ReLU_backward_SiLU(Function):
    @staticmethod
    def forward(ctx, x, inplace):
        result = F.relu(x, inplace=inplace)
        # x = x.cpu()
        ex = exp(-x)
        prod = x*ex
        # x = x.cpu()
        dx = (1 / (1 + ex) + prod / (1 +ex) ** 2) # d(x*sigmoid(x))/dx
        
        ctx.save_for_backward(dx)
        ctx.inplace = inplace
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx, = ctx.saved_tensors
        # dx = dx.cuda()
        result = grad_output * dx
        inplace = ctx.inplace
        return result, None
    
class Activaion_forward_ReLU_backward_ELU(Function):
    @staticmethod
    def forward(ctx, x, inplace):
        result = F.relu(x, inplace=inplace)
        # dx = 1 if x>=0 else exp(x) # d(ELU(x,alpha=1))/dx
        dx = exp(x)
        dx[dx >= 0] = 1
        
        ctx.save_for_backward(dx)
        ctx.inplace = inplace
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx, = ctx.saved_tensors
        result = grad_output * dx
        inplace = ctx.inplace
        return result, None
    
class Activaion_forward_ReLU_backward_GELU(Function):
    @staticmethod
    def forward(ctx, x, inplace):
        result = F.relu(x, inplace=inplace)
        dx = 0.5 + 0.5*sci.erf(x.cpu()/np.sqrt(2)).cuda() + 0.5*x*(2/np.sqrt(np.pi)*exp(-x*x/2))
        ctx.save_for_backward(dx)
        ctx.inplace = inplace
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx, = ctx.saved_tensors
        result = grad_output * dx
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
    
class ReLU_ELU(nn.Module):
    """
    Activation function is ReLU in forward.
    Activation function is SiLU in backward.
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU_ELU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        result = Activaion_forward_ReLU_backward_ELU.apply(input, self.inplace)
        return result

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
class ReLU_GELU(nn.Module):
    """
    Activation function is ReLU in forward.
    Activation function is SiLU in backward.
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU_GELU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        result = Activaion_forward_ReLU_backward_GELU.apply(input, self.inplace)
        return result

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def ReLU_to_SILU(model):
    """Swap ReLU activation to SiLU."""
    for name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(model, name, nn.SiLU())
        else:
            ReLU_to_SILU(layer)


def ReLU_to_ReLUSiLU(model):
    """Swap ReLU activation to ReLU_SiLU"""
    for name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(model, name, ReLU_SiLU())
        else:
            ReLU_to_ReLUSiLU(layer)


def swap_all_activations(model, from_activation: nn.Module, to_activation: nn.Module):
    for name, layer in model.named_children():
        if isinstance(layer, from_activation):
            if hasattr(to_activation, 'inplace'):
                setattr(model, name, to_activation(inplace=True))
            else:
                setattr(model, name, to_activation())
        else:
            swap_all_activations(layer, from_activation, to_activation)

