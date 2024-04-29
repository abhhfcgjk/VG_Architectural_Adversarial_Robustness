from torch import nn
import torch.nn.functional as F
from torch import Tensor, exp
from torch.autograd import Function


class Activaion_forward_ReLU_backward_SiLU(Function):
    @staticmethod
    def forward(ctx, x, inplace):
        result = F.relu(x, inplace=inplace)
        dx = 1 / (1 + exp(-x)) + x * exp(-x) / (1 + exp(-x)) ** 2
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

