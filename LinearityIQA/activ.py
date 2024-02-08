from typing import Any
from torch.nn import Module, ReLU, SiLU
import torch.nn.functional as F
from torch import Tensor, exp
from torch.autograd import Function

class Activaion_forward_ReLU_backward_SiLU(Function):
    @staticmethod
    def forward(ctx, input, inplace):
        ctx.save_for_backward(input)
        ctx.inplace = inplace
        return F.relu(input, inplace=inplace)
    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = None
        input, = ctx.saved_tensors
        if ctx.inplace:
            grad_inputs = grad_outputs.clone()
        else:
            grad_inputs = grad_outputs
        grad_inputs = grad_inputs/(1+exp(-input)) + grad_inputs*input*exp(-input)/(1+exp(-input))**2
        return grad_inputs

class ReLU_SiLU(Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU_SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return Activaion_forward_ReLU_backward_SiLU.apply(input, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def ReLU_to_SILU(model):
    for name,layer in model.named_children():
        if isinstance(layer, ReLU):
            setattr(model, name, SiLU())
        else:
            ReLU_to_SILU(layer)

def ReLU_to_ReLUSiLU(model):
    for name,layer in model.named_children():
        if isinstance(layer, ReLU):
            setattr(model, name, ReLU_SiLU())
        else:
            ReLU_to_SILU(layer)
