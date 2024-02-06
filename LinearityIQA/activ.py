from torch.nn import Module, ReLU, SiLU
import torch.nn.functional as F

# class SILU(Module):
#     __constants__ = ['inplace']

#     def __init__(self, inplace=False):
#         super(SILU, self).__init__()
#         self.inplace = inplace

#     def forward(self, input):
#         return input*F.sigmoid(input, inplace=self.inplace)

#     def extra_repr(self):
#         inplace_str = 'inplace=True' if self.inplace else ''
#         return inplace_str


def ReLU_to_SILU(model):
    for name,layer in model.named_children():
        if isinstance(layer, ReLU):
            setattr(model, name, SiLU())
        else:
            ReLU_to_SILU(layer)
