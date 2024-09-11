import torch
from torch.autograd import Function
from torch import nn

class Activation_smooth_relu(Function):
    def __init__(self):
        pass

class Smooth_ReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        pass