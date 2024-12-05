from models_train.gabor_layers import GaborLayer
from torchvision.models import resnet50
from torch import nn
# import torch.ao.nn.quantized as nq
import pytorch_quantization.nn as nq
from pytorch_quantization.tensor_quant import QuantDescriptor

def swap_to_gabor(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            print(name, layer)
            gabor_conv = GaborLayer(layer.in_channels, layer.out_channels,
                                    kernel_size=layer.kernel_size[0], padding=layer.padding,
                                    stride=layer.stride, kernels=1)
            setattr(model, name, gabor_conv)
        else:
            swap_to_gabor(layer)

def swap_to_quntized(model, full_copy=False):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            attrs = {
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding,
                'dilation': layer.dilation,
                'groups': layer.groups,
                'bias': layer.bias is not None,
                'padding_mode': layer.padding_mode,
            }
            input_quant = QuantDescriptor(num_bits=16, fake_quant=True)
            weight_quant = QuantDescriptor(num_bits=16, fake_quant=True)
            conv = nq.QuantConv2d(**attrs, 
                                  quant_desc_input=input_quant, 
                                  quant_desc_weight=weight_quant)
            if full_copy:
                copy_weights(conv_in=layer, conv_to=conv, bias=True)
            # conv.weight = layer.weight.data
            # conv.bias = layer.weight.data
            # conv.set_weight_bias(layer.weight.data, layer.bias.data if layer.bias else None)
            setattr(model, name, conv)
        else:
            swap_to_quntized(layer)

def copy_weights(conv_in, conv_to, bias=True):
    conv_to.weight = conv_in.weight.data
    if bias:
        conv_to.bias = conv_in.bias

if __name__=='__main__':
    resnet = resnet50(pretrained=True)
    conv = nn.Conv2d(16, 10, kernel_size=(3,3), stride=2, padding=1)
    swap_to_gabor(resnet)
    print(resnet)