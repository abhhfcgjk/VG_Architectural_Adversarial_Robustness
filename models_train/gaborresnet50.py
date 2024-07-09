from gabor_layers import GaborLayer
from torchvision.models import resnet50
from torch import nn

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

if __name__=='__main__':
    resnet = resnet50(pretrained=True)
    conv = nn.Conv2d(16, 10, kernel_size=(3,3), stride=2, padding=1)
    swap_to_gabor(resnet)
    print(resnet)