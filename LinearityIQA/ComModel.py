'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
from torch import hub, Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, WeightsEnum
from torchvision import models
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np

from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models._api import register_model
from torchvision.models.resnet import BasicBlock, Bottleneck

class ResnetCon(models.ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
    # def __init__(self, arch="resnet34"):
        super().__init__(block, layers, num_classes, zero_init_residual, groups,
                         width_per_group, replace_stride_with_dilation, norm_layer)
        self.layer_results = {"l1": None,"l2": None,"l3": None,"l4": None, 'gg': None}
        # self.model = models.__dict__[arch](pretrained=True)
        # self.arch = arch
        # print(self.model)
        # print(self.__arches)
        # self.model = self.__arches[self.arch]()
        # print(ResNet18_Weights.IMAGENET1K_V1.url)
        # print(hub.load_state_dict_from_url(ResNet18_Weights.IMAGENET1K_V1.url).keys())

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        self.layer_results["l1"] = x
        x = self.layer2(x)
        self.layer_results["l2"] = x
        x = self.layer3(x)
        self.layer_results["l3"] = x
        x = self.layer4(x)
        self.layer_results["l4"] = x

        x = self.avgpool(x)
        self.layer_results["gg"] = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResnetCon:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResnetCon(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


# @register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResnetCon:
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


# @register_model()
@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResnetCon:
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


# @register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResnetCon:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

# if __name__=='__main__':
#     l = resnet18(pretrained=True)
#     o = models.resnet18()
#     print(l.state_dict().keys()==o.state_dict().keys())


class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.b = torch.nn.Parameter(torch.zeros(1, channel_size, output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.w, self.b)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=np.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        
        output = torch.bmm(x.transpose(0,1), self.w).transpose(0,1) + self.b
        return output


class self_correlation(nn.Module):

    def __init__(self, percentage, input_channel):
        super(self_correlation, self).__init__()
        self.percentage = percentage

        self.fc1 = LinearWithChannel(self.percentage, self.percentage, input_channel)
        self.hard = nn.ReLU6()

    def self_selected(self, c, x, topk):
        
        B, C, H, W = x.shape
        one_dim_size = H * W
        
        embedding = torch.zeros((B,C,topk)).cuda()
        x = x.reshape(B, C, one_dim_size)
        c = c.reshape(B, one_dim_size)

        top = torch.topk(c, k=topk)

        selected = c - top[0][:,-1].unsqueeze(1)
        selected = (torch.floor(selected))
        selected = F.relu(selected+0.5)*2
        #print(selected.shape)
        
        indexes = top[1]
        #print(indexes.shape)
        weights = batched_index_select(selected, 1, indexes)
        #print(weights)
        img = batched_index_select(x, 2, indexes)
        
        # Select topk points
        embedding[:,:,:topk] = weights.unsqueeze(1) * img

        embedding = self.fc1(embedding)
        #print(embedding.shape)
        v3t = embedding.transpose(1,2)
        #print(v3t.shape)
        out = torch.matmul(v3t, embedding)#有点类似a'a？

        return out

    def forward(self, im, weight):

        B, C, H, W = im.shape

        embedding = self.self_selected(weight, im, self.percentage)
        
        # embedding = embedding.reshape(B,C,H,W)
        
        return embedding

class LinearAttentionBlock(nn.Module):
    
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g


class ProjectorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    
    def forward(self, inputs):
        return self.op(inputs)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class RARTFA(nn.Module):
    __arches = {"resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50}
    def __init__(self, arch="resnet34", pretrained=True):
        super().__init__()
        self.arch = arch
        # print(vars())
        self.md = self.__arches[arch](pretrained=pretrained)

        self.attn1 = LinearAttentionBlock(512)
        self.attn2 = LinearAttentionBlock(512)
        self.attn3 = LinearAttentionBlock(512)

        self.proj1 = ProjectorBlock(64, 512)
        self.proj2 = ProjectorBlock(128, 512)
        self.proj3 = ProjectorBlock(256, 512)

        self.corr1 = self_correlation(10, 64)
        self.corr2 = self_correlation(10, 128)
        self.corr3 = self_correlation(10, 256)

        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0, bias=True)


        self.classify = nn.Linear(250, 1) #20-1200,10

    def forward(self, x):
        # return self.md(x)
        x = self.md(x)
        l1, l2, l3, l4, _ = self.md.layer_results
        if gg == None:
            gg = self.dense(l4)

        c1, g1 = self.attn1(self.proj1(l1), gg)
        out1 = self.corr1(l1, c1)

        c2, g2 = self.attn2(self.proj2(l2), gg)
        out2 = self.corr2(l2, c2)

        c3, g3 = self.attn3(self.proj3(l3), gg)
        out3 = self.corr3(l3, c3)

        g = torch.cat((out1, out2, out3), dim=1)

        g = g.view(g.size(0), -1)

        out = self.classify(g)

        return out, c1, c2

# if __name__ == '__main__':
#     m = RTRTFA()
#     print(m)
#     print(m.md.layer_results)
    # print(vars()["resnet34"])