import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from pathlib import Path
from collections import OrderedDict
from unittest.mock import seal

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block, Attention

from activ import swap_all_activations, ReLU_ELU, ReLU_SiLU
from pruning.pruning import l1_prune, ln_prune, pls_prune, displs_prune, hsic_prune
from Cayley import CayleyBlockPool
from swin import SwinTransformer
from KDE import RBFAttention

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def print_sparcity(self):
        """only for resnet"""
        print("SPARCITY")
        p_list = self.prune_parameters
        print(p_list.__len__())
        for (module, attr) in p_list:
            percentage = 100. * float(torch.sum(getattr(module, attr) == 0)) / float(getattr(module, attr).nelement())

            print("Sparsity in {}.{} {}: {:.2f}%".format(module.__class__.__name__, attr,
                                                         getattr(module, attr).shape, percentage))
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=(2, 2), window_size=4, dim_mlp=768, num_heads=(4, 4),
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.is_rtoken = kwargs.get("rtoken", False)
        self.is_kde = kwargs.get("kde", False)
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        
        if self.is_rtoken:
            n_tokens = 2
            self.input_size += n_tokens
            # print(self.patches_resolution)
            self.patches_resolution = (self.input_size, self.input_size)
            dim_mlp += 58*n_tokens
            window_size = 5
            self.rtoken = torch.nn.Parameter(
                1e-2 * torch.randn(1, 58*n_tokens, embed_dim*4)
            )

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        if self.is_kde:
            self.swap_attention()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []  # Clear the list

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def swap_attention(self):
        def __help(module):
            for name, layer in module.named_children():
                if isinstance(layer, Attention):
                    n_head = layer.num_heads
                    d_model = layer.qkv.in_features
                    d_head = layer.qkv.in_features 
                    dropatt = layer.attn_drop.p
                    dropout = layer.proj_drop.p
                    setattr(module, name, RBFAttention(n_head, 
                                                       d_model, 
                                                       d_head, 
                                                       dropout=dropout, 
                                                       dropatt=dropatt))
                else:
                    __help(layer)
        __help(self)
        print(f"All {Attention} objects swaped to {RBFAttention}")

    def forward(self, x):
        import logging
        
        # logging.debug(x.shape)
        b, _, _, _ = x.shape

        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.clear()

        if self.is_rtoken:
            logging.debug(x.shape)
            logging.debug(self.rtoken.repeat(b, 1, 1).shape)
            x = torch.cat([x, self.rtoken.repeat(b, 1, 1)], dim=1)

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        logging.debug(x.shape)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / (torch.sum(w) + 1e-6)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

class ImageNormalizer(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
        std: tuple[float, float, float] = [0.229, 0.224, 0.225],
        persistent: bool = True,
    ):
        super(ImageNormalizer, self).__init__()

        self.register_buffer(
            "mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=persistent
        )
        self.register_buffer(
            "std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=persistent
        )

    def forward(self, inputs: torch.Tensor):
        return (inputs - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: tuple[float, float, float] = [0.485, 0.456, 0.406],
    std: tuple[float, float, float] = [0.229, 0.224, 0.225],
):
    # layers = OrderedDict([("crop", tv.transforms.Resize([691, 921])), ("normalize", ImageNormalizer(mean, std)), ("model", model)])
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)

