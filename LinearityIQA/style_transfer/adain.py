from torch import nn
import torch

from .adain_blocks import VGG, Decoder

class AdaIN(nn.Module):
    def __init__(self):
        pass


class StyleTransfer:
    def __init__(self, decoder_ckpt='style_transfer/checkpoints/decoder.pth', vgg_ckpt='./style_transfer/checkpoints/vgg_normalised.pth'):
        self.decoder = Decoder
        self.decoder.eval()
        self.decoder.load_state_dict(torch.load(decoder_ckpt))
        self.decoder.cuda()
        # self.decoder = nn.DataParallel(self.decoder)

        self.vgg = VGG
        self.vgg.eval()
        self.vgg.load_state_dict(torch.load(vgg_ckpt))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.vgg.cuda()
        # self.vgg = nn.DataParallel(self.vgg)

        self.shape_label: torch.Tensor
        self.texture_label: torch.Tensor
        self.debiased_label: torch.Tensor

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @staticmethod
    def adaptive_instance_normalization(content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = StyleTransfer.calc_mean_std(style_feat)
        content_mean, content_std = StyleTransfer.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def style_transfer(self, content, style, alpha=1.0, interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)
        content_f = self.vgg(content)
        style_f = self.vgg(style)
        if interpolation_weights:
            raise NotImplementedError
        else:
            feat = self.adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return self.decoder(feat)

    def __call__(self, image, label, alpha, label_mix_alpha=0):
        n, c, h, w = image.shape
        content = image.cuda()
        random_index = torch.randperm(n)
        style = image[random_index].cuda()
        from icecream import ic
        label_style = torch.cat((label[0][random_index].view(1,-1), label[1][random_index].view(1,-1)), dim=0).cuda()
        # ic(label_style)
        with torch.no_grad():
            stylized_image = self.style_transfer(content, style, alpha)
        
        # from torchvision.utils import save_image
        # from random import random
        # r = int(random()*10000)
        # save_image(stylized_image, f'delete_me{r}.png')
        # save_image(style, f'style_delete_me{r}.png')
        # save_image(content, f'content_delete_me{r}.png')
        
        self.shape_label = label
        self.texture_label = label_style
        self.debiased_label = torch.ones_like(label).cuda() * label_mix_alpha
        # if replace:
        #     return stylized_image, (self.shape_label, self.texture_label, self.debiased_label)
        # else:
        label1 = torch.cat([label, label], dim=1).cuda()
        label2 = torch.cat([torch.zeros_like(label), label_style], dim=1).cuda()
        label_weight = torch.cat([torch.zeros_like(label), torch.ones_like(label) * label_mix_alpha], dim=1).cuda()
        ret_label = (label1, label2, label_weight)
        return stylized_image, ret_label

