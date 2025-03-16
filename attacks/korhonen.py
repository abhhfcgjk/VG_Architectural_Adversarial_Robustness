import torch
from typing import List
from torch.autograd import Variable
import numpy as np
import cv2
from scipy import ndimage
from torchvision import transforms
from torch import Tensor
from tqdm import tqdm
from .base import Attacker


# def loss_fn(output, metric_range, k, b):
#     return 1 - (output[-1]*k[0]+b[0])/metric_range

# def rgb2ycbcr(im_rgb):
#     im_rgb = im_rgb.astype(np.float32)
#     im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
#     im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
#     im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0
#     im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0
#     return im_ycbcr

# def makeSpatialActivityMap(im):
#     im = im.cpu().detach().permute(0, 2, 3, 1).numpy()[0]
#     im = rgb2ycbcr(im)
#     im_sob = ndimage.sobel(im[:,:,0])
#     im_zero = np.zeros_like(im_sob)
#     im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]
#     maxval = im_zero.max()
#     if maxval == 0:
#         im_zero = im_zero + 1
#         maxval = 1
#     im_sob = im_zero /maxval
#     DF = np.array([[0, 1, 1, 1, 0], 
#                     [1, 1, 1, 1, 1], 
#                     [1, 1, 1, 1, 1], 
#                     [1, 1, 1, 1, 1], 
#                     [0, 1, 1, 1, 0]]).astype('uint8')
#     out_im = cv2.dilate(im_sob, DF)
#     return out_im

# def korhonen(
#         image_,
#         model,
#         k: List[int],
#         b: List[int],
#         metric_range=100,
#         device="cuda",
#         iters=50,
#         lr=0.002
# ):
#     sp_map = makeSpatialActivityMap(image_.clone() * 255)
#     sp_map = sp_map / 255
#     sp_map = transforms.ToTensor()(sp_map.astype(np.float32))
#     sp_map = sp_map.unsqueeze_(0)
#     sp_map = sp_map.to(device)
#     compress_image = Variable(image_.clone().to(device), requires_grad=True)
#     optimizer = torch.optim.Adam([compress_image], lr = lr)
    
#     for i in range(iters):
#         output = model(compress_image)
#         loss = loss_fn(output, metric_range, k, b)
#         loss.backward()
#         compress_image.grad = torch.nan_to_num(compress_image.grad)
#         compress_image.grad *= sp_map
#         optimizer.step()
#         compress_image.data.clamp_(0., 1.)
#         optimizer.zero_grad()
#     res_image = (compress_image).data.clamp_(0., 1.)
#     return res_image


class Korhonen(Attacker):
    def __init__(self, 
                 model,
                 loss_computer,
                 iters: int = 50,
                 lr=0.002,
                 device="cuda",
                 *args, **kwargs):
        super().__init__(model)
        self.lr = lr
        self.device = device
        self.iters = iters
        self.loss_computer = loss_computer

    @staticmethod
    def _rgb2ycbcr(im_rgb):
        im_rgb = im_rgb.astype(np.float32)
        im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
        im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
        im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0
        im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0
        return im_ycbcr

    @classmethod
    def _makeSpatialActivityMap(cls, im):
        im = im.cpu().detach().permute(0, 2, 3, 1).numpy()[0]
        im = cls._rgb2ycbcr(im)
        im_sob = ndimage.sobel(im[:,:,0])
        im_zero = np.zeros_like(im_sob)
        im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]
        maxval = im_zero.max()
        if maxval == 0:
            im_zero = im_zero + 1
            maxval = 1
        im_sob = im_zero /maxval
        DF = np.array([[0, 1, 1, 1, 0], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1], 
                        [0, 1, 1, 1, 0]]).astype('uint8')
        out_im = cv2.dilate(im_sob, DF)
        return out_im

    def run(self, inputs: Tensor, target: Tensor):
        sp_map = self._makeSpatialActivityMap(inputs.clone() * 255)
        sp_map = sp_map / 255
        sp_map = transforms.ToTensor()(sp_map.astype(np.float32))
        sp_map = sp_map.unsqueeze_(0)
        sp_map = sp_map.to(self.device)
        compress_image = Variable(inputs.clone().to(self.device), requires_grad=True)
        optimizer = torch.optim.Adam([compress_image], lr=self.lr)
        
        for i in range(self.iters):
            output = self.model(compress_image)
            loss = -self.loss_computer(output, None)
            loss.backward()
            compress_image.grad = torch.nan_to_num(compress_image.grad)
            compress_image.grad *= sp_map
            optimizer.step()
            compress_image.data.clamp_(0., 1.)
            optimizer.zero_grad()
        res_image = (compress_image).data.clamp_(0., 1.)
        return res_image
