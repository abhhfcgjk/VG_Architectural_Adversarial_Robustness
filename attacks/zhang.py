import torch

from torch.autograd import Variable
from typing import List

# from IQA_pytorch import SSIM as SSIM_OPT
from pytorch_msssim import SSIM

def loss_fn(im1, im2, output, metric_range, k, b, ssim_calc):
    return ssim_calc(im1, im2) - (output[-1]*k[0]+b[0])/metric_range

def zhang(
        image_,
        model,
        k: List[int],
        b: List[int],
        metric_range=100,
        device="cuda",
        iters=10,
        lr=0.005
):
    ssim_calc = SSIM().to(device)
    model.eval()
    in_image = image_
    compress_image = Variable(image_.clone().to(device), requires_grad=True)
    in_image.requires_grad = False
    optimizer = torch.optim.Adam([compress_image], lr=lr)
    for i in range(iters):
        score = model(compress_image.to(device))
        #loss = loss_fn(compress_image, in_image, score, metric_range, k, b, ssim_calc).to(device)
        loss = ssim_calc(compress_image, in_image) - (score[-1]*k[0]+b[0])/metric_range
        loss.backward()
        compress_image.grad.data[torch.isnan(compress_image.grad.data)] = 0
        optimizer.step()
        compress_image.data.clamp_(min=0, max=1)
        compress_image.data[torch.isnan(compress_image.data)] = 0
        optimizer.zero_grad()

    res_image = (compress_image).data.clamp_(min=0, max=1)
    return res_image
