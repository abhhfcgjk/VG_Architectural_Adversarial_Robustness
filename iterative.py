from torch.autograd import Variable
from typing import Any, List
import torch
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.transforms import Normalize

from icecream import ic

"""KonCept"""
def loss_fn(output, metric_range, k, b):
    loss = 1 - (output) / metric_range
    return loss

# iterative attack baseline (IFGSM attack)
def attack_callback(
        image_,
        model=None,
        attack_type="IFGSM",
        metric_range=100,
        device="cuda",
        eps=1.0,
        iters=10,
        delta=1 / 255
):
    """
    Attack function.
    Args:
    image: (torch.Tensor of shape [1,3,H,W]) clear image to be attacked.
    model: (PyTorch model): Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method that supports backpropagation.
    iters: (int) number of iterations. Can be ignored, during testing always set to default value.
    alpha: (float) step size for signed gradient methods. Can be ignored, during testing always set to default value.
    device (str or torch.device()): Device to use in computaions.
    eps: (float) maximum allowed pixel-wise difference between clear and attacked images (in 0-1 scale).
    Returns:
        torch.Tensor of shape [1,3,H,W]: adversarial image with same shape as image argument.
    """

    if attack_type == "IFGSM":
        additive = torch.zeros_like(image_).to(device)
    elif attack_type == "PGD":
        additive = torch.rand_like(image_).to(device)
    else:
        raise "No attack_type. Got {}. Expected IFGSM, PGD.".format(attack_type)

    im_denorm = image_.clone()
    for _ in range(iters):
        additive.data.clamp_(-10/255, 10/255)
        im = Variable(normalize(im_denorm+additive, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), requires_grad=True)
        output = model(im)
        loss = loss_fn(output, metric_range)
        model.zero_grad()
        loss.backward()
        im_grad = im.grad.data
        additive.data -= delta * im_grad.sign()
        
    perturbed_im = image_ + additive
    perturbed_im.clamp_(0.0, eps)
    return perturbed_im

