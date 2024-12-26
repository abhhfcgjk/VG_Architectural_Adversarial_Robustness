from torch.autograd import Variable
from typing import Any, List
import torch
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.transforms import Normalize

from icecream import ic

def norm(score: int, mmin: int, mmax: int, metric_range=1):
    return (score - mmin) * metric_range / (mmax - mmin)


def get_score(y: List[Any], k: List[int], b: List[int]):
    return y[-1] * k[0] + b[0]

"""Linearity"""
def loss_fn(output, metric_range, k, b):
    loss = 1 - (output[-1] * k[0] + b[0]) / metric_range
    return loss

def denorm(batch, mean, std):
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    return batch * std.view(1,-1,1,1) + mean.view(1,-1,1,1)

def fgsm_attack(data, eps, alpha, model, metric_range, k, b, loss_fn=loss_fn, device='cuda'):
    noise = torch.empty(*data.shape, device=device)
    noise.uniform_(-eps, eps)

    # print(data.shape, noise.shape, noise[:].shape)
    noisy_data = data.clone() + noise
    noisy_data.requires_grad_()

    output = model(noisy_data)
    loss = loss_fn(output, metric_range, k, b).sum()
    data_grad = torch.autograd.grad(loss, noisy_data)[0].detach()

    grad_sign = data_grad.sign()
    perturbed_data = data - alpha*grad_sign
    perturbed_data.clamp_(0, 1)
    return perturbed_data

def attack_callback(
        image_,
        model=None,
        attack_type="IFGSM",
        metric_range=100,
        device="cuda",
        eps=1.0,
        iters=10,
        delta=1 / 255,
        k: List[int] = None,
        b: List[int] = None,
        # loss_fn=lambda output, metric_range, k, b: 1 - (output[-1] * k[0] + b[0]) / metric_range
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
    # ic.enable()


    # image = Variable(image_.clone().to(device) , requires_grad=True)
    if attack_type == "IFGSM":
        additive = torch.zeros_like(image_).to(device)
    elif attack_type == "PGD":
        additive = torch.rand_like(image_).to(device)
    else:
        raise "No attack_type. Got {}. Expected IFGSM, PGD.".format(attack_type)

    # additive = Variable(additive, requires_grad=True)

    # for _ in range(iters):
    #     img = image+additive
    #     img.data.clamp_(0.0, 1.0)

    #     ic(img.shape)
    #     y = model(normalize(img,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    #     # y = model(img)

    #     loss = loss_fn(y, metric_range, k, b)

    #     model.zero_grad()
    #     if additive.grad is not None:
    #         additive.grad.zero_()

    #     loss.backward()
    #     input_grad = additive.grad.data
    #     # ic(input_grad)

    #     gradient_sign = input_grad.sign()
    #     # ic(gradient_sign)
        
    #     additive.data -= alpha * gradient_sign
    #     additive.data.clamp_(-eps, eps)
        


    im_denorm = image_.clone()
    for _ in range(iters):
        additive.data.clamp_(-10/255, 10/255)
        im = Variable(normalize(im_denorm+additive, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), requires_grad=True)
        output = model(im)
        loss = loss_fn(output, metric_range,k,b)
        model.zero_grad()
        loss.backward()
        im_grad = im.grad.data
        additive.data -= delta * im_grad.sign()
        
    perturbed_im = image_ + additive
    perturbed_im.clamp_(0.0, eps)
    return perturbed_im


    # im_denorm = image_.clone()
    
    # im = Variable(normalize(im_denorm+additive, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), requires_grad=True)
    # output = model(im)
    # loss = loss_fn(output, metric_range,k,b)
    # model.zero_grad()
    # loss.backward()
    # im_grad = im.grad.data
    # epsilon = alpha
    # for _ in range(iters):
    #     additive.data += epsilon * im_grad.sign()
    # perturbed_im = image_ - additive
    # perturbed_im.clamp_(0.0, 1.0)
    # return perturbed_im



    # im_denorm = image_.clone()
    # im = Variable(normalize(im_denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), requires_grad=True)
    # for _ in range(iters):
    #     im = im + normalize(additive,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ic(im)
    #     output = model(im)
    #     loss = loss_fn(output, metric_range,k,b)
    #     model.zero_grad()
    #     loss.backward()
    #     im_grad = im.grad.data
    #     additive.data += alpha * im_grad.sign()
    #     im.grad.zero_()
    # perturbed_im = image_ - additive
    # perturbed_im.clamp_(0.0, 1.0)
    # return perturbed_im


