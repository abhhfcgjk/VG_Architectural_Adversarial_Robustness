from torch.autograd import Variable
from typing import Any, List
import torch


def norm(score: int, mmin: int, mmax: int, metric_range=1):
    return (score - mmin) * metric_range / (mmax - mmin)


def get_score(y: List[Any], k: List[int], b: List[int]):
    return y[-1] * k[0] + b[0]


# def loss_fn(output, metric_range, k, b):
#     loss = 1 - (output[-1] * k[0] + b[0]) / metric_range
#     return loss

def loss_fn(output, metric_range, k, b):
    loss = 1 - (output) / metric_range
    return loss

# iterative attack baseline (IFGSM attack)
def attack_callback(
        image,
        model=None,
        attack_type="IFGSM",
        metric_range=100,
        device="cpu",
        eps=10 / 255,
        iters=10,
        alpha=1 / 255,
        k: List[int] = None,
        b: List[int] = None,
        mmin=0,
        mmax=100
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
    image = Variable(image.clone().to(device), requires_grad=True)
    if attack_type == "IFGSM":
        additive = torch.zeros_like(image).to(device)
    elif attack_type == "PGD":
        additive = torch.rand_like(image).to(device)
    else:
        raise "No attack_type. Got {}. Expected IFGSM, PGD.".format(attack_type)

    additive = Variable(additive, requires_grad=True)

    for _ in range(iters):
        with torch.autograd.set_detect_anomaly(True):
            img = Variable(image + additive, requires_grad=True)
            img.data.clamp_(0.0, 1.0)
            y = model(img)
            # y[-1] = norm(get_score(y, k, b),mmin, mmax)
            loss = loss_fn(y, metric_range, k, b)

            model.zero_grad()
            loss.backward()
            input_grad = img.grad.data

        gradient_sign = input_grad.sign()
        additive.data -= alpha * gradient_sign
        additive.data.clamp_(-eps, eps)

    res_image = (image + additive).data.clamp_(min=0, max=1)

    return res_image
