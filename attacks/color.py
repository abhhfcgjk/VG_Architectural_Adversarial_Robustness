
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision.transforms.functional import resize, to_tensor, normalize

# image quantization
def quantization(x):
    x_quan = torch.round(x * 255) / 255
    return x_quan


# picecwise-linear color filter
def CF(img, param, pieces):

    param = param[:, :, None, None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    for i in range(pieces):
        total_image += (
            torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
        )
    total_image *= pieces / color_curve_sum
    return total_image


def step(model, k, b, optimizer, inputs, Paras, pieces, ref_image, metric_range, device):
    const = 0.5
    batch_size = inputs.shape[0]
    Paras.data = torch.clamp(Paras.data, min=0)
    Paras_sum = torch.sum(Paras.view(batch_size, 3, -1), dim=2, keepdim=True)

    # regularization on the adjustment
    l2 = torch.sum(
        (((Paras / Paras_sum - 1 / pieces) ** 2)).view(batch_size, -1), dim=1
    )

    adv = CF(inputs, Paras, pieces)
    adv[adv > 1] = 1
    adv[adv < 0] = 0
    score = (
        model(normalize(adv.to(device), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    )
    l2_loss = (const * l2).sum()
    sign = 1
    score = (score[-1] * k[0] + b[0])
    loss = 1 - score * sign / metric_range + l2_loss.to(device)
    optimizer.zero_grad()
    loss.backward()
    Paras.grad = torch.nan_to_num(Paras.grad)
    optimizer.step()
    inputs.data.clamp_(0.0, 1.0)

    return quantization(adv).detach(), score.detach(), loss.detach()


def attack(compress_image, ref_image=None, model=None, k=None, b=None, metric_range=100, device="cpu"):
    pieces = 64
    compress_image = Variable(compress_image.clone().to(device), requires_grad=False)

    batch_size = compress_image.shape[0]
    best_score = -100000
    o_best_adversary = compress_image.clone()

    Paras = torch.ones(batch_size, 3, pieces).to(device) * 1 / pieces
    Paras.requires_grad = True
    optimizer = optim.Adam([Paras], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(10):
        # print(iteration, best_score)
        # perform the adversary
        adv, score, loss = step(
            model, k, b,
            optimizer,
            compress_image,
            Paras,
            pieces,
            ref_image,
            metric_range,
            device,
        )
        if (score > best_score):
            best_score = score
            o_best_adversary = adv.clone()

    res_image = (o_best_adversary).data.clamp_(min=0, max=1)
    return res_image
