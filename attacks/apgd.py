from torch.autograd import Variable
from typing import Any, List
import torch
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.transforms import Normalize
from tqdm import tqdm


def check_oscillation(loss_steps, cur_step, k, thr_decr):
    t = torch.zeros(loss_steps.shape[1], device=loss_steps.device, dtype=loss_steps.dtype)
    for i in range(k):
        t += (loss_steps[cur_step - i] > loss_steps[cur_step - i - 1]).float()

    return (t <= k * thr_decr * torch.ones_like(t)).float()

def loss_fn(output, metric_range, k, b):
    loss = -(1 - (output[-1] * k[0] + b[0]) / metric_range)
    return loss

def apgd(
        image_,
        model,
        k_: List[int],
        b: List[int],
        attack_type="IFGSM",
        metric_range=100,
        device="cuda",
        eps=1.0,
        iters=10,
        delta=1 / 255,
):

    n_iter = iters
    n_iter_min = max(int(0.06 * iters), 1)
    size_decr = max(int(0.03 * iters), 1)
    k = max(int(0.22 * iters), 1)

    eps = eps / 255
    alpha = delta
    thr_decr: float = 0.75
    # loss_computer = loss_computer

    k = max(int(0.22 * n_iter), 1)
    device = image_.device

    x_adv = image_.clone()
    x_best = x_adv.clone().detach()
    loss_steps = torch.zeros([n_iter, image_.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, image_.shape[0]], device=device)

    step_size = (
        alpha
        * eps
        * torch.ones(
            [image_.shape[0], *[1] * (len(image_.shape) - 1)], device=device, dtype=image_.dtype
        )
    )

    counter3 = 0

    # 1 step of the classic PGD
    x_adv.requires_grad_()
    grad_adv_input = normalize(x_adv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    grad_adv_input.requires_grad_()

    logits = model(grad_adv_input)
    loss_indiv = loss_fn(logits, metric_range, k_, b) #loss_computer(logits)
    loss = loss_indiv.sum()

    grad = torch.autograd.grad(loss, [grad_adv_input])[0].detach()
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    loss_best = loss_indiv.detach().clone()
    loss_best = loss_best.squeeze(-1)
    loss_best_last_check = loss_best.clone()
    loss_best_last_check = loss_best_last_check.squeeze(-1)
    reduced_last_check = torch.ones_like(loss_best)

    x_adv_old = x_adv.clone().detach()

    for i in range(iters):
        x_adv = x_adv.detach()
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()

        a = 0.75 if i > 0 else 1.0

        x_adv_1 = x_adv + step_size * torch.sign(grad)
        x_adv_1 = torch.clamp(
            torch.min(torch.max(x_adv_1, image_ - eps), image_ + eps), 0.0, 1.0
        )
        x_adv_1 = torch.clamp(
            torch.min(
                torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), image_ - eps),
                image_ + eps,
            ),
            0.0,
            1.0,
        )

        x_adv = x_adv_1

        if i < n_iter - 1:
            x_adv.requires_grad_()
        grad_adv_input = normalize(x_adv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        grad_adv_input.requires_grad_()
        logits = model(grad_adv_input)
        loss_indiv = loss_fn(logits, metric_range, k_, b) #self.loss_computer(logits, target)

        loss = loss_indiv.sum()

        if i < n_iter - 1:
            grad = torch.autograd.grad(loss, [grad_adv_input])[0].detach()

        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        # check step size
        y1 = loss_indiv.detach().clone()
        y1 = y1.squeeze(-1)
        loss_steps[i] = y1
        ind = (y1 > loss_best).nonzero().squeeze()
        x_best[ind] = x_adv[ind].clone()
        grad_best[ind] = grad[ind].clone()
        loss_best[ind] = y1[ind]
        loss_best_steps[i + 1] = loss_best

        counter3 += 1

        if counter3 == k:
            fl_oscillation = check_oscillation(loss_steps, i, k, thr_decr)
            fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                loss_best_last_check >= loss_best
            ).float()
            fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
            reduced_last_check = fl_oscillation.clone()
            loss_best_last_check = loss_best.clone()

            if fl_oscillation.sum() > 0:
                ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                step_size[ind_fl_osc] /= 2.0

                x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

            counter3 = 0
            k = max(k - size_decr, n_iter_min)

        # print(self.k)
    return x_best