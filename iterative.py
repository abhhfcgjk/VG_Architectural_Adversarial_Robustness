from torch.autograd import Variable
import torch


# iterative attack baseline (IFGSM attack)
def attack(
    image,
    model=None,
    metric_range=100,
    device="cpu",
    eps=10 / 255,
    iters=10,
    alpha=1 / 255,
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

    # additive = torch.zeros_like(image).to(device)
    additive = torch.rand_like(image).to(device)
    additive = Variable(additive, requires_grad=True)

    loss_fn = lambda score, m_range: 1 - score / m_range

    for _ in range(iters):
        img = Variable(image + additive, requires_grad=True)
        # img = image + additive
        img.data.clamp_(0.0, 1.0)
        output = model(img)

        avr_output = torch.stack(output).mean()
        loss = loss_fn(avr_output, metric_range)

        # print(loss)
        model.zero_grad()
        loss.backward()
        input_grad = img.grad.data

        gradient_sign = input_grad.sign()
        additive.data -= alpha * gradient_sign  # 0.25*eps*gradient_sign
        additive.data.clamp_(-eps, eps)

    res_image = image + additive
    res_image = (res_image).data.clamp_(min=0, max=1)

    return res_image
