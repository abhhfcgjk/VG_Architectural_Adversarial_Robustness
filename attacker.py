from models import Net
from args import Configs
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import resize, to_tensor, normalize, crop
from torchvision.utils import save_image
from torchvision import transforms
import data_loader_test
import random
import os

def get_range(model, test_index, config):
    mmin = float('inf')
    mmax = -float('inf')
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
									 std=(0.229, 0.224, 0.225))
    folder_path = {
        'koniq':    config.datapath,
        'nips':     config.datapath
        }
    loader = data_loader_test.DataLoader(config.dataset, folder_path[config.dataset],
                                        test_index, config.patch_size,
                                        config.test_patch_num, istrain=False)
    test_data = loader.get_data()
    model = model.cuda()
    for i, (data, label) in enumerate(test_data):
        # data = data.cuda()
        output = model(normalize(data).cuda())
        print(i, output)
        mmin = min(output[0][0][0], mmin)
        mmax = max(output[0][0][0], mmax)
    print(mmin, mmax, mmax-mmin)
    return mmax-mmin



def attack(model, normalize, inputs, eps, alpha, target, loss_computer, iters):
    # save_image(inputs[0], 'im0.png')
    eps = alpha*1.2
    noise = torch.empty(*inputs.shape, device='cuda')
    noise.uniform_(-eps, eps)

    for _ in range(iters):
        noise_with_grad = noise.detach().requires_grad_().cuda()
        noisy_inputs = inputs.cuda() + noise_with_grad.cuda()
        noisy_inputs.clamp_(0.0, 1.0)

        noisy_outputs = model(normalize(noisy_inputs).cuda())
        loss = loss_computer(noisy_outputs, target)

        grad = torch.autograd.grad(loss, [noise_with_grad])[0].detach()

        noise -= alpha * torch.sign(grad)
        noise.clamp_(-eps, eps)

    attacked = inputs.cuda() + noise.cuda()
    attacked.clamp_(0.0, 1.0)
    # logging.debug(inputs[0].shape)
    # logging.debug(attacked[0].shape)
    return attacked


def run(model, config, iters, eps):
    iters = iters
    eps = eps
    folder_path = {
        'koniq':    config.datapath,
        'nips':     config.datapath,
        }
    img_num = {
        'koniq':    list(range(0, 10073)),
        'nips':     list(range(0, 999)),
        }
    if config.seed == 0:
        pass
    else:
        print('we are using the seed = {}'.format(config.seed))
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
    total_num_images = img_num[config.dataset]
    random.shuffle(total_num_images)
    if config.dataset == 'koniq':
        test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]
    elif config.dataset == 'nips':
        test_index = img_num[config.dataset]
    loader = data_loader_test.DataLoader(config.dataset, folder_path[config.dataset],
                                        test_index, config.patch_size,
                                        config.test_patch_num, istrain=False)
    test_data = loader.get_data()
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
									 std=(0.229, 0.224, 0.225))
    metric_range = 100
    # metric_range = get_range(model, test_index, config)
    loss_ = lambda x,y: 1 - x[0]/metric_range

    result = pd.DataFrame(columns=['origin_preds', 
                                    'preds_eps=2.0', 'preds_eps=4.0', 
                                    'preds_eps=6.0', 'preds_eps=8.0',
                                    'preds_eps=10.0'])

    model.eval()
    for i, (data, label) in tqdm(enumerate(test_data), total=len(test_data)):
        d = {}
        output = model(normalize(data).cuda())
        # print(i,output)
        d['origin_preds'] = output[0][0][0].item()
        for alpha in [2/255, 4/255, 6/255, 8/255, 10/255]:
            attacked = attack(model, normalize, data, eps, alpha, None, loss_, iters)
            output_attacked = model(normalize(attacked).cuda())
            d[f'preds_eps={alpha*255}'] = output_attacked[0][0][0].item()
        # print(d)
        result.loc[len(result)] = d
        # print(result)
    result.to_csv(f'mnt/koniq_{config.vesion}_2021/sv/{config.dataset}_pgd_{iters}.csv', index=False)


if __name__=='__main__':
    config = Configs()
    model = Net(config, 'cuda')
    ckpt = torch.load('mnt/koniq_{}_2021/sv/bestmodel_{}_2021'.format(config.vesion, config.vesion))
    model.load_state_dict(ckpt["model"])
    model.eval()
    attack_pref = {'iters': (1,3,5,8), 'eps': (2.0, 4.0, 6.0, 8.0, 10.0)}

    run(model, config, config.iters, 4/255)
