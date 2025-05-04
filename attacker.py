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
import data_loader
import random
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

class TestLoader(Dataset):
    def __init__(self, dataset, dataset_path, resize=False, crop=False, resize_size_h=498, resize_size_w=664, **kwargs):
        # self.batch_size = batch_size
        self.dataset = dataset
        self.loader = kwargs.get("loader", default_loader)
        self.dataset_path = dataset_path
        self.resize_size_h, self.resize_size_w = resize_size_h, resize_size_w
        self.resize = resize
        self.crop = crop
        if self.dataset == 'NIPS':
            self.im_names = [path.name for path in Path(self.dataset_path).iterdir()]
        elif self.dataset == 'KonIQ-10k':
            datainfo = kwargs.get("data_info", None)
            assert datainfo
            Info = h5py.File(datainfo, 'r')
            index = Info['index']
            train_ratio = 0.6
            train_and_val_ratio = 0.8
            exp_id = 0
            index = index[:, exp_id % index.shape[1]]
            ref_ids = Info['ref_ids'][0, :]
            status ='test'
            if status == 'train':
                index = index[0:int(train_ratio * len(index))]
            elif status == 'val':
                index = index[int(train_ratio * len(index)):int(train_and_val_ratio * len(index))]
            elif status == 'test':
                index = index[int(train_and_val_ratio * len(index)):len(index)]
            self.index = []
            for i in range(len(ref_ids)):
                if ref_ids[i] in index:
                    self.index.append(i)
            print("# {} images: {}".format(status, len(self.index)))

            self.label = Info['subjective_scores'][0, self.index].astype(np.float32)
            self.label_std = Info['subjective_scoresSTD'][0, self.index].astype(np.float32)
            self.im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        else:
            raise KeyError(f"Dataset {self.dataset} does not exist.")
        self.ims = []
        print("DATA LOADING")
        for im_name in tqdm(self.im_names, total=len(self.im_names)):
            im = self.loader(os.path.join(self.dataset_path, im_name))
            # if resize:  # resize or not?
            #     im = resize(im, (resize_size_h, resize_size_w))  # h, w
            self.ims.append(im)

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        image = self.ims[idx]
        img = self.transform(image)
        img_name = self.im_names[idx]
        return img_name, img
    
    def transform(self, image):
        if self.crop:  # crop or not?
            image = crop(image, 0, 0, height=self.resize_size_h, width=self.resize_size_w)
        elif self.resize:  # resize or not?
            image = resize(image, (self.resize_size_h, self.resize_size_w))  #
        img = to_tensor(image).cuda()
        return img


def get_range(model, test_index, config):
    mmin = float('inf')
    mmax = -float('inf')
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
									 std=(0.229, 0.224, 0.225))
    folder_path = {
        'live':     config.datapath,
        'csiq':     config.datapath,
        'tid2013':  config.datapath,
        'kadid10k': config.datapath,
        'clive':    config.datapath,
        'koniq':    config.datapath,
        'nips':    config.datapath,
        'fblive':   config.datapath,
        }
    loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
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

def run(model, config):
    iters = 5
    eps = 10./255
    folder_path = {
        'live':     config.datapath,
        'csiq':     config.datapath,
        'tid2013':  config.datapath,
        'kadid10k': config.datapath,
        'clive':    config.datapath,
        'koniq':    config.datapath,
        'nips':    config.datapath,
        'fblive':   config.datapath,
        }

    img_num = {
        'live':     list(range(0, 29)),
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'clive':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        'nips':    list(range(0, 999)),
        'fblive':   list(range(0, 39810)),
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
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]
    test_index = img_num['nips']
    loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
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
    print(len(test_data))
    for i, (data, label) in enumerate(test_data):
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
        print(result)
    result.to_csv(f'results_{iters}.csv', index=False)



def attack(model, normalize, inputs, eps, alpha, target, loss_computer, iters):
    save_image(inputs[0], 'im0.png')
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
    print(inputs[0].shape, attacked[0].shape)
    
    save_image(attacked[0], 'im.png')
    return attacked



if __name__=='__main__':
    config = Configs()
    model = Net(config, 'cuda')
    ckpt = torch.load('mnt/koniq_1_2021/sv/bestmodel_1_2021')
    model.load_state_dict(ckpt)
    model.eval()
    run(model, config)
