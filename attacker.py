import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import resize, to_tensor, normalize, crop
import iterative
from models_train.IQAmodel import IQAModel
from models_train.Linearity import Linearity

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
import h5py
from random import random
# from LinearityIQA.activ import ReLU_to_SILU, ReLU_to_ReLUSiLU
# from clearml import Task, Logger
from icecream import ic


class Attack:
    epsilons = np.array([2, 4, 6, 8, 10]) / 255.0

    def __init__(self, model, arch, pool, use_bn_end, P6, P7, pruning,t_prune, activation, device='cpu',
                 gabor=False, gradnorm_regularization=False, cayley=False, cayley_pool=False, cayley_pair=False) -> None:
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device
        self.arch = arch
        # if arch=='resnet101':
        #     arch = 'resnext101_32x8d'
        self.model_name = model
        self.prune = pruning
        self.prune_method = t_prune
        self.activation = activation
        self.gradnorm_regularization = gradnorm_regularization
        self.cayley = cayley
        self.cayley_pool = cayley_pool
        self.cayley_pair = cayley_pair
        self.to_save_images = 700
        # self.prune 
        self.model = Linearity(arch='resnext101_32x8d' if arch=='apgd_ssim'or arch=='apgd_ssim_eps2' or arch=='free_ssim_eps2' else arch, pool=pool,
                           use_bn_end=use_bn_end,
                           P6=P6, P7=P7, activation=activation, pruning=None, gabor=gabor, 
                           cayley=cayley, cayley_pool=cayley_pool, cayley_pair=cayley_pair).to(self.device)
        ic(self.model)
        self.model.eval()
        cols = ['image_name', 'clear_val', 'attacked_eps=']
        # self.df_attack_csv = pd.DataFrame(columns=['image_name', 'clear_val'] + \
        #                                 [f'attacked_val_eps={int(val*255.0)}' for val in self.epsilons])

    def compute_output(self, x):
        # im = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # if self.model_name == "Linearity":
        return self.model(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[-1].item() * self.k[0] + self.b[0]
            # return self.model(x)[-1].item() * self.k[0] + self.b[0]
        # elif self.model_name == "KonCept":
        #     return self.model(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])).item()
            # raise NotImplementedError()
        # raise NameError(f"No model {self.model_name}.")

    def load_checkpoints(self, checkpoints_path="LinearityIQA/checkpoints/p1q2.pth"):
        self.checkpoint = torch.load(checkpoints_path, map_location=self.device)
        # ic(self.checkpoint['model']['cayley_block6.conv_cayley.alpha'])
        ic(self.model.state_dict().keys())
        self.model.load_state_dict(self.checkpoint["model"])
        self.k = self.checkpoint['k']
        self.b = self.checkpoint['b']
        self.min_train = self.checkpoint['min']
        self.max_train = self.checkpoint['max']
        self.dataset_path = '.'
        self.metric_range_train = self.checkpoint['max'] - self.checkpoint['min']
        print(self.checkpoint.keys())
        self.min_test = self.min_train
        self.max_test = self.max_train
        self.metric_range_test = self.metric_range_train

    def set_load_conf(self, dataset, dataset_path, resize, crop, resize_size_h, resize_size_w, batch_size=4, data_info=None):
        self.df_attack_csv = pd.DataFrame(columns=['image_name', 'clear_val'] + \
                                        [f'attacked_val_eps={int(val*255.0)}' for val in self.epsilons])
        self.dataset = dataset
        self.dataset_path = dataset_path
        if self.dataset_path is None:
            self.dataset_path = f"./{self.dataset}/"
        self.resize = resize
        self.crop = crop
        self.resize_size_h: int = resize_size_h
        self.resize_size_w: int = resize_size_w

        self.loader = DataLoader(
                            TestLoader(
                                dataset=self.dataset,
                                dataset_path=self.dataset_path, 
                                resize=self.resize, 
                                crop=self.crop, 
                                resize_size_h=self.resize_size_h, 
                                resize_size_w=self.resize_size_w,
                                data_info=data_info
                                ),
                            batch_size=1
                            )

    def _get_info_max_min_from_testset(self, debug=False):
        self.clear_vals = []
        count = 5
        for img_name, img_ in tqdm(
                self.loader,
                total=len(self.loader),
        ):
            # if Path(image_path).suffix not in [".png", ".jpg", ".jpeg"]:
            #     continue
            # # diffs = {int(x * 255): [] for x in self.epsilons}
            # img = Image.open(image_path).convert("RGB")
            # if self.resize:  # resize or not?
            #     img = resize(img, (self.resize_size_h, self.resize_size_w))  #
            # img = to_tensor(img).to(self.device)

            # img = img.unsqueeze(0)

            with torch.no_grad():
                clear_val = self.compute_output(img_)
                ic('OUT: ', clear_val)
                self.clear_vals.append(clear_val)

            ###########debug
            if debug:
                print('min: ',min(self.clear_vals),'max: ', max(self.clear_vals))
                count -= 1
                if not count:
                    break
            ##########
        self.min_test, self.max_test = min(self.clear_vals), max(self.clear_vals)
        self.metric_range_test = self.max_test - self.min_test


        # self.min_test, self.max_test = self.min_train, self.max_train
        # self.metric_range_test = self.metric_range_train


        print("Range: ", self.min_test, self.max_test)

    def attack(self, attack_type="IFGSM", iterations=1, debug=False):
        self.attack_type = attack_type
        self.iterations = iterations
        self.attack_type = attack_type
        
        if self.to_save_images is not None:
            debug_dir = 'debug_img'
            os.makedirs(debug_dir, exist_ok=True)
        count = 5
        self.gains = {int(x * 255): [] for x in self.epsilons}
        print("ATTACK")
        for image_num, (img_name, img_) in tqdm(
                enumerate(self.loader),
                total=len(self.loader),
        ):
            img_name = img_name[0]

            clear_val = self.compute_output(img_)
            # clear_val = self.min_max_scale(clear_val) # (clear_val - self.min_test)/(self.max_test - self.min_test)
            attacked_vals = {int(x * 255): None for x in self.epsilons}

            for _, eps in enumerate(self.epsilons):
                img_attacked_ = iterative.attack_callback(
                    img_, model=self.model, attack_type=attack_type, metric_range=self.metric_range_test,
                    device=self.device,
                    eps=1.0, iters=iterations, delta=eps, k=self.k, b=self.b
                )
                # img_attacked = normalize(img_attacked_,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                
                with torch.no_grad():
                    # save_image(img_, f'debug_img/clear{image_num}.png')
                    # save_image(img_attacked_, f'debug_img/{image_num}_{int(eps*255)}.png')
                    # diff = img_attacked - img
                    # diff = torch.clamp(diff, min=-10 / 255, max=10 / 255)
                    # # print("DIFF:", torch.sum(diff))
                    # if self.to_save_images is not None and image_num % self.to_save_images==0:
                    #     save_image(img_attacked_, f'{debug_dir}/{image_num}_{int(eps*255)}.png')

                    attacked_val = self.compute_output(img_attacked_)
                    ic(attacked_val)
                    # attacked_val = self.min_max_scale(attacked_val) # (attacked_val - self.min_test)/(self.max_test - self.min_test)
                    ic(clear_val, attacked_val)
                    gain = attacked_val - clear_val

                    self.gains[int(eps * 255)].append(gain)
                    attacked_vals[int(eps * 255)] = attacked_val
            # image_num += 1
        
            ###########debug
            if debug:
                # print(self.attacked_vals, self.clear_vals)
                # print(self.attacked_vals[image_num - 1], self.clear_vals[image_num - 1])
                count -= 1
                if not count:
                    break
            ##########
            if self.to_save_images is not None and image_num % self.to_save_images==0:
                save_image(img_, f'{debug_dir}/clear{image_num}.png')
                save_image(img_attacked_, f'{debug_dir}/attacked{image_num}_{int(self.epsilons[-1]*255)}.png')
                # Logger.current_logger().report_media(
                #     title=self.dataset,
                #     series="clear",
                #     iteration=image_num,
                #     local_path=os.path.join(debug_dir, f"clear{image_num}.png")
                # )
                # Logger.current_logger().report_media(
                #     title=self.dataset,
                #     series="max perturbation",
                #     iteration=image_num,
                #     local_path=os.path.join(debug_dir, f"attacked{image_num}_{int(self.epsilons[-1]*255)}.png")
                # )
                
            frame = {
                        'image_name': img_name,
                        'clear_val': clear_val
                    } | {
                        f'attacked_val_eps={int(eps*255.0)}': attacked_vals[int(eps * 255)] for eps in self.epsilons
                    }
            self.df_attack_csv.loc[len(self.df_attack_csv)] = frame
        
        gain_graph = [[key, np.array(values).mean()] for key, values in self.gains.items()]
        gain_graph = np.array(gain_graph)
        # Logger.current_logger().report_scatter2d(
        #     title=self.arch,
        #     series=self.dataset,
        #     iteration=0,
        #     scatter=gain_graph,
        #     xaxis='eps',
        #     yaxis='gain',
        #     mode='lines+markers'
        # )

    def save_vals_to_file(self, csv_results_dir='.'):
        # data = pd.DataFrame(columns=['clear', 'attack'])

        assert csv_results_dir
        cl = f'+cayley' if self.cayley else ''
        clp = f'+cayley_pool' if self.cayley_pool else ''
        cp = f'++cayley_pair' if self.cayley_pair else ''
        gr = f'+gr' if self.gradnorm_regularization else ''
        if self.resize:
            resize_flag = '+resize={}x{}'.format(self.resize_size_h, self.resize_size_w) if self.resize else ''
        prune = f"+{self.prune}_{self.prune_method}" if self.prune else ''
        activation =  self.activation
        arch_status = f'{self.arch}{cl}{clp}{cp}{gr}{prune}+{activation}'
        result_path = "{}_{}_{}={}{}.csv".format(
                                            self.dataset,
                                            arch_status,
                                            self.attack_type,
                                            self.iterations,
                                            resize_flag
                                            )
        csv_path = os.path.join(csv_results_dir, result_path)
        self.df_attack_csv.to_csv(csv_path)
        print(f"Results saved to {csv_path}")
        # Task.current_task().register_artifact(
        #     name=result_path,
        #     artifact=self.df_attack_csv,
        #     metadata={
        #         'Arch': self.arch, 
        #         'Cayley': cl,
        #         'Cayley pool': clp,
        #         'Cayley pair': cp,
        #         'Gradnorm regularization': gr,
        #         'Activation': activation,
        #         'Dataset': self.dataset, 
        #         'PGD': self.iterations
        #         }
        # )

    def save_results(self, csv_results_dir='.'):
        self.results = []
        degree = 0
        prune_status = f"+prune={self.prune}{self.prune_method}" if self.prune is not None and self.prune > 0 else ""
        cl = f'+cayley' if self.cayley else ''
        clp = f'+cayley_pool' if self.cayley_pool else ''
        cp = f'+cayley_pair' if self.cayley_pair else ''
        gr = f'+gr' if self.gradnorm_regularization else ''
        mdif = {'arch': self.arch + '-' + self.model_name + prune_status + gr + cl + clp + cp,
                'activation': self.activation,
                'attack': self.attack_type,
                'iterations': self.iterations}
        for int_eps in self.gains.keys():

            self.results.append(np.array(self.gains[int_eps]).mean())
            print(
                f"eps={int_eps}/255 :",
                "mean diff = {:.5f}".format(np.array(self.gains[int_eps]).mean()),
            )
            mdif.update({f'eps {int_eps}': self.results[-1] * (10 ** degree)})

        # correlation = stats.spearmanr(self.clear_vals, self.attacked_vals)
        mdif.update({'SROCC': self.checkpoint['SROCC']})
        print('SROCC:', self.checkpoint['SROCC'])
        mdif.update({'PLCC': self.checkpoint['PLCC']})
        print('PLCC:', self.checkpoint['PLCC'])

        cols = [f'eps {e}' for e in self.gains.keys()]
        cols = ['arch', 'activation', 'attack', 'iterations'] + cols + ['SROCC', 'PLCC']
        if csv_results_dir is not None:
            csv_path = os.path.join(csv_results_dir, "results.csv".format(self.activation))
            if "results.csv" not in os.listdir(csv_results_dir):
                tmp = pd.DataFrame(columns=cols)
                tmp.style.hide(axis='index')
                print(len(tmp.index))
                tmp.loc[len(tmp.index)] = mdif
                tmp.to_csv(csv_path)
            else:
                df = pd.read_csv(csv_path, usecols=[_ for _ in range(1, len(cols) + 1)])
                df.loc[len(df)] = mdif
                df.to_csv(csv_path)

    def min_max_scale(self, value):
        return (value - self.min_test)/(self.max_test - self.min_test)

    @property
    def res(self):
        if 'results' not in vars(self).keys():
            return None
            # raise AttributeError("results doesnt exist. Run self.save_results(self, csv_results_dir).")
        return self.results

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
        image = self.ims[idx] # self.loader(self.imgs_names[idx]) # Image.open(self.imgs_names[index]).convert("RGB")
        # if self.crop:  # crop or not?
        #     image = crop(image, 0, 0, height=self.resize_size_h, width=self.resize_size_w)
        # elif self.resize:  # resize or not?
        #     image = resize(image, (self.resize_size_h, self.resize_size_w))  #
        # img = to_tensor(image).cuda()
        img = self.transform(image)
        img_name = self.im_names[idx]
        # img=normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img_name, img
    
    def transform(self, image):
        if self.crop:  # crop or not?
            image = crop(image, 0, 0, height=self.resize_size_h, width=self.resize_size_w)
        elif self.resize:  # resize or not?
            image = resize(image, (self.resize_size_h, self.resize_size_w))  #
        img = to_tensor(image).cuda()
        return img