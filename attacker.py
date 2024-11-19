import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import resize, to_tensor, normalize, crop
from attack import attack_callback
# from models_train.IQAmodel import IQAModel
from argparse import ArgumentParser
from DBCNN import DBCNN
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image
from random import random
import csv
# from LinearityIQA.activ import ReLU_to_SILU, ReLU_to_ReLUSiLU
from icecream import ic


class Attack:
    epsilons = np.array([2, 4, 6, 8, 10]) / 255.0

    def __init__(self, path, options, device='cpu') -> None:
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device
        self.arch = options['backbone']
        # if arch=='resnet101':
        #     arch = 'resnext101_32x8d'
        self.model_name = options['model']
        self.prune = options['pruning']
        self.prune_method = options['pruning_type']
        self.activation = options['activation']
        self.gradnorm_regularization = options['gradnorm_regularization']
        self.cayley = options.get('cayley', None)
        self.cayley2 = options.get('cayley2', None)
        self.cayley3 = options.get('cayley3', None)
        self.cayley4 = options.get('cayley4', None)
        self.model = torch.nn.Parallel(DBCNN(path['scnn_root'], options)).to(device)
        print(self.model)
        # print(self.model)

    def compute_output(self, x):
        # im = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.model_name == "Linearity":
            return self.model(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[-1].item() * self.k[0] + self.b[0]
            # return self.model(x)[-1].item() * self.k[0] + self.b[0]
        elif self.model_name == "KonCept":
            return self.model(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])).item()
            # raise NotImplementedError()
        elif self.model_name == "DBCNN":
            return self.model(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])).item()
        raise NameError(f"No model {self.model_name}.")

    def load_checkpoints(self, checkpoints_path="./DBCNN.pt"):
        self.checkpoint = torch.load(checkpoints_path, map_location=self.device)

        self.model.load_state_dict(self.checkpoint['model'])
        self.min_train =  self.checkpoint['min']
        self.max_train =  self.checkpoint['max']
        self.dataset_path = '.'
        self.metric_range_train = self.checkpoint['max'] - self.checkpoint['min']
        print(self.checkpoint.keys())
        self.min_test = self.min_train
        self.max_test = self.max_train
        self.metric_range_test = self.metric_range_train

    def set_load_conf(self, dataset_path, resize, crop, resize_size_h, resize_size_w, batch_size=4):
        self.dataset_path = dataset_path
        self.resize = resize
        self.crop = crop
        self.resize_size_h: int = resize_size_h
        self.resize_size_w: int = resize_size_w
        self.loader = DataLoader(TestLoader(self.dataset_path, self.resize, self.crop, self.resize_size_h, self.resize_size_w),
                                 batch_size=1)

    def _get_info_max_min_from_testset(self, debug=False):
        self.clear_vals = []
        count = 5
        for img_, img in tqdm(
                self.loader,
                total=len(self.loader),
        ):
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

        print("Range: ", self.min_test, self.max_test)

    def attack(self, attack_type="IFGSM", iterations=1, debug=False):
        # self._get_info_max_min_from_testset(debug)
        self.attack_type = attack_type
        self.iterations = iterations
        self.attacked_vals = []
        image_num = 0
        count = 5
        self.gains = {int(x * 255): [] for x in self.epsilons}
        for image_num, (img_name, img_) in tqdm(
                enumerate(self.loader),
                total=len(self.loader),
        ):
            img_name = img_name[0]
            clear_val = self.compute_output(img_)
            # clear_val = (clear_val - self.min_test)/(self.max_test - self.min_test)
            attacked_vals = {int(x * 255): None for x in self.epsilons}

            for _, eps in enumerate(self.epsilons):
                img_attacked_ = attack_callback(
                    ###############
                    img_, model=self.model, attack_type=attack_type, metric_range=self.metric_range_test,
                    device=self.device,
                    #################
                    eps=1.0, iters=iterations, delta=eps
                )
                # img_attacked = normalize(img_attacked_,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                
                with torch.no_grad():
                    attacked_val = self.compute_output(img_attacked_)
                    # attacked_val = attacked_val[-1].item() * self.k[0] + self.b[0]
                    # ic(attacked_val)

                    # attacked_val = (attacked_val - self.min_test)/(self.max_test - self.min_test)
                    # ic(clear_val, attacked_val)
                    gain = attacked_val - clear_val

                    self.gains[int(eps * 255)].append(gain)
                    attacked_vals[int(eps * 255)] = attacked_val
            self.attacked_vals.append(attacked_val)  # with eps 10/255

                
            frame = {
                        'image_name': img_name,
                        'clear_val': clear_val
                    } | {
                        f'attacked_val_eps={int(eps*255.0)}': attacked_vals[int(eps * 255)] for eps in self.epsilons
                    }
            self.df_attack_csv.loc[len(self.df_attack_csv)] = frame

            image_num += 1            
            ###########debug
            if debug:
                count -= 1
                if not count:
                    break
            ##########

    # def save_vals_to_file(self, csv_results_dir='.'):
    #     se_status = "+se" if self.se else ""
    #     data = pd.DataFrame(columns=['clear', 'attack'])
    #     if csv_results_dir is not None:
    #         for i in range(len(self.clear_vals)):
    #             data.loc[len(data.index)] = [self.clear_vals[i], self.attacked_vals[i]]

    #         result_path = "results_{}_{}_{}.csv".format(self.activation,
    #                                                     self.arch + se_status,
    #                                                     self.iterations)
    #         csv_path = os.path.join(csv_results_dir, result_path)
    #         data.to_csv(csv_path)
    #         print(f"Results saved to {csv_path}")
    def save_vals_to_file(self, csv_results_dir='.'):
        # data = pd.DataFrame(columns=['clear', 'attack'])

        assert csv_results_dir
        cl = f'+cayley' if self.cayley else ''
        clp = f'+cayley_pool' if self.cayley_pool else ''
        cp = f'++cayley_pair' if self.cayley_pair else ''
        gr = f'+gr' if self.gradnorm_regularization else ''
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


    def save_results(self, csv_results_dir='.'):
        self.results = []
        degree = 0
        prune_status = f"+prune={self.prune}{self.prune_method}" if self.prune is not None and self.prune > 0 else ""
        cl = f'+cayley' if self.cayley else ''
        # clp = f'+cayley_pool' if self.cayley_pool else ''
        cp = f'+cayley2' if self.cayley2 else ''
        cbp = f'+cayley3' if self.cayley3 else ''
        gr = f'+gr' if self.gradnorm_regularization else ''
        mdif = {'arch': self.arch + '-' + self.model_name + prune_status + gr + cl + cp + cbp,
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
        mdif.update({'SROCC': self.checkpoint['SRCC']})
        print('SROCC:', self.checkpoint['SRCC'])

        cols = [f'eps {e}' for e in self.gains.keys()]
        cols = ['arch', 'activation', 'attack', 'iterations'] + cols + ['SROCC']
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
                # df.drop(labels='Unnamed: 0',axis='columns')

                df.to_csv(csv_path)

    def min_max_scale(self, value):
        return (value - self.min_test)/(self.max_test - self.min_test)

    @property
    def res(self):
        if 'results' not in vars(self).keys():
            raise AttributeError("results doesnt exist. Run self.save_results(self, csv_results_dir).")
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
            # Info = h5py.File(datainfo, 'r')
            self.label = []
            with open(datainfo, 'r') as f:
                Info = csv.DictReader(f)
                for row in Info:
                    self.im_names.append(row['image_name'])
                    mos = float(row['MOS'])
                    mos = np.array(mos)
                    mos = mos.astype(np.float32)
                    self.label.append(mos)
            index = list(range(0,10073))
            status ='test'
            if status == 'train':
                self.index = index[0:int(0.8 * len(index))]
            elif status == 'val':
                pass
            elif status == 'test':
                self.index = index[int(0.8 * len(index)):len(index)]
            print("# {} images: {}".format(status, len(self.index)))
        else:
            raise KeyError(f"Dataset {self.dataset} does not exist.")
        self.ims = []
        print("DATA LOADING")
        for im_name in tqdm(self.im_names, total=len(self.im_names)):
            im = self.loader(os.path.join(self.dataset_path, im_name))
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

