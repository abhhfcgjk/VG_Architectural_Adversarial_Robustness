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
        self.cayley = options['cayley']
        self.cayley2 = options['cayley2']
        self.cayley3 = options['cayley3']
        self.model = DBCNN(path['scnn_root'], options).to(device)
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
        # ic(self.checkpoint['model']['cayley_block6.conv_cayley.alpha'])
        # ic(self.model.state_dict().keys())
        # print(self.model.state_dict().keys())
        print(self.checkpoint['model'].keys())
        # self.checkpoint['model']['']
# 'features1.27.conv_in.weight', 'features1.27.conv_in.bias', 'features1.27.conv_cayley.weight', 'features1.27.conv_cayley.bias', 'features1.27.conv_cayley.alpha'
        
        # self.checkpoint['model']['module.features1.27.conv_cayley.alpha'] = self.checkpoint['model']['module.features1.28.conv_cayley.alpha']
        # self.checkpoint['model']['module.features1.27.conv_cayley.bias'] = self.checkpoint['model']['module.features1.28.conv_cayley.bias']
        # self.checkpoint['model']['module.features1.27.conv_cayley.weight'] = self.checkpoint['model']['module.features1.28.conv_cayley.weight']
        # self.checkpoint['model']['module.features1.27.conv_in.bias'] = self.checkpoint['model']['module.features1.28.conv_in.bias']
        # self.checkpoint['model']['module.features1.27.conv_in.weight'] = self.checkpoint['model']['module.features1.28.conv_in.weight']
        
        # del self.checkpoint['model']['module.features1.28.conv_cayley.alpha']
        # del self.checkpoint['model']['module.features1.28.conv_cayley.bias']
        # del self.checkpoint['model']['module.features1.28.conv_cayley.weight']
        # del self.checkpoint['model']['module.features1.28.conv_in.bias']
        # del self.checkpoint['model']['module.features1.28.conv_in.weight']

        print(self.model.state_dict().keys())
        weights = self.checkpoint['model']
        for key in list(weights.keys()):
            weights[key.replace('module.', '')] = weights[key]
            del weights[key]

        self.model.load_state_dict(weights)
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
        for img_, img in tqdm(
                self.loader,
                total=len(self.loader),
        ):
            clear_val = self.compute_output(img_)
            clear_val = (clear_val - self.min_test)/(self.max_test - self.min_test)

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
                    attacked_val = (attacked_val - self.min_test)/(self.max_test - self.min_test)
                    # ic(clear_val, attacked_val)
                    gain = attacked_val - clear_val
                    self.gains[int(eps * 255)].append(gain)
            self.attacked_vals.append(attacked_val)  # with eps 10/255
            image_num += 1
            
            ###########debug
            if debug:
                count -= 1
                if not count:
                    break
            ##########

    def save_vals_to_file(self, csv_results_dir='.'):
        se_status = "+se" if self.se else ""
        data = pd.DataFrame(columns=['clear', 'attack'])
        if csv_results_dir is not None:
            for i in range(len(self.clear_vals)):
                data.loc[len(data.index)] = [self.clear_vals[i], self.attacked_vals[i]]

            result_path = "results_{}_{}_{}.csv".format(self.activation,
                                                        self.arch + se_status,
                                                        self.iterations)
            csv_path = os.path.join(csv_results_dir, result_path)
            data.to_csv(csv_path)
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

    @property
    def res(self):
        if 'results' not in vars(self).keys():
            raise AttributeError("results doesnt exist. Run self.save_results(self, csv_results_dir).")
        return self.results


class TestLoader(Dataset):
    def __init__(self,  dataset_path, resize,crop=False, resize_size_h=498, resize_size_w=664):
        # self.batch_size = batch_size
        self.crop = crop
        self.resize = resize
        self.resize_size_h, self.resize_size_w = resize_size_h, resize_size_w
        self.dataset_path = dataset_path
        self.imgs_names = [_ for _ in Path(self.dataset_path).iterdir()]

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, index):
        image = Image.open(self.imgs_names[index]).convert("RGB")
        if self.crop:  # crop or not?
            image = crop(image, 0, 0, height=self.resize_size_h, width=self.resize_size_w)
        elif self.resize:  # resize or not?
            image = resize(image, (self.resize_size_h, self.resize_size_w))  #
        img = to_tensor(image).cuda()
        img_ = img
        # img=normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img_, img
    
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='koniq-10k',
                        help='dataset: live|csiq|tid2013|livec|mlive|koniq-10k')
    parser.add_argument('--pruning', dest='pruning', type=float,
                        default=0, help='Pruning percentage.')
    parser.add_argument('--tune_iters', dest='tune_iters', type=int,
                        default=1, help='Iters for tune')
    parser.add_argument('--iters', dest='iters', type=int,
                        default=1, help='PGD attack iters count.')
    parser.add_argument('--cayley', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley2', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley3', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--debug', action='store_true',
                        help='DEBUG')
    args = parser.parse_args()
    options = {
        'fc': True,
        'cayley': args.cayley,
        'cayley2': args.cayley2,
        'cayley3': args.cayley3,
        'backbone': 'VGG-16',
        'model': 'DBCNN',
        'pruning': 0,
        'pruning_type': None,
        'activation': 'relu',
        'gradnorm_regularization': False,
        'resize': False,
        'crop': False,
        'height': None,
        'width': None,
    }
    path = {
        'koniq-10k': os.path.join('dataset', 'KonIQ-10k'),
        'nips': os.path.join('dataset', 'NIPS'),
        'csv_results_dir': './rs',

        'live': os.path.join('dataset','databaserelease2'),
        'csiq': os.path.join('dataset','CSIQ'),
        'tid2013': os.path.join('dataset','TID2013'),
        'livec': os.path.join('dataset','ChallengeDB_release'),
        'mlive': os.path.join('dataset','LIVEmultidistortiondatabase'),
        'fc_model': os.path.join('fc_models'),
        'scnn_root': os.path.join('pretrained_scnn','scnn.pkl'),
        'fc_root': os.path.join('fc_models','net_params_best.pkl'),
        'db_model': os.path.join('db_models'),   
    }

    ic.disable()

    exec_: Attack = Attack(path, options, device='cuda')

    exec_.load_checkpoints(checkpoints_path=f'./DBCNN-cayley={args.cayley}'\
                                            f'-cayley2={args.cayley2}'\
                                            f'-cayley3={args.cayley3}.pt')
    exec_.set_load_conf(dataset_path=path['nips'],
                        resize=options['resize'],
                        crop=options['crop'],
                        resize_size_h=options['height'],
                        resize_size_w=options['width'])

    
    exec_.attack(attack_type='PGD',
                 iterations=args.iters, debug=args.debug)

    exec_.save_results(path['csv_results_dir'])

    print(np.array(exec_.res).mean())
