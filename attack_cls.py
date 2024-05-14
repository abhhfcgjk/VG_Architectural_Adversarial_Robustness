import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import resize, to_tensor, normalize
import iterative
from models_train.IQAmodel import IQAModel

# from LinearityIQA.activ import ReLU_to_SILU, ReLU_to_ReLUSiLU


class Attack:
    epsilons = np.array([2, 4, 6, 8, 10]) / 255.0

    def __init__(self, model, arch, pool, use_bn_end, P6, P7, se, pruning, activation, device='cpu') -> None:
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device
        self.arch = arch
        self.se = se
        self.model_name = model
        self.prune = pruning
        self.activation = activation
        # self.prune 
        self.model = IQAModel(model,arch=arch, pool=pool,
                           use_bn_end=use_bn_end,
                           P6=P6, P7=P7, activation=activation, se=se, pruning=None).to(self.device)
        self.model.eval()
        # print(self.model)

    def compute_output(self, x):
        if self.model_name == "Linearity":
            return self.model(x)[-1].item() * self.k[0] + self.b[0]
        elif self.model_name == "KonCept":
            return self.model(x).item()
        raise NameError(f"No model {self.model_name}.")

    def load_checkpoints(self, checkpoints_path="LinearityIQA/checkpoints/p1q2.pth"):
        self.checkpoint = torch.load(checkpoints_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint["model"])
        self.k = self.checkpoint['k']
        self.b = self.checkpoint['b']
        # self.min_train = self.checkpoint['min']
        # self.max_train = self.checkpoint['max']
        self.dataset_path = '.'
        self.resize = False
        # self.metric_range_train = self.checkpoint['max'] - self.checkpoint['min']
        print(self.checkpoint.keys())

    def set_load_conf(self, dataset_path, resize_size_h, resize_size_w):
        self.dataset_path = dataset_path
        self.resize = True
        self.resize_size_h: int = resize_size_h
        self.resize_size_w: int = resize_size_w

    def _get_info_max_min_from_testset(self, debug=False):
        self.clear_vals = []
        count = 5
        for image_path in tqdm(
                Path(self.dataset_path).iterdir(),
                total=len([x for x in Path(self.dataset_path).iterdir()]),
        ):
            if Path(image_path).suffix not in [".png", ".jpg", ".jpeg"]:
                continue
            # diffs = {int(x * 255): [] for x in self.epsilons}
            im = Image.open(image_path).convert("RGB")
            if self.resize:  # resize or not?
                im = resize(im, (self.resize_size_h, self.resize_size_w))  #
            im = to_tensor(im).to(self.device)

            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im = im.unsqueeze(0)

            with torch.no_grad():
                clear_val = self.compute_output(im)
                # clear_val = clear_val[-1].item() * self.k[0] + self.b[0]
                self.clear_vals.append(clear_val)

            ###########debug
            if debug:
                print(min(self.clear_vals), max(self.clear_vals))
                count -= 1
                if not count:
                    break
            ##########
        self.min_test, self.max_test = min(self.clear_vals), max(self.clear_vals)
        self.metric_range_test = self.max_test - self.min_test

    def attack(self, attack_type="IFGSM", iterations=1, debug=False):
        self._get_info_max_min_from_testset(debug)
        self.attack_type = attack_type
        self.iterations = iterations
        self.attacked_vals = []
        image_num = 0
        count = 5
        self.gains = {int(x * 255): [] for x in self.epsilons}
        for image_path in tqdm(
                Path(self.dataset_path).iterdir(),
                total=len([x for x in Path(self.dataset_path).iterdir()]),
        ):
            if Path(image_path).suffix not in [".png", ".jpg", ".jpeg"]:
                continue

            im = Image.open(image_path).convert("RGB")
            if self.resize:  # resize or not?
                im = resize(im, (self.resize_size_h, self.resize_size_w))  #
            im = to_tensor(im).to(self.device)

            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im = im.unsqueeze(0)
            # print(clear_vals[image_num], min_pred, max_pred)
            clear_val = iterative.norm(self.clear_vals[image_num],
                                       mmin=self.min_test, mmax=self.max_test)

            self.clear_vals[image_num] = clear_val

            for _, eps in enumerate(self.epsilons):
                im_attacked = iterative.attack_callback(
                    ###############
                    im, model=self.model, attack_type=attack_type, metric_range=self.metric_range_test,
                    device=self.device,
                    #################
                    eps=10 / 255, iters=iterations, alpha=eps, k=self.k, b=self.b,
                    mmin=self.min_test, mmax=self.max_test
                )

                with torch.no_grad():
                    diff = im_attacked - im
                    diff = torch.clamp(diff, min=-10 / 255, max=10 / 255)
                    # print("DIFF:", torch.sum(diff))
                    im_attacked = im + diff

                    attacked_val = self.compute_output(im_attacked)
                    # attacked_val = attacked_val[-1].item() * self.k[0] + self.b[0]
                    attacked_val = iterative.norm(attacked_val, self.min_test, self.max_test)
                    gain = attacked_val - clear_val
                    self.gains[int(eps * 255)].append(gain)
            self.attacked_vals.append(attacked_val)  # with eps 10/255
            image_num += 1

            ###########debug
            if debug:
                # print(self.attacked_vals, self.clear_vals)
                print(self.attacked_vals[image_num - 1], self.clear_vals[image_num - 1])
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
        se_status = "+se" if self.se else ""
        prune_status = f"+prune={self.prune}" if self.prune is not None and self.prune > 0 else ""
        mdif = {'arch': self.arch + '-' + self.model_name + se_status + prune_status,
                'activation': self.activation,
                'attack': self.attack_type,
                'iterations': self.iterations,
                'degree': f'10^{degree}'}
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

        cols = [f'eps {e}' for e in self.gains.keys()]
        cols = ['arch', 'activation', 'attack', 'iterations', 'degree'] + cols + ['SROCC']
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
