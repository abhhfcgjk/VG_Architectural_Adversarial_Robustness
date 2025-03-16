from datetime import datetime
import os
from pathlib import Path
import re
import time
import shutil
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn.utils import prune
from torch.nn import ReLU, SiLU, ELU, GELU
from tqdm import tqdm
import wandb

from attacks.fgsm import FGSM
from attacks.pgd import PGD, AutoPGD
from attacks.base import Attacker
from attacks.uap import UAP
from attacks.korhonen import Korhonen
from attacks.zhang import Zhang
from lr_scheduler import SimpleLRScheduler
from activ import ReLU_ELU, ReLU_SiLU, swap_all_activations
from model import TReS, normalize_model
from datasets.koniq10k_dali import get_data_loaders
from datasets.nips_dali import get_data_loader
from metrics import IQAPerformance, dump_scalar_metrics
from utils import load_config

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["KMP_WARNINGS"] = "off"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class AdversarialTrainer:
    def __init__(self, gpu: int, config_path: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.db_model_path = Path(self.config["db_model"])
        self.results_csv = self.config["attack"]["path"]["csv_name"]
        self.eval_only = self.config["eval_only"]
        self.use_mask = (
                        (self.config['options']['prune'] > 0) 
                        and (not self.config['eval_only'])
                        )
        self.is_gradnorm_regularization = self.config['train']['gr']
        self.h_gradnorm_regularization = 0.01
        self.weight_gradnorm_regularization = 0.001
        self.is_adv = True if self.config['attack']['train']['type'] != "none" else False

        self.nhead = self.config['train']['nhead']
        self.num_encoder_layers = self.config['train']['num_encoder_layers']
        self.dim_feedforward = self.config['train']['dim_feedforward']
        self.num_patches = self.config['train']['num_patches']

        self.world_size = torch.cuda.device_count()
        self.distributed = self.world_size > 1
        self.gpu = gpu
        print(f"I am gpu #{self.gpu}")

        if self.distributed:
            self.setup_distributed()

        self.model = normalize_model(
            TReS(self.nhead, self.num_encoder_layers, self.dim_feedforward, 
                 db_model=self.config['db_model'], **self.config['options']),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.model.to(self.gpu)
        self._init_logger()

    def _init_logger(self):
        if self.gpu == 0:
            self.wandb_writer = wandb.init(
                    entity="igor_-m-v-lomonosovmoscow-state-university",
                    project="TReS",
                    config=self.config
                )

            print(f"=> Logging in {self.wandb_writer}")
            self.start_training_time = time.time()

    def train(self) -> None:
        self._prepare_for_training()
        self._train_loop()

    def _prepare_for_training(self) -> None:
        self.current_epoch = 0
        self.end_epoch = self.config["train"]["epochs"]
        self.lr = float(self.config["train"]["lr"])
        self._prepair_prune()
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = get_data_loaders(
            rank=self.gpu,
            num_tasks=self.world_size,
            args=self.config["data"],
            batch_size=self.config["train"]["batch_size"],
            num_patches=self.config["train"]["num_patches"],
            patch_size=(self.config["train"]["patch_size"],self.config["train"]["patch_size"]),
            num_workers=self.config["train"]["num_workers"],
            seed=self.config["seed"],
        )

        self._init_optimizer(lr=self.lr)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu]
            )
        self.scaler = GradScaler()

        self._init_lr_scheduler()
        self.loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.clsloss =  nn.CrossEntropyLoss()
        self.train_attack = self._init_attack(
            self.config["attack"]["train"], "train"
        )

        if self.gpu == 0:
            self.val_criterion = self.config["train"]["val_criterion"]
            self.metric_computer = IQAPerformance(self.num_patches)
            self.best_val_criterion, self.best_epoch = -100, -1

    def _init_optimizer(self, lr=0.00005) -> None:
        self.optimizer = Adam(self.model.model.parameters(), lr=lr)

    def _init_lr_scheduler(self) -> None:
        if self.config['lr_scheduler']['type'] == 'simple':
            self.lr_scheduler = SimpleLRScheduler(
                self.optimizer, self.config["lr_scheduler"]["points"]
            )
        elif self.config['lr_scheduler']['type'] == 'step':
            lr_decay_step = int(
                self.end_epoch
                / (
                    1
                    + np.log(self.config['lr_scheduler']['overall_lr_decay'])
                    / np.log(self.config['lr_scheduler']['lr_decay'])
                )
            )
            self.lr_scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_decay_step,
                gamma=self.config['lr_scheduler']['lr_decay'],
            )

    def _init_attack(
        self,
        attack_config: dict[str, Any],
        phase: str,
        metric_range: Optional[Tuple[float, float]] = None,
        *args, **kwargs
    ) -> Attacker:
        attack_name = attack_config["type"]
        if attack_name == "none":
            return None

        if self.is_adv:
            # Load pretrained model for adversarial training
            path = self.config['db_model'].replace('-adv', '')
            ckpt = torch.load(path)['model']
            self.model.load_state_dict(ckpt)

        attackers = {"fgsm": FGSM, "pgd": PGD, "apgd": AutoPGD, 
                     "uap": UAP, "korhonen": Korhonen, "zhang": Zhang}
        attacker_cls = attackers.get(attack_name)

        if attacker_cls is None:
            raise RuntimeError(f"Unknown attack `{attack_name}`")

        if metric_range:
            def loss_computer(y, target):
                return -torch.sum(1 - y / metric_range)
        else:
            def loss_computer(y, target):
                return self.loss(y, target.unsqueeze(1))

        attacker = attacker_cls(
            model=self.model,
            loss_computer=loss_computer,
            **attack_config["params"],
        )

        if attack_name.upper() == 'UAP':
            dataloader = kwargs.get('dataloader')
            attacker.generate(dataloader)
        return attacker

    def _train_loop(self) -> None:
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.end_epoch:
            self.pred_scores = []
            self.gt_scores = []
            self.model.train()
            if self.config['lr_scheduler']['type'] == 'simple':
                self.lr_scheduler.adjust_learning_rate(self.current_epoch)

            done_steps = self.current_epoch * train_data_len
            batch_start_time = time.time()
            print(train_data_len)
            for step, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                metrics = self._train_step(data, step, batch_start_time)
                if self.gpu == 0:
                    dump_scalar_metrics(
                        metrics, self.wandb_writer, "train", global_step=done_steps + step,
                    )
                batch_start_time = time.time()

            if self.config['lr_scheduler']['type'] == 'step':
                self.lr_scheduler.step()
            self.current_epoch += 1

            if self.gpu:
                continue

            self.metric_computer.reset()
            self.model.eval()
            for step, data in enumerate(self.val_loader):
                self._val_step(data)

            val_criterion = self.metric_computer.plcc

            # Do no save model 3 first epochs, because obviously their srcc will be better 
            if val_criterion >= self.best_val_criterion and self.current_epoch > 0:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                }
                torch.save(checkpoint, self.db_model_path)

                self.best_val_criterion = val_criterion
                self.best_epoch = self.current_epoch
                print(
                    f'Save current best model @best_val_criterion ({self.val_criterion}):\
                            {self.best_val_criterion:.3f} @epoch: {self.best_epoch}'
                )
            else:
                print(
                    f'Model is not updated @val_criterion ({self.val_criterion}):\
                            {val_criterion:.3f} @epoch: {self.current_epoch}'
                )

        if self.gpu:
            return

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
        }
        self.metric_computer = IQAPerformance(self.num_patches)

        self.model.eval()
        self.metric_computer.reset()
        self.pred_scores = []
        self.gt_scores = []
        for step, data in enumerate(self.train_loader):
            self._val_step(data)

        preds = self.metric_computer.preds
        checkpoint['max'] = np.max(preds)
        checkpoint['min'] = np.min(preds)

        # Update min and max metric scores for the best model
        checkpoint = torch.load(self.db_model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.metric_computer.reset()
        self.pred_scores = []
        self.gt_scores = []
        for step, data in enumerate(self.train_loader):
            self._val_step(data)

        preds = self.metric_computer.preds
        checkpoint['max'] = np.max(preds)
        checkpoint['min'] = np.min(preds)
        self.save_checkpoints(checkpoint, self.db_model_path)
        torch.save(checkpoint, self.db_model_path)

    def _train_step(self, data, step: int, start_time: float) -> Dict[str, float]:
        metrics = {}
        inputs, label = data[0]['data'], data[0]['label']
        label = label.squeeze(-1)
        label = label.cuda(self.gpu, non_blocking=True)

        metrics["data_time"] = time.time() - start_time
        self.optimizer.zero_grad()
        # with autocast(enabled=True):
        if self.train_attack:
            inputs = self.train_attack.run(inputs, label)
        # print(len(inputs))
        print(label)

#====================================================================
        self.model.zero_grad()
        pred, closs = self.model(inputs)
        pred2, closs2 = self.model(torch.flip(inputs, [3])) 

        self.pred_scores = self.pred_scores + pred.flatten().cpu().tolist()
        self.gt_scores = self.gt_scores + label.cpu().tolist()

        loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
        loss_qa2 = self.l1_loss(pred2.squeeze(), label.float().detach())
        # =============================================================================
        # =============================================================================

        indexlabel = torch.argsort(label) # small--> large
        # print(indexlabel)
        anchor1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0) # d_min
        positive1 = torch.unsqueeze(pred[indexlabel[1],...].contiguous(),dim=0) # d_min+
        negative1_1 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0) # d_max+

        anchor2 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0)# d_max
        positive2 = torch.unsqueeze(pred[indexlabel[-2],...].contiguous(),dim=0)# d'_max+
        negative2_1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0)# d_min+

        # =============================================================================
        # =============================================================================

        fanchor1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)
        fpositive1 = torch.unsqueeze(pred2[indexlabel[1],...].contiguous(),dim=0)
        fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)

        fanchor2 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)
        fpositive2 = torch.unsqueeze(pred2[indexlabel[-2],...].contiguous(),dim=0)
        fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)

        # consistency = nn.L1Loss()
        assert (label[indexlabel[-1]]-label[indexlabel[1]])>=0
        assert (label[indexlabel[-2]]-label[indexlabel[0]])>=0
        triplet_loss1 = nn.TripletMarginLoss(margin=(label[indexlabel[-1]]-label[indexlabel[1]]), p=1)
        triplet_loss2 = nn.TripletMarginLoss(margin=(label[indexlabel[-2]]-label[indexlabel[0]]), p=1)
        tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
            triplet_loss2(anchor2, positive2, negative2_1)
        ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
            triplet_loss2(fanchor2, fpositive2, fnegative2_1)

        loss = loss_qa + closs + loss_qa2 + closs2 + \
                0.5*(self.l1_loss(tripletlosses,ftripletlosses.detach())+ \
                self.l1_loss(ftripletlosses,tripletlosses.detach()))+ \
                0.05*(tripletlosses+ftripletlosses)
#====================================================================

        # model_out = self.model(inputs)
        # loss = self.loss(model_out, label.unsqueeze(1))
        if self.is_gradnorm_regularization:
            loss += self.weight_gradnorm_regularization*self._gradnorm_regularization(inputs)

        loss.backward()
        self.optimizer.step()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        metrics["total_loss"] = loss.cpu().detach().numpy()
        metrics["total_time"] = time.time() - start_time
        return metrics

    def _val_step(self, data, attack: Attacker = None) ->  None:
        if len(data[0])>1:
            inputs, label = data[0]['data'], data[0]['label']
            label = label.squeeze(-1)
            label = label.cuda(self.gpu, non_blocking=True)
        else:
            inputs = data[0]['data']
            label = torch.zeros(len(data[0]['data']), 1).cuda(self.gpu, non_blocking=True)

        if attack:
            inputs = attack.run(inputs, label)

        pred, _ = self.model(inputs)
        self.pred_scores = self.pred_scores + pred.cpu().tolist()
        self.gt_scores = self.gt_scores + label.cpu().tolist()
        # print(len(pred), len(label))
        self.metric_computer.update(pred, label)

    def _gradnorm_regularization(self, images):
        # images = images.clone().detach().requires_grad_(True).cuda()
        # pred_cur = self.model(images)
        # dx = torch.autograd.grad(pred_cur, images, grad_outputs=torch.ones_like(pred_cur), retain_graph=True)
        # dx = dx[0]
        # images.requires_grad_(False)

        # v = dx.view(dx.shape[0], -1)
        # v = torch.sign(v)

        # v = v.view(dx.shape).detach()
        # x2 = images + self.h_gradnorm_regularization*v

        # pred_pert = self.model(x2)
        # dl = (pred_pert - pred_cur)/self.h_gradnorm_regularization
        # loss = dl.pow(2).mean()/2
        # return loss
        raise NotImplementedError

    def _prepair_prune(self):
        print(self.model.model)
        if self.config['options']['prune'] <= 0. or self.eval_only:
            return

        self.model.model.load_pretrained(self.db_model_path)
        self.model.model.prune(amount=self.config['options']['prune'], 
                            prtype=self.config['options']['prune_type'],
                            width=256,
                            height=192,
                            images_count=100,
                            kernel=1)
        self.model.model.print_sparcity()
        self.end_epoch = self.config['options']['prune_epochs']
        self.config['lr_scheduler']['type'] = 'simple'
        self.config['lr_scheduler']['points'][0] = self.config['options']['prune_lr']
        self.config['lr_scheduler']['points'][self.end_epoch] = self.config['options']['prune_lr']

    def test(self) -> None:
        checkpoint = torch.load(self.db_model_path)
        datasets = ['KonIQ-10k', 'NIPS']
        # datasets = ['NIPS']
        for _dataset in datasets:
            self.model.load_state_dict(checkpoint['model'])
            # self.replace_backward_activations(self.config['options']['activation'])
            self.metric_computer = IQAPerformance(self.num_patches)
            metric_range = checkpoint['max'] - checkpoint['min']
            self.hash = self.__get_options_hash(self.config['options'])
            
            print(checkpoint['min'], checkpoint['max'])
            print(f'Metric range: {metric_range}')
            print('SROCC: ', checkpoint['SROCC'])
            print('PLCC: ', checkpoint['PLCC'])
            print('Test Dataset:', _dataset)

        
            if _dataset=="NIPS":
                self.test_loader = get_data_loader(
                            directory=self.config['test_data']['NIPS']['directory'],
                            rank=self.gpu,
                            num_tasks=self.world_size,
                            batch_size=self.config['train']['batch_size'],
                            num_workers=self.config['train']['num_workers'],
                            seed=self.config['seed']
                        )
            elif _dataset=="KonIQ-10k":
                self.test_loader = get_data_loaders(
                            rank=self.gpu,
                            num_tasks=self.world_size,
                            args=self.config['test_data']['KonIQ-10k'],
                            batch_size=self.config['train']['batch_size'],
                            num_workers=self.config['train']['num_workers'],
                            seed=self.config['seed'],
                            phase="test",
                        )
            # self.test_loader = self.load_dataset(_dataset)
            # print("LOADER: ", self.test_loader)
            results = {}
            self.model.eval()

            self.metric_computer.reset()
            for step, data in enumerate(self.test_loader):
                self._val_step(data)

            orig_preds = self.metric_computer.preds.copy()
            orig_preds = np.array(orig_preds)
            orig_preds_scaled = (orig_preds - checkpoint['min']) / metric_range
            results['origin_preds'] = orig_preds
            results['orig_preds_scaled'] = orig_preds_scaled
    
            for attack_args in self.config['attack']['test']:
                attack = self._init_attack(attack_args, "test", 
                                           metric_range=metric_range, 
                                           dataloader=self.test_loader)

                self.metric_computer.reset()
                for step, data in tqdm(enumerate(self.test_loader)):
                    self._val_step(data, attack)

                att_preds = np.array(self.metric_computer.preds)
                att_preds_scaled = (att_preds - checkpoint['min']) / metric_range
                results[f'preds_eps={attack_args["params"]["eps"]}'] = att_preds
                results[f'preds_scaled={attack_args["params"]["eps"]}'] = att_preds_scaled

                abs_gain = np.mean(att_preds - orig_preds)
                print(f'Abs gain for eps={attack_args["params"]["eps"]}: {abs_gain}')

                abs_gain = np.mean(att_preds_scaled - orig_preds_scaled)
                print(f'Abs gain for eps={attack_args["params"]["eps"]}: {abs_gain}')

                rel_gain = np.mean((att_preds_scaled - orig_preds_scaled) / (orig_preds_scaled + 1))
                print(f'Relative gain for eps={attack_args["params"]["eps"]}: {rel_gain}')
                pd.DataFrame.from_dict(results).to_csv(
                    f'csv/{_dataset}_{self.results_csv}_{attack_args["type"]}'
                    f'={attack_args["params"]["iters"]}.csv'
                )

    def eval(self) -> None:
        checkpoint = torch.load(self.db_model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.metric_computer = IQAPerformance(self.num_patches)
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric range: {metric_range}')

        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.test_loader):
            self._val_step(data)

        checkpoint['SROCC'] = self.metric_computer.srcc
        checkpoint['PLCC'] = self.metric_computer.plcc
        print('SROCC: ', checkpoint['SROCC'])
        print('PLCC: ', checkpoint['PLCC'])
        self.save_checkpoints(checkpoint, self.db_model_path)
        self.wandb_writer.log_artifact(artifact_or_path=self.db_model_path, 
                                       name="checkpoints", type="checkpoints")
        # self.test()

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def setup_distributed(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

        dist.init_process_group("nccl", rank=self.gpu, world_size=self.world_size)
        torch.cuda.set_device(self.gpu)

    @staticmethod
    def __get_options_hash(options):
        def __help(options):
            x = 'KonCept'
            for key in options:
                x += f"-{key}={options[key]}"
            return x
        hash_value = __help(options)
        # print(frozen, hash_value)
        return hash_value
    
    def save_checkpoints(self, checkpoint, trained_model_file):
        self.replace_backward_activations()
        if self.use_mask:
            # model_copy = copy.deepcopy(model)
            for module, name in self.model.model.prune_parameters:
                print(name, module,[ m[0] for m in list(module.named_buffers())])
                prune.remove(module, name)
                print('REMOVE ', name, module,[ m[0] for m in list(module.named_buffers())])
            checkpoint['model'] = self.model.state_dict()
            torch.save(checkpoint, trained_model_file)
        else:
            torch.save(checkpoint, trained_model_file)

    def replace_backward_activations(self):
        if isinstance(self.model.model.Activ, ReLU_SiLU):
            swap_all_activations(self.model, ReLU_SiLU, ReLU)
        elif isinstance(self.model.model.Activ, ReLU_ELU):
            swap_all_activations(self.model, ReLU_ELU, ReLU)
        else:
            return

    @classmethod
    def run(cls, *args):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            multiprocessing.spawn(
                cls._exec_wrapper, args=args, nprocs=world_size, join=True
            )
        else:
            cls.exec(0, *args)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, config_path):
        trainer = cls(gpu=gpu, config_path=config_path)
        if trainer.eval_only:
            trainer.test()
        else:
            trainer.train()
            print(str(datetime.now())[:-4])
            if gpu == 0:
                trainer.eval()
        if trainer.distributed:
            trainer.cleanup_distributed()

    # def __del__(self):
    #     # if self.gpu == 0:
    #     #     self.writer.close()
    #     self.wandb_writer.finish()