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
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from attacks.fgsm import FGSM
from attacks.pgd import PGD, AutoPGD
from attacks.base import Attacker
from lr_scheduler import SimpleLRScheduler
from model import DBCNN, normalize_model
from datasets.koniq10k_dali import get_data_loaders
from log import dump_config
from metrics import IQAPerformance, dump_scalar_metrics
from utils import load_config

from tqdm import tqdm


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["KMP_WARNINGS"] = "off"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class AdversarialTrainer:
    def __init__(self, gpu: int, config_path: Path):
        self.config_path = config_path
        self.config = load_config(config_path)
        # self.config = config_path
        self.eval_only = self.config["eval_only"]

        self.world_size = torch.cuda.device_count()
        self.distributed = self.world_size > 1
        self.gpu = gpu
        print(f"I am gpu #{self.gpu}")

        if self.distributed:
            self.setup_distributed()

        self.fc_model_dir = Path(self.config['fc_model'])
        scnn_root = self.config['data']['scnn_root']

        self.options = self.config['options']
        self.model = normalize_model(
            DBCNN(scnn_root, **self.options),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.model.to(self.gpu)
        print(self.model.model)

        # checkpoint = torch.load()
        # self.model.load_state_dict(checkpoint['model'])

        self._init_fc()
        self._init_logger()

    # def __del__(self):
    #     if self.gpu == 0:
    #         self.writer.close()

    def _init_logger(self):
        if self.gpu == 0:
            if self.config["checkpoint_path"]:
                self.log_dir = Path(self.config["checkpoint_path"])
                test_config = self.config['attack']['test']
                self.config = load_config(Path(self.config["checkpoint_path"]) / "presets.yaml")
                self.config['attack']['test'] = test_config
                self.writer = SummaryWriter(log_dir=self.log_dir)
                return

            if self.config["attack"]["train"]["type"] == "none":
                train_method = "origin"
            elif self.config["attack"]["train"]["type"] == "apgd":
                train_method = "apgd"
            elif self.config["attack"]["train"]["params"]["mode"] == "zero":
                train_method = "fgsm"
            elif self.config["attack"]["train"]["params"]["mode"] == "uniform":
                train_method = "free_fgsm"
            else:
                raise NotImplementedError

            if train_method == "origin":
                threat = 0
            else:
                threat = self.config["attack"]["train"]["params"]["eps"]
            exp = f'ep={self.config["train"]["epochs"]}_eps={threat}'

            self.log_dir = (
                Path(self.config["log"]["directory"])
                / train_method
                / f"{self.config['label_strategy']}{('_' + self.config['penalty']) if 'penalty' in self.config else ''}"
                / f'{str(datetime.now())[:-4]}_{exp}_{self.config["lr_scheduler"]["type"]}'
            )
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.log_dir)

            print(f"=> Logging in {self.log_dir}")
            dump_config(self.config, self.writer)

            # try not to lose the best presets
            shutil.copy(self.config_path, self.log_dir / "presets.yaml")

            self.start_training_time = time.time()

    def _init_fc(self):
        if self.options['fc']:
            return
        fc_model = torch.load(self.fc_model_dir / f'{hash(frozenset(self.options))}.pkl')
        self.model.load_state_dict(fc_model)

    def train(self) -> None:
        self._prepare_for_training()
        self._train_loop()

    def _prepare_for_training(self) -> None:
        self.current_epoch = 0
        self.end_epoch = self.config["train"]["epochs"]

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = get_data_loaders(
            rank=self.gpu,
            num_tasks=self.world_size,
            args=self.config["data"],
            batch_size=self.config["train"]["batch_size"],
            num_workers=self.config["train"]["num_workers"],
            seed=self.config["seed"],
        )

        self._init_optimizer(lr=self.config['train']['lr'], 
                             weight_decay=self.config['train']['weight_decay'])
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu]
            )
        self.scaler = GradScaler()

        self._init_lr_scheduler()
        self.loss = nn.MSELoss()
        self.train_attack = self._init_attack(
            self.config["attack"]["train"], "train"
        )

        if self.gpu == 0:
            self.val_criterion = self.config["train"]["val_criterion"]
            self.metric_computer = IQAPerformance()
            self.best_val_criterion, self.best_epoch = -100, -1

    def _init_optimizer(self, lr=0.00005, weight_decay=0.0005) -> None:
        if self.config['options']['fc'] == True:
            self.optimizer = torch.optim.SGD(
                    self.model.model.fc.parameters(), lr=lr,
                    momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                    self.model.model.parameters(), lr=lr,
                    weight_decay=weight_decay)

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
        elif self.config['lr_scheduler']['type'] == 'multistep':
            self.lr_scheduler = lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.config["lr_scheduler"]["milestones"],
                gamma=self.config["lr_scheduler"]["gamma"]
            )


    def _init_attack(
        self,
        attack_config: dict[str, Any],
        phase: str,
        metric_range: Optional[Tuple[float, float]] = None,
    ) -> Attacker:
        attack_name = attack_config["type"]
        if attack_name == "none":
            return None

        attackers = {"fgsm": FGSM, "pgd": PGD, "apgd": AutoPGD}
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

        return attacker

    def _train_loop(self) -> None:
        train_data_len = len(self.train_loader)
        print(train_data_len)
        while self.current_epoch < self.end_epoch:
            self.model.train()
            if self.config['lr_scheduler']['type'] == 'simple':
                self.lr_scheduler.adjust_learning_rate(self.current_epoch)

            done_steps = self.current_epoch * train_data_len
            batch_start_time = time.time()

            for step, data in tqdm(enumerate(self.train_loader), total=train_data_len):
                metrics = self._train_step(data, step, batch_start_time)
                if self.gpu == 0:
                    dump_scalar_metrics(
                        metrics, self.writer, "train", global_step=done_steps + step
                    )

                batch_start_time = time.time()

            if self.config['lr_scheduler']['type'] == 'step':
                self.lr_scheduler.step()
            elif self.config['lr_scheduler']['type'] == 'multistep':
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
            if val_criterion >= self.best_val_criterion and self.current_epoch > 10:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                }

                if self.options['fc'] == True:
                    torch.save(self.model.state_dict(), 
                               self.fc_model_dir / f'{hash(frozenset(self.options))}.pkl')
                elif self.options['fc'] == False:
                    torch.save(checkpoint, self.log_dir / 'best_model.pth')

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

        if self.options['fc'] == True:
            return

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
        }

        self.metric_computer = IQAPerformance()

        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.train_loader):
            self._val_step(data)

        preds = self.metric_computer.preds
        checkpoint['max'] = np.max(preds)
        checkpoint['min'] = np.min(preds)
        torch.save(checkpoint, self.log_dir / 'final_model.pth')

        # Update min and max metric scores for the best model
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.metric_computer.reset()
        for step, data in enumerate(self.train_loader):
            self._val_step(data)

        preds = self.metric_computer.preds
        checkpoint['max'] = np.max(preds)
        checkpoint['min'] = np.min(preds)
        torch.save(checkpoint, self.log_dir / 'best_model.pth')

    def _train_step(self, data, step: int, start_time: float) -> Dict[str, float]:
        metrics = {}
        inputs, label = data[0]['data'], data[0]['label']

        label = label.squeeze(-1)
        label = label.cuda(self.gpu, non_blocking=True)

        metrics["data_time"] = time.time() - start_time

        self.optimizer.zero_grad()
        
        with autocast(enabled=True, device_type='cuda'):
            if self.train_attack:
                inputs = self.train_attack.run(inputs, label)

            model_out = self.model(inputs)
            loss = self.loss(model_out, label.unsqueeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics["total_loss"] = loss.cpu().detach().numpy()
        metrics["total_time"] = time.time() - start_time
        return metrics

    def _val_step(self, data, attack: Attacker = None) ->  None:
        inputs, label = data[0]['data'], data[0]['label']
        label = label.squeeze(-1)
        label = label.cuda(self.gpu, non_blocking=True)

        if attack:
            inputs = attack.run(inputs, label)

        model_out = self.model(inputs)
        self.metric_computer.update(model_out, label)

    def _gradnorm_regularize(self, images):
        get_pred = lambda output: output[-1]*self.k[0] + self.b[0]
        images = images.cuda()
        images.requires_grad_(True)
        
        output_cur = self.compute_output(images, None)

        pred_cur = get_pred(output_cur)

        dx = torch.autograd.grad(pred_cur, images, grad_outputs=torch.ones_like(pred_cur), retain_graph=True)
        dx = dx[0]
        images.requires_grad_(False)

        v = dx.view(dx.shape[0], -1)
        v = torch.sign(v)

        v = v.view(dx.shape).detach()
        x2 = images + self.h_gradnorm_regularization*v

        output_pert = self.compute_output(x2, None)
        pred_pert = get_pred(output_pert)

        dl = (pred_pert - pred_cur)/self.h_gradnorm_regularization
        loss = dl.pow(2).mean()/2

        return loss

    def eval(self) -> None:
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        results = {}
        self.model.load_state_dict(checkpoint['model'])
        self.metric_computer = IQAPerformance()
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric range: {metric_range}')

        self.test_loader = get_data_loaders(
            rank=self.gpu,
            num_tasks=self.world_size,
            args=self.config['data'],
            batch_size=self.config['train']['batch_size'],
            num_workers=self.config['train']['num_workers'],
            seed=self.config['seed'],
            phase="test",
        )

        self.model.eval()
        self.metric_computer.reset()
        for step, data in enumerate(self.test_loader):
            self._val_step(data)
        
        checkpoint['SROCC'] = self.metric_computer.srcc
        checkpoint['PLCC'] = self.metric_computer.plcc
        print('SROCC: ', checkpoint['SROCC'])
        print('PLCC: ', checkpoint['PLCC'])
        torch.save(checkpoint, self.log_dir / 'best_model.pth')

        orig_preds = self.metric_computer.preds.copy()
        orig_preds = np.array(orig_preds)
        orig_preds_scaled = (orig_preds - checkpoint['min']) / metric_range
        results['origin_preds'] = orig_preds
        results['orig_preds_scaled'] = orig_preds_scaled
 
        for attack_args in self.config['attack']['test']:
            attack = self._init_attack(attack_args, "test", metric_range=metric_range)

            self.metric_computer.reset()
            for step, data in enumerate(self.test_loader):
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
        
        if len(self.config['attack']['test']) > 0:
            pd.DataFrame.from_dict(results).to_csv(
                self.log_dir / f"{self.config['data']['dataset']}_test_{self.config['attack']['test'][0]['type']}.csv",
                index=False
            )

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def setup_distributed(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

        dist.init_process_group("nccl", rank=self.gpu, world_size=self.world_size)
        torch.cuda.set_device(self.gpu)

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
            trainer.eval()
        else:
            trainer.train()
            print(str(datetime.now())[:-4])
            if gpu == 0:
                trainer.eval()
        if trainer.distributed:
            trainer.cleanup_distributed()
