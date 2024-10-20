import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler, Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from models_train.IQAdataset import get_data_loaders
# from models_train.IQAmodel import IQAModel
from models_train.Linearity import Linearity
from models_train.IQAloss import IQALoss
from models_train.IQAperformance import IQAPerformanceLinearity, IQAPerfomanceKonCept
from models_train.pruning import PruneConv, l1_prune, pls_prune, ln_prune
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
from typing import Dict
from tqdm import tqdm
from torch.nn.utils import prune
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize
from torch import nn
# from mixer import MixData
# from style_transfer.adain import StyleTransfer
# from style_transfer.mixer import MixData

from icecream import ic
from clearml import Task, Logger
# metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']

# from dataclasses import dataclass


class Trainer:
    metrics_printed = ['SROCC', 'PLCC', 'RMSE']

    def __init__(self, device, args, prune_iter=0) -> None:
        # self.config = data
        if args.debug:
            ic.enable()
        else:
            ic.disable()
        self.args = args
        # self.base_model_name = args.model
        self.device = device
        self.arch = args.architecture
        self.device = device
        self.epochs = args.epochs
        self.current_epoch = 0
        self.gpu = 0
        self.dlayer = args.dlayer
        self.pruning = args.pruning
        self.noise_batch = args.noise
        self.gradnorm_regularization = args.gradnorm_regularization
        self.h_gradnorm_regularization = 6/255 # 10/255
        self.weight_gradnorm_regularization = 1e-1 # 1e-2
        self.cayley = args.cayley
        self.cayley_pool = args.cayley_pool
        self.cayley_pair = args.cayley_pair

        self.prune_iters = args.prune_iters
        self.width_prune = args.width_prune
        self.height_prune = args.height_prune
        self.images_count_prune = args.images_count_prune
        self.kernel_prune = args.kernel_prune
        

        self.model = Linearity(
                              arch='resnext101_32x8d' 
                                if self.args.architecture=='apgd_ssim' or 
                                self.args.architecture=='apgd_ssim_eps2' or 
                                self.args.architecture=='free_ssim_eps2' 
                                else self.args.architecture,
                              pool=self.args.pool,
                              use_bn_end=self.args.use_bn_end,
                              P6=self.args.P6, P7=self.args.P7,
                              activation=args.activation, dlayer=self.dlayer,
                              pruning=self.pruning, gabor=args.gabor,
                              cayley=self.cayley, cayley_pool=self.cayley_pool, cayley_pair=self.cayley_pair).to(self.device)


        self.scaler = GradScaler()
        self.k = [1, 1, 1]
        self.b = [0, 0, 0]

        current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        self.writer = SummaryWriter(log_dir='{}/{}-{}'.format(self.args.log_dir, self.args.format_str, current_time))
        self._optimizer()

    def train(self, train=True, val=True, test=True):
        self._prepair_train(train, val, test)
        self._train_loop()

    def compute_output(self, inputs, label):
        return self.model(inputs)

    def unpack_data(self, inputs, label, step):
        return inputs, label

    def _prepair(self, train=True, val=True, test=True, use_normalize=True):
        if train or val or test:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args, 
                                                                                    train=train, 
                                                                                    val=val,
                                                                                    test=test, 
                                                                                    use_normalize=use_normalize,
                                                                                    )
        # if self.base_model_name == "Linearity":
        self.loss_func = IQALoss(loss_type=self.args.loss_type, alpha=self.args.alpha, beta=self.args.beta,
                                p=self.args.p, q=self.args.q,
                                monotonicity_regularization=self.args.monotonicity_regularization,
                                gamma=self.args.gamma, detach=self.args.detach)
        # elif self.base_model_name == "KonCept":
        #     # self.loss_func = lambda output, label: nn.MSELoss()(output, label[0].unsqueeze(1))
        #     self.loss_func = lambda output, label: nn.MSELoss()(output, label[0].unsqueeze(1))


    def _prepair_train(self, train, val, test):
        self._prepair(train, val, test)
        if self.args.ft_lr_ratio == .0:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # if self.base_model_name=="Linearity":
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step,
                                                gamma=self.args.lr_decay)
        # elif self.base_model_name=="KonCept":
        #     points = {0: 1e-4, 40: 1e-4/5, 60: 1e-4/10} # 70 epochs
        #     self.scheduler = SimpleLRScheduler(self.optimizer, points)

        self.scaler = GradScaler()

        self.metric_computer = self._get_perfomance('val', k=[1, 1, 1], b=[0, 0, 0], mapping=True)
        self.best_val_criterion, self.best_epoch = -100, -1

        self._optimizer()

    def _optimizer(self):
        # if self.base_model_name=="Linearity":
        self.optimizer = Adam([{'params': self.model.regression.parameters()},
                            # The most important parameters. Maybe we need three levels of lrs
                            {'params': self.model.dr6.parameters()},
                            {'params': self.model.dr7.parameters()},
                            {'params': self.model.regr6.parameters()},
                            {'params': self.model.regr7.parameters()},
                            {'params': self.model.features.parameters(),
                                'lr': self.args.learning_rate * self.args.ft_lr_ratio}],
                            lr=self.args.learning_rate,
                            weight_decay=self.args.weight_decay)  # Adam can be changed to other optimizers, such as SGD, Adadelta.
        # elif self.base_model_name=="KonCept":
        #     self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        # else:
        #     raise NameError(f"No {self.base_model_name} model.")

    def _train_loop(self):
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.epochs:
            self.model.train()
            done_steps = self.current_epoch * train_data_len
            for step, (inputs, label) in tqdm(enumerate(self.train_loader), total=train_data_len):
                label = [k.cuda() for k in label]
                inputs = inputs.cuda()

                if self.noise_batch:
                    inputs, label = self.expand_batch(inputs, label, alpha=2)

                metrics = self._train_step(inputs, label, step)
                dump_scalar_metrics(
                    metrics, self.writer, 'train', global_step=done_steps + step
                )

            self.scheduler.step(self.current_epoch)
            self.current_epoch += 1

            self.metric_computer.reset()
            self.model.eval()
            for step, (inputs, label) in enumerate(self.val_loader):
                self._val_step(inputs=inputs, label=label)
            metrics = self.metric_computer.compute()
            dump_scalar_metrics(
                metrics,
                self.writer,
                'val',
                global_step=self.current_epoch,
                dataset=self.args.dataset
            )
            val_criterion = abs(metrics[self.args.val_criterion])
            # print(val_criterion, self.best_val_criterion)
            if val_criterion > self.best_val_criterion:
                # if self.args.debug:
                # print('max:', 'max_pred', 'min:', 'min_pred')
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'k': self.k,
                    'b': self.b,
                }
                torch.save(checkpoint, self.args.trained_model_file)

                self.best_val_criterion = val_criterion
                self.best_epoch = self.current_epoch
                print(
                    f'Save current best model @best_val_criterion ({self.args.val_criterion}):\
                          {self.best_val_criterion:.3f} @epoch: {self.best_epoch}'
                )
            else:
                print(
                    f'Model is not updated @val_criterion ({self.args.val_criterion}):\
                          {val_criterion:.3f} @epoch: {self.current_epoch}'
                )

        self.metric_computer = self._get_perfomance(
            'train', k=[1, 1, 1], b=[0, 0, 0], mapping=True
        )

        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.train_loader):
            self._val_step(inputs, label)
        coeffs = self.metric_computer.compute()
        preds = self.metric_computer.preds
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'k': coeffs['k'],
            'b': coeffs['b'],
            'min': preds.min(),
            'max': preds.max(),
            'epoch': self.current_epoch,
            'SROCC': self.best_val_criterion,

        }
        torch.save(checkpoint, self.args.trained_model_file)
        print('checkpoints saved')

        checkpoint = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.train_loader):
            self._val_step(inputs, label)

        coeffs = self.metric_computer.compute()
        print(coeffs)
        checkpoint['k'] = coeffs['k']
        checkpoint['b'] = coeffs['b']
        preds = self.metric_computer.preds
        checkpoint['max'] = preds.max()
        checkpoint['min'] = preds.min()
        checkpoint['SROCC'] = self.metric_computer.SROCC
        checkpoint['PLCC'] = self.metric_computer.PLCC
        checkpoint ['RMSE'] = self.metric_computer.RMSE
        torch.save(checkpoint, self.args.trained_model_file)
        Task.current_task().upload_artifact(name="Metrics", 
                                            artifact_object={
                                                'model_name': "Linearity",
                                                'min': checkpoint['min'],
                                                'max': checkpoint['max'],
                                                'SROCC': checkpoint['SROCC'],
                                                'PLCC': checkpoint['PLCC'],
                                                'RMSE': checkpoint['RMSE'],
                                            })

    def _train_step(self, inputs, label, step):
        inputs = inputs.to(self.device)

        inputs, label = self.unpack_data(inputs, label, step)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.compute_output(inputs, label)

        loss = self.loss_func(output, label) / self.args.accumulation_steps 

        if self.gradnorm_regularization:
            grad_loss = self.gradnorm_regularize(inputs)
            loss += self.weight_gradnorm_regularization*grad_loss
        ic(loss)

        with autocast(enabled=True):
            self.scaler.scale(loss).backward()
            if step % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        metrics = {'loss': loss.cpu().detach().numpy()}
        return metrics

    def _val_step(self, inputs, label):
        inputs = inputs.to(self.device)
        label = [k.to(self.device) for k in label]
        output = self.compute_output(inputs, label)
        self.metric_computer.update((output, label))

    def eval(self, loaded=False):
        if self.args.evaluate:
            self._prepair(train=False, val=True, test=True)

        checkpoint = torch.load(self.args.trained_model_file)
        # results = {}
        # print(checkpoint['model'])
        if not loaded:
            self.model.load_state_dict(checkpoint['model'])
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.metric_computer = self._get_perfomance('test', k=self.k, b=self.b, mapping=True)

        self.model.eval()
        self.metric_computer.reset()
        test_len = len(self.test_loader)
        for step, (inputs, label) in tqdm(enumerate(self.test_loader), total=test_len):
            self._val_step(inputs, label)
        metrics = self.metric_computer.compute()

        checkpoint['model'] = self.model.state_dict()
        checkpoint['SROCC'] = self.metric_computer.SROCC
        checkpoint['min'] = self.metric_computer.preds.min()
        checkpoint['max'] = self.metric_computer.preds.max()
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric_range:', metric_range)
        torch.save(checkpoint, self.args.trained_model_file)

        print('{}, {}: {:.3f}'.format(self.args.dataset, self.metrics_printed[0], metrics[self.metrics_printed[0]]))
        np.save(self.args.save_result_file, metrics)

    # def __del__(self):
    #     self.writer.close()

    def expand_batch(self, inputs, label, alpha=2):
        eps = alpha/255
        noise = torch.randn_like(inputs, device="cuda")
        inputs_noise = inputs + eps*noise
        inputs = torch.cat((inputs, inputs_noise), 0)
        label[0] = torch.cat((label[0],label[0]))
        label[1] = torch.cat((label[1],label[1]))
        ic(inputs.shape)
        ic(label)
        return inputs, label

    def _get_perfomance(self, *args, **kwargs):
        # if self.base_model_name=="Linearity":
        return IQAPerformanceLinearity(*args, **kwargs)
        # elif self.base_model_name=="KonCept":
        #     return IQAPerfomanceKonCept(*args, **kwargs)
        # raise NameError(f"No {self.base_model_name} model.")

    def prune(self):
        for i in range(self.prune_iters):
            self._prepair_prune(prune_iter=i)
            self._prune_features()
            # self.eval(loaded=True)
            self.current_epoch = 0
            if self.epochs > 0:
                if i == 0:
                    self.train()
                else:
                    self.train(train=False, val=False, test=False)
            elif self.epochs == 0 and i == 0:
                self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args, train=False, val=True,
                                                                                    test=True, use_normalize=True,
                                                                                    )

    def _prune_features(self):
        prune_parameters: tuple

        if self.args.pruning_type == 'l1':
            prune_parameters = l1_prune(self.model, self.pruning)
        elif self.args.pruning_type == 'pls':
            prune_parameters = pls_prune(self.model, self.pruning,
                                        width=self.width_prune, 
                                        height=self.height_prune, 
                                        images_count=self.images_count_prune, 
                                        kernel=self.kernel_prune) # 120, 90
        elif self.args.pruning_type == 'l2':
            prune_parameters = ln_prune(self.model, self.pruning, 2)

        self.model.print_sparcity(prune_parameters)

    def _prepair_prune(self, prune_iter=0):
        if prune_iter > 0:
            form = self.args.trained_model_file[:-1] + str(prune_iter+1) # prune_iter += 1
        elif prune_iter == 0:
            # form = f'{self.args.trained_model_file}+prune={self.args.pruning}{self.args.pruning_type}_lr={self.args.learning_rate}_e={self.args.epochs}_iters={prune_iter+1}'
            form = '{}+prune={}{}_lr={}_e={}_iters={}'.format(self.args.trained_model_file, self.args.pruning,
                                                              self.args.pruning_type, self.args.learning_rate,
                                                              self.args.epochs, prune_iter+1)
        else:
            raise Exception

        checkpoint = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.args.trained_model_file = form
        torch.save(checkpoint, self.args.trained_model_file)
        print(self.args.trained_model_file)
        
        ##################
        # convs = []
        # prune_parameters = []
        # for layer in self.model.features:
        #     if isinstance(layer, nn.Sequential):
        #         for block in layer:
        #             for conv in block.children():
        #                 if isinstance(conv, nn.Conv2d):
        #                     convs.append(conv)
        # for i in range(len(convs)):
        #     prune_parameters.append((convs[i], 'weight'))
        ######################

        # self.model.print_sparcity(prune_parameters)


    def gradnorm_regularize(self, images):
        get_pred = lambda output: output[-1]*self.k[0] + self.b[0]
        images = images.cuda()
        images.requires_grad_(True)
        
        output_cur = self.compute_output(images, None)
        ic(output_cur)

        pred_cur = get_pred(output_cur)
        ic(pred_cur)
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

    @classmethod
    def run(cls, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainer = cls(device, args)
        # print(args.evaluate)
        if args.evaluate:
            trainer.eval()
        elif args.pruning:
            trainer.prune()
            trainer.eval()
        else:
            trainer.train()
            trainer.eval()


class SimpleLRScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        points: Dict[int, float],
    ):
        self.optimizer = optimizer
        self.points = points

    def step(self, epoch: int):
        if epoch in self.points:
            self.lr = self.points[epoch]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

        print("LR=", self.lr)
        return self.lr



def dump_scalar_metrics(metrics: Dict, writer: SummaryWriter, phase: str, global_step: int = 0, dataset: str = ''):
    prefix = phase.lower() + (f'_{dataset}' if dataset else '')
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f'{metric_name}/{prefix}',
            metric_value,
            global_step=global_step,
        )
