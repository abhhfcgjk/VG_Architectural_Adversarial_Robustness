import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from IQAdataset import get_data_loaders
from IQAmodel import IQAModel
from IQAloss import IQALoss
from IQAperformance import IQAPerformance
from tensorboardX import SummaryWriter
import datetime
import numpy as np
from typing import Dict
from activ import ReLU_to_SILU, ReLU_to_ReLUSiLU
from tqdm import tqdm
from torch.nn.utils import prune
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize
from pruning import PruneConv, l1_prune, pls_prune, ln_prune
from torch import nn
# from mixer import MixData
from style_transfer.adain import StyleTransfer
from style_transfer.mixer import MixData

from icecream import ic
# metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']

# from dataclasses import dataclass


class Trainer:
    metrics_printed = ['SROCC', 'PLCC', 'RMSE']

    def __init__(self, device, args) -> None:
        # self.config = data
        self.args = args
        self.device = device
        self.arch = self.args.architecture
        self.device = device
        self.epochs = self.args.epochs
        self.current_epoch = 0
        self.gpu = 0
        self.is_se = args.squeeze_excitation
        self.pruning = args.pruning
        self.model = IQAModel(arch=self.args.architecture,
                              pool=self.args.pool,
                              use_bn_end=self.args.use_bn_end,
                              P6=self.args.P6, P7=self.args.P7,
                              activation=args.activation, se=self.is_se,
                              pruning=self.pruning).to(self.device)
        self.mgamma = args.mgamma
        self.feature_model = args.feature_model
        if args.feature_model:
            self.schedule_for_model = [30,60,90]
            self.style_transfer = StyleTransfer()
            self.mixup = MixData(args.gamma, self.device)
        else:
            # self.feature_model = None
            self.style_transfer = None
            self.mixup = None

        self.scaler = GradScaler()
        self.k = [1, 1, 1]
        self.b = [0, 0, 0]

        current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        self.writer = SummaryWriter(log_dir='{}/{}-{}'.format(self.args.log_dir, self.args.format_str, current_time))
        self._optimizer()

    def train(self):
        if self.args.pruning:
            self._prepair_prune()
        self._prepair_train()
        self._train_loop()

    def compute_output(self, inputs, label):
        if self.style_transfer:
            output1 = self.model(inputs[0])
            output2 = self.model(inputs[1])
            output = [torch.cat([o1, o2], dim=0) for o1, o2 in zip(output1, output2)]
            # label = torch.cat([
            #     torch.tensor([k.tolist() for k in label[0]]),
            #     torch.tensor([k.tolist() for k in label[1]])
            # ], dim=1).to(self.device)
            # label = ()
            return output
        return self.model(inputs)

    def unpack_data(self, inputs, label, step):
        if self.style_transfer:
            label = torch.tensor([k.tolist() for k in label]).to(self.device)
            img_size = self.img_size_scheduler(step, self.epochs, self.schedule_for_model)
            resized_inputs = F.interpolate(inputs, size=img_size)
            input_aux, targets = self.style_transfer(resized_inputs, label, 1-self.mgamma)
            inputs = (inputs, input_aux)
            # assert len(target_aux) == 3
            # batch_size = label[0].size(0)
            inputs, label = self.mixup(inputs, targets)
            if self.feature_model == 'shape':
                label[2] = 0.
            elif self.feature_model == 'texture':
                label[2] = 1.
            elif self.feature_model == 'debiased':
                pass
            else:
                raise ModuleNotFoundError('')
            # label = targets
            inputs = (
                normalize(inputs[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                normalize(inputs[1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            )

        return inputs, label

    def _prepair(self, train=True, val=True, test=True, use_normalize=True):
        # print(train,val,test)
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args, train=train, val=val,
                                                                                test=test, use_normalize=use_normalize)
        self.loss_func = IQALoss(loss_type=self.args.loss_type, alpha=self.args.alpha, beta=self.args.beta,
                                 p=self.args.p, q=self.args.q,
                                 monotonicity_regularization=self.args.monotonicity_regularization,
                                 gamma=self.args.gamma, detach=self.args.detach)
        if self.mixup is not None:
            self.mixup.criterion = self.loss_func
            self.loss_func = self.mixup.calculate_loss


    def _prepair_train(self):
        self._prepair(use_normalize=(self.feature_model is None))
        if self.args.ft_lr_ratio == .0:
            for param in self.model.features.parameters():
                param.requires_grad = False

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step,
                                             gamma=self.args.lr_decay)

        self.scaler = GradScaler()

        self.metric_computer = IQAPerformance('val', k=[1, 1, 1], b=[0, 0, 0], mapping=True)
        self.best_val_criterion, self.best_epoch = -100, -1

        self._optimizer()

    def _optimizer(self):
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

    def _train_loop(self):
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.epochs:
            self.model.train()
            done_steps = self.current_epoch * train_data_len
            for step, (inputs, label) in tqdm(enumerate(self.train_loader), total=train_data_len):

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

        self.metric_computer = IQAPerformance(
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
            'SROCC': self.best_val_criterion
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
        checkpoint['SROCC'] = abs(self.metric_computer.SROCC)
        torch.save(checkpoint, self.args.trained_model_file)

    def _train_step(self, inputs, label, step):
        inputs = inputs.to(self.device)

        inputs, label = self.unpack_data(inputs, label, step)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.compute_output(inputs, label)
        # output = self.model(inputs)
        loss = self.loss_func(output, label) / self.args.accumulation_steps
        with autocast(enabled=True):
            self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # self.optimizer.zero_grad()
        metrics = {'loss': loss.cpu().detach().numpy()}
        return metrics

    def _val_step(self, inputs, label):
        # inputs = inputs.cuda(self.gpu, non_blocking=True)
        inputs = inputs.to(self.device)
        # label = [k.cuda(self.gpu, non_blocking=True) for k in label]
        label = [k.to(self.device) for k in label]

        output = self.model(inputs)
        self.metric_computer.update((output, label))

    def eval(self):
        if self.args.evaluate:
            self._prepair(train=False, val=True, test=True)

        checkpoint = torch.load(self.args.trained_model_file)
        # results = {}
        # print(checkpoint['model'])
        self.model.load_state_dict(checkpoint['model'])
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.metric_computer = IQAPerformance('test', k=self.k, b=self.b, mapping=True)
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric_range:', metric_range)

        checkpoint = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoint['model'])
        # if prune:
        #     self.prune()
        # print(getattr(self.model.layer1.conv1, 'weight') == 0)
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.model.eval()
        self.metric_computer.reset()

        for step, (inputs, label) in enumerate(self.test_loader):
            self._val_step(inputs, label)
        metrics = self.metric_computer.compute()

        checkpoint['model'] = self.model.state_dict()
        checkpoint['SROCC'] = abs(self.metric_computer.SROCC)
        torch.save(checkpoint, self.args.trained_model_file)

        print('{}, {}: {:.3f}'.format(self.args.dataset, self.metrics_printed[0], metrics[self.metrics_printed[0]]))
        np.save(self.args.save_result_file, metrics)

    # def __del__(self):
    #     self.writer.close()

    def _prepair_prune(self):
        form = self.args.trained_model_file + f'+prune={self.args.pruning}' + self.args.pruning_type + f'_lr={self.args.learning_rate}_e={self.args.epochs}'

        try:
            checkpoint = torch.load(
                self.args.trained_model_file + f'+prune={self.args.pruning}' + self.args.pruning_type)
        except Exception:
            checkpoint = torch.load(self.args.trained_model_file)
            self.model.load_state_dict(checkpoint['model'])
            self.prune()
            checkpoint['model'] = self.model.state_dict()
            self.args.trained_model_file = self.args.trained_model_file + f'+prune={self.args.pruning}' + self.args.pruning_type
            print(self.args.trained_model_file)
            torch.save(checkpoint, self.args.trained_model_file)
            self._prepair(train=False, val=True, test=True)
            self.eval()

        self.args.trained_model_file = form
        self.model.load_state_dict(checkpoint['model'])

        ##################
        convs = []
        prune_parameters = []
        for layer in self.model.features:
            if isinstance(layer, nn.Sequential):
                for block in layer:
                    for conv in block.children():
                        if isinstance(conv, nn.Conv2d):
                            convs.append(conv)
        for i in range(len(convs)):
            prune_parameters.append((convs[i], 'weight'))
        ######################

        IQAModel.print_sparcity(prune_parameters)

    def prune(self):
        prune_parameters: tuple
        if self.args.pruning_type == 'l1':
            prune_parameters = l1_prune(self.model, self.pruning)
        elif self.args.pruning_type == 'pls':
            prune_parameters = pls_prune(self.model, self.pruning,
                                         width=120, height=90, images_count=50)
        elif self.args.pruning_type == 'l2':
            prune_parameters = ln_prune(self.model, self.pruning, 2)

        IQAModel.print_sparcity(prune_parameters)

    def img_size_scheduler(self, batch_idx, epoch, schedule, min_size=260):
        ret_size = 224.0
        if batch_idx % 2 == 0:
            ret_size /= 2 ** 0.5
        schedule = [0] + schedule + [self.epochs]
        for i in range(len(schedule) - 1):
            if schedule[i] <= epoch < schedule[i + 1]:
                if epoch < (schedule[i] + schedule[i + 1]) / 2:
                    ret_size /= 2 ** 0.5
                    ri = i
                break
        ret_size = max(int(ret_size + 0.5), min_size)
        if batch_idx == 0:
            print("img size is ", ret_size)
        return ret_size, ret_size

    @classmethod
    def run(cls, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainer = cls(device, args)
        # print(args.evaluate)
        if args.pruning:
            # print("EVAL")
            trainer.train()
            trainer.eval()
        elif args.evaluate:
            trainer.eval()
        else:
            trainer.train()
            trainer.eval()


def dump_scalar_metrics(metrics: Dict, writer: SummaryWriter, phase: str, global_step: int = 0, dataset: str = ''):
    prefix = phase.lower() + (f'_{dataset}' if dataset else '')
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f'{metric_name}/{prefix}',
            metric_value,
            global_step=global_step,
        )
