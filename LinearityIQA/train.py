import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from IQAdataset import get_data_loaders
from IQAmodel import IQAModel
from IQAloss import IQALoss
from IQAperformance import IQAPerformance
# from tensorboardX import SummaryWriter
import datetime
import numpy as np
from typing import Dict
from activ import ReLU_to_SILU, ReLU_to_ReLUSiLU
from tqdm import tqdm

# metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']

# from dataclasses import dataclass


class Trainer:
    metrics_printed = ['SROCC', 'PLCC', 'RMSE']
    def __init__(self, device, args) -> None:
        # self.config = data
        self.args = args
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
        
        # if args.activation=='silu':
        #     ReLU_to_SILU(self.model)
        # elif args.activation=='relu_silu':
        #     ReLU_to_ReLUSiLU(self.model)

        self.scaler = GradScaler()
        self.k = [1,1,1]
        self.b = [0,0,0]

        current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        # self.writer = SummaryWriter(log_dir='{}/{}-{}'.format(self.args.log_dir, self.args.format_str, current_time))
        self._optimizer()
        
    def train(self):
        self._prepair_train()
        self._train_loop()

    def _prepair(self, train=True, val=True, test=True):
        # print(train,val,test)
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args, train=train, val=val, test=test)
        self.loss_func = IQALoss(loss_type=self.args.loss_type, alpha=self.args.alpha, beta=self.args.beta, 
                            p=self.args.p, q=self.args.q, 
                            monotonicity_regularization=self.args.monotonicity_regularization, 
                            gamma=self.args.gamma, detach=self.args.detach)
    def _prepair_train(self):
        self._prepair()
        if self.args.ft_lr_ratio == .0:
            for param in self.model.features.parameters():
                param.requires_grad = False
        # self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        # self.loss_func = IQALoss(loss_type=self.args.loss_type, alpha=self.args.alpha, beta=self.args.beta, 
        #                     p=self.args.p, q=self.args.q, 
        #                     monotonicity_regularization=self.args.monotonicity_regularization, 
        #                     gamma=self.args.gamma, detach=self.args.detach)
        self.scaler = GradScaler()

        
        self.metric_computer = IQAPerformance('val', k=[1,1,1], b=[0,0,0], mapping=True)
        self.best_val_criterion, self.best_epoch = -100, -1
        # global best_val_criterion, best_epoch, max_pred, min_pred
        # best_val_criterion, best_epoch = -100, -1  # larger, better, e.g., SROCC or PLCC. If RMSE is used, best_val_criterion <- 10000
        # max_pred, min_pred = -200, 200

        self._optimizer()
    
    def _optimizer(self):
        self.optimizer = Adam([{'params': self.model.regression.parameters()}, # The most important parameters. Maybe we need three levels of lrs
                    {'params': self.model.dr6.parameters()},
                    {'params': self.model.dr7.parameters()},
                    {'params': self.model.regr6.parameters()},
                    {'params': self.model.regr7.parameters()},
                    {'params': self.model.features.parameters(), 'lr': self.args.learning_rate * self.args.ft_lr_ratio}],
                    lr=self.args.learning_rate, weight_decay=self.args.weight_decay) # Adam can be changed to other optimizers, such as SGD, Adadelta.

    def _train_loop(self):
        train_data_len = len(self.train_loader)
        while self.current_epoch < self.epochs:
            self.model.train()
            done_steps = self.current_epoch * train_data_len
            for step, (inputs, label) in tqdm(enumerate(self.train_loader),total=train_data_len):
                
                metrics = self._train_step(inputs, label, step)
                # if self.gpu == 0:
                # dump_scalar_metrics(
                #     metrics, self.writer, 'train', global_step=done_steps + step
                # )

            self.scheduler.step(self.current_epoch)
            self.current_epoch += 1
            
            self.metric_computer.reset()
            self.model.eval()
            for step, (inputs, label) in enumerate(self.val_loader):
                self._val_step(inputs=inputs, label=label)
            metrics = self.metric_computer.compute()
            # dump_scalar_metrics(
            #     metrics,
            #     self.writer,
            #     'val',
            #     global_step=self.current_epoch,
            #     dataset=self.args.dataset
            # )
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
            'train', k = [1,1,1], b=[0,0,0], mapping=True
        )

        self.metric_computer.reset()
        for step, (inputs, label) in enumerate(self.train_loader):
            self._val_step(inputs,label)
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
        # inputs = inputs.cuda(self.gpu, non_blocking=True)
        inputs = inputs.to(self.device)
        # label = [k.cuda(self.gpu, non_blocking=True) for k in label]
        label = [k.to(self.device) for k in label]
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)
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
        self.model.load_state_dict(checkpoint['model'])
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.metric_computer = IQAPerformance('test', k=self.k, b=self.b,mapping=True)
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric_range:', metric_range)

        checkpoint = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.model.eval()
        self.metric_computer.reset()

        for step, (inputs, label) in enumerate(self.test_loader):
            self._val_step(inputs, label)
        metrics = self.metric_computer.compute()

        checkpoint['SROCC'] = abs(self.metric_computer.SROCC)
        torch.save(checkpoint, self.args.trained_model_file)
        
        print('{}, {}: {:.3f}'.format(self.args.dataset, self.metrics_printed[0], metrics[self.metrics_printed[0]]))
        np.save(self.args.save_result_file, metrics)

    # def __del__(self):
    #     self.writer.close()

    @classmethod
    def run(cls, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # with open(config_path, "r") as conf:
        #     data = yaml.load(conf, Loader=yaml.SafeLoader)
        trainer = cls(device, args)
        # print(args.evaluate)
        if args.evaluate:
            # print("EVAL")
            trainer.eval()
        else:
            trainer.train()
            trainer.eval()



# def dump_scalar_metrics(metrics: Dict, writer: SummaryWriter, phase: str, global_step: int =0, dataset: str=''):
#     prefix = phase.lower() + (f'_{dataset}' if dataset else '')
#     for metric_name, metric_value in metrics.items():
#         writer.add_scalar(
#             f'{metric_name}/{prefix}',
#             metric_value,
#             global_step=global_step,
#         )