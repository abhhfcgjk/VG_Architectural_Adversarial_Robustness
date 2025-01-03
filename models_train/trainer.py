import torch
import torch.ao.quantization
from torch.optim import Adam, SGD, Adadelta, lr_scheduler, Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.ao.nn import quantized as nq
from pytorch_msssim import ssim
from models_train.IQAdataset import get_data_loaders
# from models_train.IQAmodel import IQAModel
from models_train.Linearity import Linearity
from models_train.IQAloss import IQALoss
from models_train.IQAperformance import IQAPerformanceLinearity, IQAPerfomanceKonCept
from models_train.swap_convs import swap_to_quntized
import models_train.iterative as iterative 
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
from typing import Dict
from tqdm import tqdm
from torch.nn.utils import prune
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize
from torch import nn
import copy
from icecream import ic
# metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']

from _codecs import encode
torch.serialization.add_safe_globals([np._core.multiarray.scalar, 
                                        np.dtype, np.dtypes.Float64DType,
                                        encode])

MAX_SCORE_NORM = 1.0982177
MIN_SCORE_NORM = 0.29622972

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
        self.noise_batch = args.noise
        self.gradnorm_regularization = args.gradnorm_regularization
        self.h_gradnorm_regularization = 6/255 # 10/255
        self.weight_gradnorm_regularization = 1e-1 # 1e-2
        self.cayley = args.cayley
        self.cayley_pool = args.cayley_pool
        self.cayley_pair = args.cayley_pair

        self.pruning = args.pruning
        self.prune_iters = args.prune_iters
        self.width_prune = args.width_prune
        self.height_prune = args.height_prune
        self.images_count_prune = args.pls_images
        self.kernel_prune = args.kernel_prune
        self.use_mask = True if self.pruning > 0 else False

        self.adv = self.args.adv

        self.is_quantize = args.quantize

        # if self.is_quantize:
        #     self.device = 'cpu'

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
                              cayley=self.cayley, cayley_pool=self.cayley_pool, 
                              cayley_pair=self.cayley_pair, quantize=False).to(self.device)

        print(self.model)
        # self.scaler = GradScaler()
        self.k = [1, 1, 1]
        self.b = [0, 0, 0]

        current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        self.writer = SummaryWriter(log_dir='{}/{}-{}'.format(self.args.log_dir, self.args.format_str, current_time))
        # self._optimizer()
        if self.args.quantize:
            self.quantize()

    def train(self, train=True, val=True, test=True):
        if self.adv:
            self._prepair_adv_train()
        self._prepair_train(train, val, test)
        self._train_loop()

    def compute_output(self, inputs, label):
        return self.model(inputs)

    def unpack_data(self, inputs, label, step):
        return inputs, label

    def _prepair(self, train=True, val=True, test=True, use_normalize=True):
        if self.args.quantize:
            self.__quantize()
        if self.pruning > 0.0:
            self.__prune()
        if train or val or test:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args=self.args, 
                                                                                    train=train, 
                                                                                    val=val,
                                                                                    test=test, 
                                                                                    use_normalize=use_normalize,
                                                                                    )

        self.loss_func = IQALoss(loss_type=self.args.loss_type, alpha=self.args.alpha, beta=self.args.beta,
                                p=self.args.p, q=self.args.q,
                                monotonicity_regularization=self.args.monotonicity_regularization,
                                gamma=self.args.gamma, detach=self.args.detach)

    def _prepair_adv_train(self, ):
        checkpoints = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoints['model'])
        self.args.trained_model_file += '-advirsarial'
        self.min_score = checkpoints['min']
        self.max_score = checkpoints['max']
        self.k = checkpoints['k']
        self.b = checkpoints['b']
        print("Orig SROCC, PLCC:", checkpoints['SROCC'], checkpoints['PLCC'])

    def _prepair_train(self, train, val, test):
        self._prepair(train, val, test)
        if self.args.ft_lr_ratio == .0:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # if self.base_model_name=="Linearity":
        self._optimizer()
        self.scaler = GradScaler()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step,
                                                gamma=self.args.lr_decay)
        # elif self.base_model_name=="KonCept":
        #     points = {0: 1e-4, 40: 1e-4/5, 60: 1e-4/10} # 70 epochs
        #     self.scheduler = SimpleLRScheduler(self.optimizer, points)

        # self.scaler = GradScaler()

        self.metric_computer = self._get_perfomance('val', k=[1, 1, 1], b=[0, 0, 0], mapping=True)
        self.best_val_criterion, self.best_epoch = -100, -1

        # self._optimizer()

    def __prune(self):
        # if prune_iter > 0:
        #     form = self.args.trained_model_file[:-1] + str(prune_iter+1) # prune_iter += 1
        # elif prune_iter == 0:
        #     # form = f'{self.args.trained_model_file}+prune={self.args.pruning}{self.args.pruning_type}_lr={self.args.learning_rate}_e={self.args.epochs}_iters={prune_iter+1}'
        #     form = '{}+prune={}{}_lr={}_e={}_iters={}'.format(self.args.trained_model_file, self.args.pruning,
        #                                                       self.args.pruning_type, self.args.learning_rate,
        #                                                       self.args.epochs, prune_iter+1)
        # else:
        #     raise ValueError(f"prune_iter < 0. {prune_iter=}")
        self.model.print_sparcity()
        form = '{}+prune={}{}_lr={}_e={}'.format(self.args.trained_model_file, self.args.pruning,
                                                          self.args.pruning_type, self.args.learning_rate,
                                                          self.args.prune_epochs)

        checkpoint = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.args.trained_model_file = form
        self.model.prune(amount=self.args.pruning, 
                         prtype=self.args.pruning_type,
                         width=self.args.width_prune,
                         height=self.args.height_prune,
                         images_count=self.args.pls_images,
                         kernel=self.args.kernel_prune)
        self.save_checkpoints(checkpoint, self.args.trained_model_file, 
                              use_mask=False)
        # torch.save(checkpoint, self.args.trained_model_file)
        print(self.args.trained_model_file)

        self.epochs = self.args.prune_epochs
        self.model.print_sparcity()

    def __quantize(self, precision=16):
        print(self.args.trained_model_file)
        checkpoint = torch.load(self.args.trained_model_file, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])

        form = '{}+quantize={}'.format(self.args.trained_model_file, 
                                       self.args.quantize,
                                      )
        self.args.trained_model_file = form
        self.model.quantize(precision)
        self.save_checkpoints(checkpoint, self.args.trained_model_file)
        # torch.save(checkpoint, self.args.trained_model_file)
        print("Quantized")

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
                label = [k.to(self.device) for k in label]
                inputs = inputs.to(self.device)

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
            if (val_criterion > self.best_val_criterion) and (self.current_epoch > 10):
                # if self.args.debug:
                # print('max:', 'max_pred', 'min:', 'min_pred')
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'k': self.k,
                    'b': self.b,
                }
                self.save_checkpoints(checkpoint, self.args.trained_model_file, 
                                      use_mask=False)
                # torch.save(checkpoint, self.args.trained_model_file)

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

        ckpt = torch.load(self.args.trained_model_file)
        self.model.load_state_dict(ckpt['model'])
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
        self.save_checkpoints(checkpoint, self.args.trained_model_file, 
                              use_mask=False)
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
        self.save_checkpoints(checkpoint, self.args.trained_model_file, 
                              use_mask=False)
        # torch.save(checkpoint, self.args.trained_model_file)
        # Task.current_task().upload_artifact(name="Metrics", 
        #                                     artifact_object={
        #                                         'model_name': "Linearity",
        #                                         'min': checkpoint['min'],
        #                                         'max': checkpoint['max'],
        #                                         'SROCC': checkpoint['SROCC'],
        #                                         'PLCC': checkpoint['PLCC'],
        #                                         'RMSE': checkpoint['RMSE'],
        #                                     })

    def _train_step(self, inputs, label, step):
        inputs = inputs.to(self.device)

        inputs, label = self.unpack_data(inputs, label, step)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True):
            if self.adv:
                adv_inputs = iterative.fgsm_attack(inputs, 
                                                eps=4.0/255, 
                                                alpha=5.0/255, 
                                                model=self.model, 
                                                metric_range=(self.max_score - self.min_score),
                                                k=self.k, b=self.b)
                ssim_val = ssim(adv_inputs, inputs, data_range=1, size_average=False)
                adv_label = [
                    torch.clamp(
                        label[0].clone()
                        - 1 * (1 - ssim_val) * (self.max_score - self.min_score),
                        self.min_score,
                        self.max_score,
                    ),
                    torch.clamp(
                        label[1].clone()
                        - 1
                        * (1 - ssim_val)
                        * (MAX_SCORE_NORM - MIN_SCORE_NORM),
                        MIN_SCORE_NORM,
                        MAX_SCORE_NORM,
                    ),
                ]
                inputs = torch.concat([adv_inputs, inputs])
                label = [torch.concat([adv_lab, lab]) for adv_lab, lab in zip(adv_label, label)]

            output = self.compute_output(inputs, label)

            loss = self.loss_func(output, label) / self.args.accumulation_steps 

            if self.gradnorm_regularization:
                grad_loss = self.gradnorm_regularize(inputs)
                loss += self.weight_gradnorm_regularization*grad_loss
            ic(loss)
        
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

        print(self.args.trained_model_file)
        checkpoint = torch.load(self.args.trained_model_file, weights_only=True)
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
        checkpoint['PLCC'] = self.metric_computer.PLCC
        checkpoint ['RMSE'] = self.metric_computer.RMSE
        checkpoint['min'] = self.metric_computer.preds.min()
        checkpoint['max'] = self.metric_computer.preds.max()
        metric_range = checkpoint['max'] - checkpoint['min']
        print(checkpoint['min'], checkpoint['max'])
        print(f'Metric_range:', metric_range)
        self.save_checkpoints(checkpoint, self.args.trained_model_file, use_mask=self.use_mask, model=self.model)
        # torch.save(checkpoint, self.args.trained_model_file)
        # Task.current_task().upload_artifact(name="Metrics", 
        #                                     artifact_object={
        #                                         'model_name': "Linearity",
        #                                         'min': checkpoint['min'],
        #                                         'max': checkpoint['max'],
        #                                         'SROCC': checkpoint['SROCC'],
        #                                         'PLCC': checkpoint['PLCC'],
        #                                         'RMSE': checkpoint['RMSE'],
        #                                     })

        print('{}, {}: {:.3f}'.format(self.args.dataset, self.metrics_printed[0], metrics[self.metrics_printed[0]]))
        np.save(self.args.save_result_file, metrics)
        # Task.current_task().upload_artifact(name="np metrics", artifact_object=metrics)

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

    @staticmethod
    def _set_quantized_conv(model):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Conv2d):
                attrs = {
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'dilation': layer.dilation,
                    'groups': layer.groups,
                    'bias': layer.bias is not None,
                    'padding_mode': layer.padding_mode,
                }
                conv = nq.Conv2d(**attrs)
                # conv.weight = layer.weight.data
                # conv.bias = layer.weight.data
                # conv.set_weight_bias(layer.weight.data, layer.bias.data if layer.bias else None)
                setattr(model, name, conv)
            else:
                __class__._set_quantized_conv(layer)

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

    @staticmethod
    def save_checkpoints(checkpoint, trained_model_file, model=None, use_mask=False):
        if use_mask:
            # model_copy = copy.deepcopy(model)
            for module, name in model.prune_parameters:
                prune.remove(module, name)
            checkpoint['model'] = model.state_dict()
            torch.save(checkpoint, trained_model_file)
        else:
            torch.save(checkpoint, trained_model_file)

    @staticmethod
    def get_prune_file_name(args):
        return '{}+prune={}{}_lr={}_e={}_iters={}'.format(args.trained_model_file, args.pruning,
                                                          args.pruning_type, args.learning_rate,
                                                          args.prune_epochs, args.prune_iters)
    
    # @staticmethod
    # def get_quantize_file_name(args):
    #     return '{}+quantize={}'.format(args.trained_model_file, 
    #                                               args.quantize,
    #                                               )

    @classmethod
    def run(cls, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainer = cls(device, args)
        # print(args.evaluate)
        if args.evaluate:
            if args.pruning:
                # form = '{}+prune={}{}_lr={}_e={}_iters={}'.format(args.trained_model_file, args.pruning,
                #                                                 args.pruning_type, args.learning_rate,
                #                                                 args.prune_epochs, args.prune_iters)
                form = cls.get_prune_file_name(args)
                args.trained_model_file = form
            elif args.quantize:
                form = cls.get_quantize_file_name(args)
                args.trained_model_file = form
            trainer.eval()
        # elif args.pruning:
        #     trainer.prune()
        #     trainer.eval()
        # elif args.quantize:
        #     trainer.quantize()
        #     trainer.eval()
        else:
            trainer.train()
            trainer.eval()


# class SimpleLRScheduler:
#     def __init__(
#         self,
#         optimizer: Optimizer,
#         points: Dict[int, float],
#     ):
#         self.optimizer = optimizer
#         self.points = points

#     def step(self, epoch: int):
#         if epoch in self.points:
#             self.lr = self.points[epoch]
#             for param_group in self.optimizer.param_groups:
#                 param_group["lr"] = self.lr

#         print("LR=", self.lr)
#         return self.lr



def dump_scalar_metrics(metrics: Dict, writer: SummaryWriter, phase: str, global_step: int = 0, dataset: str = ''):
    prefix = phase.lower() + (f'_{dataset}' if dataset else '')
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f'{metric_name}/{prefix}',
            metric_value,
            global_step=global_step,
        )
