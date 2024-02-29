# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2020/1/14

import torch
# from torch.optim import Adam, SGD, Adadelta, lr_scheduler
# from apex import amp
from IQAdataset import get_data_loaders
# from IQAmodel import IQAModel
# from IQAloss import IQALoss
# from IQAperformance import IQAPerformance
# from tensorboardX import SummaryWriter
# import datetime
import os
import numpy as np
import random
from argparse import ArgumentParser

# from torch.cuda import amp

# from activ import ReLU_to_SILU, ReLU_to_ReLUSiLU
from train import Trainer


metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']
# scaler = amp.grad_scaler.GradScaler()
def writer_add_scalar(writer, status, dataset, scalars, iter):
    for metric_print in metrics_printed:
        writer.add_scalar('{}/{}/{}'.format(status, dataset, metric_print), scalars[metric_print], iter)


def run(args):
    Trainer.run(args)
    # trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    # t = [torch.tensor([66.4211, 15.2294, 61.0280, 72.7750, 50.5682, 73.8152, 31.2712, 71.9091],device='cuda:0'),
    #      torch.tensor([0.5375, 0.6159, 0.5664, 0.6331, 0.5235, 0.5143, 0.5963, 0.5326],device='cuda:0')]
    # print(max(t[0]).item())
    # quit()
    parser = ArgumentParser(description='Norm-in-Norm Loss with Faster Convergence and Better Performance for Image Quality Assessment')
    parser.add_argument("--activation", default='relu',
                        help='activation function')
    
    parser.add_argument("--seed", type=int, default=19920517)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('-flr', '--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-accum', '--accumulation_steps', type=int, default=1,
                        help='accumulation_steps for training (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.1,
                        help='lr decay (default: 0.1)')
    parser.add_argument('-olrd', '--overall_lr_decay', type=float, default=0.01,
                        help='overall lr decay (default: 0.01)')
    parser.add_argument('-optl', '--opt_level', default='O1', type=str,
                        help='opt_level for amp (default: O1)')
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-valc', '--val_criterion', default='SROCC', type=str,
                        help='val_criterion: SROCC or PLCC (default: SROCC)') # If using RMSE, minor modification should be made, i.e., 

    parser.add_argument('-a', '--alpha', nargs=2, type=float, default=[1, 0],
                        help='loss coefficient alpha in total loss (default: [1, 0])')
    parser.add_argument('-b', '--beta', nargs=3, type=float, default=[.1, .1, 1],
                        help='loss coefficients for level 6, 7, and 6+7 (default: [.1, .1, 1])')

    parser.add_argument('-arch', '--architecture', default='resnext101_32x8d', type=str,
                        help='arch name (default: resnext101_32x8d)')
    parser.add_argument('-pl', '--pool', default='avg', type=str,
                        help='pool method (default: avg)')
    parser.add_argument('-ubne', '--use_bn_end', action='store_true',
                        help='Use bn at the end of the output?')
    parser.add_argument('-P6', '--P6', type=int, default=1,
                        help='P6 (default: 1)')
    parser.add_argument('-P7', '--P7', type=int, default=1,
                        help='P7 (default: 1)')
    parser.add_argument('-lt', '--loss_type', default='norm-in-norm', type=str,
                        help='loss type (default: norm-in-norm)')
    parser.add_argument('-p', '--p', type=float, default=1,
                        help='p (default: 1)')
    parser.add_argument('-q', '--q', type=float, default=2,
                        help='q (default: 2)')
    parser.add_argument('-detach', '--detach', action='store_true',
                        help='Detach in loss?')
    parser.add_argument('-monoreg', '--monotonicity_regularization', action='store_true',
                        help='use monotonicity_regularization?')
    parser.add_argument('-g', '--gamma', type=float, default=0.1,
                        help='coefficient of monotonicity regularization (default: 0.1)')

    parser.add_argument('-ds', '--dataset', default='KonIQ-10k', type=str,
                        help='dataset name (default: KonIQ-10k)')
    parser.add_argument('-eid', '--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.6,
                        help='train ratio (default: 0.6)')
    parser.add_argument('-tvr', '--train_and_val_ratio', type=float, default=0.8,
                        help='train_and_val_ratio (default: 0.8)')

    parser.add_argument('-rs', '--resize', action='store_true',
                        help='Resize?')
    parser.add_argument('-rs_h', '--resize_size_h', default=498, type=int,
                        help='resize_size_h (default: 498)')
    parser.add_argument('-rs_w', '--resize_size_w', default=664, type=int,
                        help='resize_size_w (default: 664)')

    parser.add_argument('-augment', '--augmentation', action='store_true',
                        help='Data augmentation?')
    parser.add_argument('-ag', '--angle', default=2, type=float,
                        help='angle (default: 2)')
    parser.add_argument('-cs_h', '--crop_size_h', default=498, type=int,
                        help='crop_size_h (default: 498)')
    parser.add_argument('-cs_w', '--crop_size_w', default=498, type=int,
                        help='crop_size_w (default: 498)')
    parser.add_argument('-hp', '--hflip_p', default=0.5, type=float,
                        help='hfilp_p (default: 0.5)')

    parser.add_argument('-logd', "--log_dir", type=str, default="runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('-tdt', '--test_during_training', action='store_true',
                        help='test_during_training?')  # It is better to re-make a train_loader_for_evaluation so as not to disturb the random number generator.
    parser.add_argument('-eval', '--evaluate', action='store_true',
                        help='Evaluate only?')
    parser.add_argument('-se', '--squeeze_excitation', action='store_true')

    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Debug the training by reducing dataflow to 5 batches')
    parser.add_argument('-pbar', '--pbar', action='store_true',
                        help='Use progressbar for the training')

    args = parser.parse_args()
    if args.lr_decay == 1 or args.epochs < 3:  # no lr decay
        args.lr_decay_step = args.epochs
    else:  # 
        args.lr_decay_step = int(args.epochs/(1+np.log(args.overall_lr_decay)/np.log(args.lr_decay)))

    # KonIQ-10k that train-val-test split provided by the owner
    if args.dataset == 'KonIQ-10k':
        args.train_ratio = 7058/10073
        args.train_and_val_ratio = 8058/10073
        if not args.resize:
            args.resize_size_h = 768
            args.resize_size_w = 1024

    if args.beta[1] + args.beta[-1] == .0:
        args.val_criterion = 'SROCC1'
    if args.beta[0] + args.beta[-1] == .0:
        args.val_criterion = 'SROCC2'

    args.im_dirs = {'KonIQ-10k': 'KonIQ-10k',  
                    'CLIVE': 'CLIVE' 
                    }  # ln -s database_path xxx
    args.data_info = {'KonIQ-10k': './data/KonIQ-10kinfo.mat',
                      'CLIVE': './data/CLIVEinfo.mat'}

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    args.format_str = 'activation={}-{}-loss={}-p={}-q={}-detach-{}-{}-res={}-{}x{}-se={}-aug={}-lr={}-bs={}-e={}-opt_level={}'\
                      .format(args.activation, args.architecture, args.loss_type, args.p, args.q, args.detach, 
                              args.dataset, args.resize, args.resize_size_h, args.resize_size_w,args.squeeze_excitation, args.augmentation, 
                              args.learning_rate, args.batch_size, args.epochs, args.opt_level)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/' + args.format_str
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/' + args.format_str
    print(args)
    run(args)
