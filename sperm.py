from models_train.IQAmodel import IQAModel
from models_train.IQAdataset import get_data_loaders
import torch
import numpy as np
from argparse import ArgumentParser
import os
import yaml
import scipy
from tqdm import tqdm

YAML_PATH = './path_config.yaml'

if __name__=='__main__':
    parser = ArgumentParser(
        description='Norm-in-Norm Loss with Faster Convergence and Better Performance for Image Quality Assessment')
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
                        help='val_criterion: SROCC or PLCC (default: SROCC)')  # If using RMSE, minor modification should be made, i.e.,

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
    parser.add_argument('-eval', '--evaluate', action='store_true', default=True,
                        help='Evaluate only?')

    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Debug the training by reducing dataflow to 5 batches')
    parser.add_argument('-pbar', '--pbar', action='store_true',
                        help='Use progressbar for the training')

    parser.add_argument('-prune', "--pruning", type=float,
                        help="adversarial pruning percent")
    parser.add_argument('-t_prune', "--pruning_type", type=str, default='pls')  # pls, l1, l2
    parser.add_argument('--prune_iters', type=int, default=1)
    parser.add_argument('--width_prune', type=int, default=120)
    parser.add_argument('--height_prune', type=int, default=90)
    parser.add_argument('--images_count_prune', type=int, default=50)
    parser.add_argument('--kernel_prune', type=int, default=1)

    parser.add_argument('--model', default='Linearity', type=str)    

    parser.add_argument('--colab', action='store_true', help="Train in colab")
    args = parser.parse_args()
    with open(YAML_PATH, 'r') as file:
        yaml_file = yaml.safe_load(file)
    default_dir = yaml_file['dataset']['data']['KonIQ']

    args.im_dirs = {'KonIQ-10k': COLAB_dir if args.colab else default_dir,
                    'CLIVE': 'CLIVE'
                    }  # ln -s database_path xxx

    COLAB_path = "./VG_Architectural_Adversarial_Robustness/LinearityIQA/data/KonIQ-10kinfo.mat"
    default_path = yaml_file['dataset']['labels']['KonIQ']
    args.data_info = {'KonIQ-10k': COLAB_path if args.colab else default_path,
                      'CLIVE': './data/CLIVEinfo.mat'}

    server_mnt = "~/mnt/dione/28i_mel"
    destination_path = os.path.expanduser(server_mnt)
    path = os.path.join(destination_path, "activation=relu-Linearity-resnet101-bs=8-loss=norm-in-norm-p=1.0-q=2.0-detach-False-KonIQ-10k-res=True-498x664+prune=0.05pls_lr=1e-06_e=5_iters=2")
    ckpt = torch.load(path)

    model = IQAModel('Linearity', arch='resnet101')
    model.load_state_dict(ckpt['model'])
    k = ckpt['k']
    b = ckpt['b']

    _, _, test_loader = get_data_loaders(args, train=False, val=True, test=True)

    labels = None
    outputs = None
    model.eval()

    for step, (input, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # print(input.shape)
        # print(label)
        output = model(input)[-1]*k[0] + b[0]
        output = output.detach().numpy()
        if labels is None and outputs is None:
            labels = [label[0]]
            outputs = output
        else:
            labels = np.hstack((labels, [label[0]]))
            outputs = np.hstack((outputs, output))

    print(outputs.shape, labels.shape)
    srcc = scipy.stats.spearmanr(outputs[0], labels[0])
    print(srcc)


    




