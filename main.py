import numpy as np
import argparse
import models_train
from torchvision.transforms.functional import resize, to_tensor, normalize
from attacker import Attack
import os

from icecream import ic
import yaml
# from clearml import Task, Logger

EPS = 1e-6
YAML_PATH = './path_config.yaml'

def get_format_string(args):
    format_str = 'activation={}-{}-{}-bs={}-loss=norm-in-norm-p=1.0-q=2.0-detach-False-KonIQ-10k-res={}-{}x{}' \
        .format(
        args.activation,
        args.model,
        args.architecture,
        args.batch_size,
        True,#args.resize,
        args.resize_size_h,
        args.resize_size_w, #args.mixup, args.mixup_gamma,
        
    )

    if args.gradnorm_regularization:
        format_str += f'-gr={args.gradnorm_regularization}'
    if args.adv:
        format_str += f'-advirsarial'
    if args.cayley:
        format_str += f'-cl={args.cayley}'
    if args.cayley_pool:
        format_str += f'-clp={args.cayley_pool}'
    if args.cayley_pair:
        format_str += f'-cp={args.cayley_pair}'
    if args.gabor:
        format_str += f'-gabor=True'
    if args.noise:
        format_str += '-noise=True'
    if args.quantize:
        format_str += '-quantize16=True'
    # format_str += f'+prune={args.pruning}{args.pruning_type}_lr=1e-06_e={args.prune_epochs}_iters={args.prune_iters}' if args.pruning else ''
    format_str += f'+prune={args.pruning}{args.pruning_type}_lr=1e-06_e={args.prune_epochs}' if args.pruning else ''
    return format_str

def run(args):

    exec_: Attack = Attack(args.model,
                           arch=args.architecture, pool=args.pool,
                           use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7,
                           activation=args.activation,
                           device=args.device, pruning=args.pruning, t_prune=args.pruning_type, gabor=args.gabor,
                           gradnorm_regularization=args.gradnorm_regularization, adv=args.adv,
                           cayley=args.cayley, cayley_pool=args.cayley_pool, cayley_pair=args.cayley_pair,
                           quantize=args.quantize
                           )

    exec_.load_checkpoints(checkpoints_path=args.trained_model_file)

    with open(YAML_PATH, 'r') as file:
        yaml_conf = yaml.safe_load(file)
    datasets = yaml_conf['dataset']['data']
    # datasets = {"KonIQ-10k": "KonIQ-10k/1024x768"}
    # datasets = {"NIPS": "NIPS_test"}
    data_info = yaml_conf['dataset']['labels']
    save_results_dir = yaml_conf['save']['results']
    
    result = [None, None]
    for i, (dataset, datset_path) in enumerate(datasets.items()):
        exec_.set_load_conf(dataset=dataset, 
                            dataset_path=datset_path,
                            resize=args.resize,
                            crop=args.crop,
                            resize_size_h=args.resize_size_h,
                            resize_size_w=args.resize_size_w,
                            data_info=data_info['KonIQ-10k'])

        exec_.attack(attack_type=args.attack_type,
                    iterations=args.iterations, debug=args.debug)

        exec_.save_results(args.csv_results_dir)
        exec_.save_vals_to_file(csv_results_dir=save_results_dir)
        
        result[i] = np.array(exec_.res).mean()
        print(result[i])
    return result


if __name__ == "__main__":
    # task = Task.init(project_name="Test", task_name="Linearity", reuse_last_task_id=False)

    parser = argparse.ArgumentParser(description="Test Demo for LinearityIQA")

    parser.add_argument(
        "--architecture", "-arch",
        default="resnet101",
        type=str,
        help="arch name (default: resnet101) vonenet50|resnet50|wideresnet50|resnet18|resnet34|resnext101_32x8d",
    )
    parser.add_argument(
        "--pool", default="avg", type=str, help="pool method (default: avg)"
    )
    parser.add_argument(
        "--use_bn_end", action="store_true", help="Use bn at the end of the output?"
    )
    parser.add_argument("--P6", type=int, default=1, help="P6 (default: 1)")
    parser.add_argument("--P7", type=int, default=1, help="P7 (default: 1)")

    parser.add_argument(
        "--trained_model_file",
        default="LinearityIQA/checkpoints/p1q2.pth",
        type=str,
        help="trained_model_file",
    )

    parser.add_argument("--resize", action="store_true", help="Resize?")
    parser.add_argument(
        "--resize_size_h", default=498, type=int, help="resize_h (default: 498, 384)"
    )
    parser.add_argument(
        "--resize_size_w", default=664, type=int, help="resize_w (default: 664, 512)"
    )

    parser.add_argument(
        "--batch_size", '-bs', default=8, type=int, help="resize_w (default: 664, 512)"
    )

    parser.add_argument("-iter", "--iterations", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--attack_type", type=str, default="PGD")
    # parser.add_argument("--dataset", type=str, default="NIPS", help="KonIQ-10 | NIPS")
    # parser.add_argument("--dataset_path", type=str, default=None, help="./NIPS_test/ | ./KonIQ-10k/")
    parser.add_argument("--data_info", type=str, default=None, help="./data/KonIQ-10kinfo.mat")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv_results_dir", type=str, default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-se", "--squeeze_excitation", action="store_true")
    # parser.add_argument("-weights", "--checkpoints_dir", type=str, default='weights')

    parser.add_argument('-prune', "--pruning", type=float, help="adversarial pruning percent")
    parser.add_argument('-t_prune', "--pruning_type", type=str, default='pls')  # displs, pls, l1, l2
    parser.add_argument('--prune_epochs', type=int, default=5)
    parser.add_argument('--prune_iters', type=int, default=1)

    parser.add_argument('-quant', '--quantize', action='store_true', help="Use quantization.")

    parser.add_argument('--model', default='Linearity', type=str)
    parser.add_argument('--dlayer', default=None, type=str) # d1, d2
    parser.add_argument('--gabor', action='store_true', help="Chage convs to gabor layer")
    parser.add_argument('--noise', action='store_true', help="Use normal noise on batch")
    parser.add_argument('-gr', '--gradnorm_regularization', action='store_true', help="Use gradient-norm regularization")
    parser.add_argument('-cl', '--cayley', action='store_true', help="Use cayley block with conv")
    parser.add_argument('-clp', '--cayley_pool', action='store_true', help="Use cayley block with pooling")
    parser.add_argument('-cp', '--cayley_pair', action='store_true', help="Use cayley block after conv4, conv5")
    parser.add_argument('--crop', action='store_true', help='Use crop for image')

    parser.add_argument('--adversarial', '-adv', dest='adv', action='store_true')
    args = parser.parse_args()
    # task.connect(vars(args))

    print(args.architecture, args.pruning)

    args.format_str = get_format_string(args)

    if args.debug:
        ic.enable()
        print("DEBUG")
    else:
        ic.disable()

    print("Device: ", args.device)
    with open(YAML_PATH, 'r') as file:
        yaml_file = yaml.safe_load(file)
    checkpints_path = yaml_file['save']['ckpt']
    # checkpints_path = "Linearity-ckpt"
    args.trained_model_file = os.path.join(checkpints_path, args.format_str)
    print(args.trained_model_file)
    total_score = run(args)
    if total_score[0] is not None and total_score[1] is not None:
        print(
            "Result for {} type attack: {:.4f}, {:.4f}".format(
                args.attack_type.capitalize(), total_score[0], total_score[1]
            )
        )
