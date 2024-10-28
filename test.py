import numpy as np
import argparse
from torchvision.transforms.functional import resize, to_tensor, normalize
from attacker import Attack
import os

from icecream import ic
import yaml
from clearml import Task, Logger

EPS = 1e-6
YAML_PATH = './path_config.yaml'

def get_format_string(args):
    format_str = 'activation={}-{}-{}-bs={}-KonIQ-10k-res={}-{}x{}' \
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
    if args.cayley:
        format_str += f'-cl={args.cayley}'
    if args.cayley_pool:
        format_str += f'-clp={args.cayley_pool}'
    if args.cayley_pair:
        format_str += f'-cp={args.cayley_pair}'

    format_str += f'+prune={args.pruning}{args.pruning_type}_lr=1e-06_e={args.prune_epochs}_iters={args.prune_iters}' if args.pruning else ''
    return format_str

def run(args):

    exec_: Attack = Attack(args.model,
                           arch=args.architecture,
                           activation=args.activation,
                           device=args.device, pruning=args.pruning, t_prune=args.pruning_type,
                           gradnorm_regularization=args.gradnorm_regularization,
                           cayley=args.cayley, cayley_pool=args.cayley_pool, cayley_pair=args.cayley_pair
                           )

    exec_.load_checkpoints(checkpoints_path=args.trained_model_file)

    with open(YAML_PATH, 'r') as file:
        yaml_conf = yaml.safe_load(file)
    datasets = yaml_conf['dataset']['data']
    datasets = {"KonIQ-10k": "./KonIQ-10k"}
    # datasets = {"NIPS": "./NIPS_test"}
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
                     iterations=args.iterations, 
                     debug=args.debug)

        exec_.save_results(args.csv_results_dir)
        exec_.save_vals_to_file(csv_results_dir=save_results_dir)
        
        result[i] = np.array(exec_.res).mean()
        print(result[i])
    return result


if __name__ == "__main__":
    task = Task.init(project_name="Test", task_name="KonCept", reuse_last_task_id=False)

    parser = argparse.ArgumentParser(description="Test Demo for KonCept IQA")

    parser.add_argument('--model', default='KonCept', type=str)
    parser.add_argument(
        "--architecture", "-arch",
        default="inceptionresnetv2",
        type=str,
        help="arch name (default: inceptionresnetv2)",
    )

    parser.add_argument(
        "--trained_model_file",
        default="./KonCept.pt",
        type=str,
        help="trained_model_file",
    )

    parser.add_argument("--resize", action="store_true", help="Resize?")
    parser.add_argument(
        "--resize_size_h", default=384, type=int, help="resize_h (default: 512, 384)"
    )
    parser.add_argument(
        "--resize_size_w", default=512, type=int, help="resize_w (default: 512, 384)"
    )

    parser.add_argument(
        "--batch_size", '-bs', default=16, type=int, help="batch size"
    )

    parser.add_argument("-iter", "--iterations", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--attack_type", type=str, default="PGD")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv_results_dir", type=str, default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('-prune', "--pruning", type=float, help="adversarial pruning percent")
    parser.add_argument('-t_prune', "--pruning_type", type=str, default='pls')  # pls, l1, l2
    parser.add_argument('--prune_epochs', type=int, default=5)
    parser.add_argument('--prune_iters', type=int, default=1)
    parser.add_argument('-gr', '--gradnorm_regularization', action='store_true', help="Use gradient-norm regularization")
    parser.add_argument('-cl', '--cayley', action='store_true', help="Use cayley block with conv")
    parser.add_argument('-clp', '--cayley_pool', action='store_true', help="Use cayley block with pooling")
    parser.add_argument('-cp', '--cayley_pair', action='store_true', help="Use cayley block after conv4, conv5")
    parser.add_argument('--crop', action='store_true', help='Use crop for image')

    args = parser.parse_args()
    task.connect(vars(args))

    print(args.architecture, args.pruning)

    args.format_str = get_format_string(args) + '.pt'

    if args.debug:
        ic.enable()
        print("DEBUG")
    else:
        ic.disable()

    print("Device: ", args.device)
    checkpints_path = "KonCept-ckpt"
    args.trained_model_file = os.path.join(checkpints_path, args.format_str)
    print(args.trained_model_file)
    total_score = run(args)
    if total_score[0] is not None and total_score[1] is not None:
        print(
            "Result for {} type attack: {:.4f}, {:.4f}".format(
                args.attack_type.capitalize(), total_score[0], total_score[1]
            )
        )
