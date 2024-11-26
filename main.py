from attacker import Attack
import numpy as np
import os
from argparse import ArgumentParser
from icecream import ic
import torch
import yaml

YAML_PATH = './path_config.yaml'

def get_format_string(args):
    form =  f'DBCNN-cayley={args.cayley}'\
            f'-cayley2={args.cayley2}'\
            f'-cayley3={args.cayley3}'\
            f'-cayley4={args.cayley4}'\
            f'-gr={args.gradient_regularization}'\
            f'-activation={args.activation}'
    if args.prune>0:
        form += f'-prune_type={args.prune_type}'\
                f'-prune_amount={args.prune}'\
                f'-prune_epochs={args.prune_epochs}'\
                f'-prune_lr={args.prune_lr}'

    return form + '.pt'


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='koniq-10k',
                        help='dataset: live|csiq|tid2013|livec|mlive|koniq-10k')
    
    parser.add_argument('--prune', dest='prune', type=float,
                        default=0, help='Pruning percentage.')
    parser.add_argument('--prune_epochs', dest='prune_epochs', type=int,
                        default=5, help='Pruning epochs.')
    parser.add_argument('--prune_type', dest='prune_type', type=str,
                        default='l2', help='Pruning type.')
    parser.add_argument('--prune_lr', dest='prune_lr', type=float,
                        default=1e-6, help='Pruning learning rate.')
    
    parser.add_argument('--tune_iters', dest='tune_iters', type=int,
                        default=1, help='Iters for tune')
    parser.add_argument('--iters', dest='iters', type=int,
                        default=1, help='PGD attack iters count.')
    parser.add_argument('--cayley', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley2', action='store_true',
                        help='Use two cayley blocks')
    parser.add_argument('--cayley3', action='store_true',
                        help='Use cayley block')
    parser.add_argument('--cayley4', action='store_true',
                        help='Use two cayley blocks')
    parser.add_argument('--gradient_regularization', '-gr', action='store_true',
                        help='Use gradient regularization')
    parser.add_argument('--activation', default='relu', type=str,
                        help='Use cayley block')
    parser.add_argument("--resize", action="store_true", help="Resize?")
    parser.add_argument(
        "--resize_size_h", default=498, type=int, help="resize_h (default: 498, 384)"
    )
    parser.add_argument(
        "--resize_size_w", default=664, type=int, help="resize_w (default: 664, 512)"
    )
    parser.add_argument('--debug', action='store_true',
                        help='DEBUG')
    args = parser.parse_args()
    options = {
        'fc': True,
        'cayley': args.cayley,
        'cayley2': args.cayley2,
        'cayley3': args.cayley3,
        'cayley4': args.cayley4,
        'backbone': 'VGG-16',
        'model': 'DBCNN',
        'prune': args.prune,
        'prune_type': args.prune_type,
        'activation': args.activation,
        'gradnorm_regularization': args.gradient_regularization,
        'resize': args.resize,
        'crop': False,
        'height': args.resize_size_h,
        'width': args.resize_size_w,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    path = {
        'koniq-10k': os.path.join('dataset', 'KonIQ-10k'),
        'nips': os.path.join('dataset', 'NIPS'),
        'csv_results_dir': './rs',

        'live': os.path.join('dataset','databaserelease2'),
        'csiq': os.path.join('dataset','CSIQ'),
        'tid2013': os.path.join('dataset','TID2013'),
        'livec': os.path.join('dataset','ChallengeDB_release'),
        'mlive': os.path.join('dataset','LIVEmultidistortiondatabase'),
        'fc_model': os.path.join('fc_models'),
        'scnn_root': os.path.join('pretrained_scnn','scnn.pkl'),
        'fc_root': os.path.join('fc_models','net_params_best.pkl'),
        'db_model': os.path.join('db_models'),
    }

    ic.disable()


    with open(YAML_PATH, 'r') as file:
        yaml_conf = yaml.safe_load(file)

    exec_: Attack = Attack(path, options, device='cuda')

    exec_.load_checkpoints(checkpoints_path=os.path.join(yaml_conf['save']['ckpt'], get_format_string(args)))

    datasets = yaml_conf['dataset']['data']
    datasets = {"KonIQ-10k": "./dataset/KonIQ-10k"}
    # datasets = {"NIPS": "./dataset/NIPS"}
    data_info = yaml_conf['dataset']['labels']
    save_results_dir = yaml_conf['save']['results']
    
    result = [None, None]
    for i, (dataset, datset_path) in enumerate(datasets.items()):
        exec_.set_load_conf(
                        dataset=dataset,
                        dataset_path=datset_path,
                        resize=options['resize'],
                        crop=options['crop'],
                        resize_size_h=options['height'],
                        resize_size_w=options['width'],
                        data_info=data_info['KonIQ-10k'])
    
        exec_.attack(attack_type='PGD',
                    iterations=args.iters, debug=args.debug)

        exec_.save_results(path['csv_results_dir'])
        exec_.save_vals_to_file(csv_results_dir=save_results_dir)
        
        result[i] = np.array(exec_.res).mean()
        print(result[i])

    print(result)