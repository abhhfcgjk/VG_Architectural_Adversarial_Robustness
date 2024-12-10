from argparse import ArgumentParser
import yaml

def run(args):
    with open(args.config_dir + '/default_test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['activation'] = args.activation
    config['options']['activation'] = args.activation
    config['options']['backbone'] = args.backbone
    config['options']['prune']=args.prune
    config['options']['prune_lr'] = args.prune_lr
    config['options']['prune_epochs'] = args.prune_epochs
    config['options']['prune_type'] = args.prune_type
    config['options']['nt'] = args.nt
    config['eval_only'] = args.test

    if args.test:
        config['eval_only'] = True

    for i in range(len(config['attack']['test'])):
        config['attack']['test'][i]['params']['iters'] = args.iters
    config['attack']['path']['checkpoints'] = f'/home/maindev/28i_mel/mnt/koncept-activation={args.activation}.pth'

    activation_status = f'-activation|{args.activation}'
    backbone_satatus = f'-backbone|{args.backbone}'
    prune_status = f'-prune={args.prune}{args.prune_type}' if args.prune > 0.0 else ''
    nt_status = f'-nt' if args.nt else ''

    form = 'config{}{}{}{}_PGD={}.yaml'.format(activation_status, 
                                        backbone_satatus, 
                                        nt_status, prune_status,
                                        args.iters)

    with open(form , 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return form

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='presets')
    # parser.add_argument('--config_file', type=str, default='default_train.yaml')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--backbone', type=str, default='inceptionresnetv2')
    parser.add_argument('--cayley1', action='store_true')
    parser.add_argument('--cayley2', action='store_true')
    parser.add_argument('--cayley3', action='store_true')
    parser.add_argument('--cayley4', action='store_true')
    parser.add_argument('--nt', action='store_true')
    parser.add_argument('--prune', type=float, default=0.0)
    parser.add_argument('--prune_lr', type=float, default=0.00001)
    parser.add_argument('--prune_epochs', type=int, default=15)
    parser.add_argument('--prune_type', type=str, default='l1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--iters', type=int, default=1)

    args = parser.parse_args()
    print(run(args))