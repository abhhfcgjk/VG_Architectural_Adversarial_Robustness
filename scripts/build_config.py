from argparse import ArgumentParser
import yaml

def run(args):
    if args.test:
        with open(args.config_dir + '/default_test.yaml', 'r') as file:
            config = yaml.safe_load(file)
    else:
        with open(args.config_dir + '/default_train.yaml', 'r') as file:
            config = yaml.safe_load(file)

    config['activation'] = args.activation
    config['options']['activation'] = args.activation
    config['options']['backbone'] = args.backbone
    config['options']['prune']=args.prune
    config['options']['prune_lr'] = args.prune_lr
    config['options']['prune_epochs'] = args.prune_epochs
    config['options']['prune_type'] = args.prune_type
    config['options']['cayley1'] = args.cayley1
    config['options']['cayley2'] = args.cayley2
    config['eval_only'] = args.test
    config['train']['gr'] = args.gr

    if args.adv:
        config['attack']['train']['type'] = 'fgsm'
        config['attack']['train']['params'] = {'eps': 4., 'alpha': 5., 'mode': 'uniform'}
        config['train']['epochs'] = 30

    if config['options']['prune'] > 0:
        config['train']['epochs'] = args.prune_epochs

    if args.test:
        config['eval_only'] = True

    for i in range(len(config['attack']['test'])):
        config['attack']['test'][i]['params']['iters'] = args.iters
    

    activation_status = f'-activation|{args.activation}'
    backbone_satatus = f'-backbone|{args.backbone}'
    prune_status = f'-prune={args.prune}{args.prune_type}' if args.prune > 0.0 else ''
    gr_status = f'-gr' if args.gr else ''
    adv_status = f'-adv' if args.adv else ''
    cayley1_status = '-cayley1' if args.cayley1 else ''
    cayley2_status = '-cayley2' if args.cayley2 else ''
    test_status = 'test' if args.test else ''

    config['attack']['path']['checkpoints'] = 'db_model/koncept{}{}{}{}{}{}{}.pth'.format(
                                                                        backbone_satatus, 
                                                                        cayley1_status,
                                                                        cayley2_status,
                                                                        adv_status,
                                                                        gr_status, prune_status,
                                                                        activation_status,
                                                                        )
    form = '{}config{}{}{}{}{}{}{}_PGD={}.yaml'.format(test_status,
                                               backbone_satatus, 
                                               cayley1_status,
                                               cayley2_status,
                                               gr_status, 
                                               adv_status, 
                                               prune_status,
                                               activation_status,
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
    parser.add_argument('--prune', type=float, default=0.0)
    parser.add_argument('--prune_lr', type=float, default=0.0001)
    parser.add_argument('--prune_epochs', type=int, default=15)
    parser.add_argument('--prune_type', type=str, default='l1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--gr', action='store_true')
    parser.add_argument('--adv', action='store_true')

    args = parser.parse_args()
    print(run(args))