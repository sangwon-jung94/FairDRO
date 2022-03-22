import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--result-dir', default='./results/',
                        help='directory to save results (default: ./results/)')
    parser.add_argument('--log-dir', default='./logs/',
                        help='directory to save logs (default: ./logs/)')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--t-device', default=0, type=int, help='teacher cuda device number')
    
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--evalset', default='test', choices=['all', 'train', 'test'])

    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas','utkface', 'celeba', 'cifar10s','cifar100s', 'waterbird','utkface_fairface'])
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio for cifar-10s')
    parser.add_argument('--img-size', default=176, type=int, help='img size for preprocessing')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20200101', type=str, help='experiment date')
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'lbc','mfd', 'adv', 'fairhsic', 'lgdro','gdro', 'lgdro_chi'])

    parser.add_argument('--optim', default='Adam', type=str, required=False,
                        choices=['AdamP', 'AdamW','SGD', 'SGD_momentum_decay', 'Adam'],
                        help='(default=%(default)s)')

    parser.add_argument('--lamb', default=1, type=float, help='fairness strength')
    parser.add_argument('--gamma', default=0.1, type=float, help='fairness strength')
    parser.add_argument('--model', default='', required=True, choices=['resnet12', 'resnet50', 'resnet34', 'resnet18', 'resnet101','mlp', 'resnet18_dropout', 'bert'])
    parser.add_argument('--teamodel', default='', choices=['resnet12', 'resnet50', 'resnet34', 'resnet18', 'resnet101','mlp'])    
    
    parser.add_argument('--teacher-type', default=None, choices=['mlp','resnet12','resnet18', 'resnet34', 'resnet50', 'mobilenet', 'shufflenet', 'cifar_net', 'None'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--add-attr', default=None, help='additional group attribute for celeba')

    parser.add_argument('--eta', default=0.001, type=float, help='adversary training learning rate or lr for reweighting')

    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly'], help='kernel for mmd')
    parser.add_argument('--labelwise', default=False, action='store_true', help='labelwise loader')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')
    parser.add_argument('--record', default=False, action='store_true', help='record')

    # For reweighting,
    parser.add_argument('--reweighting-target-criterion', default='eo', type=str, help='fairness criterion')
    parser.add_argument('--iteration', default=10, type=int, help='iteration for reweighting')
    
    # For ldro chi,
    parser.add_argument('--kd', default=False, action='store_true', help='kd')
    parser.add_argument('--rho', default=0.5, type=float, help='uncertainty box length')
    parser.add_argument('--ibr', default=False, action='store_true', help='iterated best response')
    args = parser.parse_args()
    args.cuda=True
    if args.mode == 'train' and args.method == 'mfd':
        if args.teacher_type is None:
            raise Exception('A teacher model needs to be specified for distillation')
        elif args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    return args
