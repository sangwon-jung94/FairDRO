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
    parser.add_argument('--val', default=False, action='store_true')
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--evalset', default='all', choices=['all', 'train', 'test'])

    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas','utkface', 'celeba', 'jigsaw'])
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio for cifar-10s')
    parser.add_argument('--img-size', default=176, type=int, help='img size for preprocessing')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--max-grad-norm', default=1, type=float, help='size for clip grad')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20200101', type=str, help='experiment date')
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'fairinf', 'fairret','lbc','mfd', 'fairhsic','gdro','fairbatch','fairdro','cov','rw','renyi', 'rvp','egr','pl', 'direct_reg', 'fairdro_wo_c'])

    parser.add_argument('--optim', default='Adam', type=str, required=False,
                        choices=['AdamP', 'AdamW','SGD', 'SGD_momentum_decay', 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--sam', default=False, action='store_true', help='sam')
    parser.add_argument('--lamb', default=1, type=float, help='fairness strength')
    parser.add_argument('--model', default='', required=True, choices=['resnet12', 'resnet50','cifar_net', 'resnet34', 'resnet18', 'resnet101','mlp', 'resnet18_dropout', 'bert','lr'])
    parser.add_argument('--teamodel', default='', choices=['resnet12', 'resnet50', 'resnet34', 'resnet18', 'resnet101','mlp'])        
    parser.add_argument('--teacher-type', default=None, choices=['mlp','resnet12','bert','resnet18', 'resnet34', 'resnet50', 'mobilenet', 'shufflenet', 'cifar_net', 'None'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--n-workers', default=1, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--add-attr', default=None, help='additional group attribute for celeba')

    parser.add_argument('--eta', default=0.001, type=float, help='adversary training learning rate or lr for reweighting')

    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly'], help='kernel for mmd')
    parser.add_argument('--balSampling', default=False, action='store_true', help='balSampling loader')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')
    parser.add_argument('--record', default=False, action='store_true', help='record')
    parser.add_argument('--analysis', default=False, action='store_true', help='analysis')
    parser.add_argument('--uc', default=False, action='store_true', help='uncertain')
    
    parser.add_argument('--fairness-criterion', default='dca', type=str, choices=['eo', 'dp', 'eopp', 'ap','dca'], help='fairness criterion')
    
    # For reweighting,
    parser.add_argument('--iteration', default=10, type=int, help='iteration for reweighting')
    
    # For lgdro chi,
    parser.add_argument('--kd', default=False, action='store_true', help='kd')
    parser.add_argument('--rho', default=0.5, type=float, help='uncertainty box length')
    parser.add_argument('--use-01loss', default=False, action='store_true', help='using 0-1 loss when updating q')
    parser.add_argument('--gamma', default=0.1, type=float, help='learning rate for q')
    parser.add_argument('--optim-q', default='pd', choices=['pd', 'ibr', 'smt_ibr'], help='the type of optimization for q')
    parser.add_argument('--q-decay', default='linear', type=str, help='the type of optimization for q')
    parser.add_argument('--label-flipped', default=False, action='store_true', help='flip a label when the corresponding q has a negative value')
    parser.add_argument('--rholr', default=0.001, type=float, help='learning rate of lambda')        

    # For fairInf,
    parser.add_argument('--alpha', default=0.02, type=float, help='uncertainty box length')    
    parser.add_argument('--beta', default=0.5, type=float, help='uncertainty box length')
   
    # For exp_grad_reduction,
    parser.add_argument('--bound_B', default=0.01, type=float, help='bound for L1 norm')
    parser.add_argument('--constraint_c', default=0.0, type=float, help='bound for constraint c')
    
    # For cotter,
    parser.add_argument('--lamblr', default=0.001, type=float, help='learning rate of lambda')    
    parser.add_argument('--epsilon', default=0.01, type=float, help='constraint penalty')
    
    # balanced cross entropy
    parser.add_argument('--balanced', default=False, action='store_true', help='whether use a balanced acc')
    
    args = parser.parse_args()
    args.cuda=True
    if args.mode == 'train' and args.method == 'mfd':
        if args.teacher_type is None:
            raise Exception('A teacher model needs to be specified for distillation')
        elif args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    return args
