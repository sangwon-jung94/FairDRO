import torch
import numpy as np
import random
import os
import torch.nn.functional as F

def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_accuracy(outputs, labels, binary=False, reduction='mean'):
    #if multi-label classification
    if len(labels.size())>1:
        outputs = (outputs>0.0).float()
        correct = ((outputs==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    
    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
        
    c = (predictions == labels).float().squeeze()
    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")

class FitnetRegressor(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(FitnetRegressor, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        #self.regressor = nn.Linear(in_feature, out_feature, bias=False)
        self.regressor = torch.nn.Conv2d(in_feature, out_feature, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.regressor.weight, mode='fan_out', nonlinearity='relu')
        self.regressor.weight.data.uniform_(-0.005, 0.005)
#         self.bn = torch.nn.BatchNorm2d(out_feature) 
#         torch.nn.init.ones_(self.bn.weight)
#         torch.nn.init.zeros_(self.bn.bias)

    def forward(self, feature):
        if feature.dim() == 2:
            feature = feature.unsqueeze(2).unsqueeze(3)

        return F.relu(self.regressor(feature))
#         return F.relu(self.bn(self.regressor(feature)))

def make_log_name(args):
    log_name = args.model

#     if args.mode == 'eva':
#         log_name = args.modelpath.split('/')[-1]
#         log_name = log_name[:-3]

#     else:
    if args.pretrained:
        log_name += '_pretrained'
    log_name += f'_seed{args.seed}_epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_{args.optim}_wd{args.weight_decay}'
    if args.method == 'adv':
        log_name += f'_lamb{args.lamb}_eta{args.eta}'

    elif args.method == 'reweighting':
        log_name += f'_eta{args.eta}_iter{args.iteration}'

    elif 'gdro' in args.method:
        if not args.ibr:
            log_name += f'_gamma{args.gamma}'

        if 'chi' in args.method:
            log_name += f'_rho{args.rho}'

    if args.labelwise:
        log_name += '_labelwise'

    if args.method in ['fairhsic', 'mfd']:
        log_name += f'_lamb{args.lamb}'
        if args.method == 'mfd':
            log_name += f'_from_{args.teacher_type}'

    if args.dataset == 'celeba':
        if args.target != 'Blond_Hair':
            log_name += f'_T{args.target}'
        if args.add_attr is not None:
            log_name += f'_A{args.add_attr}'
        
    return log_name
