import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import cvxpy as cvx
import time 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from copy import deepcopy

def chi_proj(pre_q, rho):
    #start = time.time()
    g = pre_q.shape[0]
    q = cvx.Variable(g)
    v = pre_q.cpu().numpy()
    #obj = cvx.Minimize(cvx.square(cvx.norm(q - v, 2)))
    obj = cvx.Minimize(cvx.sum(cvx.kl_div(q, v)))

    constraints = [q>= 0.0,
                   cvx.sum(q)==1.0,
                   cvx.square(cvx.norm(q-np.ones(g)/g, 2)) <= rho*2/g]
    
    prob = cvx.Problem(obj, constraints)
    prob.solve() # Returns the optimal value.
    print("optimal value : ", prob.value)
    print("pre q : ", pre_q)
    print("optimal var :", q.value)
    #end = time.time()
    #print(f'took {end-start} s')
    return q.value


# def chi_proj_nonuni(pre_q, rho, group_dist):
#     #start = time.time()
#     g = pre_q.shape[0]
#     q = cvx.Variable(g)
#     v = pre_q.cpu().numpy()
#     obj = cvx.Minimize(cvx.square(cvx.norm(q - v, 2)))

#     constraints = [q>= 0.0,
#                    cvx.sum(q)==1.0,
#                    cvx.square(q-group_dist) @ group_dist  <= rho]
    
#     prob = cvx.Problem(obj, constraints)
#     prob.solve() # Returns the optimal value.
#     print("optimal value : ", prob.value)
#     print("pre q : ", pre_q)
#     print("optimal var :", q.value)
# # 4    return q.value

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

def get_subgroup_accuracy(outputs, labels, groups, n_classes, n_groups, reduction='mean'):
    n_subgroups = n_classes*n_groups
    with torch.no_grad():
        subgroups = groups * n_classes + labels
        group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_denom = group_denom.reshape((n_groups, n_classes))
       
        predictions = torch.argmax(outputs, 1)
        c = (predictions==labels).float()

        num_correct = (group_map @ c).reshape((n_groups, n_classes))
        subgroup_acc = num_correct/group_denom
        group_acc = num_correct.sum(1) / group_denom.sum(1) 
        
    return subgroup_acc,group_acc 
    
def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")

def make_log_name(args):
    log_name = args.model

    if args.pretrained:
        log_name += '_pretrained'

    log_name += f'_seed{args.seed}_epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_{args.optim}'
    log_name += f'_wd{args.weight_decay}'

    if args.method == 'lbc':
        log_name += f'_eta{args.eta}_iter{args.iteration}'        
    
    elif args.method == 'egr':
        log_name += f'_eta{args.eta}_iter{args.iteration}_bound{args.bound_B}_constraint{args.constraint_c}'
    
    elif args.method == 'fairbatch':
        log_name += f'_gamma{args.gamma}'
        
    elif args.method == 'pl':
        log_name += f'_lamblr{args.lamblr}_eps{args.epsilon}'
        
    elif 'gdro' in args.method:
        log_name += f'_gamma{args.gamma}'

    elif args.method == 'rvp':
        log_name += f'_rho{args.rho}'        
        
    elif 'fairdro' in args.method:
        log_name += f'_{args.optim_q}'
        if args.optim_q == 'smt_ibr':
            log_name += f'_{args.q_decay}'
        log_name += f'_rho{args.rho}'
        if args.use_01loss:
            log_name +='_01loss'
        
    if 'cov' == args.method or 'fairret' == args.method:
        log_name += f'_lamb{args.lamb}'

    if 'renyi' == args.method:
        log_name += f'_lamb{args.lamb}'

    if 'direct_reg' == args.method:
        log_name += f'_lamb{args.lamb}'
   
    if args.method in ['fairhsic', 'mfd']:
        log_name += f'_lamb{args.lamb}'
        if args.method == 'mfd':
            log_name += f'_from_{args.teacher_type}'

    if args.fairness_criterion != 'dca':
        log_name += f'_criterion{args.fairness_criterion}'

    if args.balSampling:
        log_name += '_balSampling'

    if args.balanced:
        log_name += '_balanced'

    return log_name


def cal_dca(loader, model, writer, epoch):
    bs_list = [128, 256, 512,1024]
    acc_gap_dict = {}
    loss_gap_dict = {}
    for bs in bs_list:
        loader = DataLoader(loader.dataset, 
                            batch_size=bs, 
                            shuffle=False, 
                            num_workers=1, 
                            pin_memory=True, 
                            drop_last=True)
        model.train()

        n_groups = loader.dataset.n_groups
        n_classes = loader.dataset.n_classes
        n_subgroups = n_groups * n_classes        
        
        group_count_total = torch.zeros(n_subgroups).cuda()
        group_loss_total = torch.zeros(n_subgroups).cuda()        
        group_acc_total = torch.zeros(n_subgroups).cuda()        

        group_loss_list = []
        group_acc_list = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, targets, idx = data
                labels = targets
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()
                    
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()

                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                unnormalized_group_loss = (group_map @ loss.view(-1))
                unnormalized_group_acc = (group_map @ acc)
                group_loss = unnormalized_group_loss / group_denom
                group_acc = unnormalized_group_acc / group_denom
                loss = torch.mean(group_loss)

                group_count_total += group_count
                group_loss_total += unnormalized_group_loss
                group_acc_total += unnormalized_group_acc

                group_loss_list.append(group_loss)
                group_acc_list.append(group_acc)
        
        group_loss_total /= group_count_total
        group_acc_total /= group_count_total
        group_loss_total = group_loss_total.reshape((n_groups, n_classes))
        group_acc_total = group_acc_total.reshape((n_groups, n_classes))
        loss_gap_total = group_loss_total[0] - group_loss_total[1]
        acc_gap_total = group_loss_total[0] - group_loss_total[1]
        # balSampling_acc_gap = torch.max(group_acc, dim=0)[0] - torch.min(group_acc, dim=0)[0]
        gap = 0
        for group_loss in group_loss_list:
            group_loss = group_loss.reshape((n_groups, n_classes))
            loss_gap = group_loss[0] - group_loss[1]
            gap += (loss_gap_total - loss_gap).abs().mean()
        gap /= len(group_loss_list)
        loss_gap_dict[f'{bs}'] = gap

        gap = 0
        for group_acc in group_acc_list:
            group_acc = group_acc.reshape((n_groups, n_classes))
            acc_gap = group_acc[0] - group_acc[1]
            gap += (acc_gap_total - acc_gap).abs().mean()
        gap /= len(group_acc_list)
        acc_gap_dict[f'{bs}'] = gap

    writer.add_scalars('loss_gap_dict', loss_gap_dict, epoch)
    writer.add_scalars('acc_gap_dict', acc_gap_dict, epoch)
    return 

    
