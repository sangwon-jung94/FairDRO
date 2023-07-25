import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed
from adamp import AdamP
from tensorboardX import SummaryWriter
from sam.sam import SAM
from arguments import get_args
import time
import os 
from torch.utils.data import DataLoader
args = get_args()


def main():

    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    result_dir = os.path.join(args.result_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(result_dir)
    writer = None
    if args.record:
        log_dir = os.path.join(args.log_dir, args.date, dataset, args.method)
        check_log_dir(log_dir)
        writer = SummaryWriter(log_dir + '/' + log_name)

    print(log_name)    
    ########################## get dataloader ################################
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset, 
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        n_workers=args.n_workers,
                                                        target_attr=args.target,
                                                        add_attr = args.add_attr,
#                                                         skew_ratio=args.skew_ratio,
                                                        labelwise=args.labelwise,
                                                        args=args
                                                        )
    n_classes, n_groups, train_loader, test_loader = tmp
    ########################## get model ##################################
    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    elif 'cifar' in args.dataset:
        args.img_size = 32

    model = networks.ModelFactory.get_model(args.model, n_classes, args.img_size,
                                            pretrained=args.pretrained, n_groups=n_groups)

    model.cuda('cuda:{}'.format(args.device))
    
    if args.pretrained:
        if args.modelpath is not None:
            model.load_state_dict(torch.load(args.modelpath))
        elif args.model == 'mlp' and (args.teacher_path is not None and args.teacher_type):
            model.load_state_dict(torch.load(args.teacher_path))
        
    teacher = None
    if ((args.method == 'mfd' and args.teacher_path is not None) and args.mode != 'eval'):
        teacher = networks.ModelFactory.get_model(args.teacher_type, train_loader.dataset.n_classes, args.img_size)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cuda:{}'.format(args.t_device))))
        teacher.cuda('cuda:{}'.format(args.t_device))

    if ((args.method == 'lgdro_chi' and args.teacher_path is not None) and args.mode != 'eval'):
#     (args.method=='lgdro_chi' and args.kd):
        teacher = networks.ModelFactory.get_model(args.teacher_type, train_loader.dataset.n_classes, args.img_size)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cuda:{}'.format(args.t_device))))
        teacher.cuda('cuda:{}'.format(args.t_device))
        
#     set_seed(seed)
    scheduler=None
    ########################## get trainer ##################################
    if 'Adam' == args.optim:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'AdamP' == args.optim:
        optimizer = optim.AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'AdamW' == args.optim:
        if not args.model.startswith("bert"):
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # BERT uses its own scheduler and optimizer
        else: 
            from pytorch_transformers import WarmupLinearSchedule
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            optimizer = optim.AdamW(optimizer_grouped_parameters,
                                lr=args.lr,
                                eps=1e-8,
                                )
            t_total = len(train_loader) * args.epochs
            print(f"\nt_total is {t_total}\n")
            scheduler = WarmupLinearSchedule(optimizer,
                                             warmup_steps=0,
                                             t_total=t_total)        
        
    elif 'SGD' == args.optim:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.method == 'mfd':
        trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                    optimizer=optimizer, teacher=teacher, scheduler=scheduler)
    else:
        trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                    optimizer=optimizer, scheduler=scheduler)

    ####################### start training or evaluating ####################
    
    if args.mode == 'train':
        start_t = time.time()
        trainer_.train(train_loader, test_loader, args.epochs, writer=writer)
        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        trainer_.save_model(save_dir, log_name)
    
    else:
        print('Evaluation ----------------')
        model_to_load = args.modelpath
        trainer_.model.load_state_dict(torch.load(model_to_load))
        print('Trained model loaded successfully')

    if args.evalset == 'all':
        trainer_.compute_confusion_matix('train', train_loader.dataset.n_classes, train_loader, result_dir, log_name)
        trainer_.compute_confusion_matix('test', test_loader.dataset.n_classes, test_loader, result_dir, log_name)

    elif args.evalset == 'train':
        trainer_.compute_confusion_matix('train', train_loader.dataset.n_classes, train_loader, result_dir, log_name)
    else:
        trainer_.compute_confusion_matix('test', test_loader.dataset.n_classes, test_loader, result_dir, log_name)
    if writer is not None:
        writer.close()
    print('Done!')


if __name__ == '__main__':
    main()
    
    

            
            

