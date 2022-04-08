import torch
import numpy as np
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from sklearn.metrics import confusion_matrix
from utils import make_log_name


class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(method, **kwargs):
        if method == 'scratch':
            import trainer.vanilla_train as trainer
        elif method == 'mfd':
            import trainer.mfd as trainer
        elif method == 'fairhsic':
            import trainer.fairhsic as trainer
        elif method == 'adv':
            import trainer.adv_debiasing as trainer
        elif method == 'lbc':
            import trainer.lbc as trainer
        elif method == 'lgdro':
            import trainer.labelwise_groupdro as trainer
        elif method == 'gdro':
            import trainer.groupdro as trainer
        elif method == 'lgdro_chi':
            import trainer.labelwise_groupdro_chi as trainer
        else:
            raise Exception('Not allowed method')
        return trainer.Trainer(**kwargs)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''
    def __init__(self, model, args, optimizer, teacher=None, scheduler=None):
        self.get_inter = args.get_inter
        
        self.record = args.record
        self.cuda = args.cuda
        self.device = args.device
        self.t_device = args.t_device
        self.term = args.term
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm
        self.epochs = args.epochs
        self.method = args.method
        self.model_name =args.model
        self.nlp_flag = True if self.model_name.startswith("bert") else False
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.optim_type = args.optim
        self.sam = args.sam
        self.log_dir = args.log_dir
        self.criterion=torch.nn.CrossEntropyLoss(reduction='none')
        self.scheduler = None
        self.dataset = args.dataset
        self.uc = args.uc

        self.log_name = make_log_name(args)
        self.log_dir = os.path.join(args.log_dir, args.date, args.dataset, args.method)
        self.save_dir = os.path.join(args.save_dir, args.date, args.dataset, args.method)

        if scheduler is None:
            if self.optim_type == 'Adam' and self.optimizer is not None:
                if self.sam:
                    self.scheduler = ReduceLROnPlateau(self.optimizer.base_optimizer)
                else:
                    self.scheduler = ReduceLROnPlateau(self.optimizer)
            elif (self.optim_type == 'AdamP' or self.optim_type == 'AdamW') and self.optimizer is not None:
                if self.sam:
                    self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, self.epochs)
                else:
                    self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
            else: 
                if self.epochs == 70:
                    interval = [30, 60]
                else:
                    interval = [10, 20]
                if self.sam:
        #             self.scheduler = MultiStepLR(self.optimizer, [60, 120, 180], gamma=0.1)
                    self.scheduler = MultiStepLR(self.optimizer.base_optimizer, interval, gamma=0.1)
                else:
                    self.scheduler = MultiStepLR(self.optimizer, interval, gamma=0.1)
        else:
            self.scheduler = scheduler
            

    def evaluate(self, model, loader, criterion, epoch=0, device=None, train=False, record=False, writer=None):
        if record:
            assert writer is not None
            
        if not train:
            model.eval()
        else:
            model.train()
        n_groups = loader.dataset.n_groups
        n_classes = loader.dataset.n_classes
        n_subgroups = n_groups * n_classes        
        device = self.device if device is None else device

        group_count = torch.zeros(n_subgroups).cuda(device=device)
        group_loss = torch.zeros(n_subgroups).cuda(device=device)        
        group_acc = torch.zeros(n_subgroups).cuda(device=device) 
        
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
#                 if j == 100 and args.dataset=='celeba':
#                     break
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                labels = classes 
            
                if self.cuda:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    groups = groups.cuda(device)
                    
                if self.nlp_flag:
                    input_ids = inputs[:, :, 0]
                    input_masks = inputs[:, :, 1]
                    segment_ids = inputs[:, :, 2]
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=labels,
                    )[1] 
                else:
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                
                # calculate the labelwise losses
#                 if not self.uc:
                subgroups = groups * n_classes + labels                
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count += group_map.sum(1)

                group_loss += (group_map @ loss.view(-1))
                group_acc += group_map @ acc
#                 else:
#                     idxs = np.array([i * n_classes for i in range(n_groups)])
#                     for l in range(n_classes):
#                         mask = classes == l
# #                         print((groups[mask].T @ loss[mask]).float())
# #                         print(groups[mask].sum(dim=0).float())
#                         group_count[idxs+l] += groups[mask].sum(dim=0).float()
#                         group_loss[idxs+l] += (groups[mask].T @ loss[mask]).float()
#                         group_acc[idxs+l] += (groups[mask].T @ acc[mask]).float()
            loss = group_loss.sum() / group_count.sum() 
            acc = group_acc.sum() / group_count.sum() 

            group_loss /= group_count
            group_acc /= group_count

            group_loss = group_loss.reshape((n_groups, n_classes))            
            group_acc = group_acc.reshape((n_groups, n_classes))
            labelwise_acc_gap = torch.max(group_acc, dim=0)[0] - torch.min(group_acc, dim=0)[0]
            deoa = torch.mean(labelwise_acc_gap).item()
            deom = torch.max(labelwise_acc_gap).item()
            
        if record:
            self.write_record(writer, epoch, loss, acc, deom, deoa, group_loss, group_acc, train)
            
        model.train()
        return loss, acc, deom, deoa, group_acc, group_loss
    
    def write_record(self, writer, epoch, loss, acc, deom, deoa, group_loss, group_acc, train=False):
        flag = 'train' if train else 'test'
        
        n_groups = group_acc.shape[0]
        n_classes = group_loss.shape[1]
        writer.add_scalar(f'{flag}_loss', loss, epoch)
        writer.add_scalar(f'{flag}_acc', acc, epoch)
        writer.add_scalar(f'{flag}_deom', deom, epoch)
        writer.add_scalar(f'{flag}_deoa', deoa, epoch)

        acc_dict = {}
        loss_dict = {}

        for g in range(n_groups):
            for l in range(n_classes):
                acc_dict[f'g{g},l{l}'] = group_acc[g,l]
                loss_dict[f'g{g},l{l}'] = group_loss[g,l]

        writer.add_scalars(f'{flag}_subgroup_acc', acc_dict, epoch)
        writer.add_scalars(f'{flag}_subgroup_loss', loss_dict, epoch)        
    
    def save_model(self, save_dir, log_name="", model=None):
        model_to_save = self.model if model is None else model
        model_savepath = os.path.join(save_dir, log_name + '.pt')
        torch.save(model_to_save.state_dict(), model_savepath)

        print('Model saved to %s' % model_savepath)

    def compute_confusion_matix(self, dataset='test', n_classes=2,
                                dataloader=None, log_dir="", log_name=""):
        from scipy.io import savemat
        from collections import defaultdict
        self.model.eval()
        confu_mat = defaultdict(lambda: np.zeros((n_classes, n_classes)))
        print('# of {} data : {}'.format(dataset, len(dataloader.dataset)))

        predict_mat = {}
        output_set = torch.tensor([])
        group_set = torch.tensor([], dtype=torch.long)
        target_set = torch.tensor([], dtype=torch.long)
        intermediate_feature_set = torch.tensor([])
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the inputs
                inputs, _, groups, targets, _ = data
                labels = targets
                groups = groups.long()

                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                # forward                    
                if self.nlp_flag:
                    input_ids = inputs[:, :, 0]
                    input_masks = inputs[:, :, 1]
                    segment_ids = inputs[:, :, 2]
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=labels,
                    )[1] 
                else:
                    outputs = self.model(inputs)
                    
                if self.get_inter:
                    intermediate_feature = self.model.forward(inputs, get_inter=True)[-2]

                group_set = torch.cat((group_set, groups))
                target_set = torch.cat((target_set, targets))
                output_set = torch.cat((output_set, outputs.cpu()))
                if self.get_inter:
                    intermediate_feature_set = torch.cat((intermediate_feature_set, intermediate_feature.cpu()))

                pred = torch.argmax(outputs, 1)
                group_element = list(torch.unique(groups).numpy())
                for i in group_element:
                    mask = groups == i
                    if len(labels[mask]) != 0:
                        confu_mat[str(i)] += confusion_matrix(
                            labels[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                            labels=[i for i in range(n_classes)])
                        
#         print(confu_mat['0'][0,0] / confu_mat['0'][0,:].sum())
#         print(confu_mat['0'][1,1] / confu_mat['0'][1,:].sum())
#         print(confu_mat['1'][0,0] / confu_mat['1'][0,:].sum())
#         print(confu_mat['1'][1,1] / confu_mat['1'][1,:].sum())


        predict_mat['group_set'] = group_set.numpy()
        predict_mat['target_set'] = target_set.numpy()
        predict_mat['output_set'] = output_set.numpy()
        if self.get_inter:
            predict_mat['intermediate_feature_set'] = intermediate_feature_set.numpy()
            
        savepath = os.path.join(log_dir, log_name + '_{}_confu'.format(dataset))
        print('savepath', savepath)
        savemat(savepath, confu_mat, appendmat=True)

        savepath_pred = os.path.join(log_dir, log_name + '_{}_pred'.format(dataset))
        savemat(savepath_pred, predict_mat, appendmat=True)

        print('Computed confusion matrix for {} dataset successfully!'.format(dataset))
        return confu_mat


