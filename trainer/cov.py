from __future__ import print_function
from collections import defaultdict
import time
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cvxpy as cvx
# import dccp
# from dccp.problem import is_dccp
import numpy as np

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lamb
        self.fairness_criterion = args.fairness_criterion
        assert (self.fairness_criterion == 'eo' or self.fairness_criterion == 'ap')
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
            
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion)
            
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_dcam, eval_dcaa, _, _  = self.evaluate(self.model, 
                                                                             test_loader, 
                                                                             self.criterion,
                                                                             epoch, 
                                                                             train=False,
                                                                             record=self.record,
                                                                             writer=writer
                                                                            )
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_dcam, (eval_end_time - eval_start_time)))

            if self.record:
                _, _, _, _, train_subgroup_acc, train_subgroup_loss=self.evaluate(self.model, train_loader, self.criterion, epoch, 
                              train=True, 
                              record=self.record,
                              writer=writer
                             )
                cov = self.calculate_covariance(self.model, train_loader)
                n_classes = train_loader.dataset.n_classes
                covs = {}
                for l in range(n_classes):
                    covs[f'l{l}'] = cov[l]
                writer.add_scalars('covs', covs, epoch)

            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        print('Training Finished!')        

    def calculate_covariance(self, model, train_loader):
        model.train()
        
        n_groups = train_loader.dataset.n_groups
        
        FPR_total = 0
        FNR_total = 0
        for i, data in enumerate(train_loader):
            inputs, _, groups, targets, idx = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

            def closure_FPR(inputs, groups, labels, model):
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g

                outputs = model(inputs) # n by 2

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = -(labels.view(-1,1)-1)*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss

            def closure_FNR(inputs, groups, labels, model):
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g
                outputs = model(inputs) # n by 2

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = (labels.view(-1,1))*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss
            
            FPR_total  += closure_FPR(inputs, groups, labels, model)
            FNR_total += closure_FNR(inputs, groups, labels, model)
        return [torch.abs(FPR_total).item(), torch.abs(FNR_total).item()]

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
        
            inputs, _, groups, targets, idx = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            def closure():
                if self.data == 'jigsaw':
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
                    
                if self.balanced:
                    subgroups = groups * n_classes + labels
                    group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                    group_count = group_map.sum(1)
                    group_denom = group_count + (group_count==0).float() # avoid nans
                    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                    group_loss = (group_map @ loss.view(-1))/group_denom
                    loss = torch.mean(group_loss)
                else:
                    if criterion is not None:
                        loss = criterion(outputs, labels).mean()
                    else:
                        loss = self.criterion(outputs, labels).mean()
                return outputs, loss
            
            if self.fairness_criterion == 'eo':
                def closure_FPR(inputs, groups, labels, model):
                    groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                    groups_onehot = groups_onehot.float() # n by g
                    if self.data == 'jigsaw':
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
                        outputs = model(inputs) # n by 2

                    d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                    d_theta_new = -(labels.view(-1,1)-1)*(2*labels.view(-1,1)-1)*d_theta
                    g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                    z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                    loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                    loss = torch.sum(loss_groupwise)
                    return loss

                def closure_FNR(inputs, groups, labels, model):
                    groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                    groups_onehot = groups_onehot.float() # n by g
                    if self.data == 'jigsaw':
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
                        outputs = model(inputs) # n by 2
                    d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                    d_theta_new = (labels.view(-1,1))*(2*labels.view(-1,1)-1)*d_theta
                    g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                    z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                    loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                    loss = torch.sum(loss_groupwise)
                    return loss
            
            elif self.fairness_criterion == 'ap':
                def closure_OMR(inputs, groups, labels, model):
                    groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                    groups_onehot = groups_onehot.float() # n by g
                    if self.data == 'jigsaw':
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
                        outputs = model(inputs) # n by 2
                    d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                    d_theta_new = (2*labels.view(-1,1)-1)*d_theta # y*d_theta
                    g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                    z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                    loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                    loss = torch.sum(loss_groupwise)
                    return loss
                
            outputs, loss = closure()
            
            if self.fairness_criterion == 'eo':
                loss += self.lamb*closure_FPR(inputs, groups, labels, model)
                loss += self.lamb*closure_FNR(inputs, groups, labels, model)
            
            elif self.fairness_criterion == 'ap':
                loss += self.lamb*closure_OMR(inputs, groups, labels, model)
            
            loss.backward()
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
    
    
