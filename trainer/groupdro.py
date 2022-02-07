from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy
import trainer
import torch
import numpy as np


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.gamma = args.gamma # learning rate of adv_probs
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        
        self.adv_probs_dict = {}
        for l in range(num_classes):
#            self.adv_probs_dict[l] = torch.ones(num_groups).cuda() / num_groups*num_classes
            self.adv_probs = torch.ones(num_groups*num_classes).cuda() / num_groups*num_classes
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model,criterion)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_subgroup_acc  = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))
            if self.record:
                train_loss, train_acc, train_deom, train_deoa, train_subgroup_acc = self.evaluate(self.model, train_loader, self.criterion)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('train_acc', train_acc, epoch)
                writer.add_scalar('train_deom', train_deom, epoch)
                writer.add_scalar('train_deoa', train_deoa, epoch)
                writer.add_scalar('eval_loss', eval_loss, epoch)
                writer.add_scalar('eval_acc', eval_acc, epoch)
                writer.add_scalar('eval_deom', eval_deom, epoch)
                writer.add_scalar('eval_deoa', eval_deoa, epoch)
                
                eval_contents = {}
                train_contents = {}
                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_contents[f'g{g},l{l}'] = eval_subgroup_acc[g,l]
                        train_contents[f'g{g},l{l}'] = train_subgroup_acc[g,l]
                writer.add_scalars('eval_subgroup_acc', eval_contents, epoch)
                writer.add_scalars('train_subgroup_acc', train_contents, epoch)
                
            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        num_subgroups = num_classes * num_groups
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            subgroups = groups * num_classes + labels
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
            else:
                loss = self.train_criterion(outputs, labels)
            
            # calculate the groupwise losses
            group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom
            
            # update q
            robust_loss = 0
#            idxs = np.array([i * num_classes for i in range(num_groups)])
#            for l in range(num_classes):
#                label_group_loss = group_loss[idxs+l]
#                self.adv_probs_dict[l] = self.adv_probs_dict[l] * torch.exp(self.gamma * label_group_loss.data)
#                self.adv_probs_dict[l] = self.adv_probs_dict[l]/(self.adv_probs_dict[l].sum())
#                robust_loss += label_group_loss @ self.adv_probs_dict[l]
            self.adv_probs = self.adv_probs * torch.exp(self.gamma*group_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum())
            
            robust_loss = group_loss @ self.adv_probs
            
            running_loss += robust_loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            robust_loss.backward()
            self.optimizer.step()
            
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

