from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from utils import get_accuracy
from collections import defaultdict
import trainer
import pickle
from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        model = self.model
        model.train()

        n_groups = train_loader.dataset.n_groups
        n_classes = train_loader.dataset.n_classes
        
        # get statistics
        y_set, s_set = self.get_statistics(train_loader.dataset, bs=self.bs,
                                                  n_workers=self.n_workers)

        start_t = time.time()
        weight_matrix = self.get_reweight_matrix(y_set, s_set, n_groups, n_classes)  

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, weight_matrix)

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
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DCAM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_dcam, (eval_end_time - eval_start_time)))

            if self.record:
                self.evaluate(self.model, train_loader, 
                              self.criterion, epoch, 
                              train=True, 
                              record=self.record,
                              writer=writer
                             )

            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        end_t = time.time()
        train_t = int((end_t - start_t) / 60)
        print('Training Time : {} hours {} minutes '.format(int(train_t / 60), (train_t % 60)))
                                                                          
    def _train_epoch(self, epoch, train_loader, model, weight_matrix, criterion=None):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, indexes = data
            labels = targets
            groups = groups.long()
            labels = labels.long()

            weights = weight_matrix[groups, labels]
            
            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()
                
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
                loss = self.train_criterion(outputs, labels)
                group_loss = (group_map @ loss.view(-1))/group_denom
                loss = torch.mean(group_loss)
            else:
                if criterion is not None:
                    loss = criterion(outputs, labels).mean()
                else:
                    loss = self.criterion(outputs, labels).mean()

            loss.backward()
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            batch_end_time = time.time()
            avg_batch_time += batch_end_time - batch_start_time

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                avg_batch_time = 0.0

    def get_statistics(self, dataset, bs=128, n_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)

        if model != None:
            model.eval()

        y_set = []
        s_set = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                _, _, sen_attrs, targets, _ = data
                y_set.append(targets) 
                s_set.append(sen_attrs)

        y_set = torch.cat(y_set)
        s_set = torch.cat(s_set)
        return y_set.long().cuda(), s_set.long().cuda()

    # update weight
    def get_reweight_matrix(self, label, sen_attrs, n_groups, n_classes):  
        w_matrix = torch.zeros((n_groups, n_classes))
        for g in range(n_groups):
            for c in range(n_classes):
                group_mask = sen_attrs == g
                class_mask = label==c
                group_class_mask = group_mask * class_mask
                w_matrix[g,c] = group_mask.sum() * class_mask.sum() / group_class_mask.sum()
        return w_matrix

    
        
