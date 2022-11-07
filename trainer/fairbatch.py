from __future__ import print_function

import time
import os
import torch
import torch.nn as nn
from utils import get_accuracy
from collections import defaultdict
import torch.optim as optim
import trainer
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, test_loader, epochs, writer=None):
        
        dummy_loader = DataLoader(train_loader.dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=2, 
                                          pin_memory=True, drop_last=False)

        if self.nlp_flag:
            self.adjust_count = 0
            self.adjust_term = 100
        self.model.train()
        
        for epoch in range(epochs):
            train_acc, train_loss = self._train_epoch(epoch, train_loader, self.model, dummy_loader)

            eval_start_time = time.time()                
            eval_loss, eval_acc, eval_deom, eval_deoa, _, _  = self.evaluate(self.model, 
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
                   eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))

            if self.record:
                self.evaluate(self.model, train_loader, self.criterion, epoch, 
                              train=True, 
                              record=self.record,
                              writer=writer
                             )
            
            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print('Training Finished!')

        return self.model


    def _train_epoch(self, epoch, train_loader, model, dummy_loader):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        batch_start_time = time.time()
        if not self.nlp_flag:
            self.adjust_lambda(model, train_loader, dummy_loader)
        
        for i, data in enumerate(train_loader):
            if self.nlp_flag:
                if self.adjust_count % self.adjust_term == 0:
                    self.adjust_lambda(model, train_loader, dummy_loader)    
                self.adjust_count+=1
            # Get the inputs
            inputs, _, groups, labels, _ = data
            if self.cuda:
                inputs = inputs.cuda().squeeze()
                labels = labels.cuda().squeeze()
                groups = groups.cuda()

            # labels = labels.float() if num_classes == 2 else labels.long()
            labels = labels.long()

            if self.nlp_flag:
                input_ids = inputs[:, :, 0]
                input_masks = inputs[:, :, 1]
                segment_ids = inputs[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=labels,
                    output_hidden_states=True
                )
                outputs = outputs[1]
            else:
                outputs = model(inputs)

            if self.balanced:
                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                group_loss = (group_map @ loss.view(-1))/group_denom
#                 weights = self.weight_matrix.flatten().cuda()
                loss = torch.mean(group_loss)
            else:
                loss = self.criterion(outputs, labels).mean()

            running_loss += loss.item()
            # binary = True if num_classes ==2 else False
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.3f} '
                      '[{:.3f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
                
        return running_acc / self.term, running_loss / self.term
    
    def adjust_lambda(self, model, train_loader, dummy_loader):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.
        """
        
        model.train()
        
        logits = []
        labels = []
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        
        sampler = train_loader.sampler
        with torch.no_grad():
            for i, data in enumerate(dummy_loader):
                inputs, _, groups, _labels, tmp = data
                if self.cuda:
                    inputs = inputs.cuda()
                    _labels = _labels.cuda()
                    groups = groups.cuda()
                
                if self.nlp_flag:
                    input_ids = inputs[:, :, 0]
                    input_masks = inputs[:, :, 1]
                    segment_ids = inputs[:, :, 2]
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=_labels,
                        output_hidden_states=True
                    )
                    outputs = outputs[1]
                else:
                    outputs = model(inputs)

                logits.append(outputs)
                labels.append(_labels)

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        labels = labels.long()
        # TO DO
        # We should use BCELoss if a model outputs one-dim vecs
        # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        if sampler.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
#             eo_loss = criterion ((F.tanh(logits)+1)/2, (labels+1)/2)
            eo_loss = criterion(logits, labels)
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[sampler.yz_index[tmp_yz]])) / sampler.yz_len[tmp_yz]
                
            for tmp_y in sampler.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[sampler.y_index[tmp_y]])) / sampler.y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                sampler.lb1 += sampler.gamma
            else:
                sampler.lb1 -= sampler.gamma
                
            if sampler.lb1 < 0:
                sampler.lb1 = 0
            elif sampler.lb1 > 1:
                sampler.lb1 = 1 
                
        elif sampler.fairness_type == 'eo':
            
            yhat_yz = {}
            yhat_y = {}
                        
#             eo_loss = criterion ((F.tanh(logits)+1)/2, (labels+1)/2)

            eo_loss = criterion(logits, labels.long())
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[sampler.yz_index[tmp_yz]])) / sampler.yz_len[tmp_yz]
                
            for tmp_y in sampler.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[sampler.y_index[tmp_y]])) / sampler.y_len[tmp_y]

            max_diff = 0
            pos = (0, 0)

            for _l in range(n_classes):
                # max_diff = 0
                # pos = 0
                for _g in range(1,n_groups):
                    tmp_diff = abs(yhat_yz[(_l, _g)] - yhat_yz[(_l, _g-1)])
                    if max_diff < tmp_diff:
                        max_diff = tmp_diff
                        pos = (_l, _g) if yhat_yz[(_l, _g)] >= yhat_yz[(_l, _g-1)] else (_l, _g-1)

                # # lb update per label
                #     #find plus position
                # if yhat_yz[(_l, pos)] > yhat_yz[(_l, pos-1)]:
                #     target = pos-1
                # else:
                #     target = pos

            pos_label = pos[0]
            pos_group = pos[1]
            for _g in range(n_groups):
                if _g == pos_group:
                    sampler.lbs[pos_label][_g] += sampler.gamma
                else:
                    sampler.lbs[pos_label][_g] -= sampler.gamma / (n_groups-1)
                if sampler.lbs[pos_label][_g] > 1:
                    sampler.lbs[pos_label][_g] = 1
                elif sampler.lbs[pos_label][_g] < 0:
                    sampler.lbs[pos_label][_g] = 0

            #normalize
            sampler.lbs[pos_label] = [i / sum(sampler.lbs[pos_label]) for i in sampler.lbs[pos_label]]
                
            
#             y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
#             y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
#             if y1_diff > y0_diff:
#                 if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
#                     sampler.lb1 += sampler.alpha
#                 else:
#                     sampler.lb1 -= sampler.alpha
#             else:
#                 if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
#                     sampler.lb2 += sampler.alpha
#                 else:
#                     sampler.lb2 -= sampler.alpha
                    
                
#             if sampler.lb1 < 0:
#                 sampler.lb1 = 0
#             elif sampler.lb1 > 1:
#                 sampler.lb1 = 1
                
#             if sampler.lb2 < 0:
#                 sampler.lb2 = 0
#             elif sampler.lb2 > 1:
#                 sampler.lb2 = 1
                
        elif sampler.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(sampler.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
#             dp_loss = criterion((F.tanh(logits)+1)/2, ones_tensor) # Note that ones tensor puts as the true label
            dp_loss = criterion(logits, ones_tensor.long())
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[sampler.yz_index[tmp_yz]])) / sampler.z_len[tmp_yz[1]]


            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    sampler.lbs[1][1] += sampler.gamma
                    sampler.lbs[1][0] -= sampler.gamma
                else:
                    sampler.lbs[1][1] -= sampler.gamma
                    sampler.lbs[1][0] += sampler.gamma

            else:
                if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                    sampler.lbs[0][1] -= sampler.gamma
                    sampler.lbs[0][0] += sampler.gamma

                else:
                    sampler.lbs[0][1] += sampler.gamma
                    sampler.lbs[0][0] -= sampler.gamma

            # sum to c?
            if sampler.lbs[1][1] < 0:
                sampler.lbs[1][1] = 0
                sampler.lbs[1][0] = 1
            elif sampler.lbs[1][1] > 1:
                sampler.lbs[1][1] = 1
                sampler.lbs[1][0] = 0

            if sampler.lbs[0][1] < 0:
                sampler.lbs[0][1] = 0
                sampler.lbs[0][0] = 1
            elif sampler.lbs[0][1] > 1:
                sampler.lbs[0][1] = 1
                sampler.lbs[0][0] = 0

        model.train()
