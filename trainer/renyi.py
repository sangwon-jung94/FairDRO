from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from utils import get_accuracy
from collections import defaultdict
import trainer
import pickle
from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        
        self.lamb = args.lamb
        self.batch_size = args.batch_size
        self.n_workers = args.n_workers
        self.reweighting_target_criterion = args.reweighting_target_criterion
        
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        self.n_groups = train_loader.dataset.n_groups
        self.n_classes = train_loader.dataset.n_classes
        if self.reweighting_target_criterion == 'eo':
            self.weights = torch.zeros((self.n_classes, self.n_classes))
        if self.reweighting_target_criterion == 'dp':
            self.weights = torch.zeros((1, self.n_classes))
        
        

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion)
            
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

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
        
            inputs, _, groups, targets, idx = data
            groups = groups.long()
            labels = targets.long()
            weights = self.weights
            
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                weights = weights.cuda(device=self.device)
                
                
            def closure():
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
                
                if criterion is not None:
                    loss = criterion(outputs, labels).mean()
                else:
                    loss = self.criterion(outputs, labels).mean()
                return outputs, loss
            
            if self.reweighting_target_criterion == 'dp':
                def closure_renyi(inputs, groups, labels, model, weights):
                    assert (weights).shape == (1, self.n_classes)

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
                        outputs = model(inputs) # n by 2

                    output_probs = torch.nn.Softmax(dim=None)(outputs) # n by c

                    if self.n_groups == 2:
                        s_tilde = ((2*groups-1).view(len(groups),1)) * (torch.ones_like(groups).view(len(groups),1)).expand(len(groups), self.n_classes)
                        multiplier = -(weights**2)+weights*s_tilde
                        assert (multiplier).shape == (len(groups), self.n_classes)

                        sample_loss = torch.sum(multiplier*output_probs, dim=1)
                        loss = torch.mean(sample_loss)

                    else: 
                        print('not implemented')

                    return loss
                
                
            if self.reweighting_target_criterion == 'eo':
                def closure_renyi(inputs, groups, labels, model, weights):
                    assert (weights).shape == (self.n_classes, self.n_classes)
                    loss = 0
                    index_set = []
                    for c in range(self.n_classes):
                        index_set.append(torch.where(labels==c)[0])

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
                        outputs = model(inputs) # n by 2

                    output_probs = torch.nn.Softmax(dim=None)(outputs) # n by c

                    

                    if self.n_groups == 2:
                        s_tilde = ((2*groups-1).view(len(groups),1)) * (torch.ones_like(groups).view(len(groups),1)).expand(len(groups), self.n_classes)
                        
                        for c in range(self.n_classes):
                            if index_set[c] == []:
                                pass
                            else:
                                output_probs_c = output_probs[index_set[c]]
                                s_tilde_c = s_tilde[index_set[c]]
                                weights_c = weights[c]
                                multiplier_c = -(weights_c**2)+weights_c*s_tilde_c
                                assert (multiplier_c).shape == (len(index_set[c]), self.n_classes)

                                sample_loss_c = torch.sum(multiplier_c*output_probs_c, dim=1)
                                loss_c = torch.mean(sample_loss_c)
                                loss += loss_c

                    else: 
                        groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                        groups_onehot = groups_onehot.float() # n by g

                    return loss
            
            
            
            outputs, loss = closure()
            loss += self.lamb*closure_renyi(inputs, groups, labels, model, weights)
            
            loss.backward()
            if not self.sam:
                if self.nlp_flag:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self.optimizer.first_step(zero_grad=True)
                outputs, loss = closure()
                loss.backward()
                if self.nlp_flag:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
                self.optimizer.second_step(zero_grad=True)
                
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
            
            
            
        self.weights = self.update_weights(train_loader.dataset, self.batch_size, self.n_workers, model, weights) # implemented for each epoch
            
    def update_weights(self, dataset, batch_size, n_workers, model, weights):  
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)
        
        
        Y_prob_set = []
        Y_set = []
        S_set = []
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _, sen_attrs, targets, _ = data
                Y_set.append(targets) # sen_attrs = -1 means no supervision for sensitive group
                S_set.append(sen_attrs)

                if self.cuda:
                    inputs = inputs.cuda()
                    groups = sen_attrs.cuda()
                    targets = targets.cuda()
                if model != None:
                    if self.nlp_flag:
                        input_ids = inputs[:, :, 0]
                        input_masks = inputs[:, :, 1]
                        segment_ids = inputs[:, :, 2]
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=input_masks,
                            token_type_ids=segment_ids,
                            labels=targets,
                        )[1] 
                    else:
                        outputs = model(inputs)
                    output_probs = torch.nn.Softmax(dim=None)(outputs) # n by c
                    Y_prob_set.append(output_probs)
                total+= inputs.shape[0]

        Y_set = torch.cat(Y_set).long().cuda()
        S_set = torch.cat(S_set).long().cuda()
        Y_prob_set = torch.cat(Y_prob_set) if len(Y_prob_set) != 0 else torch.zeros(0)
        
        
        index_set =[]
        for c in range(self.n_classes):
            index_set.append(torch.where(Y_set==c)[0])
        
        if self.n_groups == 2:
            S_set = S_set.view(len(S_set), 1)
            S_tilde_set = 2*S_set-1
            for c in range(self.n_classes):
                denominator = 2*torch.sum(Y_prob_set[index_set[c]], dim=0).view(1, self.n_classes)
                numerator = torch.sum(S_tilde_set[index_set[c]]*Y_prob_set[index_set[c]], dim=0).view(1, self.n_classes)
                weights[c] = numerator/denominator 
        return weights