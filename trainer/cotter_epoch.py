from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.epsilon = args.epsilon 
        self.lamblr = args.lamblr # learning rate of adv_probs
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
#         self.train_criterion = torch.nn.MultiMarginLoss(reduction='none')
        self.hinge_loss = torch.nn.MultiMarginLoss()
#         self.hinge_loss = nn.CrossEntropyLoss()
        
    def stationary_distribution(self, M):
#         transition_matrix_transp = transition_matrix.T
        mat = np.array(M)
        eigenvals, eigenvects = np.linalg.eig(M)
        '''
        Find the indexes of the eigenvalues that are close to one.
        Use them to select the target eigen vectors. Flatten the result.
        '''
        close_to_1_idx = np.isclose(eigenvals,1)
        target_eigenvect = eigenvects[:,close_to_1_idx]
        target_eigenvect = target_eigenvect[:,0]
        # Turn the eigenvector elements into probabilites
        stationary_distrib = target_eigenvect / sum(target_eigenvect) 
        stationary_distrib = stationary_distrib.real
        
        return torch.tensor(stationary_distrib).cuda()
    
    def eo_constraints(self, outputs, labels, groups):
        tnr_group0_mask = ((1-labels) * (1-groups)) == 1
        tnr_group1_mask = ((1-labels) * groups) == 1
        tpr_group0_mask = (labels * (1-groups)) == 1
        tpr_group1_mask = (labels * groups) == 1

        a = tnr_group0_mask.sum()
        b = tnr_group1_mask.sum()
        c = tpr_group0_mask.sum()
        d = tpr_group1_mask.sum()
        
        tnr_group0 = self.hinge_loss(outputs[tnr_group0_mask], labels[tnr_group0_mask]) 
        tnr_group1 = self.hinge_loss(outputs[tnr_group1_mask], labels[tnr_group1_mask])
        tpr_group0 = self.hinge_loss(outputs[tpr_group0_mask], labels[tpr_group0_mask])
        tpr_group1 = self.hinge_loss(outputs[tpr_group1_mask], labels[tpr_group1_mask])
        
        g0 = tnr_group0 - tnr_group1 - self.epsilon if a*b != 0 else 0
        g1 = - tnr_group0 + tnr_group1 - self.epsilon if a*b != 0 else 0      
        g2 = tpr_group0 - tpr_group1 - self.epsilon if c*d != 0 else 0
        g3 = - tpr_group0 + tpr_group1 - self.epsilon if c*d != 0 else 0

        return (g0,g1,g2,g3)
    
    def update_M(self, station_dist, train_subgroup_acc):
        
        tnr_group0 = 1-train_subgroup_acc[0,0]
        tnr_group1 = 1-train_subgroup_acc[1,0]
        tpr_group0 = 1-train_subgroup_acc[0,1]
        tpr_group1 = 1-train_subgroup_acc[1,1]
        
        g0 = tnr_group0 - tnr_group1 - self.epsilon
        g1 = - tnr_group0 + tnr_group1 - self.epsilon       
        g2 = tpr_group0 - tpr_group1 - self.epsilon
        g3 = - tpr_group0 + tpr_group1 - self.epsilon        
        
        grad = torch.tensor([0,g0,g1,g2,g3]).unsqueeze(1)
        tmp = grad @ station_dist.cpu().unsqueeze(0)
        grad = torch.exp(self.lamblr * tmp)
        self.M = self.M * grad
        self.M = self.M / self.M.sum(0)
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        
        self.normal_loader = DataLoader(train_loader.dataset, 
                                        batch_size=128, 
                                        shuffle=False, 
                                        num_workers=2, 
                                        pin_memory=True, 
                                        drop_last=False)
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        
        self.M = torch.ones((5,5))/5
        
        for epoch in range(epochs):
            station_dist = self.stationary_distribution(self.M)
            self._train_epoch(epoch, train_loader, model, station_dist, criterion)
            _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, 
                                                               epoch,
                                                               train=True,
                                                               record=self.record,
                                                               writer=writer
                                                              )
            self.update_M(station_dist, train_subgroup_acc)

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

    def _train_epoch(self, epoch, train_loader, model, station_dist, criterion=None):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            def closure():
                subgroups = groups * n_classes + labels
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

                constraints_loss = self.eo_constraints(outputs, labels, groups)
                tmp = 0
                for i in range(4):
                    tmp += constraints_loss[i] * station_dist[i+1]
                constraints_loss = tmp
                
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
                
                return outputs, station_dist[0]*loss + constraints_loss 
            
            outputs, loss = closure()            
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




