from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from utils import get_accuracy
from collections import defaultdict
import trainer
import pickle
from torch.utils.data import DataLoader
import copy

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.eta = args.eta #learning rate for theta
        self.bound_B = args.bound_B # bound for L1 norm
        self.iteration = args.iteration #iteration for lambda
        self.batch_size = args.batch_size
        self.n_workers = args.n_workers
        self.target_criterion = args.target_criterion
        assert self.target_criterion == 'eo' #only for eo
        self.constraint_c = args.constraint_c # vector
        self.weight_decay = args.weight_decay #
        
    def train(self, train_loader, test_loader, epochs, dummy_loader=None, writer=None):
        log_set = defaultdict(list)

        self.model.train()
        self.n_groups = train_loader.dataset.n_groups
        self.n_classes = train_loader.dataset.n_classes
        assert self.n_classes == 2 #only for binary classification
        self.M_matrix = self.get_M_matrix(self.target_criterion)
        

        self.multiplier_set = []
        self.model_set = []
        
        self.theta = torch.zeros(self.n_groups * self.n_classes * 2)
        self.multiplier = torch.zeros(self.n_groups * self.n_classes * 2)
        self.best_value = float("inf")
    

        S_Y_set, Y_set, S_set, self.P_S_Y_mat, self.P_Y = self.get_statistics(train_loader.dataset, batch_size=self.batch_size, n_workers=self.n_workers)
        
        if self.cuda:
            self.theta = self.theta.cuda()
            self.M_matrix = self.M_matrix.cuda()
            self.multiplier = self.multiplier.cuda()
            self.P_S_Y_mat = self.P_S_Y_mat.cuda()
            self.P_Y = self.P_Y.cuda()
        
        if not self.nlp_flag:
            backup_model = copy.deepcopy(self.model)
        
        print('eta_learning_rate : ', self.eta)
        n_iters = self.iteration
        print('n_iters : ', n_iters)
        
        violations = 0
        for iter_ in range(n_iters):
            
#             self.multiplier = self.bound_B * (torch.exp(self.theta)/(1+torch.sum(torch.exp(self.theta))))
            self.multiplier = self.bound_B * (torch.exp(self.theta-torch.max(self.theta))/(torch.exp(-torch.max(self.theta))+torch.sum(torch.exp(self.theta-torch.max(self.theta))))) # for numerical stability
            print('self.multiplier', self.multiplier)
            self.multiplier_set.append(self.multiplier)
            
            start_t = time.time()
            
            if not self.nlp_flag:
                self.reset_model(backup_model)
            
            if self.nlp_flag:
                assert n_iters == 1
                self.weight_update_term = 1000 #For computation #100 
                self.weight_update_count = 0

            for epoch in range(epochs):
                lb_idx = self._train_epoch(epoch, train_loader, self.model)
                
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
                    
            end_t = time.time()
            train_t = int((end_t - start_t) / 60)
            print('Training Time : {} hours {} minutes / iter : {}/{}'.format(int(train_t / 60), (train_t % 60),
                                                                              (iter_ + 1), n_iters))
            
            if not self.nlp_flag:
                model=copy.deepcopy(self.model)
                self.model_set.append(model)
                
                # get statistics & calculate violation
                mu_, acc_ = self.get_mu(train_loader.dataset, batch_size=self.batch_size, n_workers=self.n_workers, model=model)
                self.theta += self.eta * (self.M_matrix @ mu_ - self.constraint_c)
#                 print('updated theta', self.theta)
#                 print('updated multiplier', mu_)
#                 print('train_acc', acc_)                        
        
        if not self.nlp_flag:
            # Get multiplier_avg
            multiplier_set_matrix = torch.stack(self.multiplier_set)
            multiplier_avg = torch.mean(multiplier_set_matrix, dim=0)
            # print('multiplier_avg', multiplier_avg)

            # Calculate lagrangian with multiplier_avg            
            print('inf_value', self.best_value)
            print('len(self.model_set)', len(self.model_set))
            for i in range(len(self.model_set)):
                print('i', i)
                value = self.Lagrangian_01(train_loader.dataset, batch_size=128, n_workers=2, model=self.model_set[i], multiplier=multiplier_avg)
                print('current_value', value)
                if value < self.best_value:
                    self.best_value = value
                    best_index = i
                    print('true')
                    print('current_best_value', self.best_value)

            print('final value', self.best_value)
            print('final index', best_index)

            # Get best deterministic model
            self.model = self.model_set[best_index]
        
#         print('last multiplier_set', self.multiplier_set)
#         print('last theta', self.theta)
#         print('last multiplier', self.multiplier)
#         print('last model_set', self.model_set)
#         print('last train acc', acc_)

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        if self.nlp_flag:
            best_model = copy.deepcopy(model)

        running_loss = 0.0
        running_acc = 0.0
        avg_batch_time = 0.0

        n_batches = len(train_loader)
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        lambda_mat = (self.multiplier[:self.n_groups * self.n_classes] - self.multiplier[self.n_groups * self.n_classes:]).reshape(self.n_groups, self.n_classes)
        
        for i, data in enumerate(train_loader):
            
            #mu updated
            #theta updated
            #lambda
            
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            # labels = labels.float() if n_classes == 2 else labels.long()
            groups = groups.long()
            labels = labels.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()
            
            if not self.balanced:
                C_0 = (labels!=0).long()
                C_1 = (labels!=1).long() + lambda_mat[groups, labels] / self.P_S_Y_mat[groups, labels] - torch.sum(lambda_mat[:, labels]) / self.P_Y[labels]
            else:
                subgroup_probs = self.P_S_Y_mat[groups, labels]
                C_0 = (labels!=0).long() / subgroup_probs
                C_1 = (labels!=1).long() / subgroup_probs + lambda_mat[groups, labels] / self.P_S_Y_mat[groups, labels] - torch.sum(lambda_mat[:, labels]) / self.P_Y[labels]
            reweights = torch.abs(C_0 - C_1)
            relabels = (C_0>C_1).long()
            
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
                    
                if self.balanced:
                    subgroups = groups * n_classes + labels
                    group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                    group_count = group_map.sum(1)
                    group_denom = group_count + (group_count==0).float() # avoid nans
                    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                    group_loss = (group_map @ loss.view(-1))/group_denom
                    loss = torch.mean(group_loss)
                else:
                    loss = torch.mean(reweights * nn.CrossEntropyLoss(reduction='none')(outputs, relabels))
                    
                return outputs, loss      

            outputs, loss = closure()
            
            preds = torch.argmax(outputs, dim=1) # not differentiable
            output_probs = torch.nn.Softmax(dim=None)(outputs) # n by c
            
            def closure_mu():
                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_map_ext = torch.cat((group_map, torch.ones(group_map.size(1)).view(1,-1).cuda()))
                group_count = group_map_ext.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans, mini-batch
                group_output = (group_map_ext @ (output_probs[:,1].view(-1).float()))/group_denom
                return group_output

    
            mu = closure_mu()

            reg = self.multiplier @ (self.M_matrix @ mu - self.constraint_c)
            loss += reg
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
            
            if self.nlp_flag:
                self.weight_update_count += 1
                if self.weight_update_count % self.weight_update_term == 0:
                    # get statistics
                    
#                     current_model=copy.deepcopy(model)
#                     self.model_set.append(current_model)
                    mu_, acc_ = self.get_mu(train_loader.dataset, batch_size=self.batch_size, n_workers=self.n_workers, model=model)
                    self.theta += self.eta * (self.M_matrix @ mu_ - self.constraint_c)
                    
                    # for numerical stability
                    self.multiplier = self.bound_B * (torch.exp(self.theta-torch.max(self.theta))/(torch.exp(-torch.max(self.theta))+torch.sum(torch.exp(self.theta-torch.max(self.theta))))) 
                    print('self.multiplier', self.multiplier)
                    self.multiplier_set.append(self.multiplier) # plus one for nlp
                    
                    lambda_mat = (self.multiplier[:self.n_groups * self.n_classes] - self.multiplier[self.n_groups * self.n_classes:]).reshape(self.n_groups, self.n_classes)
                    
                    # self.reset_model(backup_model) # no reset for NLP model
                    
                    # Get multiplier_avg
                    multiplier_set_matrix = torch.stack(self.multiplier_set)
                    multiplier_avg = torch.mean(multiplier_set_matrix, dim=0)
                    # print('multiplier_avg', multiplier_avg)
                    
                    # Calculate lagrangian with multiplier_avg                  
                    print('iter', self.weight_update_count)
                    value = self.Lagrangian_01(train_loader.dataset, batch_size=128, n_workers=2, model=model, multiplier=multiplier_avg) #eval mode
                    model.train() #train mode
                    print('current_value', value)
                    if value < self.best_value:
                        self.best_value = value
                        print('current_best_value', self.best_value)                        
                        best_model = copy.deepcopy(model)


            running_loss += loss.item()
            # binary = True if n_classes == 2 else False
            # running_acc += get_accuracy(outputs, labels, binary=binary)
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

        last_batch_idx = i
        print('loss', loss)
        
        # Get best deterministic model
        model = best_model
        
        return last_batch_idx

    def get_M_matrix(self, target_criterion): # for eo
        assert target_criterion == 'eo'
        plus_matrix = torch.eye(self.n_groups * self.n_classes)
        minus_matrix = -torch.eye(self.n_groups * self.n_classes)
        minus_vector = -torch.ones(self.n_groups * self.n_classes)
        plus_vector = torch.ones(self.n_groups * self.n_classes)
        matrix_cat = torch.cat((plus_matrix, minus_matrix))
        vector_cat = torch.cat((minus_vector, plus_vector))
        M_matrix = torch.cat((matrix_cat, vector_cat.reshape(-1,1)), dim=1)
        return M_matrix.float()

    def get_mu(self, dataset, batch_size=128, n_workers=2, model=None):
        if model != None:
            model.eval()

        mu = torch.zeros(self.n_groups * self.n_classes + 1)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)

        Y_pred_set = []
        Y_set = []
#         S_set = []
        S_Y_set = [] # subgroups = groups * n_classes + labels
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _, sen_attrs, targets, _ = data
                Y_set.append(targets)
#                 S_set.append(sen_attrs)
                S_Y_set.append(sen_attrs * self.n_classes + targets)

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
                    # Y_pred_set.append(torch.argmax(outputs, dim=1) if n_classes >2 else (torch.sigmoid(outputs) >= 0.5).float())
                    Y_pred_set.append(torch.argmax(outputs, dim=1))
                total+= inputs.shape[0]

        Y_set = torch.cat(Y_set).cuda()
#         S_set = torch.cat(S_set)
        S_Y_set = torch.cat(S_Y_set).cuda()
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        
        acc = torch.sum(Y_pred_set==Y_set)/len(Y_set)
        
        mu[-1] = torch.mean(Y_pred_set.float())
        for i in range(len(mu)-1):
            index_set = torch.where(S_Y_set==i)[0]
            mu[i] = torch.mean(Y_pred_set.float()[index_set])
        if self.cuda:
            mu = mu.cuda()
            acc = acc.cuda()
        return mu.float(), acc.float()
    

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
    
    
#     def initialize_all(self):
#         for name, param in self.model.state_dict().items():
# #             self.model.state_dict()[name][:] = torch.nn.init.normal_(param, mean=0.0, std=1.0)
#             if 'bias' not in name:
#                 self.model.state_dict()[name][:] = torch.nn.init.xavier_uniform_(param)
#             else:
#                 self.model.state_dict()[name][:] = torch.nn.init.uniform_(param)
        
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
#         print('initialized')

    def reset_model(self, backup_model):
        self.model = copy.deepcopy(backup_model)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay) # depends on method
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs) # depends on method
        print('reset')

    
    def Lagrangian_01(self, dataset, batch_size=128, n_workers=2, model=None, multiplier=None):
        if model != None:
            model.eval()
        
        mu, acc = self.get_mu(dataset, batch_size=self.batch_size, n_workers=self.n_workers, model=model)
        error_rate = 1.0 - acc
        Lagrangian = error_rate + multiplier @ (self.M_matrix @ mu - self.constraint_c)
        
        return Lagrangian.item()

    
    
    ##############################################################################################################
    
    
    def get_statistics(self, dataset, batch_size=128, n_workers=2):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)

        Y_set = []
        S_set = []
        S_Y_set = []
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _, sen_attrs, targets, _ = data
                Y_set.append(targets)
                S_set.append(sen_attrs)
                S_Y_set.append(sen_attrs * self.n_classes + targets)

                if self.cuda:
                    inputs = inputs.cuda()
                    groups = sen_attrs.cuda()
                    targets = targets.cuda()
                total+= inputs.shape[0]

        Y_set = torch.cat(Y_set).long()
        S_set = torch.cat(S_set).long()
        S_Y_set = torch.cat(S_Y_set).long()
        P_S_Y = torch.zeros(self.n_groups * self.n_classes)
        for i in range(len(P_S_Y)):
            index_set = torch.where(S_Y_set==i)[0]
            P_S_Y[i] = len(index_set) / len(S_Y_set)
        P_S_Y_mat = P_S_Y.reshape(self.n_groups, self.n_classes)
        P_Y = torch.sum(P_S_Y_mat, dim=0)
        return S_Y_set, Y_set, S_set, P_S_Y_mat, P_Y
    
