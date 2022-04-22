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

        self.batch_size = args.batch_size
        self.n_workers = args.n_workers
        
    def train(self, train_loader, test_loader, epochs, dummy_loader=None, writer=None):
        log_set = defaultdict(list)
        model = self.model
        model.train()
        n_groups = train_loader.dataset.n_groups
        n_classes = train_loader.dataset.n_classes
        
        
        # Full batch 가져오기 #통계
        _, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                  n_workers=self.n_workers)  

        start_t = time.time()
        weight_set = self.reweight(Y_train, S_train, n_groups, n_classes)  
        if self.model == 'lr':
            # initialize the models
            self.initialize_all()

        for epoch in range(epochs):
            lb_idx = self._train_epoch(epoch, train_loader, model, weight_set)

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
        print('Training Time : {} hours {} minutes '.format(int(train_t / 60), (train_t % 60)))
                                                                          

            

            
    def _train_epoch(self, epoch, train_loader, model, weight_set):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        n_batches = len(train_loader)
        n_classes = train_loader.dataset.n_classes

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, indexes = data
#             print(indexes[0], groups)
            labels = targets
            # labels = labels.float() if n_classes == 2 else labels.long()
            labels = labels.long()

            weights = weight_set[indexes[0]]

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()
                
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

                loss = torch.mean(weights * nn.CrossEntropyLoss(reduction='none')(outputs, labels))
                
                return outputs, loss

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
        return last_batch_idx

    def get_statistics(self, dataset, batch_size=128, n_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)
        n_classes = dataloader.dataset.n_classes

        if model != None:
            model.eval()

        Y_pred_set = []
        Y_set = []
        S_set = []
        total = 0
        for i, data in enumerate(dataloader):
            inputs, _, sen_attrs, targets, indexes = data
#             Y_set.append(targets[sen_attrs != -1]) # sen_attrs = -1 means no supervision for sensitive group
            Y_set.append(targets) # sen_attrs = -1 means no supervision for sensitive group
            S_set.append(sen_attrs)

            if self.cuda:
                inputs = inputs.cuda()
                groups = sen_attrs.cuda()
            if model != None:
                outputs = model(inputs) 
                # Y_pred_set.append(torch.argmax(outputs, dim=1) if n_classes >2 else (torch.sigmoid(outputs) >= 0.5).float())
                Y_pred_set.append(torch.argmax(outputs, dim=1))
            total+= inputs.shape[0]

        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda()


    # update weight
    def reweight(self, label, sen_attrs, n_groups, n_classes):  
        n_data = len(label)
        weights = torch.zeros(n_data)
        w_matrix = torch.zeros((n_groups, n_classes))
        for g in range(n_groups):
            for c in range(n_classes):
                group_idxs = torch.where(sen_attrs == g)[0]
                class_idxs = torch.where(label==c)[0]
                group_class_idxs = torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
                w_matrix[g,c] = (len(group_idxs) * len(class_idxs))/(n_data * len(group_class_idxs))
        weights = w_matrix[sen_attrs, label]
        print('w_matrix(g,c)', w_matrix)
        return weights


    def criterion(self, model, outputs, labels):
        # if n_classes == 2:
        #     return nn.BCEWithLogitsLoss()(outputs, labels)
        # else:
        return nn.CrossEntropyLoss()(outputs, labels)
    
    
    def initialize_all(self):
        from networks import ModelFactory
        self.model = ModelFactory.get_model('lr', hidden_dim=64, n_classes=2, n_layer = 1)
        self.optimizer =optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = MultiStepLR(self.optimizer, [10,20], gamma=0.1)
        
