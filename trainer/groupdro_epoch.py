from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader
class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.gamma = args.gamma # learning rate of adv_probs
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.trueloss = args.trueloss        
        self.otpim_q = agrs.optim_q
        
    def _q_update_pd(self, train_subgroup_loss, n_classes, n_groups):
        train_subgroup_loss = torch.flatten(train_subgroup_loss)
        self.adv_probs = self.adv_probs * torch.exp(self.gamma*train_subgroup_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum()) 

    def _q_update_ibr_linear_interpolation(self, train_subgroup_loss, n_classes, n_groups, epoch, epochs):
        train_subgroup_loss = torch.flatten(train_subgroup_loss)
        assert len(train_subgroup_loss) == (n_classes * n_groups)

        idxs = np.array([i * n_classes for i in range(n_groups)]) 
        q_start = copy.deepcopy(self.adv_probs)
        
        q_ibr = torch.zeros_like(train_subgroup_loss)
        pos = train_subgroup_loss.argmax().item()
        q_ibr[pos] = 1
        
        cur_step_size = 0.5 * (1 + np.cos(np.pi * (epoch/epochs)))
        self.adv_probs = q_start + cur_step_size*(q_ibr - q_start)
        print(f' label loss : {train_subgroup_loss}')
        print(f'label q_ibr values : {q_ibr}')
        print(f'label q values : {self.adv_probs}')

                 
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
        
        self.adv_probs = torch.ones(n_groups*n_classes).cuda() / n_groups*n_classes
        if self.nlp_flag:
            self.t_total = len(train_loader) * epochs
            self.q_update_term = 0
            self.total_q_update = (epochs * len(train_loader)) / 100
            self.n_q_update = 0         
        
        for epoch in range(epochs):
            
            self._train_epoch(epoch, train_loader, model,criterion)
            if not self.nlp_flag:
                _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, 
                                                                   epoch,
                                                                   train=True,
                                                                   record=self.record,
                                                                   writer=writer
                                                                  )
                if self.trueloss:
                    train_subgroup_loss = 1- train_subgroup_acc 
                
                train_subgroup_loss = torch.flatten(train_subgroup_loss)

                # update q
                if self.trueloss:
                    train_subgroup_loss = 1-train_subgroup_acc
                # q update
                if self.optim_q == 'pd':
                    self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
                elif self.optim_q == 'ibr_ip':
                    self._q_update_ibr_linear_interpolation(train_subgroup_loss, n_classes, n_groups, epoch, epochs)
                
#                 train_subgroup_loss = torch.flatten(train_subgroup_loss)
#                 self.adv_probs = self.adv_probs * torch.exp(self.gamma*train_subgroup_loss.data)
#                 self.adv_probs = self.adv_probs/(self.adv_probs.sum()) # proj


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

                loss = self.train_criterion(outputs, labels)

                # calculate the groupwise losses
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                group_loss = (group_map @ loss.view(-1))/group_denom

                robust_loss = group_loss @ self.adv_probs
                
                return outputs, robust_loss 
            
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
                
            if self.nlp_flag:
                self.q_update_term += 1
                if self.q_update_term % 100 == 0:
                    print('lets start')
                    start = time.time()
                    _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, 
                                                                       epoch,
                                                                       train=True,
                                                                       record=False,
                                                                       writer=None
                                                                      )
                    end = time.time()

                    if self.trueloss:
                        train_subgroup_loss = 1-train_subgroup_acc
                    if self.optim_q == 'pd':
                        self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
                    elif self.optim_q == 'ibr_ip':
                        self._q_update_ibr_linear_interpolation(train_subgroup_loss, n_classes, n_groups, epoch, epochs)
                    self.n_q_update+=1
                    self.q_update_term = 0                



