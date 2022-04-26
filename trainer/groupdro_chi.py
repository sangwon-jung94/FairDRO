from __future__ import print_function
from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import get_accuracy, chi_proj, chi_proj_nonuni

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.gamma = args.gamma # learning rate of adv_probs
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.rho = args.rho
        self.trueloss = args.trueloss
        self.optim_q = args.optim_q
        self.q_decay = args.q_decay
        self.margin = args.margin

    def _q_update_pd(self, train_subgroup_loss, n_classes, n_groups):
        train_subgroup_loss = torch.flatten(train_subgroup_loss)
        
        idxs = np.array([i * n_classes for i in range(n_groups)])  
            
        self.adv_probs *= torch.exp(self.gamma*train_subgroup_loss)
        self.adv_probs = torch.from_numpy(chi_proj(self.adv_probs, self.rho)).cuda(device=self.device).float()

    def _q_update_ibr_linear_interpolation(self, train_subgroup_loss, n_classes, n_groups, epoch, epochs):
        train_subgroup_loss = torch.flatten(train_subgroup_loss)
        assert len(train_subgroup_loss) == (n_classes * n_groups)
        
        q_start = copy.deepcopy(self.adv_probs)
        q_ibr = copy.deepcopy(self.adv_probs)
        
        if self.q_decay == 'cos':
            cur_step_size = 0.5 * (1 + np.cos(np.pi * (epoch/epochs)))
        elif self.q_decay == 'linear':
            cur_step_size = 1 - epoch/epochs
        else:
            raise ValueError
            
        if not self.margin:
            q_ibr = self._update_mw_bisection(train_subgroup_loss)#, self.group_dist[l])
        else:
            q_ibr = self._update_mw_margin(train_subgroup_loss)#, self.group_dist[l])
            
        self.adv_probs = q_start + cur_step_size*(q_ibr - q_start)
        print(f'loss : {train_subgroup_loss}')
        print(f'q_ibr values : {q_ibr}')
        print(f'q values : {self.adv_probs}')
                         
        
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
            
            if not self.nlp_flag or self.record:
                _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, 
                                                                   epoch,
                                                                   train=True,
                                                                   record=self.record,
                                                                   writer=writer
                                                                  )
                if self.trueloss:
                    train_subgroup_loss = 1-train_subgroup_acc
                # q update
                if self.optim_q == 'pd':
                    self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
#                 elif self.optim_q == 'ibr':
#                     self._q_update_ibr(train_subgroup_loss, n_classes, n_groups)
                elif self.optim_q == 'ibr_ip':
                    self._q_update_ibr_linear_interpolation(train_subgroup_loss, n_classes, n_groups, epoch, epochs)


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
                        
                    # q update
                    if self.optim_q == 'pd':
                        self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
#                     elif self.optim_q == 'ibr':
#                         self._q_update_ibr(train_subgroup_loss, n_classes, n_groups)
                    elif self.optim_q == 'ibr_ip':
                        self._q_update_ibr_linear_interpolation(train_subgroup_loss, n_classes, n_groups, self.n_q_update, self.total_q_update)
                    self.n_q_update+=1
                    self.q_update_term = 0

    def _update_mw_bisection(self, losses, p_train=None):

        if losses.min() < 0:
            raise ValueError

        rho = self.rho

        p_train = torch.ones(losses.shape) / losses.shape[0]
        p_train = p_train.float().cuda(device=self.device)
#        p_train = torch.from_numpy(p_train).float().cuda(device=self.device)
        if hasattr(self, 'min_prob'):
            min_prob = self.min_prob
        else:
            min_prob = 0.2
        def p(eta):
            pp = p_train * torch.relu(losses - eta)
            q = pp / pp.sum()
            cq = q / p_train
#             cq = torch.clamp(q / p_train, min=0.01)
            return cq * p_train / (cq * p_train).sum()

        def bisection_target(eta):
            pp = p(eta)
            return 0.5 * ((pp / p_train - 1) ** 2 * p_train).sum() - rho

        eta_min = -(1.0 / (np.sqrt(2 * rho + 1) - 1)) * losses.max()
        eta_max = losses.max()
        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=1000)

        q = p(eta_star)
        if hasattr(self, 'clamp_q_to_min') and self.clamp_q_to_min:
            q = torch.clamp(q, min=torch.min(self.p_train).item())
            q = q / q.sum()

#         if self.logging:
#             logger.info("EMA before-baseline losses: {}".format(
#                 " ".join(["{:.6f}".format(xx.item()) for xx in self.sum_losses[0:self.n_groups]])))
#             logger.info("EMA after-baseline losses: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in past_losses[0:self.n_groups]])))
#             logger.info("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.p_train[0:self.n_groups]])))
#             sum_weights = q[0:self.n_groups].sum().item()
#             logger.info("Group loss weights: {}".format(" ".join(["{:.6f}".format(xx.item() / sum_weights) for xx in q[0:self.n_groups]])))

#         if self.args.clear_history:
#             self.sum_losses.zero_()
        # self.count_cat.fill_(1.)

        return q        


    def _update_mw_margin(self, losses, p_train=None):

        if losses.min() < 0:
            raise ValueError

        rho = self.rho

        n_groups = len(losses)
        mean = losses.mean()
        denom = (losses - mean).norm(2)

        q = 1/n_groups + np.sqrt(2 * self.rho / n_groups)* (1/denom) * (losses - mean)
        return q


