from __future__ import print_function
from collections import defaultdict

import copy
import time
from utils import get_accuracy, chi_proj
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.rho = args.rho 
        self.optim_q = args.optim_q

        # when using gradient ascent for q
        self.gamma = args.gamma # learning rate of adv_probs
        self.tol = 1e-4

        self.data = args.dataset

        self.use_01loss = args.use_01loss
        self.q_decay = args.q_decay    

        self.update_freq = 100 # only for jigsaw
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        
        global loss_set
        model = self.model
        model.train()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        
        self.normal_loader = DataLoader(train_loader.dataset, 
                                        batch_size=128, 
                                        shuffle=False, 
                                        num_workers=2, 
                                        pin_memory=True, 
                                        drop_last=False)
        
        self.q_dict = torch.ones(n_groups*n_classes).cuda()
        
        if self.data == 'jigsaw':
            self.n_q_update = 0
            self.q_update_term = 0
            self.total_q_update = (epochs * len(train_loader)) / self.update_freq

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion)            
            if self.data != 'jigsaw' or self.record:
                _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, 
                                                                                    self.normal_loader,
                                                                                    self.train_criterion, 
                                                                                    epoch,
                                                                                    train=True,
                                                                                    record=self.record,
                                                                                    writer=writer
                                                                                    )
                if self.use_01loss:
                    train_subgroup_loss = 1-train_subgroup_acc

                # q update
                if self.optim_q == 'pd':
                    self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
                elif self.optim_q == 'ibr':
                    self.q_dict = self._q_update_ibr(self.q_dict, train_subgroup_loss, n_classes, n_groups)
                elif self.optim_q == 'smt_ibr':
                    self.q_dict, opt_q = self._q_update_ibr_linear_interpolation(self.q_dict, 
                                                                                 train_subgroup_loss, 
                                                                                 n_classes,
                                                                                 n_groups, 
                                                                                 epoch, 
                                                                                 epochs
                                                                                )

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
                q_values = {}
                
                for g in range(n_groups):
                    for l in range(n_classes):
                        q_values[f'g{g},l{l}'] = self.q_dict[l][g]
                writer.add_scalars('q_values', q_values, epoch)

                if self.optim_q == 'smt_ibr':
                    opt_q_values = {}
                    for g in range(n_groups):
                        for l in range(n_classes):
                            opt_q_values[f'g{g},l{l}'] = opt_q[l][g]
                    writer.add_scalars('opt_q_values', opt_q_values, epoch)
                
                if not self.use_01loss:
                    q_values_true, opt_q_true =self._q_update_ibr_linear_interpolation(self.q_dict, 1-train_subgroup_acc, n_classes, n_groups, epoch, epochs)
                    q_values = {}
                    for g in range(n_groups):
                        for l in range(n_classes):
                            q_values[f'g{g},l{l}'] = q_values_true[l][g]
                    writer.add_scalars('q_values_true', q_values, epoch)

                    opt_q_values = {}
                    for g in range(n_groups):
                        for l in range(n_classes):
                            opt_q_values[f'g{g},l{l}'] = opt_q_true[l][g]
                    writer.add_scalars('opt_q_values_true', opt_q_values, epoch)
                
            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
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
                
            subgroups = groups * n_classes + labels
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

            if criterion is not None:
                loss = criterion(outputs, labels)
            else:
                loss = self.train_criterion(outputs, labels)

            # calculate the balSampling losses
            group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom
            # group_loss = group_loss.reshape([n_groups, n_classes])
            robust_loss = 0
            # for l in range(n_classes):
            robust_loss += group_loss @ self.q_dict
            robust_loss /= (n_classes*n_groups)        
            self.optimizer.zero_grad()
            robust_loss.backward()                
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            running_loss += robust_loss.item()
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
                
            if self.data == 'jigsaw':
                self.q_update_term += 1
                if self.q_update_term % self.update_freq == 0:
                    print('lets start')
                    start = time.time()
                    _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, 
                                                                       epoch,
                                                                       train=True,
                                                                       record=False,
                                                                       writer=None
                                                                      )
                    end = time.time()

                    if self.use_01loss:
                        train_subgroup_loss = 1-train_subgroup_acc
                    # q update
                    if self.optim_q == 'pd':
                        self._q_update_pd(train_subgroup_loss, n_classes, n_groups)
                    elif self.optim_q == 'ibr':
                        self.q_dict = self._q_update_ibr(self.q_dict,
                                                         train_subgroup_loss, 
                                                         n_classes,
                                                         n_groups)
                    elif self.optim_q == 'smt_ibr':
                        self.q_dict, opt_q = self._q_update_ibr_linear_interpolation(self.q_dict, 
                                                                                     train_subgroup_loss, 
                                                                                     n_classes, 
                                                                                     n_groups, 
                                                                                     self.n_q_update, 
                                                                                     self.total_q_update
                                                                                     )
                    self.n_q_update += 1
                    self.q_update_term = 0
                

    def _q_update_ibr(self, q_dict, losses, n_classes, n_groups):
        opt_q = {}
        for l in range(n_classes):
            label_group_loss = losses[:,l]
            opt_q[l] = self._update_mw_margin(label_group_loss)
            print(f'{l} label loss : {label_group_loss}')
            print(f'{l} label q values : {q_dict[l]}')
        return opt_q
    
    def _q_update_pd(self, train_subgroup_loss, n_classes, n_groups):
        # train_subgroup_loss = torch.flatten(train_subgroup_loss)
        
        idxs = np.array([i * n_classes for i in range(n_groups)])  
            
        for l in range(n_classes):
            label_group_loss = train_subgroup_loss[idxs+l]
            self.adv_probs_dict[l] *= torch.exp(self.gamma*label_group_loss)
            self.adv_probs_dict[l] = torch.from_numpy(chi_proj(self.adv_probs_dict[l], self.rho)).cuda(device=self.device).float()

#                self.adv_probs_dict[l] = torch.from_numpy(chi_proj_nonuni(self.adv_probs_dict[l], self.rho, self.group_dist[l])).cuda(device=self.device).float()
#            self._q_update(train_subgroup_loss, n_classes, n_groups)            

    def _q_update_ibr_linear_interpolation(self, q_dict, subgroup_loss, n_classes, n_groups, epoch, epochs):
        if self.q_decay == 'cos': 
            cur_step_size = 0.5 * (1 + np.cos(np.pi * (epoch/epochs)))

        elif self.q_decay == 'linear':
            cur_step_size = 1 - epoch/epochs
        subgroup_loss = subgroup_loss.flatten()
        opt_q = self._update_mw_margin(subgroup_loss)
        q_dict = q_dict + cur_step_size*(opt_q - q_dict)
        print(f'loss : {subgroup_loss}')
        print(f'q values : {q_dict}')
        return q_dict, opt_q

#     Deprecated    
#     def _update_mw_bisection(self, losses, p_train=None):
        
#         if losses.min() < 0:
#             raise ValueError

#         rho = self.rho
        
#         p_train = torch.ones(losses.shape) / losses.shape[0]
#         p_train = p_train.float().cuda(device=self.device)
# #        p_train = torch.from_numpy(p_train).float().cuda(device=self.device)
#         if hasattr(self, 'min_prob'):
#             min_prob = self.min_prob
#         else:
#             min_prob = 0.2
#         def p(eta):
#             pp = p_train * torch.relu(losses - eta)
#             q = pp / pp.sum()
#             cq = q / p_train
# #             cq = torch.clamp(q / p_train, min=0.01)
#             return cq * p_train / (cq * p_train).sum()

#         def bisection_target(eta):
#             pp = p(eta)
#             return 0.5 * ((pp / p_train - 1) ** 2 * p_train).sum() - rho

#         eta_min = -(1.0 / (np.sqrt(2 * rho + 1) - 1)) * losses.max()
#         eta_max = losses.max()
#         eta_star = bisection(
#             eta_min, eta_max, bisection_target,
#             tol=self.tol, max_iter=1000)

#         q = p(eta_star)
#         if hasattr(self, 'clamp_q_to_min') and self.clamp_q_to_min:
#             q = torch.clamp(q, min=torch.min(self.p_train).item())
#             q = q / q.sum()

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

        # return q        


    def _update_mw_margin(self, losses, p_train=None):
        
        if losses.min() < 0:
            raise ValueError

        rho = self.rho
        
        n_groups = len(losses)
        mean = losses.mean()
        denom = (losses - mean).norm(2)
        if denom == 0:
            q = torch.zeros_like(losses) + 1/n_groups
        else:
            q = 1/n_groups + np.sqrt(2 * self.rho / n_groups)* (1/denom) * (losses - mean)
        return q
        
# Deprecated

# def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=1000):
#     """Expects f an increasing function and return eta in [eta_min, eta_max]
#     s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
#     lower = f(eta_min)
#     upper = f(eta_max)

#     # until the root is between eta_min and eta_max, double the length of the
#     # interval starting at either endpoint.
#     while lower > 0 or upper < 0:
#         length = eta_max - eta_min
#         if lower > 0:
#             eta_max = eta_min
#             eta_min = eta_min - 2 * length
#         if upper < 0:
#             eta_min = eta_max
#             eta_max = eta_max + 2 * length

#         lower = f(eta_min)
#         upper = f(eta_max)

#     for _ in range(max_iter):
#         eta = 0.5 * (eta_min + eta_max)

#         v = f(eta)

#         if torch.abs(v) <= tol:
#             return eta

#         if v > 0:
#             eta_max = eta
#         elif v < 0:
#             eta_min = eta
#     # if the minimum is not reached in max_iter, returns the current value
# #     logger.info('Maximum number of iterations exceeded in bisection')
#     return 0.5 * (eta_min + eta_max)
