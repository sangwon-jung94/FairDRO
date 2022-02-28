from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader

def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=1000):
    """Expects f an increasing function and return eta in [eta_min, eta_max]
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta
    print(eta_min, eta_max)
    # if the minimum is not reached in max_iter, returns the current value
#     logger.info('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.gamma = args.gamma # learning rate of adv_probs
        self.rho = args.rho
        self.ibr = args.ibr
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.tol = 1e-4

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        
        
        self.normal_loader = DataLoader(train_loader.dataset, 
                                        batch_size=128, 
                                        shuffle=False, 
                                        num_workers=2, 
                                        pin_memory=True, 
                                        drop_last=True)
        
        self.adv_probs_dict = {}
        for l in range(num_classes):
            self.adv_probs_dict[l] = torch.ones(num_groups).cuda() / num_groups
            
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
        
        total_loss = torch.zeros(num_subgroups).cuda(device=self.device)
        
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
            
            # calculate the labelwise losses
            group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom
#             total_loss += group_loss.detach().clone()
            
            # update q
            robust_loss = 0
            idxs = np.array([i * num_classes for i in range(num_groups)])            
            for l in range(num_classes):
                label_group_loss = group_loss[idxs+l]
#                 self.adv_probs_dict[l] = self._update_mw(label_group_loss)
                robust_loss += label_group_loss @ self.adv_probs_dict[l]
            
            robust_loss /= num_classes
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

        group_count = torch.zeros(num_subgroups).cuda(device=self.device)
        with torch.no_grad():
            for i, data in enumerate(self.normal_loader):
                # Get the inputs
                inputs, _, groups, targets, _ = data
                labels = targets

                if self.cuda:
                    inputs = inputs.cuda(device=self.device)
                    labels = labels.cuda(device=self.device)
                    groups = groups.cuda(device=self.device)

                subgroups = groups * num_classes + labels
                outputs = model(inputs)
                subgroups = groups * num_classes + labels
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                else:
                    loss = self.train_criterion(outputs, labels)
            
                # calculate the labelwise losses
                group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
                group_count += group_map.sum(1)
#                 group_denom = group_count + (group_count==0).float() # avoid nans
#                 group_loss = (group_map @ loss.view(-1))/group_denom                
                group_loss = (group_map @ loss.view(-1))                
                total_loss += group_loss.detach().clone()
            total_loss /= group_count
        
        idxs = np.array([i * num_classes for i in range(num_groups)])            
        for l in range(num_classes):
            label_group_loss = total_loss[idxs+l]
            self.adv_probs_dict[l] = self._update_mw(label_group_loss)
            print(f'{l} label loss : {total_loss[idxs+l]}')
            print(f'{l} label q values : {self.adv_probs_dict[l]}')
                
    def _update_mw(self, losses):
        
        if losses.min() < 0:
            raise ValueError

        rho = self.rho
        p_train = torch.ones(losses.shape) / losses.shape[0]
        p_train = p_train.cuda(device=self.device)

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
        

