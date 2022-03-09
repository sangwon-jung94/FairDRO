from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy, chi_proj, chi_proj_nonuni
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
                                        drop_last=False)
        
        self.adv_probs_dict = {}
        self.group_dist = {}        
        for l in range(num_classes):
            num_data = train_loader.dataset.num_data[:,l]
            self.group_dist[l] = num_data / num_data.sum()
            print(num_data, self.group_dist[l])
            self.adv_probs_dict[l] = torch.from_numpy(self.group_dist[l]).float().cuda(device=self.device)
#             self.adv_probs_dict[l] = torch.ones(num_groups).cuda(device=self.device) / num_groups

            
        for epoch in range(epochs):
            
            self._train_epoch(epoch, train_loader, model,criterion)            
            
            
            # gradient ascent
            train_loss, train_acc, train_deom, train_deoa, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, train=True)
            idxs = np.array([i * num_classes for i in range(num_groups)])  
            
#             for l in range(num_classes):
#                 label_group_loss = train_subgroup_loss[idxs+l]
# #                 label_group_loss = 1-train_subgroup_acc[:,l]
#                 print(label_group_loss)
#                 print(self.adv_probs_dict[l])
#                 self.adv_probs_dict[l] *= torch.exp(self.gamma*label_group_loss)
#                 print(self.adv_probs_dict[l])
# #                 self.adv_probs_dict[l] = torch.from_numpy(chi_proj(self.adv_probs_dict[l], self.rho)).cuda(device=self.device).float()
#                 self.adv_probs_dict[l] = torch.from_numpy(chi_proj_nonuni(self.adv_probs_dict[l], self.rho, self.group_dist[l])).cuda(device=self.device).float()
            self._q_update(train_subgroup_loss, num_classes, num_groups)            



            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_subgroup_acc, eval_subgroup_loss  = self.evaluate(self.model, test_loader, self.train_criterion, train=False)
            eval_end_time = time.time()            

            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))
            
            if self.record:
                train_loss, train_acc, train_deom, train_deoa, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, train=True)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('train_acc', train_acc, epoch)
                writer.add_scalar('train_deom', train_deom, epoch)
                writer.add_scalar('train_deoa', train_deoa, epoch)
                writer.add_scalar('eval_loss', eval_loss, epoch)
                writer.add_scalar('eval_acc', eval_acc, epoch)
                writer.add_scalar('eval_deom', eval_deom, epoch)
                writer.add_scalar('eval_deoa', eval_deoa, epoch)
                
                eval_subgroup_loss = eval_subgroup_loss.reshape((num_groups, num_classes))
                train_subgroup_loss = train_subgroup_loss.reshape((num_groups, num_classes))
                
                eval_contents_acc = {}
                train_contents_acc = {}
                eval_contents_loss = {}
                train_contents_loss = {}
                q_values = {}
                
                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_contents_acc[f'g{g},l{l}'] = eval_subgroup_acc[g,l]
                        train_contents_acc[f'g{g},l{l}'] = train_subgroup_acc[g,l]
                        eval_contents_loss[f'g{g},l{l}'] = eval_subgroup_loss[g,l]
                        train_contents_loss[f'g{g},l{l}'] = train_subgroup_loss[g,l]
                        q_values[f'g{g},l{l}'] = self.adv_probs_dict[l][g]

                writer.add_scalars('eval_subgroup_acc', eval_contents_acc, epoch)
                writer.add_scalars('train_subgroup_acc', train_contents_acc, epoch)
                writer.add_scalars('eval_subgroup_loss', eval_contents_loss, epoch)
                writer.add_scalars('train_subgroup_loss', train_contents_loss, epoch)
                writer.add_scalars('q_values', q_values, epoch)                
                
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
        
        idxs = np.array([i * num_classes for i in range(num_groups)])            
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
            
#            train_loss, train_acc, train_deom, train_deoa, train_subgroup_acc, train_subgroup_loss = self.evaluate(self.model, self.normal_loader, self.train_criterion, train=True)
            
            # update q
            robust_loss = 0
            idxs = np.array([i * num_classes for i in range(num_groups)])            
            for l in range(num_classes):
                label_group_loss = group_loss[idxs+l]
#                 self.adv_probs_dict[l] *= torch.exp(self.gamma*label_group_loss).detach()
#                 self.adv_probs_dict[l] = torch.from_numpy(chi_proj(self.adv_probs_dict[l], self.rho)).cuda(device=self.device).float()
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

        
    def _q_update(self, losses, num_classes, num_groups):
        assert len(losses) == (num_classes * num_groups)
        
        idxs = np.array([i * num_classes for i in range(num_groups)])
        
        for l in range(num_classes):
            label_group_loss = losses[idxs+l]
            self.adv_probs_dict[l] = self._update_mw(label_group_loss, self.group_dist[l])
            print(f'{l} label loss : {losses[idxs+l]}')
            print(f'{l} label q values : {self.adv_probs_dict[l]}')
                
    def _update_mw(self, losses, p_train):
        
        if losses.min() < 0:
            raise ValueError

        rho = self.rho
        
#         p_train = torch.ones(losses.shape) / losses.shape[0]
        p_train = torch.from_numpy(p_train).float().cuda(device=self.device)
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
        
    def evaluate(self, model, loader, criterion, device=None, train=False):

        if not train:
            model.eval()
        else:
            model.train()
            
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        num_subgroups = num_groups * num_classes
        device = self.device if device is None else device

        eval_eopp_list = torch.zeros(num_subgroups).cuda(device)
#         eval_data_count = torch.zeros(num_subgroups).cuda(device)
        
        group_count = torch.zeros(num_subgroups).cuda(device=self.device)
        group_loss = torch.zeros(num_subgroups).cuda(device=self.device)        
        group_acc = torch.zeros(num_subgroups).cuda(device=self.device)        
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, targets, _ = data
                labels = targets

                if self.cuda:
                    inputs = inputs.cuda(device=self.device)
                    labels = labels.cuda(device=self.device)
                    groups = groups.cuda(device=self.device)

                subgroups = groups * num_classes + labels
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float()
            
                # calculate the labelwise losses
                group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
                group_count += group_map.sum(1)
                
                group_loss += (group_map @ loss.view(-1))
                group_acc += group_map @ acc
                
#                 for g in range(num_groups):
#                     for l in range(num_classes):
#                         eval_eopp_list += acc[(groups == g) * (labels == l)].sum()
#                         eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

            eval_loss = group_loss.sum() / group_count.sum() 
            eval_acc = group_acc.sum() / group_count.sum() 

            group_loss /= group_count        
            group_acc /= group_count
            
            group_acc = group_acc.reshape((num_groups, num_classes))
            labelwise_acc_gap = torch.max(group_acc, dim=0)[0] - torch.min(group_acc, dim=0)[0]
            eval_avg_eopp = torch.mean(labelwise_acc_gap).item()
            eval_max_eopp = torch.max(labelwise_acc_gap).item()
        model.train()
        
        return eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, group_acc, group_loss

