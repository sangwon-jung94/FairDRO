from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from utils import get_accuracy
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, teacher=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.teacher = teacher
        self.lamb = args.lamb
        self.t_device = args.t_device 
        self.sigma = args.sigma
        self.kernel = args.kernel
        
    def train(self, train_loader, test_loader, epochs, writer=None):
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups

        distiller = MMDLoss(w_m=self.lamb, sigma=self.sigma,
                            n_classes=n_classes, n_groups=n_groups, kernel=self.kernel)
        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller)

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

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, idx = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)
            
            t_inputs = inputs.to(self.t_device)
            
            if self.data == 'jigsaw':
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
                stu_logits = outputs[1]
                f_s = outputs[2][0][:,0,:]
                with torch.no_grad():
                    t_outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=labels,
                        output_hidden_states=True
                    )
                    # tea_logits = t_outputs[1]
                    f_t = t_outputs[2][0][:,0,:]

            else:
                outputs = model(inputs, get_inter=True)
                stu_logits = outputs[-1]
                f_s = outputs[-2]
                with torch.no_grad():
                    t_outputs = teacher(t_inputs, get_inter=True)
                    # tea_logits = t_outputs[-1]
                    f_t = t_outputs[-2].detach()


            loss = self.criterion(stu_logits, labels).mean()
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels)
            loss = loss + mmd_loss 
            
            loss.backward()
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            running_acc += get_accuracy(stu_logits, labels)
            running_loss += loss.item()
            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

            
class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, n_groups, n_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1)

        mmd_loss = 0
        with torch.no_grad():
            _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

        for c in range(self.n_classes):
            if len(teacher[labels==c]) == 0:
                continue
            for g in range(self.n_groups):
                if len(student[(labels==c) * (groups == g)]) == 0:
                    continue
                K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                        sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                        sigma_avg=sigma_avg, kernel=self.kernel)

                mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()
        loss = (1/2) * self.w_m * mmd_loss
        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg
    



