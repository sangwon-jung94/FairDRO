from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import get_accuracy
import trainer
from .hsic import RbfHSIC
import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lamb
        self.sigma = args.sigma
        self.kernel = args.kernel
        
        # self.image_transformer = networks.ModelFactory.get_model('image_transformer').cuda(self.device)
        
    def train(self, train_loader, test_loader, epochs, writer=None):
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups

        hsic = RbfHSIC(1, 1)
        
        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, hsic=hsic, n_classes=n_classes)

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

    def _train_epoch(self, epoch, train_loader, model, hsic=None,n_classes=3):
        model.train()

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
            # inputs_transformed = self.image_transformer(inputs)

            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs, get_inter=True)
            # outputs_transformed = model(inputs_transformed, get_inter=True)

            stu_logits = outputs[-1]
            # stu_logits = outputs_transformed[-1]

            loss = self.criterion(stu_logits, labels).mean()

            running_acc += get_accuracy(stu_logits, labels)

            f_s = outputs[-2]
            # f_s_transformed = outputs_transformed[-2]

            # if self.slmode and self.version == 2:
            #    idxs = groups != -1 
            #    f_s = f_s[idxs]
            #    f_t = f_t[idxs]
            #    groups = groups[idxs]
            #    labels = labels[idxs]
            
            group_onehot = F.one_hot(groups).float()
            hsic_loss = 0
            for l in range(n_classes):
                mask = targets == l
                hsic_loss += hsic.unbiased_estimator(f_s[mask], group_onehot[mask])
            # hsic_loss = hsic.unbiased_estimator(f_s-f_s_transformed, group_onehot)
            # hsic_loss2 = hsic.unbiased_estimator(f_s_transformed, group_onehot)

            loss = loss + self.lamb * hsic_loss 
            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

