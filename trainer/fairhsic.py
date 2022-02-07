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
        self.slmode = True if args.sv < 1 else False
        self.version = args.version
        #if self.slmode and self.version == 2:
        #    self.lamb = self.lamb / args.sv
        
        # self.image_transformer = networks.ModelFactory.get_model('image_transformer').cuda(self.device)
        
    def train(self, train_loader, test_loader, epochs, writer=None):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        hsic = RbfHSIC(1, 1)
        
        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, hsic=hsic, num_classes=num_classes)

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

    def _train_epoch(self, epoch, train_loader, model, hsic=None,num_classes=3):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, (idx,_) = data
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

            loss = self.criterion(stu_logits, labels)

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
            for l in range(num_classes):
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
