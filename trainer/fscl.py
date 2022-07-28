from __future__ import print_function
from collections import defaultdict
import torch.nn.functional as F
import time
from utils import get_accuracy
import trainer
from torchvision import transforms
import torch
import PIL
import os
import torch.nn as nn
from data_handler.utils import get_mean_std
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
import torch.utils.data as data
import torch.optim as optim

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
}

            
class DatasetWrapper4fscl(data.Dataset):
    def __init__(self, dataset, dataset_name):
        self.dataset_name = dataset_name
        mean, std = get_mean_std(self.dataset_name)        
        
        self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
        
        self.transform = self.train_transform 
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        s, l, img_name = self.dataset.features[index]
        if self.dataset_name == 'celeba':
            img_name = os.path.join(self.dataset.root, "img_align_celeba", img_name)
            image = PIL.Image.open(os.path.join(self.root, "img_align_celeba", img_name))
        elif self.dataset_name == 'utkface':
            image_path = os.path.join(self.dataset.root, img_name)
            image = PIL.Image.open(image_path, mode='r').convert('RGB')
        
        return [self.transform(image), self.transform(image)], 1, np.float32(s), np.int64(l), index
        
class FairSupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, name='resnet18',head='mlp', feat_dim=128):
        super(FairSupConResNet, self).__init__()
        dim_in = model_dict[name]
        self.encoder = model
        del self.encoder.fc
        if head == 'linear':
            self.fc = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward_con(self, x):
        feat = self.encoder.forward_feature(x)
        feat = F.normalize(self.fc(feat), dim=1)
        return feat

    def forward(self, x):
        feat = self.encoder.forward_feature(x)
        return self.fc(feat)
    
    def make_linear_classifier(self, name='resnet18', n_classes=10):
        feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, n_classes).cuda()

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        
        # fisrt stage : representation learning
        def _init_fn(worker_id):
            np.random.seed(int(self.seed))
        train_dataset = DatasetWrapper4fscl(train_loader.dataset, self.dataset)
        train_loader_con = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                      num_workers=2, worker_init_fn=_init_fn,
                                      pin_memory=True, drop_last=True)
        
        self.model = FairSupConResNet(model=self.model).cuda()
        criterion = FairSupConLoss(temperature=0.1)
        criterion = criterion.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=1e-04)
        self.scheduler = MultiStepLR(self.optimizer, [60,75,90], gamma=0.1)
        
        print('start')
        for epoch in range(epochs):
            time1 = time.time()
            self._con_train_epoch(epoch, train_loader_con, self.model, criterion)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            
            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        
        n_classes = train_loader.dataset.n_classes
        self.model.make_linear_classifier(n_classes=n_classes)
        
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=1e-04)

        
#         for epoch in range(epochs):
        for epoch in range(10): # following their original setting (refer to the paper)
            self._train_epoch(epoch, train_loader, self.model,self.criterion)
            
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
                
#             if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
#                 self.scheduler.step(eval_loss)
#             else:
#                 self.scheduler.step()
        print('Training Finished!')        

    def _con_train_epoch(self,epoch, train_loader, model, criterion):
        batch_start_time = time.time()
        for idx, (images,_, sa, ta, _) in enumerate(train_loader):

            images = torch.cat([images[0], images[1]], dim=0)
            images = images.cuda(non_blocking=True)
            ta = ta.cuda(non_blocking=True)
            sa = sa.cuda(non_blocking=True)
            bsz = ta.shape[0]
      
            # compute loss
            features = model.forward_con(images)
        
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(features,ta,sa,1,'FSCL',epoch)

            # SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print info
            if idx % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = (time.time()-batch_start_time)/self.term
                print(f'Train: [{epoch}][{idx+1}/{len(train_loader)}]\t\
                      BT ({avg_batch_time})\t\
                      loss {loss.item()}'
#                       .format(
#                        avg_batch_time,
                        )
#                 sys.stdout.flush()

    
    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.eval()
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
        
            inputs, _, groups, targets, idx = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                
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
                    with torch.no_grad():
                        features = model.encoder.forward_feature(inputs)
                    outputs = model.fc(features)
                
                if criterion is not None:
                    loss = criterion(outputs, labels).mean()
                else:
                    loss = self.criterion(outputs, labels).mean()
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
            running_acc += get_accuracy(outputs, labels)
            
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, 10, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
                

class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    
    def forward(self, features, labels,sensitive_labels,group_norm,method, epoch,mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: target classes of shape [bsz].
            sensitive_labels: sensitive attributes of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        n_sensitive_mask=(~sensitive_mask.bool()).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if method=="FSCL":
            mask = mask * logits_mask
            logits_mask_fair=logits_mask*(~mask.bool()).float()*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        elif method=="FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
                
            logits_mask_fair=logits_mask*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
              
        # compute mean of log-likelihood over positive
        #apply group normalization
        if group_norm==1:
            mean_log_prob_pos = ((mask*log_prob)/((mask*sensitive_mask).sum(1))).sum(1)
           
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm==1:
            C=loss.size(0)/8
            norm=(1/(((mask*sensitive_mask).sum(1)+1).float()))
            loss=(loss*norm)*C
            
        loss = loss.view(anchor_count, batch_size).mean()
        return loss




