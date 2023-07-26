from __future__ import print_function
import torch
import time
from utils import get_accuracy
import trainer
from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.eta = args.eta
        self.iteration = args.iteration

    def train(self, train_loader, test_loader, epochs, writer=None):
        model = self.model
        model.train()
        self.n_groups = train_loader.dataset.n_groups
        self.n_classes = train_loader.dataset.n_classes

        self.extended_multipliers = torch.zeros((self.n_groups, self.n_classes))
        self.weight_matrix = self.get_weight_matrix(self.extended_multipliers)
        
        print('eta_learning_rate : ', self.eta)
        n_iters = self.iteration
        print('n_iters : ', n_iters)
        violations = 0
        for iter_ in range(n_iters):
            start_t = time.time()

            if self.data == 'jigsaw':
                assert n_iters == 1
                self.weight_update_term = 100
                self.weight_update_count = 0

            for epoch in range(epochs):
                self._train_epoch(epoch, train_loader, model)
                
                eval_start_time = time.time()                
                eval_loss, eval_acc, eval_dcam, eval_dcaa, _, _  = self.evaluate(self.model, 
                                                                                 test_loader, 
                                                                                 self.criterion,
                                                                                 epoch,
                                                                                 train=False,
                                                                                 record=self.record,
                                                                                 writer=writer
                                                                                 )
                            
                eval_end_time = time.time()
                print('[{}/{}] Method: {} '
                      'Test Loss: {:.3f} Test Acc: {:.2f} Test DCAM {:.2f} [{:.2f} s]'.format
                      (epoch + 1, epochs, self.method,
                       eval_loss, eval_acc, eval_dcam, (eval_end_time - eval_start_time)))

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
            print('Training Time : {} hours {} minutes / iter : {}/{}'.format(int(train_t / 60), (train_t % 60),
                                                                              (iter_ + 1), n_iters))
            
            if self.data != 'jigsaw':
                # get statistics
                pred_set, y_set, s_set = self.get_statistics(train_loader.dataset, bs=self.bs,
                                                                     n_workers=self.n_workers, model=model)

                # calculate violation
                if self.fairness_criterion == 'dp':
                    acc, violations = self.get_error_and_violations_DP(pred_set, y_set, s_set, self.n_groups, self.n_classes)
                elif self.fairness_criterion == 'dca':
                    acc, violations = self.get_error_and_violations_DCA(pred_set, y_set, s_set, self.n_groups, self.n_classes)

                self.extended_multipliers -= self.eta * violations 
                self.weight_matrix = self.get_weight_matrix(self.extended_multipliers) 

    def _train_epoch(self, epoch, train_loader, model):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()
            labels = labels.long()

            weights = self.weight_matrix[groups, labels]

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()
                
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
                
            if self.balanced:
                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                loss = self.train_criterion(outputs, labels)
                group_loss = (group_map @ loss.view(-1))/group_denom
                weights = self.weight_matrix.flatten().cuda()
                loss = torch.mean(group_loss*weights)
            else:
                loss = torch.mean(weights * self.train_criterion(outputs, labels))

            loss.backward()
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.data == 'jigsaw':
                self.weight_update_count += 1
                if self.weight_update_count % self.weight_update_term == 0:
                    # get statistics
                    pred_set, y_set, s_set = self.get_statistics(train_loader.dataset, bs=self.bs,
                                                                         n_workers=self.n_workers, model=model)

                    # calculate violation
                    if self.fairness_criterion == 'dp':
                        acc, violations = self.get_error_and_violations_DP(pred_set, y_set, s_set, self.n_groups, self.n_classes)
                    elif self.fairness_criterion == 'dca':
                        acc, violations = self.get_error_and_violations_DCA(pred_set, y_set, s_set, self.n_groups, self.n_classes)
                    self.extended_multipliers -= self.eta * violations 
                    self.weight_matrix = self.get_weight_matrix(self.extended_multipliers) 

            running_loss += loss.item()
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

    def get_statistics(self, dataset, bs=128, n_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)

        if model != None:
            model.eval()

        pred_set = []
        y_set = []
        s_set = []
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _, sen_attrs, targets, _ = data
                y_set.append(targets) # sen_attrs = -1 means no supervision for sensitive group
                s_set.append(sen_attrs)

                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                if model != None:
                    if self.data == 'jigsaw':
                        input_ids = inputs[:, :, 0]
                        input_masks = inputs[:, :, 1]
                        segment_ids = inputs[:, :, 2]
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=input_masks,
                            token_type_ids=segment_ids,
                            labels=targets,
                        )[1] 
                    else:
                        outputs = model(inputs)
                    pred_set.append(torch.argmax(outputs, dim=1))
                total+= inputs.shape[0]

        y_set = torch.cat(y_set)
        s_set = torch.cat(s_set)
        pred_set = torch.cat(pred_set) if len(pred_set) != 0 else torch.zeros(0)
        return pred_set.long(), y_set.long().cuda(), s_set.long().cuda()
    
    # Vectorized version for DP & multi-class
    def get_error_and_violations_DP(self, y_pred, label, sen_attrs, n_groups, n_classes):
        acc = torch.mean(y_pred == label)
        total_num = len(y_pred)
        violations = torch.zeros((n_groups, n_classes))

        for g in range(n_groups):
            for c in range(n_classes):
                pivot = len(torch.where(y_pred==c)[0])/total_num
                group_idxs=torch.where(sen_attrs == g)[0]
                group_pred_idxs = torch.where(torch.logical_and(sen_attrs == g, y_pred == c))[0]
                violations[g, c] = len(group_pred_idxs)/len(group_idxs) - pivot
        return acc, violations

    # Vectorized version for DCA & multi-class
    def get_error_and_violations_DCA(self, y_pred, label, sen_attrs, n_groups, n_classes):
        acc = torch.mean((y_pred == label).float())
        violations = torch.zeros((n_groups, n_classes)) 
        for g in range(n_groups):
            for c in range(n_classes):
                class_idxs = torch.where(label==c)[0]
                pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
                pivot = len(pred_class_idxs)/len(class_idxs)
                group_class_idxs=torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
                group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
                violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
        print('violations',violations)
        return acc, violations
    
    def get_weight_matrix(self, extended_multipliers):  
        w_matrix = torch.sigmoid(extended_multipliers) # g by c
        return w_matrix
    