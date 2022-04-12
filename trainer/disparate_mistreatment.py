from __future__ import print_function
from collections import defaultdict
import time
from utils import get_accuracy
import trainer
import torch
from torch.utils.data import DataLoader
import cvxpy as cvx
import dccp
from dccp.problem import is_dccp
import numpy as np

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        if args.model != 'lr':
            raise NameError('args.model is not lr !')
        self.batch_size = args.batch_size
        self.n_workers = args.n_workers
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.mu = args.mu
        self.max_iters = args.max_iters
        self.max_iter_dccp = args.max_iter_dccp
        self.seed = args.seed
        np.random.seed(self.seed)
        
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        
        # Get parameter
        model_dict = {}
        for name, params in model.named_parameters():
            model_dict[name] = params
        keys = list(model_dict.keys())
        if len(keys) == 2 :
            w_cat = torch.cat((model_dict[keys[0]], model_dict[keys[1]].reshape(-1,1)), dim=1)
        else :
            raise NameError('Not implemented')
        
        # CVXPY formulation
        w = cvx.Variable(w_cat.shape[1])
        w.value = -(torch.sum(w_cat, dim=0)).detach().cpu().numpy()
        _, Y_train, S_train, X_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                                 n_workers=self.n_workers, model=None)
        Y_train, S_train, X_train = Y_train.detach().cpu().numpy(), S_train.detach().cpu().numpy(), X_train.detach().cpu().numpy()
        X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
        
        y_train = Y_train*2 - 1
    
        loss = cvx.sum(cvx.logistic(cvx.multiply(-y_train, X_train*w)))
        N = X_train.shape[0]
        N0_idx, N1_idx = np.where(S_train==0)[0], np.where(S_train==1)[0]
        N0, N1 = len(N0_idx), len(N1_idx)

#         # FPR
#         FPR_N0 = cvx.multiply((1-y_train[N0_idx])/2, cvx.multiply( y_train[N0_idx], X_train[N0_idx] * w))
#         FPR_N1 = cvx.multiply((1-y_train[N1_idx])/2, cvx.multiply( y_train[N1_idx], X_train[N1_idx] * w))
#         constraints = [(N0)* cvx.sum( cvx.minimum(0, FPR_N1) )<= (N1)*cvx.sum( cvx.minimum(0, FPR_N0) )+self.epsilon,
#                        (N0)* cvx.sum( cvx.minimum(0, FPR_N1) )>= (N1)*cvx.sum( cvx.minimum(0, FPR_N0) )-self.epsilon]

#         # FNR
#         FNR_N0 = cvx.multiply((1+y_train[N0_idx])/2, cvx.multiply( y_train[N0_idx], X_train[N0_idx] * w))
#         FNR_N1 = cvx.multiply((1+y_train[N1_idx])/2, cvx.multiply( y_train[N1_idx], X_train[N1_idx] * w))
#         constraints = [(N0)* cvx.sum( cvx.minimum(0, FNR_N1) )<= (N1)*cvx.sum( cvx.minimum(0, FNR_N0) )+self.epsilon,
#                        (N0)* cvx.sum( cvx.minimum(0, FNR_N1) )>= (N1)*cvx.sum( cvx.minimum(0, FNR_N0) )-self.epsilon]

        #Both
        FPR_N0 = cvx.multiply((1-y_train[N0_idx])/2.0, cvx.multiply( y_train[N0_idx], X_train[N0_idx] * w))
        FPR_N1 = cvx.multiply((1-y_train[N1_idx])/2.0, cvx.multiply( y_train[N1_idx], X_train[N1_idx] * w))
        FNR_N0 = cvx.multiply((1+y_train[N0_idx])/2.0, cvx.multiply( y_train[N0_idx], X_train[N0_idx] * w))
        FNR_N1 = cvx.multiply((1+y_train[N1_idx])/2.0, cvx.multiply( y_train[N1_idx],  X_train[N1_idx] * w))
        constraints = [(N0/N)* cvx.sum( cvx.minimum(0, FPR_N1) )<= (N1)*cvx.sum( cvx.minimum(0, FPR_N0/N) )+self.epsilon,
                       (N0/N)* cvx.sum( cvx.minimum(0, FPR_N1) )>= (N1)*cvx.sum( cvx.minimum(0, FPR_N0/N) )-self.epsilon,
                       (N0/N)* cvx.sum( cvx.minimum(0, FNR_N1) )<= (N1)*cvx.sum( cvx.minimum(0, FNR_N0/N) )+self.epsilon,
                       (N0/N)* cvx.sum( cvx.minimum(0, FNR_N1) )>= (N1)*cvx.sum( cvx.minimum(0, FNR_N0/N) )-self.epsilon]

        
        problem = cvx.Problem(cvx.Minimize(loss), constraints)
        print("Problem is DCP (disciplined convex program):", problem.is_dcp())
        print("Problem is DCCP (disciplined convex-concave program):", dccp.is_dccp(problem))
        # problem.solve(verbose=True)
       
    
        tau = self.tau # default dccp parameters, need to be varied per dataset
        mu = self.mu # default dccp parameters, need to be varied per dataset
        max_iters = self.max_iters
        max_iter_dccp = self.max_iter_dccp
        
        problem.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
            solver=cvx.SCS, verbose=False,max_iters=max_iters, max_iter=max_iter_dccp)

        w_opt = np.array(w.value)
        print('status : ', problem.status)
        
        # Parameter update
        weight = np.vstack(((-1/2)*w_opt[:-1], (1/2)*w_opt[:-1]))
        bias = np.append((-1/2)*w_opt[-1], (1/2)*w_opt[-1])
        model_dict[keys[0]] = torch.from_numpy(weight).float().cuda()
        model_dict[keys[1]] = torch.from_numpy(bias).float().cuda()
        for name, params in model.named_parameters():
            params.data.copy_(model_dict[name])
        self.model = model
        
        
        

        for epoch in range(epochs):
            # self._train_epoch(epoch, train_loader, model, criterion)
            
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

#             if self.record:
#                 self.evaluate(self.model, train_loader, self.criterion, epoch, 
#                               train=True, 
#                               record=self.record,
#                               writer=writer
#                              )
                
#             if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
#                 self.scheduler.step(eval_loss)
#             else:
#                 self.scheduler.step()
#         print('Training Finished!')        

#     def _train_epoch(self, epoch, train_loader, model, criterion=None):
#         model.train()
        
        
        
#         running_acc = 0.0
#         running_loss = 0.0
#         total = 0
#         batch_start_time = time.time()
#         for i, data in enumerate(train_loader):
#             # Get the inputs
        
#             inputs, _, groups, targets, idx = data
#             labels = targets
#             if self.cuda:
#                 inputs = inputs.cuda(device=self.device)
#                 labels = labels.cuda(device=self.device)
                
#             def closure():
#                 if self.nlp_flag:
#                     input_ids = inputs[:, :, 0]
#                     input_masks = inputs[:, :, 1]
#                     segment_ids = inputs[:, :, 2]
#                     outputs = model(
#                         input_ids=input_ids,
#                         attention_mask=input_masks,
#                         token_type_ids=segment_ids,
#                         labels=labels,
#                     )[1] 
#                 else:
#                     outputs = model(inputs)
                
#                 if criterion is not None:
#                     loss = criterion(outputs, labels).mean()
#                 else:
#                     loss = self.criterion(outputs, labels).mean()
#                 return outputs, loss
            
#             outputs, loss = closure()            
#             #loss.backward()
#             if not self.sam:
#                 if self.nlp_flag:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
#                 # self.optimizer.step()
#                 # self.optimizer.zero_grad()
#             else:
#                 # self.optimizer.first_step(zero_grad=True)
#                 outputs, loss = closure()
#                 # loss.backward()
#                 if self.nlp_flag:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
#                 self.optimizer.second_step(zero_grad=True)
                
#             running_loss += loss.item()
#             running_acc += get_accuracy(outputs, labels)
            
#             if i % self.term == self.term-1: # print every self.term mini-batches
#                 avg_batch_time = time.time()-batch_start_time
#                 print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
#                       '[{:.2f} s/batch]'.format
#                       (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
#                        avg_batch_time/self.term))

#                 running_loss = 0.0
#                 running_acc = 0.0
#                 batch_start_time = time.time()


                
    def get_statistics(self, dataset, batch_size=128, n_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)
        n_classes = dataloader.dataset.n_classes

        if model != None:
            model.eval()

        Y_pred_set = []
        X_set = []
        Y_set = []
        S_set = []
        total = 0
        for i, data in enumerate(dataloader):
            inputs, _, sen_attrs, targets, indexes = data
            X_set.append(inputs)
            Y_set.append(targets)
            S_set.append(sen_attrs)

            if self.cuda:
                inputs = inputs.cuda()
            if model != None:
                outputs = model(inputs)
                Y_pred_set.append(torch.argmax(outputs, dim=1))
            total+= inputs.shape[0]

        X_set = torch.cat(X_set)
        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda(), X_set.float().cuda()

