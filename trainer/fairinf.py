from __future__ import print_function
from collections import defaultdict
import time
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import networks
import gurobipy as gp
from torch.utils.data import DataLoader
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import random
import numpy as np
from tqdm import tqdm
import sklearn.neural_network
import sklearn.linear_model
import sklearn.metrics
from scipy.linalg import cho_solve, cho_factor

import torch
from torch import nn

class IFBaseClass(ABC):
    """ Abstract base class for influence function computation on logistic regression """

    @staticmethod
    def set_sample_weight(n: int, sample_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == n
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == n
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0.
            assert max(sample_weight) <= 2.

        return sample_weight

    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True) -> np.ndarray:
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)

    @abstractmethod
    def log_loss(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
                 l2_reg: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             l2_reg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def grad_pred(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def hess(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             check_pos_def: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pred(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the predictive probability and class label """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> None:
        raise NotImplementedError


class LogisticRegression(IFBaseClass):
    """
    Logistic regression: pred = sigmoid(weight^T @ x + bias)
    Currently only support binary classification
    Borrowed from https://github.com/kohpangwei/group-influence-release
    """

    def __init__(self, l2_reg: float, fit_intercept: bool = False):
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.model = sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=(1. / l2_reg),
            fit_intercept=fit_intercept,
            tol=1e-8,
            solver="lbfgs",
            max_iter=2048,
            multi_class="ovr",
            warm_start=False,
        )

    def log_loss(self, x, y, sample_weight=None, l2_reg=False, eps=1e-16):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        pred, _, = self.pred(x)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            log_loss += self.l2_reg * np.linalg.norm(self.weight, ord=2) / 2.

        return log_loss

    def grad(self, x, y, sample_weight=None, l2_reg=False):
        """
        Compute the gradients: grad_wo_reg = (pred - y) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        indiv_grad = x * (pred - y).reshape(-1, 1)
        reg_grad = self.l2_reg * self.weight
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        if self.fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)
            reg_grad = np.concatenate([reg_grad, np.zeros(1)], axis=0)

        total_grad = np.sum(weighted_indiv_grad, axis=0)

        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)
        indiv_grad = x * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

        return total_grad, weighted_indiv_grad

    def hess(self, x, sample_weight=None, check_pos_def=False):
        """
        Compute hessian matrix: hessian = pred * (1 - pred) @ x^T @ x + lambda
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        factor = pred * (1. - pred)
        indiv_hess = np.einsum("a,ai,aj->aij", factor, x, x)
        reg_hess = self.l2_reg * np.eye(x.shape[1])

        if self.fit_intercept:
            off_diag = np.einsum("a,ai->ai", factor, x)
            off_diag = off_diag[:, np.newaxis, :]

            top_row = np.concatenate([indiv_hess, np.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = np.concatenate([off_diag, factor.reshape(-1, 1, 1)], axis=2)
            indiv_hess = np.concatenate([top_row, bottom_row], axis=1)

            reg_hess = np.pad(reg_hess, [[0, 1], [0, 1]], constant_values=0.)

        hess_wo_reg = np.einsum("aij,a->ij", indiv_hess, sample_weight)
        total_hess_w_reg = hess_wo_reg + reg_hess

        if check_pos_def:
            self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def fit(self, x, y, sample_weight=None, verbose=False):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        self.model.fit(x, y, sample_weight=sample_weight)
        self.weight: np.ndarray = self.model.coef_.flatten()
        if self.fit_intercept:
            self.bias: np.ndarray = self.model.intercept_

        if verbose:
            pred, _ = self.pred(x)
            train_loss_wo_reg = self.log_loss(x, y, sample_weight)
            reg_loss = np.sum(np.power(self.weight, 2)) * self.l2_reg / 2.
            train_loss_w_reg = train_loss_wo_reg + reg_loss

            print("Train loss: %.5f + %.5f = %.5f" % (train_loss_wo_reg, reg_loss, train_loss_w_reg))

        return

    def pred(self, x):
        return self.model.predict_proba(x)[:, 1], self.model.predict(x)


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.lamb = args.lamb

    def _vanilla_training(self, train_loader, test_loader, epochs, criterion=None):
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        model = networks.ModelFactory.get_model(self.args.model, n_classes, self.args.img_size,
                                        pretrained=self.args.pretrained, n_groups=n_groups)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        trainer_ = trainer.TrainerFactory.get_trainer('scratch', model=model, args=self.args,
                                            optimizer=optimizer, scheduler=self.scheduler)
        model = model.cuda()
        trainer_.train(train_loader, test_loader, epochs, criterion)
        return trainer_.model


    def lp(self, fair_infl, util_infl, fair_loss, alpha, beta, gamma):
        num_sample = len(fair_infl)
        max_fair = sum([v for v in fair_infl if v < 0.])
        max_util = sum([v for v in util_infl if v < 0.])

        print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f;" % (max_fair, max_util))

        all_one = np.array([1. for _ in range(num_sample)])
        fair_infl = np.array(fair_infl)
        util_infl = np.array(util_infl)
        model = gp.Model()
        x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

        if fair_loss >= -max_fair:
            print("=====> Fairness loss exceeds the maximum availability")
            model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
            model.addConstr(all_one @ x <= alpha * num_sample, name="amount")
            model.setObjective(fair_infl @ x)
            model.optimize()
        else:
            model.addConstr(fair_infl @ x <= beta * -fair_loss, name="fair")
            model.addConstr(util_infl @ x <= gamma * max_util, name="util")
            model.setObjective(all_one @ x)
            model.optimize()

        print("Total removal: %.5f; Ratio: %.3f%%\n" % (sum(x.X), (sum(x.X) / num_sample) * 100))

        return 1 - x.X

    def loss_ferm(self, loss_fn, x, y, s):
        N = x.shape[0]

        idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
        idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

        loss_grp_0_y_1 = loss_fn(x[idx_grp_0_y_1], y[idx_grp_0_y_1])
        loss_grp_1_y_1 = loss_fn(x[idx_grp_1_y_1], y[idx_grp_1_y_1])

        loss1 = np.abs((loss_grp_0_y_1 / len(idx_grp_0_y_1)) - (loss_grp_1_y_1 / len(idx_grp_1_y_1)))

        idx_grp_0_y_0 = [i for i in range(N) if s[i] == 0 and y[i] == 0]
        idx_grp_1_y_0 = [i for i in range(N) if s[i] == 1 and y[i] == 0]

        loss_grp_0_y_0 = loss_fn(x[idx_grp_0_y_0], y[idx_grp_0_y_0])
        loss_grp_1_y_0 = loss_fn(x[idx_grp_1_y_0], y[idx_grp_1_y_0])

        loss2 = np.abs((loss_grp_0_y_0 / len(idx_grp_0_y_0)) - (loss_grp_1_y_0 / len(idx_grp_1_y_0)))

        return loss1 + loss2 
    
    def grad_ferm(self, grad_fn, x, y, s):
        """
        Fair empirical risk minimization for binary sensitive attribute
        Exp(L|grp_0) - Exp(L|grp_1)
        """

        N = x.shape[0]

        idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
        idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

        grad_grp_0_y_1, _ = grad_fn(x=x[idx_grp_0_y_1], y=y[idx_grp_0_y_1])
        grad_grp_1_y_1, _ = grad_fn(x=x[idx_grp_1_y_1], y=y[idx_grp_1_y_1])

        grad1 = np.abs((grad_grp_0_y_1 / len(idx_grp_0_y_1)) - (grad_grp_1_y_1 / len(idx_grp_1_y_1)))

        idx_grp_0_y_0 = [i for i in range(N) if s[i] == 0 and y[i] == 0]
        idx_grp_1_y_0 = [i for i in range(N) if s[i] == 1 and y[i] == 0]

        grad_grp_0_y_0, _ = grad_fn(x=x[idx_grp_0_y_0], y=y[idx_grp_0_y_0])
        grad_grp_1_y_0, _ = grad_fn(x=x[idx_grp_1_y_0], y=y[idx_grp_1_y_0])

        grad2 = np.abs((grad_grp_0_y_0 / len(idx_grp_0_y_0)) - (grad_grp_1_y_0 / len(idx_grp_1_y_0)))
        return grad1 + grad2

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None, val_loader=None):
        global loss_set
        model = self.model
        model.train()

        scratch_model = self._vanilla_training(train_loader, test_loader, epochs, criterion)
        
        val_x, val_y, val_s = [], [], []
        for i, data in enumerate(val_loader):
            # Get the inputs
            inputs, _, groups, targets, idx = data
            val_x.append(inputs)
            val_y.append(targets)
            val_s.append(groups)
        val_x = torch.cat(val_x, dim=0).numpy()
        val_y = torch.cat(val_y, dim=0).numpy()
        val_s = torch.cat(val_s, dim=0).numpy()

        train_x, train_y, train_s = [], [], [] 
        tmp_train_loader =  DataLoader(train_loader.dataset, batch_size=128, shuffle=False, drop_last=False)

        for i, data in enumerate(tmp_train_loader):
            # Get the inputs
            inputs, _, groups, targets, idx = data
            train_x.append(inputs)
            train_y.append(targets)
            train_s.append(groups)
        train_x = torch.cat(train_x, dim=0).numpy()
        train_y = torch.cat(train_y, dim=0).numpy()
        train_s = torch.cat(train_s, dim=0).numpy()

        weights = self._cal_inf_score(scratch_model, val_x, val_y, val_s, train_x, train_y, train_s)
        print("Start  training")
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion, weights)
            
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
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_dcam, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        print('Vanilla Training Finished!')        


        print('Training Finished!')        

    def _cal_inf_score(self, scratch_model, x_val, y_val, s_val, x_train, y_train, s_train):
        
        weight, bias = scratch_model.head.weight.cpu().detach().numpy(), scratch_model.head.bias.cpu().detach().numpy()

        model = LogisticRegression(l2_reg=3)

        # Simulate model training to fit the necessary structure (sklearn requires this step)
        X_dummy = np.zeros((2, x_val.shape[1]))  # Dummy data with shape (n_samples, n_features)
        y_dummy = np.zeros(2)  # Dummy target
        y_dummy[1] = 1  # Set the target to 1 to avoid division by zero
        model.fit(X_dummy, y_dummy)  # Fit the model to establish `coef_` and `intercept_`

        # Set the weights (coef_) and bias (intercept_) manually
        model.model.coef_ = weight
        model.model.intercept_ = bias
        model.weight = weight
        model.bias = bias

        # print(weight.shape, bias.shape)
        
        ori_fair_loss_val = self.loss_ferm(model.log_loss, x_val, y_val, s_val)
        ori_util_loss_val = model.log_loss(x_val, y_val)

        """ compute the influence and solve lp """
        # pred_train, _ = model.pred(x_train)

        train_total_grad, train_indiv_grad = model.grad(x_train, y_train)
        util_loss_total_grad, acc_loss_indiv_grad = model.grad(x_val, y_val)
        fair_loss_total_grad = self.grad_ferm(model.grad, x_val, y_val, s_val)

        hess = model.hess(x_train)
        util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
        fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)

        util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
        fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
        # print(fair_pred_infl.shape)

        sample_weight = self.lp(fair_pred_infl, util_pred_infl, ori_fair_loss_val, self.args.alpha, self.args.beta, self.args.gamma)
        return sample_weight

        # """ train with weighted samples """

        # model.fit(x_train, y_train, sample_weight=sample_weight)

        # if args.metric == "eop":
        #     upd_fair_loss_val = loss_ferm(model.log_loss, x_val, y_val, s_val)
        # elif args.metric == "dp":
        #     pred_val, _ = model.pred(x_val)
        #     upd_fair_loss_val = loss_dp(x_val, s_val, pred_val)
        # else:
        #     raise ValueError
        # upd_util_loss_val = model.log_loss(x_val, y_val)

        # print("Fairness loss: %.5f -> %.5f; Utility loss: %.5f -> %.5f" % (
        #     ori_fair_loss_val, upd_fair_loss_val, ori_util_loss_val, upd_util_loss_val))

        # _, pred_label_val = model.pred(x_val)
        # _, pred_label_test = model.pred(x_test)

        # val_res = val_evaluator(y_val, pred_label_val)
        # test_res = test_evaluator(y_test, pred_label_test)

        # tok = time.time()
        # print("Total time: %.5fs" % (tok - tik))

        



    def _train_epoch(self, epoch, train_loader, model, criterion=None, weight=None):
        model.train()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        weight = torch.tensor(weight).to(torch.float32).cuda()
        for i, data in enumerate(train_loader):
            # Get the inputs
        
            inputs, _, groups, targets, idx = data

            # make gropus labels to onehot

            labels = targets
        
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

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

            sample_weights = weight[idx]
                
            if self.balanced:
                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                loss = loss * sample_weights
                group_loss = (group_map @ loss.view(-1))/group_denom
                # group_loss = group_loss * sample_weights
                loss = torch.mean(group_loss)
            else:
                if criterion is not None:
                    loss = criterion(outputs, labels).mean()
                else:
                    loss = self.criterion(outputs, labels).mean()
            
            loss.backward()
            if self.data == 'jigsaw':
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
                
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
    
    
