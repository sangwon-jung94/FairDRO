import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler
import torch


class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.
    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        gamma: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.
        
    """
    def __init__(self, Dataset, batch_size, gamma, target_fairness, replacement = False, seed = 0):
        """Initializes FairBatch."""
        np.random.seed(seed)
        random.seed(seed)
        y_data = []
        z_data = []
        if 'Jigsaw' not in type(Dataset).__name__:
            for rows in Dataset.features:
                z = rows[0]
                y = rows[1]
                y_data.append(y)
                z_data.append(z)
        else:
            y_data = Dataset.y_array
            z_data = Dataset.g_array
        self.y_data = torch.FloatTensor(y_data)
        self.z_data = torch.FloatTensor(z_data)
        self.gamma = gamma
        self.fairness_type = target_fairness
        self.replacement = replacement
        
        self.N = len(Dataset)
        
        self.batch_size = batch_size
        self.n_batch = int(self.N / self.batch_size)
        
        self.n_groups = Dataset.n_groups
        self.n_labels = Dataset.n_classes
        
        if target_fairness == 'eqopp' and self.n_labels < 2:
            raise ValueError
        
        # Takes the unique values of the tensors
        self.z_item = list(range(self.n_groups))
        self.y_item = list(range(self.n_labels))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = torch.nonzero(self.z_mask[tmp_z] == 1).squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = torch.nonzero(self.y_mask[tmp_y] == 1).squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = torch.nonzero(self.yz_mask[tmp_yz] == 1).squeeze()
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        self.S = {}
        
        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz])/self.N
            
        self.S_per_label = {}
        for _l in range(self.n_labels):
            tmp = 0
            for _g in range(self.n_groups):
                key = (_l, _g)
                tmp += self.S[key]
            self.S_per_label[_l] = tmp
        
        self.lbs = {}
        for _l in range(self.n_labels):
            lbs_per_label = []
            for _g in range(self.n_groups):
                key = (_l, _g)
                lbs_per_label.append(self.S[key] / self.S_per_label[_l])
            #normalize
            lbs_per_label = [i / sum(lbs_per_label) for i in lbs_per_label]
            self.lbs[_l] = lbs_per_label
    
    def select_batch_replacement(self, batch_size, full_index, n_batch, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(n_batch):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(n_batch):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    start_idx = len(full_index)-start_idx
                else:
                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index

    
    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch. 
            
        """
#         if self.fairness_type == 'original':
            
#             entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])
            
#             sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)
#             yield 3
#             for i in range(self.batch_num):
#                 yield sort_index[i]
            
#         else:
        
#             self.adjust_lambda() # Adjust the lambda values

        each_size = {}


        # Based on the updated lambdas, determine the size of each class in a batch
        if self.fairness_type == 'eqopp': # only for binary setting
            # lb1 * loss_z1 + (1-lb1) * loss_z0

#                 each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
#                 each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,1)] = round(self.lbs[1][0] * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round((1-self.lbs[1][0]) * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(0,1)] = round(self.S[(0,1)])
            each_size[(0,0)] = round(self.S[(0,0)])

        elif self.fairness_type == 'eo':
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            tmp = 0
            for _l in range(self.n_labels):
                for _g in range(self.n_groups):
                    each_size[(_l,_g)] = round(self.lbs[_l][_g] * self.S_per_label[_l])
                    tmp +=each_size[(_l,_g)] 
            if tmp != self.batch_size:
                each_size[(_l,_g)] += self.batch_size-tmp

#                 each_size[(1,1)] = round(self.lb1 * (self.S[(1,1)] + self.S[(1,0)]))
#                 each_size[(1,0)] = round((1-self.lb1) * (self.S[(1,1)] + self.S[(1,0)]))
#                 each_size[(0,1)] = round(self.lb2 * (self.S[(0,1)] + self.S[(0,0)]))
#                 each_size[(0,0)] = round((1-self.lb2) * (self.S[(0,1)] + self.S[(0,0)]))
#                 print(each_size)

        elif self.fairness_type == 'dp': # only for binary setting
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            each_size[(1,1)] = round(self.lbs[1][1] * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(1,0)] = round(self.lbs[1][0] * (self.S[(1,1)] + self.S[(1,0)]))
            each_size[(0,1)] = round(self.lbs[0][1] * (self.S[(0,1)] + self.S[(0,0)]))
            each_size[(0,0)] = round(self.lbs[0][0] * (self.S[(0,1)] + self.S[(0,0)]))

        # Get the indices for each class
        sort_index = {}
        for _l in range(self.n_labels):
            for _g in range(self.n_groups):
                key = (_l, _g)
                sort_index[key] = self.select_batch_replacement(each_size[key], self.yz_index[key], self.n_batch, self.replacement)

#             sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.yz_index[(1,1)], self.batch_num, self.replacement)
#             sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(0, 1)], self.yz_index[(0,1)], self.batch_num, self.replacement)
#             sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.yz_index[(1,0)], self.batch_num, self.replacement)
#             sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(0, 0)], self.yz_index[(0,0)], self.batch_num, self.replacement)
        finallist = []
        for i in range(self.n_batch):
            batch = None
            for _l in range(self.n_labels):
                for _g in range(self.n_groups):
                    key = (_l, _g)                        
                    if batch is None:
                        batch = sort_index[key][i].copy()
                    else:
                        batch = np.hstack((batch, sort_index[key][i].copy()))
            key_in_fairbatch = batch.tolist()
            random.shuffle(key_in_fairbatch)
            finallist.extend(key_in_fairbatch)

#                 key_in_fairbatch = np.hstack(batch)
#                 key_in_fairbatch = sort_index_y_0_z_0[i].copy()
#                 key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
#                 key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
#                 key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))
#                 random.shuffle(list(key_in_fairbatch.astype(int)))

#         print(len(finallist))
        return iter(finallist)
#                 yield iter(key_in_fairbatch)

    def __len__(self):
        """Returns the length of data."""
        return len(self.y_data)