import numpy as np
import pandas as pd
import random
import torch.utils.data as data
from data_handler.dataset_factory import GenericDataset

# class TabularDataset(data.Dataset):
class TabularDataset(GenericDataset):
    """Adult dataset."""
    # 1 idx -> sensi
    # 2 idx -> label
    # 3 idx -> filename or feature (image / tabular)
    def __init__(self, dataset, sen_attr_idx, **kwargs):
        super(TabularDataset, self).__init__(**kwargs)
        self.sen_attr_idx = sen_attr_idx
        
        dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=0)
        # features, labels = self._balance_test_set(dataset)
        self.dataset = dataset_train if self.split == 'train' else dataset_test
        
        features = np.delete(self.dataset.features, self.sen_attr_idx, axis=1)
        mean, std = self._get_mean_n_std(dataset_train.features)        
        features = (features - mean) / std
        
        self.groups = np.expand_dims(self.dataset.features[:, self.sen_attr_idx], axis=1)
        self.labels = np.squeeze(self.dataset.labels)
        
        # self.features = self.dataset.features
        self.features = np.concatenate((self.groups, self.dataset.labels, features), axis=1)

        # For prepare mean and std from the train dataset
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.n_groups, self.n_classes)
        
      #  if self.split == 'train':
      #      self._split_group()
        
    def _split_group(self):
        for l in range(self.n_classes):
            for g in range(self.n_groups):
                n_data = self.n_data[g,l]
                n_flip = 0
                for idx in self.idxs_per_group[g,l]:
                    self.features[idx][0] = 2 + g
                    n_flip += 1
                    if n_flip>(n_data/2):
                        break
        self.n_groups = 4
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.n_groups, self.n_classes)

    def get_dim(self):
        return self.dataset.features.shape[-1]

    def __getitem__(self, idx):
        features = self.features[idx]
        group = features[0]
        label = features[1]
        feature = features[2:]

        return np.float32(feature), 0, group, np.int64(label), idx

    def _get_mean_n_std(self, train_features):
        features = np.delete(train_features, self.sen_attr_idx, axis=1)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] += 1e-7
        return mean, std

