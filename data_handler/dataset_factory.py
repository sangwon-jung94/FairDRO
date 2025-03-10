import importlib
import torch.utils.data as data
import numpy as np
from collections import defaultdict

dataset_dict = {'utkface' : ['data_handler.utkface','UTKFaceDataset'],
                'utkface_fairface' : ['data_handler.utkface_fairface','UTKFaceFairface_Dataset'],
                'celeba' : ['data_handler.celeba', 'CelebA'],
                'adult' : ['data_handler.adult', 'AdultDataset_torch'],
                'compas' : ['data_handler.compas', 'CompasDataset_torch'],                
                'cifar100s' : ['data_handler.cifar100s', 'CIFAR_100S'],   
                'cifar10s' : ['data_handler.cifar10s', 'CIFAR_10S'],   
                'waterbird' : ['data_handler.waterbird', 'WaterBird'],
                'jigsaw' : ['data_handler.jigsaw_dataset', 'JigsawDataset'],
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
   # def get_dataset(name, split='Train', seed=0, sv_ratio=1, version=1, target='Attractive', add_attr=None):
    def get_dataset(name, split='train', seed=0, target_attr='Blond_Hair', add_attr=None, balSampling=False, bs=256, uc=False, method=None, val=False):
        root = f'./data/{name}' if name != 'utkface_fairface' else './data/utkface'
        kwargs = {'root':root,
                  'split':split,
                  'seed':seed,
                  'val':val
                  }
         
        if name not in dataset_dict.keys():
            raise Exception('Not allowed method')
        
        if name == 'celeba':
            kwargs['add_attr'] = add_attr
            kwargs['target_attr'] = target_attr
        elif name == 'adult':
            kwargs['target_attr'] = target_attr
        elif name == 'compas':
            kwargs['target_attr'] = target_attr
        elif name == 'jigsaw':
            kwargs['target_name'] = target_attr
            kwargs['batch_size'] = bs
            kwargs['uc']=uc
            kwargs['method']=method
        
        
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        
#         if split == 'train' and balSampling:
#             class_.sampel_weight = class_._make_weights(method)
        return class_(**kwargs)

class GenericDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, seed=0, uc=False, val=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.n_data = None
        self.uc = uc
        self.val = val
        
    def __len__(self):
        return np.sum(self.n_data)
    
    def _data_count(self, features, n_groups, n_classes):
        idxs_per_group = defaultdict(lambda: [])
        data_count = np.zeros((n_groups, n_classes), dtype=int)
    
        if self.root == './data/jigsaw':
            for s, l in zip(self.g_array, self.y_array):
                data_count[s, l] += 1
        else:
            for idx, i in enumerate(features):
                s, l = int(i[0]), int(i[1])
                data_count[s, l] += 1
                idxs_per_group[(s,l)].append(idx)

            
        print(f'mode : {self.split}')        
        for i in range(n_groups):
            print('# of %d group data : '%i, data_count[i, :])
        return data_count, idxs_per_group
            
    def _make_data(self, features, n_groups, n_classes):
        # if the original dataset not is divided into train / test set, this function is used
        import copy
        min_cnt = 100
        data_count = np.zeros((n_groups, n_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)
        
        train_data = features
        test_data = tmp
        return train_data, test_data
    
#         for s, l, _ in self.features:
        
    def _balance_test_data(self, n_data, n_groups, n_classes):
        print('balance test data...')
        # if the original dataset is divided into train / test set, this function is used        
        n_data_min = np.min(n_data)
        print('min : ', n_data_min)
        data_count = np.zeros((n_groups, n_classes), dtype=int)
        new_features = []
        for idx, i in enumerate(self.features):
            s, l = int(i[0]), int(i[1])
            if data_count[s, l] < n_data_min:
                new_features.append(i)
                data_count[s, l] += 1
            
        return new_features

    def make_weights(self, method):
#         if method == 'lgdro_chi' and self.uc:
#             group_weights = np.zeros((self.n_groups, self.n_classes))
#             print(self.gprob_array.shape)
#             for l in range(self.n_classes):
#                 tmp = self.gprob_array[self.y_array==l].sum(axis=0)
#                 group_weights[:,l] = tmp / tmp.sum()
#             weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
#             return weights
        
        if self.root != './data/jigsaw':
            if method == 'fairhsic':
                group_weights = len(self) / self.n_data.sum(axis=0)
                weights = [group_weights[int(feature[1])] for feature in self.features]
#             elif method == 'cgdro_new':
#                 weights = self.n_data.sum(axis=0) / self.n_data
#                 weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features] 
            else:
                group_weights = len(self) / self.n_data
                weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features]
        else:
            if method == 'fairhsic':
                group_weights = len(self) / self.n_data.sum(axis=0)
                weights = [group_weights[l] for g,l in zip(self.g_array,self.y_array)]
#             elif method == 'cgdro_new':
#                 weights = self.n_data.sum(axis=0) / self.n_data
#                 weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
            else:
                group_weights = len(self) / self.n_data
                weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
        return weights 
    
            
#         elif method == 'lgdro_chi':
#             group_weights = np.zeros_like(self.n_data, dtype=np.float)            
#             for l in range(self.n_classes):
#                 group_probs = 1 / self.n_data[:, l]
#                 group_weights[:,l] = group_probs / group_probs.sum()
#                 print(group_weights[:,l])
#                 group_weights[:,l] *= self.n_data[:, l].sum()
#                 print(group_weights[:,l])
#             print(group_weights)
#             weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features]

