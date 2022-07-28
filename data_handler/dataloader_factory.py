from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, batch_size=256, seed = 0, n_workers=4,
                       target_attr='Blond_Hair', add_attr=None, labelwise=False, args=None):
        if name == 'adult':
            target_attr = 'sex'
        elif name == 'compas':
            target_attr = 'race'
        elif name == 'jigsaw' : 
            target_attr = 'toxicity'

        test_dataset = DatasetFactory.get_dataset(name, split='test',
                                                  target_attr=target_attr, seed=seed, add_attr=add_attr, bs=batch_size,uc=args.uc,method=args.method)
        train_dataset = DatasetFactory.get_dataset(name, split='train',
                                                   target_attr=target_attr, seed=seed,add_attr=add_attr, bs=batch_size,uc=args.uc,method=args.method)
        
        n_classes = test_dataset.n_classes
        n_groups = test_dataset.n_groups
        
        def _init_fn(worker_id):
            np.random.seed(int(seed))

        shuffle = True
        sampler = None
        if labelwise:
#             if args.method == 'mfd':
#                 from data_handler.custom_loader import Customsampler                
#                 sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
#             else:
            from torch.utils.data.sampler import WeightedRandomSampler
            weights = train_dataset.make_weights(args.method)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#             else:
#                 from data_handler.custom_loader import Customsampler                
#                 sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
            shuffle = False
        elif args.method == 'fairbatch':
            from data_handler.fairbatch import FairBatch
            sampler = FairBatch(train_dataset, batch_size, gamma=args.gamma, target_fairness='eo', seed=seed)
            shuffle = False

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      num_workers=n_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)

        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                     num_workers=n_workers, worker_init_fn=_init_fn, pin_memory=True)

        print('# of test data : {}'.format(len(test_dataset)))
        print('# of train data : {}'.format(len(train_dataset)))
        print('Dataset loaded.')
        print('# of classes, # of groups : {}, {}'.format(n_classes, n_groups))

        return n_classes, n_groups, train_dataloader, test_dataloader

