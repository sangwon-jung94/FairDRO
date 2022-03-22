from os.path import join
from PIL import Image

import random
import numpy as np
import pandas as pd
from torchvision import transforms
from data_handler import GenericDataset
from data_handler.utils import get_mean_std

class WaterBird(GenericDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """
    
    mean, std = get_mean_std('waterbirds')
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            (224, 224),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])    
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    name = 'waterbird'
    
    def __init__(self, **kwargs):
        
        transform = self.train_transform if kwargs['split'] == 'train' else self.test_transform

        GenericDataset.__init__(self, transform=transform, **kwargs)
        
        self.metadata_df = pd.read_csv(
            join(self.root, 'metadata.csv'))

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        
        self.split_array = self.metadata_df['split'].values
        mask = (self.split_array == self.split_dict[self.split])
        
        self.y_array = self.metadata_df['y'].values[mask]
        self.g_array = self.metadata_df['place'].values[mask]
        self.filenames = self.metadata_df['img_filename'].values[mask]
        
        self.features = [[g,y,filename] for g,y,filename in zip(self.g_array, self.y_array, self.filenames)]

        self.n_groups = 2
        self.n_classes = 2
        
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.n_groups, self.n_classes)
        
    def __getitem__(self, index):
        s, l, img_name = self.features[index]
        
        image_path = join(self.root, img_name)
        image = Image.open(image_path, mode='r').convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, 1, np.float32(s), np.int64(l), (index, img_name)


