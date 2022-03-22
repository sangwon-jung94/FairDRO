import pandas as pd
from data_handler.AIF360.compas_dataset import CompasDataset
from data_handler.tabular_dataset import TabularDataset


class CompasDataset_torch(TabularDataset):
    """Compas dataset."""
    name = 'compas'
    def __init__(self, root, target_attr='race', **kwargs):

        dataset = CompasDataset(root_dir=root)
        if target_attr == 'sex':
            sen_attr_idx = 0
        elif target_attr == 'race':
            sen_attr_idx = 2
        else:
            raise Exception('Not allowed group')

        self.n_groups = 2
        self.n_classes = 2

        super(CompasDataset_torch, self).__init__(root=root, dataset=dataset, sen_attr_idx=sen_attr_idx,
                                                  **kwargs)

