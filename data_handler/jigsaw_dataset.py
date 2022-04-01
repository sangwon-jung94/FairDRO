import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# from data.confounder_dataset import ConfounderDataset
from data_handler.dataset_factory import GenericDataset
from transformers import AutoTokenizer, BertTokenizer


class JigsawDataset(GenericDataset):
    """
    Jigsaw dataset. We only consider the subset of examples with identity annotations.
    Labels are 1 if target_name > 0.5, and 0 otherwise.

    95% of tokens have max_length <= 220, and 99.9% have max_length <= 300
    """

    def __init__(
        self,
        target_name,
#         confounder_names,
        batch_size=None,
        **kwargs
    ):
        
        GenericDataset.__init__(self, **kwargs)
        
        # def __init__(self, args):
        self.dataset_name = "jigsaw"
        # self.aux_dataset = args.aux_dataset
        self.target_name = target_name
        self.confounder_names = ['black', 'white', 'asian','latino', 'other_race_or_ethnicity']
#         self.confounder_names = ['christian', 'jewish', 'muslim']
#         self.confounder_names = ['male', 'female', 'homosexual_gay_or_lesbian']
        self.model = "bert-base-uncased"

        if batch_size == 32:
            self.max_length = 128
        elif batch_size == 24:
            self.max_length = 220
        elif batch_size == 16:
            self.max_length = 300
        else:
            assert False, "Invalid batch size"

        # Read in metadata
        data_filename = "all_data_with_identities.csv"
        print("metadata_csv_name:", data_filename)
        
        self.metadata_df = pd.read_csv(
            os.path.join(self.root, data_filename), index_col=0
        )
        
        # split mask
        split_mask = (self.metadata_df["split"]==self.split).values
        
        # Get the y values
        self.y_array = (self.metadata_df[self.target_name].values >= 0.5).astype("long")
        self.y_array = self.y_array[split_mask]
        self.n_classes = len(np.unique(self.y_array))
        print('confounder : ' , self.confounder_names)

        
        # Map to groups
        attr = (self.metadata_df.loc[self.metadata_df["split"]==self.split,self.confounder_names]).values
        max_val = attr.max(axis=1)
        argmax_val = attr.argmax(axis=1)
        mask = max_val>=0.1
        self.g_array = argmax_val[mask]
        self.n_groups = len(self.confounder_names)
        
        self.y_array = self.y_array[mask]
        
        
        """
        # Confounders are all binary
        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.n_confounders = len(self.confounder_names)
        confounders = (self.metadata_df.loc[:, self.confounder_names] >= 0.5).values
        self.confounder_array = confounders @ np.power(
            2, np.arange(self.n_confounders)
        )
        """

        # Extract text
        self.text_array = self.metadata_df.loc[self.metadata_df["split"]==self.split, "comment_text"]
        self.text_array = list(self.text_array[mask])
        self.tokenizer = BertTokenizer.from_pretrained(self.model)

        self.n_data, _ = self._data_count(None, self.n_groups, self.n_classes)

#     def __len__(self):
#         return len(self.y_array)

    def get_group_array(self):
        return self.g_array

    def get_label_array(self):
        return self.y_array

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.g_array[idx]

        text = self.text_array[idx]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )  # 220
        x = torch.stack(
            (tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]),
            dim=2,
        )
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1

        return x, 1, np.float32(g), np.int64(y), idx

    def group_str(self, group_idx):
        if self.n_groups == self.n_classes:
            y = group_idx
            group_name = f"{self.target_name} = {int(y)}"
        else:
            y = group_idx // (self.n_groups / self.n_classes)
            c = group_idx % (self.n_groups // self.n_classes)

            group_name = f"{self.target_name} = {int(y)}"
            bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
            for attr_idx, attr_name in enumerate(self.confounder_names):
                group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name

