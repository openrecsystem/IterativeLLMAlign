# -*- coding: utf-8 -*-
from sysconfig import is_python_build
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json

class MovieLensDataset(Dataset):
    def __init__(self, data, meta_data, is_vec=False):
        self.data = data
        self.is_vec = is_vec
        self.is_hist = is_hist
        self.feature_offset = meta_data['feature_offset'][:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_field = ["Gender", "Age", "Job", "Zipcode", "Movie ID", "Title", "First genre"]
        offset = torch.tensor(self.feature_offset)
        input_ids = torch.tensor(row[feature_field]) + offset
        label = torch.tensor(row["labels"], dtype=torch.float32)
        if self.is_vec:
            input_embed = torch.tensor(row['embed'], dtype=torch.float32)
            return (input_ids, input_embed), label

        return input_ids, label


class ToysDataset(Dataset):
    def __init__(self, data, meta_data, is_vec=False, is_hist=False):
        self.data = data
        self.is_vec = is_vec
        self.is_hist = is_hist
        self.feature_offset = meta_data['feature_offset'][:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_field = ["Item ID", "Title", "Category", "Brand"]
        offset = torch.tensor(self.feature_offset, dtype=torch.int64)
        input_ids = torch.tensor(row[feature_field], dtype=torch.int64) + offset
        label = torch.tensor(row["labels"], dtype=torch.float32)

        if self.is_vec:
            input_embed = torch.tensor(row['embed'], dtype=torch.float32)
            return (input_ids, input_embed), label

        return input_ids, label


class BookcrossDataset(Dataset):
    def __init__(self, data, meta_data, is_vec=False, is_hist=False):
        self.data = data
        self.is_vec = is_vec
        self.is_hist = is_hist
        self.feature_offset = meta_data['feature_offset'][:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_field = ["ISBN", "Location", "Age", "Book title", "Author", "Publisher"]
        offset = torch.tensor(self.feature_offset)
        input_ids = torch.tensor(row[feature_field]) + offset
        label = torch.tensor(row["labels"], dtype=torch.float32)

        if self.is_hist:
            hist_ids = torch.tensor(eval(row["Hist"])) + offset[4]
            input_ids = torch.cat((input_ids, hist_ids))

        if self.is_vec:
            input_embed = torch.tensor(row['embed'], dtype=torch.float32)
            return (input_ids, input_embed), label

        return input_ids, label