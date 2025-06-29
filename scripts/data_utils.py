import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler
import torch

class SolarDataset(Dataset):
    def __init__(self, df, seq_len, label_len, pred_len, target_col='FLUX'):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.data = df[[target_col]].values
        self.targets = df['FLARE_TARGET_24h'].values

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        target = self.targets[s_end + self.pred_len - 1]

        return (
            torch.tensor(seq_x, dtype=torch.float),
            torch.tensor(seq_y, dtype=torch.float),
            torch.tensor(target, dtype=torch.float)
        )

def load_data(file_path, seq_len, label_len, pred_len, distributed=False):
    df = pd.read_csv(file_path, parse_dates=['DATETIME'])
    train_df = df[df['DATETIME'] < '2025-03-01']
    test_df = df[df['DATETIME'] >= '2025-03-01']

    train_set = SolarDataset(train_df, seq_len, label_len, pred_len)
    test_set = SolarDataset(test_df, seq_len, label_len, pred_len)

    train_sampler = DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(train_set, batch_size=32, shuffle=(train_sampler is None), sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader

