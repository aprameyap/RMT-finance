import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class SolarFlareDataset(Dataset):
    def __init__(self, df, encoder_len=288):
        self.encoder_len = encoder_len
        self.data = []
        self.labels = []
        le = LabelEncoder()
        df['FLARE_TARGET_24h'] = le.fit_transform(df['FLARE_TARGET_24h'])

        for i in range(len(df) - encoder_len):
            x = df.iloc[i:i+encoder_len]['FLUX_SMOOTH'].values.astype('float32')
            y = df.iloc[i+encoder_len]['FLARE_TARGET_24h']
            self.data.append(torch.tensor(x).unsqueeze(-1))  # (seq_len, 1)
            self.labels.append(torch.tensor(y).long())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

df = pd.read_csv("data/processed_solexs_tft_ready.csv", parse_dates=["DATETIME"])
split_idx = int(0.8 * len(df))

train_dataset = SolarFlareDataset(df.iloc[:split_idx])
val_dataset = SolarFlareDataset(df.iloc[split_idx:])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
