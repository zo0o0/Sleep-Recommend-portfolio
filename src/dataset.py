import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.preprocessing import preprocess

def my_mode(x):
    if x.isna().all():
        return np.nan
    return x.value_counts().idxmax()

class SleepDataset(Dataset):
    def __init__(self, data_dir, mode, window_size, stride):
        self.window_size = window_size
        self.stride = stride
        self.X, self.y = [], []

        all_dirs = sorted(glob.glob(os.path.join(data_dir, mode, '*')))
        for dir in all_dirs:
            csv_files = sorted(glob.glob(os.path.join(dir, '*.csv')))
            for file_name in csv_files:
                X_all, y_all = preprocess(file_name)
                for i in range(0, len(X_all) - window_size + 1, stride):
                    self.X.append(X_all.iloc[i:i+window_size].values)
                    y_window = y_all.iloc[i:i + window_size]
                    mode_vals = y_window.mode()
                    y_label = mode_vals.iloc[0] if not mode_vals.empty else 0
                    self.y.append(int(y_label))

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
