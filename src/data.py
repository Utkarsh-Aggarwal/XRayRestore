import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_gray(path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


class XrayDataset(Dataset):
    def __init__(self, splits_csv, split, size=256, augment=False):
        df = pd.read_csv(splits_csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = torch.from_numpy(load_gray(self.df.path[i], self.size))[None]
        if self.augment and torch.rand(1) < 0.5:
            x = x.flip(-1)
        return x, i
