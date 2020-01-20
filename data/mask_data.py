import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ColorJitter, Compose, RandomErasing,
                                    RandomGrayscale, ToPILImage, ToTensor)

from setting import CSV_PATH, DATAROOT, PREDICT_PATH
from data.label_util import mask_to_label


class ProData(Dataset):
    def __init__(self, root: str = "", transform=None, *args, **kwargs):
        super(ProData, self).__init__(*args, **kwargs)
        self.transform = ToTensor()
        self.csv = pd.read_csv(os.path.join(CSV_PATH, 'test.csv'))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        label = row['label']
        imgName = label.split('/')[-1]
        img = cv2.imread(os.path.join(PREDICT_PATH, imgName))
        img = self.transform(img)
        mask = cv2.imread(label, 0)
        mask = mask[690:, :]
        label = mask_to_label(mask)
        return img, torch.from_numpy(label)


train_loader = DataLoader(ProData(), batch_size=1, shuffle=True, num_workers=0)
