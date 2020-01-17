import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ColorJitter, Compose, RandomErasing,
                                    RandomGrayscale, ToPILImage, ToTensor)

from setting import CSV_PATH, DATAROOT
from util.label_util import mask_to_label


def crop_resize_data(image, label=None, image_size=[], offset=690):
    roi_image = image[offset:, :]
    if label is not None:
        roi_label = label[offset:, :]
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]),
                                 interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, (image_size[0], image_size[1]),
                                 interpolation=cv2.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]),
                                 interpolation=cv2.INTER_LINEAR)
        return train_image


class LanDataSet(Dataset):
    def __init__(self,
                 root: str = "",
                 size=[],
                 transform=None,
                 *args,
                 **kwargs):
        super(LanDataSet, self).__init__(*args, **kwargs)
        self.size = size
        self.transform = transform or ToTensor()
        self.csv = pd.read_csv(root)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        mask = row["label"]
		name = mask.split('/')[-1]
		img = cv2.imread(name,0)
        img = cv2.imread(mask,0)
        img, mask = crop_resize_data(img, mask, self.size)
        if self.transform:
            img = self.transform(img)
        label = mask_to_label(mask)
        label = one_hot(label)
        return img, torch.from_numpy(label), row['label']