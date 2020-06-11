import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ColorJitter, Compose, RandomErasing,
                                    RandomGrayscale, ToPILImage, ToTensor)


from util.label_util import mask_to_label


def one_hot(img):
    zs = np.array([np.zeros_like(img) for i in range(8)], dtype=np.float32)
    for i in range(8):
        zs[i][img == i] = 1
    return zs


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
                 size=[])
        super(LanDataSet, self).__init__()
        self.size = size
        self.transform = transform or ToTensor()
        

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        img, mask = row["img"], row["label"]
        img = cv2.imread(img)
        #img = jpeg.JPEG(img).decode()
        mask = cv2.imread(mask, 0)
        img, mask = crop_resize_data(img, mask, self.size)
        if self.transform:
            img = self.transform(img)
        label = mask_to_label(mask)
        label = one_hot(label)
        return img, torch.from_numpy(label), row['label']





