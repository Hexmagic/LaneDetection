import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from util.label_util import mask_to_label
from torchvision.transforms import Compose  #, ToTensor,Normalize, ColorJitter, ToPILImage
from util.img_precessing import *


def one_hot(img):
    zs = np.array([np.zeros_like(img) for i in range(8)], dtype=np.float32)
    for i in range(8):
        zs[i][img == i] = 1
    return zs


def crop_resize_data(image, label=None, image_size=[768, 256], offset=690):
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
    def __init__(self, root: str = "", transform=None, *args, **kwargs):
        super(LanDataSet, self).__init__(*args, **kwargs)
        self.transform = Compose([
            #DeformAug(),
            ScaleAug(),
            ImageAug(),
            CutOut(32, 0.5),
            #ToPILImage(),
            #ColorJitter(brightness=0.4,contrast=0.3,saturation=0.3),
            ToTensor()
        ])
        self.csv = pd.read_csv(root)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        img, mask = row["img"], row["label"]
        img = cv2.imread(img)
        mask = cv2.imread(mask, 0)
        img, mask = crop_resize_data(img, mask)
        if self.transform:
            samp = self.transform([img, mask])
        img, mask = samp['image']/255., samp['mask']
        label = mask_to_label(mask)
        label = one_hot(label)
        return img, torch.from_numpy(label)


def get_train_loader(batch_size=2):
    return DataLoader(LanDataSet("data_list/train.csv"),
                      shuffle=True,
                      batch_size=batch_size)


def get_test_loader(batch_size=2):
    return DataLoader(LanDataSet("data_list/test.csv"),
                      shuffle=True,
                      batch_size=2)


def get_valid_loader(batch_size=2):
    return DataLoader(LanDataSet("data_list/valid.csv"),
                      shuffle=True,
                      batch_size=2)
