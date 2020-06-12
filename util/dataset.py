import cv2
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import (ColorJitter, Compose, RandomErasing,
                                    RandomGrayscale, ToPILImage, ToTensor)
from util.label_util import mask_to_label
import torch.nn.functional as F


def one_hot(img):
    zs = np.array([np.zeros_like(img) for i in range(8)], dtype=np.float32)
    for i in range(8):
        zs[i][img == i] = 1
    return zs


class LaneDataSet(Dataset):
    def __init__(self, mode='train'):
        super(LaneDataSet, self).__init__()
        self.mode = mode
        self.batch_cnt = 0
        self.wid = 846
        self.hei = 255
        self.min_size = 846
        self.max_size = 846+2
        self.ratio = 846 / 255
        if self.mode == 'train':
            self.transform = Compose([
                ToPILImage(),
                ColorJitter(0.2, 0.2, 0.2),
                ToTensor(),
                RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(1, 2.5)),
            ])
        with open(f'data/{mode}.txt', 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        row = self.lines[index].strip()
        img = cv2.imread(f'data/images/{row.replace(".png",".jpg")}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f'data/labels/{row}', 0)
        img, mask = img[690:, :, :], mask[690:, :]
        if self.transform:
            img = self.transform(img)
        label = mask_to_label(mask)
        label = one_hot(label)
        return img, torch.Tensor(label)

    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size,
                              mode="bilinear",align_corners=True).squeeze(0)
        return image

    def collate_fn(self, batch):
        #import pdb; pdb.set_trace()
        imgs, labels = list(zip(*batch))
        if self.batch_cnt % 10 == 0:
            self.wid = random.choice(range(self.min_size, self.max_size))
            self.hei = int(self.wid / self.ratio)
        imgs = torch.stack(
            [self.resize(img, (self.hei, self.wid)) for img in imgs])
        labels = torch.stack(
            [self.resize(img, (self.hei, self.wid)) for img in labels])
        self.batch_cnt += 1
        return imgs, labels


if __name__ == '__main__':
    data = LaneDataSet()
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=2, collate_fn=data.collate_fn,num_workers=2)
    for batch in loader:
        print(batch[0].shape)
