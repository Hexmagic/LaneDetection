import pdb
import cv2
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomErasing,
    RandomGrayscale,
    ToPILImage,
    ToTensor,
)
from util.label_util import mask_to_label
import torch.nn.functional as F
from glob import glob


def one_hot(img):
    zs = np.array([np.zeros_like(img) for i in range(8)], dtype=np.float32)
    for i in range(8):
        zs[i][img == i] = 1
    return zs


class LaneDataSet(Dataset):
    def __init__(self, mode="train", root='data', multi_scale=False, wid=846):
        super(LaneDataSet, self).__init__()
        self.mode = mode
        self.batch_cnt = 0
        self.ratio = 846 / 255.01
        self.wid = wid
        self.multi_scale = multi_scale
        self.hei = int(wid / self.ratio)
        self.min_size = int(wid * 0.8)
        self.max_size = int(wid * 1.2)
        imgs = list(glob(f'{root}/**/*.jpg',recursive=True))
        pos = int(len(imgs) * 0.8)
        if self.mode == "train":
            self.transform = Compose([
                ToPILImage(),
                ColorJitter(0.4, 0.3, 0.3),
                ToTensor(),
                RandomErasing(p=0.5, scale=(0.03, 0.2), ratio=(0.1, 5)),
            ])
            self.files = imgs[:pos]
        else:
            self.files = imgs[pos:]
            self.transform = ToTensor()

    def __len__(self):
        return len(self.files)

    def resize_img(self, img):
        img = cv2.resize(img, (self.wid, self.hei),
                         interpolation=cv2.INTER_LINEAR)
        return img

    def resize_mask(self, mask):
        mask = cv2.resize(mask, (self.wid, self.hei),
                          interpolation=cv2.INTER_NEAREST)
        return mask

    def __getitem__(self, index):
        row = self.files[index].strip()
        img = cv2.imread(row)
        label_path = row.replace('Image/ColorImage', 'Label').replace('.jpg', '_bin.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, 0)
        img, mask = img[690:, :, :], mask[690:, :]
        img = self.resize_img(img)
        mask = self.resize_mask(mask)
        if self.transform:
            img = self.transform(img)
        label = mask_to_label(mask)
        label = one_hot(label)
        return img, torch.Tensor(label)

    def resize_timg(self, image, size):
        image = F.interpolate(image.unsqueeze(0),
                              size=size,
                              mode="bilinear",
                              align_corners=True).squeeze(0)
        return image

    def resize_tlabel(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size,
                              mode="nearest").squeeze(0)
        return image

    def collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        if self.multi_scale:
            if self.batch_cnt % 10 == 0 and self.multi_scale:
                self.wid = random.choice(range(self.min_size, self.max_size))
                self.hei = int(self.wid / self.ratio)
            imgs = torch.stack(
                [self.resize_timg(img, (self.hei, self.wid)) for img in imgs])
            labels = torch.stack([
                self.resize_tlabel(img, (self.hei, self.wid)) for img in labels
            ])
            self.batch_cnt += 1
        return imgs, labels


if __name__ == "__main__":
    data = LaneDataSet()
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        data, batch_size=1)  # collate_fn=data.collate_fn, num_workers=1)
    i = 0
    import time
    ss = time.time()
    for batch in loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
