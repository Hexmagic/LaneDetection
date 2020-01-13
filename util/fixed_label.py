from util.label_util import COLORMAP,label_to_color_mask,label_to_mask
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from seize.util.pipe import split_into_n
import os
# 3a+4b+5c
CMAP = {
    1410: 1,
    1360: 1,
    1024: 0,
    1275: 1,
    180: 0,
    540: 0,
    426: 2,
    735: 2,
    2056: 0,
    480: 0,
    1836: 0,
    1980: 3,
    2020: 0,
    1386: 0,
    640: 0,
    1280: 4,
    2628: 0,
    2021: 5,
    690: 0,
    1152: 6,
    1432: 6,
    1450: 6,
    1935: 0,
    2100: 6,
    1208: 6,
    2712: 6,
    1529: 0,
    1428: 0,
    2104: 0,
    1655: 0,
    1787: 7,
    1785: 0,
    1988: 7,
    1344: 0,
    1122: 7,
    1071: 0,
    3060: 0
}


def img_to_label(img):
    sh = img.shape
    img = img.astype(np.int32)
    label = np.zeros(sh[:2], dtype=np.uint8)
    a, b, c = cv2.split(img)
    tmp = 3 * a + 4 * b + 5 * c
    for k, v in CMAP.items():
        tmp[tmp == k] = v
    return tmp.astype(np.uint8)

import threading
def fixed(pngs):
    thname = threading.current_thread().getName()
    for ele in tqdm(pngs,desc=thname):
        img = cv2.imread(str(ele))
        label = img_to_label(img)
        mask = label_to_mask(label)
        gimg = cv2.merge([mask, mask, mask])
        path = str(ele).replace('Labels_Fixed', 'Gray_Label')
        if os.path.exists(path):
            cv2.imwrite(path,gimg)


def main():
    path = 'D:/Compressed/Labels_Fixed'
    pngs = list(Path(path).glob('*/*/*/*/*.png'))
    pool = ThreadPoolExecutor(max_workers=8)
    for _ in pool.map(fixed, pngs >> split_into_n(8)):
        _
main()