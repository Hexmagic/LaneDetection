import os
from glob import glob
import cv2
from matplotlib.pyplot import winter
from numpy.core.defchararray import join
from tqdm.std import trange
from util.label_util import mask_to_label
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sipe import split_into_n
import threading
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
par = ArgumentParser()
par.add_argument('--cpu', type=int, default=8)
arg = par.parse_args()
pool = ThreadPoolExecutor(arg.cpu)

other_paths = []
four_paths = []


def job(mask):
    thName = threading.current_thread().getName()
    global four_paths
    global other_paths
    for ele in tqdm(mask, desc=thName):
        if 'road03' in ele:
            continue
        img = cv2.imread(ele, 0)
        label = mask_to_label(img)
        if 4 in label:
            four_paths.append(ele)
        else:
            other_paths.append(ele)


def main():
    mask = list(glob('data/**/*.png', recursive=True))
    rst = []
    for result in pool.map(job, mask >> split_into_n(arg.cpu)):
        pass
    import random
    random.shuffle(four_paths)
    random.shuffle(other_paths)
    train_img, val_img = train_test_split(other_paths)
    train_img_, val_img_ = train_test_split(four_paths)
    train_img += train_img_
    val_img += val_img_
    with open('train.txt', 'w') as f:
        f.writelines('\n'.join(train_img))
    with open('val.txt', 'w') as f:
        f.writelines('\n'.join(val_img))


main()
