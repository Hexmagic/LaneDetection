import os
import random
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
from sipe import split_into_n
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from util.label_util import mask_to_label


class Spliter(object):
    def __init__(self, arg) -> None:
        super(Spliter, self).__init__()
        self.other_label_paths = []
        self.four_label_paths = []
        self.pool = ThreadPoolExecutor(arg.cpu)
        self.root = arg.root
        self.roads = arg.roads

    def job(self, mask):
        thName = threading.current_thread().getName()
        for ele in tqdm(mask, desc=thName):
            img = cv2.imread(ele, 0)
            label = mask_to_label(img)
            if 4 in label:
                self.four_label_paths.append(ele)
            else:
                self.other_label_paths.append(ele)

    def run(self):
        mask = []
        for road in self.roads:
            mask+=glob(f"{self.root}/Gray_Label/Label_road0{road}/**/*.png",recursive=True)
        print("collect image infomation ...")
        for _ in self.pool.map(self.job, mask >> split_into_n(arg.cpu)):
            pass
        print("shuffle data and split trainval")
        train_img, val_img = train_test_split(self.other_label_paths)
        train_img_, val_img_ = train_test_split(self.four_label_paths)
        train_img += train_img_
        val_img += val_img_
        random.shuffle(train_img)
        random.shuffle(val_img)
        print(f"write {len(train_img)} lines to train.txt")
        with open('train.txt', 'w') as f:
            f.writelines('\n'.join(train_img))
        print(f"write {len(val_img)} lines to val.txt")
        with open('val.txt', 'w') as f:
            f.writelines('\n'.join(val_img))


if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--cpu', type=int, default=4)
    par.add_argument('--root', type=str, default='data',help='data set directory')
    par.add_argument('--roads',type=str,default='2,4',help='roads number concat by , ')
    arg = par.parse_args()
    Spliter(arg).run()
