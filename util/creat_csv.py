import os
from pathlib import Path
from typing import List, Tuple
import cv2
import sys
import pandas as pd
from sklearn.utils import shuffle
from logzero import logger
import json
from config import RoadList
from sklearn.model_selection import train_test_split



class LaneDataFactory(object):
    '''
	获取所有的图片和对应的label位置，并按照比例割训练集和验证集，测试集
	'''
    def __init__(self):
        self.root = DATAROOT

    def saveCSV(self, name, imgs: List[str], labels: List[str]):
        dataFram = pd.DataFrame({'img': imgs, 'label': labels})
        dataFram.to_csv(os.path.join(CSV_PATH, name), index=False)

    def dump(self) -> None:
        '''保存训练测试验证的路径到csv文件'''
        imgs: List[str] = []
        labels: List[str] = []
        # 获取图片和label
        for road in RoadList:
            img, label = self.getImageAndLabel(road)
            assert len(img) == len(label)
            logger.info(f"{road} find {len(img)} Image and Label")
            imgs += img
            labels += label

        # 打乱顺序，每次开始训练获取的测试验证和训练集不同
        shuffle(imgs, labels)
        # 获取切分的索引
        length = len(imgs)
        i = int(length * TRAIN_SIZE)
        j = i + int(length * TEST_SIZE)
        # 返回训练集验证集和测试集
        train_img, other_img, train_label, other_label = train_test_split(
            imgs, labels, test_size=0.3)
        test_img, valid_img, test_label, valid_label = train_test_split(
            other_img, other_label, test_size=0.33)
        self.saveCSV('train.csv', train_img, train_label)
        self.saveCSV('test.csv', test_img, test_label)
        self.saveCSV('valid.csv', valid_img, valid_label)

    def getImageAndLabel(self, path: str) -> Tuple[List[str], List[str]]:
        '''获取对应Road下面的数据路径对饮的label路径'''
        img_list, label_list = [], []
        road = path.split('/')[-1]
        pt = Path(os.path.join(self.root, path))
        if PLAT == 'win32':
            glob = '*/*/*/*/*.jpg'
        else:
            glob = '*/*/*.jpg'
        for ele in pt.glob(glob):
            imgName = ele.name
            labelParent = str(ele.parent).replace(
                path, f'Gray_Label/Label_{road.lower()}/Label')
            labelName = imgName.replace('.jpg', '_bin.png')
            labelPath = os.path.join(labelParent,
                                     imgName.replace(".jpg", "_bin.png"))
            if PLAT == 'win32':
                path = labelPath.replace("Image_Data", "Gray_Label")
                path = path.replace(road, f'Label_{road.lower()}')
                path = path.replace(f'ColorImage_{road.lower()}\\ColorImage',
                                    'Label')
            else:
                path = os.path.join(self.root, labelPath)
                path = path.replace("Image_Data", "Gray_Label")
                path = path.replace(road, f'Label_{road.lower()}/Label')
            if not os.path.exists(path):
                import pdb
                pdb.set_trace()
                continue
            img_list.append(str(ele))
            label_list.append(path)
        return img_list, label_list


def dump():
    LaneDataFactory().dump()


dump()