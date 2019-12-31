import os
from pathlib import Path
from typing import List, Tuple
import cv2
import pandas as pd
from sklearn.utils import shuffle
from logzero import logger

CSV_PATH = 'data_list'
TRAIN_SIZE = 0.6
VALID_SIZE = 0.2
TEST_SIZE = 0.2
DATA_ROOT = 'D:/Compressed'


class LaneDataFactory(object):
    '''
    获取所有的图片和对应的label位置，并按照比例割训练集和验证集，测试集
    '''

    def __init__(self):
        self.root = DATA_ROOT

    def saveCSV(self, name, imgs: List[str], labels: List[str]):
        dataFram = pd.DataFrame({'img': imgs,
                                 'label': labels})
        dataFram.to_csv(os.path.join(CSV_PATH, name),index=False)

    def dump(self) -> None:
        '''保存训练测试验证的路径到csv文件'''
        imgs: List[str] = []
        labels: List[str] = []
        # 获取图片和label
        for road in ['Road02', 'Road04']:
            img, label = self.getImageAndLabel(road)
            assert len(img) == len(label)
            logger.info(f"{road} find {len(img)} Image and Label")
            imgs += img
            labels += label

        # 打乱顺序，每次开始训练获取的测试验证和训练集不同
        shuffle(imgs, labels)
        # 获取切分的索引
        length = len(imgs)
        i = int(length*TRAIN_SIZE)
        j = i+int(length*TEST_SIZE)
        # 返回训练集验证集和测试集
        self.saveCSV('train.csv', imgs[:i], labels[:i])
        self.saveCSV('test.csv', imgs[i:j], labels[i:j])
        self.saveCSV('valid.csv', imgs[j:], labels[j:])

    def getImageAndLabel(self, path: str) -> Tuple[List[str], List[str]]:
        '''获取对应Road下面的数据路径对饮的label路径'''
        img_list, label_list = [], []
        for ele in Path(os.path.join(self.root, path)).glob('*/*/*/*/*.jpg'):
            imgName = ele.name
            labelParent = str(ele.parent).replace(path, 'Gray_Label').replace(
                f'ColorImage_{path.lower()}\\ColorImage', f'Label_{path.lower()}\\Label'
            )
            labelPath = f'{labelParent}\\{imgName.replace(".jpg","_bin.png")}'
            if not os.path.exists(os.path.join(self.root, labelPath)):
                continue
            img_list.append(str(ele))
            label_list.append(
                os.path.join(self.root, labelPath))
        return img_list, label_list


if __name__ == "__main__":
    LaneDataFactory().dump()
