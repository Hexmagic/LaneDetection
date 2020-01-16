from setting import MODELNAME,SIZE2
from model.project import Projection
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from util.loss import DiceLoss
from tqdm import tqdm
from util.gpu import wait_gpu
from util.label_util import label_to_color_mask
import torch
import numpy as np 
from util.datagener import get_test_loader
from visdom import Visdom
import os
import sys
plt = sys.platform


class PTrain(object):
    def __init__(self):
        plt = sys.platform
        self.visdom = None if plt == 'linux' else Visdom()
        self.result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        self.ids = self.bootstrap()
        self.loss_func1 = BCEWithLogitsLoss().cuda(device=self.ids[0])
        self.loss_func2 = DiceLoss().cuda(device=self.ids[0])
        self.dataprocess = tqdm(get_test_loader(batch_size=1,size=SIZE2[0]))

    def bootstrap(self):
        '''
        获取可用的GPU
        '''
        argv = sys.argv
        if len(argv) > 1:
            ids = [int(argv[1])]
        else:
            ava_gpu_index = wait_gpu(need=7)
            ids = [ava_gpu_index]
        print(f"Use Device  {ids} Valid")
        return ids

    def encode(self, labels):
        '''
        转换label为彩色标签
        '''
        rst = []
        for ele in labels:
            ele = np.argmax(ele, axis=0)
            rst.append(label_to_color_mask(ele))
        return rst

    def miou(self):
        MIOU = 0.0
        for i in range(1, 8):
            #print(result_string)
            MIOU += self.result["TP"][i] / self.result["TA"][i]
        MIOU = MIOU / 7
        return MIOU

    def load_model(self):
        if os.path.exists(MODELNAME):
            print("load from load model")
            last_gpu_id = int(open('last_gpu.id', 'r').read().strip())
            net = torch.load(
                MODELNAME,
                map_location={f'cuda:{last_gpu_id}': f"cuda:{self.ids[0]}"})
        else:
            print("Model Not Exists")
        return net