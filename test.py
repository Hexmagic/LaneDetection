import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCELoss, BCEWithLogitsLoss
from tqdm import tqdm
from visdom import Visdom

from model.deeplabv3_plus import DeeplabV3Plus
from setting import MEMORY, MODELNAME, PREDICT_PATH, SIZE3
from data.datagener import get_test_loader, get_valid_loader
from util.gpu import wait_gpu
from data.label_util import label_to_color_mask, mask_to_label
from util.loss import DiceLoss
from util.metric import compute_iou


class Tester(object):
    def __init__(self, mode='test'):
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
        self.dataprocess = tqdm(get_test_loader(
            batch_size=1, size=SIZE3[0])) if mode == 'test' else tqdm(
                get_valid_loader(batch_size=1, size=SIZE3[0]))

    def bootstrap(self):
        '''
        获取可用的GPU
        '''
        argv = sys.argv
        if len(argv) > 1:
            ids = [int(argv[1])]
        else:
            ava_gpu_index = wait_gpu(need=MEMORY)
            ids = ava_gpu_index
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

    def run(self):
        with torch.no_grad():
            net = self.load_model()
            net.eval()
            total_mask_loss = []

            for i, batch_item in enumerate(self.dataprocess):
                i = i + 1
                image, mask, names = batch_item
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
                out = net(image)
                sig = torch.sigmoid(out)
                mask_loss = self.loss_func1(out, mask) + self.loss_func2(
                    sig, mask)
                total_mask_loss.append(mask_loss.detach().item())
                self.dataprocess.set_postfix_str("mask_loss:{:.4f}".format(
                    np.mean(total_mask_loss)))

                if i % 20 == 0:
                    '''
                    每20batch显示一次MIOU
                    '''
                    miou = self.miou()
                    self.dataprocess.set_description_str(
                        "MIOU:{}".format(miou))

                if i % 10 == 0:
                    if self.visdom:
                        self.visual(image, sig, mask, total_mask_loss)
                # 计算IOU
                pred = F.interpolate(sig, (510 * 2, 1692 * 2),
                                     mode='bilinear',
                                     align_corners=True)
                pred = torch.argmax(pred, dim=1)
                img = cv2.imread(names[0], 0)
                img = img[690:, :]
                mask = mask_to_label(img)
                mask = torch.stack([torch.from_numpy(mask)])
                self.result = compute_iou(pred, mask, self.result)

    def visual(self, img, sig, mask, total_mask_loss):
        # 预测转换为彩色标签
        pred = sig.cpu().detach().numpy().copy()
        pred = np.array(self.encode(pred))
        pred = pred.transpose((0, 3, 1, 2))

        #mask 转换为彩色标签
        label = mask.cpu().detach().numpy().copy()
        label = np.array(self.encode(label))
        label = label.transpose((0, 3, 1, 2))

        self.visdom.images(pred, win='Pred', opts=dict(title='pred'))
        self.visdom.images(label, win='mask', opts=dict(title='label'))
        self.visdom.images(img, win='Image', opts=dict(title='colorimg'))
        self.visdom.line(total_mask_loss,
                         win='valid loss',
                         opts=dict(title='train iter loss'))

        miou = self.miou()
        print(f"Mean IOU {miou}")


if __name__ == "__main__":
    Tester('test').run()
