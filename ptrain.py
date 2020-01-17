import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm
from visdom import Visdom

from model.project import Projection
from setting import MODELNAME, SIZE2
from util.gpu import wait_gpu
from util.label_util import label_to_color_mask
from util.loss import DiceLoss
from util.mask_data import train_loader
from util.metric import compute_iou

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
        self.dataprocess = tqdm(train_loader)

    def run(self):
        model = Projection()
        opt = AdamW(model.parameters())
        total_loss = []
        for epoch in range(10):
            for batch in self.dataprocess:
                img, mask = batch
                import pdb; pdb.set_trace()
                img, mask = Variable(img).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
                out = model(img)
                sig = torch.sigmoid(out)
                loss = self.loss_func2(sig, mask)
                opt.zero_grad()
                loss.backward()
                self.result = compute_iou(sig, mask, self.result)
                miou = self.miou()
                total_loss.append(loss.item())
                opt.step()
                self.dataprocess.set_postfix_str(f"loss {np.mean(total_loss)}")
                self.dataprocess.set_description_str(f"miou {miou}")

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

    def miou(self):
        MIOU = 0.0
        for i in range(1, 8):
            #print(result_string)
            MIOU += self.result["TP"][i] / self.result["TA"][i]
        MIOU = MIOU / 7
        return MIOU

if __name__ =='__main__':
    pt = PTrain()
    pt.run()