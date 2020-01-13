import torch
from torch.autograd import Variable
from torch.nn import BCELoss, BCEWithLogitsLoss
from tqdm import tqdm
import torch.nn.functional as F
from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader
from util.label_util import label_to_color_mask
import sys
from util.loss import DiceLoss
from util.metric import compute_iou
from collections import defaultdict
import numpy as np
import time
from util.gpu import wait_gpu
from visdom import Visdom


class Valider(object):
    def __init__(self):
        plt = sys.platform
        self.visdom = None if plt == 'linux' else Visdom()
        gid = wait_gpu(7)
        torch.cuda.set_device()
        self.loss_func1 = BCEWithLogitsLoss().cuda()
        self.loss_func2 = DiceLoss().cuda()
        self.result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }

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
            result_string = "{}: {:.4f} \n".format(
                i, self.result["TP"][i] / self.result["TA"][i])
            #print(result_string)
            MIOU += self.result["TP"][i] / self.result["TA"][i]
        MIOU = MIOU / 7
        return MIOU

    def run(self):
        with torch.no_grad():
            net = torch.load('laneNet.pth', map_location={'cuda:6': 'cuda:0'})
            net.eval()
            total_mask_loss = []
            dataLoader = get_test_loader(batch_size=1)
            dataprocess = tqdm(dataLoader)
            for i, batch_item in enumerate(dataprocess):
                image, mask = batch_item
                image, mask = Variable(image).cuda(), Variable(mask, ).cuda()
                out = net(image)
                sig = torch.sigmoid(out)
                mask_loss = self.loss_func1(out, mask) + self.loss_func2(
                    sig, mask)
                total_mask_loss.append(mask_loss.detach().item())
                dataprocess.set_postfix_str("mask_loss:{:.4f}".format(
                    np.mean(total_mask_loss)))

                if i % 20 == 0:
                    '''
                    每20batch显示一次MIOU
                    '''
                    miou = compute_miou(result)
                    dataprocess.set_description_str("MIOU:{}".format(miou))

                if i % 10 == 0:
                    if self.visdom:
                        self.visual()
                # 计算IOU

                pred = torch.argmax(F.softmax(out, dim=1), dim=1)
                mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
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
            self.visdom.images(image, win='Image', opts=dict(title='colorimg'))
            self.visdom.line(total_mask_loss,
                             win='valid loss',
                             opts=dict(title='train iter loss'))

            miou = self.miou()
            print(f"Mean IOU {miou}")


if __name__ == "__main__":
    validLoss()