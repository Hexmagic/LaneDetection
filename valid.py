import torch
from torch.autograd import Variable
from torch.nn import BCELoss
from tqdm import tqdm

from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader
from util.label_util import label_to_color_mask
import sys
from model.unet import Unet
from util.metric import compute_iou
from collections import defaultdict
import numpy as np
from visdom import Visdom
plt = sys.platform


def encode(labels):
    rst = []
    for i, ele in enumerate(labels):
        ele = np.argmax(ele, axis=0)
        rst.append(label_to_color_mask(ele))
    return rst


if plt == 'win32':
    vis = Visdom()


def validLoss():
    loss_fuc = BCELoss().cuda()
    model = torch.load('laneNet.pth').cuda()
    test_loader = get_test_loader()
    loss_list = []
    i = 0
    result = {"TP": defaultdict(int), "TA": defaultdict(int)}
    for batch in tqdm(test_loader):
        x, y = batch
        i += 1
        xv, yv = Variable(x).cuda(), Variable(y).cuda()
        yout = model(xv)
        yout = torch.sigmoid(yout)
        if i % 5 == 0:
            if plt != 'win32':
                continue
            _np = yout.cpu().detach().numpy().copy()
            pred = np.array(encode(_np))
            pred = pred.transpose((0, 3, 1, 2))
            bag_msk_np = yv.cpu().detach().numpy().copy()
            #mask = np.argmax(bag_msk_np, axis=1)
            label = np.array(encode(bag_msk_np))
            bag_msk_np = label.transpose((0, 3, 1, 2))
            vis.images(pred,
                       win='train_pred',
                       opts=dict(title='train prediction'))
            vis.images(bag_msk_np,
                       win='train_label',
                       opts=dict(title='train prediction'))
            vis.line(loss_list,
                     win='train_iter_loss',
                     opts=dict(title='train iter loss'))
        loss = loss_fuc(yout, yv)
        loss_list.append(loss.item())
        result = compute_iou(yout, yv, result)
    print(f'Valid Losss {sum(loss_list)/len(loss_list)}')
    MIOU = 0.0
    for i in range(8):
        print(f"Class {i} IOU {result['TP'][i]/result['TA'][i]}")
        MIOU += result["TP"][i] / result["TA"][i]
    print(f"MIOU {MIOU}")


validLoss()