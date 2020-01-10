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
    with torch.no_grad():
        net = torch.load('laneNet.pth')
        net.eval()
        total_mask_loss = []
        dataLoader = get_test_loader()
        dataprocess = tqdm(dataLoader)
        loss_func1 = BCEWithLogitsLoss().cuda()
        loss_func2 = DiceLoss().cuda()
        MIOU = 0.0
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        for batch_item in dataprocess:
            image, mask = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(), Variable(mask, ).cuda()
            out = net(image)
            sig = torch.sigmoid(out)
            loss1 = loss_func1(out, mask)
            loss2 = loss_func2(sig, mask)
            #loss2 = DiceLoss()(out, mask)
            mask_loss = loss1 + loss2
            #mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=8)(out, mask.long())
            total_mask_loss.append(mask_loss.detach().item())
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(
                np.mean(total_mask_loss)))

        for i in range(8):
            result_string = "{}: {:.4f} \n".format(
                i, result["TP"][i] / result["TA"][i])
            print(result_string)
            MIOU += result["TP"][i] / result["TA"][i]
        return MIOU / 8


validLoss()