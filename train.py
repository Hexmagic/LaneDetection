import os
import shutil
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torchvision import transforms
from tqdm import tqdm
from visdom import Visdom

from model.deeplabv3_plus import DeeplabV3Plus
from model.deeplabv3p import DeepLabV3P
from util.creat_csv import dump
from util.datagener import (get_test_loader, get_train_loader,
                            get_valid_loader, one_hot)
from util.gpu import wait_gpu
from util.label_util import label_to_color_mask
from util.loss import DiceLoss, FocalLoss
from util.metric import compute_iou

plt = sys.platform

#ava_gpu_index = wait_gpu(need=7)
#torch.cuda.set_device(ava_gpu_index)
ids = [3]


def encode(labels):
    rst = []
    for ele in labels:
        ele = np.argmax(ele, axis=0)
        rst.append(label_to_color_mask(ele))
    return rst


if plt == 'win32':
    vis = Visdom()


def train_epoch(net, epoch, dataLoader, optimizer):
    net.train()
    total_mask_loss = []
    dataprocess = tqdm(dataLoader)
    loss_func1 = BCEWithLogitsLoss().cuda(device=ids[0])
    loss_func2 = DiceLoss().cuda(device=ids[0])
    loss_func3 = FocalLoss().cuda(device=ids[0])
    #loss_func2 = FocalLoss(class_num=8).cuda()
    i = 0
    dataprocess.set_description_str("epoch:{}".format(epoch))
    for batch_item in dataprocess:
        i += 1
        image, mask = batch_item
        if torch.cuda.is_available():
            image, mask = Variable(image).cuda(
                device=ids[0]), Variable(mask).cuda(device=ids[0])
        optimizer.zero_grad()
        out = net(image)
        sig = torch.sigmoid(out)
        loss1 = loss_func1(out, mask)  # bcewithlogitsloss
        loss2 = loss_func2(sig, mask)
        loss3 = loss_func3(out, mask)
        #loss2 = DiceLoss()(out, mask)
        mask_loss = loss1 + loss2
        mask_loss.backward()

        if i % 10 == 0:
            if plt != 'win32':
                continue
            _np = sig.cpu().detach().numpy().copy(
            )  # output_np.shape (4, 2, 160, 160)
            #output_np = np.argmax(_np, axis=1)
            pred = np.array(encode(_np))
            pred = pred.transpose((0, 3, 1, 2))

            bag_msk_np = mask.cpu().detach().numpy().copy()
            #mask = np.argmax(bag_msk_np, axis=1)
            label = np.array(encode(bag_msk_np))
            bag_msk_np = label.transpose((0, 3, 1, 2))

            vis.images(pred, win='Pred', opts=dict(title='pred'))
            vis.images(bag_msk_np, win='GroudTruth', opts=dict(title='label'))
            vis.images(image, win='Image', opts=dict(title='colorimg'))
            vis.line(total_mask_loss,
                     win='train_iter_loss',
                     opts=dict(title='train iter loss'))
        total_mask_loss.append(mask_loss.item())
        #mask_loss.backward()
        optimizer.step()
        # dataprocess.set_postfix_str("mask_loss:{:.7f}".format(
        #     mask_loss.item()))
    print(f"Epoch {epoch} loss {np.mean(total_mask_loss)}")


def test(net, epoch, dataLoader):
    net.eval()
    total_mask_loss = []
    dataprocess = tqdm(dataLoader)
    loss_func1 = BCEWithLogitsLoss().cuda(device=ids[0])
    loss_func2 = DiceLoss().cuda(device=ids[0])
    MIOU = 0.0
    result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item
        if torch.cuda.is_available():
            image, mask = Variable(image).cuda(
                device=ids[0]), Variable(mask).cuda(device=ids[0])
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

    for i in range(1, 8):
        result_string = "{}: {:.4f} \n".format(
            i, result["TP"][i] / result["TA"][i])
        print(result_string)
        MIOU += result["TP"][i] / result["TA"][i]
    return MIOU / 7


def adjust_lr(optimizer, epoch):
    if epoch == 0:
        lr = 6e-4
    elif epoch == 1:
        lr = 5e-4
    elif epoch == 5:
        lr = 4e-4
    elif epoch == 10:
        lr = 2e-4
    elif epoch == 15:
        lr = 7e-5
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_data_batch = get_train_loader(batch_size=2)
    val_data_batch = get_valid_loader()
    if os.path.exists('laneNet.pth'):
        net = torch.load('laneNet.pth', map_location={'cuda:0': 'cuda:3'})
    else:
        net = DeepLabV3P(n_classes=8).cuda(device=ids[0])
    model = DataParallel(net, device_ids=ids)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(net.parameters())
    last_MIOU = 0.0
    for epoch in range(40):
        adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer)
        with torch.no_grad():
            miou = test(net, epoch, val_data_batch)
        if miou > last_MIOU:
            print(f"miou {miou} > last_MIOU {last_MIOU},save model")
            torch.save(net, os.path.join(os.getcwd(), "laneNet.pth"))
            last_MIOU = miou


if __name__ == "__main__":
    main()
