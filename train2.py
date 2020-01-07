from tqdm import tqdm
import torch
import os
import shutil
from util.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from util.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import get_test_loader, get_valid_loader, get_train_loader
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
from visdom import Visdom
import sys
from util.label_util import label_to_color_mask
device_list = [5]
plt = sys.platform


def encode(labels):
    rst = []
    for i, ele in enumerate(labels):
        rst.append(label_to_color_mask(ele))
    return rst


if plt =='win32':
    vis = Visdom()


def train_epoch(net, epoch, dataLoader, optimizer):
    net.train()
    total_mask_loss = []
    dataprocess = tqdm(dataLoader)
    i = 0
    for batch_item in dataprocess:
        i += 1
        image, mask = batch_item
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(
                device=device_list[0])
        optimizer.zero_grad()
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(8)(out, mask.long())
        if i % 10 == 0:
            if plt!='win32':
                continue
            _np = out.cpu().detach().numpy().copy(
            )  # output_np.shape (4, 2, 160, 160)
            output_np = np.argmax(_np, axis=1)
            pred = np.array(encode(output_np))
            pred = pred.transpose((0, 3, 1, 2))
            bag_msk_np = mask.cpu().detach().numpy().copy()
            #mask = np.argmax(bag_msk_np, axis=1)
            label = np.array(encode(bag_msk_np))

            bag_msk_np = label.transpose((0, 3, 1, 2))

            vis.images(pred,
                       win='train_pred',
                       opts=dict(title='train prediction'))
            vis.images(bag_msk_np,
                       win='train_label',
                       opts=dict(title='train prediction'))
            vis.line(total_mask_loss,
                     win='train_iter_loss',
                     opts=dict(title='train iter loss'))
        total_mask_loss.append(mask_loss.item())
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(
            np.mean(total_mask_loss)))
    print("Epoch:{}, mask loss is {:.4f} \n".format(
        epoch, total_mask_loss / len(dataLoader)))


def test(net, epoch, dataLoader):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(
                device=device_list[0])
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=8)(out[0], mask.long())
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))

    for i in range(8):
        result_string = "{}: {:.4f} \n".format(
            i, result["TP"][i] / result["TA"][i])
        print(result_string)


def adjust_lr(optimizer, epoch):
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_data_batch = get_train_loader()
    val_data_batch = get_valid_loader()
    net = DeeplabV3Plus(n_class=8)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=0.0006,
                                 weight_decay=1.0e-4)
    for epoch in range(30):
        # adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer)
        test(net, epoch, val_data_batch)
        if epoch % 10 == 0:
            torch.save(
                net, os.path.join(os.getcwd(), "laneNet{}.pth".format(epoch)))
    torch.save(net, os.path.join(os.getcwd(), "finalNet.pth"))


if __name__ == "__main__":
    main()
