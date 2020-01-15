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
from sync_batchnorm import patch_replication_callback
from model.deeplabv3_plus import DeeplabV3Plus
from util.datagener import (get_test_loader, get_train_loader,
                            get_valid_loader, one_hot)
from util.gpu import wait_gpu
from util.label_util import label_to_color_mask
from util.loss import DiceLoss
from util.metric import compute_iou
from config import MEMORY, EPOCH, LOGPATH, MODELNAME, SIZE1, SIZE2, SIZE3


class Trainer(object):
    def __init__(self, memory=MEMORY):
        plt = sys.platform
        self.visdom = Visdom() if plt == 'win32' or plt == 'darwin' else None
        self.trainF = open(os.path.join(LOGPATH, 'train.txt'), 'w+')
        self.testF = open(os.path.join(LOGPATH, 'test.txt'), 'w+')
        self.ids = self.bootstrap(memory)
        self.loss_func1 = BCEWithLogitsLoss().cuda(device=self.ids[0])
        self.loss_func2 = DiceLoss().cuda(device=self.ids[0])

    def bootstrap(self, memory):
        '''
        获取可用的GPU
        '''
        argv = sys.argv
        if len(argv) > 1:
            ids = [int(argv[1])]
        else:
            ava_gpu_index = wait_gpu(need=memory)
            ids = ava_gpu_index
        print(f"Use Device  {ids} Train")
        return ids

    def adjust_lr(self, optimizer, epoch):
        '''
        根据epoch衰减学习率
        '''
        if epoch == 0:
            lr = 6e-4
        elif epoch == 1:
            lr = 4e-4
        elif epoch == 5:
            lr = 3e-4
        elif epoch == 10:
            lr = 2e-4
        elif epoch == 15:
            lr = 7e-5
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def encode(self, labels):
        '''
        将传入的预测标签转换为彩色标签
        @param labels
        '''
        rst = []
        for ele in labels:
            ele = np.argmax(ele, axis=0)
            rst.append(label_to_color_mask(ele))
        return rst

    def visual(self, img, sig, mask, total_mask_loss):
        '''
        使用visdom可视化，需要启动visdom服务 python -m visdom.server
        '''
        pred = sig.cpu().detach().numpy().copy()
        pred = np.array(self.encode(pred))
        pred = pred.transpose((0, 3, 1, 2))

        ground_truth = mask.cpu().detach().numpy().copy()
        ground_truth = np.array(self.encode(ground_truth))
        ground_truth = ground_truth.transpose((0, 3, 1, 2))

        self.visdom.images(img, win='Image', opts=dict(title='colorimg'))
        self.visdom.images(pred, win='Pred', opts=dict(title='pred'))
        self.visdom.images(ground_truth,
                           win='GroudTruth',
                           opts=dict(title='label'))
        self.visdom.line(total_mask_loss,
                         win='train_iter_loss',
                         opts=dict(title='train iter loss'))

    def train(self, net, epoch, dataLoader, optimizer):
        net.train()
        total_mask_loss = []
        dataprocess = tqdm(dataLoader)
        i = 0
        dataprocess.set_description_str("epoch:{}".format(epoch))
        for i, batch_item in enumerate(dataprocess):
            if i % 100 == 0:
                self.trainF.flush()
            image, mask = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            optimizer.zero_grad()
            out = net(image)
            sig = torch.sigmoid(out)
            mask_loss = self.loss_func1(out, mask) + self.loss_func2(sig, mask)
            mask_loss.backward()
            total_mask_loss.append(mask_loss.item())
            dataprocess.set_postfix_str("mask_loss:{:.7f}".format(
                np.mean(total_mask_loss)))

            if i % 10 == 0:
                if self.visdom:
                    self.visual(image, sig, mask, total_mask_loss)

            self.trainF.write(f'Epoch {epoch} loss {mask_loss.item()}\n')
            optimizer.step()

        mean_loss = f"Epoch {epoch}  Mean loss {np.mean(total_mask_loss)}"
        print(mean_loss)
        self.trainF.write(mean_loss)
        self.trainF.flush()

    def mean_iou(self, epoch, result):
        '''
        计算miou,只计算1-7一共7个类别的iou
        '''
        miou = 0
        for i in range(1, 8):
            result_string = "{}: {:.4f} \n".format(
                i, result["TP"][i] / result["TA"][i])
            self.testF.write(f'Epoch {epoch} IOU {result_string}')
            print(result_string)
            miou += result["TP"][i] / result["TA"][i]
        return miou / 7

    def valid(self, net, epoch, dataLoader):
        net.eval()
        total_mask_loss = []
        dataprocess = tqdm(dataLoader)
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
        for batch_item in dataprocess:
            image, mask = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            out = net(image)
            sig = torch.sigmoid(out)
            mask_loss = self.loss_func1(out, mask) + self.loss_func2(sig, mask)
            total_mask_loss.append(mask_loss.detach().item())
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(mask, dim=1), dim=1)
            result = compute_iou(pred, mask, result)
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(
                np.mean(total_mask_loss)))
            self.testF.write(f'Epoch {epoch} loss {mask_loss.item()}\n')

        self.testF.flush()
        return self.mean_iou(epoch, result)

    def load_model(self):
        if os.path.exists(MODELNAME):
            print("train from load model")
            last_gpu_id = int(open('last_gpu.id', 'r').read().strip())
            net = torch.load(
                MODELNAME,
                map_location={f'cuda:{last_gpu_id}': f"cuda:{self.ids[0]}"})
        else:
            print("train from scratch")
            net = DeeplabV3Plus(n_class=8).cuda(device=self.ids[0])
            with open('last_gpu.id', 'w') as f:
                f.write(str(self.ids[0]))
        return net

    def run(self, batchsize, shape):
        train_data_batch = get_train_loader(batchsize, shape)
        val_data_batch = get_valid_loader(batchsize, shape)
        net = self.load_model()
        net = DataParallel(net, device_ids=self.ids)
        patch_replication_callback(net)
        optimizer = torch.optim.AdamW(net.parameters())
        last_MIOU = 0.0

        for epoch in range(EPOCH):
            self.adjust_lr(optimizer, epoch)
            self.train(net, epoch, train_data_batch, optimizer)
            with torch.no_grad():
                miou = self.valid(net, epoch, val_data_batch)
            if miou > last_MIOU:
                print(f"miou {miou} > last_MIOU {last_MIOU},save model")
                torch.save(net, os.path.join(os.getcwd(), "laneNet.pth"))
                last_MIOU = miou
        torch.save(net, os.path.join(os.getcwd(), "finalNet.pth"))


def main():
    for ele in [SIZE1, SIZE2, SIZE3]:
        print(f"Train Size {ele}")
        shape, batch = ele
        trainer = Trainer(memory=6 if batch == 2 else 9)
        trainer.run(batch, shape)


if __name__ == '__main__':
    main()
