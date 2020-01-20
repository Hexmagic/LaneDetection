import os
import shutil
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from tqdm import tqdm
from visdom import Visdom

from model.deeplabv3_plus import DeeplabV3Plus
from model.unet import Unet
from model.unet_plus import Unet_2D
from setting import LOGPATH, MEMORY, MODELNAME, SIZE1, SIZE2, SIZE3
from sync_batchnorm import DataParallelWithCallback
from util.datagener import (get_test_loader, get_train_loader,
                            get_valid_loader, one_hot)
from util.gpu import wait_gpu
from util.label_util import label_to_color_mask
from util.loss import DiceLoss, FocalLoss
from util.metric import compute_iou


class Trainer(object):
    def __init__(self, memory=MEMORY, model='deeplab'):
        self.model = model
        plt = sys.platform
        self.visdom = Visdom() if plt == 'win32' or plt == 'darwin' else None
        self.trainF = open(os.path.join(LOGPATH, 'train.txt'), 'w+')
        self.testF = open(os.path.join(LOGPATH, 'test.txt'), 'w+')
        self.ids = self.bootstrap(memory)

    def bootstrap(self, memory):
        '''
        获取可用的GPU
        '''
        argv = sys.argv
        if len(argv) > 1:
            ids = list(map(int, argv[1].split(',')))
        else:
            ava_gpu_index = wait_gpu(need=memory)
            ids = ava_gpu_index
        print(f"Use Device  {ids} Train")
        return ids

    def adjust_lr_adam(self, optimizer, epoch):
        '''
        根据epoch衰减学习率
        '''
        if epoch == 0:
            lr = 6e-4
        elif epoch == 2:
            lr = 4e-4
        elif epoch == 5:
            lr = 3e-4
        elif epoch == 8:
            lr = 4e-4
        elif epoch == 13:
            lr = 3e-4
        elif epoch == 18:
            lr = 2e-4
        elif epoch == 22:
            lr = 3e-4
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_lr_sgd(self, optimizer, epoch):
        '''
        根据epoch衰减学习率
        '''
        if epoch == 0:
            lr = 1e-2
        elif epoch == 2:
            lr = 4e-4
        elif epoch == 5:
            lr = 3e-4
        elif epoch == 8:
            lr = 4e-4
        elif epoch == 13:
            lr = 3e-4
        elif epoch == 18:
            lr = 2e-4
        elif epoch == 22:
            lr = 3e-4
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
        loss_func1 = BCEWithLogitsLoss().cuda(device=self.ids[0])
        loss_func2 = DiceLoss().cuda(device=self.ids[0])
        #loss_func3 = FocalLoss().cuda(device=self.ids[0])
        dataprocess.set_description_str("epoch:{}".format(epoch))
        for i, batch_item in enumerate(dataprocess):
            if i % 100 == 0:
                self.trainF.flush()
            image, mask, _ = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            optimizer.zero_grad()
            out = net(image)
            sig = torch.sigmoid(out)
            mask_loss = loss_func1(out, mask) + loss_func2(
                sig, mask)  #+ loss_func3(out, mask)
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
        loss_func1 = BCEWithLogitsLoss().cuda(device=self.ids[0])
        loss_func2 = DiceLoss().cuda(device=self.ids[0])
        #loss_func3 = FocalLoss().cuda(device=self.ids[0])
        for batch_item in dataprocess:
            image, mask, _ = batch_item
            if torch.cuda.is_available():
                image, mask = Variable(image).cuda(
                    device=self.ids[0]), Variable(mask).cuda(
                        device=self.ids[0])
            out = net(image)
            sig = torch.sigmoid(out)
            mask_loss = loss_func1(out, mask) + loss_func2(
                sig, mask)  #+ loss_func3(out, mask)
            total_mask_loss.append(mask_loss.item())
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
            last_gpu_id = int(
                open(f'{self.model}_last_gpu.id', 'r').read().strip())
            net = torch.load(
                MODELNAME,
                map_location={f'cuda:{last_gpu_id}': f"cuda:{self.ids[0]}"})
        else:
            print("train from scratch")
            if self.model == 'deeplab':
                print("Model Deeplab")
                net = DeeplabV3Plus(n_class=8).cuda(device=self.ids[0])
            elif self.model == 'unet++':
                print("Model Unet++")
                net = Unet_2D(n_classes=8).cuda(device=self.ids[0])
            else:
                print("MOdeul Unet")
                net = Unet(n_class=8).cuda(device=self.ids[0])
            with open(f'{self.model}_last_gpu.id', 'w') as f:
                f.write(str(self.ids[0]))
        return net

    def run(self, batchsize, shape, epochs):
        train_data_batch = get_train_loader(batchsize, shape)
        val_data_batch = get_valid_loader(batchsize, shape)
        net = self.load_model()
        if len(self.ids) > 1:
            print("Use Mutil GPU Train Model")
            # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DataParallel(net, device_ids=self.ids)
            # net = DistributedDataParallel(net, device_ids=self.ids)
            #net = DataParallelWithCallback(net, device_ids=self.ids)
            #patch_replication_callback(net)
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=0.001,
                                    momentum=0.95,
                                    weight_decay=0.01,
                                    nesterov=True)
        last_MIOU = 0.0

        for epoch in range(epochs):
            self.adjust_lr(optimizer, epoch)
            self.train(net, epoch, train_data_batch, optimizer)
            with torch.no_grad():
                miou = self.valid(net, epoch, val_data_batch)
            if miou > last_MIOU:
                print(f"miou {miou} > last_MIOU {last_MIOU},save model")
                torch.save(
                    net, os.path.join(os.getcwd(),
                                      f"{self.model}_laneNet.pth"))
                last_MIOU = miou
        torch.save(net, os.path.join(os.getcwd(), f"{self.model}finalNet.pth"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='输入GPU的ID,或者以逗号分隔的ID列表')
    parser.add_argument('--model', type=str, help='模型名称，deeplab或者unet++')
    parser.add_argument('--stage',type=int,help='训练阶段，默认为1',default=1)
    args = parser.parse_args()
    model = args.model
    assert model in ['unet', 'deeplab', 'unet++']
    for ele in [SIZE1, SIZE2, SIZE3]:
        print(f"Train Size {ele}")
        shape, batch, epoch = ele
        trainer = Trainer(memory=6 if batch == 2 else 9, model=model)
        trainer.run(batch, shape, epoch)


if __name__ == '__main__':
    main()
